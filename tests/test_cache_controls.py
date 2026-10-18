"""Per-request cache controls (enhancement #2):
X-Cradle-Cache-Control: no-store / no-cache / refresh, and X-Cradle-Cache-TTL.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import AuthSettings, FeatureFlags, L2Settings, Settings, UpstreamSettings
from cradle.embeddings.fake import FakeEmbedder
from tests.fake_rerank import AllowReranker


async def _upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "cached-answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=False, reconstruction=False, local_1b=False,
        ),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(_upstream), base_url="http://upstream")
    app = create_app(settings=settings, embedder=FakeEmbedder(), http=http)
    with TestClient(app) as c:
        yield c


def _ask(c: TestClient, content: str, tok: str, **headers) -> httpx.Response:
    h = {"Authorization": f"Bearer {tok}"}
    h.update(headers)
    return c.post(
        "/v1/chat/completions", headers=h,
        json={"model": "qwen", "messages": [{"role": "user", "content": content}],
              "max_tokens": 8, "temperature": 0},
    )


def test_no_store_prevents_caching(client: TestClient) -> None:
    tok = "nostore"
    a = _ask(client, "hello", tok, **{"X-Cradle-Cache-Control": "no-store"})
    assert a.headers["X-Cradle-Cache"] == "MISS"
    # A normal repeat must still MISS, because the first was never stored.
    b = _ask(client, "hello", tok)
    assert b.headers["X-Cradle-Cache"] == "MISS"


def test_no_cache_skips_read_but_still_writes(client: TestClient) -> None:
    tok = "nocache"
    _ask(client, "world", tok)  # normal: stores
    # refresh: must bypass the stored entry (fresh upstream call), MISS.
    r = _ask(client, "world", tok, **{"X-Cradle-Cache-Control": "refresh"})
    assert r.headers["X-Cradle-Cache"] == "MISS"
    # but the normal path still hits (write happened on the seed).
    hit = _ask(client, "world", tok)
    assert hit.headers["X-Cradle-Cache"] == "HIT-L1"


def test_normal_request_caches(client: TestClient) -> None:
    tok = "normal"
    a = _ask(client, "greetings", tok)
    assert a.headers["X-Cradle-Cache"] == "MISS"
    b = _ask(client, "greetings", tok)
    assert b.headers["X-Cradle-Cache"] == "HIT-L1"


def test_ttl_zero_prevents_caching(client: TestClient) -> None:
    tok = "ttl0"
    _ask(client, "ephemeral", tok, **{"X-Cradle-Cache-TTL": "0"})
    b = _ask(client, "ephemeral", tok)
    assert b.headers["X-Cradle-Cache"] == "MISS"  # ttl=0 => not stored


_ANSWER = {"n": 0}


async def _versioned_upstream(request: httpx.Request) -> httpx.Response:
    """Each call answers with a new version so a stale replay is detectable."""
    _ANSWER["n"] += 1
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": f"answer-v{_ANSWER['n']}"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )


@pytest.fixture()
def l2_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    _ANSWER["n"] = 0
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=True, reconstruction=False, local_1b=False,
        ),
        l2=L2Settings(mode="local", cosine_threshold=0.90),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(_versioned_upstream), base_url="http://upstream"
    )
    app = create_app(
        settings=settings, embedder=FakeEmbedder(), http=http, reranker=AllowReranker()
    )
    with TestClient(app) as c:
        yield c


def test_refresh_replaces_the_l2_point_too(l2_client: TestClient) -> None:
    """A refresh must re-embed so the writeback replaces the stale L2 point.

    Regression: the embed used to live inside the skipped read path, so a
    refresh wrote the fresh answer to L1 only and every later L2 replay (a
    paraphrase, or the same prompt after L1 eviction) served the old answer.
    """
    tok = "refresh-l2"
    a = _ask(l2_client, "what is the capital of france", tok)
    assert a.headers["X-Cradle-Cache"] == "MISS" and "answer-v1" in a.text
    r = _ask(
        l2_client, "what is the capital of france", tok,
        **{"X-Cradle-Cache-Control": "refresh"},
    )
    assert r.headers["X-Cradle-Cache"] == "MISS" and "answer-v2" in r.text
    # Drop the exact L1 key so the next identical request must come from L2.
    l2_client.app.state.runtime.l1.clear()
    b = _ask(l2_client, "what is the capital of france", tok)
    assert b.headers["X-Cradle-Cache"] == "HIT-L2"
    assert "answer-v2" in b.text


def test_invalid_ttl_ignored(client: TestClient) -> None:
    tok = "badttl"
    a = _ask(client, "hi", tok, **{"X-Cradle-Cache-TTL": "not-a-number"})
    assert a.headers["X-Cradle-Cache"] == "MISS"
    b = _ask(client, "hi", tok)  # invalid ttl ignored => normal caching applies
    assert b.headers["X-Cradle-Cache"] == "HIT-L1"
