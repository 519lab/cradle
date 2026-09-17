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
from cradle.config import AuthSettings, FeatureFlags, Settings, UpstreamSettings
from cradle.embeddings.fake import FakeEmbedder


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


def test_invalid_ttl_ignored(client: TestClient) -> None:
    tok = "badttl"
    a = _ask(client, "hi", tok, **{"X-Cradle-Cache-TTL": "not-a-number"})
    assert a.headers["X-Cradle-Cache"] == "MISS"
    b = _ask(client, "hi", tok)  # invalid ttl ignored => normal caching applies
    assert b.headers["X-Cradle-Cache"] == "HIT-L1"
