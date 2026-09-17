"""Integration test: the cross-encoder rerank stage forces a real miss through
the handler for an entity swap the numbers/negation guard cannot catch (#5).

A constant-vector stub embedder makes every pair collide on cosine (=1.0) so the
cosine gate always hits; the guard passes (no number/negation difference); only
the scripted reranker separates them. This mirrors the real failure mode
(France/Germany: cosine 0.905, guard blind, reranker rejects) deterministically.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import AuthSettings, FeatureFlags, Settings, UpstreamSettings
from tests.fake_rerank import ScriptedReranker


class ConstantEmbedder:
    dim = 384

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        v[0] = 1.0
        return v

    def ready(self) -> bool:
        return True


async def _fake_upstream(request: httpx.Request) -> httpx.Response:
    body = request.content.decode()
    answer = "berlin" if "Germany" in body else "paris" if "France" in body else "other"
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{answer}", "object": "chat.completion", "created": 1,
            "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": answer}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    )


def _make_client(tmp_path: Path, reranker: object) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=True, l2_rerank=True, reconstruction=False, local_1b=False,
        ),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(_fake_upstream), base_url="http://upstream"
    )
    app = create_app(
        settings=settings, embedder=ConstantEmbedder(), http=http, reranker=reranker
    )
    return TestClient(app)


def _ask(client: TestClient, content: str, token: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": "qwen", "messages": [{"role": "user", "content": content}],
              "max_tokens": 8, "temperature": 0},
    )


@pytest.fixture()
def reject_germany_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    # Reranker scores the Germany-vs-France pair below threshold, everything else high.
    reranker = ScriptedReranker(rules=[("Germany", 1.0)])
    with _make_client(tmp_path, reranker) as c:
        yield c


def test_entity_swap_rejected_by_rerank(reject_germany_client: TestClient) -> None:
    tok = "rerank-entity"
    a = _ask(reject_germany_client, "What is the capital of France?", tok)
    assert a.headers["X-Cradle-Cache"] == "MISS"
    # Germany collides on cosine (=1.0) and passes the guard, but the reranker
    # scores it 1.0 < 4.0 -> reject -> real miss -> correct 'berlin', not 'paris'.
    b = _ask(reject_germany_client, "What is the capital of Germany?", tok)
    assert b.headers["X-Cradle-Cache"] == "MISS"
    assert b.headers["X-Cradle-Rerank"].startswith("reject:")
    assert b.json()["choices"][0]["message"]["content"] == "berlin"
    assert int(b.headers["X-Cradle-Upstream-Tokens"]) > 0


def test_paraphrase_passes_rerank(reject_germany_client: TestClient) -> None:
    tok = "rerank-pass"
    _ask(reject_germany_client, "What is the capital of France?", tok)
    # A second France query: cosine hit, guard pass, reranker scores 100 -> serve.
    b = _ask(reject_germany_client, "What is the capital of France?", tok)
    # Exact repeat is L1; either way it must not be a rerank rejection.
    assert b.headers["X-Cradle-Cache"] in {"HIT-L1", "HIT-L2"}
    assert "reject:" not in b.headers.get("X-Cradle-Rerank", "")
    assert b.json()["choices"][0]["message"]["content"] == "paris"


class BoomReranker:
    def score(self, query: str, candidate: str) -> float:
        raise RuntimeError("onnx down")


def test_rerank_fail_open_serves_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken reranker must not turn L2 hits into misses (fail-open)."""
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    with _make_client(tmp_path, BoomReranker()) as c:
        _ask(c, "A unique question about aardvarks", "fo-tok")
        # Different prompt, cosine=1.0 hit, guard pass, reranker raises -> fail-open serve.
        b = _ask(c, "A different question about zebras entirely", "fo-tok")
        assert b.headers["X-Cradle-Cache"] == "HIT-L2"
        assert b.headers["X-Cradle-Rerank"] == "fail-open"
