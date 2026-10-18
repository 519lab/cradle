"""Top-K L2 (enhancement #1): when the nearest candidate is rejected by the
guard/rerank, a valid paraphrase at rank 2..K still serves instead of a miss.

Uses a constant-vector embedder so every stored point is an equal-cosine
candidate (Qdrant returns all K), and a scripted reranker that rejects one
candidate's embed_text and passes another. Without top-K this is a miss; with
top-K the passing candidate serves.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import AuthSettings, FeatureFlags, L2Settings, Settings, UpstreamSettings


class ConstantEmbedder:
    dim = 384

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        v[0] = 1.0
        return v

    def ready(self) -> bool:
        return True


async def _fake_upstream(request: httpx.Request) -> httpx.Response:
    body = request.content.decode().lower()
    ans = "reddish" if "mars" in body else "answer-for-" + body[-8:]
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": ans}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    )


class RejectFirstReranker:
    """Rejects any candidate mentioning 'venus'; passes everything else high."""

    def score(self, query: str, candidate: str) -> float:
        return -10.0 if "venus" in candidate.lower() else 100.0


@pytest.fixture()
def topk_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=True, l2_rerank=True, reconstruction=False, local_1b=False,
        ),
        l2=L2Settings(mode="local", query_top_k=5),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(_fake_upstream), base_url="http://upstream"
    )
    app = create_app(
        settings=settings, embedder=ConstantEmbedder(), http=http, reranker=RejectFirstReranker()
    )
    with TestClient(app) as c:
        yield c


def _ask(c: TestClient, content: str, token: str) -> httpx.Response:
    return c.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": "qwen", "messages": [{"role": "user", "content": content}],
              "max_tokens": 8, "temperature": 0},
    )


def test_rank2_paraphrase_serves_when_rank1_rejected(topk_client: TestClient) -> None:
    tok = "topk"
    # Seed two L2 entries under one tenant: one the reranker will reject (venus),
    # one it will accept (mars). Constant embedder => both are equal-cosine
    # candidates for any later query.
    r1 = _ask(topk_client, "Tell me about Venus the planet", tok)
    assert r1.headers["X-Cradle-Cache"] == "MISS"
    r2 = _ask(topk_client, "Tell me about Mars the planet", tok)
    assert r2.headers["X-Cradle-Cache"] == "MISS"
    mars_answer = r2.json()["choices"][0]["message"]["content"]

    # A new query. Both stored candidates match on cosine. The 'venus' one is
    # rejected by rerank; the 'mars' one passes => we must serve it (HIT-L2),
    # not fall through to a miss.
    r3 = _ask(topk_client, "Some other planet question entirely", tok)
    assert r3.headers["X-Cradle-Cache"] == "HIT-L2"
    assert r3.json()["choices"][0]["message"]["content"] == mars_answer
