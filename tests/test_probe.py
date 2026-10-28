"""Cache probe mode (innovation #3): ``X-Cradle-Cache-Control: probe`` explains
the read-side decision (L1, L2 candidates, guard, rerank) without serving a
completion, writing to either cache, or calling upstream.

A near-constant embedder makes every stored prompt a top-K candidate in a
deterministic order; a scripted reranker rejects the first one so the
explanation shows both a rejected and a served candidate.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import AuthSettings, FeatureFlags, L2Settings, Settings, UpstreamSettings
from cradle.gateway.probe import PROBE_OBJECT
from tests.fake_rerank import ScriptedReranker

_CALLS = {"upstream": 0}


class NearConstantEmbedder:
    """Every prompt embeds to the same unit vector, except prompts mentioning
    'mars', which sit at cosine ~0.995 to it. So a query ranks a 'venus' point
    first (cosine 1.0) and a 'mars' point second — deterministic top-K order."""

    dim = 384

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        if "mars" in text.lower():
            v[0], v[1] = 0.995, (1 - 0.995**2) ** 0.5
        else:
            v[0] = 1.0
        return v

    def ready(self) -> bool:
        return True


async def _upstream(request: httpx.Request) -> httpx.Response:
    _CALLS["upstream"] += 1
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "an answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )


@pytest.fixture()
def probe_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    _CALLS["upstream"] = 0
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=True, reconstruction=False, local_1b=False,
        ),
        l2=L2Settings(mode="local", cosine_threshold=0.90, query_top_k=5),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(_upstream), base_url="http://upstream")
    app = create_app(
        settings=settings,
        embedder=NearConstantEmbedder(),
        http=http,
        reranker=ScriptedReranker([("venus", -5.0)]),
    )
    with TestClient(app) as c:
        yield c


def _post(c: TestClient, content: str, *, probe: bool = False, tok: str = "probe-tok"):
    headers = {"Authorization": f"Bearer {tok}"}
    if probe:
        headers["X-Cradle-Cache-Control"] = "probe"
    return c.post(
        "/v1/chat/completions", headers=headers,
        json={"model": "qwen", "temperature": 0,
              "messages": [{"role": "user", "content": content}]},
    )


def test_probe_on_empty_cache_is_a_miss_that_calls_nothing(probe_client: TestClient) -> None:
    r = _post(probe_client, "how far is venus from the sun", probe=True)
    assert r.status_code == 200
    assert r.headers["X-Cradle-Probe"] == "1"
    assert r.headers["X-Cradle-Cache"] == "MISS"
    body = r.json()
    assert body["object"] == PROBE_OBJECT
    assert body["would_call_upstream"] is True
    assert body["l1"]["hit"] is False and body["l1"]["key"]
    assert body["l2"] == {"eligible": True, "embedded": True, "hit": False, "candidates": []}
    assert _CALLS["upstream"] == 0
    # A probe never writes: the same prompt still misses for real afterwards.
    assert _post(probe_client, "how far is venus from the sun").headers["X-Cradle-Cache"] == "MISS"


def test_probe_reports_l1_hit(probe_client: TestClient) -> None:
    _post(probe_client, "what colour is mars")  # seed
    r = _post(probe_client, "what colour is mars", probe=True)
    body = r.json()
    assert r.headers["X-Cradle-Cache"] == "HIT-L1"
    assert body["cache"] == "HIT-L1" and body["would_call_upstream"] is False
    assert body["l1"]["hit"] is True
    assert _CALLS["upstream"] == 1


def test_probe_explains_every_l2_candidate(probe_client: TestClient) -> None:
    _post(probe_client, "how far is venus from the sun")  # rerank will reject this one
    _post(probe_client, "how far is mars from the sun")   # this one serves
    calls_before = _CALLS["upstream"]
    r = _post(probe_client, "how far is the red planet from the sun", probe=True)
    body = r.json()
    assert r.headers["X-Cradle-Cache"] == "HIT-L2"
    assert body["cache"] == "HIT-L2" and body["would_call_upstream"] is False
    cands = body["l2"]["candidates"]
    assert len(cands) == 2  # best-first: venus (1.0, rejected) then mars (0.995, served)
    assert cands[0]["cosine"] == 1.0 and cands[0]["rerank"].startswith("reject:")
    assert cands[0]["served"] is False
    assert cands[1]["cosine"] == 0.995 and cands[1]["rerank"].startswith("pass:")
    assert cands[1]["served"] is True
    assert all(c["guard"] == "pass" for c in cands)
    assert _CALLS["upstream"] == calls_before
    # No L1 promote happened: the probed prompt still goes through L2 for real.
    real = _post(probe_client, "how far is the red planet from the sun")
    assert real.headers["X-Cradle-Cache"] == "HIT-L2"


def test_probe_shows_guard_rejections(probe_client: TestClient) -> None:
    _post(probe_client, "convert 100 usd to eur")
    r = _post(probe_client, "convert 200 usd to eur", probe=True)
    body = r.json()
    assert body["cache"] == "MISS"
    assert body["l2"]["candidates"] == [
        {**body["l2"]["candidates"][0], "guard": "reject:numbers", "rerank": None, "served": False}
    ]
    assert r.headers["X-Cradle-Guard"] == "reject:numbers"


def test_probe_on_uncacheable_request_is_bypass(probe_client: TestClient) -> None:
    r = probe_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer probe-tok", "X-Cradle-Cache-Control": "probe"},
        json={"model": "qwen", "n": 2, "messages": [{"role": "user", "content": "hi"}]},
    )
    body = r.json()
    assert r.headers["X-Cradle-Cache"] == "BYPASS"
    assert body["cache"] == "BYPASS" and body["would_call_upstream"] is True
    assert body["l1"]["key"] is None and body["l2"]["eligible"] is False
    assert _CALLS["upstream"] == 0
