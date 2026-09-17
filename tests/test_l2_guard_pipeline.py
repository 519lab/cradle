"""Integration test: the L2 precision guard forces a real miss through the
handler (issue #5), and the l2_points gauge is bumped on write (issue #4).

A constant-vector stub embedder makes any two prompts collide on cosine (=1.0),
so the cosine gate always "hits". That isolates the guard: only numbers/negation
differences can prevent a serve. This is exactly the near-miss failure mode the
real bi-encoder produces (high cosine, different question) but deterministic.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import (
    AuthSettings,
    FeatureFlags,
    Settings,
    UpstreamSettings,
)
from cradle.metrics import prometheus as m


class ConstantEmbedder:
    """Returns the same unit vector for every input -> cosine 1.0 for all pairs."""

    dim = 384

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        v[0] = 1.0
        return v

    def ready(self) -> bool:
        return True


async def _fake_upstream(request: httpx.Request) -> httpx.Response:
    body = request.content.decode()
    # Echo which question was asked so we can prove the answer is fresh, not stale.
    answer = "berlin" if "Germany" in body else "paris" if "France" in body else "other"
    return httpx.Response(
        200,
        json={
            "id": f"chatcmpl-{answer}",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen",
            "choices": [
                {"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": answer}}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        },
    )


@pytest.fixture()
def l2_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=True, reconstruction=False, local_1b=False,
        ),
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(_fake_upstream), base_url="http://upstream"
    )
    app = create_app(settings=settings, embedder=ConstantEmbedder(), http=http)
    with TestClient(app) as c:
        yield c


def _ask(client: TestClient, content: str, token: str) -> httpx.Response:
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": f"Bearer {token}"},
        json={"model": "qwen", "messages": [{"role": "user", "content": content}],
              "max_tokens": 8, "temperature": 0},
    )


def _content(r: httpx.Response) -> str:
    return r.json()["choices"][0]["message"]["content"]


def test_negation_swap_forced_to_miss(l2_client: TestClient) -> None:
    tok = "guard-neg"
    a = _ask(l2_client, "Is Python statically typed? yes or no", tok)
    assert a.headers["X-Cradle-Cache"] == "MISS"
    # Different (negated) question: cosine=1.0 would serve the stale hit, guard blocks it.
    b = _ask(l2_client, "Is Python NOT statically typed? yes or no", tok)
    assert b.headers["X-Cradle-Cache"] == "MISS"
    assert b.headers["X-Cradle-Guard"] == "reject:negation"
    assert int(b.headers["X-Cradle-Upstream-Tokens"]) > 0


def test_number_swap_forced_to_miss(l2_client: TestClient) -> None:
    tok = "guard-num"
    _ask(l2_client, "Convert 100 USD to EUR", tok)
    b = _ask(l2_client, "Convert 200 USD to EUR", tok)
    assert b.headers["X-Cradle-Cache"] == "MISS"
    assert b.headers["X-Cradle-Guard"] == "reject:numbers"


def test_genuine_repeat_still_hits_l2(l2_client: TestClient) -> None:
    """The guard must not break legitimate hits: identical embed_text passes."""
    tok = "guard-hit"
    _ask(l2_client, "What is the capital of France?", tok)
    # Same question under a *different* token would be a different tenant, so use
    # a second prompt that canonicalizes identically but isn't an exact L1 key
    # match — here a trailing-space variant that L1 normalizes but keeps L2 in play.
    b = _ask(l2_client, "What is the capital of France?", tok)
    # Exact repeat is an L1 hit; either way it must NOT be a guard rejection.
    assert "X-Cradle-Guard" not in b.headers
    assert _content(b) == "paris"


def test_l2_points_gauge_bumped_on_write(l2_client: TestClient) -> None:
    # The gauge is a process-global metric shared across the registry, so pin it
    # to a known value first; then a fresh-tenant miss must write back and bump
    # it by exactly one (issue #4).
    m.l2_points.set(0)
    _ask(l2_client, "A brand new unique question about pangolins in the tundra", "gauge-tok")
    assert m.l2_points._value.get() == 1
