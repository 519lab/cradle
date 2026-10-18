"""Pass-through payload fidelity (issue #26).

Cradle must forward only the sampling fields the client actually set. It must
not inject its own model defaults (temperature 1.0, top_p 1.0, n 1, penalties
0.0, stream false) onto a backend that has its own defaults (llama.cpp, vLLM,
Ollama). These tests capture the body the fake upstream receives and assert on
which top-level fields survive.
"""

from __future__ import annotations

import time

import httpx
import pytest
from fastapi.testclient import TestClient

import tests.fake_upstream as fu  # last_payload side channel
from cradle.app import create_app
from cradle.config import (
    AuthSettings,
    FeatureFlags,
    L2Settings,
    Settings,
    UpstreamSettings,
)
from cradle.embeddings.fake import FakeEmbedder
from tests.fake_rerank import AllowReranker
from tests.fake_upstream import fake_app

_INJECTED = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "n", "stream")


class PromptCollidingEmbedder:
    """Every role-framed prompt -> one constant vector (so differently-worded
    prompts collide as an L2 candidate), answers -> content-hash (FakeEmbedder).
    Mirrors tests/test_audit.py; lets a paraphrase force an L2 hit + audit
    without also being an L1 exact-key hit."""

    dim = 384

    def __init__(self) -> None:
        self._fake = FakeEmbedder()

    def embed(self, text: str) -> list[float]:
        if text.startswith(("user:", "system:", "developer:")):
            v = [0.0] * self.dim
            v[0] = 1.0
            return v
        return self._fake.embed(text)

    def ready(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_payload() -> None:
    fu.last_payload = None


def _post(client: TestClient, auth_header: dict[str, str], body: dict) -> httpx.Response:
    return client.post("/v1/chat/completions", headers=auth_header, json=body)


def test_omitted_sampling_fields_are_not_forwarded(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    r = _post(
        client,
        auth_header,
        {"model": "qwen", "messages": [{"role": "user", "content": "hi there"}]},
    )
    assert r.status_code == 200
    assert fu.last_payload is not None
    for field in _INJECTED:
        assert field not in fu.last_payload, f"Cradle injected {field!r} the client never sent"
    # The two fields the client did send always survive.
    assert fu.last_payload["model"] == "qwen"
    assert "messages" in fu.last_payload


def test_explicit_sampling_fields_are_forwarded(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    r = _post(
        client,
        auth_header,
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi there"}],
            "temperature": 0.2,
            "top_p": 0.8,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
        },
    )
    assert r.status_code == 200
    assert fu.last_payload is not None
    assert fu.last_payload["temperature"] == 0.2
    assert fu.last_payload["top_p"] == 0.8
    assert fu.last_payload["presence_penalty"] == 0.5
    assert fu.last_payload["frequency_penalty"] == 0.3


def test_explicit_default_value_is_forwarded(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    # temperature: 1.0 sent deliberately — proves we key on *set*, not on value.
    r = _post(
        client,
        auth_header,
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi there"}],
            "temperature": 1.0,
        },
    )
    assert r.status_code == 200
    assert fu.last_payload is not None
    assert fu.last_payload["temperature"] == 1.0


def test_extra_vendor_field_is_forwarded(
    client: TestClient, auth_header: dict[str, str]
) -> None:
    # A backend-specific knob (top_k) the client set must ride through untouched.
    r = _post(
        client,
        auth_header,
        {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi there"}],
            "top_k": 40,
        },
    )
    assert r.status_code == 200
    assert fu.last_payload is not None
    assert fu.last_payload["top_k"] == 40


def _audit_client(tmp_path) -> TestClient:
    """A client with L2 on, audit_rate=1.0, an always-agree rerank judge, and a
    prompt-colliding embedder so a paraphrase lands as an L2 hit (not L1) and
    every hit triggers exactly one audit upstream call."""
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        upstream=UpstreamSettings(
            base_url="http://upstream/v1",
            models_passthrough=False,
            pass_through_client_auth=True,
        ),
        features=FeatureFlags(
            cache=True,
            compression=False,
            structure=False,
            l2=True,
            reconstruction=False,
            local_1b=False,
        ),
        l2=L2Settings(mode="local", cosine_threshold=0.90, audit_rate=1.0, audit_judge="rerank"),
    )
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream"
    )
    app = create_app(
        settings=settings,
        embedder=PromptCollidingEmbedder(),
        http=http,
        reranker=AllowReranker(),
    )
    return TestClient(app)


def _drain(c: TestClient, timeout_s: float = 5.0) -> None:
    rt = c.app.state.runtime
    deadline = time.monotonic() + timeout_s
    while rt.audit_tasks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not rt.audit_tasks, "audit tasks did not finish"


def test_audit_payload_omits_injected_sampling_fields(tmp_path) -> None:
    # Seed L2 with one prompt, then ask a differently-worded prompt: it collides
    # at L2 (not L1), serves an L2 hit, and its audit (rate 1.0) re-asks upstream.
    # The audit's forwarded body is what we check.
    with _audit_client(tmp_path) as c:
        r1 = c.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "audit me please"}]},
        )
        assert r1.status_code == 200
        assert r1.headers.get("X-Cradle-Cache") == "MISS"
        fu.last_payload = None
        r2 = c.post(
            "/v1/chat/completions",
            json={"model": "qwen", "messages": [{"role": "user", "content": "please audit me now"}]},
        )
        assert r2.status_code == 200
        assert r2.headers.get("X-Cradle-Cache") == "HIT-L2"
        assert r2.headers.get("X-Cradle-Audit") == "scheduled"
        _drain(c)
    # The audit ran and re-asked upstream; its payload must be a pass-through.
    assert fu.last_payload is not None
    for field in _INJECTED:
        assert field not in fu.last_payload, f"audit injected {field!r} the client never sent"
    assert fu.last_payload["model"] == "qwen"
