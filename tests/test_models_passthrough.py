"""/v1/models auto-passthrough: in a single-backend (fallback-only) deploy,
Cradle reflects the upstream's real models; with named routes it keeps the
static list; and an upstream error falls back to the static list (never 500)."""

from __future__ import annotations

import httpx
import pytest

from cradle.config import Settings, UpstreamSettings
from cradle.upstream.openai import list_models


def _upstream_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json={"object": "list", "data": [{"id": "real-model-xyz", "object": "model"}]}
    )


def _upstream_down(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"error": "unavailable"})


async def _call(settings: Settings, transport: httpx.MockTransport) -> dict:
    async with httpx.AsyncClient(transport=transport, base_url="http://up") as client:
        return await list_models(client, settings)


@pytest.mark.asyncio
async def test_single_backend_reflects_real_upstream(monkeypatch) -> None:
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    s = Settings(upstream=UpstreamSettings(base_url="http://up/v1", models=["placeholder"]))
    out = await _call(s, httpx.MockTransport(_upstream_ok))
    ids = [m["id"] for m in out["data"]]
    assert ids == ["real-model-xyz"], "single-backend should reflect the real upstream model"


@pytest.mark.asyncio
async def test_upstream_down_falls_back_to_static(monkeypatch) -> None:
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    s = Settings(upstream=UpstreamSettings(base_url="http://up/v1", models=["placeholder"]))
    out = await _call(s, httpx.MockTransport(_upstream_down))
    ids = [m["id"] for m in out["data"]]
    assert ids == ["placeholder"], "upstream down must fall back to the static list, not 500"


@pytest.mark.asyncio
async def test_named_routes_keep_static_list(monkeypatch) -> None:
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    s = Settings(
        upstream=UpstreamSettings(base_url="http://up/v1", models=["fallback-m"]),
        upstreams={"openai": UpstreamSettings(base_url="http://oai/v1", models=["gpt-x"])},
        routes=[{"model": "gpt-*", "to": "openai"}],
    )
    # Even with the upstream reachable, named routes => static advertised list.
    out = await _call(s, httpx.MockTransport(_upstream_ok))
    ids = sorted(m["id"] for m in out["data"])
    assert "real-model-xyz" not in ids, "named-route config must not proxy one backend's models"
    assert "fallback-m" in ids
