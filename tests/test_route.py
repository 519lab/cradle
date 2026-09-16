from __future__ import annotations

import pytest
from pydantic import ValidationError

from cradle.config import RouteRule, Settings, UpstreamSettings
from cradle.upstream.route import advertised_models, resolve_upstream


def _settings() -> Settings:
    return Settings(
        upstream=UpstreamSettings(base_url="http://llama:8080/v1", models=["local-llama"]),
        upstreams={
            "openai": UpstreamSettings(base_url="https://api.openai.com/v1", models=["gpt-4o"]),
            "xai": UpstreamSettings(base_url="https://api.x.ai/v1", models=["grok-4"]),
        },
        routes=[
            RouteRule(model="gpt-*", to="openai"),
            RouteRule(model="o1-*", to="openai"),
            RouteRule(model="grok-*", to="xai"),
            RouteRule(model="*", to="default"),
        ],
    )


def test_resolve_gpt_and_grok() -> None:
    s = _settings()
    name, u = resolve_upstream(s, "gpt-4o-mini")
    assert name == "openai"
    assert u.base_url.startswith("https://api.openai.com")
    name, u = resolve_upstream(s, "grok-4")
    assert name == "xai"
    name, u = resolve_upstream(s, "local-llama")
    assert name == "default"
    assert "llama:8080" in u.base_url


def test_no_routes_uses_default() -> None:
    s = Settings(upstream=UpstreamSettings(base_url="http://only:1/v1"))
    name, u = resolve_upstream(s, "gpt-4o")
    assert name == "default"
    assert u.base_url.endswith(":1/v1")


def test_unknown_route_target_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown upstream"):
        Settings(routes=[RouteRule(model="gpt-*", to="missing")])


def test_advertised_models_union() -> None:
    s = _settings()
    ids = advertised_models(s)
    assert ids[0] == "local-llama"
    assert "gpt-4o" in ids
    assert "grok-4" in ids
