from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import tests.fake_upstream as fu  # last_authorization side channel
from cradle.app import create_app
from cradle.config import AuthSettings, FeatureFlags, Settings, UpstreamSettings
from cradle.embeddings.fake import FakeEmbedder
from cradle.tenancy import principal_from_forwarded_token
from tests.fake_upstream import fake_app


@pytest.fixture
def intercept_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True,
            compression=False,
            structure=False,
            l2=False,
            reconstruction=False,
            local_1b=False,
        ),
        upstream=UpstreamSettings(
            base_url="http://upstream/v1",
            pass_through_client_auth=True,
        ),
    )
    http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream"
    )
    app = create_app(settings=settings, embedder=FakeEmbedder(), http=http)
    with TestClient(app) as c:
        yield c


def test_no_cradle_key_forwards_client_bearer(intercept_client: TestClient) -> None:
    r = intercept_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-client-secret"},
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert fu.last_authorization == "Bearer sk-client-secret"


def test_no_authorization_still_proxies(intercept_client: TestClient) -> None:
    r = intercept_client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert fu.last_authorization in (None, "")


def test_cache_isolated_by_forwarded_token(intercept_client: TestClient) -> None:
    payload = {"model": "m", "messages": [{"role": "user", "content": "same-prompt-intercept"}]}
    a = intercept_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-alice"},
        json=payload,
    )
    b = intercept_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-bob"},
        json=payload,
    )
    assert a.status_code == 200
    assert b.status_code == 200
    assert a.headers["X-Cradle-Cache"] != "HIT-L1"
    assert b.headers["X-Cradle-Cache"] != "HIT-L1"
    a2 = intercept_client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer sk-alice"},
        json=payload,
    )
    assert a2.headers["X-Cradle-Cache"] == "HIT-L1"


def test_metrics_open_in_intercept_mode(intercept_client: TestClient) -> None:
    r = intercept_client.get("/metrics")
    assert r.status_code == 200
    assert b"cradle_" in r.content


def test_forwarded_token_principal_stable() -> None:
    p1 = principal_from_forwarded_token("sk-alice")
    p2 = principal_from_forwarded_token("sk-alice")
    p3 = principal_from_forwarded_token("sk-bob")
    assert p1.tenant_id == p2.tenant_id == p1.user_id
    assert p1.tenant_id != p3.tenant_id
    assert p1.key_id == "forwarded"
