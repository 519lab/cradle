"""Client request headers reach the upstream unless denylisted (#81)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import tests.fake_upstream as fu  # last_headers side channel
from cradle.app import create_app
from cradle.config import AuthKey, AuthSettings, FeatureFlags, Settings, UpstreamSettings
from cradle.embeddings.fake import FakeEmbedder
from cradle.upstream.openai import forward_request_headers
from tests.fake_upstream import fake_app

APP_HEADERS = {
    "X-Session-Id": "sess-123",
    "X-OpenWebUI-Chat-Id": "chat-456",
    "X-OpenWebUI-User-Id": "u-789",
    "X-Foo": "bar",
}


def _client(tmp_path: Path, auth: AuthSettings, upstream: UpstreamSettings) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=auth,
        features=FeatureFlags(
            cache=True, compression=False, structure=False, l2=False,
            reconstruction=False, local_1b=False,
        ),
        upstream=upstream,
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_app), base_url="http://upstream")
    return TestClient(create_app(settings=settings, embedder=FakeEmbedder(), http=http))


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    upstream = UpstreamSettings(base_url="http://upstream/v1", models_passthrough=True)
    with _client(tmp_path, AuthSettings(keys=[]), upstream) as c:
        yield c


def _chat(client: TestClient, headers: dict[str, str], **body) -> httpx.Response:
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **body}
    return client.post("/v1/chat/completions", headers=headers, json=payload)


def _assert_app_headers_forwarded() -> None:
    for name, value in APP_HEADERS.items():
        assert fu.last_headers.get(name.lower()) == value, name


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({}, id="json-miss"),
        pytest.param({"stream": True}, id="wrap-stream"),
        pytest.param({"stream": True, "n": 2}, id="bypass-stream"),
        pytest.param({"stream": True, "tools": [{"type": "function", "function": {"name": "f"}}]},
                     id="tool-stream"),
    ],
)
def test_app_headers_reach_upstream_on_every_path(client: TestClient, body: dict) -> None:
    fu.last_headers = {}
    r = _chat(client, {**APP_HEADERS, "Authorization": "Bearer sk-client"}, **body)
    assert r.status_code == 200
    _assert_app_headers_forwarded()
    assert fu.last_headers["authorization"] == "Bearer sk-client"
    assert fu.last_headers["content-type"] == "application/json"


def test_denylisted_headers_are_stripped(client: TestClient) -> None:
    fu.last_headers = {}
    r = _chat(
        client,
        {
            **APP_HEADERS,
            "X-Cradle-Cache-Control": "no-store",
            "X-Cradle-Cache-TTL": "5",
            "Accept-Encoding": "zstd",
            "Proxy-Authorization": "Basic cHJveHk=",
        },
    )
    assert r.status_code == 200
    _assert_app_headers_forwarded()
    assert not any(k.startswith("x-cradle-") for k in fu.last_headers)
    assert "proxy-authorization" not in fu.last_headers
    # httpx sets its own accept-encoding; the client's zstd must not be relayed.
    assert "zstd" not in fu.last_headers.get("accept-encoding", "")
    assert fu.last_headers["host"] == "upstream"


def test_models_passthrough_forwards_headers(client: TestClient) -> None:
    fu.last_headers = {}
    r = client.get("/v1/models", headers=APP_HEADERS)
    assert r.status_code == 200
    _assert_app_headers_forwarded()


def test_keyed_mode_never_leaks_cradle_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRADLE_TEST_KEY", "cradle-secret")
    monkeypatch.setenv("CRADLE_UPSTREAM_API_KEY", "sk-upstream")
    auth = AuthSettings(keys=[AuthKey(token_env="CRADLE_TEST_KEY", tenant_id="t", user_id="u")])
    upstream = UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=False)
    with _client(tmp_path, auth, upstream) as c:
        fu.last_headers = {}
        r = _chat(c, {**APP_HEADERS, "Authorization": "Bearer cradle-secret"})
    assert r.status_code == 200
    _assert_app_headers_forwarded()
    assert fu.last_headers["authorization"] == "Bearer sk-upstream"
    assert "cradle-secret" not in str(fu.last_headers)


def test_forward_request_headers_denylist() -> None:
    out = forward_request_headers(
        {
            "Host": "cradle:8000",
            "Content-Length": "12",
            "Content-Type": "text/plain",
            "Connection": "keep-alive, X-Hop",
            "Keep-Alive": "timeout=5",
            "X-Hop": "per-hop",
            "Transfer-Encoding": "chunked",
            "Upgrade": "h2c",
            "TE": "trailers",
            "Expect": "100-continue",
            "Authorization": "Bearer x",
            "X-CRADLE-Cache-Control": "probe",
            "X-Session-Id": "s1",
            "Traceparent": "00-abc-def-01",
        }
    )
    assert out == {"x-session-id": "s1", "traceparent": "00-abc-def-01"}


def test_forward_request_headers_empty() -> None:
    assert forward_request_headers({}) == {}
