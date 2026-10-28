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


def _raw(pairs: list[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    return [(k.encode(), v.encode()) for k, v in pairs]


def test_forward_request_headers_denylist() -> None:
    out = forward_request_headers(
        _raw([
            ("Host", "cradle:8000"),
            ("Content-Length", "12"),
            ("Content-Type", "text/plain"),
            ("Connection", "keep-alive, X-Hop"),
            ("Keep-Alive", "timeout=5"),
            ("X-Hop", "per-hop"),
            ("Transfer-Encoding", "chunked"),
            ("Upgrade", "h2c"),
            ("TE", "trailers"),
            ("Expect", "100-continue"),
            ("Content-Digest", "sha-256=:abc=:"),
            ("Digest", "SHA-256=abc="),
            ("Content-MD5", "abc=="),
            ("Authorization", "Bearer x"),
            ("X-CRADLE-Cache-Control", "probe"),
            ("X-Session-Id", "s1"),
            ("Traceparent", "00-abc-def-01"),
        ])
    )
    assert out == _raw([("x-session-id", "s1"), ("traceparent", "00-abc-def-01")])


def test_forward_request_headers_empty() -> None:
    assert forward_request_headers([]) == []


def test_non_ascii_header_value_is_relayed_not_a_500(client: TestClient) -> None:
    """A UTF-8 display name must pass through byte-exact, not crash httpx's ASCII encode."""
    name = "José 名前".encode()
    r = client.post(
        "/v1/chat/completions",
        content=b'{"model":"m","messages":[{"role":"user","content":"hi"}]}',
        headers=[("content-type", "application/json"), ("x-openwebui-user-name", name)],
    )
    assert r.status_code == 200
    assert (b"x-openwebui-user-name", name) in fu.last_raw_headers


def test_repeated_headers_are_all_relayed(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        content=b'{"model":"m","messages":[{"role":"user","content":"repeat"}]}',
        headers=[
            ("content-type", "application/json"),
            ("cookie", "a=1"), ("cookie", "b=2"),
            ("x-trace", "first"), ("x-trace", "second"),
        ],
    )
    assert r.status_code == 200
    raw = fu.last_raw_headers
    assert [v for k, v in raw if k == b"cookie"] == [b"a=1", b"b=2"]
    assert [v for k, v in raw if k == b"x-trace"] == [b"first", b"second"]


def test_duplicate_authorization_forwards_the_authenticated_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keyed + pass-through: the FIRST authorization authenticates, so it is the one forwarded."""
    monkeypatch.setenv("CRADLE_TEST_KEY", "cradle-secret")
    auth = AuthSettings(keys=[AuthKey(token_env="CRADLE_TEST_KEY", tenant_id="t", user_id="u")])
    upstream = UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True)
    with _client(tmp_path, auth, upstream) as c:
        r = c.post(
            "/v1/chat/completions",
            content=b'{"model":"m","messages":[{"role":"user","content":"hi"}]}',
            headers=[
                ("content-type", "application/json"),
                ("authorization", "Bearer cradle-secret"),
                ("authorization", "Bearer totally-bogus"),
            ],
        )
    assert r.status_code == 200
    assert [v for k, v in fu.last_raw_headers if k == b"authorization"] == [b"Bearer cradle-secret"]


def test_body_session_fields_forwarded_and_share_cache(client: TestClient) -> None:
    """#82 end to end: session fields reach upstream, and a second session hits L1."""
    def ask(session: str) -> httpx.Response:
        return _chat(
            client, {"Authorization": "Bearer t"},
            session_id=session, chat_id=f"chat-{session}", metadata={"s": session}, store=True,
            messages=[{"role": "user", "content": "capital of france"}],
        )

    first = ask("a")
    assert first.headers["x-cradle-cache"] == "MISS"
    assert fu.last_payload is not None
    assert fu.last_payload["session_id"] == "a"
    assert fu.last_payload["chat_id"] == "chat-a"
    assert fu.last_payload["metadata"] == {"s": "a"}
    assert fu.last_payload["store"] is True
    assert ask("b").headers["x-cradle-cache"] == "HIT-L1"
