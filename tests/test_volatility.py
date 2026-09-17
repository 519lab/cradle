"""Volatility guard (innovation #4): time-sensitive prompts get a short TTL.

Unit tests pin every pattern in the table; pipeline tests prove the TTL that
actually lands in the stored record, that an explicit client TTL wins, that a
system prompt carrying the date does not trip the guard, and that
``volatile_ttl_s: 0`` means "never store".
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.cache import l1 as l1mod
from cradle.cache.records import CanonicalMessage
from cradle.cache.volatility import volatile_reason, volatile_reason_for
from cradle.config import AuthSettings, CacheSettings, FeatureFlags, Settings, UpstreamSettings
from cradle.embeddings.fake import FakeEmbedder
from cradle.gateway.models import ChatRequest
from cradle.normalize import cache_namespace, canonicalize, l1_key
from cradle.tenancy import principal_from_forwarded_token
from cradle.upstream.route import resolve_upstream


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("what is the latest version of python", "time"),
        ("Who is the current CEO of Ford?", "time"),
        ("what's the date today", "time"),
        ("what time is it in Tokyo", "time"),
        ("as of this month, how many users", "time"),
        ("what is the price of a Tesla Model 3", "market"),
        ("USD to EUR exchange rate", "market"),
        ("how much does a gallon of milk cost", "market"),
        ("what is the weather in Toronto", "weather"),
        ("give me the forecast for the weekend", "weather"),
        ("any news about the election", "news"),
        ("what happened in Ottawa", "news"),
    ],
)
def test_volatile_patterns(text: str, reason: str) -> None:
    assert volatile_reason(text) == reason


@pytest.mark.parametrize(
    "text",
    [
        "explain version control best practices",
        "now write a function that sorts a list",
        "score this essay out of ten",
        "what is the capital of France",
        "summarize the following meeting notes",
        "how do I set a timeout in httpx",
    ],
)
def test_stable_prompts_are_not_volatile(text: str) -> None:
    assert volatile_reason(text) is None


def test_system_prompt_date_does_not_count() -> None:
    msgs = [
        CanonicalMessage(role="system", content="Today's date is 2026-09-17. Be brief."),
        CanonicalMessage(role="user", content="what is the capital of France"),
    ]
    assert volatile_reason_for(msgs) is None
    msgs.append(CanonicalMessage(role="user", content="and the latest population figure?"))
    assert volatile_reason_for(msgs) == "time"


async def _upstream(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "an answer"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    )


def _make_client(tmp_path: Path, cache: CacheSettings) -> TestClient:
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=False, reconstruction=False, local_1b=False,
        ),
        cache=cache,
        upstream=UpstreamSettings(base_url="http://upstream/v1", pass_through_client_auth=True),
    )
    http = httpx.AsyncClient(transport=httpx.MockTransport(_upstream), base_url="http://upstream")
    return TestClient(create_app(settings=settings, embedder=FakeEmbedder(), http=http))


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    with _make_client(tmp_path, CacheSettings(ttl_s=86400, volatile_ttl_s=300)) as c:
        yield c


@pytest.fixture()
def never_store_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    with _make_client(tmp_path, CacheSettings(ttl_s=86400, volatile_ttl_s=0)) as c:
        yield c


def _ask(c: TestClient, content: str, **headers) -> httpx.Response:
    h = {"Authorization": "Bearer vol-tok", **headers}
    return c.post(
        "/v1/chat/completions", headers=h,
        json={"model": "qwen", "temperature": 0,
              "messages": [{"role": "user", "content": content}]},
    )


def _stored_ttl(c: TestClient, content: str) -> int | None:
    rt = c.app.state.runtime
    req = ChatRequest(model="qwen", temperature=0,
                      messages=[{"role": "user", "content": content}])
    principal = principal_from_forwarded_token("vol-tok")
    name, up = resolve_upstream(rt.settings, "qwen")
    canonical = canonicalize(req, principal, rt.settings,
                             backend_namespace=cache_namespace(name, up.base_url))
    rec = l1mod.get_sync(rt.l1, l1_key(canonical))
    return None if rec is None else rec.ttl_s


def test_volatile_prompt_gets_short_ttl(client: TestClient) -> None:
    r = _ask(client, "what is the latest version of python")
    assert r.headers["X-Cradle-Cache"] == "MISS"
    assert r.headers["X-Cradle-Volatile"] == "time"
    assert _stored_ttl(client, "what is the latest version of python") == 300
    # It is still cached (short TTL, not no-store).
    assert _ask(client, "what is the latest version of python").headers["X-Cradle-Cache"] == "HIT-L1"


def test_stable_prompt_keeps_default_ttl(client: TestClient) -> None:
    r = _ask(client, "what is the capital of France")
    assert "X-Cradle-Volatile" not in r.headers
    assert _stored_ttl(client, "what is the capital of France") == 86400


def test_client_ttl_header_wins_over_guard(client: TestClient) -> None:
    r = _ask(client, "what is the weather in Toronto", **{"X-Cradle-Cache-TTL": "3600"})
    assert "X-Cradle-Volatile" not in r.headers
    assert _stored_ttl(client, "what is the weather in Toronto") == 3600


def test_volatile_ttl_zero_never_stores(never_store_client: TestClient) -> None:
    r = _ask(never_store_client, "bitcoin price")
    assert r.headers["X-Cradle-Volatile"] == "market"
    assert _stored_ttl(never_store_client, "bitcoin price") is None
    assert _ask(never_store_client, "bitcoin price").headers["X-Cradle-Cache"] == "MISS"


def test_guard_can_be_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    with _make_client(tmp_path, CacheSettings(volatility_guard=False)) as c:
        r = _ask(c, "what is the latest version of python")
        assert "X-Cradle-Volatile" not in r.headers
        assert _stored_ttl(c, "what is the latest version of python") == 86400
