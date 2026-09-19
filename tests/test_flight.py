"""Single-flight (#57): concurrent identical cacheable misses share one upstream call.

TestClient is synchronous and serializes requests, so it cannot produce concurrency.
These tests drive the ASGI app directly on one event loop with an httpx.AsyncClient
and asyncio.gather, and count upstream calls via a controllable fake upstream whose
handler blocks on an Event until the test observes the expected follower count in
runtime.flights — deterministic, no sleeps-as-synchronization.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import asynccontextmanager

import httpx
import pytest

from cradle.app import create_app
from cradle.config import (
    AuthKey,
    AuthSettings,
    CacheSettings,
    FeatureFlags,
    L2Settings,
    Settings,
    UpstreamSettings,
)
from cradle.embeddings.fake import FakeEmbedder
from tests.fake_rerank import AllowReranker


class GatedUpstream:
    """A fake upstream that counts calls and can block the leader until released."""

    def __init__(self) -> None:
        self.calls = 0
        self.release = asyncio.Event()
        self.blocking = True  # when False, respond immediately (no gating)

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        body = json.loads(request.content)
        stream = body.get("stream", False)
        if self.blocking:
            await self.release.wait()
        content = "ACK-" + str(body["messages"][-1]["content"])[:8]
        if stream:
            def gen() -> Iterator[bytes]:
                cid, created, model = "up-1", 1, body["model"]
                for frag in ["role", content]:
                    if frag == "role":
                        d = {"id": cid, "created": created, "model": model,
                             "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                    else:
                        d = {"id": cid, "created": created, "model": model,
                             "choices": [{"index": 0, "delta": {"content": frag}, "finish_reason": None}]}
                    yield f"data: {json.dumps(d)}\n\n".encode()
                fin = {"id": cid, "created": created, "model": model,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}}
                yield f"data: {json.dumps(fin)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            return httpx.Response(200, content=b"".join(gen()),
                                  headers={"content-type": "text/event-stream"})
        completion = {
            "id": "up-1", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        return httpx.Response(200, json=completion)


def _settings(tmp_path, api_key: str, singleflight: bool = True) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]),
        features=FeatureFlags(cache=True, compression=True, l2=False, reconstruction=True),
        l2=L2Settings(mode="local", cosine_threshold=0.90),
        upstream=UpstreamSettings(base_url="http://upstream/v1", models_passthrough=False),
        cache=CacheSettings(singleflight=singleflight),
    )


@asynccontextmanager
async def _driver(tmp_path, api_key, singleflight=True):
    up = GatedUpstream()
    http = httpx.AsyncClient(transport=httpx.MockTransport(up.handler), base_url="http://upstream")
    app = create_app(
        settings=_settings(tmp_path, api_key, singleflight),
        embedder=FakeEmbedder(), http=http, reranker=AllowReranker(),
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield up, app, client
    await http.aclose()


async def _wait_for_followers(app, n: int, timeout: float = 5.0) -> str:
    """Poll runtime.flights until a flight has n followers; return its key."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        for k, f in app.state.runtime.flights.items():
            if f.followers >= n:
                return k
        await asyncio.sleep(0.005)
    raise AssertionError(f"no flight reached {n} followers; flights={app.state.runtime.flights}")


def _auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _body(content: str, stream: bool = False, **extra) -> dict:
    b = {"model": "m", "messages": [{"role": "user", "content": content}], "stream": stream}
    b.update(extra)
    return b


@pytest.mark.asyncio
async def test_n_concurrent_json_one_upstream_call(tmp_path, api_key):
    async with _driver(tmp_path, api_key) as (up, app, client):
        N = 5
        auth = _auth(api_key)
        tasks = [
            asyncio.ensure_future(client.post("/v1/chat/completions", headers=auth, json=_body("hi")))
            for _ in range(N)
        ]
        # Wait until N-1 followers have coalesced onto the one leader, then release.
        await _wait_for_followers(app, N - 1)
        up.release.set()
        resps = await asyncio.gather(*tasks)

    assert up.calls == 1, f"expected 1 upstream call, got {up.calls}"
    assert all(r.status_code == 200 for r in resps)
    bodies = [r.json()["choices"][0]["message"]["content"] for r in resps]
    assert len(set(bodies)) == 1, f"all callers must get the same answer: {bodies}"
    followers = sum(1 for r in resps if r.headers.get("X-Cradle-Flight") == "follower")
    assert followers == N - 1, f"expected {N - 1} followers, got {followers}"
    # registry drained
    assert app.state.runtime.flights == {}


@pytest.mark.asyncio
async def test_n_concurrent_stream_one_upstream_call(tmp_path, api_key):
    async with _driver(tmp_path, api_key) as (up, app, client):
        N = 4
        auth = _auth(api_key)

        async def get(content: str, include_usage: bool):
            body = _body(content, stream=True)
            if include_usage:
                body["stream_options"] = {"include_usage": True}
            r = await client.post("/v1/chat/completions", headers=auth, json=body)
            return r

        # One follower asks for usage, the rest do not — proves per-follower usage.
        tasks = [asyncio.ensure_future(get("streamq", i == 0)) for i in range(N)]
        await _wait_for_followers(app, N - 1)
        up.release.set()
        resps = await asyncio.gather(*tasks)

    assert up.calls == 1, f"expected 1 upstream call, got {up.calls}"
    texts = [r.text for r in resps]
    # Every caller sees the same answer content and a terminating [DONE].
    for t in texts:
        assert "ACK-" in t
        assert "data: [DONE]" in t
    followers = sum(1 for r in resps if r.headers.get("X-Cradle-Flight") == "follower")
    assert followers == N - 1
    # Exactly the include_usage caller(s) get a usage frame.
    with_usage = [t for t in texts if '"usage"' in t and "completion_tokens" in t]
    assert len(with_usage) == 1, "only the include_usage follower/leader emits usage"
    assert app.state.runtime.flights == {}


@pytest.mark.asyncio
async def test_leader_generator_abort_fails_flight_and_clears_registry(tmp_path, api_key):
    """A leader whose response generator is aborted mid-stream (client disconnect →
    Starlette throws GeneratorExit/CancelledError into it) must fail() the flight so
    waiting followers get an error, and pop it from the registry. Driven directly on
    _wrap_stream because in-memory ASGI transport does not replicate a real disconnect's
    cancellation of the server-side generator."""
    import cradle.gateway.stream as st
    from cradle.cache.records import Principal
    from cradle.compress.engine import compress
    from cradle.gateway.context import RequestContext
    from cradle.gateway.flight import Flight
    from cradle.gateway.models import ChatMessage, ChatRequest
    from cradle.gateway.sse import StreamAccumulator
    from cradle.normalize import canonicalize

    s = _settings(tmp_path, api_key)
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k")
    req = ChatRequest(model="m", stream=True, messages=[ChatMessage(role="user", content="dc")])
    canonical = canonicalize(req, principal, s, backend_namespace="ns")
    compressed = compress(req.messages, s, "m")
    flight = Flight("k:s")

    class _RT:
        settings = s
        flights = {"k:s": flight}

    rt = _RT()

    class _Resp:
        async def aiter_lines(self):
            yield 'data: {"choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}'
            # Simulate the client disconnecting mid-stream: the consumer stops pulling
            # and the generator is finalized. We emulate that by never sending finish.
            await asyncio.sleep(3600)  # will be cancelled by the abort below
            yield ""

        async def aclose(self):
            pass

    ctx = RequestContext(request_id="r", principal=principal)
    ctx.canonical = canonical
    ctx.layer_hit = "miss"
    ctx.flight = flight
    acc = StreamAccumulator(outbound_id="o", outbound_created=1, model="m")

    gen = st._wrap_stream(rt, req, ctx, None, compressed, _Resp(), acc)
    # Pull the first frame (role), which also publishes it to the flight.
    await gen.__anext__()
    assert len(flight.frames) >= 1  # leader published a frame
    # Now abort the generator as Starlette does on disconnect.
    await gen.aclose()  # raises GeneratorExit inside → except/finally runs

    assert flight.done.is_set(), "aborted leader must resolve the flight"
    assert flight.error is not None, "abort must FAIL the flight, not finish it"
    assert rt.flights == {}, "aborted leader must clear the registry"


@pytest.mark.asyncio
async def test_follower_of_aborted_leader_gets_error_frame(tmp_path, api_key):
    """End-to-end at the flight boundary: a stream follower on a flight the leader
    failed receives an error frame + [DONE] and does not hang."""
    from cradle.cache.records import Principal
    from cradle.gateway.context import RequestContext
    from cradle.gateway.flight import Flight, _follow_stream
    from cradle.gateway.models import ChatMessage, ChatRequest

    s = _settings(tmp_path, api_key)
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k")
    req = ChatRequest(model="m", stream=True, messages=[ChatMessage(role="user", content="x")])
    flight = Flight("k:s")
    flight.followers = 1

    class _RT:
        settings = s

    ctx = RequestContext(request_id="r", principal=principal)
    ctx.layer_hit = "miss"

    gen = _follow_stream(_RT(), req, ctx, flight, timeout=5.0)
    # Leader fails after the follower starts tailing.
    async def abort_soon():
        await asyncio.sleep(0.01)
        flight.fail({"error": {"message": "leader aborted", "type": "server_error"}})

    abort = asyncio.ensure_future(abort_soon())
    chunks = [c async for c in gen]
    await abort
    text = b"".join(chunks).decode()
    assert "error" in text.lower()
    assert "data: [DONE]" in text
    assert flight.followers == 0, "follower decremented its count in finally"


async def _assert_no_coalesce_concurrent(app, client, headers, body, expect_calls, up):
    """Fire N identical requests CONCURRENTLY under a blocking upstream. If the
    predicate excludes them, no flight forms, so every one calls upstream; the
    registry stays empty throughout. (Sequential requests would pass even if the
    predicate were broken — they must race.)"""
    N = 3
    tasks = [
        asyncio.ensure_future(client.post("/v1/chat/completions", headers=headers, json=body))
        for _ in range(N)
    ]
    # Give the requests time to reach upstream and (if wrongly eligible) register a
    # flight, then assert none did before releasing.
    for _ in range(40):
        await asyncio.sleep(0.005)
        if up.calls >= N:
            break
    assert app.state.runtime.flights == {}, "no flight may form for an ineligible request"
    up.release.set()
    resps = await asyncio.gather(*tasks)
    assert up.calls == expect_calls, f"expected {expect_calls} upstream calls, got {up.calls}"
    return resps


@pytest.mark.asyncio
async def test_no_cache_request_does_not_coalesce(tmp_path, api_key):
    async with _driver(tmp_path, api_key) as (up, app, client):
        # no-cache skips the L1 read, so two concurrent no-cache requests genuinely
        # race — this is the path the eligibility predicate exists to block.
        hdr = {**_auth(api_key), "X-Cradle-Cache-Control": "no-cache"}
        resps = await _assert_no_coalesce_concurrent(app, client, hdr, _body("nc"), 3, up)
    assert all(r.headers.get("X-Cradle-Flight") is None for r in resps)


@pytest.mark.asyncio
async def test_bypass_request_does_not_coalesce(tmp_path, api_key):
    async with _driver(tmp_path, api_key) as (up, app, client):
        # n=2 is uncacheable → bypass → never a leader/follower, even concurrently.
        resps = await _assert_no_coalesce_concurrent(
            app, client, _auth(api_key), _body("by", n=2), 3, up
        )
    assert all(r.headers["X-Cradle-Cache"] == "BYPASS" for r in resps)


@pytest.mark.asyncio
async def test_json_follower_timeout_returns_504(tmp_path, api_key, monkeypatch):
    """A leader that never resolves within the real follower timeout yields a 504,
    not a hang (the registry-leak backstop). Exercises the real follow() wiring:
    timeout = upstream.timeout_s + _FOLLOWER_TIMEOUT_SLACK_S, both driven tiny."""
    import cradle.gateway.flight as flmod
    from cradle.cache.records import Principal
    from cradle.gateway.context import RequestContext
    from cradle.gateway.flight import Flight, follow
    from cradle.gateway.models import ChatMessage, ChatRequest

    monkeypatch.setattr(flmod, "_FOLLOWER_TIMEOUT_SLACK_S", 0.05)
    s = _settings(tmp_path, api_key)
    s.upstream.timeout_s = 0  # real timeout = 0 + 0.05
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k")
    req = ChatRequest(model="m", messages=[ChatMessage(role="user", content="t")])
    flight = Flight("k:j")  # never resolved

    class _RT:
        settings = s

    ctx = RequestContext(request_id="r", principal=principal)
    ctx.layer_hit = "miss"
    resp = await follow(_RT(), req, ctx, flight)
    assert resp.status_code == 504
    assert flight.followers == 0  # decremented in the timeout's finally


@pytest.mark.asyncio
async def test_stream_follower_timeout_emits_error_frame(tmp_path, api_key, monkeypatch):
    """The stream-follower analogue: a never-resolving leader makes the follower's
    tail time out, emitting an error frame + [DONE] instead of hanging."""
    import cradle.gateway.flight as flmod
    from cradle.cache.records import Principal
    from cradle.gateway.context import RequestContext
    from cradle.gateway.flight import Flight, follow
    from cradle.gateway.models import ChatMessage, ChatRequest

    monkeypatch.setattr(flmod, "_FOLLOWER_TIMEOUT_SLACK_S", 0.05)
    s = _settings(tmp_path, api_key)
    s.upstream.timeout_s = 0
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k")
    req = ChatRequest(model="m", stream=True, messages=[ChatMessage(role="user", content="t")])

    class _RT:
        settings = s

    ctx = RequestContext(request_id="r", principal=principal)
    ctx.layer_hit = "miss"
    flight = Flight("k:s")  # never resolved
    resp = await follow(_RT(), req, ctx, flight)
    chunks = [c async for c in resp.body_iterator]
    text = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks).decode()
    assert "timed out" in text.lower() or "error" in text.lower()
    assert "data: [DONE]" in text
    assert flight.followers == 0


@pytest.mark.asyncio
async def test_flag_off_no_coalescing(tmp_path, api_key):
    async with _driver(tmp_path, api_key, singleflight=False) as (up, app, client):
        up.blocking = False  # respond immediately; no gating needed
        auth = _auth(api_key)
        # Distinct prompts so nothing L1-hits; each must call upstream.
        tasks = [
            asyncio.ensure_future(client.post("/v1/chat/completions", headers=auth, json=_body(f"q{i}")))
            for i in range(3)
        ]
        resps = await asyncio.gather(*tasks)
    assert up.calls == 3, "flag off => no coalescing => one upstream call each"
    assert all(r.headers.get("X-Cradle-Flight") is None for r in resps)


@pytest.mark.asyncio
async def test_upstream_error_propagates_to_followers(tmp_path, api_key):
    async with _driver(tmp_path, api_key) as (up, app, client):
        auth = _auth(api_key)

        async def err_handler(request: httpx.Request) -> httpx.Response:
            up.calls += 1
            await up.release.wait()
            return httpx.Response(500, json={"error": {"message": "boom", "type": "server_error"}})

        # Swap the handler to always 500.
        app.state.runtime.http = httpx.AsyncClient(
            transport=httpx.MockTransport(err_handler), base_url="http://upstream"
        )
        N = 3
        tasks = [
            asyncio.ensure_future(client.post("/v1/chat/completions", headers=auth, json=_body("boom")))
            for _ in range(N)
        ]
        await _wait_for_followers(app, N - 1)
        up.release.set()
        resps = await asyncio.gather(*tasks)
        await app.state.runtime.http.aclose()

    assert up.calls == 1
    assert all(r.status_code == 502 for r in resps), [r.status_code for r in resps]
    assert app.state.runtime.flights == {}
