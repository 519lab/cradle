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
from cradle.metrics import prometheus as m
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


def _parse_error_frame(sse_text: str) -> dict:
    """Return the JSON of the single error `data:` frame in an SSE stream (not
    [DONE]). Lets a test assert the error frame's SHAPE, not just a substring —
    the substring check `"error" in text` passes on a double-wrapped frame (#72)."""
    for line in sse_text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        if isinstance(obj, dict) and "error" in obj:
            return obj
    raise AssertionError(f"no error frame found in SSE stream: {sse_text!r}")


def _assert_clean_stream_end(sse_text: str) -> None:
    """Assert the SSE stream ends cleanly: [DONE] is the last data frame and NO data
    frame carries an `error` key. A substring scan for "error" can't verify this —
    it both misses an error frame appended AFTER [DONE] (the #70 corruption) and
    false-positives on answer text containing the word 'error'. Structural instead."""
    payloads = [
        line[len("data: "):].strip()
        for line in sse_text.splitlines()
        if line.startswith("data: ")
    ]
    assert payloads, f"no data frames in stream: {sse_text!r}"
    assert payloads[-1] == "[DONE]", f"stream must end at [DONE], not {payloads[-1]!r}"
    for p in payloads:
        if p == "[DONE]":
            continue
        obj = json.loads(p)
        assert not (isinstance(obj, dict) and "error" in obj), (
            f"error frame in a stream that should have ended cleanly: {p!r}"
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
    # A genuine client disconnect labels leader_disconnect (not upstream_error).
    assert m.flight_aborts.labels(reason="leader_disconnect")._value.get() >= 1


@pytest.mark.asyncio
async def test_reaper_clears_an_unstarted_leaked_generator(tmp_path, api_key):
    """#67: an UNSTARTED _wrap_stream generator (Starlette cancels the response before
    it ever pulls a frame) never runs its finally, so the registered flight leaks —
    `done` unset, slot retained. The background reaper must resolve+evict it once it
    goes stale. Deliberately NEVER starts the generator (no __anext__), which is the
    exact path the disconnect-mid-stream test above does not cover."""
    import cradle.gateway.stream as st
    from cradle.cache.records import Principal
    from cradle.compress.engine import compress
    from cradle.gateway.context import RequestContext
    from cradle.gateway.flight import Flight, reap_stale_flights
    from cradle.gateway.models import ChatMessage, ChatRequest
    from cradle.gateway.sse import StreamAccumulator

    s = _settings(tmp_path, api_key)
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k")
    req = ChatRequest(model="m", stream=True, messages=[ChatMessage(role="user", content="leak")])
    compressed = compress(req.messages, s, "m")
    flight = Flight("k:s")

    class _RT:
        settings = s
        flights = {"k:s": flight}

    rt = _RT()

    class _Resp:
        async def aiter_lines(self):
            yield ""  # never reached — the generator is never started

        async def aclose(self):
            pass

    ctx = RequestContext(request_id="r", principal=principal)
    ctx.layer_hit = "miss"
    ctx.flight = flight

    # Construct the generator but NEVER iterate it: its finally will never run.
    gen = st._wrap_stream(rt, req, ctx, None, compressed, _Resp(),
                          StreamAccumulator(outbound_id="o", outbound_created=1, model="m"))

    timeout_s = s.upstream.timeout_s
    # A fresh flight is not yet stale: the reaper must leave it alone.
    assert reap_stale_flights(rt, timeout_s) == 0
    assert rt.flights == {"k:s": flight}
    assert not flight.done.is_set()

    # Age it past the staleness bound (the leaked generator never publishes, so in
    # production last_progress_at simply never advances; here we backdate it).
    flight.last_progress_at -= timeout_s + 1
    reaped = reap_stale_flights(rt, timeout_s)

    assert reaped == 1
    assert flight.done.is_set(), "reaper must resolve the leaked flight so followers wake"
    assert flight.error is not None, "reaped flight fails (not finishes) so followers get an error"
    assert rt.flights == {}, "reaper must evict the leaked slot so the key is no longer poisoned"

    await gen.aclose()  # tidy the never-started generator


def test_resolution_is_idempotent_first_wins(tmp_path, api_key):
    """#67 follow-up: fail()/finish() are no-ops once the flight is resolved, so a
    leader that finishes AFTER the reaper already failed it in the same tick does not
    overwrite the error (which would make _follow_json serve an error for a leader
    that actually succeeded), and vice-versa. The FIRST resolution wins."""
    from cradle.gateway.flight import Flight

    # reaper failed it, then the real leader finishes: error stays, completion ignored.
    f = Flight("k:j")
    f.fail({"error": {"message": "reaped", "type": "server_error"}})
    assert f.done.is_set()
    f.finish({"id": "late", "choices": [{"message": {"content": "hi"}}]})
    assert f.error is not None, "a late finish must not clear the reaper's error"
    assert f.completion is None, "a late finish must not overwrite the resolved state"

    # leader finished, then a stray fail (e.g. raise-guard after finally): completion stays.
    g = Flight("k:j")
    g.finish({"id": "ok", "choices": [{"message": {"content": "answer"}}]})
    g.fail({"error": {"message": "late abort", "type": "server_error"}})
    assert g.completion is not None, "a late fail must not clobber a successful completion"
    assert g.error is None, "a late fail must not set an error on a finished flight"


@pytest.mark.asyncio
async def test_reaper_loop_runs_under_lifespan_and_evicts_a_leak(tmp_path, api_key):
    """#67: the reaper background task wired into the lifespan actually evicts a leaked
    flight (covers the loop glue, not just reap_stale_flights). Uses a tiny
    upstream.timeout_s so the sweep fires fast, and polls with a deadline (the
    codebase's non-flaky pattern) rather than asserting on a fixed sleep."""
    from cradle.config import UpstreamSettings
    from cradle.gateway.flight import Flight

    s = _settings(tmp_path, api_key)
    # Rebuild with a tiny staleness/sweep interval so the reaper fires promptly.
    s = s.model_copy(update={
        "upstream": UpstreamSettings(base_url="http://upstream/v1", models_passthrough=False, timeout_s=0.05),
    })
    http = httpx.AsyncClient(transport=httpx.MockTransport(GatedUpstream().handler), base_url="http://upstream")
    app = create_app(settings=s, embedder=FakeEmbedder(), http=http, reranker=AllowReranker())
    async with app.router.lifespan_context(app):
        rt = app.state.runtime
        leaked = Flight("k:s")
        leaked.last_progress_at -= 10  # already stale for a 0.05s timeout
        rt.flights["k:s"] = leaked
        # Poll until the reaper loop's next sweep evicts it (deadline, not a fixed sleep).
        deadline = asyncio.get_event_loop().time() + 5.0
        while "k:s" in rt.flights and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(0.02)
        assert "k:s" not in rt.flights, "reaper loop must evict the leaked flight"
        assert leaked.done.is_set(), "reaper resolved it so any follower would wake"
    await http.aclose()


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
    assert "data: [DONE]" in text
    # #72: the follower's error frame must be the leader's error object as-is,
    # NOT double-wrapped. flight.error is {"error": {...}}; the old code passed it
    # through error_frame() → {"error": {"error": {...}}}. Assert the parsed shape:
    # exactly one "error" level whose value is the leader's inner error dict.
    err = _parse_error_frame(text)
    assert set(err) == {"error"}, f"expected a single top-level 'error' key, got {err}"
    inner = err["error"]
    assert "error" not in inner, f"error frame is double-wrapped: {err}"
    assert inner.get("message") == "leader aborted"
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
async def test_no_store_request_does_not_coalesce(tmp_path, api_key):
    async with _driver(tmp_path, api_key) as (up, app, client):
        # no-store requests reach BOTH flight seams (unlike bypass), so this is the
        # eligibility clause with no other coverage. Concurrent no-store requests
        # must not coalesce (flight_eligible excludes cache_no_store in v1).
        hdr = {**_auth(api_key), "X-Cradle-Cache-Control": "no-store"}
        resps = await _assert_no_coalesce_concurrent(app, client, hdr, _body("ns"), 3, up)
    assert all(r.headers.get("X-Cradle-Flight") is None for r in resps)


@pytest.mark.asyncio
async def test_bypass_request_does_not_coalesce(tmp_path, api_key):
    async with _driver(tmp_path, api_key) as (up, app, client):
        # n=2 is uncacheable → bypass → returns before the seams. Structural, not via
        # the predicate; kept as a regression guard that bypass never registers a flight.
        resps = await _assert_no_coalesce_concurrent(
            app, client, _auth(api_key), _body("by", n=2), 3, up
        )
    assert all(r.headers["X-Cradle-Cache"] == "BYPASS" for r in resps)


@pytest.mark.asyncio
async def test_late_join_follower_gets_all_frames(tmp_path, api_key):
    """A follower that joins AFTER the leader has already published some frames must
    receive the already-buffered frames then live-tail the rest with none dropped or
    duplicated at the join boundary (exercises the tail() cursor + event-ordering)."""
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

    # Leader has already published two frames before the follower joins.
    flight.publish(b"data: FRAME-A\n\n")
    flight.publish(b"data: FRAME-B\n\n")

    gen = _follow_stream(_RT(), req, ctx, flight, timeout=5.0)

    async def leader_continues():
        await asyncio.sleep(0.02)
        flight.publish(b"data: FRAME-C\n\n")  # a frame appended WHILE the follower tails
        await asyncio.sleep(0.02)
        flight.finish({"id": "o", "created": 1, "model": "m",
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"},
                                    "finish_reason": "stop"}], "usage": {}})

    cont = asyncio.ensure_future(leader_continues())
    chunks = [c async for c in gen]
    await cont
    text = b"".join(chunks).decode()
    # All three frames present, in order, exactly once; then the follower's own [DONE].
    assert text.count("FRAME-A") == 1 and text.count("FRAME-B") == 1 and text.count("FRAME-C") == 1
    assert text.index("FRAME-A") < text.index("FRAME-B") < text.index("FRAME-C")
    assert "data: [DONE]" in text


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


@pytest.mark.asyncio
async def test_json_followers_relay_leader_upstream_status_and_headers(tmp_path, api_key):
    """#73: a JSON follower must reflect the leader's REAL upstream status (429) and
    forwardable headers (retry-after, x-ratelimit-*), not a generic 502 with no
    backoff header — otherwise a coalesced client can't honor the upstream's backoff.
    The leader and its followers should all carry the same status + retry-after."""
    async with _driver(tmp_path, api_key) as (up, app, client):
        auth = _auth(api_key)

        async def rate_limited(request: httpx.Request) -> httpx.Response:
            up.calls += 1
            await up.release.wait()
            return httpx.Response(
                429,
                json={"error": {"message": "slow down", "type": "rate_limit", "code": "rate_limited"}},
                headers={"retry-after": "42", "x-ratelimit-remaining": "0"},
            )

        app.state.runtime.http = httpx.AsyncClient(
            transport=httpx.MockTransport(rate_limited), base_url="http://upstream"
        )
        N = 3
        tasks = [
            asyncio.ensure_future(client.post("/v1/chat/completions", headers=auth, json=_body("hi")))
            for _ in range(N)
        ]
        await _wait_for_followers(app, N - 1)
        up.release.set()
        resps = await asyncio.gather(*tasks)
        await app.state.runtime.http.aclose()

    assert up.calls == 1, "one upstream call; the other two coalesced as followers"
    for r in resps:
        assert r.status_code == 429, f"follower must relay the leader's 429, got {r.status_code}"
        assert r.headers.get("retry-after") == "42", (
            f"follower must carry the leader's retry-after: {dict(r.headers)}"
        )
        assert r.headers.get("x-ratelimit-remaining") == "0"
        assert r.json()["error"]["message"] == "slow down", "follower relays the real upstream body"
    # At least the two followers carried X-Cradle-Flight: follower.
    assert sum(r.headers.get("X-Cradle-Flight") == "follower" for r in resps) == N - 1
    assert app.state.runtime.flights == {}


@pytest.mark.asyncio
async def test_stream_leader_midstream_error_gives_followers_clean_message(tmp_path, api_key):
    """#72 Defect B: a wrap-stream leader whose upstream emits an error frame
    MID-STREAM (HTTP 200 body carrying `data: {"error": {...}}`) must give its
    followers an error whose `message` is the upstream text, not the whole error
    dict stuffed into the string message field. Distinct from the #63 open-failure
    path (HTTP 500 at open) — this asserts the upstream's `"boom"` reached the
    follower, which proves the acc.error branch (stream.py) ran, not the open path."""
    async with _driver(tmp_path, api_key) as (up, app, client):
        auth = _auth(api_key)

        async def stream_err_handler(request: httpx.Request) -> httpx.Response:
            up.calls += 1
            await up.release.wait()

            def gen() -> Iterator[bytes]:
                # A valid role frame, then an upstream error frame mid-stream.
                role = {"id": "up-1", "created": 1, "model": "m",
                        "choices": [{"index": 0, "delta": {"role": "assistant"},
                                     "finish_reason": None}]}
                yield f"data: {json.dumps(role)}\n\n".encode()
                yield b'data: {"error": {"message": "boom", "type": "server_error", "code": "upstream_error"}}\n\n'

            return httpx.Response(200, content=b"".join(gen()),
                                  headers={"content-type": "text/event-stream"})

        app.state.runtime.http = httpx.AsyncClient(
            transport=httpx.MockTransport(stream_err_handler), base_url="http://upstream"
        )
        N = 3
        tasks = [
            asyncio.ensure_future(
                client.post("/v1/chat/completions", headers=auth, json=_body("boom", stream=True))
            )
            for _ in range(N)
        ]
        await _wait_for_followers(app, N - 1)
        up.release.set()
        resps = await asyncio.gather(*tasks)
        await app.state.runtime.http.aclose()

    assert up.calls == 1
    # Every response is a follower or the leader; each carries an error frame whose
    # message is the upstream string, exactly one error level (Defect A also holds).
    for r in resps:
        err = _parse_error_frame(r.text)
        assert set(err) == {"error"}, f"expected single top-level error, got {err}"
        inner = err["error"]
        assert "error" not in inner, f"double-wrapped error frame: {err}"
        assert inner.get("message") == "boom", (
            f"message must be the upstream string, not a nested dict: {inner!r}"
        )
    assert app.state.runtime.flights == {}


@pytest.mark.asyncio
async def test_stream_writeback_failure_does_not_corrupt_followers(tmp_path, api_key, monkeypatch):
    """#70: a cache writeback backend error on the wrap-stream LEADER must not fail
    its followers — the answer already streamed to them. Before the fix, completion_out
    was set only after writeback, so a raise failed the flight and appended an error
    frame + [DONE] onto an already-completed follower stream."""
    from cradle.gateway import stream as st
    from cradle.gateway import writeback as wbmod

    async def _boom(runtime, canonical, vec, record):
        raise RuntimeError("l1 backend down")

    # New code writes via writeback_best_effort, which calls the module-global
    # writeback in wbmod. Pre-fix stream.py called `writeback` bound into it by a
    # from-import, so also patch that name (raising=False → no-op on current code) —
    # otherwise the fail-before run never injects the failure and proves nothing.
    monkeypatch.setattr(wbmod, "writeback", _boom)
    monkeypatch.setattr(st, "writeback", _boom, raising=False)
    errors_before = m.cache_write_errors._value.get()

    async with _driver(tmp_path, api_key) as (up, app, client):
        auth = _auth(api_key)
        N = 3
        tasks = [
            asyncio.ensure_future(
                client.post("/v1/chat/completions", headers=auth, json=_body("hi", stream=True))
            )
            for _ in range(N)
        ]
        await _wait_for_followers(app, N - 1)
        up.release.set()
        resps = await asyncio.gather(*tasks)

    assert up.calls == 1, "leader made the single upstream call; followers coalesced"
    for r in resps:
        text = r.text
        # The corruption #70 names: an error frame appended AFTER [DONE]. Assert
        # structurally that the stream ends at [DONE] with no error frame anywhere,
        # and still carries the real answer.
        _assert_clean_stream_end(text)
        assert "ACK-" in text, f"follower must carry the real answer: {text!r}"
    # The failure was counted + swallowed (proves the best-effort helper ran).
    assert m.cache_write_errors._value.get() == errors_before + 1
    assert app.state.runtime.flights == {}


@pytest.mark.asyncio
async def test_json_writeback_failure_does_not_fail_followers(tmp_path, api_key, monkeypatch):
    """#70 (JSON path): a writeback backend error on a JSON leader must not turn its
    success into a 502 for every follower. Before the fix, `out` was set only after
    writeback, so a raise failed the flight and followers got 502 for an answer the
    leader had successfully obtained."""
    from cradle.gateway import pipeline as pl
    from cradle.gateway import writeback as wbmod

    async def _boom(runtime, canonical, vec, record):
        raise RuntimeError("l1 backend down")

    # See the stream test: patch both the helper's module-global writeback and the
    # name pre-fix pipeline.py bound via from-import (raising=False → no-op now).
    monkeypatch.setattr(wbmod, "writeback", _boom)
    monkeypatch.setattr(pl, "writeback", _boom, raising=False)
    errors_before = m.cache_write_errors._value.get()

    async with _driver(tmp_path, api_key) as (up, app, client):
        auth = _auth(api_key)
        N = 3
        tasks = [
            asyncio.ensure_future(client.post("/v1/chat/completions", headers=auth, json=_body("hi")))
            for _ in range(N)
        ]
        await _wait_for_followers(app, N - 1)
        up.release.set()
        resps = await asyncio.gather(*tasks)

    assert up.calls == 1
    for r in resps:
        assert r.status_code == 200, f"writeback failure must not 502 a follower: {r.status_code}"
        assert r.json()["choices"][0]["message"]["content"], "follower carries the real answer"
    assert m.cache_write_errors._value.get() == errors_before + 1
    assert app.state.runtime.flights == {}


# --- Registry-correctness regressions (#63/#64/#65) --------------------------


def test_stale_leader_does_not_evict_its_replacement(tmp_path, api_key):
    """#64: a stale leader whose slot was replaced must NOT delete the replacement
    flight when it finally exits. resolve_and_release pops by IDENTITY, not by key —
    an unconditional pop(key) would remove the live replacement, defeating coalescing
    and spawning a duplicate upstream call. Unit-level: no followers needed."""
    from cradle.gateway.flight import Flight, resolve_and_release

    class _RT:
        flights: dict = {}

    rt = _RT()
    a = Flight("k:s")
    b = Flight("k:s")
    rt.flights["k:s"] = a          # A is the leader
    rt.flights["k:s"] = b          # A went stale; register seam replaced it with B
    assert rt.flights["k:s"] is b
    # A finishes late and resolves. It no longer owns the slot, so it must leave B.
    resolve_and_release(rt, a, completion={"ok": 1})
    assert rt.flights.get("k:s") is b, "A's exit must not evict the live replacement B"
    # B finishes normally and correctly removes itself.
    resolve_and_release(rt, b, completion={"ok": 2})
    assert rt.flights == {}


@pytest.mark.asyncio
async def test_stream_open_failure_resolves_and_removes_flight(tmp_path, api_key):
    """#63: when the upstream fails at stream OPEN, _miss_stream returns an error
    before _wrap_stream is ever built — its resolving finally never runs. The flight
    (registered by the leader) must still be resolved and removed, so later identical
    requests do not join a dead flight and hang ~upstream.timeout_s."""
    async with _driver(tmp_path, api_key) as (up, app, client):
        # Swap upstream for one that 500s at open (no gating — fail immediately).
        async def err_handler(request: httpx.Request) -> httpx.Response:
            up.calls += 1
            return httpx.Response(500, json={"error": {"message": "open boom", "type": "server_error"}})

        app.state.runtime.http = httpx.AsyncClient(
            transport=httpx.MockTransport(err_handler), base_url="http://upstream"
        )
        auth = _auth(api_key)
        r = await client.post("/v1/chat/completions", headers=auth, json=_body("openfail", stream=True))
        assert r.status_code >= 500
        # The key must be free — a stream leader that failed at open left no dead flight.
        assert app.state.runtime.flights == {}, "open-failure must resolve+remove the flight"
        # A second identical request leads afresh (calls upstream again), not hangs.
        r2 = await client.post("/v1/chat/completions", headers=auth, json=_body("openfail", stream=True))
        assert r2.status_code >= 500
        assert up.calls == 2, "second request must lead afresh, not join a dead flight"
        await app.state.runtime.http.aclose()


def test_healthy_publishing_leader_is_not_stale(tmp_path, api_key):
    """#65: staleness is measured from last progress, not creation. A leader whose
    total lifetime exceeds upstream.timeout_s but which is still publishing frames is
    NOT stale (a long/slow stream), while a silent leader past the timeout IS."""
    import time as _time

    from cradle.gateway.flight import Flight, is_stale

    f = Flight("k:s")
    # Force the flight "old" by creation, but keep progress recent.
    f.created_at = _time.monotonic() - 1000.0
    f.last_progress_at = f.created_at
    assert is_stale(f, timeout_s=120.0), "a silent long-lived flight is stale"
    f.publish(b"data: frame\n\n")  # a live leader publishes → bumps last_progress_at
    assert not is_stale(f, timeout_s=120.0), "a leader that just published is NOT stale"


@pytest.mark.asyncio
async def test_follower_not_timed_out_by_slow_own_consumption(tmp_path, api_key, monkeypatch):
    """#65 follower deadline must bound only the wait for the next LEADER frame, not
    downstream delivery. A follower whose own client consumes slowly, while the leader
    keeps publishing, must NOT hit the timeout (regression for the codex finding: the
    old asyncio.timeout_at wrapped the yield, so a slow follower tripped its own
    deadline even with a healthy leader)."""
    import cradle.gateway.flight as flmod
    from cradle.cache.records import Principal
    from cradle.gateway.context import RequestContext
    from cradle.gateway.flight import Flight, follow
    from cradle.gateway.models import ChatMessage, ChatRequest

    monkeypatch.setattr(flmod, "_FOLLOWER_TIMEOUT_SLACK_S", 0.0)
    s = _settings(tmp_path, api_key)
    s.upstream.timeout_s = 0.1  # follower per-frame progress deadline = 0.1s (slack 0)
    principal = Principal(tenant_id="t1", user_id="u1", key_id="k")
    req = ChatRequest(model="m", stream=True, messages=[ChatMessage(role="user", content="t")])

    class _RT:
        settings = s

    ctx = RequestContext(request_id="r", principal=principal)
    ctx.layer_hit = "miss"
    flight = Flight("k:s")

    # The leader publishes each frame promptly — never idle longer than the 0.1s
    # deadline — then finishes. The FOLLOWER's own client is slow: 0.15s per pull,
    # LONGER than the deadline. Under the old code the deadline spanned `yield frame`,
    # so a single slow downstream delivery (0.15s > 0.1s) tripped it even though the
    # leader was healthy. The fix bounds only the wait for the next leader frame, so a
    # slow follower can never trip it.
    async def leader():
        for i in range(4):
            flight.publish(f"data: f{i}\n\n".encode())
            await asyncio.sleep(0.02)
        flight.finish({"id": "x", "created": 1, "model": "m", "usage": {}})

    resp = await follow(_RT(), req, ctx, flight)
    task = asyncio.ensure_future(leader())
    chunks = []
    async for c in resp.body_iterator:
        chunks.append(c)
        await asyncio.sleep(0.15)  # slow follower client: > the 0.1s deadline
    await task
    text = b"".join(c if isinstance(c, bytes) else c.encode() for c in chunks).decode()
    assert "timed out" not in text.lower(), f"healthy leader must not time out a slow follower: {text!r}"
    assert "data: [DONE]" in text
    assert flight.followers == 0
