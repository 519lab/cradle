"""Single-flight coordination: coalesce concurrent identical cacheable misses.

The first cacheable miss for a given key is the *leader* — it opens upstream,
streams normally, and writes back. Concurrent identical arrivals are *followers*
that replay the leader's buffered outbound frames and then live-tail the rest
(streaming), or await the leader's completion dict (JSON). Only the leader calls
upstream and only the leader writes back (#57, ADR-0008).

Coordination is a per-key ``Flight`` on ``Runtime.flights`` plus ``asyncio``
primitives on the single event loop — no locks, no threads. The single hardest
correctness surface is registry cleanup: a leader that exits for ANY reason
(success, upstream error, cancellation) must resolve-or-fail its flight so
waiters wake, then remove it; followers additionally await with a timeout so a
leak degrades to a per-request error, never an infinite hang.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, StreamingResponse

from cradle.gateway.context import RequestContext
from cradle.gateway.errors import openai_error
from cradle.gateway.models import ChatRequest
from cradle.gateway.responses import _headers, _include_usage, _observe
from cradle.gateway.sse import encode_chunk, encode_done, error_frame, usage_frame
from cradle.logging_setup import log_request
from cradle.metrics import prometheus as m

if TYPE_CHECKING:
    from cradle.runtime import Runtime


class Flight:
    """One in-flight leader upstream call that followers coalesce onto."""

    __slots__ = (
        "key",
        "created_at",
        "last_progress_at",
        "frames",
        "done",
        "_new_frame",
        "completion",
        "error",
        "followers",
    )

    def __init__(self, key: str) -> None:
        self.key = key
        self.created_at = time.monotonic()  # observability only (monotonic, not wall)
        # Staleness is measured from the LAST progress, not creation (#65): a stream
        # leader is only resolved after its whole body streams to its own client,
        # which can far exceed upstream.timeout_s for a long/slow-client stream —
        # measuring from created_at would treat a healthy leader as leaked. Stamped
        # here and bumped on every publish(); a JSON leader publishes no frames, so
        # this stays at registration and JSON keeps creation-based semantics (its
        # chat() call is genuinely bounded by the httpx timeout).
        self.last_progress_at = self.created_at
        self.frames: list[bytes] = []  # published outbound frames, up to finish_frame
        self.done = asyncio.Event()  # set when the leader finishes or fails
        self._new_frame = asyncio.Event()  # pulsed on each publish so tailers wake
        self.completion: dict | None = None  # canonical result for JSON followers
        self.error: dict | None = None  # OpenAI-shaped error object if leader failed
        self.followers = 0  # live follower count (feeds the metric AND tests)

    def publish(self, frame: bytes) -> None:
        """Append an outbound frame and wake any live tailers. Called by the leader
        for every frame up to and including finish_frame (NOT the usage frame or
        [DONE] — each follower emits its own, per its own include_usage flag)."""
        self.frames.append(frame)
        self.last_progress_at = time.monotonic()  # a live leader is not stale (#65)
        self._new_frame.set()
        self._new_frame = asyncio.Event()

    def finish(self, completion: dict) -> None:
        """Leader succeeded: publish the completion dict and wake all waiters."""
        self.completion = completion
        self.done.set()

    def fail(self, error: dict) -> None:
        """Leader failed/aborted: publish an OpenAI-shaped error and wake waiters."""
        self.error = error
        self.done.set()

    async def tail(self, cursor: int) -> AsyncIterator[bytes]:
        """Yield frames appended after ``cursor`` until the leader is done.

        ``cursor`` is the number of frames a follower has already replayed from
        the initial ``frames`` snapshot, so no frame is dropped or duplicated at
        the join boundary. Returns promptly if the leader is already done (a late
        follower's cursor already covers every frame)."""
        while True:
            # Snapshot the next-frame event BEFORE draining, so it predates any
            # frame we have not yet seen. publish() sets the current event then
            # swaps in a fresh one; if a publish lands between our drain and our
            # await, `waiter` is the event it already set, so the await returns at
            # once and the next loop's len() check sees the frame. (Reading the
            # event AFTER draining could hand us the fresh event and stall a frame
            # until the next publish.)
            waiter = self._new_frame
            while cursor < len(self.frames):
                yield self.frames[cursor]
                cursor += 1
            if self.done.is_set():
                return
            done_wait = asyncio.ensure_future(self.done.wait())
            frame_wait = asyncio.ensure_future(waiter.wait())
            try:
                await asyncio.wait(
                    {done_wait, frame_wait}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                done_wait.cancel()
                frame_wait.cancel()


# --- Eligibility and keying ------------------------------------------------

_PASS_THROUGH_TOP = ("system_fingerprint", "service_tier")
# Extra grace over the upstream timeout before a follower gives up on its leader.
# A follower's leader is bounded by upstream.timeout_s; this slack covers the
# leader's own reconstruction/writeback after upstream returns.
_FOLLOWER_TIMEOUT_SLACK_S = 5.0


def flight_eligible(ctx: RequestContext) -> bool:
    """Whether this request may lead or join a flight.

    Coalesce only a cacheable, non-bypass, non-#43-tee, non-no-cache, non-probe
    request. no-store is excluded in v1 (a no-store leader never writes back — a
    fine leader — but keeping it out minimises the interaction surface). Bypass is
    already excluded because it returns before the flight seams are reached, but
    the predicate stays explicit."""
    return (
        ctx.layer_hit != "bypass"
        and not ctx.cacheable_passthrough_stream
        and not ctx.cache_no_read
        and not ctx.cache_no_store
        and not ctx.cache_probe
    )


def flight_key(key: str, stream: bool) -> str:
    """Registry key: the L1 key plus stream-ness. JSON and streaming callers share
    an L1 key (hash_input excludes ``stream``), but a JSON leader publishes no
    frames, so a stream follower behind it would replay nothing. Keying on
    stream-ness keeps a flight homogeneous; cross-mode coalescing is a v1 non-goal
    (ADR-0008). Bursts that motivate the feature are homogeneous anyway."""
    return f"{key}:{'s' if stream else 'j'}"


def resolve_and_release(
    runtime: Runtime, flight: Flight, *, completion: dict | None = None, error: dict | None = None
) -> None:
    """Resolve a leader's flight and remove it from the registry — the one place
    that mutates ``runtime.flights`` on leader exit (#64).

    The pop is **identity-checked**: it removes the slot only if it still holds
    *this* flight. A leader that was declared stale and replaced (register seam)
    no longer owns its key — an unconditional ``pop(flight.key)`` would delete the
    *replacement* leader's live flight, defeating coalescing and spawning a
    duplicate upstream call. `finish`/`fail` are idempotent (`done` is an Event),
    so calling this twice (e.g. a raise-guard then a finally) is safe."""
    if error is not None:
        flight.fail(error)
    else:
        flight.finish(completion or {})
    if runtime.flights.get(flight.key) is flight:
        del runtime.flights[flight.key]


def is_stale(flight: Flight, timeout_s: float) -> bool:
    """A flight with no progress for longer than the upstream timeout is assumed
    leaked; replace it. Measured from last_progress_at, not created_at (#65), so a
    healthy stream leader that keeps publishing frames is never treated as stale
    no matter how long its total lifetime; a JSON leader (no frames) or a wedged
    stream leader goes stale after timeout_s of silence."""
    return (time.monotonic() - flight.last_progress_at) > timeout_s


# --- Follower response paths -----------------------------------------------


def _follower_headers(ctx: RequestContext) -> dict[str, str]:
    h = _headers(ctx)
    h["X-Cradle-Flight"] = "follower"
    return h


async def follow(
    runtime: Runtime, req: ChatRequest, ctx: RequestContext, flight: Flight
) -> JSONResponse | StreamingResponse:
    """Serve a follower from the leader's flight. The follower never calls upstream
    and never writes back; it is a miss that avoided an upstream call."""
    ctx.layer_hit = "miss"
    ctx.upstream_prompt_tokens = 0
    flight.followers += 1
    m.flight_followers.inc()
    timeout = runtime.settings.upstream.timeout_s + _FOLLOWER_TIMEOUT_SLACK_S
    if req.stream:
        return StreamingResponse(
            _follow_stream(runtime, req, ctx, flight, timeout),
            media_type="text/event-stream",
            headers={**_follower_headers(ctx), "Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    return await _follow_json(runtime, req, ctx, flight, timeout)


async def _follow_json(
    runtime: Runtime, req: ChatRequest, ctx: RequestContext, flight: Flight, timeout: float
) -> JSONResponse:
    try:
        await asyncio.wait_for(flight.done.wait(), timeout=timeout)
    except TimeoutError:
        m.flight_aborts.labels(reason="timeout").inc()
        log_request(runtime.settings, ctx, None, error_status=504)
        return openai_error(
            "upstream single-flight leader timed out", "server_error", "upstream_error", 504,
            _follower_headers(ctx),
        )
    finally:
        flight.followers -= 1
    if flight.error is not None:
        m.flight_aborts.labels(reason="leader_error").inc()
        _observe(ctx)
        log_request(runtime.settings, ctx, None, error_status=502)
        return JSONResponse(flight.error, status_code=502, headers=_follower_headers(ctx))
    _observe(ctx)
    log_request(runtime.settings, ctx, flight.completion)
    return JSONResponse(flight.completion, headers=_follower_headers(ctx))


async def _follow_stream(
    runtime: Runtime, req: ChatRequest, ctx: RequestContext, flight: Flight, timeout: float
) -> AsyncIterator[bytes]:
    cursor = 0
    try:
        # Replay frames already buffered, then live-tail until the leader is done.
        snapshot = len(flight.frames)
        for i in range(snapshot):
            yield flight.frames[i]
            cursor = i + 1
        try:
            # Progress deadline, not a total-duration bound (#65): a follower of a
            # healthy long stream must not time out while the leader is still
            # publishing. The timeout must bound ONLY the wait for the next leader
            # frame, never the downstream `yield` — a slow follower CLIENT taking a
            # while to consume a frame it already received must not count as the
            # leader being stuck. So wait_for wraps __anext__ (re-armed per frame),
            # and the yield sits outside it. wait_for cancels the pending __anext__
            # on timeout, so aclose() lets tail()'s finally cancel its own waiters.
            frames = flight.tail(cursor)
            try:
                while True:
                    try:
                        frame = await asyncio.wait_for(frames.__anext__(), timeout)
                    except StopAsyncIteration:
                        break
                    yield frame
            finally:
                await frames.aclose()
        except TimeoutError:
            m.flight_aborts.labels(reason="timeout").inc()
            yield encode_chunk(error_frame("upstream single-flight leader timed out"))
            yield encode_done()
            return
        if flight.error is not None:
            m.flight_aborts.labels(reason="leader_error").inc()
            yield encode_chunk(error_frame(flight.error))
            yield encode_done()
            return
        # Success: the leader published through finish_frame. Emit this follower's
        # OWN usage frame (per its own include_usage) + [DONE] — never the leader's,
        # whose include_usage flag may differ.
        completion = flight.completion or {}
        usage = completion.get("usage") or {}
        if _include_usage(req) and usage:
            # completion["id"]/created/model are the leader's outbound_id etc.
            # (see _wrap_stream), so this usage frame matches the replayed frames.
            extra = {k: completion[k] for k in _PASS_THROUGH_TOP if completion.get(k) is not None}
            rec_id = str(completion.get("id") or "")
            created = int(completion.get("created") or 0)
            model = str(completion.get("model") or req.model)
            yield encode_chunk(usage_frame(rec_id, created, model, usage, extra or None))
        yield encode_done()
        _observe(ctx)
        log_request(runtime.settings, ctx, flight.completion)
    finally:
        flight.followers -= 1
