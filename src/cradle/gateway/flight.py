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
import logging
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

log = logging.getLogger("cradle.flight")


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
        "error_status",
        "error_headers",
        "upstream_resp",
        "is_stream",
        "followers",
    )

    def __init__(self, key: str, *, is_stream: bool | None = None) -> None:
        self.key = key
        # Whether this flight is a streaming leader. The reaper only reaps a stream
        # flight (#76), so this must be explicit, not inferred from response state.
        # Defaults to the key's stream marker (flight_key appends ':s'/':j') so
        # existing callers that build a keyed flight keep working; the production
        # register seam passes it explicitly from req.stream.
        self.is_stream = key.endswith(":s") if is_stream is None else is_stream
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
        # A JSON leader's real upstream status + forwardable headers on an upstream
        # error, so a JSON follower relays the leader's true 429/503 + retry-after/
        # x-ratelimit-* instead of a generic 502 (#73). None → the follower uses the
        # 502 default. Stream followers can't use these: their StreamingResponse
        # status/headers are committed at join time (follow()), before the leader's
        # outcome is known, so a stream leader's status/headers reach followers only
        # inside a frame body (already handled by #72).
        self.error_status: int | None = None
        self.error_headers: dict[str, str] | None = None
        # The open upstream streaming response for a stream leader (#77). Stashed at
        # the _miss_stream wrap seam so the reaper can close it if the leader's
        # generator leaks (never iterated → its own finally, which owns aclose(),
        # never runs). None for a JSON leader (its chat() response is already read).
        self.upstream_resp: object | None = None
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
        """Leader succeeded: publish the completion dict and wake all waiters.
        A no-op if already resolved (idempotent): the FIRST resolution wins, so a
        leader that finishes after the reaper already failed its flight (#67) does
        not overwrite the error, and a raise-guard + finally double-call is safe."""
        if self.done.is_set():
            return
        self.completion = completion
        self.done.set()

    def fail(
        self,
        error: dict,
        *,
        status: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Leader failed/aborted: publish an OpenAI-shaped error and wake waiters.
        ``status``/``headers`` carry the leader's real upstream status + forwardable
        headers so a JSON follower relays them instead of a generic 502 (#73).
        A no-op if already resolved (idempotent): the FIRST resolution wins, so the
        reaper cannot clobber a flight that finished successfully in the same tick,
        and a raise-guard + finally double-call is safe."""
        if self.done.is_set():
            return
        self.error = error
        self.error_status = status
        self.error_headers = headers
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
    runtime: Runtime,
    flight: Flight,
    *,
    completion: dict | None = None,
    error: dict | None = None,
    error_status: int | None = None,
    error_headers: dict[str, str] | None = None,
) -> None:
    """Resolve a leader's flight and remove it from the registry — the one place
    that mutates ``runtime.flights`` on leader exit (#64).

    The pop is **identity-checked**: it removes the slot only if it still holds
    *this* flight. A leader that was declared stale and replaced (register seam)
    no longer owns its key — an unconditional ``pop(flight.key)`` would delete the
    *replacement* leader's live flight, defeating coalescing and spawning a
    duplicate upstream call. `finish`/`fail` are idempotent (`done` is an Event),
    so calling this twice (e.g. a raise-guard then a finally) is safe.

    ``error_status``/``error_headers`` (error path only) carry the leader's real
    upstream status + forwardable headers for a JSON follower to relay (#73)."""
    if error is not None:
        flight.fail(error, status=error_status, headers=error_headers)
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


def _is_reapable(flight: Flight, timeout_s: float) -> bool:
    """Whether the reaper may evict this flight (#67, narrowed by #76).

    ONLY a leaked *unstarted stream leader* qualifies: a **stream** flight (key
    ends ``:s``) with **zero published frames** that is stale and not done. The
    narrowing is what makes the reaper safe against a live leader (#76):
    - A *started* `_wrap_stream` publishes the role frame on its first line, so any
      live stream leader — even one backpressured at a downstream `yield`, or parked
      in a hung writeback — has `frames` non-empty and is never reaped. The only
      stream flight with zero frames past `timeout_s` is one whose generator was
      never iterated (Starlette cancelled the response before pulling a frame, so
      its `finally` never ran and it leaked).
    - A **JSON** flight cannot leak this way at all: `_miss`'s ``try/finally``
      resolves it on every path (including the upstream-error early return), behind
      the register-site raise-guard, so it is excluded — a slow-but-live JSON leader
      is never reaped."""
    return (
        flight.is_stream
        and not flight.frames
        and not flight.done.is_set()
        and is_stale(flight, timeout_s)
    )


async def reap_stale_flights(runtime: Runtime, timeout_s: float) -> int:
    """Sweep the registry once, resolving+removing any LEAKED flight (#67). Returns
    the count reaped. See `_is_reapable` for exactly what qualifies (a leaked
    unstarted stream leader — narrowed by #76 so a live-but-quiet leader is never
    failed out from under its followers).

    A flight leaked with no follow-up arrival would otherwise sit in the registry
    poisoning the key until an identical arrival evicts it; this background sweep
    evicts it regardless. Resolution goes through `resolve_and_release` (the
    identity-checked pop, #64), so a slot already replaced by a live leader is never
    evicted by the reaper.

    The registry mutation is done synchronously (no await between the reapable check
    and the pop, so the identity invariant holds); the leaked upstream responses are
    closed afterwards (#77) — a leaked unstarted `_wrap_stream` never ran its own
    finally, so its `resp.aclose()` was never called. aclose() is idempotent, so a
    generator that later does run its finally double-closing is harmless."""
    to_close: list[object] = []
    reaped = 0
    for flight in list(runtime.flights.values()):
        if not _is_reapable(flight, timeout_s):
            continue
        m.flight_aborts.labels(reason="stale").inc()
        if flight.upstream_resp is not None:
            to_close.append(flight.upstream_resp)
        resolve_and_release(runtime, flight, error={
            "error": {"message": "single-flight leader made no progress; reaped",
                      "type": "server_error", "code": "upstream_error"}
        })
        reaped += 1
    for resp in to_close:
        try:
            await resp.aclose()
        except Exception:
            log.warning("reaper: closing a leaked upstream response failed", exc_info=True)
    return reaped


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
        # Relay the leader's real upstream status + forwardable headers (retry-after,
        # x-ratelimit-*) so a follower's backoff on a 429/503 works, matching what the
        # leader's own client got (#73). >=500 collapses to 502 exactly as
        # _upstream_error_response does; a leader that failed with no captured status
        # (a non-UpstreamError abort) keeps the 502 default.
        status = 502
        if flight.error_status is not None:
            status = 502 if flight.error_status >= 500 else flight.error_status
        headers = {**_follower_headers(ctx), **(flight.error_headers or {})}
        log_request(runtime.settings, ctx, None, error_status=status)
        return JSONResponse(flight.error, status_code=status, headers=headers)
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
            # flight.error is ALREADY the OpenAI-shaped object ({"error": {...}}) that
            # every leader resolve site sets and _follow_json returns verbatim (#72).
            # error_frame() would wrap it a second time → {"error": {"error": {...}}};
            # emit it directly so the stream follower's error matches the JSON one.
            yield encode_chunk(flight.error)
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
