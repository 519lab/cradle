"""Streaming miss paths: the SSE response builders and their per-mode generators.

Three streaming shapes live here, extracted verbatim from ``pipeline.py`` to keep
every module under the 600-line rule (behavior unchanged):

- **bypass** (``_passthrough_bytes``): uncacheable, tee upstream bytes verbatim.
- **tool-stream passthrough-cache** (``_passthrough_cache_stream`` + ``#43``):
  tee verbatim while accumulating a copy, cache a clean no-tool-call response.
- **wrap** (``_wrap_stream``): the cacheable miss — rebuild the client stream from
  locally synthesized frames and write back.

``_miss_stream`` opens the upstream stream and dispatches to one of the three.
Shared response/observability helpers come from ``gateway/responses.py`` so this
module and ``pipeline.py`` both depend on a leaf and the import graph stays acyclic.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse, StreamingResponse

from cradle.gateway.context import RequestContext
from cradle.gateway.responses import (
    _client_auth,
    _effective_ttl,
    _headers,
    _include_usage,
    _observe,
    _upstream_error_response,
)
from cradle.gateway.sse import (
    StreamAccumulator,
    content_frame,
    encode_chunk,
    encode_done,
    error_frame,
    finish_frame,
    parse_and_accumulate,
    reasoning_frame,
    role_frame,
    usage_frame,
)
from cradle.gateway.writeback import (
    cache_skip_reason,
    passthrough_skip_reason,
    record_from,
    writeback,
)
from cradle.logging_setup import log_request
from cradle.metrics import prometheus as m
from cradle.reconstruct.merge import wrap_content, wrap_prefix, wrap_suffix
from cradle.upstream.openai import UpstreamError, forwardable_headers, start_chat_stream

if TYPE_CHECKING:
    from cradle.runtime import Runtime


def _sse_headers(ctx: RequestContext) -> dict[str, str]:
    headers = _headers(ctx)
    headers["Cache-Control"] = "no-cache"
    headers["X-Accel-Buffering"] = "no"
    # On a streaming miss the real upstream token count is only known after the body
    # has streamed — too late for a response header. Drop it rather than report a
    # false 0. (It lands in the cache record and the cradle_upstream_prompt_tokens
    # metric; a later cache-hit replay reports the true value.)
    headers.pop("X-Cradle-Upstream-Tokens", None)
    return headers


async def _miss_stream(runtime, req, ctx, vec, compressed, payload, target) -> JSONResponse | StreamingResponse:
    t0 = time.perf_counter()
    try:
        resp = await start_chat_stream(
            runtime.http, target, payload, authorization=_client_auth(ctx)
        )
    except UpstreamError as exc:
        return _upstream_error_response(runtime, ctx, exc)
    ctx.t_upstream_s = time.perf_counter() - t0
    headers = _sse_headers(ctx)
    if ctx.layer_hit == "bypass":
        # Bypass tees the body verbatim; also relay the allowlisted upstream headers
        # (x-ratelimit-*, request id) so a passthrough response carries quota state.
        headers.update(forwardable_headers(resp.headers))
        return StreamingResponse(
            _passthrough_bytes(runtime, resp, ctx),
            media_type="text/event-stream",
            headers=headers,
        )
    if ctx.cacheable_passthrough_stream:
        # Tool-enabled stream (#43): tee verbatim like bypass (so a tool call relays
        # intact) but accumulate a copy and cache a no-tool-call response afterward.
        headers.update(forwardable_headers(resp.headers))
        acc = StreamAccumulator(
            outbound_id=f"chatcmpl-{uuid.uuid4().hex}",
            outbound_created=int(time.time()),
            model=req.model,
        )
        return StreamingResponse(
            _passthrough_cache_stream(runtime, req, ctx, vec, compressed, resp, acc),
            media_type="text/event-stream",
            headers=headers,
        )
    outbound_id = f"chatcmpl-{uuid.uuid4().hex}"
    outbound_created = int(time.time())
    acc = StreamAccumulator(
        outbound_id=outbound_id, outbound_created=outbound_created, model=req.model
    )
    return StreamingResponse(
        _wrap_stream(runtime, req, ctx, vec, compressed, resp, acc),
        media_type="text/event-stream",
        headers=headers,
    )


async def _passthrough_bytes(runtime: Runtime, resp, ctx: RequestContext) -> AsyncIterator[bytes]:
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
        _observe(ctx)
        # Bypass never inspects the body (verbatim tee), so no completion text.
        log_request(runtime.settings, ctx, None)
    except asyncio.CancelledError:
        log_request(runtime.settings, ctx, None, disconnected=True)
        raise
    finally:
        await resp.aclose()


# Cap the line-accumulation buffer for the passthrough-cache path. A single SSE
# frame far larger than this is treated as unparseable → caching disabled (the
# client still gets every byte). 4 MiB is generous for a chat completion frame.
_PASSTHROUGH_LINE_BUF_MAX = 4 * 1024 * 1024


async def _passthrough_cache_stream(
    runtime: Runtime, req, ctx, vec, compressed, resp, acc: StreamAccumulator
) -> AsyncIterator[bytes]:
    """Tool-enabled stream (#43): tee upstream bytes to the client VERBATIM while
    accumulating a copy; cache only a clean, no-tool-call response afterward.

    The client's bytes are never derived from parsing — any accumulation failure
    only disables caching (acc.cache_disabled), never the stream.
    """
    buf = ""
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk  # verbatim to the client, always
            if acc.cache_disabled:
                continue  # already un-cacheable; keep teeing, skip parsing
            try:
                buf += chunk.decode("utf-8")
            except UnicodeDecodeError:
                # A byte boundary split a codepoint, or non-UTF-8: we can't safely
                # reassemble lines, so stop trusting the accumulation.
                acc.cache_disabled = True
                continue
            if len(buf) > _PASSTHROUGH_LINE_BUF_MAX:
                acc.cache_disabled = True
                buf = ""
                continue
            # Consume only complete lines; keep the trailing partial in buf.
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                try:
                    parse_and_accumulate(line, acc)
                except Exception:  # noqa: BLE001 - aux parsing must never kill the stream
                    acc.cache_disabled = True
        # Stream ended. Record the real upstream prompt-token count BEFORE _observe
        # so the cradle_upstream_prompt_tokens_total counter sees it — the tool-call
        # accumulation finishes here, not before (#59). _observe on a tool-stream
        # otherwise counts 0, which zeroes the metric for all tool-bearing traffic.
        usage = acc.usage or {}
        ctx.upstream_prompt_tokens = int(usage.get("prompt_tokens") or compressed.compressed_tokens)
        _observe(ctx)
        await _maybe_cache_passthrough(runtime, req, ctx, vec, compressed, acc)
    except asyncio.CancelledError:
        acc.client_connected = False
        log_request(runtime.settings, ctx, None, disconnected=True)
        raise
    except GeneratorExit:
        acc.client_connected = False
        log_request(runtime.settings, ctx, None, disconnected=True)
        raise
    finally:
        await resp.aclose()


async def _maybe_cache_passthrough(runtime, req, ctx, vec, compressed, acc: StreamAccumulator) -> None:
    """Fail-closed writeback for the passthrough-cache path (#43)."""
    # Reconstruct the SAME representation the JSON miss path caches (merge()):
    # JSON and streaming share a cache key (stream is excluded from hash_input),
    # so a cross-mode hit must return an identical body.
    body = wrap_content(compressed.template, acc.content)
    usage = acc.usage or {}
    # ctx.upstream_prompt_tokens was set by the caller (_passthrough_cache_stream)
    # before _observe, from this same acc.usage; the cache record below reads it (#59).
    message: dict[str, Any] = {"role": "assistant", "content": body}
    # Store reasoning RAW (not through wrap_content — brand prefix/suffix are for the
    # answer only) under the key the upstream used, so a cache HIT replays the
    # thinking faithfully, matching the JSON path (#46/#49).
    if acc.reasoning and acc.reasoning_key:
        message[acc.reasoning_key] = acc.reasoning
    completion = {
        "id": acc.outbound_id,
        "object": "chat.completion",
        "created": acc.outbound_created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": acc.finish_reason,
            }
        ],
        "usage": usage,
        **acc.extra_top,
    }
    ttl = _effective_ttl(runtime, ctx)
    # Fail-closed gate: every condition must hold, or we cache nothing. The ordered
    # reason logic (root cause before shape artifact) lives in passthrough_skip_reason
    # so it is unit-testable without a live stream.
    skip = passthrough_skip_reason(
        completion=completion,
        error=acc.error,
        cache_disabled=acc.cache_disabled,
        saw_done=acc.saw_done,
        tool_call_seen=acc.tool_call_seen,
        no_store=(ctx.cache_no_store or ttl == 0),
    )
    if acc.client_connected and ctx.canonical is not None and skip is None:
        rec = record_from(
            ctx.canonical,
            completion,
            ctx.inbound_prompt_tokens,
            ctx.upstream_prompt_tokens,
            ttl,
        )
        await writeback(runtime, ctx.canonical, vec, rec)
    elif skip is not None:
        m.cache_write_skips.labels(reason=skip).inc()
    log_request(runtime.settings, ctx, completion if skip is None else None)


async def _wrap_stream(runtime, req, ctx, vec, compressed, resp, acc: StreamAccumulator) -> AsyncIterator[bytes]:
    outbound_id = acc.outbound_id
    outbound_created = acc.outbound_created
    try:
        first = True
        async for line in resp.aiter_lines():
            if first:
                first = False
                yield encode_chunk(role_frame(outbound_id, outbound_created, req.model))
                prefix = wrap_prefix(compressed.template)
                if prefix:
                    yield encode_chunk(content_frame(outbound_id, outbound_created, req.model, prefix))
            piece = parse_and_accumulate(line, acc)
            if acc.error:
                yield encode_chunk(error_frame(acc.error_payload or "upstream error"))
                yield encode_done()
                return
            if acc.tool_call_seen:
                yield encode_chunk(error_frame("unexpected tool_calls on wrap path"))
                yield encode_done()
                return
            # Forward reasoning to the client on its own frame (#49). Previously the
            # wrap path dropped reasoning entirely — parse_and_accumulate returns only
            # content, so a reasoning model's thinking never reached the client and
            # the stripped answer was cached. last_reasoning is the reasoning chunk
            # (if any) from this line.
            if acc.last_reasoning:
                yield encode_chunk(
                    reasoning_frame(
                        outbound_id,
                        outbound_created,
                        req.model,
                        acc.last_reasoning,
                        key=acc.reasoning_key or "reasoning_content",
                    )
                )
            if piece:
                yield encode_chunk(content_frame(outbound_id, outbound_created, req.model, piece))
        if acc.finish_reason is None:
            yield encode_chunk(error_frame("upstream stream ended without finish_reason"))
            yield encode_done()
            return
        suffix = wrap_suffix(compressed.template, acc.content)
        if suffix:
            yield encode_chunk(content_frame(outbound_id, outbound_created, req.model, suffix))
        yield encode_chunk(finish_frame(outbound_id, outbound_created, req.model, acc.finish_reason))
        # The client only sees a usage chunk when it asked for one (OpenAI semantics),
        # even though Cradle always requests usage upstream on the wrap path.
        if _include_usage(req) and acc.usage:
            yield encode_chunk(
                usage_frame(
                    outbound_id, outbound_created, req.model, acc.usage, acc.extra_top or None
                )
            )
        yield encode_done()
        usage = acc.usage or {}
        ctx.upstream_prompt_tokens = int(usage.get("prompt_tokens") or compressed.compressed_tokens)
        body = wrap_content(compressed.template, acc.content)
        message: dict[str, Any] = {"role": "assistant", "content": body}
        # Store reasoning raw so a cache HIT replays the thinking the live client
        # just saw (#49). Same representation the passthrough and JSON paths use.
        if acc.reasoning and acc.reasoning_key:
            message[acc.reasoning_key] = acc.reasoning
        completion = {
            "id": outbound_id,
            "object": "chat.completion",
            "created": outbound_created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": acc.finish_reason,
                }
            ],
            "usage": acc.usage or {},
            # Provider metadata (system_fingerprint, service_tier) so a cached replay
            # carries the same top-level fields a live wrap-stream response does.
            **acc.extra_top,
        }
        ttl = _effective_ttl(runtime, ctx)
        skip = cache_skip_reason(completion)
        if ctx.cache_no_store or ttl == 0:
            skip = skip or "no_store"
        if (
            acc.client_connected
            and ctx.layer_hit != "bypass"
            and ctx.canonical is not None
            and skip is None
        ):
            rec = record_from(
                ctx.canonical,
                completion,
                ctx.inbound_prompt_tokens,
                ctx.upstream_prompt_tokens,
                ttl,
            )
            await writeback(runtime, ctx.canonical, vec, rec)
        elif skip is not None:
            m.cache_write_skips.labels(reason=skip).inc()
        _observe(ctx)
        log_request(runtime.settings, ctx, completion)
    except asyncio.CancelledError:
        acc.client_connected = False
        # The success-path line runs after the body streams, so a client that
        # disconnects mid-stream would otherwise leave no request line at all.
        log_request(runtime.settings, ctx, None, disconnected=True)
        raise
    except GeneratorExit:
        acc.client_connected = False
        log_request(runtime.settings, ctx, None, disconnected=True)
        raise
    finally:
        await resp.aclose()
