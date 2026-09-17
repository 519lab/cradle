from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi.responses import JSONResponse, StreamingResponse

from cradle.cache import l1 as l1mod
from cradle.cache import l2 as l2mod
from cradle.cache.guard import guard_reason
from cradle.cache.records import L2Filter, L2Hit
from cradle.cache.volatility import volatile_reason_for
from cradle.compress.engine import compress
from cradle.gateway.audit import schedule_audit, should_audit
from cradle.gateway.context import RequestContext
from cradle.gateway.errors import openai_error
from cradle.gateway.models import ChatMessage, ChatRequest
from cradle.gateway.probe import candidate_entry, probe_response
from cradle.gateway.sse import (
    StreamAccumulator,
    content_frame,
    encode_chunk,
    encode_done,
    error_frame,
    finish_frame,
    parse_and_accumulate,
    role_frame,
    synthesize_sse,
    usage_frame,
)
from cradle.gateway.writeback import promote_l2_hit, record_from, writeback
from cradle.metrics import prometheus as m
from cradle.normalize import cache_namespace, canonicalize, is_cacheable, l1_key, l2_eligible
from cradle.reconstruct.merge import merge, wrap_content, wrap_prefix, wrap_suffix
from cradle.reconstruct.templates import template_for
from cradle.tokens import count_chat_prompt
from cradle.upstream.openai import (
    UpstreamError,
    chat,
    forwardable_headers,
    start_chat_stream,
)
from cradle.upstream.route import resolve_upstream

if TYPE_CHECKING:
    from cradle.runtime import Runtime

log = logging.getLogger("cradle.pipeline")


def _headers(ctx: RequestContext) -> dict[str, str]:
    cache = {
        "l1": "HIT-L1",
        "l2": "HIT-L2",
        "miss": "MISS",
        "bypass": "BYPASS",
    }[ctx.layer_hit]
    h = {
        "X-Request-ID": ctx.request_id,
        "X-Cradle-Cache": cache,
        "X-Cradle-Pipeline": ctx.canonical.pipeline_version if ctx.canonical else "",
        "X-Cradle-Inbound-Tokens": str(ctx.inbound_prompt_tokens),
        "X-Cradle-Upstream-Tokens": str(ctx.upstream_prompt_tokens),
        "X-Cradle-Upstream": ctx.upstream_name,
    }
    if ctx.l2_score is not None:
        h["X-Cradle-Similarity"] = f"{ctx.l2_score:.6f}"
    if ctx.l2_guard_reason is not None:
        h["X-Cradle-Guard"] = f"reject:{ctx.l2_guard_reason}"
    if ctx.l2_rerank_note is not None:
        h["X-Cradle-Rerank"] = ctx.l2_rerank_note
    if ctx.volatile_reason is not None:
        h["X-Cradle-Volatile"] = ctx.volatile_reason
    if ctx.audit_scheduled:
        h["X-Cradle-Audit"] = "scheduled"
    return h


def _apply_volatility_guard(runtime: Runtime, ctx: RequestContext) -> None:
    """Clamp the TTL of time-sensitive prompts unless the client set one."""
    settings = runtime.settings.cache
    if not settings.volatility_guard or ctx.cache_ttl_override is not None:
        return
    assert ctx.canonical is not None
    reason = volatile_reason_for(ctx.canonical.messages)
    if reason is None:
        return
    ctx.volatile_reason = reason
    ctx.cache_ttl_override = settings.volatile_ttl_s
    m.volatile_prompts.labels(reason=reason).inc()


def _include_usage(req: ChatRequest) -> bool:
    opts = req.stream_options or {}
    return bool(opts.get("include_usage"))


def _upstream_payload(req: ChatRequest, messages: list[ChatMessage]) -> dict[str, Any]:
    payload = req.model_dump(exclude_none=True)
    payload["messages"] = [m.model_dump(exclude_none=True) for m in messages]
    return payload


def _client_auth(ctx: RequestContext) -> str | None:
    return ctx.headers.get("authorization") or None


def _upstream_error_response(ctx: RequestContext, exc: UpstreamError) -> JSONResponse:
    m.upstream_errors.labels(status=str(exc.status)).inc()
    status = 502 if exc.status >= 500 else exc.status
    # Relay upstream retry/quota headers (retry-after, x-ratelimit-*, request id) so a
    # client's backoff on a 429/503 still works even though Cradle re-frames the body.
    headers = {**_headers(ctx), **exc.headers}
    if isinstance(exc.body, dict):
        return JSONResponse(exc.body, status_code=status, headers=headers)
    return openai_error(str(exc.body), "server_error", "upstream_error", status, headers)


async def _maybe_embed(runtime: Runtime, text: str, ctx: RequestContext) -> list[float] | None:
    if runtime.embedder is None:
        return None
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        vec = await asyncio.wait_for(
            loop.run_in_executor(runtime.embed_pool, runtime.embedder.embed, text),
            timeout=runtime.settings.l2.embed_timeout_s,
        )
        ctx.t_embed_s = time.perf_counter() - t0
        m.latency_seconds.labels(stage="embed").observe(ctx.t_embed_s)
        return vec
    except Exception:
        m.embed_errors.inc()
        ctx.t_embed_s = time.perf_counter() - t0
        return None


async def _rerank_ok(runtime: Runtime, query_text: str, candidate_text: str) -> tuple[bool, str]:
    """Run the rerank stage in the embed pool with a timeout. Fail-open.

    Returns ``(serve, note)`` where ``note`` is the X-Cradle-Rerank value.
    A missing reranker, a timeout, or a model error all serve the candidate
    (it already passed cosine + guard) and are recorded as fail-open.
    """
    if runtime.reranker is None:
        return True, "off"
    loop = asyncio.get_running_loop()
    threshold = runtime.settings.l2.rerank_threshold
    try:
        score = await asyncio.wait_for(
            loop.run_in_executor(
                runtime.embed_pool, runtime.reranker.score, query_text, candidate_text
            ),
            timeout=runtime.settings.l2.rerank_timeout_s,
        )
    except Exception:  # noqa: BLE001 - fail open on timeout or model error
        m.l2_rerank_fail_open.inc()
        return True, "fail-open"
    if score >= threshold:
        return True, f"pass:{score:.4f}"
    m.l2_rerank_rejects.inc()
    return False, f"reject:{score:.4f}"


# finish_reasons that indicate a complete, trustworthy answer worth caching.
# `length` (truncated), `content_filter` (refused), tool_calls, and empty bodies
# would otherwise become the permanent cached answer for a prompt and its
# paraphrases (enhancement #3, write-quality gate).
_CACHEABLE_FINISH = frozenset({"stop", "eos"})


def _effective_ttl(runtime: Runtime, ctx: RequestContext) -> int:
    ttl = ctx.cache_ttl_override
    return ttl if ttl is not None else runtime.settings.cache.ttl_s


def _response_cache_skip_reason(completion: dict) -> str | None:
    """Return a skip reason if this response must NOT be cached, else None."""
    choices = completion.get("choices") or []
    if not choices:
        return "no_choices"
    choice = choices[0]
    finish = choice.get("finish_reason")
    if finish not in _CACHEABLE_FINISH:
        return f"finish_{finish}"
    content = (choice.get("message") or {}).get("content")
    if not content or not content.strip():
        return "empty_content"
    return None


def _observe(ctx: RequestContext) -> None:
    m.inbound_prompt_tokens.inc(ctx.inbound_prompt_tokens)
    m.upstream_prompt_tokens.inc(ctx.upstream_prompt_tokens)
    m.latency_seconds.labels(stage="l1").observe(ctx.t_l1_s)
    if ctx.t_l2_s:
        m.latency_seconds.labels(stage="l2").observe(ctx.t_l2_s)
    if ctx.t_compress_s:
        m.latency_seconds.labels(stage="compress").observe(ctx.t_compress_s)
    if ctx.t_upstream_s:
        m.latency_seconds.labels(stage="upstream").observe(ctx.t_upstream_s)
    if ctx.t_reconstruct_s:
        m.latency_seconds.labels(stage="reconstruct").observe(ctx.t_reconstruct_s)
    if ctx.layer_hit in {"l1", "l2"}:
        m.cache_hits.labels(layer=ctx.layer_hit).inc()
    elif ctx.layer_hit == "miss":
        m.cache_misses.inc()
    m.requests_total.labels(
        endpoint="chat", status="200", cache=ctx.layer_hit
    ).inc()


async def handle_chat(runtime: Runtime, req: ChatRequest, ctx: RequestContext) -> JSONResponse | StreamingResponse:
    # Resolve the backend BEFORE canonicalizing so the resolved upstream folds
    # into the cache identity (bug A): entries never cross backends.
    ctx.upstream_name, upstream = resolve_upstream(runtime.settings, req.model)
    ns = cache_namespace(ctx.upstream_name, upstream.base_url)
    canonical = canonicalize(req, ctx.principal, runtime.settings, backend_namespace=ns)
    ctx.canonical = canonical
    ctx.inbound_prompt_tokens = count_chat_prompt(req.messages, req.model)
    cacheable = is_cacheable(canonical, req, runtime.settings)
    if not cacheable:
        ctx.layer_hit = "bypass"
        if ctx.cache_probe:
            return _probe(ctx, l1_key=None, l2_eligible=False)
        return await _miss(runtime, req, ctx, vec=None)
    _apply_volatility_guard(runtime, ctx)

    key = l1_key(canonical)
    # Per-request no-cache/refresh (enhancement #2): skip the read, still write.
    if runtime.l1 is not None and not ctx.cache_no_read:
        t0 = time.perf_counter()
        rec = await l1mod.get(runtime.l1, key)
        ctx.t_l1_s = time.perf_counter() - t0
        if rec is not None:
            ctx.layer_hit = "l1"
            ctx.upstream_prompt_tokens = 0
            if ctx.cache_probe:
                return _probe(ctx, l1_key=key, l2_eligible=False)
            return _replay(req, ctx, rec)

    vec: list[float] | None = None
    eligible = l2_eligible(canonical, req, runtime.settings) and runtime.qdrant is not None
    if eligible:
        # Embed even on no-cache/refresh: only the L2 *read* is skipped. The
        # writeback needs the vector to replace the stale L2 point, otherwise a
        # refresh updates L1 alone and paraphrases keep replaying the old answer.
        vec = await _maybe_embed(runtime, canonical.embed_text, ctx)
        if vec is not None and not ctx.cache_no_read:
            t0 = time.perf_counter()
            candidates = await l2mod.query(
                runtime.qdrant,
                runtime.settings,
                vec,
                L2Filter(
                    tenant_id=canonical.tenant_id,
                    user_id=canonical.user_id,
                    model=canonical.model,
                    backend_namespace=canonical.backend_namespace,
                    system_prompt_version=canonical.system_prompt_version,
                    pipeline_version=canonical.pipeline_version,
                    sampling_fingerprint=canonical.sampling_fingerprint,
                    now_unix=int(time.time()),
                ),
            )
            ctx.t_l2_s = time.perf_counter() - t0
            # Top-K (enhancement #1): try candidates best-first; serve the first
            # that survives the guard + rerank. A near-miss at rank 1 no longer
            # forces a full miss when a true paraphrase sits at rank 2..K.
            for hit in candidates:
                ctx.l2_score = hit.score
                # Stage 1b (verified L2): an entry that an audit already judged
                # wrong at this similarity or higher refuses to serve.
                floor = hit.record.audit_floor
                if floor is not None and hit.score <= floor:
                    ctx.l2_guard_reason = "audit-floor"
                    m.l2_guard_rejects.labels(reason="audit-floor").inc()
                    continue
                # Stage 2 (issue #5): cheap precision guard.
                reason = guard_reason(canonical.embed_text, hit.record.embed_text)
                if reason is not None:
                    ctx.l2_guard_reason = reason
                    m.l2_guard_rejects.labels(reason=reason).inc()
                    _note_candidate(ctx, hit, guard=reason, rerank=None, served=False)
                    continue
                # Stage 3: cross-encoder rerank for entity swaps the guard cannot
                # see. Fail-open: an unavailable reranker still serves.
                serve, note = await _rerank_ok(
                    runtime, canonical.embed_text, hit.record.embed_text
                )
                ctx.l2_rerank_note = note
                _note_candidate(ctx, hit, guard=None, rerank=note, served=serve)
                if serve:
                    ctx.l2_guard_reason = None  # a later candidate cleared the guard
                    ctx.layer_hit = "l2"
                    ctx.upstream_prompt_tokens = 0
                    if ctx.cache_probe:
                        return _probe(ctx, l1_key=key, l2_eligible=True)
                    rec = await promote_l2_hit(
                        runtime,
                        canonical,
                        hit.record,
                        ctx.inbound_prompt_tokens,
                        ttl_s=_effective_ttl(runtime, ctx),
                    )
                    if should_audit(runtime.settings.l2.audit_rate):
                        ctx.audit_scheduled = True
                        schedule_audit(
                            runtime,
                            req=req,
                            ctx=ctx,
                            canonical=canonical,
                            query_vec=vec,
                            hit=hit,
                            served=rec,
                            target=upstream,
                        )
                    return _replay(req, ctx, rec)

    ctx.layer_hit = "miss"
    if ctx.cache_probe:
        return _probe(ctx, l1_key=key, l2_eligible=eligible)
    return await _miss(runtime, req, ctx, vec=vec)


def _note_candidate(
    ctx: RequestContext, hit: L2Hit, *, guard: str | None, rerank: str | None, served: bool
) -> None:
    """Record an examined L2 candidate for probe mode (no-op otherwise)."""
    if ctx.cache_probe:
        ctx.probe_candidates.append(
            candidate_entry(
                key=hit.record.key, score=hit.score, guard=guard, rerank=rerank, served=served
            )
        )


def _probe(ctx: RequestContext, *, l1_key: str | None, l2_eligible: bool) -> JSONResponse:
    """Probe mode terminal: explain the decision, write nothing, call nothing."""
    m.cache_probes.labels(cache=ctx.layer_hit).inc()
    return probe_response(ctx, _headers(ctx), l1_key=l1_key, l2_eligible=l2_eligible)


def _replay(req: ChatRequest, ctx: RequestContext, rec) -> JSONResponse | StreamingResponse:
    _observe(ctx)
    headers = _headers(ctx)
    if req.stream:
        headers["Cache-Control"] = "no-cache"
        headers["X-Accel-Buffering"] = "no"

        async def gen() -> AsyncIterator[bytes]:
            for chunk in synthesize_sse(rec, include_usage=_include_usage(req)):
                yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
    return JSONResponse(rec.response, headers=headers)


async def _miss(
    runtime: Runtime, req: ChatRequest, ctx: RequestContext, vec: list[float] | None
) -> JSONResponse | StreamingResponse:
    assert ctx.canonical is not None
    t0 = time.perf_counter()
    compressed = compress(req.messages, runtime.settings, req.model)
    ctx.t_compress_s = time.perf_counter() - t0
    tenant_tmpl = template_for(runtime.settings, ctx.principal.tenant_id)
    compressed.template.brand_prefix = tenant_tmpl.brand_prefix
    compressed.template.brand_suffix = tenant_tmpl.brand_suffix
    compressed.template.mode = tenant_tmpl.mode
    payload = _upstream_payload(req, compressed.messages)
    _name, target = resolve_upstream(runtime.settings, req.model)
    ctx.upstream_name = _name
    if req.stream:
        # On the wrap path Cradle rebuilds the outbound stream and caches the result,
        # so it must know the real token usage even when the client did not request it
        # (otherwise the record stores empty usage and every later include_usage hit
        # replays {}). Ask upstream for the usage chunk. Never touch the bypass payload:
        # that path is a byte-exact passthrough and must stay verbatim.
        if ctx.layer_hit != "bypass":
            payload["stream_options"] = {
                **(payload.get("stream_options") or {}),
                "include_usage": True,
            }
        return await _miss_stream(runtime, req, ctx, vec, compressed, payload, target)
    return await _miss_json(runtime, req, ctx, vec, compressed, payload, target)


async def _miss_json(runtime, req, ctx, vec, compressed, payload, target) -> JSONResponse:
    t0 = time.perf_counter()
    try:
        completion = await chat(
            runtime.http, target, payload, authorization=_client_auth(ctx)
        )
    except UpstreamError as exc:
        return _upstream_error_response(ctx, exc)
    ctx.t_upstream_s = time.perf_counter() - t0
    usage = completion.get("usage") or {}
    ctx.upstream_prompt_tokens = int(usage.get("prompt_tokens") or compressed.compressed_tokens)
    t1 = time.perf_counter()
    if ctx.layer_hit == "bypass":
        out = completion
    else:
        out = merge(completion, compressed.template)
    ctx.t_reconstruct_s = time.perf_counter() - t1
    ttl = _effective_ttl(runtime, ctx)
    skip = _response_cache_skip_reason(out)
    if ctx.cache_no_store or ttl == 0:
        skip = skip or "no_store"
    if ctx.layer_hit != "bypass" and ctx.canonical is not None and skip is None:
        rec = record_from(
            ctx.canonical,
            out,
            ctx.inbound_prompt_tokens,
            ctx.upstream_prompt_tokens,
            ttl,
        )
        await writeback(runtime, ctx.canonical, vec, rec)
    elif skip is not None:
        m.cache_write_skips.labels(reason=skip).inc()
    _observe(ctx)
    return JSONResponse(out, headers=_headers(ctx))


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
        return _upstream_error_response(ctx, exc)
    ctx.t_upstream_s = time.perf_counter() - t0
    headers = _sse_headers(ctx)
    if ctx.layer_hit == "bypass":
        # Bypass tees the body verbatim; also relay the allowlisted upstream headers
        # (x-ratelimit-*, request id) so a passthrough response carries quota state.
        headers.update(forwardable_headers(resp.headers))
        return StreamingResponse(
            _passthrough_bytes(resp, ctx),
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


async def _passthrough_bytes(resp, ctx: RequestContext) -> AsyncIterator[bytes]:
    try:
        async for chunk in resp.aiter_bytes():
            yield chunk
        _observe(ctx)
    except asyncio.CancelledError:
        raise
    finally:
        await resp.aclose()


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
        completion = {
            "id": outbound_id,
            "object": "chat.completion",
            "created": outbound_created,
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": body},
                    "finish_reason": acc.finish_reason,
                }
            ],
            "usage": acc.usage or {},
            # Provider metadata (system_fingerprint, service_tier) so a cached replay
            # carries the same top-level fields a live wrap-stream response does.
            **acc.extra_top,
        }
        ttl = _effective_ttl(runtime, ctx)
        skip = _response_cache_skip_reason(completion)
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
    except asyncio.CancelledError:
        acc.client_connected = False
        raise
    except GeneratorExit:
        acc.client_connected = False
        raise
    finally:
        await resp.aclose()
