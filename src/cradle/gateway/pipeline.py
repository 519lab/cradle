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
from cradle.cache.records import L2Filter
from cradle.compress.engine import compress
from cradle.gateway.context import RequestContext
from cradle.gateway.errors import openai_error
from cradle.gateway.models import ChatMessage, ChatRequest
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
from cradle.normalize import canonicalize, is_cacheable, l1_key, l2_eligible
from cradle.reconstruct.merge import merge, wrap_content, wrap_prefix, wrap_suffix
from cradle.reconstruct.templates import template_for
from cradle.tokens import count_chat_prompt
from cradle.upstream.openai import UpstreamError, chat, chat_stream

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
    }
    if ctx.l2_score is not None:
        h["X-Cradle-Similarity"] = f"{ctx.l2_score:.6f}"
    return h


def _include_usage(req: ChatRequest) -> bool:
    opts = req.stream_options or {}
    return bool(opts.get("include_usage"))


def _upstream_payload(req: ChatRequest, messages: list[ChatMessage]) -> dict[str, Any]:
    payload = req.model_dump(exclude_none=True)
    payload["messages"] = [m.model_dump(exclude_none=True) for m in messages]
    return payload


def _choice_content(completion: dict[str, Any]) -> str:
    choices = completion.get("choices") or [{}]
    msg = (choices[0] or {}).get("message") or {}
    content = msg.get("content") or ""
    return content if isinstance(content, str) else str(content)


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
    canonical = canonicalize(req, ctx.principal, runtime.settings)
    ctx.canonical = canonical
    ctx.inbound_prompt_tokens = count_chat_prompt(req.messages, req.model)
    cacheable = is_cacheable(canonical, req, runtime.settings)
    if not cacheable:
        ctx.layer_hit = "bypass"
        return await _miss(runtime, req, ctx, vec=None)

    if runtime.l1 is not None:
        t0 = time.perf_counter()
        rec = await l1mod.get(runtime.l1, l1_key(canonical))
        ctx.t_l1_s = time.perf_counter() - t0
        if rec is not None:
            ctx.layer_hit = "l1"
            ctx.upstream_prompt_tokens = 0
            return _replay(req, ctx, rec)

    vec: list[float] | None = None
    if l2_eligible(canonical, req, runtime.settings) and runtime.qdrant is not None:
        vec = await _maybe_embed(runtime, canonical.embed_text, ctx)
        if vec is not None:
            t0 = time.perf_counter()
            hit = await l2mod.query(
                runtime.qdrant,
                runtime.settings,
                vec,
                L2Filter(
                    tenant_id=canonical.tenant_id,
                    user_id=canonical.user_id,
                    model=canonical.model,
                    system_prompt_version=canonical.system_prompt_version,
                    pipeline_version=canonical.pipeline_version,
                    sampling_fingerprint=canonical.sampling_fingerprint,
                    now_unix=int(time.time()),
                ),
            )
            ctx.t_l2_s = time.perf_counter() - t0
            if hit is not None:
                ctx.layer_hit = "l2"
                ctx.l2_score = hit.score
                ctx.upstream_prompt_tokens = 0
                rec = await promote_l2_hit(runtime, canonical, hit.record, ctx.inbound_prompt_tokens)
                return _replay(req, ctx, rec)

    ctx.layer_hit = "miss"
    return await _miss(runtime, req, ctx, vec=vec)


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
    if req.stream:
        return await _miss_stream(runtime, req, ctx, vec, compressed, payload)
    return await _miss_json(runtime, req, ctx, vec, compressed, payload)


async def _miss_json(runtime, req, ctx, vec, compressed, payload) -> JSONResponse:
    t0 = time.perf_counter()
    try:
        completion = await chat(runtime.http, runtime.settings, payload)
    except UpstreamError as exc:
        m.upstream_errors.labels(status=str(exc.status)).inc()
        status = 502 if exc.status >= 500 else exc.status
        if isinstance(exc.body, dict):
            return JSONResponse(exc.body, status_code=status, headers=_headers(ctx))
        return openai_error(str(exc.body), "server_error", "upstream_error", status)
    ctx.t_upstream_s = time.perf_counter() - t0
    usage = completion.get("usage") or {}
    ctx.upstream_prompt_tokens = int(usage.get("prompt_tokens") or compressed.compressed_tokens)
    t1 = time.perf_counter()
    wrapped = merge(completion, compressed.template)
    ctx.t_reconstruct_s = time.perf_counter() - t1
    if ctx.layer_hit != "bypass" and ctx.canonical is not None:
        rec = record_from(
            ctx.canonical,
            wrapped,
            ctx.inbound_prompt_tokens,
            ctx.upstream_prompt_tokens,
            runtime.settings.cache.ttl_s,
        )
        await writeback(runtime, ctx.canonical, vec, rec)
    _observe(ctx)
    return JSONResponse(wrapped, headers=_headers(ctx))


async def _miss_stream(runtime, req, ctx, vec, compressed, payload) -> JSONResponse | StreamingResponse:
    outbound_id = f"chatcmpl-{uuid.uuid4().hex}"
    outbound_created = int(time.time())
    acc = StreamAccumulator(
        outbound_id=outbound_id, outbound_created=outbound_created, model=req.model
    )
    headers = _headers(ctx)
    headers["Cache-Control"] = "no-cache"
    headers["X-Accel-Buffering"] = "no"

    async def gen() -> AsyncIterator[bytes]:
        t0 = time.perf_counter()
        try:
            stream = chat_stream(runtime.http, runtime.settings, payload)
            first = True
            async for line in stream:
                if first:
                    first = False
                    yield encode_chunk(role_frame(outbound_id, outbound_created, req.model))
                    prefix = wrap_prefix(compressed.template)
                    if prefix:
                        yield encode_chunk(content_frame(outbound_id, outbound_created, req.model, prefix))
                piece = parse_and_accumulate(line, acc)
                if acc.error:
                    yield encode_chunk(error_frame("upstream error"))
                    yield encode_done()
                    return
                if piece:
                    yield encode_chunk(content_frame(outbound_id, outbound_created, req.model, piece))
            ctx.t_upstream_s = time.perf_counter() - t0
            if acc.tool_call_seen:
                if acc.finish_reason:
                    yield encode_chunk(
                        finish_frame(outbound_id, outbound_created, req.model, acc.finish_reason)
                    )
                yield encode_done()
                return
            if acc.finish_reason is None:
                yield encode_chunk(finish_frame(outbound_id, outbound_created, req.model, "stop"))
                yield encode_done()
                return
            suffix = wrap_suffix(compressed.template, acc.content)
            if suffix:
                yield encode_chunk(content_frame(outbound_id, outbound_created, req.model, suffix))
            yield encode_chunk(finish_frame(outbound_id, outbound_created, req.model, acc.finish_reason))
            if _include_usage(req) and acc.usage:
                yield encode_chunk(usage_frame(outbound_id, outbound_created, req.model, acc.usage))
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
            }
            if ctx.layer_hit != "bypass" and ctx.canonical is not None and not acc.error:
                rec = record_from(
                    ctx.canonical,
                    completion,
                    ctx.inbound_prompt_tokens,
                    ctx.upstream_prompt_tokens,
                    runtime.settings.cache.ttl_s,
                )
                await writeback(runtime, ctx.canonical, vec, rec)
            _observe(ctx)
        except UpstreamError as exc:
            m.upstream_errors.labels(status=str(exc.status)).inc()
            if acc.content_parts or acc.saw_done:
                yield encode_chunk(error_frame("upstream error"))
                yield encode_done()
                return
            raise

    try:
        # Open stream inside generator so connect-fail before first byte can be JSON.
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
    except UpstreamError as exc:
        status = 502 if exc.status >= 500 else exc.status
        return openai_error(str(exc.body), "server_error", "upstream_error", status)
