from __future__ import annotations

import asyncio
import logging
import time
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
from cradle.gateway.flight import (
    Flight,
    flight_eligible,
    flight_key,
    follow,
    is_stale,
    resolve_and_release,
)
from cradle.gateway.models import ChatMessage, ChatRequest
from cradle.gateway.probe import candidate_entry, probe_response
from cradle.gateway.responses import (
    _client_auth,
    _effective_ttl,
    _headers,
    _include_usage,
    _observe,
    _upstream_error_response,
)
from cradle.gateway.sse import (
    synthesize_sse,
)
from cradle.gateway.stream import _miss_stream
from cradle.gateway.writeback import (
    cache_skip_reason,
    promote_l2_hit,
    record_from,
    writeback_best_effort,
)
from cradle.logging_setup import log_request
from cradle.metrics import prometheus as m
from cradle.normalize import cache_namespace, canonicalize, is_cacheable, l1_key, l2_eligible
from cradle.reconstruct.merge import merge
from cradle.reconstruct.templates import template_for
from cradle.tokens import count_chat_prompt
from cradle.upstream.openai import (
    UpstreamError,
    chat,
)
from cradle.upstream.route import resolve_upstream

if TYPE_CHECKING:
    from cradle.runtime import Runtime

log = logging.getLogger("cradle.pipeline")


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


def _upstream_payload(req: ChatRequest, messages: list[ChatMessage]) -> dict[str, Any]:
    # Pass-through contract (issue #26): forward only fields the client actually set,
    # so Cradle never injects its own sampling defaults (temperature/top_p/penalties/n)
    # onto a backend that has its own (llama.cpp, vLLM, Ollama). exclude_unset drops
    # unset defaults; exclude_none keeps the result a strict subset of the old payload
    # (an explicitly-sent optional null is not re-forwarded). Cradle sets stream_options
    # deliberately on the wrap path; that is added by the caller, not here.
    payload = req.model_dump(exclude_unset=True, exclude_none=True)
    payload["messages"] = [m.model_dump(exclude_none=True) for m in messages]
    return payload


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
        # An embed failure silently disables L2 for this request; without the
        # traceback it is invisible past the counter. Log it (no prompt text).
        log.warning("embed failed for request %s; L2 read skipped", ctx.request_id, exc_info=True)
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
        # Fail-open means a possibly-wrong hit is served; the traceback is the
        # only way to tell a timeout from a model crash. Log it (no prompt text).
        log.warning("rerank failed; serving the twice-gated hit (fail-open)", exc_info=True)
        m.l2_rerank_fail_open.inc()
        return True, "fail-open"
    if score >= threshold:
        return True, f"pass:{score:.4f}"
    m.l2_rerank_rejects.inc()
    return False, f"reject:{score:.4f}"


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
    # A cacheable stream+tools request (#43) is a normal miss whose miss path must
    # tee verbatim (the wrap path cannot carry a tool call). Flag the wire strategy;
    # layer_hit stays "miss". An L1 hit still replays normally (content-only).
    if req.stream and canonical.has_tools:
        ctx.cacheable_passthrough_stream = True
    _apply_volatility_guard(runtime, ctx)

    key = l1_key(canonical)
    ctx.l1_cache_key = key  # stash for the request log line (avoid re-hashing)
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
            return _replay(runtime, req, ctx, rec)

    # Single-flight follow-check (#57), before embed/L2: if an identical request is
    # already in flight, join it as a follower and skip embed + the whole L2 query
    # (embed_pool is a single worker — N followers would serialise on it). A fresh
    # leader answer beats an L2 near-match, so skipping L2 costs nothing.
    if runtime.settings.cache.singleflight and flight_eligible(ctx):
        fkey = flight_key(key, req.stream)
        existing = runtime.flights.get(fkey)
        if existing is not None and not is_stale(existing, runtime.settings.upstream.timeout_s):
            return await follow(runtime, req, ctx, existing)

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
                    log.debug(
                        "L2 candidate rejected req=%s key=%s score=%.6f reason=audit-floor floor=%.6f",
                        ctx.request_id, hit.record.key, hit.score, floor,
                    )
                    continue
                # Stage 2 (issue #5): cheap precision guard.
                reason = guard_reason(canonical.embed_text, hit.record.embed_text)
                if reason is not None:
                    ctx.l2_guard_reason = reason
                    m.l2_guard_rejects.labels(reason=reason).inc()
                    _note_candidate(ctx, hit, guard=reason, rerank=None, served=False)
                    log.debug(
                        "L2 candidate rejected req=%s key=%s score=%.6f reason=guard:%s",
                        ctx.request_id, hit.record.key, hit.score, reason,
                    )
                    continue
                # Stage 3: cross-encoder rerank for entity swaps the guard cannot
                # see. Fail-open: an unavailable reranker still serves.
                serve, note = await _rerank_ok(
                    runtime, canonical.embed_text, hit.record.embed_text
                )
                ctx.l2_rerank_note = note
                _note_candidate(ctx, hit, guard=None, rerank=note, served=serve)
                if not serve:
                    log.debug(
                        "L2 candidate rejected req=%s key=%s score=%.6f reason=rerank:%s",
                        ctx.request_id, hit.record.key, hit.score, note,
                    )
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
                    return _replay(runtime, req, ctx, rec)

    ctx.layer_hit = "miss"
    if ctx.cache_probe:
        return _probe(ctx, l1_key=key, l2_eligible=eligible)
    # Single-flight register (#57), after the probe return so a probe never registers
    # a flight it won't resolve. A leader reaching here is committed to calling
    # upstream (all hit/replay paths returned above). setdefault is atomic on the one
    # event loop with no intervening await — that is the concurrency control: if
    # another request registered first this tick, become a late follower instead.
    if runtime.settings.cache.singleflight and flight_eligible(ctx):
        fkey = flight_key(key, req.stream)
        mine = Flight(fkey)
        existing = runtime.flights.setdefault(fkey, mine)
        if existing is not mine and not is_stale(existing, runtime.settings.upstream.timeout_s):
            return await follow(runtime, req, ctx, existing)
        if existing is not mine:  # stale flight held the slot — replace it and lead
            m.flight_aborts.labels(reason="stale").inc()
            runtime.flights[fkey] = mine
        ctx.flight = mine
    if ctx.flight is None:
        return await _miss(runtime, req, ctx, vec=vec)
    # Leader raise-guard (#63): the leaf miss paths resolve the flight in their own
    # finally, but an exception RAISED before that finally is reached would leak the
    # flight and poison the key. Fail+release it here if still unresolved, then
    # re-raise (BaseException so CancelledError is covered; the unconditional re-raise
    # preserves cancellation). Scope note: this catches a *raise*, not an early
    # `return` — a leaf path that returns without resolving is NOT covered here (that
    # is why _miss_stream's own UpstreamError branch resolves the flight directly).
    # Idempotent with the leaf finally via the done.is_set() guard.
    try:
        return await _miss(runtime, req, ctx, vec=vec)
    except BaseException:
        if not ctx.flight.done.is_set():
            resolve_and_release(runtime, ctx.flight, error={
                "error": {"message": "single-flight leader failed", "type": "server_error",
                          "code": "upstream_error"}
            })
        raise


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


def _replay(runtime: Runtime, req: ChatRequest, ctx: RequestContext, rec) -> JSONResponse | StreamingResponse:
    _observe(ctx)
    log_request(runtime.settings, ctx, rec.response)
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
    # Record the compressed size under Cradle's own tokenizer so the true saving
    # (inbound - compressed, one accounting) is observable via a header (#52).
    # Only on a genuine miss: a BYPASS request also flows through _miss (it is an
    # uncacheable passthrough), but a compression saving is not a meaningful figure
    # there, so the header stays absent rather than misleading.
    if ctx.layer_hit == "miss":
        ctx.compressed_prompt_tokens = compressed.compressed_tokens
    tenant_tmpl = template_for(runtime.settings, ctx.principal.tenant_id)
    compressed.template.brand_prefix = tenant_tmpl.brand_prefix
    compressed.template.brand_suffix = tenant_tmpl.brand_suffix
    compressed.template.mode = tenant_tmpl.mode
    payload = _upstream_payload(req, compressed.messages)
    _name, target = resolve_upstream(runtime.settings, req.model)
    ctx.upstream_name = _name
    if req.stream:
        # On the WRAP path Cradle rebuilds the outbound stream and caches the result,
        # so it must know the real token usage even when the client did not request it
        # (otherwise the record stores empty usage and every later include_usage hit
        # replays {}). Ask upstream for the usage chunk. Do NOT touch the payload on
        # the bypass path (byte-exact passthrough) NOR the tool-stream passthrough-cache
        # path (#43): both tee the client verbatim, so injecting stream_options would
        # surface an unsolicited usage frame and can 400 backends that lack it. The
        # passthrough-cache path caches usage: acc.usage or {} — synthesize_sse omits
        # the usage frame on empty usage, so an empty-usage record replays cleanly.
        if ctx.layer_hit != "bypass" and not ctx.cacheable_passthrough_stream:
            payload["stream_options"] = {
                **(payload.get("stream_options") or {}),
                "include_usage": True,
            }
        return await _miss_stream(runtime, req, ctx, vec, compressed, payload, target)
    return await _miss_json(runtime, req, ctx, vec, compressed, payload, target)


async def _miss_json(runtime, req, ctx, vec, compressed, payload, target) -> JSONResponse:
    # Single-flight leader (#57): a try/finally is correct here (work is awaited
    # inline). Resolution is centralised in the finally on one local, out: it stays
    # None on the upstream-error return, so that path fail()s the flight; only a
    # clean success sets it and finish()es. The finally always removes the flight so
    # a later identical prompt L1-hits (or leads afresh) instead of hanging. A JSON
    # flight publishes no frames — only the completion dict — and only JSON callers
    # ever join it (the flight key is stream-scoped).
    out: dict | None = None
    upstream_exc: UpstreamError | None = None
    try:
        t0 = time.perf_counter()
        try:
            completion = await chat(
                runtime.http, target, payload, authorization=_client_auth(ctx)
            )
        except UpstreamError as exc:
            # Remember it so the flight finally can hand followers the leader's REAL
            # upstream status + retry/quota headers, not a generic 502 (#73).
            upstream_exc = exc
            return _upstream_error_response(runtime, ctx, exc)
        ctx.t_upstream_s = time.perf_counter() - t0
        usage = completion.get("usage") or {}
        ctx.upstream_prompt_tokens = int(usage.get("prompt_tokens") or compressed.compressed_tokens)
        t1 = time.perf_counter()
        if ctx.layer_hit == "bypass":
            merged = completion
        else:
            merged = merge(completion, compressed.template)
        ctx.t_reconstruct_s = time.perf_counter() - t1
        ttl = _effective_ttl(runtime, ctx)
        skip = cache_skip_reason(merged)
        if ctx.cache_no_store or ttl == 0:
            skip = skip or "no_store"
        # The upstream answer is complete and reconstructed; the leader has
        # succeeded. Commit followers to it (out = merged) BEFORE the cache write
        # (#70) so a writeback backend error can't flip this success into a 502 for
        # every follower — the write is best-effort (counted + logged, not raised).
        out = merged  # marks a clean success for the finally
        if ctx.layer_hit != "bypass" and ctx.canonical is not None and skip is None:
            rec = record_from(
                ctx.canonical,
                merged,
                ctx.inbound_prompt_tokens,
                ctx.upstream_prompt_tokens,
                ttl,
            )
            await writeback_best_effort(runtime, ctx.canonical, vec, rec)
        elif skip is not None:
            m.cache_write_skips.labels(reason=skip).inc()
        _observe(ctx)
        log_request(runtime.settings, ctx, merged)
        return JSONResponse(merged, headers=_headers(ctx))
    finally:
        if ctx.flight is not None:
            if out is not None:
                resolve_and_release(runtime, ctx.flight, completion=out)
            elif upstream_exc is not None:
                # A real upstream failure: give followers the leader's actual error
                # body, status and forwardable headers so their backoff matches the
                # leader's own client (#73).
                m.flight_aborts.labels(reason="upstream_error").inc()
                body = upstream_exc.body if isinstance(upstream_exc.body, dict) else {
                    "error": {"message": str(upstream_exc.body), "type": "server_error",
                              "code": "upstream_error"}
                }
                resolve_and_release(
                    runtime, ctx.flight, error=body,
                    error_status=upstream_exc.status, error_headers=upstream_exc.headers,
                )
            else:
                # An abort with no captured upstream status (e.g. a non-UpstreamError
                # raise). Followers get the generic 502 default.
                m.flight_aborts.labels(reason="upstream_error").inc()
                resolve_and_release(runtime, ctx.flight, error={
                    "error": {"message": "single-flight leader failed upstream",
                              "type": "server_error", "code": "upstream_error"}
                })
