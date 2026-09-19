"""Response-shaping and observability helpers shared by the JSON and streaming
miss paths.

These are leaf helpers — they derive headers, an effective TTL, an error
response, a request flag, the client auth, and the per-request metric/observe
fan-out from ``(runtime, ctx, req)`` and call nothing in the pipeline
orchestration. They live here (not in ``pipeline.py`` or ``stream.py``) so both
importers depend on a leaf module and the import graph stays acyclic. Extracted
from ``pipeline.py`` to keep every module under the 600-line rule; behavior
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from cradle.gateway.context import RequestContext
from cradle.gateway.errors import openai_error
from cradle.gateway.models import ChatRequest
from cradle.logging_setup import log_request
from cradle.metrics import prometheus as m
from cradle.upstream.openai import UpstreamError

if TYPE_CHECKING:
    from cradle.runtime import Runtime


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
    if ctx.compressed_prompt_tokens is not None:
        # Present on a genuine miss (JSON and streaming alike), the only path where
        # compression ran and a saving is meaningful. Absent on hits and bypass so
        # it is never misread as "compressed to 0" (#52). Compare it to
        # X-Cradle-Inbound-Tokens (same tokenizer) for the true saving, NOT to
        # X-Cradle-Upstream-Tokens (a different backend tokenizer + chat template).
        h["X-Cradle-Compressed-Tokens"] = str(ctx.compressed_prompt_tokens)
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


def _include_usage(req: ChatRequest) -> bool:
    opts = req.stream_options or {}
    return bool(opts.get("include_usage"))


def _client_auth(ctx: RequestContext) -> str | None:
    return ctx.headers.get("authorization") or None


def _upstream_error_response(
    runtime: Runtime, ctx: RequestContext, exc: UpstreamError
) -> JSONResponse:
    m.upstream_errors.labels(status=str(exc.status)).inc()
    status = 502 if exc.status >= 500 else exc.status
    # An upstream 429/5xx is the most likely reason someone is watching the logs,
    # so it must still produce a request line (with the status), not vanish into
    # uvicorn's access log alone. _observe is deliberately NOT called: it hardcodes
    # status="200" and would mislabel cradle_requests_total; the error is already
    # counted in cradle_upstream_errors_total above.
    log_request(runtime.settings, ctx, None, error_status=status, upstream_status=exc.status)
    # Relay upstream retry/quota headers (retry-after, x-ratelimit-*, request id) so a
    # client's backoff on a 429/503 still works even though Cradle re-frames the body.
    headers = {**_headers(ctx), **exc.headers}
    if isinstance(exc.body, dict):
        return JSONResponse(exc.body, status_code=status, headers=headers)
    return openai_error(str(exc.body), "server_error", "upstream_error", status, headers)


def _effective_ttl(runtime: Runtime, ctx: RequestContext) -> int:
    ttl = ctx.cache_ttl_override
    return ttl if ttl is not None else runtime.settings.cache.ttl_s


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
