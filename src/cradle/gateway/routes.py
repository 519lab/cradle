from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from pydantic import ValidationError

from cradle.gateway.context import RequestContext
from cradle.gateway.errors import openai_error
from cradle.gateway.models import ChatRequest
from cradle.gateway.pipeline import handle_chat
from cradle.metrics.prometheus import render
from cradle.runtime import Runtime
from cradle.tenancy import authenticate
from cradle.upstream.openai import list_models

router = APIRouter()


def _runtime(request: Request) -> Runtime:
    return request.app.state.runtime


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(request: Request) -> Response:
    rt = _runtime(request)
    ok = True
    if rt.settings.features.cache and not rt.l1_ready:
        ok = False
    if rt.settings.features.l2 and not (rt.l2_ready and rt.embedder_ready):
        ok = False
    # When rerank is enabled it is a correctness control (issue #5): if it failed
    # to load, L2 would serve entity-swap near-misses unverified. Report not-ready
    # rather than take traffic in that silently-degraded state.
    if rt.settings.features.l2 and rt.settings.features.l2_rerank and not rt.reranker_ready:
        ok = False
    if ok:
        return Response(content='{"status":"ready"}', media_type="application/json")
    return Response(content='{"status":"not_ready"}', media_type="application/json", status_code=503)


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    rt = _runtime(request)
    if rt.settings.metrics.require_auth:
        result = await authenticate(request, rt.settings, rt.principals)
        if isinstance(result, JSONResponse):
            return result
    body, ctype = render()
    return PlainTextResponse(body, media_type=ctype)


@router.get("/v1/models")
async def models(request: Request):
    rt = _runtime(request)
    result = await authenticate(request, rt.settings, rt.principals)
    if isinstance(result, JSONResponse):
        return result
    auth = request.headers.get("authorization")
    return await list_models(rt.http, rt.settings, authorization=auth)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    rt = _runtime(request)
    principal = await authenticate(request, rt.settings, rt.principals)
    if isinstance(principal, JSONResponse):
        return principal
    raw = await request.body()
    if len(raw) > rt.settings.server.max_body_bytes:
        return openai_error("payload too large", "invalid_request_error", "payload_too_large", 413)
    try:
        body = ChatRequest.model_validate_json(raw)
    except ValidationError:
        return openai_error("invalid request body", "invalid_request_error", "invalid_request", 400)
    ctx = RequestContext(
        request_id=str(uuid.uuid4()),
        principal=principal,
        headers={"authorization": request.headers.get("authorization") or ""},
    )
    _apply_cache_directives(ctx, request, rt.settings.cache.ttl_s)
    return await handle_chat(rt, body, ctx)


def _apply_cache_directives(ctx, request: Request, max_ttl: int) -> None:
    """Per-request cache controls (enhancement #2), from request headers.

    X-Cradle-Cache-Control: comma list of no-store / no-cache / refresh / probe.
    X-Cradle-Cache-TTL: seconds, clamped to [0, cache.ttl_s].
    """
    control = (request.headers.get("x-cradle-cache-control") or "").lower()
    directives = {d.strip() for d in control.split(",") if d.strip()}
    if "no-cache" in directives or "refresh" in directives:
        ctx.cache_no_read = True
    if "no-store" in directives:
        ctx.cache_no_store = True
    if "probe" in directives:
        ctx.cache_probe = True
    ttl_raw = request.headers.get("x-cradle-cache-ttl")
    if ttl_raw:
        try:
            ttl = int(ttl_raw)
        except ValueError:
            return
        ctx.cache_ttl_override = max(0, min(ttl, max_ttl))
