from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, Response

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
    if ok:
        return Response(content='{"status":"ready"}', media_type="application/json")
    return Response(content='{"status":"not_ready"}', media_type="application/json", status_code=503)


@router.get("/metrics")
async def metrics(request: Request) -> Response:
    rt = _runtime(request)
    if rt.settings.metrics.require_auth:
        result = await authenticate(request, rt.settings, rt.principals)
        if hasattr(result, "status_code"):
            return result
    body, ctype = render()
    return PlainTextResponse(body, media_type=ctype)


@router.get("/v1/models")
async def models(request: Request):
    rt = _runtime(request)
    result = await authenticate(request, rt.settings, rt.principals)
    if hasattr(result, "status_code"):
        return result
    return await list_models(rt.http, rt.settings)


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    rt = _runtime(request)
    cl = request.headers.get("content-length")
    if cl and int(cl) > rt.settings.server.max_body_bytes:
        return openai_error("payload too large", "invalid_request_error", "payload_too_large", 413)
    principal = await authenticate(request, rt.settings, rt.principals)
    if hasattr(principal, "status_code"):
        return principal
    ctx = RequestContext(request_id=str(uuid.uuid4()), principal=principal)
    return await handle_chat(rt, body, ctx)
