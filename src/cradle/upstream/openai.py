from __future__ import annotations

import json
import os
from typing import Any

import httpx

from cradle.config import Settings, UpstreamSettings
from cradle.upstream.route import advertised_models


class UpstreamError(Exception):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"upstream {status}")
        self.status = status
        self.body = body


def _url(upstream: UpstreamSettings, path: str) -> str:
    return upstream.base_url.rstrip("/") + path


def _headers(upstream: UpstreamSettings, client_authorization: str | None = None) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if upstream.pass_through_client_auth and client_authorization:
        headers["authorization"] = client_authorization
        return headers
    key = os.environ.get(upstream.api_key_env, "")
    if key:
        headers["authorization"] = f"Bearer {key}"
    return headers


def _error_body(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": {"message": raw, "type": "server_error", "code": "upstream_error"}}


async def chat(
    client: httpx.AsyncClient,
    upstream: UpstreamSettings,
    payload: dict[str, Any],
    authorization: str | None = None,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop("stream", None)
    try:
        resp = await client.post(
            _url(upstream, "/chat/completions"),
            json=body,
            headers=_headers(upstream, authorization),
            timeout=upstream.timeout_s,
        )
    except httpx.RequestError as exc:
        raise UpstreamError(
            502,
            {"error": {"message": str(exc), "type": "server_error", "code": "upstream_error"}},
        ) from exc
    if resp.status_code >= 400:
        try:
            data = resp.json()
        except Exception:
            data = _error_body(resp.text)
        raise UpstreamError(resp.status_code, data)
    return resp.json()


async def start_chat_stream(
    client: httpx.AsyncClient,
    upstream: UpstreamSettings,
    payload: dict[str, Any],
    authorization: str | None = None,
) -> httpx.Response:
    body = dict(payload)
    body["stream"] = True
    request = client.build_request(
        "POST",
        _url(upstream, "/chat/completions"),
        json=body,
        headers=_headers(upstream, authorization),
        timeout=upstream.timeout_s,
    )
    try:
        resp = await client.send(request, stream=True)
    except httpx.RequestError as exc:
        raise UpstreamError(
            502,
            {"error": {"message": str(exc), "type": "server_error", "code": "upstream_error"}},
        ) from exc
    if resp.status_code >= 400:
        raw = (await resp.aread()).decode("utf-8", "replace")
        await resp.aclose()
        raise UpstreamError(resp.status_code, _error_body(raw))
    return resp


async def list_models(
    client: httpx.AsyncClient,
    settings: Settings,
    authorization: str | None = None,
) -> dict[str, Any]:
    if not settings.upstream.models_passthrough:
        return {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "owned_by": "cradle"}
                for m in advertised_models(settings)
            ],
        }
    resp = await client.get(
        _url(settings.upstream, "/models"),
        headers=_headers(settings.upstream, authorization),
        timeout=settings.upstream.timeout_s,
    )
    resp.raise_for_status()
    return resp.json()
