from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from cradle.config import Settings, UpstreamSettings
from cradle.upstream.route import advertised_models

log = logging.getLogger("cradle.upstream")


# Response headers worth forwarding to the caller so client-side retry/backoff and
# quota accounting keep working. Deliberately small: never forward hop-by-hop or
# body-framing headers (content-length/content-encoding/transfer-encoding) — those
# describe Cradle's re-encoded body, not the upstream's, and would corrupt it.
_FORWARD_HEADER_PREFIXES = ("x-ratelimit-",)
_FORWARD_HEADER_NAMES = frozenset({"retry-after"})
# Upstream request id, renamed so it never clobbers Cradle's own X-Request-ID.
_UPSTREAM_REQUEST_ID_HEADERS = ("x-request-id", "openai-request-id", "x-amzn-requestid")


def forwardable_headers(headers: Any) -> dict[str, str]:
    """Pick the allowlisted upstream response headers to relay to the caller."""
    out: dict[str, str] = {}
    for name, value in headers.items():
        low = name.lower()
        if low in _FORWARD_HEADER_NAMES or low.startswith(_FORWARD_HEADER_PREFIXES):
            out[name] = value
        elif low in _UPSTREAM_REQUEST_ID_HEADERS and "x-cradle-upstream-request-id" not in out:
            out["x-cradle-upstream-request-id"] = value
    return out


# Request direction (#81) is the opposite policy: a DENYLIST, so any header an
# application sends (session/chat ids, tracing, vendor routing) reaches the
# upstream without Cradle knowing about it. Stripped are only headers that are
# hop-by-hop (RFC 9110 §7.6.1), that describe the client→Cradle hop rather than
# Cradle's re-serialized body, or that are credentials/control not meant for the
# upstream. `authorization` is decided separately by `_headers` (a Cradle key in
# keyed mode must never leak); `accept-encoding` is left to httpx, which only
# advertises encodings it can decode.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-connection", "te", "trailer",
    "transfer-encoding", "upgrade",
})
_CRADLE_OWNED = frozenset({
    "host", "content-length", "content-type", "accept-encoding", "expect",
    "authorization", "proxy-authorization",
})
_CRADLE_CONTROL_PREFIX = "x-cradle-"


def forward_request_headers(headers: Any) -> dict[str, str]:
    """Client request headers to relay upstream: everything except the denylist."""
    named_by_connection = {
        token.strip().lower()
        for value in _values(headers, "connection")
        for token in value.split(",")
    }
    out: dict[str, str] = {}
    for name, value in headers.items():
        low = name.lower()
        if (
            low in _HOP_BY_HOP
            or low in _CRADLE_OWNED
            or low in named_by_connection
            or low.startswith(_CRADLE_CONTROL_PREFIX)
        ):
            continue
        out[low] = value
    return out


def _values(headers: Any, name: str) -> list[str]:
    return [v for k, v in headers.items() if k.lower() == name]


class UpstreamError(Exception):
    def __init__(self, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
        super().__init__(f"upstream {status}")
        self.status = status
        self.body = body
        # Allowlisted upstream response headers (retry-after, x-ratelimit-*, request id).
        self.headers = headers or {}


def _url(upstream: UpstreamSettings, path: str) -> str:
    return upstream.base_url.rstrip("/") + path


def _headers(upstream: UpstreamSettings, client_headers: Any = None) -> dict[str, str]:
    client_headers = client_headers or {}
    headers = forward_request_headers(client_headers)
    headers["content-type"] = "application/json"
    client_authorization = next(iter(_values(client_headers, "authorization")), None)
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
    client_headers: Any = None,
) -> dict[str, Any]:
    body = dict(payload)
    body.pop("stream", None)
    try:
        resp = await client.post(
            _url(upstream, "/chat/completions"),
            json=body,
            headers=_headers(upstream, client_headers),
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
        raise UpstreamError(resp.status_code, data, forwardable_headers(resp.headers))
    return resp.json()


async def start_chat_stream(
    client: httpx.AsyncClient,
    upstream: UpstreamSettings,
    payload: dict[str, Any],
    client_headers: Any = None,
) -> httpx.Response:
    body = dict(payload)
    body["stream"] = True
    request = client.build_request(
        "POST",
        _url(upstream, "/chat/completions"),
        json=body,
        headers=_headers(upstream, client_headers),
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
        headers = forwardable_headers(resp.headers)
        await resp.aclose()
        raise UpstreamError(resp.status_code, _error_body(raw), headers)
    return resp


def _advertised_list(settings: Settings) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "cradle"}
            for m in advertised_models(settings)
        ],
    }


def _single_fallback_backend(settings: Settings) -> bool:
    """True when there is exactly one backend: no named upstreams/routes.

    In that case Cradle's `/v1/models` should reflect what the fallback upstream
    actually serves, rather than the static `upstream.models` placeholder — so a
    client's model dropdown shows the real model. With named routes present,
    proxying one backend's `/models` would be ambiguous, so keep the static list.
    """
    return not settings.upstreams and not settings.routes


async def list_models(
    client: httpx.AsyncClient,
    settings: Settings,
    client_headers: Any = None,
) -> dict[str, Any]:
    # Passthrough when explicitly requested, OR automatically in single-backend
    # (fallback-only) deployments so /v1/models reflects the real upstream model.
    passthrough = settings.upstream.models_passthrough or _single_fallback_backend(settings)
    if not passthrough:
        return _advertised_list(settings)
    try:
        resp = await client.get(
            _url(settings.upstream, "/models"),
            headers=_headers(settings.upstream, client_headers),
            timeout=settings.upstream.timeout_s,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        # /v1/models is polled by clients on every connection check, so a down or
        # slow upstream must not turn it into a 500 — fall back to the static list.
        log.warning("upstream /models unavailable; serving advertised model list")
        return _advertised_list(settings)
