from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from cradle.config import Settings


class UpstreamError(Exception):
    def __init__(self, status: int, body: Any) -> None:
        super().__init__(f"upstream {status}")
        self.status = status
        self.body = body


def _url(settings: Settings, path: str) -> str:
    return settings.upstream.base_url.rstrip("/") + path


def _headers(settings: Settings) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    import os

    key = os.environ.get(settings.upstream.api_key_env, "")
    if key:
        headers["authorization"] = f"Bearer {key}"
    return headers


async def chat(client: httpx.AsyncClient, settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("stream", None)
    resp = await client.post(_url(settings, "/chat/completions"), json=body, headers=_headers(settings))
    if resp.status_code >= 400:
        try:
            data = resp.json()
        except Exception:
            data = {"error": {"message": resp.text, "type": "server_error", "code": "upstream_error"}}
        raise UpstreamError(resp.status_code, data)
    return resp.json()


async def chat_stream(
    client: httpx.AsyncClient, settings: Settings, payload: dict[str, Any]
) -> AsyncIterator[str]:
    body = dict(payload)
    body["stream"] = True
    async with client.stream(
        "POST",
        _url(settings, "/chat/completions"),
        json=body,
        headers=_headers(settings),
    ) as resp:
        if resp.status_code >= 400:
            raw = (await resp.aread()).decode("utf-8", "replace")
            raise UpstreamError(resp.status_code, raw)
        async for line in resp.aiter_lines():
            yield line


async def list_models(client: httpx.AsyncClient, settings: Settings) -> dict[str, Any]:
    if not settings.upstream.models_passthrough:
        return {
            "object": "list",
            "data": [{"id": m, "object": "model", "owned_by": "cradle"} for m in settings.upstream.models],
        }
    resp = await client.get(_url(settings, "/models"), headers=_headers(settings))
    resp.raise_for_status()
    return resp.json()
