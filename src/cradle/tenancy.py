from __future__ import annotations

import hmac
import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from cradle.cache.records import Principal
from cradle.config import Settings
from cradle.gateway.errors import openai_error

log = logging.getLogger("cradle.tenancy")


def load_principals(settings: Settings) -> list[tuple[str, Principal]]:
    out: list[tuple[str, Principal]] = []
    for key in settings.auth.keys:
        secret = os.environ.get(key.token_env, "")
        if not secret:
            continue
        out.append(
            (
                secret,
                Principal(tenant_id=key.tenant_id, user_id=key.user_id, key_id=key.token_env),
            )
        )
    return out


def assert_auth_ready(settings: Settings, principals: list[tuple[str, Principal]]) -> None:
    if principals:
        return
    bind = settings.server.host
    loopback = bind in {"127.0.0.1", "::1", "localhost"}
    if settings.auth.allow_insecure_loopback and loopback:
        log.warning("no API keys configured; insecure loopback auth allowed")
        return
    raise RuntimeError(
        "no proxy API keys resolved from auth.keys token_env; "
        "set CRADLE_API_KEY or enable auth.allow_insecure_loopback on loopback"
    )


def _same_length_compare(supplied: str, secret: str) -> bool:
    if len(supplied) != len(secret):
        hmac.compare_digest(secret.encode("utf-8"), secret.encode("utf-8"))
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), secret.encode("utf-8"))


async def authenticate(
    request: Request, settings: Settings, principals: list[tuple[str, Principal]]
) -> Principal | JSONResponse:
    header = request.headers.get("authorization") or ""
    supplied = ""
    if header.lower().startswith("bearer "):
        supplied = header[7:].strip()
    found: Principal | None = None
    dummy = "0" * 32
    if not principals:
        hmac.compare_digest(dummy.encode(), dummy.encode())
        if settings.auth.allow_insecure_loopback:
            return Principal(tenant_id="default", user_id="default", key_id="insecure")
        return _unauth()
    for secret, principal in principals:
        candidate = supplied if supplied else dummy
        if supplied and _same_length_compare(candidate, secret):
            found = principal
        else:
            _same_length_compare(secret, secret)
    if found is None:
        return _unauth()
    return found


def _unauth():
    return openai_error("invalid api key", "invalid_request_error", "invalid_api_key", 401)
