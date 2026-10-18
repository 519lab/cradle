from __future__ import annotations

import hashlib
import hmac
import logging
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from cradle.cache.records import Principal
from cradle.config import Settings
from cradle.gateway.errors import openai_error

log = logging.getLogger("cradle.tenancy")

ANON = Principal(tenant_id="anon", user_id="anon", key_id="anon")


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
    if settings.auth.keys and not principals:
        raise RuntimeError(
            "auth.keys is set but no token_env values resolved; "
            "unset auth.keys for intercept mode (forward client credentials) "
            "or export the listed env vars"
        )


def bearer_token(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def principal_from_forwarded_token(token: str) -> Principal:
    if not token:
        return ANON
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return Principal(tenant_id=digest, user_id=digest, key_id="forwarded")


def _same_length_compare(supplied: str, secret: str) -> bool:
    if len(supplied) != len(secret):
        hmac.compare_digest(secret.encode("utf-8"), secret.encode("utf-8"))
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), secret.encode("utf-8"))


async def authenticate(
    request: Request, settings: Settings, principals: list[tuple[str, Principal]]
) -> Principal | JSONResponse:
    supplied = bearer_token(request)
    if not principals:
        return principal_from_forwarded_token(supplied)
    found: Principal | None = None
    dummy = "0" * 32
    for secret, principal in principals:
        candidate = supplied if supplied else dummy
        if supplied and _same_length_compare(candidate, secret):
            found = principal
        else:
            _same_length_compare(secret, secret)
    if found is None:
        return openai_error("invalid api key", "invalid_request_error", "invalid_api_key", 401)
    return found
