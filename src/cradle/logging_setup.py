"""Logging setup and the per-request log line.

Two responsibilities, both driven by ``settings.logging``:

* ``configure_logging`` installs one handler on the ``cradle`` logger tree so
  that ``log.info``/``log.debug`` calls actually escape the process. Without it,
  ``python -m cradle`` runs under uvicorn's default config, the ``cradle`` loggers
  sit at WARNING, and every request-path line is silently dropped — the reason
  Cradle appeared to log only uvicorn's access lines. Uvicorn's own loggers are
  left untouched, so its access log still shows.

* ``log_request`` is the single redaction chokepoint for the request-completion
  line. Every field that could carry prompt or response text passes through here
  and is gated on ``logging.content`` (``none`` → keys/scores/timings only), so
  the "no prompt text in logs by default" guarantee is enforced in one place and
  is unit-testable, rather than scattered across call sites. It is the request-path
  analogue of ``l2.audit_log_text``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cradle.config import LoggingSettings, Settings
    from cradle.gateway.context import RequestContext

# Ordered so a numeric compare answers "is content >= prompts?".
_CONTENT_RANK = {"none": 0, "prompts": 1, "prompts_and_completions": 2}

_LINE_LOGGER = "cradle.request"


def configure_logging(settings: Settings) -> None:
    """Install a handler + level on the ``cradle`` logger tree. Idempotent.

    Safe to call more than once (tests build ``create_app`` directly) and safe
    to never call (the handler is only added if absent). Only the ``cradle``
    namespace is configured; uvicorn's loggers keep their own handlers.
    """
    root = logging.getLogger("cradle")
    root.setLevel(settings.logging.level)
    # propagate=False keeps cradle lines off the root logger, so they are not
    # double-emitted through uvicorn's root handler.
    root.propagate = False
    handler = _existing_handler(root)
    if handler is None:
        handler = logging.StreamHandler()
        handler._cradle = True  # type: ignore[attr-defined]  # idempotency marker
        root.addHandler(handler)
    handler.setFormatter(_formatter(settings.logging.format))


def _existing_handler(root: logging.Logger) -> logging.StreamHandler | None:
    for h in root.handlers:
        if isinstance(h, logging.StreamHandler) and getattr(h, "_cradle", False):
            return h
    return None


def _formatter(fmt: str) -> logging.Formatter:
    if fmt == "json":
        return _JsonFormatter()
    return logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        return json.dumps(payload, separators=(",", ":"))


def _truncate(text: str, limit: int) -> str:
    if limit >= 0 and len(text) > limit:
        return text[:limit] + f"…(+{len(text) - limit})"
    return text


def _completion_text(completion: dict[str, Any] | None) -> str:
    if not completion:
        return ""
    choices = completion.get("choices") or [{}]
    return str(((choices[0] or {}).get("message") or {}).get("content") or "")


def request_fields(
    settings: LoggingSettings, ctx: RequestContext, completion: dict[str, Any] | None
) -> dict[str, Any]:
    """Build the field dict for one request line, redacted per logging.content.

    THE redaction chokepoint. Non-sensitive fields are always present; prompt
    and completion text appear only when logging.content opts in, and are then
    truncated to max_text_chars.
    """
    canonical = ctx.canonical
    fields: dict[str, Any] = {
        "request_id": ctx.request_id,
        "cache": ctx.layer_hit,
        "upstream": ctx.upstream_name,
        "inbound_tok": ctx.inbound_prompt_tokens,
        "upstream_tok": ctx.upstream_prompt_tokens,
    }
    if canonical is not None:
        fields["model"] = canonical.model
        fields["tenant"] = canonical.tenant_id  # already a token hash (tenancy.py)
        # The pipeline stashed l1_key(canonical) on ctx; None on the bypass path.
        if ctx.l1_cache_key is not None:
            fields["key"] = ctx.l1_cache_key
    # Decision annotations — only present when they fired.
    if ctx.l2_score is not None:
        fields["sim"] = round(ctx.l2_score, 6)
    if ctx.l2_guard_reason is not None:
        fields["guard"] = ctx.l2_guard_reason
    if ctx.l2_rerank_note is not None:
        fields["rerank"] = ctx.l2_rerank_note
    if ctx.volatile_reason is not None:
        fields["volatile"] = ctx.volatile_reason
    if ctx.audit_scheduled:
        fields["audit"] = "scheduled"
    # Per-stage timings (seconds), only the stages that ran.
    for name, val in (
        ("t_l1", ctx.t_l1_s),
        ("t_embed", ctx.t_embed_s),
        ("t_l2", ctx.t_l2_s),
        ("t_compress", ctx.t_compress_s),
        ("t_upstream", ctx.t_upstream_s),
        ("t_reconstruct", ctx.t_reconstruct_s),
    ):
        if val:
            fields[name] = round(val, 6)
    # Sensitive text, gated.
    rank = _CONTENT_RANK.get(settings.content, 0)
    if rank >= _CONTENT_RANK["prompts"] and canonical is not None:
        fields["prompt"] = _truncate(canonical.embed_text, settings.max_text_chars)
    if rank >= _CONTENT_RANK["prompts_and_completions"]:
        text = _completion_text(completion)
        if text:
            fields["completion"] = _truncate(text, settings.max_text_chars)
    return fields


def log_request(
    settings: Settings,
    ctx: RequestContext,
    completion: dict[str, Any] | None = None,
    *,
    disconnected: bool = False,
    error_status: int | None = None,
    upstream_status: int | None = None,
) -> None:
    """Emit one request-completion line, redacted per settings.logging.content."""
    fields = request_fields(settings.logging, ctx, completion)
    if disconnected:
        fields["disconnected"] = True
    if error_status is not None:
        fields["status"] = error_status
    if upstream_status is not None:
        fields["upstream_status"] = upstream_status
    log = logging.getLogger(_LINE_LOGGER)
    log.info(_render(fields, settings.logging.format), extra={"fields": fields})


def _render(fields: dict[str, Any], fmt: str) -> str:
    if fmt == "json":
        # The JSON formatter serializes `fields`; the message stays a short tag.
        return "request"
    return " ".join(f"{k}={_fmt_value(v)}" for k, v in fields.items())


def _fmt_value(v: Any) -> str:
    s = str(v)
    # Keep a key=value line parseable: quote anything with whitespace. Keep it
    # readable for a human watching stdout: ensure_ascii=False so a truncated
    # prompt shows "…(+199)" rather than "…(+199)".
    if any(c.isspace() for c in s):
        return json.dumps(s, ensure_ascii=False)
    return s
