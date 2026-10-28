from __future__ import annotations

from typing import Any

from cradle.gateway.models import ReconstructionTemplate


def wrap_prefix(template: ReconstructionTemplate) -> str:
    if template.mode == "passthrough":
        return ""
    return template.brand_prefix


def wrap_suffix(template: ReconstructionTemplate, upstream_content: str) -> str:
    if template.mode == "passthrough":
        return ""
    missing = [line for line in template.format_instructions if line not in upstream_content]
    fmt = ("\n" + "\n".join(missing)) if missing else ""
    return fmt + template.brand_suffix


def wrap_content(template: ReconstructionTemplate, upstream_content: str) -> str:
    return wrap_prefix(template) + upstream_content + wrap_suffix(template, upstream_content)


def merge(completion: dict[str, Any], template: ReconstructionTemplate) -> dict[str, Any]:
    out = dict(completion)
    choices = list(out.get("choices") or [])
    if not choices:
        return out
    choice = dict(choices[0])
    message = dict(choice.get("message") or {})
    if message.get("tool_calls"):
        return out
    upstream = message.get("content") or ""
    if not isinstance(upstream, str):
        upstream = str(upstream)
    message["content"] = wrap_content(template, upstream)
    choice["message"] = message
    out["choices"] = [choice, *choices[1:]]
    return out
