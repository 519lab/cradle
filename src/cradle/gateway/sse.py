from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from cradle.cache.records import CacheRecord


def encode_chunk(obj: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


def encode_done() -> bytes:
    return b"data: [DONE]\n\n"


def _base(acc_id: str, created: int, model: str) -> dict[str, Any]:
    return {
        "id": acc_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
    }


def content_frame(acc_id: str, created: int, model: str, text: str) -> dict[str, Any]:
    obj = _base(acc_id, created, model)
    obj["choices"][0]["delta"] = {"content": text}
    obj["choices"][0]["finish_reason"] = None
    return obj


def role_frame(acc_id: str, created: int, model: str) -> dict[str, Any]:
    obj = _base(acc_id, created, model)
    obj["choices"][0]["delta"] = {"role": "assistant"}
    return obj


def finish_frame(acc_id: str, created: int, model: str, reason: str) -> dict[str, Any]:
    obj = _base(acc_id, created, model)
    obj["choices"][0]["delta"] = {}
    obj["choices"][0]["finish_reason"] = reason
    return obj


def usage_frame(acc_id: str, created: int, model: str, usage: dict[str, int]) -> dict[str, Any]:
    obj = _base(acc_id, created, model)
    obj["choices"] = []
    obj["usage"] = usage
    return obj


def error_frame(message: str) -> dict[str, Any]:
    return {"error": {"message": message, "type": "server_error", "code": "upstream_error"}}


@dataclass
class StreamAccumulator:
    outbound_id: str
    outbound_created: int
    model: str
    role: str = "assistant"
    content_parts: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    saw_done: bool = False
    error: bool = False
    tool_call_seen: bool = False
    client_connected: bool = True

    @property
    def content(self) -> str:
        return "".join(self.content_parts)


def parse_and_accumulate(line: str, acc: StreamAccumulator) -> str | None:
    """Parse one SSE line. Return string delta.content to tee, else None."""
    raw = line.strip()
    if not raw or raw.startswith(":"):
        return None
    if raw.startswith("data:"):
        payload = raw[5:].lstrip()
    else:
        payload = raw
    if payload == "[DONE]":
        acc.saw_done = True
        return None
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("error"):
        acc.error = True
        return None
    choices = obj.get("choices") or []
    if not choices:
        if isinstance(obj.get("usage"), dict):
            acc.usage = obj["usage"]
        return None
    choice = choices[0]
    delta = choice.get("delta") or {}
    finish = choice.get("finish_reason")
    if finish:
        acc.finish_reason = finish
    if delta.get("tool_calls"):
        acc.tool_call_seen = True
    if isinstance(obj.get("usage"), dict):
        acc.usage = obj["usage"]
    content = delta.get("content")
    if isinstance(content, str) and content:
        acc.content_parts.append(content)
        return content
    return None


def synthesize_sse(record: CacheRecord, *, include_usage: bool) -> Iterator[bytes]:
    resp = record.response
    rec_id = str(resp.get("id") or record.key)
    created = int(resp.get("created") or record.created_at)
    model = str(resp.get("model") or record.model)
    choices = resp.get("choices") or [{}]
    message = (choices[0] or {}).get("message") or {}
    body = message.get("content") or ""
    finish = (choices[0] or {}).get("finish_reason") or "stop"
    yield encode_chunk(role_frame(rec_id, created, model))
    for i in range(0, len(body), 16):
        yield encode_chunk(content_frame(rec_id, created, model, body[i : i + 16]))
    yield encode_chunk(finish_frame(rec_id, created, model, finish))
    usage = resp.get("usage")
    if include_usage and isinstance(usage, dict):
        yield encode_chunk(usage_frame(rec_id, created, model, usage))
    yield encode_done()
