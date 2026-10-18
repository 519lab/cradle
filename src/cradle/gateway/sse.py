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


def reasoning_frame(
    acc_id: str, created: int, model: str, text: str, *, key: str = "reasoning_content"
) -> dict[str, Any]:
    """A reasoning delta (a model's thinking). Emitted before content frames so a
    replay matches the real upstream shape. ``key`` preserves whichever name the
    upstream used (``reasoning_content`` or ``reasoning``)."""
    obj = _base(acc_id, created, model)
    obj["choices"][0]["delta"] = {key: text}
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


def usage_frame(
    acc_id: str,
    created: int,
    model: str,
    usage: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    obj = _base(acc_id, created, model)
    obj["choices"] = []
    obj["usage"] = usage
    if extra:
        obj.update(extra)
    return obj


# Top-level, provider-set completion fields that describe the response but are not
# regeneration inputs. Forwarded verbatim on the wrap path (DESIGN.md wrap contract):
# unlike `id`/`created`, which are deliberately local, these carry upstream truth the
# caller may need (e.g. `system_fingerprint` for reproducibility audits).
PASS_THROUGH_TOP_FIELDS = ("system_fingerprint", "service_tier")
# Delta fields the content-only cache representation replays as-is (#43).
# `tool_calls` is here so it doesn't trip the allowlist on its own — it is gated
# separately via tool_call_seen (a tool-call response is never cached anyway).
_CACHEABLE_DELTA_KEYS = frozenset({"role", "content", "tool_calls"})
# Reasoning deltas (a model's thinking). Auxiliary, not semantically load-bearing:
# they are accumulated, stored, and replayed on their own frame (#46/#49), so they
# do NOT disable caching. Two source key names are seen in the wild; both fold to
# category B and the observed key is preserved for faithful replay. Ordered (not a
# set) so that if a message ever carried both, replay is deterministic and prefers
# the canonical `reasoning_content`.
_REASONING_DELTA_KEYS: tuple[str, ...] = ("reasoning_content", "reasoning")
# Any delta key outside A ∪ B (legacy function_call, refusal, annotations, audio, …)
# genuinely changes meaning if dropped, so it still disables caching (fail-closed).
# `id`/`created` are deliberately local on the wrap path (DESIGN.md wrap contract);
# they must never be forwarded, so they must never enter the pass-through allowlist.
assert not ({"id", "created", "object"} & set(PASS_THROUGH_TOP_FIELDS))


def error_frame(error: str | dict[str, Any]) -> dict[str, Any]:
    """Build an SSE error frame.

    A dict is an upstream error object (`{message,type,code,...}`) forwarded
    verbatim so the caller sees the real upstream failure. A string is a
    Cradle-originated message wrapped in the OpenAI error shape.
    """
    if isinstance(error, dict):
        return {"error": error}
    return {"error": {"message": error, "type": "server_error", "code": "upstream_error"}}


@dataclass
class StreamAccumulator:
    outbound_id: str
    outbound_created: int
    model: str
    role: str = "assistant"
    content_parts: list[str] = field(default_factory=list)
    # Reasoning deltas (a reasoning model's thinking), accumulated separately from
    # content so they can be stored and replayed on their own SSE frame (#46/#49).
    # reasoning_key remembers whichever name upstream used ("reasoning_content" or
    # "reasoning"); last_reasoning is the chunk parsed from the most recent line, so
    # the wrap path can emit a reasoning frame for it (parse_and_accumulate's return
    # value is reserved for the content tee — see its docstring).
    reasoning_parts: list[str] = field(default_factory=list)
    reasoning_key: str | None = None
    last_reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict[str, int] | None = None
    saw_done: bool = False
    error: bool = False
    error_payload: dict[str, Any] | None = None
    tool_call_seen: bool = False
    # Fail-closed cache gate for the tool-stream passthrough path (#43). Set when
    # a chunk carries anything the content-only cache representation cannot
    # faithfully replay: a delta key outside {role, content}, a legacy
    # function_call, or an unparseable/non-object payload. The passthrough path
    # refuses to cache when this is set; the wrap path ignores it (it has its own
    # tool_call_seen abort). Purely additive — never affects what is teed.
    cache_disabled: bool = False
    client_connected: bool = True
    # Provider-set top-level fields (system_fingerprint, service_tier) seen on any
    # chunk, forwarded to the caller and stored so cached replays match live streams.
    extra_top: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return "".join(self.content_parts)

    @property
    def reasoning(self) -> str:
        return "".join(self.reasoning_parts)


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
        # Unparseable frame: content teed verbatim is fine, but we can no longer
        # trust the accumulated body, so refuse to cache (fail-closed, #43).
        acc.cache_disabled = True
        return None
    if not isinstance(obj, dict):
        acc.cache_disabled = True
        return None
    if obj.get("error"):
        acc.error = True
        err = obj["error"]
        acc.error_payload = err if isinstance(err, dict) else {"message": str(err)}
        return None
    for f in PASS_THROUGH_TOP_FIELDS:
        if obj.get(f) is not None:
            acc.extra_top[f] = obj[f]
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
    # 3-way delta classification. Classify ALL keys before acting, so a delta
    # carrying both a reasoning key and a category-C key (e.g. refusal) disables
    # caching and does NOT pollute reasoning_parts. Category A (role/content/
    # tool_calls) replays as-is; category B (reasoning) is accumulated + replayed on
    # its own frame; anything else disables caching (fail-closed, #43/#46/#49).
    acc.last_reasoning = None
    if isinstance(delta, dict):
        has_c_key = any(
            k not in _CACHEABLE_DELTA_KEYS and k not in _REASONING_DELTA_KEYS for k in delta
        )
        if has_c_key:
            acc.cache_disabled = True
        else:
            for k in _REASONING_DELTA_KEYS:
                r = delta.get(k)
                if isinstance(r, str) and r:
                    acc.reasoning_parts.append(r)
                    acc.reasoning_key = k
                    acc.last_reasoning = r
                    break
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
    extra = {f: resp[f] for f in PASS_THROUGH_TOP_FIELDS if resp.get(f) is not None}
    yield encode_chunk(role_frame(rec_id, created, model))
    # Reasoning frames precede content, matching the real upstream shape, so a cache
    # HIT replay carries the thinking a live stream would (#46/#49). The stored
    # message uses whichever key the upstream emitted; replay preserves it.
    for rkey in _REASONING_DELTA_KEYS:
        reasoning = message.get(rkey)
        if isinstance(reasoning, str) and reasoning:
            for i in range(0, len(reasoning), 16):
                yield encode_chunk(
                    reasoning_frame(rec_id, created, model, reasoning[i : i + 16], key=rkey)
                )
            break
    for i in range(0, len(body), 16):
        yield encode_chunk(content_frame(rec_id, created, model, body[i : i + 16]))
    yield encode_chunk(finish_frame(rec_id, created, model, finish))
    usage = resp.get("usage")
    if include_usage and isinstance(usage, dict) and usage:
        yield encode_chunk(usage_frame(rec_id, created, model, usage, extra or None))
    yield encode_done()
