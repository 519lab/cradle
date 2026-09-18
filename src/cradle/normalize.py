from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any

from cradle.cache.records import CanonicalMessage, CanonicalRequest, Principal
from cradle.compress.guards import extract_protected, restore_protected
from cradle.config import Settings
from cradle.gateway.models import ChatMessage, ChatRequest

# Bumped to 2: cache identity now includes the resolved backend namespace and
# excludes routing-only hints (see cache_namespace + _ROUTING_HINTS). Old L1/L2
# entries written under schema 1 miss safely and age out on TTL.
HASH_SCHEMA_VERSION = 2
_KNOWN_REQUEST = set(ChatRequest.model_fields)
_KNOWN_MESSAGE = set(ChatMessage.model_fields)

# Undeclared request fields that steer PROVIDER-side caching/routing/abuse
# monitoring but do not change the generated answer. They must not enter the
# cache key, or two identical prompts differing only by a hint split into
# separate L1 entries (bug B). Same rationale as excluding `stream`. The
# declared `user` field is already dropped (never carried into the canonical
# form), so it is not listed here.
_ROUTING_HINTS = frozenset({
    "prompt_cache_key",
    "prompt_cache_retention",
    "safety_identifier",
})


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _sort_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return json.loads(_stable_json(value))
    return value


def _empty_to_none(value: Any) -> Any:
    if value in (None, [], {}):
        return None
    return value


def _collapse_ws(text: str) -> str:
    return " ".join(text.split()).strip()


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _canonicalize_text(text: str) -> str:
    nfkc = _nfkc(text)
    placeholders, masked, _spans = extract_protected(nfkc)
    collapsed = _collapse_ws(masked)
    return restore_protected(collapsed, placeholders)


def _content_text(content: str | list[Any] | None) -> tuple[str, bool]:
    if content is None:
        return "", False
    if isinstance(content, str):
        return content, False
    parts: list[str] = []
    non_text = False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(str(part.get("text") or ""))
        elif isinstance(part, str):
            parts.append(part)
        else:
            non_text = True
    return "\n".join(parts), non_text


def hash_input(c: CanonicalRequest) -> dict[str, Any]:
    return {
        "v": HASH_SCHEMA_VERSION,
        "tenant_id": c.tenant_id,
        "user_id": c.user_id,
        "model": c.model,
        "backend_namespace": c.backend_namespace,
        "system_prompt_version": c.system_prompt_version,
        "pipeline_version": c.pipeline_version,
        "temperature": c.temperature,
        "top_p": c.top_p,
        "max_tokens": c.max_tokens,
        "max_completion_tokens": c.max_completion_tokens,
        "n": c.n,
        "stop": c.stop,
        "seed": c.seed,
        "response_format": c.response_format,
        "presence_penalty": c.presence_penalty,
        "frequency_penalty": c.frequency_penalty,
        "logit_bias": c.logit_bias,
        "tools": c.tools,
        "tool_choice": c.tool_choice,
        "parallel_tool_calls": c.parallel_tool_calls,
        "logprobs": c.logprobs,
        "top_logprobs": c.top_logprobs,
        "messages": [m.model_dump() for m in c.messages],
        "extras": c.extras,
    }


def l1_key(canonical: CanonicalRequest) -> str:
    return hashlib.sha256(_stable_json(hash_input(canonical)).encode("utf-8")).hexdigest()


def sampling_fingerprint(c: CanonicalRequest) -> str:
    payload = hash_input(c)
    payload.pop("tenant_id", None)
    payload.pop("user_id", None)
    payload.pop("messages", None)
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _round6(v: float) -> float:
    return round(float(v), 6)


def _system_prompt_version(messages: list[CanonicalMessage]) -> str:
    parts = [m.content for m in messages if m.role in {"system", "developer"}]
    if not parts:
        return "none"
    blob = _nfkc("\n\n".join(parts))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def has_non_text_parts(messages: list[ChatMessage]) -> bool:
    for m in messages:
        if isinstance(m.content, list):
            for part in m.content:
                if isinstance(part, dict) and part.get("type") not in (None, "text"):
                    return True
                if not isinstance(part, (dict, str)):
                    return True
    return False


def cache_namespace(target_name: str, base_url: str) -> str:
    """Stable, non-secret namespace for the resolved backend (bug A).

    Two named backends serving the same model glob (or a `routes:` change that
    repoints a glob) must not share cache entries, or Cradle replays the old
    backend's answers. Hash the route target name + normalized base URL.
    """
    norm = base_url.rstrip("/").lower()
    blob = f"{target_name}|{norm}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def canonicalize(
    req: ChatRequest,
    principal: Principal,
    settings: Settings,
    backend_namespace: str = "",
) -> CanonicalRequest:
    uncacheable: str | None = None
    messages: list[CanonicalMessage] = []
    for m in req.messages:
        text, non_text = _content_text(m.content)
        if non_text:
            uncacheable = "non_text_part"
        extras = {k: _sort_json(v) for k, v in (m.model_extra or {}).items() if k not in _KNOWN_MESSAGE}
        tool_calls = _empty_to_none(_sort_json(m.tool_calls))
        tool_call_id = m.tool_call_id
        content = _canonicalize_text(text) if text else ""
        if not content and not tool_calls and not tool_call_id:
            continue
        messages.append(
            CanonicalMessage(
                role=m.role,
                content=content,
                name=m.name,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
                extras=extras,
            )
        )

    extras = {
        k: _sort_json(v)
        for k, v in (req.model_extra or {}).items()
        if k not in _KNOWN_REQUEST and k not in _ROUTING_HINTS
    }
    tools = _empty_to_none(_sort_json(req.tools))
    tool_choice = _empty_to_none(_sort_json(req.tool_choice))
    logit_bias = _empty_to_none(_sort_json(req.logit_bias))
    has_tools = tools is not None
    # embed_text drives L2 response matching. Exclude system/developer turns: they
    # are (a) already folded into the cache identity via system_prompt_version and
    # the L1 key, and (b) frequently a large fixed block (open-webui/RAG/agent
    # frames) that dominates the embedding and collides unrelated user questions at
    # high cosine — the #40 wrong-hit. Match on the discriminating user/assistant
    # content only. (The system prompt still gates correctness and saves tokens; it
    # just no longer pollutes similarity.)
    embed_text = "".join(
        f"{m.role}: {m.content}\n" for m in messages if m.role not in {"system", "developer"}
    )
    c = CanonicalRequest(
        tenant_id=principal.tenant_id,
        user_id=principal.user_id,
        model=req.model,
        backend_namespace=backend_namespace,
        system_prompt_version="",
        pipeline_version=settings.pipeline_version,
        temperature=_round6(req.temperature),
        top_p=_round6(req.top_p),
        max_tokens=req.max_tokens,
        max_completion_tokens=req.max_completion_tokens,
        n=req.n,
        stop=req.stop,
        seed=req.seed,
        response_format=_sort_json(req.response_format),
        presence_penalty=_round6(req.presence_penalty),
        frequency_penalty=_round6(req.frequency_penalty),
        logit_bias=logit_bias,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=req.parallel_tool_calls,
        logprobs=req.logprobs,
        top_logprobs=req.top_logprobs,
        messages=messages,
        extras=extras,
        sampling_fingerprint="",
        embed_text=embed_text,
        stream=req.stream,
        has_tools=has_tools,
        uncacheable_reason=uncacheable,
    )
    c.system_prompt_version = _system_prompt_version(messages)
    c.sampling_fingerprint = sampling_fingerprint(c)
    return c


def is_cacheable(canonical: CanonicalRequest, req: ChatRequest, settings: Settings) -> bool:
    if not settings.features.cache:
        return False
    if canonical.uncacheable_reason:
        return False
    if canonical.n != 1:
        return False
    if has_non_text_parts(req.messages):
        return False
    if req.stream and req.logprobs:
        # Cached logprobs would be wrong on replay — always bypass.
        return False
    if req.stream and canonical.has_tools and not settings.cache.cache_tool_streams:
        # Tool-enabled streams bypass unless the passthrough-cache path is enabled
        # (#43). When enabled they are cacheable: teed verbatim, cached only if the
        # response carries no tool call. The pipeline routes them via
        # ctx.cacheable_passthrough_stream, not the tool-call-incapable wrap path.
        return False
    return True


def l2_eligible(canonical: CanonicalRequest, req: ChatRequest, settings: Settings) -> bool:
    if not is_cacheable(canonical, req, settings):
        return False
    if not settings.features.l2:
        return False
    if canonical.has_tools:
        return False
    if len(canonical.messages) > settings.l2.max_messages:
        return False
    if canonical.temperature > settings.cache.max_temperature:
        return False
    # embed_text now excludes system/developer turns (#40); a request that carries
    # only a system prompt (no user/assistant content) yields an empty embed_text,
    # which must never be embedded — an empty vector matches anything.
    if not canonical.embed_text.strip():
        return False
    return True
