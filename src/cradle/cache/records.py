from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class Principal(BaseModel):
    tenant_id: str
    user_id: str
    key_id: str


class CanonicalMessage(BaseModel):
    role: str
    content: str
    name: str | None = None
    tool_calls: Any | None = None
    tool_call_id: str | None = None
    extras: dict[str, Any] = Field(default_factory=dict)


class CanonicalRequest(BaseModel):
    tenant_id: str
    user_id: str
    model: str
    system_prompt_version: str
    pipeline_version: str
    temperature: float
    top_p: float
    max_tokens: int | None
    max_completion_tokens: int | None
    n: int
    stop: list[str] | str | None
    seed: int | None
    response_format: Any | None
    presence_penalty: float
    frequency_penalty: float
    logit_bias: dict[str, float] | None
    tools: Any | None
    tool_choice: Any | None
    parallel_tool_calls: bool | None
    logprobs: bool | None
    top_logprobs: int | None
    messages: list[CanonicalMessage]
    extras: dict[str, Any]
    sampling_fingerprint: str
    embed_text: str
    stream: bool
    has_tools: bool
    uncacheable_reason: str | None = None


class L2Filter(BaseModel):
    tenant_id: str
    user_id: str
    model: str
    system_prompt_version: str
    pipeline_version: str
    sampling_fingerprint: str
    now_unix: int


class CacheRecord(BaseModel):
    schema_version: int = 1
    key: str
    tenant_id: str
    user_id: str
    model: str
    system_prompt_version: str
    pipeline_version: str
    prompt_hash: str
    embed_text_hash: str
    # Raw role-framed embed text, kept so the L2 precision guard (issue #5) can
    # compare a candidate against the query at serve time. Empty on records
    # written before the guard existed; the guard fails those closed ("no-text").
    embed_text: str = ""
    response: dict[str, Any]
    created_at: int
    expires_at: int
    ttl_s: int
    inbound_prompt_tokens: int
    upstream_prompt_tokens: int
    l2_score: float | None = None
    sampling_fingerprint: str
    temperature: float
    top_p: float
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int = 1
    seed: int | None = None
    stop: Any | None = None
    response_format: Any | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class L2Hit(BaseModel):
    record: CacheRecord
    score: float


class ChatCompletion(TypedDict, total=False):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]
