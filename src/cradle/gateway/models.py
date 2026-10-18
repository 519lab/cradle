from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str | list[Any] | None = None
    name: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    n: int = 1
    stop: str | list[str] | None = None
    seed: int | None = None
    response_format: Any | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[Any] | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    user: str | None = None


class ReconstructionTemplate(BaseModel):
    version: str = "v1"
    mode: str = "wrap"
    brand_prefix: str = ""
    brand_suffix: str = ""
    format_instructions: list[str] = Field(default_factory=list)


class CompressedPrompt(BaseModel):
    messages: list[ChatMessage]
    template: ReconstructionTemplate
    inbound_tokens: int
    compressed_tokens: int
    savings_ratio: float
    protected_span_count: int
