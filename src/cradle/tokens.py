from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken

from cradle.gateway.models import ChatMessage

TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
REPLY_PRIMER = 3


@lru_cache(maxsize=16)
def encoding_for_model_name(model: str) -> tiktoken.Encoding:
    name = model.lower()
    if any(s in name for s in ("gpt-4o", "gpt-4.1", "o1", "o3", "o4", "gpt-5")):
        return tiktoken.get_encoding("o200k_base")
    return tiktoken.get_encoding("cl100k_base")


def message_text(m: ChatMessage | Any) -> str:
    content = getattr(m, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def count_chat_prompt(messages: list[ChatMessage] | list[Any], model: str) -> int:
    enc = encoding_for_model_name(model)
    n = 0
    for m in messages:
        n += TOKENS_PER_MESSAGE
        n += len(enc.encode(message_text(m)))
        name = getattr(m, "name", None)
        if name:
            n += TOKENS_PER_NAME
            n += len(enc.encode(name))
    n += REPLY_PRIMER
    return n
