from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cradle.compress.engine import compress
from cradle.config import Settings
from cradle.gateway.models import ChatMessage
from cradle.tokens import count_chat_prompt


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def savings_for_row(row: dict[str, Any], settings: Settings) -> float:
    messages = [ChatMessage.model_validate(m) for m in row["messages"]]
    model = row.get("model") or "gpt-4o-mini"
    result = compress(messages, settings, model)
    inbound = count_chat_prompt(messages, model)
    if inbound <= 0:
        return 0.0
    return (inbound - result.compressed_tokens) / inbound
