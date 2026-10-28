from __future__ import annotations

from fastapi.testclient import TestClient

from cradle.compress.engine import compress
from cradle.config import FeatureFlags, Settings
from cradle.gateway.models import ChatMessage

ORACLE = (
    "Please just really process this.\n"
    "```python\nprint('secret-block')\n```\n"
    "respond only in JSON\n"
)


def test_guards_keep_code_and_format() -> None:
    s = Settings(features=FeatureFlags(compression=True))
    out = compress([ChatMessage(role="user", content=ORACLE)], s, "gpt-4o-mini")
    text = out.messages[0].content or ""
    assert "print('secret-block')" in text
    assert "respond only in JSON" in text.lower() or "respond only in JSON" in text


def test_client_keeps_format_instruction(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": ORACLE}]}
    r = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert r.status_code == 200
    body = r.json()["choices"][0]["message"]["content"]
    assert "respond only in JSON" in body
    s = client.post("/v1/chat/completions", headers=auth_header, json={**payload, "stream": True})
    assert s.headers["X-Cradle-Cache"] == "HIT-L1"
    import json as jsonlib

    pieces = []
    for line in s.text.splitlines():
        if line.startswith("data: ") and line[6:] != "[DONE]":
            obj = jsonlib.loads(line[6:])
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if isinstance(delta.get("content"), str):
                pieces.append(delta["content"])
    assert "respond only in JSON" in "".join(pieces)
