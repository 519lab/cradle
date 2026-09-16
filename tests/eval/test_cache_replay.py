from __future__ import annotations

from fastapi.testclient import TestClient


def test_miss_then_l1(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "replay-cache-1"}]}
    a = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    b = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert a.headers["X-Cradle-Cache"] != "HIT-L1"
    assert b.headers["X-Cradle-Cache"] == "HIT-L1"


def test_stream_hit_done(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "replay-stream-1"}]}
    client.post("/v1/chat/completions", headers=auth_header, json=payload)
    s = client.post("/v1/chat/completions", headers=auth_header, json={**payload, "stream": True})
    assert s.headers["X-Cradle-Cache"] == "HIT-L1"
    assert s.content.endswith(b"data: [DONE]\n\n") or b"data: [DONE]\n\n" in s.content
