from __future__ import annotations

from fastapi.testclient import TestClient


def test_l1_hit_after_miss(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "unique-l1-prompt-xyz"}],
    }
    a = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert a.status_code == 200
    assert a.headers["X-Cradle-Cache"] in {"MISS", "HIT-L2"}
    b = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert b.status_code == 200
    assert b.headers["X-Cradle-Cache"] == "HIT-L1"
    assert a.json()["choices"][0]["message"]["content"] == b.json()["choices"][0]["message"]["content"]


def test_stream_tools_bypass(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "stream": True,
        "tools": [{"type": "function", "function": {"name": "x", "parameters": {}}}],
        "messages": [{"role": "user", "content": "call x"}],
    }
    r = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] == "BYPASS"
    assert b"data: [DONE]\n\n" in r.content


def test_json_and_stream_same_stored_body(client: TestClient, auth_header: dict[str, str]) -> None:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "identical-canonical-body-1"}],
    }
    j = client.post("/v1/chat/completions", headers=auth_header, json=payload)
    assert j.headers["X-Cradle-Cache"] in {"MISS", "HIT-L1", "HIT-L2"}
    stored = j.json()["choices"][0]["message"]["content"]
    s = client.post("/v1/chat/completions", headers=auth_header, json={**payload, "stream": True})
    assert s.status_code == 200
    assert s.headers["X-Cradle-Cache"] == "HIT-L1"
    assert b"data: [DONE]\n\n" in s.content
    # replay concatenates stored body
    assert stored == "ACK"
    assert "ACK" in s.text
