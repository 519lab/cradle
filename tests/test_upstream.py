from __future__ import annotations

from fastapi.testclient import TestClient


def test_chat_passthrough(client: TestClient, auth_header: dict[str, str]) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "ping"}]},
    )
    assert r.status_code == 200
    assert r.headers["X-Cradle-Cache"] in {"MISS", "HIT-L1", "HIT-L2"}
    assert r.json()["choices"][0]["message"]["content"] == "ACK"


def test_developer_role_not_422(client: TestClient, auth_header: dict[str, str]) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert r.status_code == 200


def test_unknown_field_forwarded(client: TestClient, auth_header: dict[str, str]) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers=auth_header,
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "foo_extra": 1,
        },
    )
    assert r.status_code == 200
