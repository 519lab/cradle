from __future__ import annotations

from fastapi.testclient import TestClient


def test_missing_key(client: TestClient) -> None:
    r = client.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"


def test_bad_key(client: TestClient) -> None:
    r = client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert r.status_code == 401


def test_models_requires_auth(client: TestClient, auth_header: dict[str, str]) -> None:
    assert client.get("/v1/models").status_code == 401
    r = client.get("/v1/models", headers=auth_header)
    assert r.status_code == 200
    assert r.json()["object"] == "list"


def test_metrics_requires_auth(client: TestClient, auth_header: dict[str, str]) -> None:
    assert client.get("/metrics").status_code == 401
    r = client.get("/metrics", headers=auth_header)
    assert r.status_code == 200
    assert b"cradle_" in r.content


def test_invalid_json_is_400(client: TestClient, auth_header: dict[str, str]) -> None:
    r = client.post(
        "/v1/chat/completions",
        headers={**auth_header, "content-type": "application/json"},
        content=b"{not-json",
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"
