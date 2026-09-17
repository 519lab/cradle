from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import (
    AuthKey,
    AuthSettings,
    FeatureFlags,
    L2Settings,
    Settings,
    UpstreamSettings,
)
from cradle.embeddings.fake import FakeEmbedder
from tests.fake_rerank import AllowReranker
from tests.fake_upstream import fake_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api_key(monkeypatch: pytest.MonkeyPatch) -> str:
    key = "test-key-aaaaaaaa"
    monkeypatch.setenv("CRADLE_API_KEY", key)
    for name in list(os.environ):
        if name.startswith("CRADLE_") and name not in {"CRADLE_API_KEY"}:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(ROOT / "config" / "cradle.yaml"))
    return key


@pytest.fixture
def settings(tmp_path: Path, api_key: str) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(
            keys=[AuthKey(token_env="CRADLE_API_KEY", tenant_id="t1", user_id="u1")]
        ),
        features=FeatureFlags(
            cache=True,
            compression=True,
            structure=False,
            l2=True,
            reconstruction=True,
            local_1b=False,
        ),
        l2=L2Settings(mode="local", cosine_threshold=0.90),
        upstream=UpstreamSettings(base_url="http://upstream/v1", models_passthrough=False),
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    transport = httpx.ASGITransport(app=fake_app)
    http = httpx.AsyncClient(transport=transport, base_url="http://upstream")
    app = create_app(
        settings=settings, embedder=FakeEmbedder(), http=http, reranker=AllowReranker()
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_header(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}
