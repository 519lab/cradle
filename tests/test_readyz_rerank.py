"""/readyz must report not-ready when rerank is enabled but the reranker failed
to load — otherwise a container serves L2 with the #5 protection silently off.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from cradle.app import create_app
from cradle.config import AuthSettings, FeatureFlags, L2Settings, Settings, UpstreamSettings
from cradle.embeddings.fake import FakeEmbedder


class BoomReranker:
    def score(self, query: str, candidate: str) -> float:
        raise RuntimeError("model failed to load")


class OkReranker:
    def score(self, query: str, candidate: str) -> float:
        return 100.0


def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, l2_rerank: bool,
            reranker: object) -> Iterator[TestClient]:
    monkeypatch.delenv("CRADLE_API_KEY", raising=False)
    settings = Settings(
        data_dir=tmp_path / "data",
        auth=AuthSettings(keys=[]),
        features=FeatureFlags(
            cache=True, compression=False, structure=False,
            l2=True, l2_rerank=l2_rerank, reconstruction=False, local_1b=False,
        ),
        l2=L2Settings(mode="local"),
        upstream=UpstreamSettings(base_url="http://upstream/v1"),
    )
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        base_url="http://upstream",
    )
    app = create_app(settings=settings, embedder=FakeEmbedder(), http=http, reranker=reranker)
    with TestClient(app) as c:
        yield c


def test_readyz_not_ready_when_reranker_load_fails(tmp_path, monkeypatch) -> None:
    for c in _client(tmp_path, monkeypatch, l2_rerank=True, reranker=BoomReranker()):
        r = c.get("/readyz")
        assert r.status_code == 503
        assert r.json()["status"] == "not_ready"


def test_readyz_ready_when_reranker_loads(tmp_path, monkeypatch) -> None:
    for c in _client(tmp_path, monkeypatch, l2_rerank=True, reranker=OkReranker()):
        r = c.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"


def test_readyz_ignores_reranker_when_disabled(tmp_path, monkeypatch) -> None:
    """rerank off: a broken reranker is irrelevant; still ready."""
    for c in _client(tmp_path, monkeypatch, l2_rerank=False, reranker=BoomReranker()):
        r = c.get("/readyz")
        assert r.status_code == 200
