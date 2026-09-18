from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from cradle.config import L2Settings, load_settings


def test_yaml_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(Path("config/cradle.yaml")))
    s = load_settings()
    assert s.server.port == 8000
    assert s.upstream.base_url.endswith(":8080/v1")
    assert s.features.structure is False
    assert s.pipeline_version == "v3"
    assert s.auth.keys == []
    assert s.upstream.pass_through_client_auth is True


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRADLE_CONFIG", str(Path("config/cradle.yaml")))
    monkeypatch.setenv("CRADLE_SERVER__PORT", "9001")
    monkeypatch.setenv("CRADLE_PIPELINE_VERSION", "vtest")
    s = load_settings()
    assert s.server.port == 9001
    assert s.pipeline_version == "vtest"


def test_flat_upstream_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRADLE_CONFIG", str(Path("config/cradle.yaml")))
    monkeypatch.setenv("CRADLE_UPSTREAM_BASE_URL", "http://127.0.0.1:9999/v1/")
    s = load_settings()
    assert s.upstream.base_url == "http://127.0.0.1:9999/v1"


def test_cosine_floor() -> None:
    with pytest.raises(ValidationError):
        L2Settings(cosine_threshold=0.84)


def test_rerank_defaults() -> None:
    s = L2Settings()
    assert s.rerank_model == "BAAI/bge-reranker-base"
    assert s.rerank_threshold == 4.0
    assert s.rerank_device == "cpu"
    assert s.rerank_device_ids is None


def test_rerank_device_cuda_ok() -> None:
    s = L2Settings(rerank_device="cuda", rerank_device_ids=[0])
    assert s.rerank_device == "cuda"
    assert s.rerank_device_ids == [0]


def test_rerank_device_rejects_garbage() -> None:
    with pytest.raises(ValidationError):
        L2Settings(rerank_device="metal")


def test_example_config_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tracked config/cradle.yaml.example must load through the real Settings.

    Several sub-configs use extra="forbid", so a key removed from the code (e.g.
    #27's dead cache.evict_old_pipeline) but left in the example is a hard load
    failure — the exact drift #36 wants caught in review, not on a box at boot.
    The example is what operators copy to the (gitignored, bind-mounted) runtime
    config, so it must always be a valid config for the current code.
    """
    example = Path(__file__).resolve().parents[1] / "config" / "cradle.yaml.example"
    assert example.is_file(), f"tracked example config missing at {example}"
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(example))
    load_settings()  # raises ValidationError if the example drifts from the schema
