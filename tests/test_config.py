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
    load_settings()  # raises ConfigError (unknown key) or ValidationError if the example drifts


# --- Friendly config errors for stale keys (#61) -----------------------------


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cradle.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_stale_yaml_key_raises_configerror_naming_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dead key in the YAML file → ConfigError that names the key AND the file."""
    from cradle.config import ConfigError

    cfg = _write_yaml(tmp_path, "compress:\n  min_tokens: 16\n")
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(cfg))
    with pytest.raises(ConfigError) as ei:
        load_settings()
    msg = str(ei.value)
    assert "compress.min_tokens" in msg
    assert str(cfg) in msg  # points at the real file, not a hardcoded default
    assert "remove or rename" in msg


def test_stale_env_key_message_does_not_claim_its_in_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SAME error loc from an env var must NOT tell the operator to edit the
    file (the flaw the guard exists to avoid): the key isn't in the YAML."""
    from cradle.config import ConfigError

    cfg = _write_yaml(tmp_path, "compress:\n  min_savings_ratio: 0.02\n")  # valid file
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(cfg))
    monkeypatch.setenv("CRADLE_COMPRESS__MIN_TOKENS", "16")  # dead key via env
    with pytest.raises(ConfigError) as ei:
        load_settings()
    msg = str(ei.value)
    assert "compress.min_tokens" in msg
    assert f"remove or rename it in {cfg}" not in msg  # must not send them to the file
    assert "environment" in msg


def test_multiple_stale_keys_all_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every extra_forbidden key is surfaced at once (no second round trip)."""
    from cradle.config import ConfigError

    cfg = _write_yaml(
        tmp_path,
        "compress:\n  min_tokens: 16\ncache:\n  evict_old_pipeline: true\n",
    )
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(cfg))
    with pytest.raises(ConfigError) as ei:
        load_settings()
    msg = str(ei.value)
    assert "compress.min_tokens" in msg
    assert "cache.evict_old_pipeline" in msg


def test_stale_key_in_list_element_does_not_falsely_claim_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale key inside a list element (loc carries an int index, e.g.
    routes.0.weight) has no clean env spelling and no flat file key to match, so
    the message must NOT claim it's absent from the file / an env var — it IS in
    the file. Regression for the list-walk gap (#61)."""
    from cradle.config import ConfigError

    cfg = _write_yaml(
        tmp_path,
        "upstreams:\n  b1:\n    base_url: http://x/v1\n"
        "routes:\n  - model: 'gpt*'\n    to: b1\n    weight: 1\n",  # weight is not a RouteRule field
    )
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(cfg))
    with pytest.raises(ConfigError) as ei:
        load_settings()
    msg = str(ei.value)
    assert "routes.0.weight" in msg
    assert "not in" not in msg  # must not claim the in-file key is absent
    assert str(cfg) in msg  # still points at the file as a place to fix it


def test_non_extra_forbidden_error_still_raises_validationerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real validation failure (not a stale key) keeps its ValidationError so its
    message is not swallowed by the friendly path."""
    cfg = _write_yaml(tmp_path, "l2:\n  cosine_threshold: 0.5\n")  # below the 0.85 floor
    monkeypatch.delenv("CRADLE_UPSTREAM_BASE_URL", raising=False)
    monkeypatch.setenv("CRADLE_CONFIG", str(cfg))
    with pytest.raises(ValidationError):
        load_settings()
