from __future__ import annotations

from pathlib import Path

from cradle.embeddings.fake import FakeEmbedder
from cradle.embeddings.fastembed import resolve_cache_dir


def test_fake_dim_and_stable() -> None:
    e = FakeEmbedder()
    a = e.embed("hello")
    b = e.embed("hello")
    c = e.embed("world")
    assert len(a) == 384
    assert a == b
    assert a != c
    assert e.ready()


def test_resolve_cache_dir_defaults_under_data(tmp_path: Path) -> None:
    assert resolve_cache_dir(tmp_path) == tmp_path / "models" / "fastembed"


def test_resolve_cache_dir_env_overrides_volume(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/opt/cradle/models/fastembed")
    assert resolve_cache_dir(tmp_path) == Path("/opt/cradle/models/fastembed")
