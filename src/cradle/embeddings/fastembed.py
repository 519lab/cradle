from __future__ import annotations

import os
from pathlib import Path


def resolve_cache_dir(data_dir: Path) -> Path:
    env = os.environ.get("FASTEMBED_CACHE_PATH", "").strip()
    if env:
        return Path(env)
    return data_dir / "models" / "fastembed"


class FastEmbedEmbedder:
    dim = 384

    def __init__(
        self,
        cache_dir: Path,
        model_name: str = "BAAI/bge-small-en-v1.5",
        threads: int | None = None,
        dim: int = 384,
    ) -> None:
        from fastembed import TextEmbedding

        self.dim = dim
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir),
            threads=threads,
            lazy_load=False,
            providers=["CPUExecutionProvider"],
        )
        self._ready = False

    def embed(self, text: str) -> list[float]:
        vec = next(self._model.embed([text]))
        out = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        if len(out) != self.dim:
            raise RuntimeError(f"embed dim {len(out)} != {self.dim}")
        self._ready = True
        return out

    def ready(self) -> bool:
        return self._ready
