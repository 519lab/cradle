from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...

    def ready(self) -> bool: ...
