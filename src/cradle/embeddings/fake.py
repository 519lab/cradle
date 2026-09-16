from __future__ import annotations

import hashlib
import math


class FakeEmbedder:
    dim = 384

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = (digest * (self.dim // len(digest) + 1))[: self.dim]
        vec = [(b / 127.5) - 1.0 for b in raw]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def ready(self) -> bool:
        return True
