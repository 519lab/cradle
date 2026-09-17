from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx
from diskcache import Cache
from qdrant_client import QdrantClient

from cradle.cache.records import Principal
from cradle.cache.rerank import Reranker
from cradle.config import Settings
from cradle.embeddings.base import Embedder


@dataclass
class Runtime:
    settings: Settings
    http: httpx.AsyncClient
    embed_pool: ThreadPoolExecutor
    principals: list[tuple[str, Principal]] = field(default_factory=list)
    l1: Cache | None = None
    qdrant: QdrantClient | None = None
    embedder: Embedder | None = None
    reranker: Reranker | None = None
    l1_ready: bool = False
    l2_ready: bool = False
    embedder_ready: bool = False
    reranker_ready: bool = False
