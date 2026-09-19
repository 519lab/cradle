from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx
from diskcache import Cache
from qdrant_client import QdrantClient

from cradle.cache.records import Principal
from cradle.cache.rerank import Reranker
from cradle.config import Settings
from cradle.embeddings.base import Embedder

if TYPE_CHECKING:
    from cradle.gateway.flight import Flight


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
    # In-flight background L2 audits (gateway/audit.py). Tracked so shutdown can
    # drain them and tests can wait for them; a task removes itself when done.
    audit_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    # In-flight single-flight leaders (gateway/flight.py), keyed by the L1 key +
    # stream-ness. Concurrent identical misses coalesce onto the leader's entry;
    # the leader removes it in a finally. Request-scoped, not background tasks, so
    # no drain method — each resolves when its leader's request ends.
    flights: dict[str, Flight] = field(default_factory=dict)

    async def drain_audits(self, timeout_s: float = 30.0) -> None:
        if self.audit_tasks:
            await asyncio.wait(set(self.audit_tasks), timeout=timeout_s)
