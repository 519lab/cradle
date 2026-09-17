from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, Range

from cradle.cache.records import CacheRecord, L2Filter, L2Hit
from cradle.config import Settings

log = logging.getLogger("cradle.l2")


def open_l2(settings: Settings) -> QdrantClient:
    if settings.l2.mode == "server":
        raise NotImplementedError(
            "l2.mode=server is reserved; v1 uses Qdrant local. Stay on l2.mode=local."
        )
    path = settings.data_dir / "l2"
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    names = [c.name for c in client.get_collections().collections]
    if settings.l2.collection not in names:
        client.create_collection(
            collection_name=settings.l2.collection,
            vectors_config=VectorParams(size=settings.l2.dim, distance=Distance.COSINE),
        )
    return client


def _must_filter(filt: L2Filter) -> Filter:
    return Filter(
        must=[
            FieldCondition(key="tenant_id", match=MatchValue(value=filt.tenant_id)),
            FieldCondition(key="user_id", match=MatchValue(value=filt.user_id)),
            FieldCondition(key="model", match=MatchValue(value=filt.model)),
            FieldCondition(
                key="backend_namespace", match=MatchValue(value=filt.backend_namespace)
            ),
            FieldCondition(
                key="system_prompt_version", match=MatchValue(value=filt.system_prompt_version)
            ),
            FieldCondition(key="pipeline_version", match=MatchValue(value=filt.pipeline_version)),
            FieldCondition(
                key="sampling_fingerprint", match=MatchValue(value=filt.sampling_fingerprint)
            ),
            FieldCondition(key="expires_at", range=Range(gte=filt.now_unix)),
        ]
    )


def query_sync(
    client: QdrantClient, settings: Settings, vec: list[float], filt: L2Filter
) -> list[L2Hit]:
    """Return up to `query_top_k` candidates above the cosine floor, best-first."""
    k = max(1, settings.l2.query_top_k)
    result = client.query_points(
        collection_name=settings.l2.collection,
        query=vec,
        query_filter=_must_filter(filt),
        limit=k,
        with_payload=True,
    )
    hits: list[L2Hit] = []
    for point in result.points:
        score = float(point.score)
        if score < settings.l2.cosine_threshold:
            break  # points are score-desc; once below the floor, so is the rest
        payload: dict[str, Any] = dict(point.payload or {})
        record = CacheRecord.model_validate(payload)
        hits.append(L2Hit(record=record, score=score))
    return hits


async def query(
    client: QdrantClient, settings: Settings, vec: list[float], filt: L2Filter
) -> list[L2Hit]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, query_sync, client, settings, vec, filt)


def upsert_sync(client: QdrantClient, settings: Settings, vec: list[float], record: CacheRecord) -> None:
    pid = str(uuid.uuid5(uuid.NAMESPACE_URL, record.key))
    client.upsert(
        collection_name=settings.l2.collection,
        points=[
            PointStruct(
                id=pid,
                vector=vec,
                payload=record.model_dump(mode="json"),
            )
        ],
    )


async def upsert(
    client: QdrantClient, settings: Settings, vec: list[float], record: CacheRecord
) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, upsert_sync, client, settings, vec, record)


def purge_expired_sync(client: QdrantClient, settings: Settings, now: int | None = None) -> int:
    now = now or int(time.time())
    # Local mode: scroll and delete expired.
    expired_ids: list[str] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=settings.l2.collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for rec in records:
            payload = rec.payload or {}
            if int(payload.get("expires_at") or 0) < now:
                expired_ids.append(rec.id)
        if offset is None:
            break
    if expired_ids:
        client.delete(collection_name=settings.l2.collection, points_selector=expired_ids)
    return len(expired_ids)


async def purge_expired(client: QdrantClient, settings: Settings, now: int | None = None) -> int:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, purge_expired_sync, client, settings, now)


def count_points_sync(client: QdrantClient, settings: Settings) -> int:
    info = client.get_collection(settings.l2.collection)
    return int(info.points_count or 0)


async def count_points(client: QdrantClient, settings: Settings) -> int:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, count_points_sync, client, settings)
