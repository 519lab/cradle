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
    pid = point_id(record.key)
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


def point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def record_audit_sync(
    client: QdrantClient,
    settings: Settings,
    key: str,
    *,
    query_similarity: float,
    agree: bool,
) -> CacheRecord | None:
    """Fold one audit observation into the stored point's payload.

    A disagreement raises the entry's ``audit_floor`` to the query similarity
    (never lowers it), so the entry refuses future matches at or below the
    similarity that already produced a wrong answer. Returns the updated
    record, or None if the point no longer exists (expired/purged).
    """
    pid = point_id(key)
    found = client.retrieve(
        collection_name=settings.l2.collection, ids=[pid], with_payload=True, with_vectors=False
    )
    if not found:
        return None
    record = CacheRecord.model_validate(dict(found[0].payload or {}))
    if agree:
        record.audit_agree += 1
    else:
        record.audit_disagree += 1
        floor = record.audit_floor or 0.0
        record.audit_floor = max(floor, query_similarity)
    client.set_payload(
        collection_name=settings.l2.collection,
        payload={
            "audit_floor": record.audit_floor,
            "audit_agree": record.audit_agree,
            "audit_disagree": record.audit_disagree,
        },
        points=[pid],
    )
    return record


async def record_audit(
    client: QdrantClient,
    settings: Settings,
    key: str,
    *,
    query_similarity: float,
    agree: bool,
) -> CacheRecord | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: record_audit_sync(
            client, settings, key, query_similarity=query_similarity, agree=agree
        ),
    )


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
