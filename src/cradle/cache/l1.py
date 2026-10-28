from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from diskcache import Cache

from cradle.cache.records import CacheRecord

if TYPE_CHECKING:
    from cradle.config import Settings


def open_l1(settings: Settings) -> Cache:
    directory = settings.data_dir / "l1"
    directory.mkdir(parents=True, exist_ok=True)
    return Cache(
        directory=str(directory),
        size_limit=settings.cache.l1_size_limit_bytes,
        cull_limit=10,
        eviction_policy="least-recently-stored",
        tag_index=True,
        sqlite_journal_mode="wal",
    )


def get_sync(cache: Cache, key: str) -> CacheRecord | None:
    raw = cache.get(key, default=None, retry=True)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return CacheRecord.model_validate_json(raw)
    if isinstance(raw, str):
        return CacheRecord.model_validate_json(raw.encode("utf-8"))
    return CacheRecord.model_validate(raw)


async def get(cache: Cache, key: str) -> CacheRecord | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_sync, cache, key)


def set_sync(cache: Cache, key: str, record: CacheRecord, ttl_s: int, tag: str) -> None:
    cache.set(
        key,
        record.model_dump_json().encode("utf-8"),
        expire=ttl_s,
        tag=tag,
        retry=True,
    )


async def set(cache: Cache, key: str, record: CacheRecord, ttl_s: int, tag: str) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, set_sync, cache, key, record, ttl_s, tag)


def evict_tag_sync(cache: Cache, tag: str) -> int:
    return int(cache.evict(tag))


async def evict_tag(cache: Cache, tag: str) -> int:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, evict_tag_sync, cache, tag)
