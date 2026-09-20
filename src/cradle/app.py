from __future__ import annotations

import asyncio
import logging
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI

from cradle.cache import l1 as l1mod
from cradle.cache import l2 as l2mod
from cradle.config import Settings, load_settings
from cradle.embeddings.base import Embedder
from cradle.gateway.flight import reap_stale_flights
from cradle.gateway.routes import router
from cradle.metrics import prometheus as m
from cradle.runtime import Runtime
from cradle.tenancy import assert_auth_ready, load_principals

log = logging.getLogger("cradle")


def _reject_multi_worker(settings: Settings) -> None:
    if not settings.features.l2:
        return
    for var in ("WEB_CONCURRENCY", "UVICORN_WORKERS"):
        raw = os.environ.get(var, "").strip()
        if raw and raw != "1":
            raise RuntimeError(
                f"{var}={raw} is incompatible with Qdrant local L2; v1 requires a single worker"
            )


def _ensure_data_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, stat.S_IRWXU)
    except PermissionError:
        if not os.access(path, os.W_OK | os.X_OK):
            raise


def _build_embedder(settings: Settings, override: Embedder | None) -> Embedder | None:
    if override is not None:
        return override
    if not settings.features.l2:
        return None
    from cradle.embeddings.fastembed import FastEmbedEmbedder, resolve_cache_dir

    cache_dir = resolve_cache_dir(settings.data_dir)
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)
    return FastEmbedEmbedder(
        cache_dir=cache_dir,
        model_name=settings.l2.model,
        threads=settings.l2.onnx_threads,
        dim=settings.l2.dim,
    )


def _build_reranker(settings: Settings, override: object | None):
    if override is not None:
        return override
    if not (settings.features.l2 and settings.features.l2_rerank):
        return None
    from cradle.cache.rerank import FastEmbedReranker
    from cradle.embeddings.fastembed import resolve_cache_dir

    cache_dir = resolve_cache_dir(settings.data_dir)
    return FastEmbedReranker(
        model_name=settings.l2.rerank_model,
        cache_dir=str(cache_dir),
        cuda=settings.l2.rerank_device == "cuda",
        device_ids=settings.l2.rerank_device_ids,
    )


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    http: httpx.AsyncClient | None = None,
    reranker: object | None = None,
) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.features.local_1b:
            try:
                import llama_cpp  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("features.local_1b requires extra cradle[local-1b]") from exc
        _ensure_data_dir(settings.data_dir)
        principals = load_principals(settings)
        assert_auth_ready(settings, principals)

        l1 = None
        l1_ready = False
        if settings.features.cache:
            l1 = l1mod.open_l1(settings)
            l1_ready = True
            m.ready_gauge.labels(component="l1").set(1)

        qdrant = None
        l2_ready = False
        if settings.features.l2:
            qdrant = l2mod.open_l2(settings)
            l2_ready = True
            m.ready_gauge.labels(component="l2").set(1)

        resolved_embedder = _build_embedder(settings, embedder)
        embedder_ready = False
        if resolved_embedder is not None:
            resolved_embedder.embed("ok")
            embedder_ready = resolved_embedder.ready()
            m.ready_gauge.labels(component="embedder").set(1 if embedder_ready else 0)

        _reject_multi_worker(settings)

        # After the cheap startup checks so a misconfigured worker count fails
        # fast without paying the reranker model load.
        resolved_reranker = _build_reranker(settings, reranker)
        reranker_ready = False
        if resolved_reranker is not None:
            # Warm the model once so the first request does not pay init cost. A
            # warm-up failure must not crash startup, but it does mark the
            # reranker not-ready so /readyz reports the degraded state instead of
            # silently serving L2 with the #5 protection off.
            try:
                resolved_reranker.score("ok", "ok")
                reranker_ready = True
                m.ready_gauge.labels(component="reranker").set(1)
            except Exception:
                log.exception("reranker warm-up failed; L2 rerank is degraded")
                m.ready_gauge.labels(component="reranker").set(0)

        embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cradle-embed")
        client = http or httpx.AsyncClient(timeout=settings.upstream.timeout_s)
        runtime = Runtime(
            settings=settings,
            http=client,
            embed_pool=embed_pool,
            principals=principals,
            l1=l1,
            qdrant=qdrant,
            embedder=resolved_embedder,
            reranker=resolved_reranker,
            l1_ready=l1_ready,
            l2_ready=l2_ready,
            embedder_ready=embedder_ready,
            reranker_ready=reranker_ready,
        )
        app.state.runtime = runtime

        stop = asyncio.Event()

        async def purge_loop() -> None:
            while not stop.is_set():
                try:
                    if runtime.l1 is not None:
                        runtime.l1.expire()
                    if runtime.qdrant is not None:
                        await l2mod.purge_expired(runtime.qdrant, settings)
                        n = await l2mod.count_points(runtime.qdrant, settings)
                        m.l2_points.set(n)
                        if n > settings.l2.points_warn:
                            log.warning(
                                "Qdrant local collection has %s points (threshold %s); "
                                "scale path is l2.mode=server later",
                                n,
                                settings.l2.points_warn,
                            )
                except Exception:
                    log.exception("purge failed")
                try:
                    await asyncio.wait_for(stop.wait(), timeout=settings.cache.purge_interval_s)
                except TimeoutError:
                    pass

        task = asyncio.create_task(purge_loop())

        # Single-flight reaper (#67): sweep runtime.flights for leaked leaders (a
        # flight left unresolved because Starlette cancelled a stream response before
        # its _wrap_stream generator was ever iterated, so its finally never ran).
        # Only meaningful when singleflight is on; otherwise the registry stays empty,
        # so skip the wakeup loop entirely on the default deployment.
        reaper_task: asyncio.Task[None] | None = None
        if settings.cache.singleflight:
            timeout_s = settings.upstream.timeout_s

            async def reaper_loop() -> None:
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=timeout_s)
                    except TimeoutError:
                        pass
                    if stop.is_set():
                        return
                    try:
                        reap_stale_flights(runtime, timeout_s)
                    except Exception:
                        log.exception("flight reaper sweep failed")

            reaper_task = asyncio.create_task(reaper_loop())

        yield
        stop.set()
        task.cancel()
        if reaper_task is not None:
            reaper_task.cancel()
        # Let in-flight L2 audits finish (bounded) so their observations land.
        await runtime.drain_audits()
        embed_pool.shutdown(wait=False)
        if http is None:
            await client.aclose()
        if l1 is not None:
            l1.close()
        if qdrant is not None:
            qdrant.close()

    application = FastAPI(title="Cradle", lifespan=lifespan)
    application.include_router(router)
    return application


app = create_app()
