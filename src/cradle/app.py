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
from cradle.gateway.routes import router
from cradle.metrics import prometheus as m
from cradle.runtime import Runtime
from cradle.tenancy import assert_auth_ready, load_principals

log = logging.getLogger("cradle")


def _ensure_data_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, stat.S_IRWXU)


def _build_embedder(settings: Settings, override: Embedder | None) -> Embedder | None:
    if override is not None:
        return override
    if not settings.features.l2:
        return None
    from cradle.embeddings.fastembed import FastEmbedEmbedder

    cache_dir = settings.data_dir / "models" / "fastembed"
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir)
    return FastEmbedEmbedder(
        cache_dir=cache_dir,
        model_name=settings.l2.model,
        threads=settings.l2.onnx_threads,
        dim=settings.l2.dim,
    )


def create_app(
    settings: Settings | None = None,
    embedder: Embedder | None = None,
    http: httpx.AsyncClient | None = None,
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

        workers = os.environ.get("WEB_CONCURRENCY", "1")
        if settings.features.l2 and str(workers) not in {"1", ""}:
            log.warning("uvicorn workers>1 is unsupported with Qdrant local (v1 is workers=1)")

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
            l1_ready=l1_ready,
            l2_ready=l2_ready,
            embedder_ready=embedder_ready,
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
        yield
        stop.set()
        task.cancel()
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
