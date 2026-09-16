# Cradle

Pass-through API gateway for OpenAI-compatible LLM APIs: cache similar prompts, compress misses, forward the rest upstream.

## Branch model

- Long-lived: `develop` (integration) and `main` (blessed).
- Feature branches from `develop`. PRs target `develop`, never `main`.
- Never commit directly to `develop` or `main`.
- Conventional Commits with scope (`feat(cache):`, `fix(gateway):`).
- Changelog under `## [Unreleased]`.
- Remote: `https://github.com/519lab/cradle`. No PyPI in v1.

## Commands

```bash
uv sync --group dev
uv run ruff check .
uv run pytest
uv run pytest --cov=cradle --cov-report=term-missing --cov-fail-under=90
uv run python -m cradle
```

Optional real-BGE pair eval (downloads the ONNX model):

```bash
uv run pytest -m embed
```

Noisy CI latency gate:

```bash
CRADLE_SKIP_LATENCY=1 uv run pytest
```

Default config: `config/cradle.yaml`. Override upstream with `CRADLE_UPSTREAM_BASE_URL`. Proxy listen is `127.0.0.1:8000`. One `CRADLE_API_KEY` maps to one `user_id`.

L2 (Qdrant local) requires a **single** uvicorn worker. `WEB_CONCURRENCY` / `UVICORN_WORKERS` other than `1` is a startup error. Bypass streams (`stream+tools`, `n!=1`, logprobs) are raw SSE passthrough; cacheable stream misses still wrap text completions.

## Layout

`src/cradle/` — gateway, cache, embeddings, compress, reconstruct, upstream, metrics.
Modules stay ≤ 600 lines. No `ProxyService` god object.

Design contract: `DESIGN.md`. Product: `PRD.md`.
