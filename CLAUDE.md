# Cradle

Local semantic cache + token-compression proxy in front of an OpenAI-compatible LLM.

## Branch model

- Long-lived: `develop` (integration) and `main` (blessed).
- Feature branches from `develop`. PRs target `develop`, never `main`.
- Never commit directly to `develop` or `main`.
- Conventional Commits with scope (`feat(cache):`, `fix(gateway):`).
- Changelog under `## [Unreleased]`.
- Local git only in v1 — no GitHub remote, no PyPI.

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

## Layout

`src/cradle/` — gateway, cache, embeddings, compress, reconstruct, upstream, metrics.
Modules stay ≤ 600 lines. No `ProxyService` god object.

Design contract: `DESIGN.md`. Product: `PRD.md`.
