# Cradle

Self-hosted OpenAI-compatible proxy that sits in front of a local (or remote) LLM API:

1. **L1** exact-match cache (SHA-256, diskcache, p99 &lt; 2 ms lookup)
2. **L2** semantic cache (FastEmbed `BAAI/bge-small-en-v1.5` + Qdrant local, p99 &lt; 25 ms including embed on a warm, single in-flight model)
3. On miss: strip conversational fluff, call upstream, wrap the answer with a local prefix/suffix, write back both caches

No Redis. No Qdrant server process. No Ollama. Caching/compression run on CPU.

## Quick start

```bash
cp .env.example .env   # set CRADLE_API_KEY
uv sync --group dev
uv run python -m cradle
```

Cradle listens on `http://127.0.0.1:8000`. Default upstream is `http://127.0.0.1:8080/v1` (llama.cpp / vLLM OpenAI-compat). Override with `CRADLE_UPSTREAM_BASE_URL` (example: `https://api.openai.com/v1`).

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $CRADLE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

One proxy key ⇒ one `user_id`. Add more keys in `config/cradle.yaml` `auth.keys`.

## Ops notes

- **Qdrant local is not recommended above 20,000 points** (`QdrantLocal.LARGE_DATA_THRESHOLD`). The scale path is a Qdrant **server** (`l2.mode: server` later, not implemented in v1), not more local-mode tuning. Watch `cradle_l2_points`.
- Dockerfile does **not** declare `VOLUME /data`. CI compose uses tmpfs on `/data`.
- `/metrics` requires the same Bearer key by default.
- MIT license. Local git only — no GitHub remote and no PyPI in v1.
- Reconstruction is a prefix/suffix envelope (partial PRD FR-3.1). Structural distillation is off (`features.structure: false`).

## Metrics

Token savings ratio:

```
(inbound_prompt_tokens - upstream_prompt_tokens) / inbound_prompt_tokens * 100
```

Prometheus: `cradle_inbound_prompt_tokens_total`, `cradle_upstream_prompt_tokens_total`, `cradle_cache_hits_total{layer}`, `cradle_cache_misses_total`, `cradle_latency_seconds{stage}`.

Eval without a live LLM: `uv run pytest tests/eval`.

See `DESIGN.md` and `CLAUDE.md`.
