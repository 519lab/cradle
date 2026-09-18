# Cradle

Pass-through API gateway for OpenAI-compatible LLM APIs. Point clients at Cradle; it caches similar prompts, compresses misses, and forwards the rest upstream.

Same `/v1/chat/completions` contract as the provider. Caching and compression run on CPU in the gateway process — they are not a separate “local LLM app.”

1. **L1** exact-match cache (SHA-256, diskcache, p99 &lt; 2 ms lookup)
2. **L2** semantic cache (FastEmbed `BAAI/bge-small-en-v1.5` + Qdrant local, p99 &lt; 25 ms including embed on a warm, single in-flight model)
3. On miss: strip conversational fluff, call upstream, wrap the answer with a local prefix/suffix, write back both caches

No Redis. No Qdrant server process. No Ollama.

## Quick start

Point the OpenAI SDK (or any compatible client) at Cradle’s base URL instead of the provider. Keep the same `Authorization` header; Cradle forwards it upstream. No Cradle-issued API key.

```bash
cp .env.example .env                       # CRADLE_UPSTREAM_BASE_URL if the provider is not on localhost:8080
cp config/cradle.yaml.example config/cradle.yaml   # your local config; the real file is gitignored
uv sync --group dev
uv run python -m cradle
```

Dev listen: `http://127.0.0.1:8000`. Default upstream in YAML is `http://127.0.0.1:8080/v1`; set `CRADLE_UPSTREAM_BASE_URL` to the real provider (e.g. `https://api.openai.com/v1`).

## Docker

```bash
cp .env.example .env                               # optional CRADLE_UPSTREAM_BASE_URL
cp config/cradle.yaml.example config/cradle.yaml   # your local config; the real file is gitignored
docker compose up --build
```

Gateway is at `http://127.0.0.1:8000`. Compose default upstream is `http://host.docker.internal:8080/v1` (a server on the host, not `127.0.0.1` inside the container). Override with `CRADLE_UPSTREAM_BASE_URL`. L1/L2 state is the `cradle-data` volume (`/data`). FastEmbed weights are baked at `/opt/cradle/models/fastembed` so the volume does not hide them.

The image does **not** declare `VOLUME /data`. CI uses `compose.ci.yml` with tmpfs on `/data` and L2 off.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'
```

Drop-in: point the client at Cradle and keep its normal API key. Cache entries are isolated by SHA-256 of the presented Bearer token. Optional `auth.keys` in YAML is an allowlist if you want Cradle-issued keys instead.

### Clients (OpenAI-compatible)

| Client | Point at Cradle |
|---|---|
| Codex CLI | `openai_base_url` or `[model_providers.cradle] base_url` in `~/.codex/config.toml` |
| Grok CLI | `model.<id>.base_url` / `GROK_XAI_API_BASE_URL` toward Cradle |
| llama.cpp / LiteLLM / OpenAI SDK | `OPENAI_BASE_URL=http://<cradle>:8000/v1` |

Cradle picks the **backend** from the request `model` via `routes` in `config/cradle.yaml` (`gpt-*` → OpenAI, `grok-*` → xAI, `*` → local llama.cpp / LiteLLM). Unmatched models use `upstream.base_url`.

**Claude Code** talks the Anthropic Messages API (`ANTHROPIC_BASE_URL`, `/v1/messages`), not OpenAI `/v1/chat/completions`. Point it at LiteLLM (or similar) that already speaks Anthropic, or wait for a Cradle Anthropic adapter. Do not set `ANTHROPIC_BASE_URL` to Cradle today.

## Ops notes

- **Qdrant local is not recommended above 20,000 points** (`QdrantLocal.LARGE_DATA_THRESHOLD`). The scale path is a Qdrant **server** (`l2.mode: server` later, not implemented in v1), not more local-mode tuning. Watch `cradle_l2_points`.
- Dockerfile does **not** declare `VOLUME /data`. CI compose uses tmpfs on `/data`.
- `/metrics` requires the same Bearer key by default.
- MIT license. GitHub: `519lab/cradle`. No PyPI in v1.
- Reconstruction is a prefix/suffix envelope (partial PRD FR-3.1). Structural distillation is off (`features.structure: false`).
- This is a gateway, not an inference runtime: it does not load a chat model. Upstream is whatever OpenAI-compatible API you configure.
- **GPU rerank** (`l2.rerank_device: cuda`, ~2–5 ms/hit vs ~15–40 ms on CPU) is a separate image — `docker compose -f docker-compose.yml -f compose.gpu.yml up --build` on a host with an NVIDIA GPU + Container Toolkit. The default image is CPU; `cuda` on it crash-loops with an actionable error. See `RUNBOOK.md` §1.4.
- **Volatility guard** (`cache.volatility_guard`, default on): prompts that ask about time-sensitive things — "latest version", "current price", "today", weather, news — are cached for `cache.volatile_ttl_s` (default 300 s, `0` = never) instead of 24 h, so a correct-but-stale answer is not replayed all day. The reason is returned as `X-Cradle-Volatile`; an explicit `X-Cradle-Cache-TTL` header always wins.

## Per-request cache controls

| Header | Effect |
|---|---|
| `X-Cradle-Cache-Control: no-store` | Serve normally, do not store the response. |
| `X-Cradle-Cache-Control: no-cache` / `refresh` | Skip the cache read, call upstream, store the fresh answer. |
| `X-Cradle-Cache-TTL: <seconds>` | Per-entry TTL, clamped to `cache.ttl_s`; `0` = do not store. |
| `X-Cradle-Cache-Control: probe` | **Dry run.** Explain what Cradle would do (L1/L2/guard/rerank outcome per candidate) without serving, writing, or calling upstream. |

Probe example — tune thresholds without spending a token:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -H "X-Cradle-Cache-Control: probe" \
  -d '{"model":"gpt-4o-mini","temperature":0,"messages":[{"role":"user","content":"Which city is the capital of France?"}]}'
# (after "What is the capital of France?" was cached — real bge-small + bge-reranker output)
# {"object":"cradle.probe","cache":"HIT-L2","would_call_upstream":false,
#  "l1":{"key":"408a1a4e…","hit":false},
#  "l2":{"eligible":true,"embedded":true,"hit":true,
#        "candidates":[{"key":"3e35a800…","cosine":0.973013,"guard":"pass","rerank":"pass:8.3960","served":true}]}}
```

Every candidate the pipeline examined is listed best-first, so a rejected near-miss shows up as `"guard":"reject:numbers"` or `"rerank":"reject:<score>"` with `"served":false`.

## Verified L2 (audit sampling)

Semantic hits are only as good as the thresholds behind them, and a wrong hit returns `200`. Set `l2.audit_rate` (e.g. `0.02`) and Cradle re-asks upstream for that fraction of served L2 hits **in the background**, judges the fresh answer against the served one with the cross-encoder it already loads (`l2.audit_judge: auto`; answer-embedding cosine is the weak fallback when rerank is off), and:

- counts the verdict in `cradle_l2_audit_total{verdict}` — a measured false-hit rate on your real traffic;
- teaches the served entry a floor: judged wrong at similarity `s`, it never serves at `≤ s` again (`X-Cradle-Guard: reject:audit-floor`);
- self-heals: the fresh answer is cached under the querying prompt's own key;
- appends a labeled row to `data/audits.jsonl` (`query_similarity`, `judge`, `answer_score`, `verdict`, keys; prompt text only with `audit_log_text: true`) — the calibration set for `cosine_threshold` / `rerank_threshold`.

Verified live: "Which planet is farthest from the Sun?" hit the cached *closest*-planet answer (cosine 0.911, reranker 5.08 — the documented antonym gap). The audit judged the fresh answer against it at 0.53 (threshold 4.0), floored the entry at 0.911, and cached the correct answer under the new prompt; the next identical request served Neptune from L1.

Sampled hits carry `X-Cradle-Audit: scheduled`. Audits are real upstream calls made with the client's forwarded credentials after its response has completed, so keep the rate small.

## Metrics

Token savings ratio:

```
(inbound_prompt_tokens - upstream_prompt_tokens) / inbound_prompt_tokens * 100
```

Prometheus: `cradle_inbound_prompt_tokens_total`, `cradle_upstream_prompt_tokens_total`, `cradle_cache_hits_total{layer}`, `cradle_cache_misses_total`, `cradle_latency_seconds{stage}`.

Eval without a live LLM: `uv run pytest tests/eval`.

See `DESIGN.md` and `CLAUDE.md`.
