# Cradle production runbook

Operational procedures for running Cradle in production: deploy, verify, monitor,
diagnose, tune, recover. **Architecture** lives in `DESIGN.md`, **feature
explanation and quickstart** in `README.md`, **decisions** in `DECISIONS.md` — this
file cross-links them and does not restate them. When a fact here and a fact there
disagree, the code wins; fix both.

*§2 (health/readiness, POST headers, `/v1/models`), §4 (metric names), and §6.1 (probe)
verified against the live container `192.168.50.30:8000` (single llama.cpp upstream serving
`unsloth/Qwen3.8-27B-GGUF`) on 2026-09-17. §1, §3, §5, §7 derived from the code at that
date — drive them against a running instance before relying on them and update this line.*

> **Keeping this current is not optional.** Any change touching a config key, an env
> var, a metric name, a health/readiness condition, a capacity limit, or a per-request
> header updates this runbook **in the same PR** (see `DECISIONS.md` ADR-0002 and the
> lock-step rule in `CLAUDE.md`). Bump the "Verified against" date when you re-check a
> command against a running instance.

---

## 0. At a glance

| Thing | Value | Source |
|---|---|---|
| Listen (dev) | `127.0.0.1:8000` | `server.host`/`server.port` |
| Listen (compose) | `0.0.0.0:8000` | `CRADLE_SERVER__HOST=0.0.0.0` |
| Workers | **exactly 1** while `features.l2` on | hard ceiling, no horizontal scale |
| L1 cache | diskcache under `{data_dir}` | exact-match, SHA-256 key |
| L2 cache | Qdrant **local** under `{data_dir}/l2` | semantic; **≤ ~20k points** |
| State | `{data_dir}` (compose: `/data` volume `cradle-data`) | survives restart |
| Models | baked at `/opt/cradle/models/fastembed` in the image | not under `/data` |
| Liveness | `GET /healthz` → `200 {"status":"ok"}` (always) | unauthenticated |
| Readiness | `GET /readyz` → `200`/`503` | unauthenticated |
| Metrics | `GET /metrics` | auth iff `metrics.require_auth` (default **true**) |

**Cradle does not run a model.** It proxies an OpenAI-compatible upstream you
configure. "Latency", "the model is slow", "wrong answer content" that reproduce
against the upstream directly are upstream problems, not Cradle.

---

## 1. Deploy

### 1.1 First deploy (compose)

```bash
cp .env.example .env                                # set CRADLE_UPSTREAM_BASE_URL
cp config/cradle.yaml.example config/cradle.yaml    # the real file is GITIGNORED — this is a real first-deploy trap
docker compose up --build
```

- `config/cradle.yaml` is **gitignored**; a fresh clone has no `config/cradle.yaml`,
  only `.example`. Missing it means Cradle falls back to `CRADLE_CONFIG=config/cradle.yaml`
  and the pydantic defaults. Copy the example first.
- Compose reaches a host-side upstream via `host.docker.internal` (default
  `http://host.docker.internal:8080/v1`). Override with `CRADLE_UPSTREAM_BASE_URL`.
- L1/L2 state is the `cradle-data` named volume mounted at `/data`. It is **not**
  wiped on `docker compose up`/`restart`; it is wiped only by `docker compose down -v`.
- The image bakes the embedder (`BAKE_EMBEDDINGS=1`) and the reranker
  (`BAKE_RERANK=1`) so a fresh container is self-contained — no ~1.1 GB pull at
  warm-up. Set `BAKE_RERANK=0` only when `features.l2_rerank` is off. `HF_TOKEN` is a
  **build arg** for gated/rate-limited model pulls (empty = anonymous).

### 1.2 Dev / bare-metal

```bash
uv sync --group dev
uv run python -m cradle        # binds server.host:server.port, workers=1, factory=False
```

`python -m cradle` hard-codes `workers=1`, so the single-worker rule can't be
violated by this launch path. It **can** be violated by a raw `uvicorn`/`gunicorn`
CLI — see §3.1.

### 1.3 What fails startup (won't come up)

These raise during the lifespan startup and the process exits — a crash-loop, not a
`/readyz` 503:

| Condition | Error / cause | Fix |
|---|---|---|
| `WEB_CONCURRENCY`/`UVICORN_WORKERS` ≠ 1 with L2 on | `RuntimeError: …incompatible with Qdrant local L2; v1 requires a single worker` | Unset the var or set `1` (§3.1) |
| `l2.mode: server` | `NotImplementedError: l2.mode=server is reserved; v1 uses Qdrant local` | Stay on `l2.mode: local` (§3.2) |
| `auth.keys` set but no `token_env` resolves | `RuntimeError: auth.keys is set but no token_env values resolved…` | Export the listed env vars, or clear `auth.keys` for intercept mode |
| `data_dir` not writable/executable | `PermissionError` re-raised | Fix perms (dir is created `0700`) |
| Embedder dim mismatch | `RuntimeError: embed dim N != 384` | Wrong `l2.model`/`l2.dim` |
| `features.local_1b` on without extra | `RuntimeError: features.local_1b requires extra cradle[local-1b]` | Install extra or turn the flag off |

**Reranker warm-up failure does NOT crash startup.** It is caught, logged
(`reranker warm-up failed; L2 rerank is degraded`), leaves `reranker_ready=False`, and
makes `/readyz` return **503** (see §2.2, §5). This is deliberate: a container with the
#5 entity-swap protection silently off must not take traffic with a green check.

---

## 2. Verify it's up — and that it's actually Cradle

### 2.1 Liveness & readiness

```bash
curl -s http://<host>:8000/healthz   # {"status":"ok"}    always 200 if the process is alive
curl -s http://<host>:8000/readyz    # {"status":"ready"} 200, or {"status":"not_ready"} 503
```

Both are **unauthenticated** — safe for load balancers and uptime checks. `/healthz`
is liveness only (unconditional). `/readyz` gates traffic; wire the LB to it.

### 2.2 What makes `/readyz` say `not_ready` (503)

`ok` requires every **enabled-feature** check to pass (`src/cradle/gateway/routes.py:30-45`):

- `features.cache` on and L1 not ready → 503
- `features.l2` on and (L2 or embedder) not ready → 503
- `features.l2` **and** `features.l2_rerank` on and reranker not ready → 503

The Docker `HEALTHCHECK` polls `/readyz` (`--start-period=60s`), so a degraded reranker
shows as an unhealthy container.

### 2.3 Confirm it's Cradle, not the bare upstream

**Gotcha (cost real time on 2026-09-17):** `GET /v1/models` on a single-backend
deploy is a straight upstream passthrough and carries **no `x-cradle-*` headers** — it
looks exactly like the upstream. Do **not** use `/v1/models` to decide "is this
Cradle?".

Confirm via a `POST`, whose response carries the Cradle headers:

```bash
curl -s -D - -o /dev/null http://<host>:8000/v1/chat/completions \
  -H "Authorization: Bearer any-token" -H "Content-Type: application/json" \
  -d '{"model":"qwen","temperature":0,"messages":[{"role":"user","content":"ping"}]}'
# Look for:  x-cradle-cache: MISS|HIT-L1|HIT-L2|BYPASS
#            x-cradle-pipeline: v2   x-cradle-upstream: <backend>   x-request-id: <uuid>
```

In **intercept mode** (`auth.keys: []`, the default) any Bearer works and is forwarded
upstream / used as the cache tenant (SHA-256 of the token). If `auth.keys` is set,
only the configured tokens authenticate.

### 2.4 Response headers you'll read while operating (`pipeline._headers`)

| Header | Meaning |
|---|---|
| `X-Cradle-Cache` | `HIT-L1` / `HIT-L2` / `MISS` / `BYPASS` |
| `X-Cradle-Upstream` | resolved backend name (`default` or a named upstream) |
| `X-Cradle-Pipeline` | served entry's `pipeline_version` |
| `X-Cradle-Inbound-Tokens` / `X-Cradle-Upstream-Tokens` | compression savings = inbound − upstream (upstream-tokens omitted on streaming misses) |
| `X-Cradle-Similarity` | L2 cosine of the served/examined candidate |
| `X-Cradle-Guard` | `reject:<reason>` — an L2 candidate the precision guard/audit-floor rejected |
| `X-Cradle-Rerank` | `pass:<score>` / `reject:<score>` / `fail-open` / `off` |
| `X-Cradle-Volatile` | the volatility guard fired; value = reason |
| `X-Cradle-Audit` | `scheduled` — a verified-L2 audit was queued for this hit |
| `X-Request-ID` | Cradle's own request id (upstream's is relayed as `x-cradle-upstream-request-id`) |

---

## 3. Capacity & hard limits

### 3.1 Single worker — no horizontal scaling within one process

While `features.l2` is on, Cradle runs **exactly one uvicorn worker**. Qdrant local is
an in-process embedded store; multiple workers would each open it and corrupt/lock it.
Enforced at startup (`src/cradle/app.py:26-34`): if `WEB_CONCURRENCY` or
`UVICORN_WORKERS` is set to anything other than `1`/empty, startup raises and the
process exits.

**Consequence for scaling:** you cannot add workers while L2 is on. The guard early-returns
when `features.l2` is off — so it *permits* multi-worker in an L1-only deploy — but that
path is untested here and `python -m cradle` hard-codes `workers=1`, so there is no
supported launch for it; don't rely on it without verifying concurrent L1 diskcache access
and the per-worker purge/audit loops. To scale, run multiple Cradle **processes** — but
each gets its **own** L1/L2 state (`data_dir`), so hit rate does not pool across them and
tenants can land on different caches behind a load balancer. There is no shared-cache story
in v1. Treat one Cradle process as the unit of deployment.

### 3.2 Qdrant local ≤ ~20,000 points — advisory ceiling, no in-version remedy

Qdrant local mode is not recommended above ~20k points. Cradle **warns** but does not
evict or block: every purge cycle (`cache.purge_interval_s`, default 300 s) it sets the
`cradle_l2_points` gauge and, when the count exceeds `l2.points_warn` (default 20000),
logs `Qdrant local collection has N points (threshold 20000); scale path is
l2.mode=server later`.

The scale path is a Qdrant **server** (`l2.mode: server`), which is **not implemented in
v1** (setting it is a startup error). So this is a real hard ceiling. When you approach
it, the levers are: lower `cache.ttl_s` / `cache.volatile_ttl_s` so points age out
faster, or accept the cap. Do not "tune around it" in local mode.

**Alert on `cradle_l2_points`** approaching 20k (e.g. > 15000) so the ceiling is a
planned conversation, not a surprise.

### 3.3 Request body cap

Bodies over `server.max_body_bytes` (default 1 MiB) get a `413`. Raise it in config if a
legitimate client sends larger prompts.

---

## 4. Monitoring

`/metrics` (Prometheus text). **Auth:** required iff `metrics.require_auth` (config
default **true**; the live test container runs it **false**). All metrics are on a
private registry — only `cradle_*` series appear.

### 4.1 The dashboard that matters

| Signal | Metric(s) | Read it as |
|---|---|---|
| **Hit rate** | `cradle_cache_hits_total{layer}` vs `cradle_cache_misses_total` | The product working. Split L1 (exact) vs L2 (semantic). |
| **Token savings** | `(inbound − upstream) / inbound` from `cradle_inbound_prompt_tokens_total`, `cradle_upstream_prompt_tokens_total` | Compression payoff. |
| **Upstream health** | `cradle_upstream_errors_total{status}` | Climbing 429/5xx = upstream trouble, surfaced through Cradle. |
| **Latency** | `cradle_latency_seconds{stage}` — stages `l1`, `l2`, `embed`, `compress`, `upstream`, `reconstruct` | `l1` p99 should be ~ms; `embed`+`l2` the semantic cost; `upstream` dominates on misses. |
| **Readiness** | `cradle_ready{component}` — `l1`/`l2`/`embedder`/`reranker` (1/0) | `reranker=0` = #5 protection off (also 503s `/readyz`). |
| **L2 growth** | `cradle_l2_points` | The 20k ceiling (§3.2). |

### 4.2 Correctness / cache-quality signals

| Metric | What it tells you |
|---|---|
| `cradle_l2_guard_rejects_total{reason}` | Near-miss L2 candidates the precision guard / `audit-floor` blocked (numbers/negation/entity/audit-floor). Healthy — the cache refusing wrong hits. |
| `cradle_l2_rerank_rejects_total` | Entity-swap candidates the cross-encoder blocked. Healthy. |
| `cradle_l2_rerank_fail_open_total` | L2 hits served **without** rerank verification (reranker unavailable/timed out). **Should stay ~flat.** A climbing rate = rerank is effectively off and wrong hits can slip; investigate the reranker. |
| `cradle_cache_write_skips_total{reason}` | Responses the write-quality gate refused to cache (empty/`length`/`content_filter`/tool-calls). A spike in `length`/empty = upstream returning junk (see §5, #24). |
| `cradle_l2_audit_total{verdict}` | If `l2.audit_rate>0`: **measured false-hit rate** on real traffic. Watch `disagree`; it is the real wrong-hit signal. `error` = the audit's own upstream call failed. |
| `cradle_l2_audit_answer_score{judge}` | Distribution of judge scores; calibration input for thresholds. |
| `cradle_volatile_prompts_total{reason}` | How often the volatility guard applied a short TTL. |
| `cradle_embed_errors_total` | Embedder failures/timeouts → those requests fell through to a real miss. |

Also emitted: `cradle_requests_total{endpoint,status,cache}` (top-line request counter —
use for RPS and the request-weighted hit ratio), `cradle_cache_probes_total{cache}` (probe
dry-run volume), `cradle_over_compression_blocks_total{reason}` (compression skipped
because a protected span blocked it).

### 4.3 Suggested alerts

- `cradle_ready{component="reranker"} == 0` for > 5 min (with `l2_rerank` on) — degraded correctness.
- `rate(cradle_l2_rerank_fail_open_total[10m]) > 0` sustained — rerank effectively off.
- `cradle_l2_points > 15000` — approaching the local-mode ceiling.
- `rate(cradle_upstream_errors_total{status=~"5.."}[5m])` elevated — upstream trouble.
- `/readyz` != 200 — pull the instance from the LB.

---

## 5. Troubleshooting (symptom → check → cause → action)

These are drawn from real production incidents (referenced issues are in `CHANGELOG.md`).

| Symptom | Check | Likely cause | Action |
|---|---|---|---|
| Container "healthy-ish" but `/readyz` = 503 | `cradle_ready{component}`; startup logs | A component didn't warm up. Most common: reranker load failed (embedder/L2 crash startup instead) | Fix the model/weights; if `l2_rerank` isn't needed, set `features.l2_rerank: false`. Don't route traffic to a 503 instance. |
| Startup crash-loop | container logs for `RuntimeError`/`NotImplementedError`/`PermissionError` | One of the §1.3 conditions (workers≠1, `l2.mode=server`, bad `auth.keys`, unwritable `data_dir`, dim mismatch) | Fix per §1.3. This is a startup failure, not a readiness issue. |
| Wrong / stale answer served as `HIT-L2` | Re-run with `X-Cradle-Cache-Control: probe` (§6.1); read `X-Cradle-Similarity`/`X-Cradle-Guard`/`X-Cradle-Rerank` | A near-miss cleared cosine + guard + rerank (e.g. antonym/negation edge — the documented residual gap) | Enable `l2.audit_rate` to measure it; raise `l2.cosine_threshold`/`l2.rerank_threshold`; the audit-floor will self-heal the specific entry. Purge via `pipeline_version` bump (§7.2) if widespread. |
| Empty answers appearing / cached | `cradle_cache_write_skips_total{reason}`; upstream direct | Upstream returned `200` with empty content (`finish_reason: stop`, no text) — seen live from llama.cpp (#24). Gate now refuses to cache it | Investigate the upstream. The write-quality gate + audit already prevent caching/self-healing empties; a spike means the upstream is misbehaving. |
| Paraphrase keeps replaying an old answer after a `refresh` | `X-Cradle-Cache`; whether L2 point updated | Pre-fix behavior (#18) left the stale L2 point when only L1 was refreshed | On current code the prompt is always embedded on refresh; if seen, confirm the running image is current (`X-Cradle-Pipeline`). |
| Streaming cache hit replays empty `usage` | client `stream_options.include_usage`; `X-Cradle-Pipeline` | Pre-v2 wrap-stream entries stored empty usage (the wrap-stream usage fix). `pipeline_version` bumped v1→v2 so they miss and age out | Ensure the running pipeline is `v2` (it is, live); old entries expire on TTL. |
| High latency on hits | `cradle_latency_seconds{stage}` | `l1` slow → disk pressure on `data_dir`; `embed`/`l2` slow → CPU contention (single embed thread) | Move `data_dir` to fast disk; give the box CPU headroom; the embed pool is a single thread by design. |
| All requests are `MISS`/`BYPASS`, cache never fills | `X-Cradle-Cache`; `cradle_cache_write_skips_total` | Requests uncacheable: `temperature > cache.max_temperature`, streaming+tools / `n!=1` / logprobs (bypass), or write-skips | Confirm client params; bypass is by design for those shapes. |
| Upstream errors surfaced to clients | `cradle_upstream_errors_total{status}`; `x-cradle-upstream-request-id` on the response | The upstream 4xx/5xx'd; Cradle relays body + `retry-after`/`x-ratelimit-*` | It's an upstream problem. Cradle re-frames 5xx as `502`; the client's backoff has the relayed `retry-after`. |
| `/v1/models` "doesn't look like Cradle" | — | Single-backend passthrough has no `x-cradle-*` headers (§2.3) | Not a bug. Confirm via `POST` instead. |

For the reranker fail-open policy (transient timeout serves fail-open and counts
`cradle_l2_rerank_fail_open_total`; a persistent load failure 503s `/readyz` instead), see
DESIGN decision 23.

---

## 6. Tuning (do it with the probe — zero upstream cost)

### 6.1 The probe: explain a decision without spending a token

```bash
curl -s http://<host>:8000/v1/chat/completions \
  -H "Authorization: Bearer any-token" -H "Content-Type: application/json" \
  -H "X-Cradle-Cache-Control: probe" \
  -d '{"model":"qwen","temperature":0,"messages":[{"role":"user","content":"Which city is the capital of France?"}]}'
# → {"object":"cradle.probe","cache":"HIT-L2","would_call_upstream":false,
#     "l1":{...},"l2":{"eligible":true,"embedded":true,"hit":true,
#       "candidates":[{"key":"…","cosine":0.973,"guard":"pass","rerank":"pass:8.40","served":true}]}}
```

Every examined candidate is listed best-first with its `cosine`, `guard`, `rerank`, and
`served`. A rejected near-miss shows `"guard":"reject:numbers"` or
`"rerank":"reject:3.5"` with `"served":false`. **This is the primary tuning tool** — it
never writes and never calls upstream.

### 6.2 The knobs (defaults in parentheses; full tree in `config.py`)

| Knob | Default | Effect / when to change |
|---|---|---|
| `l2.cosine_threshold` | `0.90` (min 0.85) | Semantic recall floor. Lower = more L2 hits + more false hits; raise if the probe shows wrong candidates clearing it. |
| `l2.rerank_threshold` | `4.0` | Cross-encoder logit floor. Paraphrases score ≥ ~4.8, entity swaps ≤ ~3.5. |
| `l2.query_top_k` | `5` | Candidates examined per query; `1` = nearest-neighbor only. Higher recovers a true paraphrase behind a near-miss. |
| `cache.ttl_s` | `86400` | Default TTL and the clamp ceiling for `X-Cradle-Cache-TTL`. Lower to age entries out faster (also relieves the 20k ceiling). |
| `cache.volatile_ttl_s` | `300` (`0`=never) | TTL for prompts the volatility guard flags time-sensitive. |
| `l2.audit_rate` | `0.0` (off) | Fraction of L2 hits re-verified upstream in the background. **Sensible range 0.01–0.05** — it spends real upstream calls. The way to *measure* your false-hit rate. |
| `l2.audit_judge` | `auto` | `auto`=reranker when loaded, else embed cosine. |
| `cache.max_temperature` | `1.0` | Requests above this aren't cached. |

Per-request overrides (headers): `X-Cradle-Cache-Control: no-store | no-cache | refresh
| probe`, and `X-Cradle-Cache-TTL: <seconds>` (clamped to `cache.ttl_s`; always wins
over the volatility guard).

Changing a threshold does **not** rewrite existing entries — it changes future
serve/reject decisions. Probe before and after to confirm.

---

## 7. Recovery & rollback

### 7.1 Restart

```bash
docker compose restart cradle    # keeps the cradle-data volume (L1/L2 survive)
```

In-flight verified-L2 audits are drained (bounded, ~30 s) at shutdown so their
observations land; the purge loop is cancelled; L1/Qdrant are closed cleanly.

### 7.2 Invalidate the whole cache the *safe* way — bump `pipeline_version`

`pipeline_version` (default `v2`) folds into the L1 key and the L2 filter. Bumping it
(e.g. `v2`→`v3`, via `pipeline_version` in config or `CRADLE_PIPELINE_VERSION`) makes
**every** existing entry miss and age out on TTL — no file deletion, reversible by
setting it back **within the entries' TTL** (they are filtered out, not deleted; the purge
loop only removes TTL-expired entries). This is the preferred lever after a bad-cache
incident or a change to compression/reconstruction behavior. Prefer it over wiping state.

> Note: the `cache.evict_old_pipeline` config knob (default `true`) is currently **not
> wired to anything** — nothing in the code reads it, so it does not actively delete
> old-version entries. The reversibility above holds regardless of its value. (Flagged as a
> dead config field to remove or implement; tracked separately from this runbook.)

### 7.3 Destructive resets — **Greg runs these, not Claude/automation**

State the target host explicitly and confirm it's the intended instance before running.
These are irreversible.

```bash
# Full cache/state wipe (compose): removes the cradle-data volume.
docker compose down -v          # DESTROYS L1 + L2. Re-warms from empty on next start.

# Bare-metal: remove the data dir Cradle recreates on boot.
rm -rf <data_dir>               # e.g. ./data — DESTROYS L1 + L2.
```

Do not `down -v` or `rm -rf <data_dir>` against a shared/live instance to "clear a bad
entry" — use §7.2 (`pipeline_version`) or the per-request `no-cache`/`no-store` headers,
which are surgical and reversible.

### 7.4 Rolling back a release

Cradle is stateless-per-request; the state is the cache. To roll back code, redeploy the
prior image — the `cradle-data` volume is compatible as long as the L1 schema and
`pipeline_version` match. If the new release bumped `pipeline_version`, rolling back the
code **and** the version is clean (old entries were already aging out); leaving a newer
`pipeline_version` on older code just means those entries miss. When in doubt, §7.2.

---

## 8. Backup

The only durable state is `{data_dir}` (compose volume `cradle-data`): L1 diskcache and
the Qdrant-local collection. It is a **regenerable cache**, not a system of record —
losing it costs hit rate until it re-warms, nothing more. Back it up only if re-warm
cost matters; otherwise no backup is needed. There is no external database.

The verified-L2 audit log (`{data_dir}/audits.jsonl`, when `l2.audit_log` on) is the one
piece worth keeping — it's your threshold-calibration dataset. Copy it off before a wipe.
