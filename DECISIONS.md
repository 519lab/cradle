# Decisions

## ADR-0001: Locked v1 stack

**Date:** 2026-09-16
**Status:** Accepted
**Phase:** Architecture
**Deciders:** Greg

### Context

PRD listed Python vs Rust, Redis vs embedded KV, Qdrant server vs local, and optional 1B reconstruct.

### Decision

Python 3.12 FastAPI; L1 `diskcache`; L2 Qdrant local + FastEmbed BGE-small; rule-based compression (structure flag off); llama-cpp extra off; OpenAI wire; one Bearer key ⇒ one user_id; local git MIT, no remote/PyPI; default upstream `http://127.0.0.1:8080/v1`, Cradle listens on 8000.

### Rationale

Solo maintainability. ONNX is the L2 bottleneck regardless of language. Qdrant in-process local mode is a Python-client feature. Fewer daemons for edge/CPU.

### Consequences

workers=1. Qdrant local warns at 20k points. FR-3.1 reconstruction is prefix/suffix only.

See DESIGN.md Key Decisions.

## ADR-0002: RUNBOOK.md is operations-only, and kept in lock-step

**Date:** 2026-09-17
**Status:** Accepted
**Phase:** Operations
**Deciders:** Greg

### Context

Production operation of Cradle (deploy, verify, monitor, diagnose, tune, recover) had no
single home. The knowledge lived across README (quickstart/features), DESIGN.md
(architecture), the CHANGELOG (incident history), and session memory (the live test
container, the `/v1/models` passthrough gotcha, the `fuser` vs `pkill` trap). An operator
had nowhere to look during an incident.

### Decision

Add a top-level `RUNBOOK.md` scoped to **operational procedures only**. It cross-links
DESIGN.md (architecture), README.md (quickstart/feature explanation), and DECISIONS.md,
and does **not** restate them — a fact duplicated in two docs drifts apart. It carries a
"Verified against the live system on <date>" header, and every load-bearing command in it
is verified against a running instance, not invented.

It is **lock-step material**: any change touching a config key, an env var, a metric name,
a health/readiness condition, a capacity limit, or a per-request header updates RUNBOOK.md
in the **same PR** — the same rule DESIGN.md and CLAUDE.md already carry.

### Rationale

A runbook that drifts is worse than none: an operator runs a stale command mid-incident.
The scope boundary keeps it from growing into a second DESIGN.md. The lock-step rule and
the dated "verified against" header make staleness a PR-review item and a visible fact
rather than a silent rot.

### Consequences

RUNBOOK.md exists and is maintained on every relevant change. Its troubleshooting section
is built from real incidents in the CHANGELOG (e.g. #24 empty self-heal, #18 refresh L2
staleness, wrap-stream usage loss), not hypotheticals. Destructive recovery steps are
documented but marked as Greg's to run; they are never executed against a live instance to
"verify" them.

## ADR-0003: Runtime config is bind-mounted, not baked into the image

**Date:** 2026-09-18
**Status:** Accepted
**Phase:** Operations
**Deciders:** Greg

### Context

`config/cradle.yaml` is deliberately gitignored (local edits don't conflict on `git pull`;
see the CHANGELOG entry that introduced `.example` as the tracked template). But the Docker
images **also** `COPY config /app/config` and pointed `CRADLE_CONFIG` at the baked copy,
while compose mounted only the `/data` volume. That combination made a gitignored config a
build-time artifact, which is the worst of both: (1) editing the host file and restarting
did nothing — the container read the copy frozen at build time, so a raised body cap sat
unapplied through two rebuilds on the LAN GPU box (#36, triggered by #34); and (2) `COPY
config` captured whatever was in the working tree, unreviewed and free to drift from
`.example` or to still list a key a later code version rejects under `extra="forbid"` —
crash-looping a box after an upgrade (the #27 shape).

### Decision

Decouple the runtime config from the image build. The compose templates bind-mount the
`config/` directory read-only (`./config:/app/config:ro`); the images bake **only**
`cradle.yaml.example`, never a runtime `cradle.yaml`. `CRADLE_CONFIG=/app/config/cradle.yaml`
is unchanged, so a mounted host file is read and an edit is a `docker compose restart`, not
a rebuild.

The **directory** is mounted, not the file. The host `config/` dir always exists in a clone
(`.example` is tracked), so the mount never triggers compose's `create_host_path`; the host
`config/cradle.yaml` never exists in a fresh clone, so mounting *it* would auto-create it as
an empty directory and silently break config loading. With no host file present, Cradle
falls back to the built-in pydantic defaults — byte-identical to the pre-existing
"missing config → defaults" behavior.

### Rationale

This is not a reversal of the gitignore decision — the "copy the example, edit locally"
workflow is kept intact. It removes the *second* coupling (config → image) that made an
edit require a rebuild and made drift invisible. Directory-over-file is the specific choice
that keeps the graceful-fallback requirement while eliminating the empty-directory footgun.
A CI drift guard (`test_example_config_validates`) loads the tracked `.example` through the
real settings class, so a removed key surfaces in review, not at an operator's boot.

### Consequences

A config change is a restart. Deploys must copy a compose template to the gitignored
`docker-compose.yml` **after** this change to pick up the new `volumes:` mount — the fix is
in the image only if the running compose file has the mount. The bake reversal is partial:
models and the example config are still baked (self-contained image), only the runtime
config is externalized. Verified end-to-end on the CPU image (see RUNBOOK.md §1.1/§7.1).

## ADR-0004: Structured request logging; prompt/response text is opt-in and off by default

**Date:** 2026-09-18
**Status:** Accepted
**Phase:** Operations
**Deciders:** Greg

### Context

Watching Cradle's logs showed only uvicorn's access lines. Two causes: `__main__.py` started
uvicorn with no logging configuration, so the `cradle` loggers sat at WARNING and every
request-path line was dropped before it could escape the process; and the request path emitted
no lines at all — every per-request fact (cache layer, similarity, guard/rerank outcome,
per-stage timings, token counts) went only into response headers and Prometheus, never a log.
Greg asked for configurable logging that can include prompt/completion text or stay
non-sensitive. DESIGN.md carried a flat "No prompt text in logs," which this ADR reinterprets
rather than deletes.

### Decision

Add one structured request-completion log line and make verbosity configurable on **two
independent axes**:

- `logging.level` — how much detail. `INFO` emits one line per request; `DEBUG` adds a line
  per rejected L2 candidate (guard / rerank / audit-floor), carrying the candidate **key** and
  score, never the compared texts.
- `logging.content` — how sensitive. `none` (default) logs only keys, hashes, the cache
  decision, scores and timings; `prompts` adds the request text; `prompts_and_completions` adds
  the response text. Logged text is truncated to `logging.max_text_chars`.

The two axes are deliberately not collapsed: a single dial would make `DEBUG` imply dumping
prompt text, so a production tenant couldn't be debugged without leaking its prompts. All
potentially-sensitive fields pass through **one redaction helper** (`logging_setup.request_fields`),
so the "no text at `none`" guarantee lives in one place and is unit-tested on every terminal
path. `logging.format` defaults to human-readable `key=value` (Greg watches stdout); `json` is
the opt-in for aggregators. Setup configures only the `cradle` logger tree, leaving uvicorn's
access log intact.

DESIGN.md's "No prompt text in logs" becomes "**by default**; opt-in via `logging.content`."
This is the request-path analogue of the existing `l2.audit_log_text` flag.

### Rationale

`content: none` as the default keeps the safe behavior the old flat rule guaranteed, while
letting Greg raise verbosity deliberately on a scratch instance. One chokepoint plus a
`content: none` test that asserts no prompt/response text appears on any path turns the
guarantee from a convention into a regression test. Two swallowed exceptions found in the same
code (`_maybe_embed`, `_rerank_ok` discarded the traceback into a bare counter) are fixed here
too — an embed or rerank failure now logs its traceback.

### Consequences

New `logging:` config section and `CRADLE_LOGGING__*` env vars — lock-step material, so
RUNBOOK.md §4 documents them. `content` above `none` writes PII to the logs; the config
comment and this ADR mark it. A client that disconnects mid-stream now still emits a request
line (`disconnected=true`), closing a prior observability hole where the streaming success
line ran only after the body completed.

## ADR-0005: The L2 match embedding excludes the system prompt

**Date:** 2026-09-18
**Status:** Accepted
**Phase:** Correctness
**Deciders:** Greg

### Context

A live instance served an L2 hit whose answer belonged to a *different* user task (issue #40):
under one large fixed system prompt (open-webui traffic), a "riff on these creative prompts"
request was served a title-generation-shaped answer as a `200`, at `sim=0.987 rerank=pass:7.59`.
`embed_text` was built from **all** messages, system prompt included. When the system prompt is
a large block identical across requests, it dominates the embedding: the short discriminating
user turn is a fraction of the vector, so unrelated user questions land at very high cosine, and
both the cosine gate and the cross-encoder rerank are diluted the same way and pass.

Measured on the real bge models (thresholds cosine 0.90 / rerank 4.0), two genuinely different
user tasks under one shared system prompt:

| embed_text | cosine | rerank |
|---|---|---|
| system+user (before) | 0.93 | 6.27 |
| user-only (after) | 0.61 | −4.61 |

### Decision

`embed_text` is built from **user/assistant turns only**; system and developer turns are
excluded from the match embedding. The system prompt stays in the cache *identity* —
`system_prompt_version` scoping plus the L1 key — so a different system prompt still separates
entries (correctness) and the token-savings/caching behavior is unchanged. It simply no longer
pollutes similarity. A request with no user/assistant content (system-only) has an empty
`embed_text` and is L2-ineligible (an empty vector matches anything). `pipeline_version` is
bumped **v2→v3**: existing L2 vectors and records were computed from system+user text, and
mixing a user-only query against them at the vector/guard/rerank stages is unpredictable, so the
bump makes pre-fix entries miss and age out on TTL.

### Rationale

The two jobs of the system prompt were conflated. It legitimately belongs to *whether a cached
answer may be replayed* (a different system prompt can change the answer) and to token savings —
both handled by identity scoping. It does **not** belong to *which stored answer this question
matches* — that is the discriminating user content. Separating them fixes the false hit at the
root (a whole class of open-webui/RAG/agent shapes with a big shared frame) rather than papering
over it with a higher threshold, which would also suppress genuine paraphrases. `embed_text` now
means "matchable content," not "the prompt"; the guard, rerank, `audits.jsonl` rows, and the
`logging.content` `prompt=` field all read it under that meaning.

### Consequences

Operator-visible: the `pipeline_version` bump ages out the entire existing L2 (and L1) cache on
deploy — hit rate drops until the caches refill and old entries pass TTL (see RUNBOOK §7.2, the
same lever as the v1→v2 wrap-stream fix). The `prompt=` field in the request log lines and the
`query_text`/`candidate_text` rows in `audits.jsonl` now carry user/assistant turns only, not the
system prompt — shorter lines, and less prompt PII when `logging.content`/`audit_log_text` are on.
A separate, pre-existing eval finding surfaced while proving this (three `l2_pairs.jsonl` hit pairs
fall below the 0.90 floor with real bge) is tracked independently; it is the inverse failure —
false negatives at the threshold, not #40's false positives.

## ADR-0006: Tool-enabled streams are cacheable via verbatim tee + post-stream decision

**Date:** 2026-09-18
**Status:** Accepted
**Phase:** Performance / Correctness
**Deciders:** Greg

### Context

Measurement showed Cradle's dominant real traffic — Open-WebUI chat turns, which carry `tools`
and `stream:true` — was **100% BYPASS**: not cached, full upstream cost every time. Only
Open-WebUI's internal task calls (title/tags/follow-ups, no tools) benefited. Cause:
`is_cacheable` rejected `stream and (has_tools or logprobs)` up front, because the streaming
**wrap path** (`_wrap_stream`) rebuilds the outbound stream as content-only frames and cannot
carry a tool-call response. Non-streaming tool requests were already cached (text answers only;
`cache_skip_reason` refuses `finish_tool_calls`), so only the streaming variant was excluded.

Greg asked whether caching a text answer to a tool-enabled request is even correct — a valid
concern: if the model answers in text when it should have called a tool, a cached replay
suppresses the tool. Measured live: the model's answer-vs-call routing is largely deterministic
per prompt, and the **volatility guard already refuses to cache** the dangerous cases
(time/date/weather/"latest"), so the risk overlap is contained for the common shapes. The
residual gap is a stateful tool with no time-word (balance/inventory), accepted under a flag.

### Decision

Add a third streaming path, `_passthrough_cache_stream`, gated by `cache.cache_tool_streams`
(default **on**). It tees upstream bytes to the client **verbatim** (like bypass — a tool call
relays intact) while feeding a **copy** through a bounded line-buffer into `parse_and_accumulate`.
After the stream, it caches the response **only** when a fail-closed allowlist passes: parsed
terminal `finish_reason in {stop, eos}`, no error, no tool call, no delta key outside
`{role, content, tool_calls}` (subsumes legacy `function_call`, refusal, reasoning, …), no
unparseable framing, client still connected, and `cache_skip_reason` clean. The cached body is
the **`merge()`-reconstructed** representation the JSON path stores (JSON and streaming share a
cache key — `stream` is excluded from `hash_input` — so cross-mode replay must match). No
`include_usage` injection: the client stream stays verbatim, and `usage: acc.usage or {}` is
cached (`synthesize_sse` omits the usage frame on empty usage). `layer_hit` stays `"miss"`; a
separate `ctx.cacheable_passthrough_stream` flag carries the wire strategy (cache outcome and
transport are orthogonal — a new `layer_hit` value would disturb 18 read sites). `logprobs`
streams still bypass.

### Rationale

Verbatim-to-the-client makes the tool-call case impossible to corrupt: the client always gets
exactly the upstream bytes, and a parse failure only disables *caching*, never the stream.
Fail-closed-by-allowlist means a future provider delta shape defaults to "don't cache" rather
than silently dropping fields on replay. Reconstructed-not-raw keeps JSON and stream entries
interchangeable under their shared key (a real cross-mode bug caught in review). Design reviewed
by codex (read-only) + a stronger advisor; both found holes in the first draft (raw-vs-merge
representation, not-fail-closed gate, an `include_usage` injection that would have broken
`test_bypass_stream_payload_untouched`).

### Consequences

Cradle's chat path is cacheable for the first time (measured 0% → the L1 tier of non-streaming
tool requests). `cache.cache_tool_streams: false` restores the old always-bypass behavior without
a redeploy. Accepted product risk, stated plainly: the volatility guard is a regex over user text
and does not inspect tool semantics, so a stateful tool with no time-word can serve a stale
cached text answer under the flag. `sse.py` gained a `cache_disabled` accumulator flag; the wrap
path ignores it (it has its own tool_call abort). No new `pipeline_version` bump — this only adds
newly-cacheable entries; it does not change how existing entries are keyed or read.
