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

---

## ADR-0007: Compression is gated by expected benefit, not run unconditionally

**Date:** 2026-09-19
**Status:** Accepted
**Phase:** Efficiency
**Deciders:** Greg

### Context

The PRD (line 15) promises "token compression on cache misses" as a core value proposition,
and DESIGN goal G4 pins **≥40% token savings** on a documented golden fixture of *verbose*
prompts. But live measurement (#52) showed rule-based fluff-stripping removes **~0 tokens** from
real Open-WebUI traffic, whose user turns are terse and carry no leading/trailing pleasantry or
filler word for the rules to strip. On that traffic the stage does nothing useful while still
mutating the payload (whitespace-collapsed content) and running a reconstruction pass — pure cost.

The naive fix (default `features.compression` off) conflicts with the PRD goal and silently
retires the feature for users whose traffic *is* verbose, where it still hits its ≥40% target.

### Decision

Compression is **gated by expected benefit** rather than defaulted off. `compress()` computes the
achievable saving under one tokenizer and, when it is below `compress.min_savings_ratio`
(default **0.02**), forwards the **original** messages untouched with a bare reconstruction
template — a per-request no-op. Terse turns auto-skip; verbose turns (which clear the floor
easily) are compressed as before, so G4's aggregate is unchanged (measured mean 0.4979). The dead
`compress.min_tokens` knob (declared since v1, never read) is removed in the same change; it was
the vestigial, unwired version of this gate.

The `over_compression_blocks` metric is incremented only when a compression is actually used, so
a gated (discarded) rewrite no longer inflates the counter.

### Rationale

Gate-by-benefit honors **both** constraints: the PRD/G4 promise survives for the workload it was
written for, and the dominant real workload stops paying for an inert stage. It needs no default
change and is self-tuning — no operator has to know to flip a flag. `min_savings_ratio: 0.0`
restores "compress on any positive saving"; a high value effectively disables compression.

### Consequences

`compress.min_savings_ratio` is a new config key (default 0.02). `compress.min_tokens` is gone;
`compress` uses `extra="forbid"`, so a `cradle.yaml` that still lists `min_tokens` now fails to
load — remove the line (RUNBOOK §1.3 covers the unknown-key startup failure). No `pipeline_version`
bump: the cache key hashes the canonical (uncompressed) text, not the compressed payload, so gating
a request does not change what it caches or reads. This ADR does not settle whether rule-based
compression should eventually be replaced by a real neural compressor for long prompts — that
remains future work.

---

## ADR-0008: Streaming single-flight — coalesce concurrent identical cacheable misses

**Date:** 2026-09-19
**Status:** Accepted (v1 default OFF)
**Phase:** Performance / cost
**Deciders:** Greg

### Context

A burst of identical cacheable prompts arriving before the first writeback (eval loops, a bot
under load, shared-key clients) pays upstream N times and writes back N times. `hash_input`
excludes `stream`, so JSON and streaming callers already share an L1 cache key. The industry ships
non-streaming single-flight only; Cradle's local-frame wrap contract (every wrap-path frame is
synthesized locally with a local `id`/`created`) makes the streaming case natural.

### Decision

An in-flight registry `Runtime.flights: dict[str, Flight]` keyed on the **L1 key plus stream-ness**.
The first cacheable, non-bypass, non-#43-passthrough, non-`no-cache`, non-`no-store`, non-probe miss
is the leader; identical concurrent arrivals are followers that replay the leader's buffered frames
then live-tail (stream) or await its completion dict (JSON). Only the leader calls upstream and only
the leader writes back. Followers are marked `X-Cradle-Flight: follower`, count as misses, and carry
`upstream_prompt_tokens = 0`. Coordination is `dict.setdefault` + `asyncio.Event` on the single event
loop — no locks, no threads. The follow-check sits **before** embed/L2 (a follower skips the whole L2
query — `embed_pool` is a single worker, so N followers would serialise on it; a fresh leader answer
beats an L2 near-match). The leader registers **after** the probe return, so a probe never registers a
flight it would not resolve.

The flight key includes stream-ness because a **JSON leader publishes no frames** (only a completion
dict), so a stream follower behind a JSON leader would replay nothing. **Cross-mode coalescing is a v1
non-goal**; the bursts that motivate the feature are homogeneous. Tenant/principal safety is structural:
`hash_input` includes `tenant_id` and `user_id`, so a flight key can never cross tenants or principals.
In default intercept mode the principal derives from the client `Authorization`, so a shared-key burst
coalesces but a distinct-key burst does not (documented, not a bug).

### Leader disconnect (v1)

Leader abort (client disconnect → Starlette throws `GeneratorExit`/`CancelledError` into the response
generator) → followers receive an SSE error frame and `[DONE]`; the flight is failed and removed. The
alternative (leader keeps consuming upstream for followers) needs a detached background task that breaks
the `client_connected` writeback gate and the disconnect accounting — deferred. Tradeoff: followers lose
work the leader already paid for; acceptable because it is rare and a follower can retry and become the
new leader, and correctness (no hung requests, no wrong bytes) beats efficiency here.

### Registry safety (the highest-risk surface)

A leaked flight hangs every later identical prompt forever. Three layers, in order: (1) **resolve-or-fail
in the leader's generator `finally`**, keyed on a single `completion_out` local that only a clean success
sets — every early return / exception fails the flight, so a new exit path added later is safe by default;
(2) **followers await with `upstream.timeout_s + slack`** so any leak degrades to a per-request 504 /
error frame, not an infinite hang; (3) a **stale flight** (older than `upstream.timeout_s`) is replaced,
not joined. The stream leader resolves the flight **before** `resp.aclose()` so a failure there can never
strand followers.

### Default OFF

`cache.singleflight` defaults **false**. The failure mode of a leaked flight is uniquely bad (a hung
request), unlike the other default-on hot-path flags. Flip to default-on after a LAN soak against the
test container shows zero stuck flights and `cradle_flight_followers_total > 0` under a concurrent-identical
burst.

**Do not enable it before that soak.** The three CRITICAL concurrency defects the pre-merge multi-model
review of #58 found — which had merged unfixed — are now **fixed** (they were latent only because the flag
defaults off): **#63** — a stream leader whose upstream fails *on open* now resolve-and-releases its flight
(a point fix in `_miss_stream` plus a register-site raise-guard) instead of leaking it and poisoning the
key; **#64** — the registry pop is now identity-checked (`resolve_and_release()` in `flight.py`, the one
place that mutates `runtime.flights` on leader exit), so a replaced stale leader can no longer delete the
live replacement; **#65** — `is_stale` now measures from `last_progress_at` (bumped on every `publish()`),
not flight creation, so a healthy long or backpressured stream is never treated as leaked, and the stream
follower's timeout is a per-frame progress deadline (`asyncio.wait_for` around each `tail()` step, with the
downstream `yield` outside it so a slow follower client cannot trip its own deadline) rather than a
total-duration bound. Two lower-severity, self-healing gaps remain open and gated off the same soak:
**#67** (an unstarted `_wrap_stream` generator leaks its flight on a client disconnect during
`response.start` — bounded and evicted by staleness, not closeable from the request path) and **#68** (four
HIGH/MEDIUM response-correctness gaps from the same review: post-finish writeback failure, the
follow-check/register burst race, a double-wrapped follower error frame, and JSON followers losing the
leader's upstream status). The soak gate stands regardless: the failure mode of a leaked flight is uniquely bad
(a hung request), so default-on waits on the soak evidence above even now that these bugs are closed.

### Consequences

One upstream call and one writeback per burst; followers count as misses with `upstream_prompt_tokens = 0`
(so `cradle_upstream_prompt_tokens_total` does not over-count). New config key `cache.singleflight`, new
response header `X-Cradle-Flight: follower`, new metrics `cradle_flight_followers_total` /
`cradle_flight_aborts_total{reason}`. New module `gateway/flight.py`. No `pipeline_version` bump — a follower
serves the leader's answer under the shared key, changing neither what caches nor how entries are read. The
#43 tool-stream tee and bypass are structurally excluded (each client needs the raw upstream bytes /
per-response headers); coalescing them is mechanically possible later but has no cache benefit.
