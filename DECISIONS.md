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
