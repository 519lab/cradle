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
