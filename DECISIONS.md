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
