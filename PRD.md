# Product Requirements Document (PRD)

**Project Name:** Local Semantic Caching & Token Compression Layer (Zero-API Cost Proxy)

**Document Status:** Draft

**Target Architecture:** CPU / Edge-Deployable Middleware

---

## 1. Executive Summary & Goals

### 1.1 Objective

Build an open-source, self-hosted proxy layer positioned directly between user clients and upstream LLM APIs. The system intercepts user queries to perform **zero-cost semantic cache hits** and **token compression** on cache misses—reducing upstream LLM token consumption by 40%–60% while cutting average response times for recurring queries to under 20ms.

### 1.2 Core Capabilities

* **Dual-Layer Caching:** Combined SHA-256 exact matching with local vector similarity search.
* **Client-Side Compression:** Fluff stripping and structural distillation before API execution.
* **Local Response Assembly:** Rebuilding full user-facing responses using locally stored templates and small local models.
* **Zero External Dependencies:** Built entirely on open-source, non-commercial software running locally on CPU hardware.

---

## 2. System Architecture & Component Design

```
+-------------------------------------------------------------------------------+
|                                INBOUND REQUEST                                |
+-------------------------------------------------------------------------------+
                                       |
                                       v
                     +-----------------------------------+
                     |      1. Exact Match Engine        |
                     |       (SHA-256 Lookup)            |
                     +-----------------------------------+
                                       |
                         +-------------+-------------+
                         |                           |
                       [ HIT ]                     [ MISS ]
                         |                           |
                         v                           v
              +---------------------+    +-----------------------+
              | Return Cached Text  |    |  2. Semantic Search   |
              |     (< 2ms)         |    |   (Local Embedding)   |
              +---------------------+    +-----------------------+
                                                     |
                                       +-------------+-------------+
                                       |                           |
                                     [ HIT ]                     [ MISS ]
                                       |                           |
                                       v                           v
                            +---------------------+    +-----------------------+
                            | Return Cached Text  |    | 3. Local Compression  |
                            |     (< 20ms)        |    |    (Regex/NER/1B)     |
                            +---------------------+    +-----------------------+
                                                                   |
                                                                   v
                                                       +-----------------------+
                                                       | 4. Upstream LLM Call  |
                                                       | (Minimal Token Payload|
                                                       +-----------------------+
                                                                   |
                                                                   v
                                                       +-----------------------+
                                                       | 5. Local Response     |
                                                       |    Reconstruction     |
                                                       +-----------------------+
                                                                   |
                                                                   v
                                                       +-----------------------+
                                                       | 6. Update Local Cache |
                                                       +-----------------------+

```

---

## 3. Functional Requirements

### FR-1: Low-Latency Multi-Tier Caching

#### FR-1.1: Exact Match Layer (L1)

* **Mechanism:** Hash the raw, normalized input string using `SHA-256`.
* **Store:** Key-Value Store (Redis or local RocksDB).
* **Latency Budget:** $< 2\text{ ms}$.

#### FR-1.2: Semantic Match Layer (L2)

* **Embedding Execution:** Generate vector embeddings locally on CPU using quantized models (`ONNX` / `FastEmbed`).
* **Default Model:** `BAAI/bge-small-en-v1.5` (384-dimensional) or `all-MiniLM-L6-v2`.
* **Vector Store:** Local Redis Vector Search or embedded Qdrant instance.
* **Cosine Distance Threshold:** Configurable range ($0.88 - 0.93$). Similarity scores above the threshold trigger an immediate cache hit.

---

### FR-2: Input Token Compression

On an L1/L2 cache miss, the proxy modifies the input prompt before forwarding it upstream:

#### FR-2.1: Rule-Based Stripping

* Strip conversational opening/closing boilerplate, pleasantries, and redundant filler words using local regex patterns.

#### FR-2.2: Named Entity & Core Intent Extraction

* Isolate actionable instructions, system prompts, key entities, and variables while preserving original meaning.
* *Optional Engine:* Local `spaCy` NER or an ONNX-quantized small model (`Llama-3.2-1B` / `Qwen-2.5-1.5B` via `llama.cpp`).

#### FR-2.3: Structural Formatting

* Reformat extracted data into ultra-dense JSON or bulleted key-value representations to minimize token footprint.

---

### FR-3: Response Reconstruction & Cache Management

#### FR-3.1: Assembly Layer

* Receive raw, compressed completions from the upstream LLM.
* Merge raw outputs with local layout templates, brand framing, and context metadata locally before returning to the user client.

#### FR-3.2: Cache Writeback & Invalidation

* Write the original raw prompt embedding and full reconstructed response back into the L1/L2 stores.
* Support Time-To-Live (TTL) auto-eviction policy (default: 86,400 seconds / 24 hours).
* Invalidate entries upon changes to system prompts or model pipeline versions.

---

## 4. Technical Stack Specifications

| Component | Technology | Rationale |
| --- | --- | --- |
| **Proxy Gateway** | Python (FastAPI / `uvicorn`) or Rust | High-concurrency async handling for fast pipeline routing. |
| **Cache & Vector Store** | Redis (Docker) / Qdrant (Local Mode) | Unifies key-value storage and vector index search with zero cloud API costs. |
| **Embedding Engine** | `FastEmbed` / `onnxruntime` | CPU-optimized vector generation ($10–20\text{ ms}$ per execution). |
| **Local Model Engine** | `llama.cpp` or `ollama` | Lightweight CPU inference for tiny models ($1\text{B}–3\text{B}$ parameters). |

---

## 5. Key Metrics & Success Criteria

```
                        (Inbound Prompt Tokens - Upstream Tokens Sent)
Token Savings Ratio (%) = ------------------------------------------------  x 100
                                    Inbound Prompt Tokens

```

| Metric | Target |
| --- | --- |
| **L1 Exact Match Latency** | $< 2\text{ ms}$ |
| **L2 Semantic Match Latency** | $< 25\text{ ms}$ (including local embedding CPU execution) |
| **Prompt Token Savings** | $\ge 40\%$ reduction on cache misses |
| **Cache Hit Ratio (Production)** | Target $\ge 25\%$ (environment dependent) |
| **Infrastructure Overhead** | $\$0.00$ external API costs (runs entirely on local hardware) |

---

## 6. Security, Multi-Tenancy, & Risks

### 6.1 Data Isolation & Privacy

* **Tenant Scoping:** Cache keys must be partitioned by `tenant_id` or `user_id` to prevent cross-user data leaks via semantic hits.
* **System Versioning:** Hash signatures must include the `system_prompt_version` so changes in instruction logic automatically skip stale cache results.

### 6.2 Risk Mitigation

* **False Semantic Hits:** Setting cosine similarity thresholds too low ($<0.85$) will lead to inaccurate answers. *Mitigation:* Require strict default thresholds ($\ge 0.90$) and optional 1-shot cross-encoder validation for low-confidence hits.
* **Over-Compression:** Over-aggressive stripping can alter prompt intent. *Mitigation:* Provide fallback rules that preserve complex code blocks, technical syntax, and explicit formatting instructions.
