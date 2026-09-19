from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def _config_path() -> Path:
    raw = os.environ.get("CRADLE_CONFIG", "config/cradle.yaml")
    return Path(raw)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], yaml_file: Path) -> None:
        super().__init__(settings_cls)
        self.yaml_file = yaml_file
        self._data: dict[str, Any] = {}
        if yaml_file.is_file():
            with yaml_file.open(encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config {yaml_file} must be a mapping")
            self._data = loaded

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def prepare_field_value(
        self, field_name: str, field: Any, field_value: Any, value_is_complex: bool
    ) -> Any:
        return field_value

    def __call__(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field_name, field in self.settings_cls.model_fields.items():
            value, _key, is_complex = self.get_field_value(field, field_name)
            if value is not None:
                out[field_name] = self.prepare_field_value(field_name, field, value, is_complex)
        return out


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    host: str = "127.0.0.1"
    port: int = 8000
    # Max request body. 128 MiB: a 1M-token text prompt is ~4-6 MB, but inline
    # base64 images dominate (~30-45 MB for a handful of photos), so the cap is
    # sized for multimodal + long-context traffic while still rejecting
    # gigabyte-scale abuse. A pass-through gateway must not reject requests the
    # upstream would accept. Lower it for text-only deploys (issue #34).
    max_body_bytes: int = 134217728


class AuthKey(BaseModel):
    model_config = ConfigDict(extra="forbid")
    token_env: str
    tenant_id: str
    user_id: str


class AuthSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allow_insecure_loopback: bool = False
    keys: list[AuthKey] = Field(default_factory=list)


class UpstreamSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key_env: str = "CRADLE_UPSTREAM_API_KEY"
    timeout_s: float = 120
    pass_through_client_auth: bool = True
    models_passthrough: bool = False
    models: list[str] = Field(default_factory=lambda: ["gpt-4o-mini"])


class RouteRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    to: str


class FeatureFlags(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cache: bool = True
    compression: bool = True
    structure: bool = False
    l2: bool = True
    # Cross-encoder rerank of L2 candidates (issue #5). Default on: it closes the
    # entity-swap gap the numbers/negation guard cannot. Set false to disable if
    # the reranker misbehaves; L2 then serves on cosine + guard alone.
    l2_rerank: bool = True
    reconstruction: bool = True
    local_1b: bool = False


class CacheSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ttl_s: int = 86400
    max_temperature: float = 1.0
    purge_interval_s: int = 300
    l1_size_limit_bytes: int = 1_000_000_000
    # Volatility guard (cache/volatility.py): prompts that ask about something
    # time-sensitive ("latest version", "current price", "today", weather,
    # news) get volatile_ttl_s instead of ttl_s so a correct-but-stale answer
    # is not replayed for 24 h. 0 = never store volatile prompts. An explicit
    # client X-Cradle-Cache-TTL always wins over the guard.
    volatility_guard: bool = True
    volatile_ttl_s: int = 300
    # Cache tool-enabled STREAMING requests (#43). Without this, a stream+tools
    # request bypasses the cache entirely (the wrap path cannot carry a tool
    # call). With it, such a request is teed verbatim to the client while a copy
    # is accumulated; if the response contains NO tool call it is cached (same
    # representation as the JSON path), otherwise nothing is cached. Non-stream
    # tool requests are cached regardless (unaffected). Default on. NOTE: the
    # volatility guard is a regex over user text and does not inspect tool
    # semantics, so a stateful tool with no time-word ("what's my balance") can
    # serve a stale cached answer under this flag; set false to disable.
    cache_tool_streams: bool = True

    @field_validator("volatile_ttl_s")
    @classmethod
    def _volatile_ttl(cls, v: int) -> int:
        if v < 0:
            raise ValueError("cache.volatile_ttl_s must be >= 0")
        return v


class L2Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = "local"
    collection: str = "cradle_l2"
    cosine_threshold: float = 0.90
    max_messages: int = 2
    embed_timeout_s: float = 2.0
    model: str = "BAAI/bge-small-en-v1.5"
    dim: int = 384
    onnx_threads: int | None = None
    points_warn: int = 20_000
    # Cross-encoder reranker (features.l2_rerank). Threshold is a raw logit from
    # rerank_model; on BAAI/bge-reranker-base, genuine paraphrases score >= ~4.8
    # and entity swaps <= ~3.5, so ~4.0 sits in the gap (calibrated on a small
    # fixture; widen before trusting the exact value). rerank_timeout_s fails
    # open on expiry.
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_threshold: float = 4.0
    rerank_timeout_s: float = 2.0
    # Retrieve the top-K L2 candidates above the cosine floor and serve the first
    # that survives the guard + rerank, instead of only the single nearest
    # neighbor. Raises hit rate without lowering the cosine floor: when the
    # nearest neighbor is a near-miss the guard/rerank rejects, a true paraphrase
    # at rank 2..K can still serve (enhancement #1). 1 = original behavior.
    query_top_k: int = 5
    # Rerank execution device. "cpu" (default) keeps the thin CPU-only deploy and
    # the ~15-40ms/hit cost. "cuda" runs the cross-encoder on a GPU (~2-5ms, back
    # under the L2 p99 budget) but requires the GPU image (docker/Dockerfile.gpu +
    # docker-compose-gpu.yml) or the onnxruntime-gpu extra (cradle[rerank-gpu]) on a
    # CUDA host — "cuda" on the CPU image crash-loops at startup with an actionable
    # error (rerank.py, issue #31). device_ids selects GPU(s) for cuda.
    rerank_device: str = "cpu"
    rerank_device_ids: list[int] | None = None
    # Verified L2 (gateway/audit.py). After serving an L2 hit, with probability
    # audit_rate Cradle also calls upstream in the background and judges
    # whether the fresh answer agrees with the served one. Every audit is a
    # labeled observation: it feeds cradle_l2_audit_total{verdict} (a measured
    # false-hit rate), an optional JSONL log under {data_dir}/audits.jsonl, and
    # the served entry's own floor - an entry judged wrong at similarity s
    # refuses future hits at <= s. A "disagree" also writes the fresh answer
    # back under the querying prompt's key (self-heal). 0.0 = off. Audits
    # spend real upstream calls.
    audit_rate: float = 0.0
    # Judge. "auto" = the cross-encoder reranker when loaded, else embedding
    # cosine. Measured on bge models: the reranker separates same-meaning
    # answers (>= 7.2) from contradictory ones (<= 2.1) cleanly, so 4.0 sits
    # in the gap; bi-encoder cosine does NOT separate them (same 0.83-0.98
    # overlaps wrong 0.73-0.92), so the embed judge is a weak fallback whose
    # high threshold prefers false "disagree" (a miss) over a missed error.
    audit_judge: str = "auto"
    audit_rerank_threshold: float = 4.0
    audit_embed_threshold: float = 0.90
    audit_log: bool = True
    # Include the query/candidate prompt text in the JSONL rows (off: hashes
    # and keys only, per the no-prompt-text-in-logs rule).
    audit_log_text: bool = False

    @field_validator("audit_rate")
    @classmethod
    def _audit_rate(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("l2.audit_rate must be within [0, 1]")
        return v

    @field_validator("audit_judge")
    @classmethod
    def _audit_judge(cls, v: str) -> str:
        if v not in {"auto", "rerank", "embed"}:
            raise ValueError("l2.audit_judge must be 'auto', 'rerank' or 'embed'")
        return v

    @field_validator("rerank_device")
    @classmethod
    def _check_rerank_device(cls, v: str) -> str:
        if v not in {"cpu", "cuda"}:
            raise ValueError("l2.rerank_device must be 'cpu' or 'cuda'")
        return v

    @field_validator("cosine_threshold")
    @classmethod
    def _clamp_cosine(cls, v: float) -> float:
        if v < 0.85:
            raise ValueError("l2.cosine_threshold must be >= 0.85")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in {"local", "server"}:
            raise ValueError("l2.mode must be 'local' or 'server'")
        return v


class CompressSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    structure_min_chars: int = 400
    # Skip compression for a request when the fluff-stripping would save less than
    # this fraction of the prompt's tokens (#52). Rule-based stripping removes ~0
    # from terse turns (no pleasantry/filler) but adds cost — whitespace-collapsed
    # content upstream + a reconstruction pass — so below the floor Cradle forwards
    # the ORIGINAL messages untouched and skips reconstruction. Verbose prompts
    # (the workload compression targets) clear this comfortably. 0.0 = compress on
    # any positive saving; 1.0 = effectively disable compression.
    min_savings_ratio: float = 0.02


class ReconstructTenant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    brand_prefix: str = ""
    brand_suffix: str = ""


class ReconstructSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str = "wrap"
    tenants: dict[str, ReconstructTenant] = Field(default_factory=dict)


class MetricsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    require_auth: bool = True


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Two independent axes (logging_setup.py). `level` is how much detail:
    # INFO gives one request-completion line; DEBUG adds per-candidate
    # guard/rerank/audit-floor rejection lines. `content` is how sensitive:
    # "none" logs only keys, hashes, scores and timings; "prompts" adds the
    # request text; "prompts_and_completions" adds the response text too.
    # content != "none" writes PROMPT/RESPONSE TEXT (PII) to the logs — off by
    # default on purpose (the request-path analogue of l2.audit_log_text).
    level: str = "INFO"
    content: str = "none"
    # Default "text": one human-readable key=value line per request for a
    # person watching stdout. "json" emits a compact JSON object per line for a
    # log aggregator to parse.
    format: str = "text"
    # Truncation ceiling for any logged prompt/response text. max_body_bytes is
    # 128 MiB, so an inline-base64 multimodal prompt or a long completion would
    # otherwise produce an unreadable line and unbounded log growth.
    max_text_chars: int = 2000

    @field_validator("level")
    @classmethod
    def _level(cls, v: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        up = v.upper()
        if up not in allowed:
            raise ValueError(f"logging.level must be one of {sorted(allowed)}")
        return up

    @field_validator("content")
    @classmethod
    def _content(cls, v: str) -> str:
        allowed = {"none", "prompts", "prompts_and_completions"}
        if v not in allowed:
            raise ValueError(f"logging.content must be one of {sorted(allowed)}")
        return v

    @field_validator("format")
    @classmethod
    def _format(cls, v: str) -> str:
        if v not in {"text", "json"}:
            raise ValueError("logging.format must be 'text' or 'json'")
        return v

    @field_validator("max_text_chars")
    @classmethod
    def _max_text_chars(cls, v: int) -> int:
        if v < 0:
            raise ValueError("logging.max_text_chars must be >= 0")
        return v


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CRADLE_",
        env_nested_delimiter="__",
        extra="forbid",
        nested_model_default_partial_update=True,
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    data_dir: Path = Path("./data")
    # Bumped v2→v3 with the #40 embed_text fix: L2 vectors and records were computed
    # from system+user text, but the query now embeds user/assistant turns only.
    # Mixing a user-only query against system+user records at the vector/guard/rerank
    # stages is unpredictable, so the bump makes pre-fix entries miss and age out on
    # TTL. (v1→v2 was the wrap-stream usage fix.) pipeline_version folds into the L1
    # key and the L2 filter.
    pipeline_version: str = "v3"
    auth: AuthSettings = Field(default_factory=AuthSettings)
    upstream: UpstreamSettings = Field(default_factory=UpstreamSettings)
    upstreams: dict[str, UpstreamSettings] = Field(default_factory=dict)
    routes: list[RouteRule] = Field(default_factory=list)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    l2: L2Settings = Field(default_factory=L2Settings)
    compress: CompressSettings = Field(default_factory=CompressSettings)
    reconstruct: ReconstructSettings = Field(default_factory=ReconstructSettings)
    metrics: MetricsSettings = Field(default_factory=MetricsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def _flat_env_overrides(self) -> Settings:
        flat_upstream = os.environ.get("CRADLE_UPSTREAM_BASE_URL")
        if flat_upstream:
            self.upstream.base_url = flat_upstream.rstrip("/")
        else:
            self.upstream.base_url = self.upstream.base_url.rstrip("/")
        data_dir = os.environ.get("CRADLE_DATA_DIR")
        if data_dir:
            self.data_dir = Path(data_dir)
        for rule in self.routes:
            if rule.to != "default" and rule.to not in self.upstreams:
                raise ValueError(f"route model={rule.model!r} references unknown upstream {rule.to!r}")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_source = YamlConfigSettingsSource(settings_cls, _config_path())
        return (env_settings, init_settings, yaml_source)


def load_settings(**overrides: Any) -> Settings:
    return Settings(**overrides)
