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
    max_body_bytes: int = 1048576


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
    evict_old_pipeline: bool = True
    purge_interval_s: int = 300
    l1_size_limit_bytes: int = 1_000_000_000


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
    # under the L2 p99 budget) but requires the onnxruntime-gpu extra
    # (cradle[rerank-gpu]) and a CUDA host. device_ids selects GPU(s) for cuda.
    rerank_device: str = "cpu"
    rerank_device_ids: list[int] | None = None

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
    min_tokens: int = 16


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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CRADLE_",
        env_nested_delimiter="__",
        extra="forbid",
        nested_model_default_partial_update=True,
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    data_dir: Path = Path("./data")
    pipeline_version: str = "v1"
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
