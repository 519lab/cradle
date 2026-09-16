from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from cradle.cache.records import CanonicalRequest, Principal


@dataclass
class RequestContext:
    request_id: str
    principal: Principal
    canonical: CanonicalRequest | None = None
    layer_hit: Literal["l1", "l2", "miss", "bypass"] = "miss"
    l2_score: float | None = None
    inbound_prompt_tokens: int = 0
    upstream_prompt_tokens: int = 0
    t_l1_s: float = 0.0
    t_embed_s: float = 0.0
    t_l2_s: float = 0.0
    t_compress_s: float = 0.0
    t_upstream_s: float = 0.0
    t_reconstruct_s: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)
