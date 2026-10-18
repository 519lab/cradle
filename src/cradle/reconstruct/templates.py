from __future__ import annotations

from cradle.config import Settings
from cradle.gateway.models import ReconstructionTemplate


def template_for(settings: Settings, tenant_id: str) -> ReconstructionTemplate:
    mode = settings.reconstruct.mode if settings.features.reconstruction else "passthrough"
    prefix = ""
    suffix = ""
    tenant = settings.reconstruct.tenants.get(tenant_id)
    if tenant and mode == "wrap":
        prefix = tenant.brand_prefix
        suffix = tenant.brand_suffix
    return ReconstructionTemplate(mode=mode, brand_prefix=prefix, brand_suffix=suffix)
