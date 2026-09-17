from __future__ import annotations

import fnmatch

from cradle.config import Settings, UpstreamSettings


def resolve_upstream(settings: Settings, model: str) -> tuple[str, UpstreamSettings]:
    for rule in settings.routes:
        if fnmatch.fnmatch(model, rule.model):
            if rule.to == "default":
                return "default", settings.upstream
            return rule.to, settings.upstreams[rule.to]
    return "default", settings.upstream


def advertised_models(settings: Settings) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in settings.upstream.models:
        if name not in seen:
            seen.add(name)
            out.append(name)
    for target in settings.upstreams.values():
        for name in target.models:
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out
