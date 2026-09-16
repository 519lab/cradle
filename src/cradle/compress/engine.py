from __future__ import annotations

from cradle.compress import structure as structure_mod
from cradle.compress.guards import extract_protected, format_instruction_lines, restore_protected
from cradle.compress.rules import strip_fluff
from cradle.config import Settings
from cradle.gateway.models import ChatMessage, CompressedPrompt, ReconstructionTemplate
from cradle.metrics.prometheus import over_compression_blocks
from cradle.tokens import count_chat_prompt


def _rewrite_user(text: str, settings: Settings) -> tuple[str, int, list[str]]:
    placeholders, masked, spans = extract_protected(text)
    for s in spans:
        over_compression_blocks.labels(reason=s.reason if s.reason in {"code", "json", "format"} else "code").inc()
    stripped = strip_fluff(masked)
    if settings.features.structure:
        stripped = structure_mod.to_dense(stripped, settings.compress.structure_min_chars)
    restored = restore_protected(stripped, placeholders)
    return restored, len(spans), format_instruction_lines(spans)


def compress(messages: list[ChatMessage], settings: Settings, model: str) -> CompressedPrompt:
    inbound = count_chat_prompt(messages, model)
    template = ReconstructionTemplate(mode=settings.reconstruct.mode)
    if not settings.features.compression:
        return CompressedPrompt(
            messages=messages,
            template=template,
            inbound_tokens=inbound,
            compressed_tokens=inbound,
            savings_ratio=0.0,
            protected_span_count=0,
        )
    out: list[ChatMessage] = []
    protected = 0
    for m in messages:
        if m.role != "user" or not isinstance(m.content, str):
            out.append(m)
            continue
        rewritten, n, fmt = _rewrite_user(m.content, settings)
        protected += n
        template.format_instructions.extend(fmt)
        dumped = m.model_dump()
        dumped["content"] = rewritten
        out.append(ChatMessage.model_validate(dumped))
    compressed = count_chat_prompt(out, model)
    savings = 0.0 if inbound <= 0 else max(0.0, (inbound - compressed) / inbound)
    return CompressedPrompt(
        messages=out,
        template=template,
        inbound_tokens=inbound,
        compressed_tokens=compressed,
        savings_ratio=savings,
        protected_span_count=protected,
    )
