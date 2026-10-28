from __future__ import annotations

from cradle.compress import structure as structure_mod
from cradle.compress.guards import extract_protected, format_instruction_lines, restore_protected
from cradle.compress.rules import strip_fluff
from cradle.config import Settings
from cradle.gateway.models import ChatMessage, CompressedPrompt, ReconstructionTemplate
from cradle.metrics.prometheus import over_compression_blocks
from cradle.tokens import count_chat_prompt


def _span_reason(reason: str) -> str:
    return reason if reason in {"code", "json", "format"} else "code"


def _rewrite_user(text: str, settings: Settings) -> tuple[str, list[str], list[str]]:
    """Rewrite one user message. Returns (text, protected-span reasons, format lines).

    Does NOT touch the over_compression_blocks metric: the caller increments it
    only once the benefit gate decides the compression is actually used, so a
    gated (discarded) rewrite never inflates the counter (#52)."""
    placeholders, masked, spans = extract_protected(text)
    stripped = strip_fluff(masked)
    if settings.features.structure:
        stripped = structure_mod.to_dense(stripped, settings.compress.structure_min_chars)
    restored = restore_protected(stripped, placeholders)
    return restored, [_span_reason(s.reason) for s in spans], format_instruction_lines(spans)


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
    span_reasons: list[str] = []
    for m in messages:
        if m.role != "user" or not isinstance(m.content, str):
            out.append(m)
            continue
        rewritten, reasons, fmt = _rewrite_user(m.content, settings)
        span_reasons.extend(reasons)
        template.format_instructions.extend(fmt)
        dumped = m.model_dump()
        dumped["content"] = rewritten
        out.append(ChatMessage.model_validate(dumped))
    compressed = count_chat_prompt(out, model)
    savings = 0.0 if inbound <= 0 else max(0.0, (inbound - compressed) / inbound)
    if savings < settings.compress.min_savings_ratio:
        # Benefit gate (#52): the strip saved too little to be worth the cost of
        # sending whitespace-collapsed content upstream and running reconstruction.
        # Forward the ORIGINAL messages untouched with a bare template — a true
        # no-op, identical in shape to features.compression=False for this request.
        # Do NOT record over_compression_blocks: this compression was discarded.
        return CompressedPrompt(
            messages=messages,
            template=ReconstructionTemplate(mode=settings.reconstruct.mode),
            inbound_tokens=inbound,
            compressed_tokens=inbound,
            savings_ratio=0.0,
            protected_span_count=0,
        )
    # Compression is used: now (and only now) count the protected spans it handled.
    for reason in span_reasons:
        over_compression_blocks.labels(reason=reason).inc()
    return CompressedPrompt(
        messages=out,
        template=template,
        inbound_tokens=inbound,
        compressed_tokens=compressed,
        savings_ratio=savings,
        protected_span_count=len(span_reasons),
    )
