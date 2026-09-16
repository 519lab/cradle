from __future__ import annotations

from cradle.cache.records import CacheRecord
from cradle.gateway.sse import StreamAccumulator, encode_done, parse_and_accumulate, synthesize_sse


def test_encode_done_literal() -> None:
    assert encode_done() == b"data: [DONE]\n\n"
    assert b'"[DONE]"' not in encode_done()


def test_combined_content_and_finish() -> None:
    acc = StreamAccumulator(outbound_id="id", outbound_created=1, model="m")
    line = (
        'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}'
    )
    piece = parse_and_accumulate(line, acc)
    assert piece == "hi"
    assert acc.finish_reason == "stop"


def test_done_not_forwarded() -> None:
    acc = StreamAccumulator(outbound_id="id", outbound_created=1, model="m")
    assert parse_and_accumulate("data: [DONE]", acc) is None
    assert acc.saw_done is True


def test_synthesize_has_done_and_finish_after_content() -> None:
    rec = CacheRecord(
        key="k",
        tenant_id="t",
        user_id="u",
        model="m",
        system_prompt_version="none",
        pipeline_version="v1",
        prompt_hash="k",
        embed_text_hash="e",
        response={
            "id": "id1",
            "created": 1,
            "model": "m",
            "choices": [
                {"message": {"role": "assistant", "content": "abcdef"}, "finish_reason": "stop"}
            ],
        },
        created_at=1,
        expires_at=10,
        ttl_s=9,
        inbound_prompt_tokens=1,
        upstream_prompt_tokens=0,
        sampling_fingerprint="s",
        temperature=1.0,
        top_p=1.0,
    )
    chunks = list(synthesize_sse(rec, include_usage=False))
    joined = b"".join(chunks)
    assert b"data: [DONE]\n\n" in joined
    text = joined.decode()
    finish_at = text.rfind("finish_reason")
    last_content = text.rfind('"content"')
    assert finish_at > last_content
