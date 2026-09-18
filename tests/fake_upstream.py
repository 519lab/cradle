from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

fake_app = FastAPI()
last_authorization: str | None = None
last_payload: dict | None = None  # side channel: the last forwarded request body


def _reply_from(body: dict) -> str:
    messages = body.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "user":
            return "ACK"
    return "ACK"


@fake_app.post("/v1/chat/completions")
async def completions(request: Request):
    global last_authorization, last_payload
    last_authorization = request.headers.get("authorization")
    body = await request.json()
    last_payload = body
    model = body.get("model") or "fake"
    if model == "fail-401":
        return JSONResponse(
            {"error": {"message": "nope", "type": "invalid_request_error", "code": "invalid_api_key"}},
            status_code=401,
        )
    if model == "fail-429":
        return JSONResponse(
            {"error": {"message": "slow down", "type": "rate_limit_error", "code": "rate_limited"}},
            status_code=429,
            headers={
                "retry-after": "30",
                "x-ratelimit-remaining-requests": "0",
                "x-request-id": "req_upstream_abc",
                "content-length": "999",  # must NOT be forwarded (would corrupt the body)
            },
        )
    if body.get("n", 1) == 2 and not body.get("stream"):
        text = _reply_from(body)
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"},
                    {"index": 1, "message": {"role": "assistant", "content": "ALT"}, "finish_reason": "stop"},
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
    text = _reply_from(body)
    if body.get("stream"):
        want_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        async def gen():
            if model == "err-stream":
                # Mid-stream upstream error object (real provider message/type/code).
                yield 'data: {"error":{"message":"upstream boom","type":"rate_limit_error","code":"rate_limited"}}\n\n'
                return
            if model == "trunc-stream":
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": "half"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                return
            if model == "tools-plain-stream":
                # A tools request the model answers in TEXT (no tool call) — the
                # #43 cacheable case. Emits plain content despite tools present.
                for delta, fin in (({"role": "assistant"}, None), ({"content": "ACK"}, None), ({}, "stop")):
                    ch = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                          "created": 1, "model": model,
                          "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                    yield f"data: {json.dumps(ch)}\n\n"
                yield "data: [DONE]\n\n"
                return
            if model == "tools-mixed-stream":
                # Adversarial (#43): real content AND a tool_call delta, finishing
                # with stop. Must NOT be cached — proves detection isn't just the
                # finish-reason gate.
                yield f'data: {json.dumps({"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})}\n\n'
                yield f'data: {json.dumps({"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{"content":"here"},"finish_reason":None}]})}\n\n'
                yield f'data: {json.dumps({"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"x","arguments":"{}"}}]},"finish_reason":None}]})}\n\n'
                yield f'data: {json.dumps({"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]})}\n\n'
                yield "data: [DONE]\n\n"
                return
            if model == "reasoning-stream":
                # A delta field outside {role,content,tool_calls} (#43 allowlist):
                # reaches the client but can't be replayed from cache → not cached.
                yield f'data: {json.dumps({"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":None}]})}\n\n'
                yield f'data: {json.dumps({"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{"reasoning":"thinking..."},"finish_reason":None}]})}\n\n'
                yield f'data: {json.dumps({"id":"chatcmpl-fake","object":"chat.completion.chunk","created":1,"model":model,"choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":"stop"}]})}\n\n'
                yield "data: [DONE]\n\n"
                return
            if body.get("tools"):
                role = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(role)}\n\n"
                tc = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "x", "arguments": "{}"},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(tc)}\n\n"
                fin = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                }
                yield f"data: {json.dumps(fin)}\n\n"
                yield "data: [DONE]\n\n"
                return
            chunk = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "system_fingerprint": "fp_fake_123",
                "service_tier": "default",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            mid = len(text) // 2 or 1
            for part, finish in ((text[:mid], None), (text[mid:], "stop")):
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"content": part}, "finish_reason": finish}
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            if want_usage:
                # OpenAI emits a final usage-only chunk (empty choices) when
                # stream_options.include_usage is set.
                usage_chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": model,
                    "system_fingerprint": "fp_fake_123",
                    "service_tier": "default",
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        stream_headers = {"x-ratelimit-remaining-requests": "5", "x-request-id": "req_bypass_1"}
        return StreamingResponse(
            gen(), media_type="text/event-stream", headers=stream_headers
        )
    return JSONResponse(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )


@fake_app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "fake", "object": "model", "owned_by": "fake"}]}
