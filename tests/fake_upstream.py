from __future__ import annotations

import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

fake_app = FastAPI()


def _reply_from(body: dict) -> str:
    messages = body.get("messages") or []
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            return "ACK"
    return "ACK"


@fake_app.post("/v1/chat/completions")
async def completions(request: Request):
    body = await request.json()
    text = _reply_from(body)
    if body.get("stream"):
        async def gen():
            chunk = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": body.get("model") or "fake",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            mid = len(text) // 2 or 1
            for part, finish in ((text[:mid], None), (text[mid:], "stop")):
                chunk = {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": body.get("model") or "fake",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": part},
                            "finish_reason": finish,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")
    return JSONResponse(
        {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model") or "fake",
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
