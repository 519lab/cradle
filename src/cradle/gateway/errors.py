from __future__ import annotations

from fastapi.responses import JSONResponse


def openai_error(message: str, type_: str, code: str, status: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": type_, "code": code}},
        status_code=status,
    )
