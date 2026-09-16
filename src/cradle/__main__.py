from __future__ import annotations

import uvicorn

from cradle.config import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "cradle.app:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=1,
        factory=False,
    )


if __name__ == "__main__":
    main()
