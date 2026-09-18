from __future__ import annotations

import uvicorn

from cradle.config import load_settings
from cradle.logging_setup import configure_logging


def main() -> None:
    settings = load_settings()
    # Install the cradle-logger handler BEFORE uvicorn starts, so request-path
    # info/debug lines escape the process instead of being dropped at WARNING.
    configure_logging(settings)
    uvicorn.run(
        "cradle.app:app",
        host=settings.server.host,
        port=settings.server.port,
        workers=1,
        factory=False,
    )


if __name__ == "__main__":
    main()
