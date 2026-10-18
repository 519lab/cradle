from __future__ import annotations

import sys

import uvicorn

from cradle.config import ConfigError, load_settings
from cradle.logging_setup import configure_logging


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        # An operator-actionable config problem (stale key after an upgrade, #61):
        # print one message and exit, not a repeating traceback wall. Logging is
        # not configured yet, so write straight to stderr.
        print(f"cradle: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
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
