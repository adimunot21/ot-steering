"""Project-wide logging factory.

All library code uses ``get_logger(__name__)`` rather than ``print``. Scripts
and notebooks may print, but anything imported from ``src/`` should not.

The default handler writes ISO-8601 timestamped lines to stderr at INFO level.
The level can be overridden globally via the ``OT_STEERING_LOG_LEVEL`` env var
(e.g. ``DEBUG``, ``WARNING``).
"""

from __future__ import annotations

import logging
import os
import sys

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"
_ENV_LEVEL_VAR = "OT_STEERING_LOG_LEVEL"


def _resolve_level() -> int:
    """Read the log level from env, defaulting to INFO if unset/invalid."""
    raw = os.environ.get(_ENV_LEVEL_VAR, "INFO").upper()
    level = logging.getLevelName(raw)
    if not isinstance(level, int):
        return logging.INFO
    return level


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    The first call configures a stderr handler on the root project logger
    (``ot_steering``); subsequent calls reuse it. Loggers obtained through this
    helper do not propagate to the Python root logger, so they do not pollute
    third-party log configuration.

    Args:
        name: Typically ``__name__`` of the calling module. Loggers obtained
            for sub-modules inherit from the project root logger.

    Returns:
        A ``logging.Logger`` configured to write to stderr with a timestamp,
        level, and module name.

    Raises:
        ValueError: If ``name`` is empty.
    """
    if not name:
        raise ValueError("logger name must be a non-empty string")

    root = logging.getLogger("ot_steering")
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
        root.addHandler(handler)
        root.setLevel(_resolve_level())
        root.propagate = False

    if name == "ot_steering" or name.startswith("ot_steering."):
        return logging.getLogger(name)
    # External callers (tests, notebooks) get a logger under the project root
    # so their output flows through the same handler.
    return logging.getLogger(f"ot_steering.ext.{name}")
