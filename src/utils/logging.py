"""Structured logging setup."""
from __future__ import annotations

import logging
import os
import threading

_lock = threading.Lock()
_configured = False


def setup_logging(level: str = "INFO") -> None:
    global _configured
    with _lock:
        if _configured:
            return
        level_int = getattr(logging, level.upper(), logging.INFO)
        fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        try:
            from rich.logging import RichHandler
            handler: logging.Handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                log_time_format="[%X]",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(fmt))
        root = logging.getLogger()
        if root.handlers:
            root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level_int)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    env_level = os.environ.get("MOCAP_LOG_LEVEL", "INFO")
    setup_logging(env_level)
    return logging.getLogger(name)
