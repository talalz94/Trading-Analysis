"""Structured logging + progress-bar helpers, shared across the package."""
from __future__ import annotations

import logging
import re
from typing import Optional

_FMT = "[%(asctime)s] %(levelname)s | %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str = "quant", level: str = "INFO") -> logging.Logger:
    """Return a configured logger that doesn't double-attach handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    return logger


_SECRET = re.compile(r"(?i)\b(api_?key|access_token|token)=[^&\s'\"<>)]+")


def redact(text) -> str:
    """`text` with every key or token in a URL blanked ("api_key=***"). A failed FRED download's
    message carries the whole URL, api_key and all, and messages are written to logs and state
    files -- where a secret must never sit (2 Nov 2026: the key was in data/news/update.log)."""
    return _SECRET.sub(lambda m: f"{m.group(1)}=***", str(text))


class RedactFilter(logging.Filter):
    """A log handler's filter: the message and any traceback leave with their secrets blanked."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact(record.getMessage()), None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def maybe_tqdm(enabled: bool, total: Optional[int] = None, desc: str = "", unit: str = "it"):
    """Return a tqdm bar if enabled and available, else None (safe no-op)."""
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit=unit, leave=True)
    except Exception:
        return None
