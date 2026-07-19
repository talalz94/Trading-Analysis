"""Structured logging + progress-bar helpers, shared across the package."""
from __future__ import annotations

import logging
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


def maybe_tqdm(enabled: bool, total: Optional[int] = None, desc: str = "", unit: str = "it"):
    """Return a tqdm bar if enabled and available, else None (safe no-op)."""
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
        return tqdm(total=total, desc=desc, unit=unit, leave=True)
    except Exception:
        return None
