"""Plug-and-play strategies."""
from __future__ import annotations

from .base import Strategy
from .ema_ribbon import EmaRibbon

# Registry so strategies can be selected by name (e.g. from a config/CLI).
REGISTRY: dict[str, type[Strategy]] = {
    EmaRibbon.name: EmaRibbon,
}

__all__ = ["Strategy", "EmaRibbon", "REGISTRY"]
