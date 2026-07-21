"""
Binance data source.

Wraps the proven, battle-tested incremental fetcher in the repo-root `data.py`
(retries/backoff, durable partial checkpoints, gap backfill). We deliberately reuse it
rather than reimplement — it is the most mature part of the legacy codebase.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .base import DataSource, validate_ohlcv


class BinanceSource:
    """DataSource backed by `data.fetch_binance_klines` (public data API, no keys needed)."""

    name = "binance"

    def __init__(self, market: str = "spot", cache_dir: Optional[str] = None):
        self.market = market
        self.cache_dir = cache_dir

    def fetch(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: Optional[str] = None,
        *,
        progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        from ._binance_fetch import fetch_binance_klines

        df = fetch_binance_klines(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            market=self.market,
            cache=True,
            cache_dir=self.cache_dir,
            progress=progress,
            **kwargs,
        )
        return validate_ohlcv(df, source=self.name)


# Registry so callers can select a source by name (extend with oanda/metals/csv later).
_SOURCES: dict[str, type] = {"binance": BinanceSource}
_KNOWN = ["binance", "dukascopy"]


def get_source(name: str = "binance", **kwargs) -> DataSource:
    key = name.lower().strip()
    if key == "dukascopy":
        from .dukascopy import DukascopySource       # lazy: only needs dukascopy-python when used
        return DukascopySource(**kwargs)
    if key not in _SOURCES:
        raise ValueError(f"Unknown data source '{name}'. Available: {_KNOWN}")
    return _SOURCES[key](**kwargs)
