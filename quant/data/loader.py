"""
High-level data access: `get_ohlcv(...)`.

Fast path: if the symbol/interval is already cached, load the requested slice via the
Parquet store (column projection + date pushdown). Slow path: delegate to the selected
DataSource, which fetches only missing ranges (incremental) and updates the cache.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd

from ..config import SETTINGS
from ..logging_utils import get_logger
from . import store
from .binance import get_source

_log = get_logger("quant.data")


def get_ohlcv(
    symbol: str,
    interval: str,
    start: str,
    end: Optional[str] = None,
    *,
    source: str = "binance",
    market: str = "spot",
    tz: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    refresh: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Return tidy OHLCV for [start, end], adding a display-tz column `t`.

    refresh=False and a warm cache => pushdown load (fast, low memory).
    refresh=True or cold cache     => incremental fetch via the source.
    """
    cached = store.exists(symbol, interval, source=source, market=market)
    covered = cached and not refresh and _cache_covers(
        symbol, interval, start, end, source=source, market=market)

    if covered:
        _log.info("cache HIT  %s %s %s -> pushdown load", source, symbol, interval)
        df = store.load(
            symbol, interval, start=start, end=end,
            columns=columns, source=source, market=market,
        )
    else:
        why = "REFRESH" if (cached and refresh) else ("EXTEND" if cached else "MISS")
        _log.info("cache %s %s %s %s -> incremental fetch",
                  why, source, symbol, interval)
        src = get_source(source, market=market)
        full = src.fetch(symbol, interval, start=start, end=end, progress=progress)
        df = _slice(full, start, end)

    tzname = tz or SETTINGS.display_tz
    df = df.copy()
    df["t"] = df["open_time"].dt.tz_convert(tzname)
    return df


def _cache_covers(symbol, interval, start, end, *, source, market) -> bool:
    """True only if the cache already spans the requested [start, end] window.

    Prevents the silent-truncation bug where a warm cache is returned even though the request
    extends beyond it — in that case we fall through to the incremental fetch to fill the gap.
    """
    from .cache import _interval_delta
    cmin, cmax = store.cache_range(symbol, interval, source=source, market=market)
    if cmin is None:
        return False
    delta = _interval_delta(interval)
    req_start = pd.Timestamp(start, tz="UTC")
    if req_start < cmin:
        return False                                   # need earlier data
    if end is None:
        # "latest": covered only if the cache is fresh (within a couple of bars of now)
        return pd.Timestamp.now(tz="UTC") <= cmax + 2 * delta
    req_end = pd.Timestamp(end, tz="UTC")
    return req_end <= cmax + delta                     # need later data if beyond the last cached bar


def _slice(df: pd.DataFrame, start: str, end: Optional[str]) -> pd.DataFrame:
    m = df["open_time"] >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        m &= df["open_time"] <= pd.Timestamp(end, tz="UTC")
    return df[m].reset_index(drop=True)
