"""
Fast local Parquet store.

Loads cached OHLCV with **column projection** and **timestamp predicate pushdown** via
polars `scan_parquet`, so a run only materializes the columns and date range it needs
(low memory). Falls back to pandas/pyarrow if polars is unavailable.

Cache file naming matches the legacy layout so existing data/ files work unchanged:
    data/binance_{market}_{SYMBOL}_{interval}.parquet
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from ..config import SETTINGS
from .base import BASE_COLS, TIME_COL, validate_ohlcv


def cache_path(
    symbol: str,
    interval: str,
    *,
    source: str = "binance",
    market: str = "spot",
    data_dir: Union[str, Path, None] = None,
) -> Path:
    d = Path(data_dir) if data_dir is not None else SETTINGS.data_dir
    return d / f"{source}_{market}_{symbol.upper()}_{interval}.parquet"


def exists(symbol: str, interval: str, **kw) -> bool:
    return cache_path(symbol, interval, **kw).exists()


def cache_range(symbol: str, interval: str, **kw):
    """Return (min_open_time, max_open_time) of the cache as tz-aware UTC, or (None, None)."""
    path = cache_path(symbol, interval, **kw)
    if not path.exists():
        return None, None
    try:
        s = pd.read_parquet(path, columns=[TIME_COL])[TIME_COL]
        if s.empty:
            return None, None
        return pd.to_datetime(s.min(), utc=True), pd.to_datetime(s.max(), utc=True)
    except Exception:
        return None, None


def load(
    symbol: str,
    interval: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    columns: Optional[Sequence[str]] = None,
    source: str = "binance",
    market: str = "spot",
    data_dir: Union[str, Path, None] = None,
) -> pd.DataFrame:
    """Load cached OHLCV(+features) with column/date pushdown. Raises if not cached."""
    path = cache_path(symbol, interval, source=source, market=market, data_dir=data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No cache at {path}. Fetch it first via quant.data.loader.get_ohlcv(...)."
        )

    want = list(columns) if columns is not None else None
    if want is not None:
        for c in BASE_COLS:
            if c not in want:
                want.append(c)

    df = _load_polars(path, start, end, want)
    if df is None:
        df = _load_pandas(path, start, end, want)

    return validate_ohlcv(df, source=source)


def _load_polars(path, start, end, want) -> Optional[pd.DataFrame]:
    try:
        import polars as pl
    except Exception:
        return None

    lf = pl.scan_parquet(path)
    schema_names = lf.collect_schema().names()

    if want is not None:
        want = [c for c in want if c in schema_names]
        lf = lf.select(want)

    if start is not None or end is not None:
        tcol = pl.col(TIME_COL)
        # Cast the (tz-aware) timestamp bounds to match the column dtype at predicate time.
        if start is not None:
            lf = lf.filter(tcol >= pl.lit(pd.Timestamp(start, tz="UTC")).cast(pl.Datetime("ns", "UTC")))
        if end is not None:
            lf = lf.filter(tcol <= pl.lit(pd.Timestamp(end, tz="UTC")).cast(pl.Datetime("ns", "UTC")))

    return lf.collect().to_pandas()


def _load_pandas(path, start, end, want) -> pd.DataFrame:
    # pyarrow column projection; date filter applied after load.
    df = pd.read_parquet(path, columns=want)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    if start is not None:
        df = df[df[TIME_COL] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df[TIME_COL] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)
