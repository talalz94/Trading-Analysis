"""
Generic incremental Parquet cache for range-based data providers.

Any provider that can fetch an arbitrary [start, end] window can become incremental + cached by
routing through `incremental_fetch`: it loads the existing cache, fetches only the missing head/
tail ranges, merges, de-dups, writes, and returns the full cached frame. (Internal weekend gaps
are left alone — for FX/metals they are real market closures, not missing data.)

The Binance provider keeps its own richer fetcher (`_binance_fetch.py`); this helper is for new
providers like Dukascopy / OANDA.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

from .base import TIME_COL, validate_ohlcv

RawFetch = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


def _interval_delta(interval: str) -> pd.Timedelta:
    n = int("".join(c for c in interval if c.isdigit()) or "1")
    unit = interval[-1]
    return {"m": pd.Timedelta(minutes=n), "h": pd.Timedelta(hours=n),
            "d": pd.Timedelta(days=n), "w": pd.Timedelta(weeks=n)}.get(unit, pd.Timedelta(minutes=n))


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
        return df
    except Exception:
        return pd.DataFrame()


def incremental_fetch(
    raw_fetch: RawFetch,
    path: Path,
    start: str,
    end: Optional[str],
    interval: str,
    *,
    source: str = "",
    logger=None,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else None
    delta = _interval_delta(interval)

    existing = _read(path)
    ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    eff_end = end_ts if end_ts is not None else pd.Timestamp.now(tz="UTC")

    if existing.empty:
        ranges.append((start_ts, eff_end))
    else:
        cmin, cmax = existing[TIME_COL].min(), existing[TIME_COL].max()
        if start_ts < cmin:
            ranges.append((start_ts, cmin - delta))
        if eff_end > cmax:
            ranges.append((cmax + delta, eff_end))

    parts = [existing] if not existing.empty else []
    for s, e in ranges:
        if e < s:
            continue
        if logger:
            logger.info("[%s] fetch %s -> %s", source, s, e)
        got = raw_fetch(s, e)
        if got is not None and not got.empty:
            parts.append(got)

    if not parts:
        return pd.DataFrame()

    merged = pd.concat(parts, ignore_index=True)
    merged[TIME_COL] = pd.to_datetime(merged[TIME_COL], utc=True)
    merged = merged.drop_duplicates(TIME_COL, keep="last").sort_values(TIME_COL).reset_index(drop=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)
    return validate_ohlcv(merged, source=source)
