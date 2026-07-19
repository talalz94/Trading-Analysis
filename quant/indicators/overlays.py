"""
Vectorized overlay indicators (pure compute, no plotting).

Everything here returns pandas Series aligned to the input index. Fully vectorized.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def ema(series: pd.Series, period: int, *, adjust: bool = False) -> pd.Series:
    """Exponential moving average (matches legacy MovingAverage ema: span, adjust=False)."""
    return series.ewm(span=int(period), adjust=adjust, min_periods=int(period)).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=int(period), min_periods=int(period)).mean()


def add_emas(
    df: pd.DataFrame,
    periods: Iterable[int],
    *,
    source: str = "close",
    prefix: str = "ema",
    adjust: bool = False,
) -> pd.DataFrame:
    """Add EMA columns `{prefix}_{p}` for each period. Returns a new frame."""
    out = df.copy()
    src = out[source]
    for p in periods:
        out[f"{prefix}_{int(p)}"] = ema(src, p, adjust=adjust)
    return out


def add_smas(df: pd.DataFrame, periods: Iterable[int], *, source: str = "close",
             prefix: str = "sma") -> pd.DataFrame:
    out = df.copy()
    src = out[source]
    for p in periods:
        out[f"{prefix}_{int(p)}"] = sma(src, p)
    return out
