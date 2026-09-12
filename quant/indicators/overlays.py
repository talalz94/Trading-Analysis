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


def add_mtf_emas(base_df, frames=None, pairs=None, *, time_col: str = "t", source: str = "close"):
    """`{tf}__ema_{period}` columns on the base grid, one timeframe at a time.

    `build_mtf` can do this by RESAMPLING the base frame, which is fine until the period is long:
    an EMA200 wants 200 bars of ITS OWN timeframe, and a base frame that does not reach back that
    far silently yields NaN. Passing the frames from `quant.data.tf_frames` uses the history they
    were loaded with instead.

    `pairs` is `{tf: [periods]}`. The key `""` means the base frame itself, whose EMAs are added
    unprefixed (`ema_200`). Higher timeframes go through `align_timeframes(shift_bars=1)`, so a 15m
    EMA is invisible to base bars until that 15m candle closed.
    """
    from ..data.timeframe import align_timeframes

    out = base_df
    own = list((pairs or {}).get("", []))
    if own:
        out = add_emas(out, own, source=source)
    others = {}
    for tf, periods in (pairs or {}).items():
        if tf == "" or not periods:
            continue
        f = (frames or {}).get(tf)
        if f is None:
            raise KeyError(f"add_mtf_emas: no frame for {tf!r}. Load it with quant.data.tf_frames, "
                           f"or the EMA would have to be resampled from a base frame that may not "
                           f"reach back {max(periods)} of its bars")
        others[tf] = add_emas(f.reset_index(drop=True), sorted(set(periods)), source=source)
    if not others:
        return out
    return align_timeframes(out, others, time_col=time_col, shift_bars=1)
