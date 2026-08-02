"""Volatility indicators: ATR (Wilder) and Supertrend (recursive part in numba)."""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def w(f):
            return f
        return w(a[0]) if a and callable(a[0]) else w


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


@njit(cache=True)
def _supertrend(close, upperband, lowerband):
    n = close.shape[0]
    st = np.empty(n, np.float64)
    direction = np.empty(n, np.int8)  # +1 uptrend, -1 downtrend
    fub = upperband.copy()
    flb = lowerband.copy()
    for i in range(n):
        if i == 0:
            st[i] = upperband[i]
            direction[i] = -1
            continue
        if close[i - 1] <= fub[i - 1]:
            if upperband[i] < fub[i - 1]:
                fub[i] = upperband[i]
            else:
                fub[i] = fub[i - 1]
        if close[i - 1] >= flb[i - 1]:
            if lowerband[i] > flb[i - 1]:
                flb[i] = lowerband[i]
            else:
                flb[i] = flb[i - 1]
        # direction
        if direction[i - 1] == -1 and close[i] > fub[i - 1]:
            direction[i] = 1
        elif direction[i - 1] == 1 and close[i] < flb[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
        st[i] = flb[i] if direction[i] == 1 else fub[i]
    return st, direction


def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0,
                   *, prefix: str = "st") -> pd.DataFrame:
    """Adds {prefix} (line), {prefix}_dir (+1/-1), {prefix}_up (bool uptrend)."""
    out = df.copy()
    a = atr(out, period).to_numpy(np.float64)
    hl2 = ((out["high"] + out["low"]) / 2.0).to_numpy(np.float64)
    upper = hl2 + multiplier * a
    lower = hl2 - multiplier * a
    close = out["close"].to_numpy(np.float64)
    upper = np.nan_to_num(upper, nan=np.inf)
    lower = np.nan_to_num(lower, nan=-np.inf)
    st, direction = _supertrend(close, upper, lower)
    out[prefix] = st
    out[f"{prefix}_dir"] = direction
    out[f"{prefix}_up"] = direction == 1
    return out


def add_atr(df: pd.DataFrame, period: int = 14, *, prefix: str = "atr") -> pd.DataFrame:
    out = df.copy()
    out[f"{prefix}_{int(period)}"] = atr(out, period)
    return out


def add_compression(df: pd.DataFrame, *, short: int = 60, long: int = 480, nr_k: int = 120,
                    rank_window: int = 2000, prefix: str = "cmp") -> pd.DataFrame:
    """VOLATILITY COMPRESSION — how quiet is right now, relative to this instrument's own recent past.

    Five independent measures, because "stagnant" has no single definition and a result that only
    exists under one of them is a parameter artifact:

      {prefix}_range_ratio  range of the last `short` bars / range of the `long` bars before it
      {prefix}_atr_ratio    ATR(short) / ATR(long)
      {prefix}_nr           1.0 when the last `short`-bar range is the NARROWEST of the last `nr_k`
                            bars (the NR7 idea, generalised), else its rank in [0, 1]
      {prefix}_bbw          Bollinger width (2sd) / mid  — the classic squeeze gauge
      {prefix}_pct          PERCENTILE RANK of `atr_ratio` in its own trailing `rank_window`

    Causality: every window looks strictly backwards, the two range windows do not overlap, and the
    percentile rank is shifted one bar so a threshold never uses the bar it is judging. `_pct` is the
    one to condition on — an absolute ATR cut silently becomes a date filter, because ATR scales with
    price and with the era.
    """
    out = df.copy()
    s, l = int(short), int(long)
    hi, lo, c = out["high"], out["low"], out["close"]

    rng_s = hi.rolling(s, min_periods=s).max() - lo.rolling(s, min_periods=s).min()
    # the PRIOR block, shifted so the two windows never overlap
    rng_l = (hi.rolling(l, min_periods=l).max() - lo.rolling(l, min_periods=l).min()).shift(s)
    out[f"{prefix}_range_ratio"] = (rng_s / rng_l.replace(0, np.nan)).to_numpy()

    a_s, a_l = atr(out, s), atr(out, l)
    ratio = a_s / a_l.replace(0, np.nan)
    out[f"{prefix}_atr_ratio"] = ratio.to_numpy()

    out[f"{prefix}_nr"] = rng_s.rolling(int(nr_k), min_periods=int(nr_k) // 4) \
                               .rank(pct=True).to_numpy()

    mid = c.rolling(s, min_periods=s).mean()
    sd = c.rolling(s, min_periods=s).std(ddof=0)
    out[f"{prefix}_bbw"] = (4.0 * sd / mid.replace(0, np.nan)).to_numpy()

    out[f"{prefix}_pct"] = ratio.rolling(int(rank_window), min_periods=int(rank_window) // 4) \
                                .rank(pct=True).shift(1).to_numpy()
    return out
