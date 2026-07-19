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
