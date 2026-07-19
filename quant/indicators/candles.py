"""Heikin Ashi transform (recursive ha_open computed in a compiled loop)."""
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


@njit(cache=True)
def _ha_open(ha_close, open0):
    n = ha_close.shape[0]
    ha_open = np.empty(n, np.float64)
    if n == 0:
        return ha_open
    ha_open[0] = open0
    for i in range(1, n):
        ha_open[i] = 0.5 * (ha_open[i - 1] + ha_close[i - 1])
    return ha_open


def add_heikin_ashi(df: pd.DataFrame, *, prefix: str = "ha") -> pd.DataFrame:
    """Adds {prefix}_open/high/low/close and {prefix}_green (bool: ha_close > ha_open)."""
    out = df.copy()
    o = out["open"].to_numpy(np.float64)
    h = out["high"].to_numpy(np.float64)
    l = out["low"].to_numpy(np.float64)
    c = out["close"].to_numpy(np.float64)

    ha_close = (o + h + l + c) / 4.0
    open0 = 0.5 * (o[0] + c[0]) if len(o) else 0.0
    ha_open = _ha_open(ha_close, open0)
    ha_high = np.maximum(h, np.maximum(ha_open, ha_close))
    ha_low = np.minimum(l, np.minimum(ha_open, ha_close))

    out[f"{prefix}_open"] = ha_open
    out[f"{prefix}_high"] = ha_high
    out[f"{prefix}_low"] = ha_low
    out[f"{prefix}_close"] = ha_close
    out[f"{prefix}_green"] = ha_close > ha_open
    return out
