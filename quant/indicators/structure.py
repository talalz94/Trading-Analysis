"""
Market structure: swing highs/lows (pivots), lookahead-safe last-swing levels, and classic
(floor-trader) pivot points. The last_swing_* columns are the intended ref_col stop sources.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..data.timeframe import resample_ohlcv


def add_swings(df: pd.DataFrame, left: int = 10, right: int = 10, *,
               prefix: str = "swing") -> pd.DataFrame:
    """Detect pivot highs/lows and expose lookahead-safe rolling swing levels.

    A pivot high at bar p is the local max over [p-left, p+right]; it is only *confirmed*
    `right` bars later. `{prefix}_last_high` / `{prefix}_last_low` forward-fill the last
    confirmed pivot and are safe to use as stop references (no future leakage).
    """
    out = df.copy()
    w = int(left) + int(right) + 1
    high, low = out["high"], out["low"]

    roll_max = high.rolling(w, center=True, min_periods=w).max()
    roll_min = low.rolling(w, center=True, min_periods=w).min()
    is_ph = (high == roll_max)
    is_pl = (low == roll_min)

    ph_price = high.where(is_ph)
    pl_price = low.where(is_pl)

    # confirmed (and thus usable) `right` bars after the pivot bar
    out[f"{prefix}_high"] = is_ph.shift(right, fill_value=False)
    out[f"{prefix}_low"] = is_pl.shift(right, fill_value=False)
    out[f"{prefix}_last_high"] = ph_price.shift(right).ffill()
    out[f"{prefix}_last_low"] = pl_price.shift(right).ffill()
    return out


def add_pivot_points(df: pd.DataFrame, rule: str = "1D", *, prefix: str = "piv") -> pd.DataFrame:
    """Classic floor-trader pivot points from the PRIOR `rule` period (default daily).

    Adds {prefix}_pp/_r1/_s1/_r2/_s2, aligned onto the base grid and shifted one period so
    only the previous period's levels are visible (lookahead-safe).
    """
    out = df.copy()
    agg = resample_ohlcv(out, rule)
    pp = (agg["high"] + agg["low"] + agg["close"]) / 3.0
    r1 = 2 * pp - agg["low"]
    s1 = 2 * pp - agg["high"]
    r2 = pp + (agg["high"] - agg["low"])
    s2 = pp - (agg["high"] - agg["low"])
    lv = pd.DataFrame({
        "open_time": agg["open_time"],
        f"{prefix}_pp": pp, f"{prefix}_r1": r1, f"{prefix}_s1": s1,
        f"{prefix}_r2": r2, f"{prefix}_s2": s2,
    })
    # shift by one period so the CURRENT period sees only the PRIOR period's levels
    for c in [c for c in lv.columns if c != "open_time"]:
        lv[c] = lv[c].shift(1)
    merged = pd.merge_asof(out.sort_values("open_time"), lv.sort_values("open_time"),
                           on="open_time", direction="backward")
    return merged
