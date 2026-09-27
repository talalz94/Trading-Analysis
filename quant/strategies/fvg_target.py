"""
A GAP NEAR THE TARGET -- close the trade when price fills it, rather than waiting for the target.

Price is often drawn to an unfilled fair value gap and turns once it has filled it. When such a gap
sits just short of a trade's target, the target is a few points further than where the move is
likely to stop, and the trade can come within reach of it and still turn into a loss. This names
the gap, per trade, at the bar the trade would open:

    for a SHORT, a live BULLISH gap below the entry whose band lies within `near_r` R of the target
    (either side of it); for a LONG, the mirror -- a live BEARISH gap above.

If several qualify, the one NEAREST THE ENTRY wins, because price reaches it first and the rule is
"the first gap near the target that gets filled". The level returned is where the exit sits:
`at="touch"` the gap's near edge, `"ce"` its midpoint, `"full"` its far edge.

WHY A COLUMN, NOT AN EXIT SIGNAL. The target differs per trade -- it is R multiples of that trade's
own stop -- so "a gap near the target" is a question about one trade, not about a bar. The column
holds, for every bar, the level a trade opened ON that bar would use; the engine reads it once at
entry (`BacktestConfig.tp_ref_short_col`) and carries it with the trade. Gaps that form AFTER the
entry are not considered: the trade is committed to what was knowable when it opened.

The stop is recomputed here from the same reference column the stop method uses, with the same
buffer, so the R this measures in is the trade's R to within the entry fill's slippage -- close
enough for "within 1R", and it never touches the price the trade actually exits at.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

__all__ = ["fvg_target", "FVG_TARGET_AT"]

FVG_TARGET_AT = ("touch", "ce", "full")


@njit(cache=True)
def _pick(close, stop_ref, side, buf, rr, near, at, g_from, g_to, g_kind, g_top, g_bot):
    n = close.shape[0]
    out = np.full(n, np.nan)
    pick = np.full(n, -1, np.int64)          # which gap (row of `gaps`, in from_i order)
    want = 1 if side == -1 else -1          # a short aims at a BULLISH gap below, a long a BEARISH above
    for i in range(n):
        c = close[i]
        s = stop_ref[i]
        if not (c == c) or not (s == s):
            continue
        stop = s * (1.0 - side * buf)        # the buffer pushes the stop further from the entry
        r = side * (c - stop)
        if not (r > 0.0):
            continue
        tp = c + side * rr * r
        lo_w, hi_w = tp - near * r, tp + near * r
        best = np.nan
        best_d = np.inf
        for g in range(g_from.shape[0]):
            if g_from[g] > i:
                break                        # sorted by from_i: nothing later is live yet
            if g_to[g] <= i or g_kind[g] != want:
                continue
            top, bot = g_top[g], g_bot[g]
            if side == -1:
                if top >= c:
                    continue                 # must lie wholly on the profit side
            else:
                if bot <= c:
                    continue
            if bot > hi_w or top < lo_w:
                continue                     # not within `near` R of the target
            d = (c - top) if side == -1 else (bot - c)
            if d < best_d:
                best_d = d
                pick[i] = g
                if at == 0:
                    best = top if side == -1 else bot
                elif at == 1:
                    best = 0.5 * (top + bot)
                else:
                    best = bot if side == -1 else top
        out[i] = best
    return out, pick


def fvg_target(df: pd.DataFrame, gaps: pd.DataFrame, *, side: str, stop_ref: np.ndarray,
               stop_buffer_pct: float, tp_rr: float, near_r: float = 1.0,
               at: str = "ce", return_gap: bool = False):
    """Per bar, the exit level of the gap a trade opened on that bar would aim at; NaN for none.

    `return_gap=True` also returns, per bar, the index LABEL in `gaps` of the gap chosen (-1 for
    none), so a trade can say which gap -- which timeframe, which band -- it exited on.

    `gaps` is `quant.indicators.fvg_on_grid(df, frames, **fvg_kwargs)` on this same frame.
    `stop_ref` is the stop method's reference column (the highest high of N bars for a short),
    `stop_buffer_pct` its buffer, and `tp_rr` the target in R -- the trade's own geometry.
    """
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    if at not in FVG_TARGET_AT:
        raise ValueError(f"at must be one of {FVG_TARGET_AT}, got {at!r}")
    n = len(df)
    if not len(gaps):
        empty = np.full(n, np.nan)
        return (empty, np.full(n, -1, np.int64)) if return_gap else empty
    g = gaps.sort_values("from_i", kind="stable")
    lv, pos = _pick(df["close"].to_numpy(np.float64), np.asarray(stop_ref, np.float64),
                 1 if side == "long" else -1, float(stop_buffer_pct) / 100.0, float(tp_rr),
                 float(near_r), FVG_TARGET_AT.index(at),
                 g["from_i"].to_numpy(np.int64), g["to_i"].to_numpy(np.int64),
                 g["kind"].to_numpy(np.int64), g["top"].to_numpy(np.float64),
                 g["bot"].to_numpy(np.float64))
    if not return_gap:
        return lv
    labels = g.index.to_numpy()
    return lv, np.where(pos >= 0, labels[np.clip(pos, 0, None)], -1)
