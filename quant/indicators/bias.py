"""
A market's BIAS in one word, from two things a chart reader looks at:

  structure  the last confirmed swings (add_zigzag_structure's `zz_trend`): +1 after a higher
             high AND a higher low, -1 after a lower high AND a lower low, 0 when they disagree
  EMAs       +1 when close > fast EMA > slow EMA, -1 when close < fast < slow, 0 otherwise (and
             while the slow EMA is still warming up)

Their sum is the bias, -2..+2: strongly bearish, bearish, neutral, bullish, strongly bullish.
Causal: a swing counts from the candle that confirms it, an EMA from the candle it closes on.

It DESCRIBES the market; it is not a signal. Measured here as a trade filter, an EMA bias and
HH/HL structure added nothing (docs/WHAT_WORKS.md: HTF bias with and against, phase 8). It is the
reference exness/bias.py is held to, to the last digit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .overlays import ema
from .structure import add_zigzag_structure

BIAS_LABELS = {2: "Strongly bullish", 1: "Bullish", 0: "Neutral", -1: "Bearish",
               -2: "Strongly bearish"}


def add_bias(df: pd.DataFrame, *, fast: int = 50, slow: int = 200, atr_mult: float = 3.5,
             atr_period: int = 14, prefix: str = "bias") -> pd.DataFrame:
    """Adds {prefix}_structure, {prefix}_ema (each -1/0/+1), {prefix} (-2..+2) and
    {prefix}_label. `atr_mult` is the swing size, as in the trendlines (3.5 x ATR)."""
    if not 0 < int(fast) < int(slow):
        raise ValueError(f"need 0 < fast < slow, got fast={fast}, slow={slow}")
    out = df.copy()
    z = add_zigzag_structure(out[["open", "high", "low", "close"]], atr_mult=atr_mult,
                             atr_period=atr_period)
    structure = z["zz_trend"].to_numpy(np.int64)
    c = out["close"]
    ef, es = ema(c, fast).to_numpy(float), ema(c, slow).to_numpy(float)
    cv = c.to_numpy(float)
    with np.errstate(invalid="ignore"):
        side = np.where((cv > ef) & (ef > es), 1, np.where((cv < ef) & (ef < es), -1, 0))
    out[f"{prefix}_structure"] = structure
    out[f"{prefix}_ema"] = side.astype(np.int64)
    out[prefix] = structure + side
    out[f"{prefix}_label"] = [BIAS_LABELS[int(v)] for v in out[prefix]]
    return out
