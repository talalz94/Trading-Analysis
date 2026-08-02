"""
PHASE 12A — do patterns predict ANYTHING? The null-differenced screen.

81 patterns (62 candlestick, 19 chart) x 4 horizons, on the nine-window balanced cycle, each measured
against a within-day bar-shuffled null.

Why that null is unusually good here: shuffling bars inside a day keeps every candle's exact OHLC
shape but destroys the ORDER. So a hammer is still a hammer, but its trend context is gone; a morning
star can only form by accident; a double top forms from randomly ordered swings. The shuffled series
therefore has the same candle-shape distribution and the same daily drift and volatility as the real
one, and none of the sequence information the patterns claim to read. Anything a pattern scores above
that null is real; anything at or below it is the pattern counting shapes that would occur anyway.

Three things are measured, because a pattern can be useful without predicting direction:
  DIRECTION   signed forward return in the pattern's own direction, in ATR units
  HIT RATE    how often the move goes the pattern's way (50% = a coin flip)
  EXPANSION   forward absolute travel — a pattern that predicts a BIG move but not its sign is still
              tradable (a straddle), and would be invisible to a directional screen
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                        # noqa: E402
from quant.indicators import (add_atr, add_candle_patterns, add_chart_patterns,   # noqa: E402
                              CANDLE_PATTERN_DIRECTION, CHART_PATTERN_DIRECTION)
from experiments.null_control import shuffle_within_day                 # noqa: E402
from experiments.phase9_long_fair_test import WINDOWS                   # noqa: E402

TF = "15m"
HORIZONS = [5, 10, 20, 40]
N_NULL = 2
MIN_HITS = 40          # below this a per-window number is noise and is dropped


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = add_candle_patterns(d)
    d = add_chart_patterns(d)
    return add_atr(d, 14).reset_index(drop=True)


def _fwd_travel(hi, lo, h):
    def ext(arr, how):
        r = pd.Series(arr[::-1]).rolling(h, min_periods=h)
        m = (r.max() if how == "max" else r.min()).to_numpy()[::-1]
        return np.r_[m[1:], np.nan]
    return ext(hi, "max") - ext(lo, "min")


def measure(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (pattern, horizon): signed edge, hit rate, expansion, count."""
    c = d["close"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    hi, lo = d["high"].to_numpy(float), d["low"].to_numpy(float)
    rows = []
    specs = ([(f"cdl_{k}", v, "candle") for k, v in CANDLE_PATTERN_DIRECTION.items()]
             + [(f"pat_{k}", v, "chart") for k, v in CHART_PATTERN_DIRECTION.items()])
    base_travel = {}
    for h in HORIZONS:
        fwd = (np.r_[c[h:], np.full(h, np.nan)] - c) / atr
        travel = _fwd_travel(hi, lo, h) / atr
        ok = np.isfinite(fwd) & np.isfinite(travel)
        base_travel[h] = float(np.nanmedian(travel[ok]))
        for col, direction, family in specs:
            if col not in d.columns:
                continue
            m = d[col].to_numpy(bool) & ok
            n = int(m.sum())
            if n < MIN_HITS:
                continue
            sgn = direction if direction != 0 else 1
            signed = sgn * fwd[m]
            rows.append(dict(pattern=col.split("_", 1)[1], family=family, dir=direction, h=h,
                             n=n, edge=float(signed.mean()),
                             hit=float((signed > 0).mean()),
                             travel=float(np.median(travel[m])),
                             travel_ratio=float(np.median(travel[m])) / base_travel[h]))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    real_parts, null_parts = [], []
    for k, (a, b) in WINDOWS.items():
        raw = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC")
        r = measure(prepare(raw.copy()))
        r["window"] = k
        real_parts.append(r)
        for s in range(N_NULL):
            nz = measure(prepare(shuffle_within_day(raw.copy(), seed=s * 31 + len(k))))
            nz["window"], nz["seed"] = k, s
            null_parts.append(nz)
        print(f"  screened {k}", flush=True)
    real = pd.concat(real_parts, ignore_index=True)
    null = pd.concat(null_parts, ignore_index=True)

    R = real.groupby(["pattern", "family", "dir", "h"]).agg(
        n=("n", "sum"), edge=("edge", "median"), hit=("hit", "median"),
        travel_ratio=("travel_ratio", "median"),
        pos_windows=("edge", lambda s: int((s > 0).sum())),
        windows=("edge", "size")).reset_index()
    N = null.groupby(["pattern", "h"]).agg(
        null_edge=("edge", "median"), null_hit=("hit", "median"),
        null_travel=("travel_ratio", "median")).reset_index()
    M = R.merge(N, on=["pattern", "h"], how="left")
    M["excess"] = M.edge - M.null_edge
    M["hit_excess"] = M.hit - M.null_hit
    M["travel_excess"] = M.travel_ratio - M.null_travel
    M.to_csv("experiments/results_phase12_pattern_screen.csv", index=False)

    print("\n" + "=" * 108)
    print("A. BY FAMILY — is there any directional signal at all? (median over patterns)")
    print("=" * 108)
    print(M.groupby(["family", "h"]).agg(
        patterns=("pattern", "nunique"), edge=("edge", "median"),
        null_edge=("null_edge", "median"), excess=("excess", "median"),
        hit=("hit", "median"), hit_excess=("hit_excess", "median")).round(4).to_string())

    print("\n" + "=" * 108)
    print("B. BEST DIRECTIONAL PATTERNS — net of their own null, needing consistency across windows")
    print("=" * 108)
    good = M[(M.pos_windows >= M.windows - 2) & (M.n >= 300)].nlargest(20, "excess")
    print(good[["pattern", "family", "dir", "h", "n", "edge", "null_edge", "excess",
                "hit", "hit_excess", "pos_windows", "windows"]].round(4).to_string(index=False))

    print("\n" + "=" * 108)
    print("C. WORST — patterns that reliably predict the OPPOSITE of their textbook direction")
    print("=" * 108)
    bad = M[(M.n >= 300)].nsmallest(12, "excess")
    print(bad[["pattern", "family", "dir", "h", "n", "edge", "null_edge", "excess", "hit"]]
          .round(4).to_string(index=False))

    print("\n" + "=" * 108)
    print("D. EXPANSION — patterns that predict a BIG MOVE regardless of direction")
    print("   travel_ratio > 1 means more forward range than a typical bar; excess is net of null")
    print("=" * 108)
    exp = M[(M.n >= 300) & (M.h == 20)].nlargest(15, "travel_excess")
    print(exp[["pattern", "family", "n", "travel_ratio", "null_travel", "travel_excess",
               "hit", "excess"]].round(4).to_string(index=False))

    print("\n" + "=" * 108)
    print("E. COUNT AGAINST CHANCE")
    print("=" * 108)
    tot = len(M)
    pos = int((M.excess > 0).sum())
    strong = int(((M.excess > 0) & (M.pos_windows >= M.windows - 1) & (M.n >= 300)).sum())
    print(f"  cells measured                       : {tot}")
    print(f"  positive excess over null            : {pos} ({100*pos/tot:.1f}%)  -- chance is 50%")
    print(f"  positive AND consistent AND >=300 hits: {strong}")
    print(f"  chance for that filter (~0.5^{int(M.windows.median())} x n)   : "
          f"{tot * 0.5 ** int(M.windows.median()):.1f}")
    print("\nsaved -> experiments/results_phase12_pattern_screen.csv")
    return M


if __name__ == "__main__":
    main()
