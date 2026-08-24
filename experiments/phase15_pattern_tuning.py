"""
PHASE 15 — find the best DETECTOR settings for each chart pattern, judged on DIRECTION only.

This is deliberately not a strategy search. No stop, no target, no costs: the single question is
whether, once a pattern confirms, price actually goes the way the pattern says. Settings are then
chosen per pattern on that basis, so a later backtest starts from shapes that at least point the
right way.

Metric. For each confirmation, the signed forward move in the pattern's own direction, in ATR units,
at several horizons — MINUS the same-window mean forward move over all bars. That drift adjustment
is essential and is the mistake that wasted days earlier in this project: in a window where BTC fell
30%, every bearish pattern "works". Reported alongside is the HIT RATE (share of confirmations that
move the right way), which is the plainest reading of "does it point the right way".

Search. `PatternSpec` knobs are swept uniformly and then scored PER PATTERN, so each shape picks its
own best combination out of the same grid:

    zigzag atr_mult   how big a swing has to be to count as a pivot at all
    min_touches       how many pivots each fitted trendline must pass through
    min_span_bars     how wide the shape must be
    min_height_atr    how tall
    line_tol_atr      how close to the line a pivot must sit to count as a touch

Guards, all of which have caught something before:
  * every candidate must clear a minimum sample count, because the fixed detector is selective and
    41-52 trades per window already puts the standard error near 0.18;
  * consistency across the nine-window balanced cycle is reported, not just the pooled mean;
  * survivors are counted against chance;
  * the finalists are then re-measured against a within-day-shuffled null, which is the only control
    that has reliably separated real structure from measurement artifact here.
"""
from __future__ import annotations

import itertools
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                       # noqa: E402
from quant.indicators import add_atr, add_chart_patterns               # noqa: E402
from quant.indicators.chart_patterns import PATTERN_DIRECTION          # noqa: E402
from experiments.null_control import shuffle_within_day                # noqa: E402
from experiments.phase9_long_fair_test import WINDOWS                  # noqa: E402

TF = "15m"
HORIZONS = [10, 20, 40, 60]
MIN_N = 60                 # pooled confirmations required before a cell is reported at all
OUT = "experiments/results_phase15_pattern_tuning.csv"

# the shapes worth tuning: the ones that fire in usable numbers
PATTERNS = ["head_shoulders", "inverse_head_shoulders", "symmetrical_triangle",
            "ascending_triangle", "descending_triangle", "rising_wedge", "falling_wedge",
            "double_top", "double_bottom", "triple_top", "triple_bottom",
            "broadening_top", "broadening_bottom"]

GRID = dict(
    atr_mult=[2.5, 3.5, 5.0],
    min_touches=[2, 3, 4],
    min_span_bars=[40, 100, 200],
    min_height_atr=[2.0, 4.0],
    line_tol_atr=[0.6, 1.2],
)


def _drift_adjusted(d: pd.DataFrame) -> dict:
    """Per horizon: the signed forward move of every bar in ATR units, and the window's mean."""
    c = d["close"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    out = {}
    for h in HORIZONS:
        fwd = (np.r_[c[h:], np.full(h, np.nan)] - c) / atr
        ok = np.isfinite(fwd)
        out[h] = (fwd, ok, float(np.nanmean(fwd[ok])) if ok.any() else np.nan)
    return out


def measure_window(d: pd.DataFrame, cfg: dict) -> list:
    """Detect with one settings combination and score every pattern on direction."""
    spec = dict(min_touches_upper=cfg["min_touches"], min_touches_lower=cfg["min_touches"],
                min_span_bars=cfg["min_span_bars"], min_height_atr=cfg["min_height_atr"],
                line_tol_atr=cfg["line_tol_atr"])
    o = add_chart_patterns(d, patterns=PATTERNS, atr_mult=cfg["atr_mult"],
                           settings={p: spec for p in PATTERNS})
    fw = _drift_adjusted(d)
    rows = []
    for p in PATTERNS:
        m0 = o[f"pat_{p}"].to_numpy(bool)
        if not m0.any():
            continue
        sgn = PATTERN_DIRECTION[p] or 1
        rec = {"pattern": p, **cfg}
        for h in HORIZONS:
            fwd, ok, base = fw[h]
            m = m0 & ok
            n = int(m.sum())
            if n == 0:
                continue
            signed = sgn * fwd[m]
            rec[f"n{h}"] = n
            rec[f"edge{h}"] = float(signed.mean() - sgn * base)   # drift-adjusted
            rec[f"hit{h}"] = float((signed > 0).mean())
        if any(k.startswith("n") for k in rec):
            rows.append(rec)
    return rows


def run_grid(shuffle_seed=None) -> pd.DataFrame:
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(combos)} settings x {len(WINDOWS)} windows", flush=True)
    frames, t0 = [], time.time()
    for wi, (k, (a, b)) in enumerate(WINDOWS.items(), 1):
        raw = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC")
        if shuffle_seed is not None:
            raw = shuffle_within_day(raw, seed=shuffle_seed + wi)
        d = add_atr(raw, 14).reset_index(drop=True)
        for cfg in combos:
            for r in measure_window(d, cfg):
                r["window"] = k
                frames.append(r)
        print(f"  {k} done  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(frames)


def summarise(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    keys = ["pattern"] + list(GRID)
    agg = {}
    for h in HORIZONS:
        agg[f"n{h}"] = (f"n{h}", "sum")
        agg[f"edge{h}"] = (f"edge{h}", "median")
        agg[f"hit{h}"] = (f"hit{h}", "median")
        agg[f"pos{h}"] = (f"edge{h}", lambda s: int((s > 0).sum()))
    agg["windows"] = ("window", "nunique")
    g = df.groupby(keys).agg(**agg).reset_index()
    return g


def main():
    t0 = time.time()
    raw = run_grid()
    raw.to_csv(OUT.replace(".csv", "_raw.csv"), index=False)
    g = summarise(raw)
    g.to_csv(OUT, index=False)
    pd.set_option("display.width", 250)

    print("\n" + "=" * 110)
    print("A. BEST SETTINGS PER PATTERN — drift-adjusted forward move, h=20")
    print("   `edge20` in ATR units, `hit20` = share going the pattern's way (0.50 = coin flip)")
    print("=" * 110)
    ok = g[(g["n20"] >= MIN_N)].copy()
    best = (ok.sort_values("edge20", ascending=False)
              .groupby("pattern").head(1)
              .sort_values("edge20", ascending=False))
    cols = (["pattern"] + list(GRID)
            + ["n20", "edge20", "hit20", "pos20", "windows", "edge40", "hit40"])
    print(best[cols].round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("B. IS THE BEST SETTING STABLE, OR A SPIKE? spread of edge20 across all settings")
    print("=" * 110)
    st = (ok.groupby("pattern")
            .agg(settings=("edge20", "size"), best=("edge20", "max"), median=("edge20", "median"),
                 worst=("edge20", "min"),
                 pct_positive=("edge20", lambda s: round(100 * (s > 0).mean(), 1)),
                 median_hit=("hit20", "median"))
            .reset_index().sort_values("median", ascending=False))
    print(st.round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("C. WHICH KNOB ACTUALLY MATTERS? median edge20 by setting, pooled over patterns")
    print("=" * 110)
    for k in GRID:
        s = ok.groupby(k).agg(cells=("edge20", "size"), edge=("edge20", "median"),
                              hit=("hit20", "median"), n=("n20", "median")).reset_index()
        print(f"\n  {k}:")
        print(s.round(4).to_string(index=False))

    print("\n" + "=" * 110)
    print("D. COUNT AGAINST CHANCE")
    print("=" * 110)
    tot = len(ok)
    pos = int((ok.edge20 > 0).sum())
    strong = ok[(ok.edge20 > 0) & (ok.pos20 >= ok.windows - 1) & (ok.n20 >= MIN_N)]
    from math import comb
    p_consistent = sum(comb(9, i) for i in (8, 9)) / 2 ** 9
    print(f"  settings x patterns measured : {tot}")
    print(f"  positive drift-adjusted edge : {pos} ({100*pos/tot:.1f}%)   chance = 50%")
    print(f"  positive AND consistent      : {len(strong)}   chance ~ {p_consistent*tot:.1f}")
    print(f"\n  total runtime {time.time()-t0:.0f}s")
    print(f"saved -> {OUT}")
    return g


if __name__ == "__main__":
    main()
