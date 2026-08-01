"""
NULL CONTROL — does the screen report an edge on data that provably has none?

The day-matched screen (phase1b) subtracts each calendar day's mean forward return. That control
contains bars from AFTER the event, so any signal firing near a day's extremes can show spurious
mean-reversion: within a finite path, points at the top of the day's range are followed by lower
prices *by construction*, whether or not the market actually reverts.

So: rebuild the series by SHUFFLING BARS WITHIN EACH DAY. Each day keeps its exact drift, its
volatility, its candle shapes and its volume distribution — only the ORDER is destroyed, which is
where every level effect must live. Levels computed on the shuffled series cannot predict anything.
Run the identical screen on it. Whatever edge survives is measurement artifact, and the real number
must beat the null, not merely be positive.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                      # noqa: E402
from experiments.phase1b_matched_screen import screen_matched         # noqa: E402


def shuffle_within_day(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Permute bar order inside each calendar day, preserving that day's drift, vol and candle shapes.

    Close-to-close returns are permuted and re-chained; each bar's open/high/low keep their own
    proportional offsets from their close, so candles stay realistic and highs still exceed lows.
    """
    rng = np.random.default_rng(seed)
    d = df.reset_index(drop=True).copy()
    c = d["close"].to_numpy(float)
    ret = np.r_[0.0, np.diff(np.log(c))]
    ro = np.log(d["open"].to_numpy(float) / c)
    rh = np.log(d["high"].to_numpy(float) / c)
    rl = np.log(d["low"].to_numpy(float) / c)
    v = d["volume"].to_numpy(float)
    day = (d["t"].dt.year * 10000 + d["t"].dt.month * 100 + d["t"].dt.day).to_numpy()
    order = np.arange(len(d))
    _, starts = np.unique(day, return_index=True)
    for a, b in zip(starts, np.r_[starts[1:], len(d)]):
        order[a:b] = a + rng.permutation(b - a)
    new_ret = ret[order]
    for a, b in zip(starts, np.r_[starts[1:], len(d)]):
        new_ret[a] = ret[a]                     # keep the day's opening gap where it belongs
    lc = np.log(c[0]) + np.cumsum(new_ret)
    d["close"] = np.exp(lc)
    d["open"] = np.exp(lc + ro[order])
    d["high"] = np.exp(lc + rh[order])
    d["low"] = np.exp(lc + rl[order])
    d["volume"] = v[order]
    d["high"] = np.maximum.reduce([d["high"], d["open"], d["close"]])
    d["low"] = np.minimum.reduce([d["low"], d["open"], d["close"]])
    return d


def main():
    spans = {"a": ("2025-01-01", "2025-07-29"), "b": ("2025-07-29", "2026-01-28"),
             "c": ("2026-01-28", "2026-07-29")}
    real = {k: get_ohlcv("BTCUSDT", "1m", start=a, end=b, source="binance", tz="UTC")
            for k, (a, b) in spans.items()}
    print("screening REAL BTCUSDT 1m ...", flush=True)
    r_real = screen_matched(real, 1000.0)
    nulls = []
    for s in (0, 1, 2):
        fake = {k: shuffle_within_day(v, seed=s * 10 + i) for i, (k, v) in enumerate(real.items())}
        print(f"screening SHUFFLED seed={s} ...", flush=True)
        n = screen_matched(fake, 1000.0)
        n["seed"] = s
        nulls.append(n)
    null = pd.concat(nulls, ignore_index=True)
    pd.set_option("display.width", 250)

    print("\n" + "=" * 96)
    print("REAL vs WITHIN-DAY-SHUFFLED NULL  (median edge in ATR units, BTC 1m)")
    print("=" * 96)
    rows = []
    for ev in ["first_break", "fade", "sweep", "retest"]:
        for bet in ["long", "short"]:
            a = real_v = r_real[(r_real.event == ev) & (r_real.bet == bet)]
            b = null[(null.event == ev) & (null.bet == bet)]
            if not len(a) or not len(b):
                continue
            rec = {"event": ev, "bet": bet}
            for h in (30, 60):
                rv, nv = a[f"h{h}"].median(), b[f"h{h}"].median()
                nsd = b.groupby("seed")[f"h{h}"].median().std()
                rec[f"real_h{h}"] = round(float(rv), 4)
                rec[f"null_h{h}"] = round(float(nv), 4)
                rec[f"excess_h{h}"] = round(float(rv - nv), 4)
                rec[f"z_h{h}"] = round(float((rv - nv) / nsd), 2) if nsd and nsd == nsd else np.nan
            rows.append(rec)
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False))
    print("\n  null_h* far from 0 => the SCREEN manufactures that number; only excess_h* is signal.")
    cmp.to_csv("experiments/results_null_control.csv", index=False)
    print("\nsaved -> experiments/results_null_control.csv")
    return cmp


if __name__ == "__main__":
    main()
