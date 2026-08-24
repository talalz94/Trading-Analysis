"""
PHASE 15b — put the Phase-15 winners through the controls before believing them.

Phase 15 reported a drift-adjusted forward move of ~1.26 ATR for `rising_wedge`, positive in 9 of 9
windows at a 64% hit rate. That is roughly five times any other result in this project, and every
result of that size here has so far turned out to be an artifact. Three specific ways it could be
one, each of which has caught something before:

1. THE DENOMINATOR TRAP (methodology rule 10). The forward move is divided by ATR *at the
   confirmation bar*. A wedge is by definition a CONVERGING shape, so ATR is compressed exactly where
   the measurement is taken, and the move afterwards is then divided by an unusually small number. A
   "1.79x expansion after quiet" earlier in this project was entirely this. Fix: re-measure in
   percent, and in units of the WINDOW median ATR, which cannot move with the signal.

2. BREAKOUT MOMENTUM, NOT SHAPE. `rising_wedge` fires on the bar that CLOSES DOWN through the rising
   lower line — a big down bar by construction. If BTC 15m has any short-horizon continuation, then
   "a large move just happened" predicts the next move and the wedge geometry contributes nothing.
   Fix: a MATCHED control — compare each confirmation against all bars in the same trailing-move and
   volatility bucket, which is the forward-return analogue of the metronome.

3. NO SEQUENCE STRUCTURE AT ALL (methodology rule 1). Fix: re-run on within-day-shuffled bars.

Checks 1, 2 and the volatility diagnostic all reuse a single detection pass. The shuffled null runs
only on whatever is still standing afterwards, because it costs another full pass.
"""
from __future__ import annotations

import json
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
H = 20                 # the horizon Phase 15 selected on
TRAIL = 20             # trailing window for the momentum match
NBIN = 5               # quantile bins per axis of the matched control
OUT = "experiments/results_phase15b_controls.csv"

#: the per-pattern winners from Phase 15 section A
FINALISTS = {
    "rising_wedge":           dict(atr_mult=2.5, min_touches=2, min_span_bars=40,  min_height_atr=4.0, line_tol_atr=1.2),
    "falling_wedge":          dict(atr_mult=5.0, min_touches=2, min_span_bars=200, min_height_atr=2.0, line_tol_atr=1.2),
    "descending_triangle":    dict(atr_mult=2.5, min_touches=2, min_span_bars=200, min_height_atr=2.0, line_tol_atr=1.2),
    "broadening_bottom":      dict(atr_mult=3.5, min_touches=3, min_span_bars=100, min_height_atr=2.0, line_tol_atr=1.2),
    "symmetrical_triangle":   dict(atr_mult=3.5, min_touches=3, min_span_bars=40,  min_height_atr=4.0, line_tol_atr=0.6),
    "triple_top":             dict(atr_mult=5.0, min_touches=3, min_span_bars=40,  min_height_atr=4.0, line_tol_atr=1.2),
    "head_shoulders":         dict(atr_mult=5.0, min_touches=3, min_span_bars=40,  min_height_atr=2.0, line_tol_atr=0.6),
    "ascending_triangle":     dict(atr_mult=5.0, min_touches=2, min_span_bars=100, min_height_atr=4.0, line_tol_atr=1.2),
    "inverse_head_shoulders": dict(atr_mult=5.0, min_touches=2, min_span_bars=40,  min_height_atr=4.0, line_tol_atr=1.2),
}


def _bins(x: np.ndarray, k: int) -> np.ndarray:
    """Quantile-bin, tolerating ties and NaNs (NaN -> bin 0; the caller masks those rows out)."""
    b = np.zeros(len(x), int)
    ok = np.isfinite(x)
    if ok.sum() > k:
        qs = np.unique(np.nanquantile(x[ok], np.linspace(0, 1, k + 1)[1:-1]))
        b[ok] = np.searchsorted(qs, x[ok])
    return np.clip(b, 0, k - 1)


def measure(d: pd.DataFrame, grouped: dict) -> list:
    """One detection pass per distinct settings combination; every metric off the same masks."""
    c = d["close"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    med = float(np.nanmedian(atr))              # a CONSTANT denominator: cannot move with the signal

    fwd = np.r_[c[H:], np.full(H, np.nan)] - c
    ok = np.isfinite(fwd) & np.isfinite(atr) & (atr > 0)
    f_atr = np.where(ok, fwd / np.where(atr > 0, atr, np.nan), np.nan)   # what Phase 15 measured
    f_fix = fwd / med                                                    # immovable denominator
    f_pct = 100.0 * fwd / c                                              # price terms

    # matched control: trailing move (in fixed ATR) x volatility state
    trail = np.r_[np.full(TRAIL, np.nan), c[TRAIL:] - c[:-TRAIL]] / med
    cell = _bins(trail, NBIN) * NBIN + _bins(atr, NBIN)
    cmean = np.full(NBIN * NBIN, np.nan)
    okc = ok & np.isfinite(trail)
    for k in range(NBIN * NBIN):
        s = okc & (cell == k)
        if s.sum() >= 30:
            cmean[k] = f_fix[s].mean()

    base_atr = float(np.nanmean(f_atr[ok]))
    base_fix = float(np.nanmean(f_fix[ok]))
    base_pct = float(np.nanmean(f_pct[ok]))

    rows = []
    for pats, cfg in grouped.items():
        spec = dict(min_touches_upper=cfg["min_touches"], min_touches_lower=cfg["min_touches"],
                    min_span_bars=cfg["min_span_bars"], min_height_atr=cfg["min_height_atr"],
                    line_tol_atr=cfg["line_tol_atr"])
        o = add_chart_patterns(d, patterns=list(pats), atr_mult=cfg["atr_mult"],
                               settings={p: spec for p in pats})
        for p in pats:
            m = o[f"pat_{p}"].to_numpy(bool) & ok
            n = int(m.sum())
            if n == 0:
                continue
            sgn = PATTERN_DIRECTION[p] or 1
            mm = m & okc & np.isfinite(cmean[cell])
            rows.append(dict(
                pattern=p, n=n,
                edge_atr=float((sgn * f_atr[m]).mean() - sgn * base_atr),
                edge_fix=float((sgn * f_fix[m]).mean() - sgn * base_fix),
                edge_pct=float((sgn * f_pct[m]).mean() - sgn * base_pct),
                edge_matched=(float(sgn * (f_fix[mm].mean() - cmean[cell[mm]].mean()))
                              if mm.any() else np.nan),
                n_matched=int(mm.sum()),
                hit=float((sgn * fwd[m] > 0).mean()),
                atr_ratio=float(np.nanmean(atr[m]) / med),   # <1 == compressed where measured
            ))
    return rows


def run(shuffle_seed=None, only=None) -> pd.DataFrame:
    by_cfg = {}
    for p, cfg in FINALISTS.items():
        if only and p not in only:
            continue
        by_cfg.setdefault(tuple(sorted(cfg.items())), []).append(p)
    grouped = {tuple(v): dict(k) for k, v in by_cfg.items()}
    frames, t0 = [], time.time()
    for wi, (k, (a, b)) in enumerate(WINDOWS.items(), 1):
        raw = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC")
        if shuffle_seed is not None:
            raw = shuffle_within_day(raw, seed=shuffle_seed + wi)
        d = add_atr(raw, 14).reset_index(drop=True)
        for r in measure(d, grouped):
            r["window"] = k
            frames.append(r)
        print(f"  {k} done ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(frames)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("pattern")
              .agg(n=("n", "sum"), windows=("window", "nunique"),
                   edge_atr=("edge_atr", "median"), edge_fix=("edge_fix", "median"),
                   edge_pct=("edge_pct", "median"), edge_matched=("edge_matched", "median"),
                   pos_atr=("edge_atr", lambda s: int((s > 0).sum())),
                   pos_matched=("edge_matched", lambda s: int((s > 0).sum())),
                   hit=("hit", "median"), atr_ratio=("atr_ratio", "median"))
              .reset_index().sort_values("edge_matched", ascending=False))


def main():
    t0 = time.time()
    print("REAL DATA", flush=True)
    real = run()
    real.to_csv(OUT, index=False)
    g = summarise(real)
    pd.set_option("display.width", 250)

    print("\n" + "=" * 108)
    print("CONTROLS 1+2. THE SAME CONFIRMATIONS, MEASURED FOUR WAYS  (h=20)")
    print("   edge_atr     as Phase 15 measured it — forward move / ATR AT THE CONFIRMATION BAR")
    print("   edge_fix     identical, but divided by the WINDOW MEDIAN ATR (denominator cannot move)")
    print("   edge_pct     percent of price — no ATR anywhere")
    print("   edge_matched vs bars in the same trailing-move x volatility bucket (shape vs momentum)")
    print("   atr_ratio    ATR where measured / window median. <1 means the denominator was small.")
    print("=" * 108)
    print(g.round(4).to_string(index=False))

    surv = g[(g.edge_matched > 0) & (g.pos_matched >= 7)].pattern.tolist()
    print(f"\n   still standing after the matched control: {surv or 'NONE'}")
    json.dump(surv, open("experiments/_p15b_survivors.json", "w"))

    if surv:
        print("\nSHUFFLED NULL (survivors only)", flush=True)
        gn = summarise(run(shuffle_seed=1000, only=surv))
        print("\n" + "=" * 108)
        print("CONTROL 3. THE SAME DETECTOR ON WITHIN-DAY-SHUFFLED BARS — nothing here is findable")
        print("=" * 108)
        print(gn.round(4).to_string(index=False))
        j = g.merge(gn, on="pattern", suffixes=("", "_null"))
        j["real_minus_null"] = j.edge_matched - j.edge_matched_null
        print("\n   real minus null (matched edge):")
        print(j[["pattern", "edge_matched", "edge_matched_null", "real_minus_null"]]
              .round(4).to_string(index=False))
    print(f"\n  total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
