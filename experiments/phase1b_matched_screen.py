"""
PHASE 1b — level screen with DAY-MATCHED controls.

Phase 1 subtracted the whole-window mean forward return. Not enough: level events cluster in trending
stretches, so a short-side event set inherits the drift of the days it lands on. The tell was
unmistakable — every SHORT cell positive and every LONG cell negative across all four interaction
types. A real mean-reversion effect must be symmetric.

Fix: each event is compared with the mean forward return of **all bars on the same calendar day**, so
whatever the market did that day is differenced out and only the level's own contribution survives.

Also runs an explicit SYMMETRY test: an effect is believable only if the long version at a
support-like level and the short version at the mirrored resistance-like level are BOTH positive.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                      # noqa: E402
from quant.data.timeframe import resample_ohlcv                       # noqa: E402
from quant.indicators import level_events                             # noqa: E402
from experiments.phase1_level_screen import BET, EVENTS, LEVELS, prepare   # noqa: E402

HORIZONS = [5, 15, 30, 60, 120]
MIRRORS = [("pd_high", "pd_low"), ("wk_high", "wk_low"),
           ("sess_prev_high", "sess_prev_low"), ("sess_run_high", "sess_run_low"),
           ("sess_or_high", "sess_or_low"), ("rnd_above", "rnd_below"),
           ("sr_resistance", "sr_support"), ("vwap_u1", "vwap_l1")]


def _excess_by_horizon(d: pd.DataFrame) -> dict:
    """Forward move in ATR units MINUS that calendar day's mean — computed once per horizon."""
    c = d["close"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    day = (d["t"].dt.year * 10000 + d["t"].dt.month * 100 + d["t"].dt.day).to_numpy()
    out = {}
    for h in HORIZONS:
        fwd = np.r_[c[h:], np.full(h, np.nan)] - c
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = fwd / atr
        s = pd.Series(norm)
        day_mean = s.groupby(day).transform("mean").to_numpy()
        out[h] = norm - day_mean
    return out


def screen_matched(windows: dict, round_step: float, or_bars: int = 30,
                   tol_pct: float = 0.05) -> pd.DataFrame:
    prepped, excess, masks = {}, {}, {}
    for k, v in windows.items():
        d = prepare(v.copy(), round_step, or_bars)
        prepped[k] = d
        excess[k] = _excess_by_horizon(d)
        masks[k] = {(lvl, side): level_events(d, lvl, tol_pct=tol_pct, side=side)
                    for lvl, side in LEVELS if lvl in d.columns}
    rows = []
    for lvl, side in LEVELS:
        for ev in EVENTS:
            bet = BET[(side, ev)]
            rec = {"level": lvl, "side": side, "event": ev,
                   "bet": "long" if bet > 0 else "short"}
            for h in HORIZONS:
                es, ns = [], []
                for k in prepped:
                    if (lvl, side) not in masks[k]:
                        continue
                    m = masks[k][(lvl, side)][ev]
                    x = excess[k][h]
                    sel = m & np.isfinite(x)
                    if sel.sum() < 30:
                        continue
                    es.append(float(bet * x[sel].mean()))
                    ns.append(int(sel.sum()))
                if not es:
                    continue
                rec[f"h{h}"] = round(float(np.mean(es)), 4)
                rec[f"h{h}_min"] = round(float(np.min(es)), 4)
                rec[f"h{h}_pos"] = int(sum(1 for x in es if x > 0))
                rec[f"h{h}_n"] = int(np.sum(ns))
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    runs = [("BTCUSDT", "binance", "spot", 1000.0, "1m", 30),
            ("BTCUSDT", "binance", "spot", 1000.0, "5m", 6),
            ("BTCUSDT", "binance", "spot", 1000.0, "15m", 2),
            ("XAUUSD", "dukascopy", "cfd", 50.0, "1m", 30),
            ("XAUUSD", "dukascopy", "cfd", 50.0, "5m", 6),
            ("PAXGUSDT", "binance", "spot", 50.0, "5m", 6)]
    spans = {"a": ("2025-01-01", "2025-07-29"), "b": ("2025-07-29", "2026-01-28"),
             "c": ("2026-01-28", "2026-07-29")}
    out = []
    for sym, src, mkt, step, tf, orb in runs:
        base = {}
        for k, (a, b) in spans.items():
            d = get_ohlcv(sym, "1m", start=a, end=b, source=src, market=mkt, tz="UTC")
            base[k] = (d if tf == "1m"
                       else resample_ohlcv(d, tf.replace("m", "min")).reset_index(drop=True))
        r = screen_matched(base, step, or_bars=orb)
        r["symbol"], r["tf"] = sym, tf
        out.append(r)
        print(f"  screened {sym} {tf}", flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_csv("experiments/results_phase1b_matched.csv", index=False)
    pd.set_option("display.width", 250)

    print("\n" + "=" * 108)
    print("A. BY INTERACTION TYPE (day-matched). Long AND short must be positive for a real effect")
    print("=" * 108)
    piv = (res.groupby(["event", "bet"])
              .agg(cells=("h30", "size"), med_h5=("h5", "median"), med_h15=("h15", "median"),
                   med_h30=("h30", "median"), med_h60=("h60", "median"),
                   med_h120=("h120", "median"),
                   pct_pos_h60=("h60", lambda s: round(100 * (s > 0).mean(), 1)))
              .reset_index())
    print(piv.round(4).to_string(index=False))

    print("\n" + "=" * 108)
    print("B. SYMMETRY TEST — mirrored level pairs. both_pos = the effect exists on BOTH sides")
    print("=" * 108)
    rows = []
    for ev in EVENTS:
        for hi, lo in MIRRORS:
            a = res[(res.level == hi) & (res.event == ev)]
            b = res[(res.level == lo) & (res.event == ev)]
            if not len(a) or not len(b):
                continue
            for h in (30, 60):
                ma, mb = a[f"h{h}"].median(), b[f"h{h}"].median()
                if ma != ma or mb != mb:
                    continue
                rows.append(dict(event=ev, pair=f"{hi}/{lo}", h=h,
                                 short_side=round(ma, 4), long_side=round(mb, 4),
                                 both_pos=bool(ma > 0 and mb > 0),
                                 sym_edge=round(min(ma, mb), 4)))
    sd = pd.DataFrame(rows).sort_values("sym_edge", ascending=False)
    print(sd.head(25).to_string(index=False))
    n = int(sd.both_pos.sum())
    print(f"\n  BOTH sides positive: {n}/{len(sd)} pairs   (chance ~ {len(sd)//4})")
    print("\n  Top symmetric candidates:")
    top = sd[sd.both_pos].head(8)
    print(top.to_string(index=False) if len(top) else "  NONE")
    print("\nsaved -> experiments/results_phase1b_matched.csv")
    return res


if __name__ == "__main__":
    main()
