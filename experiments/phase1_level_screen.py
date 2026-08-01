"""
PHASE 1 — do reference levels predict anything at all?

For every (level x interaction) pair, measure the forward move implied by the trade you would put on,
in ATR units, MINUS the same-horizon baseline drift over all bars. Drift-adjustment is essential: in a
-28% window every short "works", which is how the original break entry fooled this project for days.

Directional convention (what you'd actually trade):
    level ABOVE price (resistance-like)   break -> long   fade -> short   sweep -> short  retest -> long
    level BELOW price (support-like)      break -> short  fade -> long    sweep -> long   retest -> short

Output: edge per regime, and a verdict flagging only those with a CONSISTENT SIGN across all regimes.
Nothing gets a backtest until it passes here.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                    # noqa: E402
from quant.data.timeframe import resample_ohlcv                     # noqa: E402
from quant.indicators import (add_atr, add_prev_day_levels, add_round_levels,  # noqa: E402
                              add_session_levels, add_sr_zones, add_volume_features,
                              add_vwap, add_week_levels, level_events)

HORIZONS = [5, 15, 30, 60, 120]
EVENTS = ["first_break", "fade", "sweep", "retest"]
# bet direction: +1 = long, -1 = short
BET = {("above", "first_break"): +1, ("above", "fade"): -1,
       ("above", "sweep"): -1, ("above", "retest"): +1,
       ("below", "first_break"): -1, ("below", "fade"): +1,
       ("below", "sweep"): +1, ("below", "retest"): -1}

LEVELS = [("pd_high", "above"), ("pd_low", "below"), ("pd_mid", "above"), ("pd_close", "above"),
          ("wk_high", "above"), ("wk_low", "below"),
          ("sess_prev_high", "above"), ("sess_prev_low", "below"),
          ("sess_run_high", "above"), ("sess_run_low", "below"),
          ("sess_or_high", "above"), ("sess_or_low", "below"),
          ("rnd_above", "above"), ("rnd_below", "below"),
          ("sr_resistance", "above"), ("sr_support", "below"),
          ("vwap", "above"), ("vwap", "below"),
          ("vwap_u1", "above"), ("vwap_l1", "below")]


def prepare(d: pd.DataFrame, round_step: float, or_bars: int = 30) -> pd.DataFrame:
    d = add_prev_day_levels(d)
    d = add_week_levels(d)
    d = add_session_levels(d, or_bars=or_bars)
    d = add_vwap(d)
    d = add_volume_features(d)
    d = add_round_levels(d, round_step)
    d = add_sr_zones(d, left=10, right=10, cluster_pct=0.15)
    d = add_atr(d, 14)
    return d.reset_index(drop=True)


def screen(windows: dict, round_step: float, tol_pct: float = 0.05,
           or_bars: int = 30) -> pd.DataFrame:
    prepped = {k: prepare(v.copy(), round_step, or_bars) for k, v in windows.items()}
    rows = []
    for lvl, side in LEVELS:
        for ev in EVENTS:
            bet = BET[(side, ev)]
            rec = {"level": lvl, "side": side, "event": ev,
                   "bet": "long" if bet > 0 else "short"}
            for h in HORIZONS:
                edges, ns = [], []
                for wname, d in prepped.items():
                    if lvl not in d.columns:
                        continue
                    c = d["close"].to_numpy(float)
                    atr = d["atr_14"].to_numpy(float)
                    fwd = np.r_[c[h:], np.full(h, np.nan)] - c
                    with np.errstate(invalid="ignore", divide="ignore"):
                        norm = fwd / atr
                    mask = level_events(d, lvl, tol_pct=tol_pct, side=side)[ev]
                    ok = np.isfinite(norm)
                    sel = mask & ok
                    if sel.sum() < 30:
                        continue
                    edge = bet * (norm[sel].mean() - norm[ok].mean())
                    edges.append(edge)
                    ns.append(int(sel.sum()))
                if not edges:
                    continue
                rec[f"h{h}"] = round(float(np.mean(edges)), 4)
                rec[f"h{h}_min"] = round(float(np.min(edges)), 4)
                rec[f"h{h}_pos"] = int(sum(1 for e in edges if e > 0))
                rec[f"h{h}_n"] = int(np.sum(ns))
            rows.append(rec)
    df = pd.DataFrame(rows)
    # a level/event is interesting only if the SIGN holds across every regime at some horizon
    consist = []
    for _, r in df.iterrows():
        best, besth = 0.0, None
        for h in HORIZONS:
            k, kp = f"h{h}_min", f"h{h}_pos"
            if k in r and r[k] == r[k] and r.get(kp, 0) == len(windows) and r[k] > best:
                best, besth = r[k], h
        consist.append((best, besth))
    df["edge_all_regimes"] = [c[0] for c in consist]
    df["best_horizon"] = [c[1] for c in consist]
    return df.sort_values("edge_all_regimes", ascending=False).reset_index(drop=True)


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "1m"
    source = sys.argv[3] if len(sys.argv) > 3 else "binance"
    step = 1000.0 if symbol.startswith("BTC") else 10.0
    spans = {"bull25": ("2025-01-01", "2025-07-29"),
             "bear25": ("2025-07-29", "2026-01-28"),
             "bear26": ("2026-01-28", "2026-07-29")}
    base = {k: get_ohlcv(symbol, "1m", start=a, end=b, source=source, tz="UTC")
            for k, (a, b) in spans.items()}
    if tf != "1m":
        base = {k: resample_ohlcv(v, tf).reset_index(drop=True) for k, v in base.items()}
    or_bars = {"1m": 30, "5m": 6, "15m": 2}.get(tf, 30)
    print(f"=== PHASE 1 LEVEL SCREEN — {symbol} {tf} ===")
    for k, v in base.items():
        print(f"  {k}: {len(v):>8,} bars  {v.close.iloc[-1]/v.close.iloc[0]-1:+.1%}")
    res = screen(base, step, or_bars=or_bars)
    pd.set_option("display.width", 250)
    cols = ["level", "side", "event", "bet", "h5", "h15", "h30", "h60", "h120",
            "edge_all_regimes", "best_horizon", "h30_n"]
    print("\n--- ALL (edge in ATR units, drift-adjusted; mean across regimes) ---")
    print(res[cols].to_string(index=False))
    print("\n--- SAME SIGN IN ALL 3 REGIMES (the only ones worth a backtest) ---")
    good = res[res.edge_all_regimes > 0]
    print(good[cols].to_string(index=False) if len(good) else "  NONE")
    out = f"experiments/results_phase1_levels_{symbol}_{tf}.csv"
    res.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    return res


if __name__ == "__main__":
    main()
