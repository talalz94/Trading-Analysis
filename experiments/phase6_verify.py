"""
PHASE 6 — the robustness gates, applied to the Phase-3 candidates.

Phase 3 produced 60 configs that are positive in all three regimes. Every one of them uses a TRAILING
stop, and that is precisely the shape this project has already been fooled by once: a 1 x ATR initial
stop with a 0.5% trail means risking ~1R to chase a ~13R runner, so the result is carried by a handful
of enormous winners and `med_R` sits at -1.0. A previous "edge" of this shape turned out to be one
+202R trade with everything else summing to negative.

`expR_ex_top` only removes ONE trade per window. That is not enough for a distribution like this. The
gates here are the ones that actually discriminate:

  A. trade-distribution profile — how concentrated is the profit?
  B. TRIM-TOP-K — expectancy after deleting the best 1 / 5 / 10 / 25 trades and the top 1% / 5%
  C. cost sensitivity — spread x1, x2, x4
  D. parameter plateau — is the pick a basin or a spike?
  E. rotating out-of-sample — train on two windows, verify on the untouched third
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.mgmt_sweep import EntryBundle
from experiments.search_harness import CFG_LEVERED, SPREAD_BTC, make_windows

CANDIDATES = [
    ("pdlow_fade_long/t30", "level_event_long@v1",
     {"lv_col": "pd_low", "lv_side": "below", "lv_event": "fade"},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 1.0, "trail_pct": 0.5}, 30),
    ("pdlow_fade_long/t15", "level_event_long@v1",
     {"lv_col": "pd_low", "lv_side": "below", "lv_event": "fade"},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 1.0, "trail_pct": 0.5}, 15),
    ("choch_long/t60", "bos_long@v1", {"bos_mode": "choch", "zz_atr_mult": 4.0},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 1.0, "trail_pct": 0.5}, 60),
    ("vwapz2_short/t60", "vwap_rev_short@v1", {"vr_sigma": 2.0},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 1.0, "trail_pct": 0.5}, 60),
    ("rtdiv_short/t120", "rsi_div_short@v1", {"div_mode": "realtime"},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 1.0, "trail_pct": 0.5}, 120),
    ("NULL_short/t60", "time_entry_short@v1", {"te_every": 200},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 1.0, "trail_pct": 0.5}, 60),
    ("NULL_long/t30", "time_entry_long@v1", {"te_every": 200},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 1.0, "trail_pct": 0.5}, 30),
]


def R_of(run) -> np.ndarray:
    t = run.trades
    if not len(t):
        return np.array([])
    risk = (t["stop_price"] - t["entry_price"]).abs() * t["qty"]
    return (t["pnl"] / risk).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)


def trimmed(R: np.ndarray, k: int) -> float:
    if len(R) <= k:
        return np.nan
    return float(np.sort(R)[:len(R) - k].mean())


def main():
    w = make_windows("BTCUSDT", "1m")
    base = {**CFG_LEVERED, "spread": SPREAD_BTC}
    pd.set_option("display.width", 260)

    # build once per (candidate, window)
    bundles = {}
    for tag, skey, ep, *_ in CANDIDATES:
        for k, d in w.items():
            bundles[(tag, k)] = EntryBundle(d, skey, ep)

    def runs_for(tag, spread=None, params_override=None, tstop=None):
        row = next(c for c in CANDIDATES if c[0] == tag)
        _, _, _, stop, tp, mp, ts = row
        mp = {**mp, **(params_override or {})}
        ts = ts if tstop is None else tstop
        cfg = {**base, **({"spread": spread} if spread else {}),
               "max_bars_in_trade": int(ts)}
        return {k: bundles[(tag, k)].run(stop, tp, mp, cfg) for k in w}

    print("=" * 104)
    print("A. TRADE DISTRIBUTION — where does the profit come from?")
    print("=" * 104)
    rows = []
    store = {}
    for tag, *_ in CANDIDATES:
        rs = runs_for(tag)
        store[tag] = rs
        for k, run in rs.items():
            R = R_of(run)
            if len(R) < 50:
                continue
            srt = np.sort(R)[::-1]
            tot = srt.sum()
            rows.append(dict(cfg=tag, win=k, n=len(R), expR=round(R.mean(), 4),
                             win_pct=round(100 * (R > 0).mean(), 1),
                             med_R=round(float(np.median(R)), 3),
                             avg_win=round(float(R[R > 0].mean()), 2) if (R > 0).any() else np.nan,
                             max_R=round(float(srt[0]), 1),
                             top1_share=round(100 * srt[0] / tot, 1) if tot > 0 else np.nan,
                             top5_share=round(100 * srt[:5].sum() / tot, 1) if tot > 0 else np.nan,
                             top1pct_share=round(100 * srt[:max(1, len(R)//100)].sum() / tot, 1)
                             if tot > 0 else np.nan))
    prof = pd.DataFrame(rows)
    print(prof.to_string(index=False))

    print("\n" + "=" * 104)
    print("B. TRIM-TOP-K — expectancy after DELETING the best trades (per window, worst shown)")
    print("   A real edge degrades gracefully. A lottery goes negative as soon as the tail is cut.")
    print("=" * 104)
    rows = []
    for tag, rs in store.items():
        rec = {"cfg": tag}
        Rs = [R_of(r) for r in rs.values()]
        Rs = [R for R in Rs if len(R) >= 50]
        if not Rs:
            continue
        rec["n_min"] = min(len(R) for R in Rs)
        for k in (0, 1, 5, 10, 25):
            rec[f"drop{k}"] = round(min(trimmed(R, k) for R in Rs), 4)
        for pct in (1, 5):
            rec[f"drop{pct}%"] = round(min(trimmed(R, max(1, len(R) * pct // 100)) for R in Rs), 4)
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 104)
    print("C. COST SENSITIVITY — worst-window expR at spread x1 / x2 / x4")
    print("=" * 104)
    rows = []
    for tag, *_ in CANDIDATES:
        rec = {"cfg": tag}
        for mult in (1, 2, 4):
            rs = runs_for(tag, spread=SPREAD_BTC * mult)
            vals = [R_of(r).mean() for r in rs.values() if len(R_of(r)) >= 50]
            rec[f"x{mult}"] = round(min(vals), 4) if vals else np.nan
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 104)
    print("D. PARAMETER PLATEAU — vary one knob at a time around each pick (worst-window expR)")
    print("=" * 104)
    for tag in ["pdlow_fade_long/t30", "choch_long/t60", "vwapz2_short/t60"]:
        print(f"\n  {tag}")
        for knob, vals in [("stop_atr_mult", [0.5, 0.75, 1.0, 1.5, 2.5]),
                           ("trail_pct", [0.25, 0.5, 0.75, 1.0, 2.0])]:
            out = []
            for v in vals:
                rs = runs_for(tag, params_override={knob: v})
                e = [R_of(r).mean() for r in rs.values() if len(R_of(r)) >= 50]
                out.append(f"{v:>5g}: {min(e):+.4f}" if e else f"{v:>5g}:   n/a")
            print(f"    {knob:15s} " + "   ".join(out))
        out = []
        for v in [0, 15, 30, 60, 120, 240]:
            rs = runs_for(tag, tstop=v)
            e = [R_of(r).mean() for r in rs.values() if len(R_of(r)) >= 50]
            out.append(f"{v:>4d}: {min(e):+.4f}" if e else f"{v:>4d}:   n/a")
        print(f"    {'time_stop':15s} " + "   ".join(out))

    print("\n" + "=" * 104)
    print("E. ROTATING OUT-OF-SAMPLE — each window held out in turn")
    print("=" * 104)
    rows = []
    for tag, rs in store.items():
        per = {k: R_of(r) for k, r in rs.items()}
        for held in per:
            tr = [v for k, v in per.items() if k != held and len(v) >= 50]
            te = per[held]
            if not tr or len(te) < 50:
                continue
            rows.append(dict(cfg=tag, held_out=held,
                             train=round(float(np.concatenate(tr).mean()), 4),
                             test=round(float(te.mean()), 4), test_n=len(te)))
    oos = pd.DataFrame(rows)
    print(oos.to_string(index=False))
    print("\n  by config:")
    print(oos.groupby("cfg").agg(min_test=("test", "min"), mean_test=("test", "mean"),
                                 all_pos=("test", lambda s: bool((s > 0).all()))
                                 ).round(4).to_string())
    oos.to_csv("experiments/results_phase6_verify_oos.csv", index=False)
    prof.to_csv("experiments/results_phase6_verify_profile.csv", index=False)
    print("\nsaved -> experiments/results_phase6_verify_*.csv")


if __name__ == "__main__":
    main()
