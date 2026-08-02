"""
PHASE 9 — the LONG side, on a fair sample.

Longs were never ignored: every entry family is registered long and short, and longs ran through
Phases 2, 3, 4 and 5. But the window set was 2025-01 -> 2026-08, over which **BTC fell 32.8%**, with
two of the three windows bear legs. The headline metric is the WORST window, so a long strategy had
to survive two bear markets while a short only had to survive one bull leg. That is not a fair test,
and Phase 8 (the improvement study) compounded it by running short-only.

This phase fixes both. Eight windows spanning a real cycle:

    2022 H1/H2   deep bear   (BTC -57% over the year)
    2023 H1/H2   recovery    (+155%)
    2024 H1/H2   bull        (+120%)
    2025 H1      bull leg    (+26%)
    2025 H2, 2026 H1  bear   (-24%, -30%)

Long and short are then measured on the SAME windows with the SAME protocol, and the reporting is
split three ways: score on the bull windows, on the bear windows, and on the worst of ALL of them.
The last is the honest number for something you would actually run unattended.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from quant.data import get_ohlcv
from quant.indicators import add_emas, add_sr_zones, add_zigzag_structure

from experiments.mgmt_sweep import sweep

TF = sys.argv[1] if len(sys.argv) > 1 else "15m"   # timeframe under test
WINDOWS = {
    "2022H1_bear": ("2022-01-01", "2022-07-01"),
    "2022H2_bear": ("2022-07-01", "2023-01-01"),
    "2023H1_bull": ("2023-01-01", "2023-07-01"),
    "2023H2_bull": ("2023-07-01", "2024-01-01"),
    "2024H1_bull": ("2024-01-01", "2024-07-01"),
    "2024H2_bull": ("2024-07-01", "2025-01-01"),
    "2025H1_bull": ("2025-01-01", "2025-07-29"),
    "2025H2_bear": ("2025-07-29", "2026-01-28"),
    "2026H1_bear": ("2026-01-28", "2026-08-01"),
}

MGMT = [dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.5, "tp_rr": 1.5},
             tag="atr2.5/rr1.5"),
        dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.5, "tp_rr": 2.0},
             tag="atr2.5/rr2"),
        dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 1.5, "tp_rr": 2.0},
             tag="atr1.5/rr2")]


def entries() -> list:
    """Every family, both sides, so the long mirror gets exactly the test the short one got."""
    out = []
    for side in ("long", "short"):
        for lvl in (45.0, 53.0, 60.0):
            out.append((f"wt{lvl:g}_{side}", f"wt_cross_{side}@v1", {"wt_level": lvl}, None))
        out.append((f"rtdiv_{side}", f"rsi_div_{side}@v1", {"div_mode": "realtime"}, None))
        out.append((f"div_{side}", f"rsi_div_{side}@v1", {"div_mode": "confirmed"}, None))
        for s in (1.5, 2.0):
            out.append((f"vwapz{s:g}_{side}", f"vwap_rev_{side}@v1", {"vr_sigma": s}, None))
        for m in ("bos", "choch"):
            out.append((f"{m}_{side}", f"bos_{side}@v1",
                        {"bos_mode": m, "zz_atr_mult": 2.5}, None))
        out.append((f"sqz_{side}", f"squeeze_{side}@v1", {"sqz_kc_mult": 1.5}, None))
        out.append((f"grad_{side}", f"grad_extreme_{side}@v1", {"ge_quantile": 0.9}, None))
        out.append((f"NULL_{side}", f"time_entry_{side}@v1", {"te_every": 60}, None))
    return out


def load() -> dict:
    out = {}
    for k, (a, b) in WINDOWS.items():
        d = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC")
        d = add_emas(d, [50, 200])
        d = add_sr_zones(d, left=10, right=10, cluster_pct=0.15)
        d = add_zigzag_structure(d, atr_mult=2.5)
        out[k] = d.reset_index(drop=True)
    return out


def main():
    pd.set_option("display.width", 260)
    w = load()
    print(f"=== TIMEFRAME {TF} ===")
    print("WINDOWS (BTC move over each):")
    for k, d in w.items():
        print(f"  {k:12s} {len(d):>6,} bars   {100*(d.close.iloc[-1]/d.close.iloc[0]-1):+7.1f}%")
    bulls = [k for k in w if k.endswith("bull")]
    bears = [k for k in w if k.endswith("bear")]

    ents = entries()
    print(f"\n{len(ents)} entries x {len(MGMT)} management x {len(w)} windows "
          f"= {len(ents)*len(MGMT)} configs", flush=True)
    t0 = time.time()
    res = sweep(w, ents, MGMT, verbose=False)
    print(f"done in {time.time()-t0:.0f}s")
    res["side"] = np.where(res.entry.str.endswith("_long"), "long", "short")
    res["is_null"] = res.entry.str.startswith("NULL")
    # per-regime aggregates from the per-window columns the sweep already wrote
    for lab, keys in (("bull", bulls), ("bear", bears)):
        cols = [f"expR_{k}" for k in keys if f"expR_{k}" in res.columns]
        res[f"expR_{lab}_min"] = res[cols].min(axis=1)
        res[f"expR_{lab}_mean"] = res[cols].mean(axis=1)
    res.to_csv(f"experiments/results_phase9_long_fair_{TF}.csv", index=False)
    if res.error.astype(bool).any():
        print("ERRORS:", res[res.error.astype(bool)].error.value_counts().head(3).to_dict())

    real = res[~res.is_null]
    print("\n" + "=" * 104)
    print("A. LONG vs SHORT on a BALANCED sample (median over all configs)")
    print("=" * 104)
    print(real.groupby("side").agg(
        configs=("expR_min", "size"),
        all_windows_min=("expR_min", "median"), all_windows_mean=("expR_mean", "median"),
        bull_min=("expR_bull_min", "median"), bull_mean=("expR_bull_mean", "median"),
        bear_min=("expR_bear_min", "median"), bear_mean=("expR_bear_mean", "median"),
        trades=("n_total", "median")).round(4).to_string())

    print("\n  the same, but BEST config per side (this is what selection actually picks):")
    print(real.groupby("side").agg(
        best_all=("expR_min", "max"), best_bull=("expR_bull_min", "max"),
        best_bear=("expR_bear_min", "max")).round(4).to_string())

    print("\n" + "=" * 104)
    print("B. TOP 20 BY WORST-OF-ALL-NINE-WINDOWS  (the honest unattended-run number)")
    print("=" * 104)
    cols = ["tag", "side", "n_total", "expR_min", "expR_mean", "expR_bull_min", "expR_bear_min",
            "expR_ex_top_min", "n_windows_pos", "dd_max"]
    print(real.nlargest(20, "expR_min")[cols].round(4).to_string(index=False))

    print("\n" + "=" * 104)
    print("C. BEST LONGS specifically (ranked on the bull windows, then checked on the bears)")
    print("=" * 104)
    lo = real[real.side == "long"]
    print(lo.nlargest(15, "expR_bull_min")[cols].round(4).to_string(index=False))

    print("\n" + "=" * 104)
    print("D. DOES ANYTHING SURVIVE ALL NINE WINDOWS?")
    print("=" * 104)
    nb = res[res.is_null].expR_min.max()
    print(f"  null ceiling across all 9 windows: {nb:.4f}")
    n_win = len([c for c in res.columns if c.startswith("expR_20")])
    surv = real[(real.expR_min > 0) & (real.n_total >= 300) & (real.expR_ex_top_min > 0)
                & (real.expR_min > nb)]
    print(f"  positive in EVERY window, >=300 trades, outlier-proof, beats the null: "
          f"{len(surv)} of {len(real)}   (chance ~{len(real)*0.5**n_win:.2f})")
    print(surv.sort_values("expR_min", ascending=False)[cols].round(4).to_string(index=False)
          if len(surv) else "  NONE")

    print("\n" + "=" * 104)
    print("E. THE 15m WINNER (wt53_short) RE-TESTED ON THE FULL CYCLE")
    print("=" * 104)
    wcols = [c for c in res.columns if c.startswith("expR_20")]
    for tag in ("wt53_short", "wt53_long"):
        r = real[(real.entry == tag) & (real.mgmt == "atr2.5/rr1.5")]
        if len(r):
            row = r.iloc[0]
            print(f"\n  {tag} | atr2.5/rr1.5   trades={int(row.n_total)}")
            print("   " + "  ".join(f"{c.replace('expR_',''):>12s}" for c in wcols))
            print("   " + "  ".join(f"{row[c]:>12.4f}" if row[c] == row[c] else f"{'n/a':>12s}"
                                    for c in wcols))
    print("\nsaved -> experiments/results_phase9_long_fair.csv")
    return res


if __name__ == "__main__":
    main()
