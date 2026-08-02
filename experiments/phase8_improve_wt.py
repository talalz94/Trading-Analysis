"""
PHASE 8 — can the ONE surviving strategy be improved? S/R zones, EMA bias, HH/HL structure, faster TF.

`wt_cross_short` on BTC 15m (stop 2.5xATR, target 1.5R) was the best result of the whole search:
44% wins, +0.074/+0.148/+0.065R across three regimes, 8-17% drawdown. Everything that might improve
it had been tested only as a STANDALONE ENTRY, never as a filter on top of it:

    S/R zones        tested as a trigger (level_event on sr_support / sr_resistance) -- never as a gate
    HH/HL structure  tested as a trigger (BOS / CHoCH) -- never as a gate
    EMA bias         gated in Phase 4, but over an entry list containing wt53_LONG, not the short
    faster TFs       wt_cross ran at 1m and 5m, but unconditioned

This phase does exactly that: base entry x 16 conditioners x 3 timeframes, with the two controls that
decide it --
  1. the SAME entry unconditioned (does the filter add anything at all?), and
  2. the metronome under the SAME conditioner (a filter that lifts a random entry equally is removing
     costly bars, not finding good ones).

Reported as NET LIFT = (conditioned - base) for the real entry MINUS the same difference for the null.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant.data import get_ohlcv
from quant.data.timeframe import resample_ohlcv
from quant.indicators import add_emas, add_sr_zones, add_zigzag_structure
from quant.strategies.conditions import (Cond, above, below, bounce_off_resistance,
                                         near_resistance, near_support, raw)
from quant.strategies.regimes import trendiness, vol_regime, vol_shock

from experiments.mgmt_sweep import sweep

SPANS = {"w1": ("2025-01-01", "2025-07-29"), "w2": ("2025-07-29", "2026-01-28"),
         "w3": ("2026-01-28", "2026-08-01")}
TFS = ["1min", "5min", "15min"]

# the winner's management, plus two neighbours so a result cannot be a single management cell
MGMT = [dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.5, "tp_rr": 1.5},
             tag="atr2.5/rr1.5"),
        dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.5, "tp_rr": 2.0},
             tag="atr2.5/rr2"),
        dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 1.5, "tp_rr": 1.5},
             tag="atr1.5/rr1.5")]


def structure_down(*, mult: float = 2.5) -> Cond:
    """Market structure is making LOWER highs and LOWER lows (adaptive zigzag trend = -1)."""
    return raw(lambda df: df["zz_trend"].to_numpy(float) < 0, "struct_LH_LL")


def structure_up(*, mult: float = 2.5) -> Cond:
    return raw(lambda df: df["zz_trend"].to_numpy(float) > 0, "struct_HH_HL")


def conditioners() -> list:
    """(name, Cond or None) — everything asked for, plus the regime gates for comparison."""
    return [
        ("none", None),
        # --- S/R zones ---
        ("at_resistance_0.1", near_resistance(0.1)),
        ("at_resistance_0.25", near_resistance(0.25)),
        ("reject_resistance", bounce_off_resistance(0.25)),
        ("far_from_support", ~near_support(0.25)),
        # --- EMA bias (same timeframe) ---
        ("below_ema200", below("close", "ema_200")),
        ("above_ema200", above("close", "ema_200")),
        ("below_ema50", below("close", "ema_50")),
        # --- HH/HL structure ---
        ("struct_down", structure_down()),
        ("struct_up", structure_up()),
        # --- the combinations the theory actually implies ---
        ("resist+struct_down", near_resistance(0.25) & structure_down()),
        ("resist+below_ema", near_resistance(0.25) & below("close", "ema_200")),
        ("struct_down+below_ema", structure_down() & below("close", "ema_200")),
        ("resist+struct+ema", (near_resistance(0.25) & structure_down()
                               & below("close", "ema_200"))),
        # --- regime gates, for reference ---
        ("vol_high", vol_regime("high")),
        ("trending", trendiness("trending")),
        ("no_shock", ~vol_shock(within=60)),
    ]


def load(tf: str) -> dict:
    """Windows with every conditioner's columns precomputed, so each gate is a plain comparison."""
    out = {}
    for k, (a, b) in SPANS.items():
        d = get_ohlcv("BTCUSDT", "1m", start=a, end=b, source="binance", tz="UTC")
        if tf != "1min":
            d = resample_ohlcv(d, tf).reset_index(drop=True)
        d = add_emas(d, [50, 200])
        d = add_sr_zones(d, left=10, right=10, cluster_pct=0.15)
        d = add_zigzag_structure(d, atr_mult=2.5)
        out[k] = d.reset_index(drop=True)
    return out


def main():
    pd.set_option("display.width", 250)
    frames = []
    for tf in TFS:
        w = load(tf)
        entries = []
        for cname, cond in conditioners():
            entries.append((f"wt_short@{cname}", "wt_cross_short@v1", {"wt_level": 53.0}, cond))
            entries.append((f"NULL@{cname}", "time_entry_short@v1", {"te_every": 60}, cond))
        t0 = time.time()
        r = sweep(w, entries, MGMT, verbose=False)
        r["tf"] = tf
        frames.append(r)
        print(f"  {tf:6s} {len(r)} configs in {time.time()-t0:.0f}s "
              f"({sum(len(v) for v in w.values()):,} bars)", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res[["base_entry", "cond"]] = res.entry.str.split("@", n=1, expand=True)
    res["is_null"] = res.base_entry.eq("NULL")
    res.to_csv("experiments/results_phase8_improve_wt.csv", index=False)
    if res.error.astype(bool).any():
        print("ERRORS:", res[res.error.astype(bool)].error.value_counts().head(3).to_dict())

    base = res[res.cond == "none"].set_index(["base_entry", "tf", "mgmt"]).expR_min
    res["lift"] = [r.expR_min - base.get((r.base_entry, r.tf, r.mgmt), np.nan)
                   for r in res.itertuples()]

    print("\n" + "=" * 100)
    print("A. THE BASE, UNCONDITIONED, BY TIMEFRAME  (does a faster TF help on its own?)")
    print("=" * 100)
    b = res[(res.cond == "none")].pivot_table(index=["tf", "mgmt"], columns="base_entry",
                                              values=["expR_min", "n_total"])
    print(b.round(4).to_string())

    print("\n" + "=" * 100)
    print("B. NET LIFT per conditioner = real lift MINUS the null's lift under the SAME filter")
    print("   positive = the filter finds better bars, not just fewer costly ones")
    print("=" * 100)
    rl = (res[~res.is_null & (res.cond != "none")].groupby(["tf", "cond"]).lift.median())
    nl = (res[res.is_null & (res.cond != "none")].groupby(["tf", "cond"]).lift.median())
    adj = pd.DataFrame({"real_lift": rl, "null_lift": nl})
    adj["net"] = adj.real_lift - adj.null_lift
    for tf in TFS:
        print(f"\n  --- {tf} ---")
        print(adj.loc[tf].sort_values("net", ascending=False).round(4).to_string())

    print("\n" + "=" * 100)
    print("C. BEST CONFIGS OVERALL (real entries only)")
    print("=" * 100)
    cols = ["tf", "tag", "n_total", "expR_min", "expR_mean", "expR_ex_top_min", "medR_min",
            "n_windows_pos", "ret_min", "dd_max", "lift"]
    real = res[~res.is_null]
    print(real.nlargest(20, "expR_min")[cols].round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("D. DID ANYTHING BEAT THE UNCONDITIONED 15m BASE?")
    print("=" * 100)
    ref = real[(real.tf == "15min") & (real.cond == "none")
               & (real.mgmt == "atr2.5/rr1.5")].expR_min.iloc[0]
    print(f"  reference (wt_cross_short 15m, atr2.5/rr1.5, no filter): expR_min = {ref:.4f}")
    better = real[(real.expR_min > ref) & (real.n_windows_pos == 3) & (real.n_total >= 300)
                  & (real.expR_ex_top_min > 0)]
    print(f"  configs beating it with >=300 trades, all 3 regimes positive, outlier-proof: "
          f"{len(better)} of {len(real)}")
    print(better.sort_values("expR_min", ascending=False)[cols].round(4).to_string(index=False)
          if len(better) else "  NONE")
    print("\nsaved -> experiments/results_phase8_improve_wt.csv")
    return res


if __name__ == "__main__":
    main()
