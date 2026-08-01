"""
PHASE 3 — management: stops x targets x TIME STOP, on the entries that led Phase 2.

Phase 2's finding sets the question. Every family lost after costs and `medR_min` sat at about -1.0,
meaning the median trade was a full stop-out; the null entry scored -0.045R, so roughly that much is
pure cost drag. Management can only help in one of three ways: fewer and better trades, a payoff big
enough that costs stop mattering, or getting out of losers sooner. The **time stop** targets the
third and has never been tested here — it is also the most direct expression of "quick trades".

The null entries stay in the sweep. If a management scheme lifts the metronome as much as it lifts a
real signal, the improvement belongs to the management, not the entry, and it is not an edge.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.mgmt_sweep import mgmt_grid, sweep
from experiments.search_harness import make_windows

# Entry configs: the Phase-2 leaders, kept diverse across families, plus both null controls.
ENTRIES = [
    ("choch_long", "bos_long@v1", {"bos_mode": "choch", "zz_atr_mult": 4.0}, None),
    ("choch_short", "bos_short@v1", {"bos_mode": "choch", "zz_atr_mult": 4.0}, None),
    ("pdlow_fade_long", "level_event_long@v1",
     {"lv_col": "pd_low", "lv_side": "below", "lv_event": "fade"}, None),
    ("orlow_break_short", "level_event_short@v1",
     {"lv_col": "sess_or_low", "lv_side": "below", "lv_event": "first_break"}, None),
    ("vwapz2_short", "vwap_rev_short@v1", {"vr_sigma": 2.0}, None),
    ("vwapz2_long", "vwap_rev_long@v1", {"vr_sigma": 2.0}, None),
    ("rtdiv_short", "rsi_div_short@v1", {"div_mode": "realtime"}, None),
    ("wt53_long", "wt_cross_long@v1", {"wt_level": 53.0}, None),
    ("NULL_long", "time_entry_long@v1", {"te_every": 200}, None),
    ("NULL_short", "time_entry_short@v1", {"te_every": 200}, None),
]

STOPS = [("atr@v1", {"stop_atr_mult": 1.0}), ("atr@v1", {"stop_atr_mult": 1.5}),
         ("atr@v1", {"stop_atr_mult": 2.5}),
         ("pct@v1", {"stop_pct": 0.2}), ("pct@v1", {"stop_pct": 0.5})]
TPS = [("rr@v1", {"tp_rr": 1.0}), ("rr@v1", {"tp_rr": 1.5}), ("rr@v1", {"tp_rr": 2.0}),
       ("rr@v1", {"tp_rr": 3.0}), ("trailing@v1", {"trail_pct": 0.5}),
       ("rr_partials@v1", {"tp_partial_rr1": 1.0, "tp_partial_pct1": 50.0, "tp_rr": 2.5})]
TIME_STOPS = [0, 15, 30, 60, 120]


def main():
    w = make_windows("BTCUSDT", "1m")
    grid = mgmt_grid(STOPS, TPS, TIME_STOPS)
    print(f"Phase 3: {len(ENTRIES)} entries x {len(grid)} management combos "
          f"= {len(ENTRIES)*len(grid)} configs x {len(w)} windows", flush=True)
    t0 = time.time()
    res = sweep(w, ENTRIES, grid)
    res.to_csv("experiments/results_phase3_mgmt.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"done in {time.time()-t0:.0f}s")
    if res.error.astype(bool).any():
        print("\nERRORS:", res[res.error.astype(bool)].error.value_counts().head().to_dict())

    nulls = res[res.entry.str.startswith("NULL")]
    real = res[~res.entry.str.startswith("NULL")]
    nb_min, nb_mean = nulls.expR_min.max(), nulls.expR_mean.max()
    print("\n" + "=" * 100)
    print(f"NULL CEILING across ALL management: expR_min {nb_min:.4f}, expR_mean {nb_mean:.4f}")
    print("=" * 100)

    print("\nTIME STOP — does capping the holding period help? (median expR_min per setting)")
    ts = (res.assign(tstop=res.mgmt.str.extract(r"/t(\d+)$")[0].fillna("0").astype(int))
             .groupby(["tstop"])
             .agg(configs=("expR_min", "size"), med_min=("expR_min", "median"),
                  best_min=("expR_min", "max"), med_mean=("expR_mean", "median"),
                  med_trades=("n_total", "median"), med_dd=("dd_max", "median"))
             .reset_index())
    print(ts.round(4).to_string(index=False))

    print("\nTIME STOP x ENTRY (median expR_min) — a real entry should improve more than the null")
    piv = (res.assign(tstop=res.mgmt.str.extract(r"/t(\d+)$")[0].fillna("0").astype(int))
              .pivot_table(index="entry", columns="tstop", values="expR_min", aggfunc="median"))
    print(piv.round(4).to_string())

    print("\nSTOP / TARGET families (median expR_min over everything else)")
    for key, pat in [("stop", r"^([a-z_]+[0-9.]*)/"), ("target", r"/([a-z_]+[0-9.]*)")]:
        g = (res.assign(k=res.mgmt.str.extract(pat)[0])
                .groupby("k").agg(configs=("expR_min", "size"), med=("expR_min", "median"),
                                  best=("expR_min", "max")).reset_index()
                .sort_values("med", ascending=False))
        print(f"\n  by {key}:")
        print(g.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("TOP 20 CONFIGS")
    print("=" * 100)
    cols = ["tag", "n_total", "expR_min", "expR_mean", "expR_ex_top_min", "medR_min",
            "n_windows_pos", "ret_min", "dd_max"]
    print(res.head(20)[cols].to_string(index=False))

    surv = real[(real.expR_min > 0) & (real.n_windows_pos == 3) & (real.n_total >= 300)
                & (real.expR_ex_top_min > 0) & (real.expR_min > nb_min)]
    print("\n" + "=" * 100)
    print(f"SURVIVORS (positive in all 3 regimes, >=300 trades, survives removing each window's "
          f"best trade, and beats the null ceiling): {len(surv)} of {len(real)}")
    print(f"chance alone predicts ~{len(real) * 0.5 ** 3:.0f} on the regime test alone")
    print("=" * 100)
    print(surv[cols].to_string(index=False) if len(surv) else "  NONE")
    print("\nsaved -> experiments/results_phase3_mgmt.csv")
    return res


if __name__ == "__main__":
    main()
