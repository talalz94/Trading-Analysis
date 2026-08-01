"""
PHASE 2 — every entry family as the PRIMARY trigger, measured against a null entry.

The control that makes this meaningful: `time_entry` opens trades on a fixed bar schedule with no
signal whatsoever. It runs the same stops, the same targets, the same costs and the same windows. An
entry rule that cannot beat a metronome is not an entry rule — whatever it scores belongs to the exit
logic, which is the exact mistake that made ~150 earlier configs look interesting.

Judged on the WORST regime window (`expR_min`), because a config that only works in one regime is
directional beta. Reported per family: its best config, and the margin over the null's best.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.search_harness import SPREAD_BTC, make_windows, search

# (stop, tp, params) pairs held IDENTICAL across families so only the entry differs
MGMT = [("atr@v1", "rr@v1", {"stop_atr_mult": 1.5, "tp_rr": 2.0}),
        ("atr@v1", "rr@v1", {"stop_atr_mult": 2.5, "tp_rr": 1.5})]

# The BET convention: the direction a trader would actually take on each level interaction.
#   level ABOVE price   break->long   fade->short  sweep->short  retest->long
#   level BELOW price   break->short  fade->long   sweep->long   retest->short
_LEVELS = [("pd_high", "above"), ("pd_low", "below"),
           ("sess_prev_high", "above"), ("sess_prev_low", "below"),
           ("sess_or_high", "above"), ("sess_or_low", "below"),
           ("vwap_u1", "above"), ("vwap_l1", "below"),
           ("sr_resistance", "above"), ("sr_support", "below")]
_BET = {("above", "first_break"): "long", ("above", "fade"): "short",
        ("above", "sweep"): "short", ("above", "retest"): "long",
        ("below", "first_break"): "short", ("below", "fade"): "long",
        ("below", "sweep"): "long", ("below", "retest"): "short"}


def entry_specs() -> list:
    """One spec per (entry family x its own parameters x side). Management is added later."""
    out = []
    for lv, side in _LEVELS:
        for ev in ["first_break", "fade", "sweep", "retest"]:
            d = _BET[(side, ev)]
            out.append((f"level_event_{d}@v1",
                        {"lv_col": lv, "lv_side": side, "lv_event": ev},
                        f"level:{lv}/{ev}/{d}"))
    for d in ("long", "short"):
        for mode in ("bos", "choch"):
            for m in (1.5, 2.5, 4.0):
                out.append((f"bos_{d}@v1", {"bos_mode": mode, "zz_atr_mult": m},
                            f"{mode}:atr{m}/{d}"))
        for z in (2.0, 3.0):
            out.append((f"vol_thrust_{d}@v1", {"vt_z": z}, f"vol:z{z}/{d}"))
        out.append((f"vol_thrust_{d}@v1", {"vt_use_tod": True, "vt_tod_ratio": 3.0},
                   f"vol:tod3/{d}"))
        for m in (1.0, 1.5):
            out.append((f"squeeze_{d}@v1", {"sqz_kc_mult": m}, f"sqz:kc{m}/{d}"))
        for lvl in (45.0, 53.0, 60.0):
            out.append((f"wt_cross_{d}@v1", {"wt_level": lvl}, f"wt:{lvl:g}/{d}"))
        for mode in ("confirmed", "realtime"):
            out.append((f"rsi_div_{d}@v1", {"div_mode": mode}, f"div:{mode}/{d}"))
        for s in (1.5, 2.0, 2.5):
            out.append((f"vwap_rev_{d}@v1", {"vr_sigma": s}, f"vwapz:{s}/{d}"))
        for q in (0.8, 0.9):
            out.append((f"grad_extreme_{d}@v1", {"ge_quantile": q}, f"grad:q{q}/{d}"))
        for n in (100, 200, 400):
            out.append((f"time_entry_{d}@v1", {"te_every": n}, f"NULL:every{n}/{d}"))
    return out


def build_grid() -> list:
    grid = []
    for scen, ep, tag in entry_specs():
        for i, (stop, tp, mp) in enumerate(MGMT):
            grid.append(dict(scenario=scen, stop=stop, tp=tp,
                             params={**ep, **mp}, tag=f"{tag}|m{i}"))
    return grid


def family_of(tag: str) -> str:
    head = tag.split(":")[0]
    return "NULL" if head == "NULL" else head


def main():
    w = make_windows("BTCUSDT", "1m")
    for k, v in w.items():
        print(f"  {k}: {len(v):,} bars  {v.close.iloc[-1]/v.close.iloc[0]-1:+.1%}")
    grid = build_grid()
    print(f"\nPhase 2: {len(grid)} configs x {len(w)} windows", flush=True)
    t0 = time.time()
    res = search(w, grid)
    res["family"] = res.tag.map(family_of)
    res["side"] = res.tag.str.rsplit("/", n=1).str[-1].str.split("|").str[0]
    res.to_csv("experiments/results_phase2_entries.csv", index=False)
    pd.set_option("display.width", 260)
    print(f"done in {time.time()-t0:.0f}s")

    null_best = res[res.family == "NULL"].expR_min.max()
    null_mean = res[res.family == "NULL"].expR_mean.max()
    print("\n" + "=" * 104)
    print(f"NULL BENCHMARK (no signal, fixed schedule): best expR_min = {null_best:.4f}, "
          f"best expR_mean = {null_mean:.4f}")
    print("Every family below must clear those numbers to have contributed anything.")
    print("=" * 104)
    fam = (res.groupby("family")
              .agg(configs=("expR_min", "size"), best_min=("expR_min", "max"),
                   best_mean=("expR_mean", "max"), med_min=("expR_min", "median"),
                   n_pos_all_windows=("n_windows_pos", lambda s: int((s == 3).sum())),
                   trades=("n_total", "median"))
              .reset_index().sort_values("best_min", ascending=False))
    fam["beats_null_min"] = (fam.best_min > null_best)
    fam["beats_null_mean"] = (fam.best_mean > null_mean)
    print(fam.round(4).to_string(index=False))

    print("\n" + "=" * 104)
    print("TOP 25 CONFIGS overall (worst-window expectancy in R, after costs)")
    print("=" * 104)
    cols = ["tag", "family", "n_total", "expR_min", "expR_mean", "expR_ex_top_min",
            "medR_min", "n_windows_pos", "ret_min", "dd_max"]
    print(res.head(25)[cols].to_string(index=False))

    surv = res[(res.expR_min > 0) & (res.n_windows_pos == 3) & (res.n_total >= 300)]
    print("\n" + "=" * 104)
    print(f"POSITIVE IN ALL 3 REGIMES with >=300 trades: {len(surv)} of {len(res)}")
    exp_chance = len(res) * 0.5 ** 3
    print(f"chance alone predicts ~{exp_chance:.0f}")
    print("=" * 104)
    print(surv[cols].to_string(index=False) if len(surv) else "  NONE")
    print("\nsaved -> experiments/results_phase2_entries.csv")
    return res


if __name__ == "__main__":
    main()
