"""
PHASE 11 — the three-part close, swept properly on the balanced nine-window cycle.

The proposed structure:
  1. plain stop, no target hit                       (owned by the StopMethod)
  2. salvage: while in profit, bail on m opposite candles once the trade has interacted with the EMA
  3. once the fixed TP is reached, don't just exit -- go to breakeven, ratchet, trail, or hand the
     runner to a candle rule, to squeeze more out of the winners

What the Phase-10 diagnostic already established, which decides what is worth sweeping:

  * Fixed targets already capture 87-93% of the favourable move. Trailing captures only 37%.
    There is very little to gain by "holding longer" -- trailing gives back two thirds.
  * 44.8% of ALL trades reach +1R at some point, and **30.1% of those end up LOSING**. That is the
    actual leak: not winners cut short, but winners turning into losers.
  * Continuation after exit is identical for the NULL entry (cont30 1.07 vs 1.00), so "the move kept
    going" is the max-of-a-random-walk effect, not a property of the strategy.

So the sweep is weighted towards protecting the +1R trades (breakeven, partial scale) rather than
towards chasing runners, and every variant is run against the metronome.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant.data import get_ohlcv
from quant.indicators import add_ema_gradient, add_emas, add_sr_zones
from experiments.mgmt_sweep import sweep
from experiments.phase9_long_fair_test import WINDOWS

TF = "15m"

ENTRIES = [
    ("wt53_short", "wt_cross_short@v1", {"wt_level": 53.0}, None),
    ("wt53_long", "wt_cross_long@v1", {"wt_level": 53.0}, None),
    ("vwapz2_long", "vwap_rev_long@v1", {"vr_sigma": 2.0}, None),
    ("NULL_short", "time_entry_short@v1", {"te_every": 60}, None),
    ("NULL_long", "time_entry_long@v1", {"te_every": 60}, None),
]

STOP = ("atr@v1", {"stop_atr_mult": 2.5})


def mgmt() -> list:
    """Baselines first, then the three-part variants."""
    st, sp = STOP
    out = []
    # --- baselines: plain fixed targets and a plain trail ---
    for rr in (1.5, 2.0, 3.0):
        out.append(dict(stop=st, tp="rr@v1", params={**sp, "tp_rr": rr}, tag=f"base_rr{rr:g}"))
    for tr in (0.5, 1.0, 2.0):
        out.append(dict(stop=st, tp="trailing@v1", params={**sp, "trail_pct": tr},
                        tag=f"base_trail{tr:g}%"))
    # --- part 3a: breakeven at N R, then a final target ---
    for be in (0.5, 1.0, 1.5):
        for rr in (2.0, 3.0):
            out.append(dict(stop=st, tp="be_ladder@v1",
                            params={**sp, "be_at_rr": be, "be_scale_pct": 0.0, "tp_rr": rr},
                            tag=f"be{be:g}/rr{rr:g}"))
    # --- part 3b: scale out at N R AND move to breakeven ---
    for be in (1.0, 1.5):
        for pct in (33.0, 50.0):
            out.append(dict(stop=st, tp="be_ladder@v1",
                            params={**sp, "be_at_rr": be, "be_scale_pct": pct, "tp_rr": 3.0},
                            tag=f"scale{pct:g}@{be:g}/rr3"))
    # --- part 3c: breakeven then RATCHET a trail behind the high-water mark ---
    for be in (1.0,):
        for tr in (0.5, 1.0, 2.0):
            out.append(dict(stop=st, tp="be_ladder@v1",
                            params={**sp, "be_at_rr": be, "be_scale_pct": 0.0, "tp_rr": 6.0,
                                    "be_trail_pct": tr},
                            tag=f"be{be:g}+trail{tr:g}%"))
    # --- part 3d: reach the TP but DON'T close -- hand the runner to a candle rule ---
    for rr in (1.5, 2.0):
        for n in (1, 2):
            out.append(dict(stop=st, tp="rr@v1",
                            params={**sp, "tp_rr": rr, "sec_opp_n": n, "sec_arm_on_tp": True},
                            secondary="run_past_tp@v1", tag=f"runpast_rr{rr:g}_opp{n}"))
    # --- part 2: the salvage rule (arm on EMA interaction, bail on m opposite candles in profit) ---
    for n in (1, 2):
        for rr in (2.0, 3.0):
            out.append(dict(stop=st, tp="rr@v1",
                            params={**sp, "tp_rr": rr, "sec_opp_n": n,
                                    "sec_arm_mode": "near", "sec_arm_tol_pct": 0.1},
                            secondary="shaky_trend_bail@v1", tag=f"salvage_opp{n}_rr{rr:g}"))
    # --- parts 2 AND 3 together ---
    for rr in (2.0, 3.0):
        out.append(dict(stop=st, tp="be_ladder@v1",
                        params={**sp, "be_at_rr": 1.0, "be_scale_pct": 50.0, "tp_rr": rr,
                                "sec_opp_n": 1, "sec_arm_mode": "near", "sec_arm_tol_pct": 0.1},
                        secondary="shaky_trend_bail@v1", tag=f"scale50+be+salvage/rr{rr:g}"))
    return out


def load() -> dict:
    out = {}
    for k, (a, b) in WINDOWS.items():
        d = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC")
        d = add_emas(d, [50, 200])
        d = add_sr_zones(d, left=10, right=10, cluster_pct=0.15)
        # the salvage rule (shaky_trend_bail) arms off the smoothed EMA line, so that column has to
        # exist even though none of these ENTRIES builds it
        d = add_ema_gradient(d, period=200, smooth=9)
        out[k] = d.reset_index(drop=True)
    return out


def main():
    pd.set_option("display.width", 260)
    w = load()
    m = mgmt()
    print(f"Phase 11: {len(ENTRIES)} entries x {len(m)} exit designs x {len(w)} windows "
          f"= {len(ENTRIES)*len(m)} configs", flush=True)
    t0 = time.time()
    res = sweep(w, ENTRIES, m, verbose=False)
    print(f"done in {time.time()-t0:.0f}s")
    res["is_null"] = res.entry.str.startswith("NULL")
    bulls = [f"expR_{k}" for k in w if k.endswith("bull")]
    bears = [f"expR_{k}" for k in w if k.endswith("bear")]
    res["bull_mean"] = res[bulls].mean(axis=1)
    res["bear_mean"] = res[bears].mean(axis=1)
    res.to_csv("experiments/results_phase11_exit_ladder.csv", index=False)
    if res.error.astype(bool).any():
        print("ERRORS:", res[res.error.astype(bool)].error.value_counts().head(3).to_dict())

    real = res[~res.is_null]
    print("\n" + "=" * 104)
    print("A. EXIT DESIGN RANKED — median over the real entries, and the SAME for the metronome")
    print("=" * 104)
    r = real.groupby("mgmt").agg(expR_mean=("expR_mean", "median"),
                                 expR_min=("expR_min", "median"),
                                 win_windows=("n_windows_pos", "median"),
                                 trades=("n_total", "median"),
                                 dd=("dd_max", "median"))
    nl = res[res.is_null].groupby("mgmt").expR_mean.median().rename("null_expR_mean")
    t = r.join(nl)
    t["net_vs_null"] = t.expR_mean - t.null_expR_mean
    print(t.sort_values("expR_mean", ascending=False).round(4).to_string())

    print("\n" + "=" * 104)
    print("B. DID ANY EXIT DESIGN BEAT ITS OWN BASELINE?  (same entry, plain rr target)")
    print("=" * 104)
    base = real[real.mgmt == "base_rr3"].set_index("entry").expR_mean
    real2 = real.copy()
    real2["vs_base_rr3"] = real2.expR_mean - real2.entry.map(base)
    g = (real2.groupby("mgmt").vs_base_rr3.median().sort_values(ascending=False))
    print(g.round(4).to_string())

    print("\n" + "=" * 104)
    print("C. TOP 15 CONFIGS BY MEAN EXPECTANCY ACROSS ALL NINE WINDOWS")
    print("=" * 104)
    cols = ["tag", "n_total", "expR_mean", "expR_min", "bull_mean", "bear_mean",
            "expR_ex_top_min", "medR_min", "n_windows_pos", "dd_max"]
    print(real.nlargest(15, "expR_mean")[cols].round(4).to_string(index=False))

    print("\n" + "=" * 104)
    print("D. SURVIVORS — positive in EVERY one of the nine windows")
    print("=" * 104)
    nb = res[res.is_null].expR_min.max()
    surv = real[(real.expR_min > 0) & (real.n_total >= 300) & (real.expR_ex_top_min > 0)
                & (real.expR_min > nb)]
    print(f"  null ceiling {nb:+.4f}   survivors {len(surv)} of {len(real)}")
    print(surv.sort_values("expR_min", ascending=False)[cols].round(4).to_string(index=False)
          if len(surv) else "  NONE")
    print("\nsaved -> experiments/results_phase11_exit_ladder.csv")
    return res


if __name__ == "__main__":
    main()
