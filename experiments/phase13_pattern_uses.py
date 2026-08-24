"""
PHASE 13 — every way to actually USE the patterns, tested on the balanced nine-window cycle.

What the Phase-12 screen established, which decides what is worth testing:

  * CANDLESTICK patterns carry no directional information: median excess over the shuffled null is
    about -0.01 at every horizon and the hit rate is 50.1-50.5%. Only `marubozu_bull` is consistent
    (9 of 9 windows, +0.22 excess) and it is a momentum bar, not a reversal shape.
  * CHART patterns have a POSITIVE mean edge but a SUB-50% hit rate (46-47%). Right less than half
    the time, but bigger when right — a positively skewed, breakout-shaped payoff.
  * CHART patterns predict EXPANSION: 12-19% more forward range, and unlike the Phase-7 compression
    result this survives measurement in price terms, so it is not an ATR-denominator artifact.
  * Directional survivors overall: 6 where chance predicts 4.5. Noise.

That combination implies a specific prescription rather than a vague "test everything": if a signal is
right under half the time but the move afterwards is larger than normal, it needs a HIGH
reward-to-risk target and will be destroyed by a tight one. Test 1 is built to falsify exactly that.

Six uses, each against the metronome:
  1. chart-pattern entries across a target ladder      -- does the skew hypothesis hold?
  2. the pattern's OWN stop/target geometry vs ATR     -- is the shape better at defining risk?
  3. candlestick patterns as entries                   -- the screen says no; confirm on cost
  4. patterns as EXITS on an unrelated entry           -- Phase 11's best exit was a candle rule
  5. candlestick confirmation ON a chart pattern       -- the classic confluence claim
  6. fading the patterns that invert                   -- hanging man, three-inside-down, etc.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant.data import get_ohlcv
from quant.indicators import add_candle_patterns, add_chart_patterns, add_emas, add_sr_zones
from quant.strategies.conditions import candle_any, candle_bias, near_resistance, near_support
from experiments.mgmt_sweep import sweep
from experiments.phase9_long_fair_test import WINDOWS

TF = "15m"

# patterns the screen liked, and the ones it says are INVERTED
# Chosen from the phase-12 re-run on the CAUSAL detector, ranked by EXPANSION rather than by
# direction -- expansion is the finding that survived (travel_ratio 1.18-1.42 against a null of
# 0.97), while every one of these has a sub-50% directional hit rate. rising_wedge and
# falling_wedge are excluded: they show no expansion at all (1.01, 0.97) on only 52-80 hits,
# despite rising_wedge having been the headline of the pre-fix Phase-15 run.
GOOD_CHART = ["triple_bottom", "triple_top", "head_shoulders", "symmetrical_triangle",
              "inverse_head_shoulders"]
INVERTED_CDL = ["hanging_man", "three_inside_down", "dragonfly_doji", "advance_block",
                "counterattack_bull", "matching_low"]
MOMENTUM_CDL = ["marubozu_bull", "marubozu_bear"]
CLASSIC_CDL_BULL = ["hammer", "engulfing_bull", "morning_star", "piercing_line", "three_outside_up"]
CLASSIC_CDL_BEAR = ["shooting_star", "engulfing_bear", "evening_star", "dark_cloud_cover",
                    "three_outside_down"]


def load() -> dict:
    out = {}
    for k, (a, b) in WINDOWS.items():
        d = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC")
        d = add_emas(d, [50, 200])
        d = add_sr_zones(d, left=10, right=10, cluster_pct=0.15)
        d = add_candle_patterns(d)
        out[k] = d.reset_index(drop=True)
    return out


def report(res: pd.DataFrame, title: str, note: str = "") -> pd.DataFrame:
    pd.set_option("display.width", 250)
    res = res.copy()
    res["is_null"] = res.entry.str.startswith("NULL")
    bulls = [f"expR_{k}" for k in WINDOWS if k.endswith("bull")]
    bears = [f"expR_{k}" for k in WINDOWS if k.endswith("bear")]
    res["bull"] = res[bulls].mean(axis=1)
    res["bear"] = res[bears].mean(axis=1)
    real = res[~res.is_null]
    nb = res[res.is_null].expR_mean.max() if res.is_null.any() else np.nan
    print("\n" + "=" * 104)
    print(title)
    if note:
        print("   " + note)
    print("=" * 104)
    cols = ["tag", "n_total", "expR_mean", "expR_min", "bull", "bear", "expR_ex_top_min",
            "n_windows_pos", "dd_max"]
    print(real.nlargest(12, "expR_mean")[cols].round(4).to_string(index=False))
    if res.is_null.any():
        print(f"\n   metronome ceiling (expR_mean): {nb:+.4f}   "
              f"real configs beating it: {int((real.expR_mean > nb).sum())} of {len(real)}")
    return res


def main():
    w = load()
    t0 = time.time()
    allres = []

    # ---- 1. the skew hypothesis: chart patterns need a HIGH R:R target -------------
    entries = [(f"chart_{s}", f"chart_pattern_{s}@v1", {"pat_names": GOOD_CHART}, None)
               for s in ("long", "short")]
    entries += [(f"NULL_{s}", f"time_entry_{s}@v1", {"te_every": 60}, None)
                for s in ("long", "short")]
    mgmt = [dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.0, "tp_rr": rr},
                 tag=f"rr{rr:g}") for rr in (0.5, 1.0, 2.0, 3.0, 5.0, 8.0)]
    mgmt += [dict(stop="atr@v1", tp="trailing@v1", params={"stop_atr_mult": 2.0, "trail_pct": 2.0},
                  tag="trail2%")]
    r1 = sweep(w, entries, mgmt, verbose=False)
    r1["test"] = "1_target_ladder"
    allres.append(r1)
    print("\n" + "=" * 104)
    print("TEST 1. THE SKEW HYPOTHESIS — expectancy by target size, chart patterns vs metronome")
    print("   a sub-50% signal with an oversized move should IMPROVE as the target widens")
    print("=" * 104)
    piv = r1.assign(kind=np.where(r1.entry.str.startswith("NULL"), "NULL", "pattern")) \
            .pivot_table(index="mgmt", columns="kind", values="expR_mean", aggfunc="median")
    piv["pattern_minus_null"] = piv["pattern"] - piv["NULL"]
    order = ["rr0.5", "rr1", "rr2", "rr3", "rr5", "rr8", "trail2%"]
    print(piv.reindex([o for o in order if o in piv.index]).round(4).to_string())

    # ---- 2. the pattern's own geometry vs a generic ATR stop -----------------------
    ent2 = [(f"chart_{s}", f"chart_pattern_{s}@v1", {"pat_names": GOOD_CHART}, None)
            for s in ("long", "short")]
    mg2 = [dict(stop="pattern_stop@v1", tp="pattern_target@v1", params={}, tag="own_stop/own_target"),
           dict(stop="pattern_stop@v1", tp="rr@v1", params={"tp_rr": 3.0}, tag="own_stop/rr3"),
           dict(stop="atr@v1", tp="pattern_target@v1", params={"stop_atr_mult": 2.0},
                tag="atr/own_target"),
           dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.0, "tp_rr": 3.0},
                tag="atr/rr3")]
    r2 = sweep(w, ent2, mg2, verbose=False)
    r2["test"] = "2_geometry"
    allres.append(r2)
    print("\n" + "=" * 104)
    print("TEST 2. IS THE PATTERN'S OWN GEOMETRY BETTER AT DEFINING RISK THAN A GENERIC ATR STOP?")
    print("=" * 104)
    print(r2.pivot_table(index="mgmt", columns="entry",
                         values=["expR_mean", "n_total"]).round(4).to_string())

    # ---- 3. candlestick entries ----------------------------------------------------
    ent3 = []
    for s, names in (("long", CLASSIC_CDL_BULL), ("short", CLASSIC_CDL_BEAR)):
        ent3.append((f"classic_{s}", f"candle_pattern_{s}@v1", {"cdl_names": names}, None))
    ent3.append(("marubozu_long", "candle_pattern_long@v1", {"cdl_names": ["marubozu_bull"]}, None))
    ent3.append(("marubozu_short", "candle_pattern_short@v1", {"cdl_names": ["marubozu_bear"]}, None))
    for s in ("long", "short"):
        ent3.append((f"score2_{s}", f"candle_pattern_{s}@v1", {"cdl_min_n": 2, "cdl_net": True}, None))
        ent3.append((f"NULL_{s}", f"time_entry_{s}@v1", {"te_every": 60}, None))
    mg3 = [dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.0, "tp_rr": rr},
                tag=f"rr{rr:g}") for rr in (1.5, 3.0)]
    r3 = sweep(w, ent3, mg3, verbose=False)
    r3["test"] = "3_candles"
    allres.append(r3)
    report(r3, "TEST 3. CANDLESTICK PATTERNS AS ENTRIES",
           "the screen said no directional information; this is the cost-aware confirmation")

    # ---- 4. patterns as EXITS on an unrelated entry ---------------------------------
    ent4 = [("wt53_short", "wt_cross_short@v1", {"wt_level": 53.0}, None),
            ("vwapz2_long", "vwap_rev_long@v1", {"vr_sigma": 2.0}, None),
            ("NULL_short", "time_entry_short@v1", {"te_every": 60}, None)]
    mg4 = [dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.5, "tp_rr": 2.0},
                tag="baseline_rr2"),
           dict(stop="atr@v1", tp="opposite_candle@v1",
                params={"stop_atr_mult": 2.5, "tp_opp_n": 1}, tag="opp_candle"),
           dict(stop="atr@v1", tp="rr@v1",
                params={"stop_atr_mult": 2.5, "tp_rr": 2.0, "sec_opp_n": 1, "sec_arm_on_tp": True},
                secondary="run_past_tp@v1", tag="runpast_opp1")]
    r4 = sweep(w, ent4, mg4, verbose=False)
    r4["test"] = "4_exits"
    allres.append(r4)
    report(r4, "TEST 4. PATTERN-STYLE EXITS ON AN UNRELATED ENTRY",
           "Phase 11's best exit was 'hit the target then leave on the first opposite candle'")

    # ---- 5. candlestick confirmation ON a chart pattern -----------------------------
    conf_long = candle_bias("bull", min_n=1)
    conf_short = candle_bias("bear", min_n=1)
    at_sup = near_support(0.25)
    at_res = near_resistance(0.25)
    ent5 = [("chart_long", "chart_pattern_long@v1", {"pat_names": GOOD_CHART}, None),
            ("chart_long+cdl", "chart_pattern_long@v1", {"pat_names": GOOD_CHART}, conf_long),
            ("chart_long+support", "chart_pattern_long@v1", {"pat_names": GOOD_CHART}, at_sup),
            ("chart_short", "chart_pattern_short@v1", {"pat_names": GOOD_CHART}, None),
            ("chart_short+cdl", "chart_pattern_short@v1", {"pat_names": GOOD_CHART}, conf_short),
            ("chart_short+resist", "chart_pattern_short@v1", {"pat_names": GOOD_CHART}, at_res)]
    mg5 = [dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.0, "tp_rr": 3.0}, tag="rr3")]
    r5 = sweep(w, ent5, mg5, verbose=False)
    r5["test"] = "5_confluence"
    allres.append(r5)
    print("\n" + "=" * 104)
    print("TEST 5. CONFLUENCE — does a candlestick confirmation or an S/R level improve a chart "
          "pattern?")
    print("   judged at CONSTANT TRADE COUNT: a filter that only cuts trades always looks better")
    print("=" * 104)
    print(r5[["entry", "n_total", "expR_mean", "expR_min", "n_windows_pos", "dd_max"]]
          .round(4).to_string(index=False))

    # ---- 6. fade the inverted patterns ---------------------------------------------
    ent6 = [("fade_inverted_long", "candle_pattern_long@v1", {"cdl_names": INVERTED_CDL}, None),
            ("with_inverted_short", "candle_pattern_short@v1", {"cdl_names": INVERTED_CDL}, None),
            ("NULL_long", "time_entry_long@v1", {"te_every": 60}, None)]
    r6 = sweep(w, ent6, mg3, verbose=False)
    r6["test"] = "6_fade"
    allres.append(r6)
    report(r6, "TEST 6. FADING THE PATTERNS THAT INVERT",
           "the screen said hanging man / three-inside-down / dragonfly doji predict the OPPOSITE; "
           "'fade_inverted_long' buys those bearish signals")

    out = pd.concat(allres, ignore_index=True)
    out.to_csv("experiments/results_phase13_pattern_uses.csv", index=False)
    print(f"\ntotal runtime {time.time()-t0:.0f}s")
    print("saved -> experiments/results_phase13_pattern_uses.csv")
    return out


if __name__ == "__main__":
    main()
