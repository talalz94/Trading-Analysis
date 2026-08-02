"""
PHASE 7b — the compression breakout as an actual strategy, every variant.

The screen (phase7) split the theory in two and gave a split answer:
  * in ABSOLUTE terms quiet periods are followed by SMALLER moves (57 vs 103 bps) — volatility is
    persistent, not mean-reverting, so "stagnant then a big move" is false as literally stated;
  * relative to the COMPRESSED RANGE, forward travel is ~1.8x larger, of which only +4..14% survives
    the shuffled-null control. A breakout risks the range, so that is the number that matters;
  * the 200-EMA cross carries NO direction: hit rates 46-53%, split along long/short lines, which is
    the same residual short-bias artifact diagnosed in Phase 1.

No direction + a possible range-relative expansion means the only honest form of this idea is a
TWO-SIDED BREAKOUT (a straddle): take both sides, let the size of the move pay for the whipsaws.
Whether that pays cannot be settled by a screen, so here it is backtested across every variant.

THE CONTROL THAT DECIDES IT: `cb_quiet_q = 1.0` runs the identical breakout with the compression
filter switched OFF. If the quiet-filtered version does not beat it, the compression contributes
nothing and the whole theory reduces to "breakouts", which was already tested and killed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.mgmt_sweep import mgmt_grid, sweep
from experiments.search_harness import make_windows

QUIET = [0.1, 0.2, 0.35, 1.0]          # 1.0 = compression filter OFF (the control)
SHORTS = [30, 60, 120]
TRIGGERS = ["range_break", "ema_cross"]


def build_entries() -> list:
    out = []
    for side in ("long", "short"):
        for trig in TRIGGERS:
            for fade in (False, True):
                for q in QUIET:
                    for s in SHORTS:
                        tag = (f"{trig[:5]}{'F' if fade else 'B'}_s{s}_"
                               f"{'ALL' if q >= 1 else f'q{q:g}'}_{side}")
                        out.append((tag, f"compression_break_{side}@v1",
                                    {"cb_trigger": trig, "cb_fade": fade,
                                     "cb_quiet_q": q, "cb_short": s}, None))
    for n in (200, 400):
        for side in ("long", "short"):
            out.append((f"NULL_e{n}_{side}", f"time_entry_{side}@v1", {"te_every": n}, None))
    return out


# stop = the OTHER SIDE of the broken range (the natural breakout stop), plus ATR alternatives
MGMT = (mgmt_grid([("high_of_n@v1", {"stop_n": 60}), ("atr@v1", {"stop_atr_mult": 1.5})],
                  [("rr@v1", {"tp_rr": 2.0}), ("rr@v1", {"tp_rr": 3.0}),
                   ("trailing@v1", {"trail_pct": 0.5})],
                  [0, 60]))


def main():
    w = make_windows("BTCUSDT", "1m",
                     spans={"w1": ("2025-01-01", "2025-07-29"),
                            "w2": ("2025-07-29", "2026-01-28"),
                            "w3": ("2026-01-28", "2026-08-01")})
    entries = build_entries()
    print(f"Phase 7b: {len(entries)} entry variants x {len(MGMT)} management "
          f"= {len(entries)*len(MGMT)} configs x {len(w)} windows", flush=True)
    t0 = time.time()
    res = sweep(w, entries, MGMT, verbose=False)
    res["is_null"] = res.entry.str.startswith("NULL")
    parts = res.entry.str.extract(r"^(?P<trig>\w{5})(?P<mode>[BF])_s(?P<short>\d+)_"
                                  r"(?P<q>ALL|q[\d.]+)_(?P<side>long|short)$")
    res = pd.concat([res, parts], axis=1)
    res.to_csv("experiments/results_phase7b_compression.csv", index=False)
    pd.set_option("display.width", 260)
    print(f"done in {time.time()-t0:.0f}s")
    if res.error.astype(bool).any():
        print("ERRORS:", res[res.error.astype(bool)].error.value_counts().head(3).to_dict())

    real = res[~res.is_null]
    nb = res[res.is_null].expR_min.max()
    print("\n" + "=" * 100)
    print(f"NULL CEILING (metronome, best of all management): expR_min {nb:.4f}")
    print("=" * 100)

    print("\nA. DOES THE COMPRESSION FILTER ADD ANYTHING?  median expR_min by quiet threshold")
    print("   'ALL' is the SAME breakout with the filter off. Tighter q must beat it.")
    piv = real.pivot_table(index="q", columns="mode", values="expR_min", aggfunc="median")
    piv.columns = ["Breakout" if c == "B" else "Fade" for c in piv.columns]
    print(piv.round(4).to_string())
    print("\n   same, best rather than median:")
    print(real.pivot_table(index="q", columns="mode", values="expR_min",
                           aggfunc="max").round(4).to_string())

    print("\nB. BY TRIGGER x MODE x SIDE (median expR_min)")
    print(real.pivot_table(index=["trig", "mode"], columns="side",
                           values="expR_min", aggfunc="median").round(4).to_string())

    print("\nC. BY RANGE LENGTH (cb_short)")
    print(real.pivot_table(index="short", columns="q", values="expR_min",
                           aggfunc="median").round(4).to_string())

    print("\n" + "=" * 100)
    print("D. TOP 20 CONFIGS")
    print("=" * 100)
    cols = ["tag", "n_total", "expR_min", "expR_mean", "expR_ex_top_min", "medR_min",
            "n_windows_pos", "ret_min", "dd_max"]
    print(real.head(20)[cols].to_string(index=False))

    surv = real[(real.expR_min > 0) & (real.n_windows_pos == 3) & (real.n_total >= 300)
                & (real.expR_ex_top_min > 0) & (real.expR_min > nb)]
    print("\n" + "=" * 100)
    print(f"SURVIVORS (all 3 regimes, >=300 trades, outlier-proof, beat the null ceiling): "
          f"{len(surv)} of {len(real)}")
    print(f"chance predicts ~{len(real)*0.125:.0f} on the regime test alone")
    print("=" * 100)
    print(surv[cols].to_string(index=False) if len(surv) else "  NONE")

    # the paired comparison: same entry structure, filter ON vs OFF
    print("\n" + "=" * 100)
    print("E. PAIRED — identical breakout, compression filter ON (q=0.2) vs OFF, same management")
    print("=" * 100)
    key = ["trig", "mode", "short", "side", "mgmt"]
    on = real[real.q == "q0.2"].set_index(key).expR_min
    off = real[real.q == "ALL"].set_index(key).expR_min
    pair = pd.DataFrame({"filter_on": on, "filter_off": off}).dropna()
    pair["delta"] = pair.filter_on - pair.filter_off
    print(f"  pairs: {len(pair)}   compression HELPED in {int((pair.delta > 0).sum())} "
          f"({100*(pair.delta > 0).mean():.1f}%)   median delta {pair.delta.median():+.4f}")
    print("\n  best 10 by delta:")
    print(pair.sort_values("delta", ascending=False).head(10).round(4).to_string())
    print("\nsaved -> experiments/results_phase7b_compression.csv")
    return res


if __name__ == "__main__":
    main()
