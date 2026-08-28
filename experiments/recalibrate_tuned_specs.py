"""
Re-derive TUNED_SPECS after a detector change, by the SAME outcome-blind rule that produced it.

The rule, unchanged: fix the quality knobs (`min_touches=3`, `line_tol_atr=0.6`) from the pooled
Phase-15 scan, then take the STRICTEST remaining combination that still yields at least
`MIN_PER_WINDOW` detections in EVERY one of the nine windows. Selection uses detection COUNT only and
never touches forward returns, because the argmax of a score surface moved for 13 of 13 patterns
under a 21% perturbation and is therefore noise.

Why it needs re-running: the containment, apex and minimum-risk gates reject shapes the old settings
were calibrated to produce, so the same numbers now mean something stricter than intended. Measured
after those gates went in, only 7 of 22 patterns still cleared the floor.

A pattern that clears no combination is REPORTED AS RARE rather than loosened until it appears.
Manufacturing detections by relaxing geometry is how a detector starts finding shapes that are not
there, which is the failure this whole line of work exists to prevent.
"""
from __future__ import annotations

import itertools
import json
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                       # noqa: E402
from quant.indicators import chart_pattern_shapes, CHART_PATTERNS      # noqa: E402
from quant.indicators.chart_patterns import TUNED_COMMON               # noqa: E402
from experiments.phase9_long_fair_test import WINDOWS                  # noqa: E402

TF = "15m"
MIN_PER_WINDOW = 10
OUT = "experiments/results_recalibration.csv"

GRID = dict(atr_mult=[2.5, 3.5, 5.0],
            min_span_bars=[40, 100, 200],
            min_height_atr=[2.0, 4.0])

#: strictness order, most strict first — the tie-break when several combinations clear the floor
_STRICT = ("atr_mult", "min_span_bars", "min_height_atr")


def count_all() -> pd.DataFrame:
    """Detections per (pattern, setting, window). One detection pass per atr_mult, not per pattern."""
    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    print(f"{len(combos)} settings x {len(WINDOWS)} windows", flush=True)
    rows: List[dict] = []
    t0 = time.time()
    for wi, (k, (a, b)) in enumerate(WINDOWS.items(), 1):
        d = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC").reset_index(drop=True)
        for cfg in combos:
            spec = {p: {**TUNED_COMMON, "min_span_bars": cfg["min_span_bars"],
                        "min_height_atr": cfg["min_height_atr"]} for p in CHART_PATTERNS}
            s = chart_pattern_shapes(d, patterns=list(CHART_PATTERNS),
                                     atr_mult=cfg["atr_mult"], settings=spec)
            vc = s["pattern"].value_counts() if len(s) else {}
            for p in CHART_PATTERNS:
                rows.append(dict(window=k, pattern=p, n=int(vc.get(p, 0)), **cfg))
        print(f"  [{wi}/{len(WINDOWS)}] {k} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def pick(counts: pd.DataFrame) -> pd.DataFrame:
    """Per pattern: the strictest setting whose WORST window still clears the floor."""
    g = (counts.groupby(["pattern"] + list(GRID))
                .agg(total=("n", "sum"), min_win=("n", "min"), median=("n", "median"))
                .reset_index())
    out = []
    for p, sub in g.groupby("pattern"):
        ok = sub[sub.min_win >= MIN_PER_WINDOW]
        if len(ok):
            r = ok.sort_values(list(_STRICT), ascending=False).iloc[0]
            out.append(dict(pattern=p, clears=True, **{k: r[k] for k in GRID},
                            total=int(r.total), min_win=int(r.min_win)))
        else:
            r = sub.sort_values("min_win", ascending=False).iloc[0]     # the best available
            out.append(dict(pattern=p, clears=False, **{k: r[k] for k in GRID},
                            total=int(r.total), min_win=int(r.min_win)))
    return pd.DataFrame(out).sort_values(["clears", "min_win"], ascending=[False, False])


def main():
    t0 = time.time()
    counts = count_all()
    counts.to_csv(OUT, index=False)
    best = pick(counts)
    pd.set_option("display.width", 220)

    print("\n" + "=" * 100)
    print(f"STRICTEST SETTING CLEARING >={MIN_PER_WINDOW} DETECTIONS IN EVERY WINDOW")
    print("=" * 100)
    print(best.to_string(index=False))

    clears = best[best.clears]
    rare = best[~best.clears]
    print(f"\n  {len(clears)} of {len(best)} patterns clear the floor")
    if len(rare):
        print(f"  genuinely RARE (no setting clears it; best worst-window count in brackets):")
        for r in rare.itertuples():
            print(f"    {r.pattern:<24} {r.min_win}")

    print("\n" + "=" * 100)
    print("TUNED_SPECS BLOCK — paste into quant/indicators/chart_patterns.py")
    print("=" * 100)
    for r in best.sort_values(["atr_mult", "pattern"], ascending=[False, True]).itertuples():
        print(f'    "{r.pattern}":{" " * max(1, 24 - len(r.pattern))}'
              f'dict(atr_mult={r.atr_mult}, min_span_bars={int(r.min_span_bars)}),'
              f'{"" if r.clears else "   # rare: %d/window at best" % r.min_win}')
    json.dump({r.pattern: dict(atr_mult=float(r.atr_mult),
                               min_span_bars=int(r.min_span_bars),
                               min_height_atr=float(r.min_height_atr))
               for r in best.itertuples()},
              open("experiments/_recalibrated_specs.json", "w"), indent=1)
    print(f"\n  {time.time()-t0:.0f}s   saved -> {OUT}")


if __name__ == "__main__":
    main()
