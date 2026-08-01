"""
PHASE 1c — the level screen, done correctly: REAL minus SHUFFLED NULL.

Two earlier attempts both measured artifacts:
  * whole-window drift adjustment  -> residual short bias (every short cell +, every long cell -)
  * day-matched control            -> within-day range-position bias; `null_control.py` showed the
                                      identical "edge" appears on bar-shuffled data with no structure

So the control has to be the screen's own output on data that provably has no level effect. For each
symbol/timeframe the identical pipeline runs on the real series and on 3 within-day bar shuffles;
the reported edge is `real - mean(null)`, and `z` is that difference in units of the null's spread.
A level effect is real only if it beats its own null, in both directions, across instruments.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                       # noqa: E402
from quant.data.timeframe import resample_ohlcv                        # noqa: E402
from experiments.phase1b_matched_screen import MIRRORS, screen_matched  # noqa: E402
from experiments.null_control import shuffle_within_day                # noqa: E402

RUNS = [("BTCUSDT", "binance", "spot", 1000.0, "1m", 30),
        ("BTCUSDT", "binance", "spot", 1000.0, "5m", 6),
        ("BTCUSDT", "binance", "spot", 1000.0, "15m", 2),
        ("XAUUSD", "dukascopy", "cfd", 50.0, "1m", 30),
        ("XAUUSD", "dukascopy", "cfd", 50.0, "5m", 6),
        ("PAXGUSDT", "binance", "spot", 50.0, "5m", 6)]
SPANS = {"a": ("2025-01-01", "2025-07-29"), "b": ("2025-07-29", "2026-01-28"),
         "c": ("2026-01-28", "2026-07-29")}
HS = [15, 30, 60, 120]
KEY = ["level", "side", "event", "bet"]
N_SEEDS = 3


def run_one(sym, src, mkt, step, tf, orb):
    base = {}
    for k, (a, b) in SPANS.items():
        d = get_ohlcv(sym, "1m", start=a, end=b, source=src, market=mkt, tz="UTC")
        base[k] = (d if tf == "1m"
                   else resample_ohlcv(d, tf.replace("m", "min")).reset_index(drop=True))
    real = screen_matched(base, step, or_bars=orb).set_index(KEY)
    nulls = []
    for s in range(N_SEEDS):
        fake = {k: shuffle_within_day(v, seed=s * 97 + i) for i, (k, v) in enumerate(base.items())}
        nulls.append(screen_matched(fake, step, or_bars=orb).set_index(KEY))
    out = real[[]].copy()
    for h in HS:
        col = f"h{h}"
        stack = pd.concat([n[col] for n in nulls], axis=1)
        nm, ns = stack.mean(axis=1), stack.std(axis=1).replace(0, np.nan)
        out[f"real_{col}"] = real[col].round(4)
        out[f"null_{col}"] = nm.round(4)
        out[f"exc_{col}"] = (real[col] - nm).round(4)
        out[f"z_{col}"] = ((real[col] - nm) / ns).round(2)
        out[f"n_{col}"] = real.get(f"{col}_n")
    return out.reset_index()


def main():
    frames = []
    for r in RUNS:
        t0 = time.time()
        f = run_one(*r)
        f["symbol"], f["tf"] = r[0], r[4]
        frames.append(f)
        print(f"  {r[0]:9s} {r[4]:4s}  done in {time.time() - t0:.0f}s", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv("experiments/results_phase1c_null_diff.csv", index=False)
    pd.set_option("display.width", 260)

    print("\n" + "=" * 100)
    print("A. BY INTERACTION TYPE — edge NET OF ITS OWN NULL, pooled over levels/symbols/TFs")
    print("=" * 100)
    agg = (res.groupby(["event", "bet"])
              .agg(cells=("exc_h30", "size"),
                   exc_h15=("exc_h15", "median"), exc_h30=("exc_h30", "median"),
                   exc_h60=("exc_h60", "median"), exc_h120=("exc_h120", "median"),
                   pct_pos_h30=("exc_h30", lambda s: round(100 * (s > 0).mean(), 1)),
                   pct_pos_h60=("exc_h60", lambda s: round(100 * (s > 0).mean(), 1)))
              .reset_index().sort_values("exc_h30", ascending=False))
    print(agg.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("B. SYMMETRY on the NULL-DIFFERENCED edge — both sides must beat their own null")
    print("=" * 100)
    rows = []
    for ev in res.event.unique():
        for hi, lo in MIRRORS:
            a = res[(res.level == hi) & (res.event == ev)]
            b = res[(res.level == lo) & (res.event == ev)]
            if not len(a) or not len(b):
                continue
            for h in (30, 60):
                ma, mb = a[f"exc_h{h}"].median(), b[f"exc_h{h}"].median()
                if ma != ma or mb != mb:
                    continue
                rows.append(dict(event=ev, pair=f"{hi}/{lo}", h=h,
                                 short_side=round(ma, 4), long_side=round(mb, 4),
                                 both_pos=bool(ma > 0 and mb > 0), sym_edge=round(min(ma, mb), 4)))
    sd = pd.DataFrame(rows).sort_values("sym_edge", ascending=False)
    print(sd.to_string(index=False))
    print(f"\n  BOTH sides beat null: {int(sd.both_pos.sum())}/{len(sd)}   (chance ~ {len(sd)//4})")

    print("\n" + "=" * 100)
    print("C. INDIVIDUAL CELLS worth a backtest: positive excess in >=5 of 6 runs, |z|>2")
    print("=" * 100)
    surv = []
    for h in (30, 60):
        g = (res.groupby(KEY)
                .agg(runs=(f"exc_h{h}", "size"),
                     pos=(f"exc_h{h}", lambda s: int((s > 0).sum())),
                     med_exc=(f"exc_h{h}", "median"), med_z=(f"z_h{h}", "median"),
                     n=(f"n_h{h}", "sum")).reset_index())
        g["h"] = h
        surv.append(g[(g.runs >= 5) & (g.pos >= 5) & (g.med_z.abs() > 2) & (g.med_exc > 0)])
    s = pd.concat(surv, ignore_index=True).sort_values("med_exc", ascending=False)
    print(s.round(4).to_string(index=False) if len(s) else "  NONE — no level/event beats its null")
    print("\nsaved -> experiments/results_phase1c_null_diff.csv")
    return res


if __name__ == "__main__":
    main()
