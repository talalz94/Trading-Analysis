"""
PHASE 14 — the gates, on the one pattern result that survived Phase 13.

The finding to break: chart-pattern entries are positive at EVERY target size and improve
monotonically as the target widens (+0.0026 at 0.5R to +0.0640 at 8R), while the metronome is
negative almost everywhere. That is exactly what the screen predicted from a sub-50% hit rate with
12-19% more forward range, so the mechanism and the backtest agree — which is the only reason it is
worth the gate battery rather than the bin.

Every gate that has killed something in this project, applied at once:
  A. per-window detail          -- is it a few windows carrying it?
  B. trim-top-K                 -- deleting the best trades; this killed every Phase-3 survivor
  C. cost sensitivity           -- spread x1 / x2 / x4
  D. cross-instrument           -- gold; this is what retracted the Phase-5 winner
  E. bootstrap per window       -- IID and moving-block lower bounds, never pooled
  F. parameter plateau          -- across zigzag sensitivity and pattern selection
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data import get_ohlcv
from quant.indicators import add_emas
from experiments.mgmt_sweep import EntryBundle
from experiments.phase6_bootstrap import boot_ci
from experiments.phase6_verify import R_of, trimmed
from experiments.phase9_long_fair_test import WINDOWS
from experiments.phase13_pattern_uses import GOOD_CHART
from experiments.search_harness import CFG_LEVERED, SPREAD_BTC

TF = "15m"
SPREAD_GOLD = 0.25
CONFIGS = [
    ("chart_long_rr8", "chart_pattern_long@v1", {"pat_names": GOOD_CHART},
     "atr@v1", "rr@v1", {"stop_atr_mult": 2.0, "tp_rr": 8.0}),
    ("chart_short_rr8", "chart_pattern_short@v1", {"pat_names": GOOD_CHART},
     "atr@v1", "rr@v1", {"stop_atr_mult": 2.0, "tp_rr": 8.0}),
    ("chart_long_trail2", "chart_pattern_long@v1", {"pat_names": GOOD_CHART},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 2.0, "trail_pct": 2.0}),
    ("chart_short_trail2", "chart_pattern_short@v1", {"pat_names": GOOD_CHART},
     "atr@v1", "trailing@v1", {"stop_atr_mult": 2.0, "trail_pct": 2.0}),
    ("NULL_long_rr8", "time_entry_long@v1", {"te_every": 60},
     "atr@v1", "rr@v1", {"stop_atr_mult": 2.0, "tp_rr": 8.0}),
    ("NULL_short_rr8", "time_entry_short@v1", {"te_every": 60},
     "atr@v1", "rr@v1", {"stop_atr_mult": 2.0, "tp_rr": 8.0}),
]


def windows(symbol="BTCUSDT", source="binance", market="spot"):
    out = {}
    for k, (a, b) in WINDOWS.items():
        try:
            d = get_ohlcv(symbol, TF, start=a, end=b, source=source, market=market, tz="UTC")
        except Exception:
            continue
        if len(d) < 500:
            continue
        out[k] = add_emas(d, [200]).reset_index(drop=True)
    return out


def run_all(w, spread):
    cfg = {**CFG_LEVERED, "spread": spread}
    store = {}
    for tag, skey, ep, stop, tp, mp in CONFIGS:
        per = {}
        for k, d in w.items():
            try:
                per[k] = EntryBundle(d, skey, ep).run(stop, tp, mp, cfg)
            except Exception:
                continue
        store[tag] = per
    return store


def main():
    pd.set_option("display.width", 250)
    w = windows()
    store = run_all(w, SPREAD_BTC)

    print("=" * 104)
    print("A. PER-WINDOW DETAIL — expectancy in every one of the nine regimes")
    print("=" * 104)
    rows = []
    for tag, per in store.items():
        rec = {"cfg": tag, "trades": sum(len(r.trades) for r in per.values())}
        for k, r in per.items():
            R = R_of(r)
            rec[k.replace("_bear", "-").replace("_bull", "+")] = (
                round(float(R.mean()), 3) if len(R) >= 20 else np.nan)
        rows.append(rec)
    A = pd.DataFrame(rows)
    print(A.to_string(index=False))
    wc = [c for c in A.columns if c not in ("cfg", "trades")]
    A["pos"] = (A[wc] > 0).sum(axis=1)
    print("\n  windows positive: " + ", ".join(f"{r.cfg} {int(r.pos)}/{len(wc)}"
                                               for r in A.itertuples()))

    print("\n" + "=" * 104)
    print("B. TRIM-TOP-K — expectancy after deleting the best trades (worst window shown)")
    print("=" * 104)
    rows = []
    for tag, per in store.items():
        Rs = [R_of(r) for r in per.values()]
        Rs = [R for R in Rs if len(R) >= 40]
        if not Rs:
            continue
        rec = {"cfg": tag, "n_min": min(len(R) for R in Rs)}
        for k in (0, 1, 5, 10, 25):
            rec[f"drop{k}"] = round(min(trimmed(R, k) for R in Rs), 4)
        rec["drop5%"] = round(min(trimmed(R, max(1, len(R) * 5 // 100)) for R in Rs), 4)
        allR = np.concatenate(Rs)
        srt = np.sort(allR)[::-1]
        rec["top1pct_share"] = round(100 * srt[:max(1, len(allR) // 100)].sum() / srt.sum(), 1)
        rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 104)
    print("C. COST SENSITIVITY — mean expectancy at spread x1 / x2 / x4")
    print("=" * 104)
    rows = []
    for mult in (1, 2, 4):
        st = run_all(w, SPREAD_BTC * mult) if mult > 1 else store
        for tag, per in st.items():
            vals = [R_of(r).mean() for r in per.values() if len(R_of(r)) >= 40]
            rows.append(dict(cfg=tag, spread=f"x{mult}",
                             expR_mean=round(float(np.mean(vals)), 4) if vals else np.nan,
                             expR_min=round(float(np.min(vals)), 4) if vals else np.nan))
    print(pd.DataFrame(rows).pivot(index="cfg", columns="spread",
                                   values=["expR_mean", "expR_min"]).round(4).to_string())

    print("\n" + "=" * 104)
    print("D. CROSS-INSTRUMENT — the identical rules on GOLD")
    print("=" * 104)
    wg = windows("XAUUSD", "dukascopy", "cfd")
    if wg:
        sg = run_all(wg, SPREAD_GOLD)
        rows = []
        for tag, per in sg.items():
            vals = [R_of(r).mean() for r in per.values() if len(R_of(r)) >= 40]
            rows.append(dict(cfg=tag, windows=len(vals),
                             trades=sum(len(r.trades) for r in per.values()),
                             expR_mean=round(float(np.mean(vals)), 4) if vals else np.nan,
                             expR_min=round(float(np.min(vals)), 4) if vals else np.nan,
                             pos=int(sum(1 for v in vals if v > 0))))
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("  gold data unavailable for these windows")

    print("\n" + "=" * 104)
    print("E. BOOTSTRAP — 5th-percentile lower bound per window (IID and moving-block)")
    print("=" * 104)
    rows = []
    for tag, per in store.items():
        for name, blk in (("iid", 0), ("block", 25)):
            los = []
            for k, r in per.items():
                ci = boot_ci(R_of(r), n_boot=5000, block=blk, seed=abs(hash((tag, k, name))) % 9999)
                if ci:
                    los.append(ci["lo5"])
            if los:
                rows.append(dict(cfg=tag, boot=name, windows=len(los),
                                 min_lo5=round(float(np.min(los)), 4),
                                 median_lo5=round(float(np.median(los)), 4),
                                 all_pos=bool(np.all(np.array(los) > 0))))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n" + "=" * 104)
    print("F. PARAMETER PLATEAU — zigzag sensitivity and pattern selection")
    print("=" * 104)
    cfg = {**CFG_LEVERED, "spread": SPREAD_BTC}
    rows = []
    for am in (1.5, 2.0, 3.0, 4.0):
        for names, lab in ((GOOD_CHART, "screened8"), (None, "all19"),
                           (["double_bottom", "double_top"], "doubles"),
                           (["head_shoulders", "inverse_head_shoulders"], "H&S")):
            vals = []
            for side in ("long", "short"):
                ep = {"pat_atr_mult": am}
                if names:
                    ep["pat_names"] = names
                for k, d in w.items():
                    try:
                        r = EntryBundle(d, f"chart_pattern_{side}@v1", ep).run(
                            "atr@v1", "rr@v1", {"stop_atr_mult": 2.0, "tp_rr": 8.0}, cfg)
                    except Exception:
                        continue
                    R = R_of(r)
                    if len(R) >= 20:
                        vals.append(float(R.mean()))
            rows.append(dict(zz_atr_mult=am, patterns=lab, n_cells=len(vals),
                             expR_mean=round(float(np.mean(vals)), 4) if vals else np.nan))
    P = pd.DataFrame(rows)
    print(P.pivot(index="zz_atr_mult", columns="patterns", values="expR_mean").round(4).to_string())
    P.to_csv("experiments/results_phase14_pattern_gates.csv", index=False)
    print("\nsaved -> experiments/results_phase14_pattern_gates.csv")


if __name__ == "__main__":
    main()
