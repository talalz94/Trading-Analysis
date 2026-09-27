"""
EXIT RULES SEARCH — do "move the target after a near miss" or "exit when a gap near the target
fills" improve notebook 21's trendline-break strategy?

WHY THIS EXISTS. Both rules were proposed from ONE trade (#04, 2026-06-09: price came within 0.1R of
a 3R target, turned, and stopped out). On that window both look excellent, and they have to -- they
were drawn around the trade. An exit rule is only worth having if it helps the trades nobody looked
at, so this runs notebook 21's EXACT settings over four and a half years of quarters and asks what
the rules do there.

WHAT IS HELD FIXED. The entries: every arm trades the same breaks (the rules only change how a trade
ENDS). An earlier exit can free the one-position slot for a trade the baseline had to skip, so trade
counts may differ by a few; that is part of the rule's effect, not noise to remove.

HOW IT IS SCORED -- decided before the run:
  * per quarter, the arm's TOTAL R minus the baseline's (exits do not filter, so total R is the
    right unit -- unlike a veto, which is scored per trade);
  * selection on TRAIN (2022Q1-2024Q2), verdict on FORWARD (2024Q3-2026Q3), never the other way;
  * quarters beaten, against the coin-flip count; the whole grid's survivors against
    n_configs x 0.5^n_windows (the project's standing rule);
  * the same totals with each arm's 3 BEST trades removed, because an exit rule that looks good
    only through one or two outsized trades is a lottery ticket (standing rule four).

THE GRID (fixed before any result was seen):
  move target   near {0.6, 0.8, 0.9} x back {0.2, 0.4} x to {0.4, 0.6}      12
                time only: after_bars {144, 288} (12h, 1 day), to 0.5          2
  fvg exit      tfs {15m+1h+4h, 1h+4h} x near_r {0.5, 1.0} x at {touch, ce, full}   12

4h bars are built from 1h (resample_ohlcv is exact going down; the cached 4h file starts 2025-08),
the same substitution experiments/sr_zone_search.py makes. Drawing-only settings are switched off
(S/R, missed setups, 1m EMAs); the test below proves they do not change a single trade.

Run:  python -m experiments.exit_rules_search            (~10 min)
Out:  experiments/results_exit_rules.csv                  one row per (quarter, arm, trade)
"""
from __future__ import annotations

import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import quant.data as qdata                                              # noqa: E402
from quant.data import resample_ohlcv                                   # noqa: E402
from quant.data.timeframe import tf_frames as _tf_frames               # noqa: E402
from quant.research import fvg_lab, variant                             # noqa: E402

OUT = ROOT / "experiments" / "results_exit_rules.csv"

TRAIN = [f"{y}Q{q}" for y in (2022, 2023) for q in (1, 2, 3, 4)] + ["2024Q1", "2024Q2"]
FORWARD = ["2024Q3", "2024Q4", "2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1", "2026Q2",
           "2026Q3"]                      # 2026Q3 = July-August, where the cache ends


def _bounds(q: str):
    p = pd.Period(q, freq="Q")
    start = p.start_time
    end = min(p.end_time.normalize() + pd.Timedelta(days=1), pd.Timestamp("2026-08-31"))
    return start, end


def _arms():
    arms = {}
    for near, back, to in itertools.product((0.6, 0.8, 0.9), (0.2, 0.4), (0.4, 0.6)):
        arms[f"move n{near} b{back} t{to}"] = dict(
            USE_TP_MOVE=True, TP_MOVE=dict(near=near, back=back, after_bars=0, to=to))
    for after in (144, 288):
        arms[f"move after{after} t0.5"] = dict(
            USE_TP_MOVE=True, TP_MOVE=dict(near=0.0, back=0.0, after_bars=after, to=0.5))
    for tfs, near_r, at in itertools.product((("15m", "1h", "4h"), ("1h", "4h")), (0.5, 1.0),
                                             ("touch", "ce", "full")):
        arms[f"fvg {'+'.join(tfs)} r{near_r} {at}"] = dict(
            USE_FVG_EXIT=True, FVG_EXIT=dict(tfs=tfs, near_r=near_r, at=at))
    return arms


def _frames_4h_from_1h(symbol, tfs, *, view_start, end, warmup_bars=800, **kw):
    """tf_frames, except 4h is resampled from 1h -- exact, and the 1h cache reaches 2021."""
    tfs = tuple(dict.fromkeys(tfs))
    out = _tf_frames(symbol, tuple(t for t in tfs if t != "4h"), view_start=view_start, end=end,
                     warmup_bars=warmup_bars, **kw)
    if "4h" in tfs:
        h = _tf_frames(symbol, ("1h",), view_start=view_start, end=end,
                       warmup_bars=warmup_bars * 4, **kw)["1h"]
        out = {"4h": resample_ohlcv(h, "240min").reset_index(drop=True), **out}
    return out


def base_cfg():
    import nb21_config as CFG
    # drawing only -- none of these is read when trading (test_exit_rules proves the trades match)
    return variant(CFG, SHOW_SR_ZONES=False, SHOW_REJECTED=False, MTF_EMAS=[], REWIND=False,
                   AS_OF=None, USE_TP_MOVE=False, USE_FVG_EXIT=False)


def run(quarters=None):
    qdata.tf_frames = _frames_4h_from_1h
    try:
        base = base_cfg()
        arms = _arms()
        rows = []
        t0 = time.time()
        for q in quarters or TRAIN + FORWARD:
            s, e = _bounds(q)
            cfg = variant(base, VIEW_START=s.strftime("%Y-%m-%d"), END=e.strftime("%Y-%m-%d"),
                          WARMUP_START=(s - pd.DateOffset(months=1)).strftime("%Y-%m-%d"))
            lab = fvg_lab(cfg)
            for name, lb in [("baseline", lab)] + [(k, lab.retrade(**v)) for k, v in arms.items()]:
                t = lb.trades
                for r in t.itertuples():
                    rows.append(dict(quarter=q, arm=name, entry=str(r.entry_time)[:16],
                                     side=r.side, R=float(r.R), why=r.close_reason,
                                     bars=int(r.bars_held)))
                if not len(t):
                    rows.append(dict(quarter=q, arm=name, entry="", side="", R=np.nan, why="",
                                     bars=0))
            print(f"{q}: {len(lab.trades)} baseline trades  {time.time() - t0:.0f}s", flush=True)
    finally:
        qdata.tf_frames = _tf_frames
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    return out


def _trim(r: pd.Series, k: int = 3) -> float:
    r = r.dropna()
    return float(r.sort_values().iloc[:-k].sum()) if len(r) > k else 0.0


def report(t: pd.DataFrame) -> pd.DataFrame:
    per = t.groupby(["arm", "quarter"]).R.sum().unstack("quarter")
    d = per.sub(per.loc["baseline"], axis=1)             # arm minus baseline, per quarter
    rows = []
    for arm in per.index:
        if arm == "baseline":
            continue
        tr, fw = d.loc[arm, TRAIN], d.loc[arm, FORWARD]
        a = t[t.arm == arm]
        b = t[t.arm == "baseline"]
        rows.append(dict(
            arm=arm,
            train_dR=round(tr.sum(), 2), train_beat=f"{int((tr > 0).sum())}/{int((tr != 0).sum())}",
            fwd_dR=round(fw.sum(), 2), fwd_beat=f"{int((fw > 0).sum())}/{int((fw != 0).sum())}",
            fwd_dR_trim3=round(_trim(a[a.quarter.isin(FORWARD)].R)
                               - _trim(b[b.quarter.isin(FORWARD)].R), 2),
            all_beat=int((d.loc[arm] > 0).sum()),
            fired=int(a.why.isin(["tp_moved", "fvg_exit"]).sum())))
    rep = pd.DataFrame(rows).sort_values("train_dR", ascending=False).reset_index(drop=True)
    n_q = len(TRAIN) + len(FORWARD)
    b = t[t.arm == "baseline"]
    print(f"\nbaseline: {b.R.notna().sum()} trades  train {b[b.quarter.isin(TRAIN)].R.sum():+.1f}R  "
          f"forward {b[b.quarter.isin(FORWARD)].R.sum():+.1f}R")
    print(f"\n{len(rep)} arms x {n_q} quarters. Beating the baseline in EVERY quarter by chance: "
          f"{len(rep) * 0.5 ** n_q:.4f} arms expected; did: {int((rep.all_beat == n_q).sum())}")
    tr_rank, fw_rank = rep.train_dR.rank(), rep.fwd_dR.rank()
    print(f"train -> forward rank correlation of the arms: {tr_rank.corr(fw_rank):+.2f}\n")
    print(rep.to_string(index=False))
    return rep


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(pd.read_csv(OUT))
    else:
        report(run())
