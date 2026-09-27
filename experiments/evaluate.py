"""
EVALUATE A STRATEGY FILE — the strategy as set, and every arm of its COMPARE, quarter by quarter.

The lab shows a strategy on one window, and one window is where every rule in this project has
looked good: notebook 21 made +4R on its 20 days and loses 0.16R a trade over four and a half years.
This is the check to run BEFORE believing a window. It is experiments/exit_rules_search.py made
general: any strategy in notebooks/strategies/, any arms, the same scoring.

    from experiments.evaluate import evaluate
    ev = evaluate(evaluate_config())     # ~8 s a quarter
    ev.baseline()   # the strategy itself: R per trade, quarters positive -- and at zero cost
    ev.arms()       # each arm against it: train and forward, quarters beaten, t, trimmed

HOW IT SCORES -- the project's standing rules, applied the same way every time:
  * per quarter, an arm's TOTAL R minus the strategy's as set. Arms change entries, exits or costs;
    none of them is a filter scored per trade, so total R is the unit;
  * TRAIN picks, FORWARD judges. `rank_corr` is whether the arms' train ranking survives forward --
    near zero or negative means the ranking was noise, whatever the best arm's number says;
  * `t` is the mean quarterly difference over its standard error, across ALL quarters. With k arms
    tried, the best |t| by chance grows with k: below ~2.5 it means nothing here;
  * `fwd_trim3` is the forward difference with each run's 3 best trades removed -- an arm that is
    only good through one or two outsized trades is a lottery ticket.

FRAMES. The cached 4h file starts 2025-08, so 4h is built from 1h (resample_ohlcv is exact going
down). Drawing is off (no S/R, no missed setups, no EMAs beyond what the rules need); neither changes
a trade -- tests/test_workspace.py checks the evaluation's window reproduces the lab's trades.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import quant.data as qdata
from quant.data import resample_ohlcv
from quant.data.timeframe import tf_frames as _tf_frames
from quant.research import fvg_lab
from quant.research.fvg_lab import RETRADE_KEYS

__all__ = ["evaluate", "Evaluation", "quarters", "quarter_window", "window_cfg",
           "frames_4h_from_1h"]

AS_SET = "as set"
ZERO = "as set, zero cost"


# ---------------------------------------------------------------- windows and frames
def quarters(first: str, last: str) -> List[str]:
    """'2022Q1', '2024Q2' -> every quarter from the first to the last, inclusive."""
    return [str(p) for p in pd.period_range(first, last, freq="Q")]


def quarter_window(q: str, data_end: Optional[str] = None):
    p = pd.Period(q, freq="Q")
    end = p.end_time.normalize() + pd.Timedelta(days=1)
    if data_end is not None:
        end = min(end, pd.Timestamp(data_end))
    return p.start_time, end


def frames_4h_from_1h(symbol, tfs, *, view_start, end, warmup_bars=800, **kw):
    """`tf_frames`, except 4h is resampled from 1h -- exact, and the 1h cache reaches 2021."""
    tfs = tuple(dict.fromkeys(tfs))
    out = _tf_frames(symbol, tuple(t for t in tfs if t != "4h"), view_start=view_start, end=end,
                     warmup_bars=warmup_bars, **kw)
    if "4h" in tfs:
        h = _tf_frames(symbol, ("1h",), view_start=view_start, end=end,
                       warmup_bars=warmup_bars * 4, **kw)["1h"]
        out = {"4h": resample_ohlcv(h, "240min").reset_index(drop=True), **out}
    return out


# the settings a window needs that a strategy file does not state -- time, and drawing switched off
_WINDOW_ONLY = dict(AS_OF=None, MTF_EMAS=[], EMAS=(), SHOW_SR_ZONES=False, SHOW_REJECTED=False,
                    REWIND=False)


def window_cfg(cfg, q: str, data_end: Optional[str] = None, warmup_months: int = 1):
    """`cfg`'s trading settings on one quarter, with a month of warm-up before it."""
    s, e = quarter_window(q, data_end)
    w = {k: v for k, v in vars(cfg).items() if k.isupper()}
    w.update(_WINDOW_ONLY)
    w.update(VIEW_START=s.strftime("%Y-%m-%d"), END=e.strftime("%Y-%m-%d"),
             WARMUP_START=(s - pd.DateOffset(months=warmup_months)).strftime("%Y-%m-%d"))
    return SimpleNamespace(**w)


def _changes_nothing(cfg, over: dict) -> bool:
    return all(repr(getattr(cfg, k, None)) == repr(v) for k, v in over.items())


# ---------------------------------------------------------------- the evaluation
@dataclass
class Evaluation:
    trades: pd.DataFrame                  # one row per (quarter, run, trade)
    train: List[str]
    forward: List[str]
    same_as_set: List[str] = field(default_factory=list)   # arms that change nothing

    def _per_quarter(self) -> pd.DataFrame:
        t = self.trades
        per = t.groupby(["run", "quarter"]).R.sum().unstack("quarter")
        return per.reindex(columns=self.train + self.forward).fillna(0.0)

    def by_quarter(self) -> pd.DataFrame:
        """Total R per quarter, one row per run -- the strategy and each arm."""
        return self._per_quarter().round(1)

    def baseline(self) -> pd.DataFrame:
        """The strategy on its own. The row that decides whether an arm is worth reading at all."""
        rows = []
        per = self._per_quarter()
        for run in [r for r in (AS_SET, ZERO) if r in per.index]:
            t = self.trades[(self.trades.run == run) & self.trades.R.notna()]
            tr, fw = t[t.quarter.isin(self.train)], t[t.quarter.isin(self.forward)]
            rows.append({"run": run, "trades": len(t),
                         "R / trade": round(t.R.mean(), 3) if len(t) else np.nan,
                         "win %": round(100 * (t.R > 0).mean(), 1) if len(t) else np.nan,
                         "train R": round(tr.R.sum(), 1), "forward R": round(fw.R.sum(), 1),
                         "quarters > 0": f"{int((per.loc[run] > 0).sum())}/{per.shape[1]}"})
        return pd.DataFrame(rows)

    def arms(self) -> pd.DataFrame:
        """Each arm against the strategy as set. Read `rank_corr` and `t` before any R column."""
        per = self._per_quarter()
        if AS_SET not in per.index:
            return pd.DataFrame()
        d = per.sub(per.loc[AS_SET], axis=1).drop(index=[r for r in (AS_SET, ZERO)
                                                          if r in per.index])
        base = self.trades[self.trades.run == AS_SET]

        def trim(x, k=3):
            x = x.dropna().sort_values()
            return float(x.iloc[:-k].sum()) if len(x) > k else 0.0

        rows = []
        for arm in d.index:
            tr, fw, all_ = d.loc[arm, self.train], d.loc[arm, self.forward], d.loc[arm]
            a = self.trades[self.trades.run == arm]
            sd = all_.std(ddof=1)
            rows.append({
                "arm": arm,
                "train dR": round(tr.sum(), 1),
                "train beat": f"{int((tr > 0).sum())}/{int((tr != 0).sum())}",
                "forward dR": round(fw.sum(), 1),
                "forward beat": f"{int((fw > 0).sum())}/{int((fw != 0).sum())}",
                "fwd trim3": round(trim(a[a.quarter.isin(self.forward)].R)
                                   - trim(base[base.quarter.isin(self.forward)].R), 1),
                "t": round(all_.mean() / (sd / np.sqrt(len(all_))), 2) if sd > 0 else np.nan,
                "trades": int(a.R.notna().sum()) - int(base.R.notna().sum()),
            })
        out = pd.DataFrame(rows)
        rc = (float(out["train dR"].rank().corr(out["forward dR"].rank()))
              if len(out) > 2 else np.nan)
        out = out.sort_values("train dR", ascending=False).reset_index(drop=True)
        out.attrs["rank_corr"] = rc
        return out

    def verdict(self) -> str:
        """The two numbers that decide whether the arms table means anything, in words."""
        a = self.arms()
        if not len(a):
            return "no arms"
        rc = a.attrs.get("rank_corr", np.nan)
        best = a["t"].abs().max()
        lines = [f"{len(a)} arms over {len(self.train) + len(self.forward)} quarters.",
                 ("train -> forward rank correlation: n/a (needs 3 or more arms)" if rc != rc
                  else f"train -> forward rank correlation: {rc:+.2f}"
                  + ("  (the train ranking did not survive)" if not rc > 0.3 else "")),
                 f"largest |t|: {best:.2f}" + ("  -- below 2.5, nothing here beats noise"
                                               if best < 2.5 else "  -- worth a closer look")]
        if self.same_as_set:
            lines.append(f"identical to the strategy as set, so not run: {self.same_as_set}")
        return "\n".join(lines)


def evaluate(cfg, arms: Optional[Dict[str, dict]] = None, *, train=None, forward=None,
             data_end=None, zero_cost=None, progress: bool = True) -> Evaluation:
    """Run the strategy in `cfg` (usually `evaluate_config()`) and its arms over the quarters.

    `arms` defaults to the strategy's COMPARE; `train`/`forward`/`data_end`/`zero_cost` to the
    config's TRAIN/FORWARD/DATA_END/ZERO_COST. Arms that change trading settings only re-trade the
    quarter's lab; any other arm rebuilds it.
    """
    train = quarters(*(train or cfg.TRAIN))
    forward = quarters(*(forward or cfg.FORWARD))
    data_end = data_end or getattr(cfg, "DATA_END", None)
    zero_cost = getattr(cfg, "ZERO_COST", True) if zero_cost is None else zero_cost
    arms = dict(getattr(cfg, "COMPARE", {}) if arms is None else arms)
    same = [k for k, v in arms.items() if _changes_nothing(cfg, v)]
    arms = {k: v for k, v in arms.items() if k not in same}

    rows = []

    def keep(q, run, t):
        if not len(t):
            rows.append(dict(quarter=q, run=run, entry="", side="", R=np.nan, why="", bars=0))
        for r in t.itertuples():
            rows.append(dict(quarter=q, run=run, entry=str(r.entry_time)[:16], side=r.side,
                             R=float(r.R), why=r.close_reason, bars=int(r.bars_held)))

    qdata.tf_frames = frames_4h_from_1h
    t0 = time.time()
    try:
        for q in train + forward:
            w = window_cfg(cfg, q, data_end)
            lab = fvg_lab(w)
            keep(q, AS_SET, lab.trades)
            if zero_cost:
                free = {**w.COSTS, "fee_bps": 0, "slippage_bps": 0, "spread": 0.0,
                        "commission_per_lot": 0.0}
                keep(q, ZERO, lab.retrade(COSTS=free).trades)
            for name, over in arms.items():
                if set(over) <= RETRADE_KEYS:
                    keep(q, name, lab.retrade(**over).trades)
                else:
                    keep(q, name, fvg_lab(SimpleNamespace(**{**vars(w), **over})).trades)
            if progress:
                print(f"{q}: {len(lab.trades)} trades  {lab.trades.R.sum():+.1f}R  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        qdata.tf_frames = _tf_frames
    return Evaluation(trades=pd.DataFrame(rows), train=train, forward=forward, same_as_set=same)
