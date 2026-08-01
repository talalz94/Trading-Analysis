"""
Evaluation harness for the automated strategy search.

Everything this project has learned the hard way is baked in, because each of these traps produced a
result that looked good and wasn't:

1. **Costs are non-optional.** A rule averaging <0.5R per win over thousands of trades dies to a
   sub-basis-point spread. Every evaluation carries a spread.
2. **One window proves nothing.** A short-only setup made +294% in a -46% year and lost 43% in a +26%
   one. Configs are scored across SEVERAL regimes and judged on the WORST, not the mean.
3. **Position sizing must actually bind.** `sizing_mode='risk_pct_equity'` silently caps at ~1x equity
   unless `allow_leverage=True` AND `max_notional_pct` is raised — realised risk was 0.25%, not 1%.
4. **Expectancy in R, not %.** R normalises away sizing so configs are comparable.
5. **Selection needs a hold-out.** Picking the best of N on the same data is how the weekday filter
   scored top-decile in-sample and bottom-15% out-of-sample.

`score()` returns one row per config; `search()` sweeps a grid; `select_and_verify()` picks on TRAIN
and reports the untouched TEST result plus where that pick ranked among all candidates.
"""
from __future__ import annotations

import itertools
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from quant import scenarios as sc

# risk sizing that actually binds (see trap 3)
CFG_LEVERED = dict(initial_cash=10_000, sizing_mode="risk_pct_equity", sizing_value=1.0,
                   allow_leverage=True, max_notional_pct=5000)
# BTC ~60-120k, so spread 6.0 price units ~= 0.7 bp round trip. Deliberately optimistic.
SPREAD_BTC = 6.0


def r_stats(run) -> dict:
    """Per-trade expectancy in R plus the usual descriptives. R = pnl / (|stop-entry| * qty)."""
    t = run.trades
    if not len(t):
        return dict(n=0, win=np.nan, avg_win_R=np.nan, avg_loss_R=np.nan, expR=np.nan,
                    totR=np.nan, ret=0.0, dd=0.0, sharpe=np.nan, pf=np.nan)
    risk = (t["stop_price"] - t["entry_price"]).abs() * t["qty"]
    R = (t["pnl"] / risk).replace([np.inf, -np.inf], np.nan).dropna()
    w, l = R[R > 0], R[R <= 0]
    wr = len(w) / len(R) if len(R) else np.nan
    aw = float(w.mean()) if len(w) else 0.0
    al = float(abs(l.mean())) if len(l) else 0.0
    s = run.stats
    tot = float(R.sum())
    # OUTLIER GUARD. A trailing stop that catches one runaway move can post a huge mean R off a
    # single trade (seen: best +202R with totR 192 -> everything else summed to NEGATIVE). So also
    # report the total with the best trade removed, and the median trade.
    top = float(R.max()) if len(R) else 0.0
    tot_ex = tot - top
    n_ex = max(len(R) - 1, 1)
    return dict(n=len(t), win=round(100 * wr, 1) if wr == wr else np.nan,
                avg_win_R=round(aw, 3), avg_loss_R=round(-al, 3),
                expR=round(wr * aw - (1 - wr) * al, 4) if wr == wr else np.nan,
                expR_ex_top=round(tot_ex / n_ex, 4),
                med_R=round(float(R.median()), 4) if len(R) else np.nan,
                top_R=round(top, 1),
                totR=round(tot, 1), totR_ex_top=round(tot_ex, 1),
                ret=round(s["total_return_pct"], 2),
                dd=round(s["max_drawdown_pct"], 2), sharpe=round(s["sharpe"], 2),
                pf=round(s["profit_factor"], 2))


def score(windows: Dict[str, pd.DataFrame], scenario: str, stop: str, tp: str, *,
          secondary=None, params: Optional[dict] = None, spread: float = SPREAD_BTC,
          cfg_extra: Optional[dict] = None, entry_filter=None, tag: str = "") -> dict:
    """Run one config over every window. Reports the WORST window as the headline number, because a
    config that only works in one regime is directional beta, not edge."""
    cfg = {**CFG_LEVERED, "spread": spread, **(cfg_extra or {})}
    per = {}
    for name, d in windows.items():
        try:
            run = sc.run_pieces(d, scenario, stop, tp, secondary=secondary,
                                params=params or {}, cfg=cfg, entry_filter=entry_filter)
            per[name] = r_stats(run)
        except Exception as exc:                       # a bad combo must not kill the sweep
            per[name] = dict(n=0, expR=np.nan, ret=np.nan, dd=np.nan, error=str(exc)[:80])
    def _vals(key):
        # `None == None` is True, so a plain NaN test lets error rows through as None and the
        # subsequent min() compares None with None. Require an actual number.
        out = []
        for v in per.values():
            x = v.get(key)
            if isinstance(x, (int, float)) and x == x:
                out.append(float(x))
        return out

    exps, exps_ex = _vals("expR"), _vals("expR_ex_top")
    meds, rets = _vals("med_R"), _vals("ret")
    errs = sorted({v["error"] for v in per.values() if v.get("error")})
    row = dict(tag=tag or f"{scenario}|{stop}|{tp}",
               scenario=scenario, stop=stop, tp=tp,
               secondary=("+".join(secondary) if isinstance(secondary, (list, tuple))
                          else (secondary or "-")),
               n_total=int(sum(v.get("n", 0) for v in per.values())),
               expR_min=round(min(exps), 4) if exps else np.nan,
               expR_mean=round(float(np.mean(exps)), 4) if exps else np.nan,
               ret_min=round(min(rets), 2) if rets else np.nan,
               dd_max=round(max((v.get("dd", 0) or 0) for v in per.values()), 2),
               expR_ex_top_min=round(min(exps_ex), 4) if exps_ex else np.nan,
               medR_min=round(min(meds), 4) if meds else np.nan,
               top_R_max=round(max((v.get("top_R") or 0) for v in per.values()), 1),
               n_windows_pos=int(sum(1 for e in exps if e > 0)),
               n_windows_pos_ex_top=int(sum(1 for e in exps_ex if e > 0)),
               n_windows=len(exps), error=("; ".join(errs)[:120] if errs else ""))
    for name, v in per.items():
        row[f"expR_{name}"] = v.get("expR")
        row[f"n_{name}"] = v.get("n")
    return row


def search(windows: Dict[str, pd.DataFrame], grid: List[dict], *, spread: float = SPREAD_BTC,
           verbose: bool = True) -> pd.DataFrame:
    """Score every spec in `grid`. Each spec is the kwargs of `score` minus `windows`."""
    rows, t0 = [], time.time()
    for i, spec in enumerate(grid, 1):
        rows.append(score(windows, spread=spread, **spec))
        if verbose and (i % 25 == 0 or i == len(grid)):
            print(f"  ...{i}/{len(grid)}  ({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    return df.sort_values("expR_min", ascending=False).reset_index(drop=True)


def expand(**axes) -> List[dict]:
    """Cartesian product of the given axes -> a list of `score` kwargs."""
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(axes[k] for k in keys))]


def select_and_verify(train: Dict[str, pd.DataFrame], test: Dict[str, pd.DataFrame],
                      grid: List[dict], *, spread: float = SPREAD_BTC,
                      metric: str = "expR_min", top_k: int = 5) -> pd.DataFrame:
    """The honest protocol: rank on TRAIN, then report the untouched TEST result AND where that pick
    lands among ALL candidates scored on TEST. If the trained pick is not near the top of the test
    ranking, the selection is noise — that is exactly how the weekday filter failed."""
    tr = search(train, grid, spread=spread, verbose=False)
    te = search(test, grid, spread=spread, verbose=False)
    te_rank = {row.tag: i + 1 for i, row in enumerate(te.itertuples())}
    te_by_tag = te.set_index("tag")
    out = []
    for _, r in tr.head(top_k).iterrows():
        t = te_by_tag.loc[r.tag] if r.tag in te_by_tag.index else None
        out.append(dict(tag=r.tag, train_rank=int(tr.index[tr.tag == r.tag][0]) + 1,
                        train=r[metric],
                        test=(t[metric] if t is not None else np.nan),
                        test_rank=te_rank.get(r.tag),
                        of=len(te),
                        test_top_decile=(te_rank.get(r.tag, 10**9) <= max(1, len(te) // 10))))
    res = pd.DataFrame(out)
    res.attrs["train"] = tr
    res.attrs["test"] = te
    return res


def make_windows(symbol: str = "BTCUSDT", interval: str = "1m", source: str = "binance",
                 spans: Optional[Dict[str, tuple]] = None) -> Dict[str, pd.DataFrame]:
    """Load the standard regime windows. Defaults span a +26% bull and two bear legs, so no config
    can score well by being directionally lucky."""
    from quant.data import get_ohlcv
    spans = spans or {
        "bull25": ("2025-01-01", "2025-07-29"),      # BTC +26%
        "bear25": ("2025-07-29", "2026-01-28"),      # BTC -24%
        "bear26": ("2026-01-28", "2026-07-29"),      # BTC -28%
    }
    out = {}
    for name, (a, b) in spans.items():
        out[name] = get_ohlcv(symbol, interval, start=a, end=b, source=source, tz="UTC")
    return out
