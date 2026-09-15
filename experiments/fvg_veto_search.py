"""
Does an unfilled fair value gap behind a trendline break tell you to skip the trade?

WHY THIS FILE EXISTS. Notebook 19 answers the question on one nine-day window, where the whole sim
takes eight trades. Across 96 gate combinations "barrier" tended to block losers and "retrace" to
block winners -- and that is worth exactly nothing: with 4 winners and 4 losers, a three-trade block
that happens to be all losers arises by chance with probability C(4,3)/C(8,3) = 7.1%, so 96
combinations produce about seven such cells for free. Worse, the 96 cells are not 96 observations;
they all score the SAME eight trades.

So this runs the comparison the way the rest of the project's research is bound to run
(`CLAUDE.md` section 0):

  * MANY WINDOWS, and the score is HOW MANY CAME OUT POSITIVE, not the total R. Total R over a
    fat-tailed sample is one trade's opinion.
  * THE CHANCE BASELINE IS PRINTED BESIDE IT. With `n_configs` settings and `n_windows` windows,
    `n_configs * 0.5 ** n_windows` of them clear every window by luck. A survivor count below that
    is not a survivor count.
  * BOTH MODES, ALWAYS, because they are exact mirrors -- a filter that helps under `retrace` and
    hurts under `barrier` is saying something; one that looks the same under both is noise.
  * TRIM-TOP-K, because a single 3R winner decides a nine-day window.

The null to beat is NOT zero. It is the SAME SIM WITHOUT THE VETO, run on the same window, because
the veto can only ever remove trades -- and removing trades at random from a positive-expectancy
strategy lowers total R while leaving R-per-trade alone. R per trade is therefore the number that
matters here, and "windows where the veto improved R per trade" is the score.

Run it:  python experiments/fvg_veto_search.py
"""
from __future__ import annotations

import itertools
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from quant.data import get_ohlcv, tf_frames                       # noqa: E402
from quant.indicators import mtf_fvgs                             # noqa: E402
from quant.indicators.trendlines import mtf_trendlines            # noqa: E402
from quant.scenarios import run_pieces                            # noqa: E402
from quant.strategies import fvg_clear                            # noqa: E402
from quant.strategies.conditions import raw                       # noqa: E402

description = "Is a trendline break with an unfilled FVG behind it worth skipping?"

SYMBOL, TF = "BTCUSDT", "5m"
LINE_TFS = ("15m", "1h")
GAP_TFS = ("15m", "1h", "4h")
WARMUP_DAYS = 30

# six non-overlapping windows; the sweep is scored by how many of them each setting wins
WINDOWS = [("2026-01-05", "2026-01-25"), ("2026-02-05", "2026-02-25"),
           ("2026-03-05", "2026-03-25"), ("2026-04-05", "2026-04-25"),
           ("2026-05-05", "2026-05-25"), ("2026-06-01", "2026-06-10")]

GRID = dict(
    mode=("retrace", "barrier"),
    timeframes=(("1h", "4h"), ("15m", "1h", "4h"), None),
    max_age_bars=(20, 60),
    max_dist_atr=(1.0, 3.0),
)
MIN_SIZE_ATR = 0.5

TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0)
COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)


def _prepare(start: str, end: str):
    warm = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    df = get_ohlcv(SYMBOL, TF, start=warm, end=end, tz="UTC").reset_index(drop=True)
    lf = tf_frames(SYMBOL, LINE_TFS, view_start=start, end=end, warmup_bars=800)
    gf = tf_frames(SYMBOL, GAP_TFS, view_start=start, end=end, warmup_bars=800)
    D = mtf_trendlines(df, lf, **TRENDLINE)
    fv = mtf_fvgs(df, gf, mitigation="ce", min_size_atr=MIN_SIZE_ATR, max_age_bars=500)
    for c in [c for c in fv.columns if "fvg_" in c]:
        D[c] = fv[c].to_numpy()
    names = {"": TF, **{f"{k}__": k for k in lf}}
    params = dict(tl_prefixes=tuple(names), tl_tf_names=names, tl_close_n={"*": 1},
                  tl_full_m=0, tl_touch_k=0, tl_combine="all",
                  stop_n=20, stop_buffer_pct=0.05, be_at_rr=0, tp_rr=3.0)
    vs = pd.Timestamp(start, tz="UTC")
    in_window = raw(lambda d: (pd.to_datetime(d["t"]) >= vs).to_numpy(), "in_window")
    return D, params, in_window


def _trades(D, params, in_window, veto) -> pd.DataFrame:
    out = []
    for side in ("long", "short"):
        f = in_window if veto is None else (in_window & fvg_clear(side, **veto))
        r = run_pieces(D, f"mtf_trendline_{side}@v1", "high_of_n@v1", "be_ladder@v1",
                       params=params, cfg=COSTS, entry_filter=f)
        out.append(r.trades)
    t = pd.concat(out, ignore_index=True)
    if not len(t):
        return t.assign(R=[])
    risk = (t["entry_price"] - t["stop_price"]).abs()
    return t.assign(R=t["pnl"] / (risk * t["qty"])).sort_values("entry_time")


def _score(t: pd.DataFrame, trim: int = 0) -> float:
    """Mean R per trade, optionally with the top `trim` winners removed.

    R PER TRADE, not total: the veto can only remove trades, and removing them at random lowers
    total R while leaving the per-trade number alone. Only a per-trade improvement means the
    removed trades were worse than the ones kept.
    """
    if not len(t):
        return np.nan
    r = np.sort(t["R"].to_numpy(float))
    if trim and len(r) > trim:
        r = r[:-trim]
    return float(np.mean(r)) if len(r) else np.nan


def main() -> None:
    keys = list(GRID)
    configs = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    configs = [dict(c, min_size_atr=0.0) for c in configs]
    print(f"{len(configs)} settings x {len(WINDOWS)} windows, {SYMBOL} {TF}\n")

    rows: List[Dict] = []
    for start, end in WINDOWS:
        D, params, inw = _prepare(start, end)
        base = _trades(D, params, inw, None)
        b0, b1 = _score(base), _score(base, trim=1)
        print(f"{start}..{end}:  no veto -> {len(base):3d} trades, "
              f"{b0:+.3f} R/trade ({b1:+.3f} trimmed)")
        for c in configs:
            t = _trades(D, params, inw, c)
            rows.append(dict(window=start, **{k: str(v) for k, v in c.items()},
                             n=len(t), r=_score(t), r_trim=_score(t, trim=1),
                             base_n=len(base), base_r=b0, base_r_trim=b1))

    res = pd.DataFrame(rows)
    res["beat"] = res["r"] > res["base_r"]
    res["beat_trim"] = res["r_trim"] > res["base_r_trim"]
    gcols = [k for k in GRID]
    agg = (res.groupby(gcols)
              .agg(windows_won=("beat", "sum"), windows_won_trim=("beat_trim", "sum"),
                   traded=("n", "sum"), blocked=("base_n", "sum"))
              .reset_index())
    agg["blocked"] = agg["blocked"] - agg["traded"]
    agg = agg.sort_values(["windows_won_trim", "windows_won"], ascending=False)

    nw = len(WINDOWS)
    chance = len(configs) * 0.5 ** nw
    print(f"\n{'':-<92}")
    print(agg.to_string(index=False))
    print(f"{'':-<92}")
    print(f"\nCHANCE BASELINE: {len(configs)} settings x 0.5^{nw} = {chance:.2f} settings clear all "
          f"{nw} windows by luck alone.")
    best = int(agg["windows_won_trim"].max()) if len(agg) else 0
    n_best = int((agg["windows_won_trim"] == nw).sum())
    print(f"settings clearing all {nw} windows (trimmed): {n_best}   "
          f"-> {'NOT above chance' if n_best <= chance else 'above chance, look closer'}")
    print(f"best any setting managed: {best}/{nw} windows")
    print("\nA veto that removes trades at random lowers total R and leaves R/trade alone, which is "
          "why R/trade is the score. Read `blocked` too: a setting that blocked nothing won its "
          "windows by doing nothing.")
    out = "experiments/results_fvg_veto.csv"
    res.to_csv(out, index=False)
    print(f"\nper-window detail -> {out}")


if __name__ == "__main__":
    main()
