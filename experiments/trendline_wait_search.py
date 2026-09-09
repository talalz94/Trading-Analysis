"""
How many candles is it worth waiting for before trading a trendline break?

`TL_CLOSE_N` in notebook 18 was one number for every timeframe, which cannot be right: two 15m
closes is half an hour and two 1m closes is two minutes. This sweeps the wait PER TIMEFRAME and per
side, so 5m can want 2 and 15m can want 3.

WHY THIS DOES NOT RUN ON THE NOTEBOOK'S NINE-DAY WINDOW. Nine days gives a handful of trades per
cell, and picking the argmax of six settings on ten trades is how this project has manufactured
every result it later had to retract. So:

  * the sweep runs over MONTHS, split into consecutive windows;
  * the score is WINDOWS POSITIVE, not total R -- a setting that wins because of one huge trade in
    one window is exactly what trim-top-K exists to catch;
  * the count of winners is compared against CHANCE, `n_settings x 0.5^n_windows`, so "6 settings
    over 6 windows will hand you a 5-of-6 winner about 19% of the time by luck alone" is stated
    rather than left for the reader to work out;
  * each timeframe is swept ALONE, so a good 15m number cannot be an artifact of 5m trades that
    happened to be open at the time.

Output: `experiments/results_trendline_wait.csv` plus a recommended dict printed at the end. That
dict is a starting point, NOT a validated setting -- read the survivor count next to it.

    python experiments/trendline_wait_search.py
    python experiments/trendline_wait_search.py --tf 5m --waits 1 2 3 4
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv, tf_frames                                # noqa: E402
from quant.indicators.trendlines import mtf_trendlines                     # noqa: E402
from quant.scenarios import run_pieces                                     # noqa: E402
from quant.strategies.conditions import raw                                # noqa: E402

SYMBOL, SOURCE = "BTCUSDT", "binance"
CHART_TF = "5m"                     # execution timeframe — matches notebook 18's default
LINE_TFS = ("5m", "15m", "1h")      # every timeframe whose lines we might trade
WAITS = (1, 2, 3, 4, 5, 6)          # candidate "wait N closes beyond the line"
START, END = "2026-01-01", "2026-06-10"
N_WINDOWS = 6
WARMUP_DAYS = 30                    # before the first window, so the lines have pivots

TRENDLINE = dict(atr_mult=3.5, min_touches=3, lookback_pivots=16, line_tol_atr=0.6,
                 min_span_bars=40, max_age_bars=500, break_buffer_atr=0.0)
EXIT = dict(stop_n=20, stop_buffer_pct=0.05, be_at_rr=1.0, tp_rr=3.0)
COSTS = dict(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
             sizing_mode="risk_pct_equity", sizing_value=1.0,
             exit_enabled=False, max_open_trades=1)
OUT = "experiments/results_trendline_wait.csv"


def windows(start: str, end: str, k: int):
    """`k` consecutive windows spanning [start, end)."""
    edges = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), periods=k + 1)
    return list(zip(edges[:-1], edges[1:]))


def run_one(D: pd.DataFrame, pref: str, side: str, wait: int, lo, hi, names) -> dict:
    """One timeframe, one side, one wait, one window."""
    p = dict(tl_prefixes=(pref,), tl_tf_names=names, tl_close_n=wait, tl_full_m=0,
             tl_touch_k=0, tl_combine="all", tl_tol_atr=0.0, **EXIT)
    win = raw(lambda d, lo=lo, hi=hi: ((pd.to_datetime(d["t"]) >= lo)
                                       & (pd.to_datetime(d["t"]) < hi)).to_numpy(), "w")
    run = run_pieces(D, f"mtf_trendline_{side}@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=COSTS, entry_filter=win)
    tr = run.trades
    if not len(tr):
        return dict(n=0, total_R=0.0, avg_R=np.nan, win=np.nan)
    risk = (tr.entry_price - tr.stop_price).abs() * tr.qty
    R = (tr.pnl / risk).to_numpy(float)
    return dict(n=len(tr), total_R=float(np.nansum(R)), avg_R=float(np.nanmean(R)),
                win=float((tr.pnl > 0).mean()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", nargs="*", default=list(LINE_TFS))
    ap.add_argument("--waits", nargs="*", type=int, default=list(WAITS))
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    a = ap.parse_args()

    warm = (pd.Timestamp(a.start, tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)).strftime("%Y-%m-%d")
    t0 = time.time()
    base = get_ohlcv(SYMBOL, CHART_TF, start=warm, end=a.end, source=SOURCE,
                     tz="UTC").reset_index(drop=True)
    extra = tf_frames(SYMBOL, [t for t in a.tf if t != CHART_TF],
                      view_start=warm, end=a.end, warmup_bars=1500, source=SOURCE)
    D = mtf_trendlines(base, extra, **TRENDLINE)
    names = {"": CHART_TF, **{f"{k}__": k for k in extra}}
    prefs = {v: k for k, v in names.items()}
    print(f"{len(base):,} {CHART_TF} bars + {len(extra)} extra timeframes  ({time.time()-t0:.1f}s)")

    wins = windows(a.start, a.end, a.windows)
    print(f"{a.windows} windows of ~{(wins[0][1]-wins[0][0]).days} days, "
          f"{len(a.tf)} timeframes x 2 sides x {len(a.waits)} waits\n")

    rows = []
    for tf in a.tf:
        pref = prefs.get(tf)
        if pref is None:
            continue
        for side in ("long", "short"):
            for wait in a.waits:
                for wi, (lo, hi) in enumerate(wins):
                    r = run_one(D, pref, side, wait, lo, hi, names)
                    rows.append(dict(tf=tf, side=side, wait=wait, window=wi, **r))
    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False)

    # ---- score: WINDOWS POSITIVE first, total R only as a tie-break -------------------
    g = (res.groupby(["tf", "side", "wait"])
         .agg(trades=("n", "sum"), windows_traded=("n", lambda z: int((z > 0).sum())),
              windows_pos=("total_R", lambda z: int((z > 0).sum())),
              total_R=("total_R", "sum"), avg_R=("avg_R", "mean"))
         .reset_index())
    g["avg_R"] = g["avg_R"].round(3)
    g["total_R"] = g["total_R"].round(2)

    print(g.to_string(index=False))
    chance = len(a.waits) * 0.5 ** a.windows
    print(f"\nCHANCE: {len(a.waits)} settings x 0.5^{a.windows} = {chance:.2f} settings are expected "
          f"to clear ALL {a.windows} windows by luck alone.")
    print("A winner is only interesting if it beats that, and if `trades` is big enough to mean "
          "anything.\n")

    best = {}
    print("best wait per timeframe and side (windows_pos, then total R):")
    for (tf, side), sub in g.groupby(["tf", "side"]):
        sub = sub.sort_values(["windows_pos", "total_R"], ascending=False)
        top = sub.iloc[0]
        best.setdefault(tf, {})[side] = int(top.wait)
        flag = "" if top.trades >= 30 else f"   <- only {int(top.trades)} trades, do not trust this"
        print(f"   {tf:>4} {side:<5} wait={int(top.wait)}   "
              f"{int(top.windows_pos)}/{a.windows} windows positive, "
              f"{top.total_R:+.1f}R over {int(top.trades)} trades{flag}")

    print("\nTL_CLOSE_N = {")
    for tf, sides in best.items():
        same = sides["long"] == sides["short"]
        note = "" if same else f"   # long wants {sides['long']}, short {sides['short']}"
        print(f'    "{tf}": {sides["short"] if not same else sides["long"]},{note}')
    print("}")
    print(f"\nwrote {OUT}   ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
