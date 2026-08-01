"""
PHASE 5 — the structural test: TIMEFRAME and a SECOND INSTRUMENT.

Phases 1-4 all ran on BTC 1-minute bars, and the cost arithmetic says that was the problem, not the
signals. Spread as a percentage of one R (1.5 x ATR stop, median ATR over 2026 H1):

    BTCUSDT   1m 10.8%   5m 3.8%   15m 2.0%   1h 0.9%   4h 0.4%
    XAUUSD    1m  7.0%   5m 2.9%   15m 1.6%   1h 0.8%   4h 0.4%

On BTC 1m an entry must produce more than 0.108R of raw edge before it pays for the spread alone —
which is why the null benchmark scored -0.045R and every family clustered just below zero. Moving to
15m cuts that hurdle by 5x, and to 1h by 12x.

So the same entries, the same management and the same protocol are re-run across 5m / 15m / 1h and on
**gold as a second instrument** (robustness gate 7, never previously used). Gold is costed to the
standing house model: spread 0.25 USD/oz and NO commission.

The null entry travels with them. A higher timeframe lifts everything (the null included) simply by
paying less spread; only lift ABOVE the null counts as edge.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant.data import get_ohlcv
from quant.data.timeframe import resample_ohlcv

from experiments.mgmt_sweep import mgmt_grid, sweep

SPANS = {"w1": ("2025-01-01", "2025-07-29"), "w2": ("2025-07-29", "2026-01-28"),
         "w3": ("2026-01-28", "2026-07-29")}
MARKETS = [("BTCUSDT", "binance", "spot", 6.0),      # ~0.7 bp, deliberately optimistic
           ("XAUUSD", "dukascopy", "cfd", 0.25)]     # house model: 0.25 USD/oz, no commission
TFS = ["5min", "15min", "1h"]

ENTRIES = [
    ("choch_long", "bos_long@v1", {"bos_mode": "choch", "zz_atr_mult": 4.0}, None),
    ("choch_short", "bos_short@v1", {"bos_mode": "choch", "zz_atr_mult": 4.0}, None),
    ("pdlow_fade_long", "level_event_long@v1",
     {"lv_col": "pd_low", "lv_side": "below", "lv_event": "fade"}, None),
    ("pdhigh_fade_short", "level_event_short@v1",
     {"lv_col": "pd_high", "lv_side": "above", "lv_event": "fade"}, None),
    ("vwapz2_short", "vwap_rev_short@v1", {"vr_sigma": 2.0}, None),
    ("vwapz2_long", "vwap_rev_long@v1", {"vr_sigma": 2.0}, None),
    ("rtdiv_short", "rsi_div_short@v1", {"div_mode": "realtime"}, None),
    ("wt53_long", "wt_cross_long@v1", {"wt_level": 53.0}, None),
    ("wt53_short", "wt_cross_short@v1", {"wt_level": 53.0}, None),
    ("NULL_long", "time_entry_long@v1", {"te_every": 40}, None),
    ("NULL_short", "time_entry_short@v1", {"te_every": 40}, None),
]

STOPS = [("atr@v1", {"stop_atr_mult": 1.5}), ("atr@v1", {"stop_atr_mult": 2.5})]
TPS = [("rr@v1", {"tp_rr": 1.5}), ("rr@v1", {"tp_rr": 2.0}),
       ("trailing@v1", {"trail_pct": 1.0})]
TIME_STOPS = [0, 40]


def windows_for(symbol, source, market, tf):
    out = {}
    for k, (a, b) in SPANS.items():
        d = get_ohlcv(symbol, "1m", start=a, end=b, source=source, market=market, tz="UTC")
        out[k] = resample_ohlcv(d, tf).reset_index(drop=True)
    return out


def main():
    grid = mgmt_grid(STOPS, TPS, TIME_STOPS)
    frames = []
    for sym, src, mkt, spread in MARKETS:
        for tf in TFS:
            w = windows_for(sym, src, mkt, tf)
            t0 = time.time()
            r = sweep(w, ENTRIES, grid, spread=spread, verbose=False)
            r["symbol"], r["tf"] = sym, tf
            frames.append(r)
            bars = {k: len(v) for k, v in w.items()}
            print(f"  {sym:9s} {tf:5s} bars={sum(bars.values()):>8,}  "
                  f"{len(r)} configs in {time.time()-t0:.0f}s", flush=True)
    res = pd.concat(frames, ignore_index=True)
    res["is_null"] = res.entry.str.startswith("NULL")
    res.to_csv("experiments/results_phase5_tf_instrument.csv", index=False)
    pd.set_option("display.width", 250)
    if res.error.astype(bool).any():
        print("ERRORS:", res[res.error.astype(bool)].error.value_counts().head(3).to_dict())

    print("\n" + "=" * 104)
    print("A. COST RELIEF IS REAL — median expR_min by symbol x timeframe, null vs real")
    print("=" * 104)
    piv = (res.pivot_table(index=["symbol", "tf"], columns="is_null", values="expR_min",
                           aggfunc="median")
              .rename(columns={False: "real", True: "NULL"}))
    piv["real_minus_null"] = piv["real"] - piv["NULL"]
    print(piv.round(4).to_string())

    print("\n" + "=" * 104)
    print("B. BEST REAL CONFIG per symbol x timeframe, against that cell's NULL ceiling")
    print("=" * 104)
    rows = []
    for (sym, tf), g in res.groupby(["symbol", "tf"]):
        real, nul = g[~g.is_null], g[g.is_null]
        if not len(real):
            continue
        b = real.loc[real.expR_min.idxmax()]
        rows.append(dict(symbol=sym, tf=tf, best=b.tag, expR_min=b.expR_min,
                         expR_mean=b.expR_mean, ex_top=b.expR_ex_top_min, medR=b.medR_min,
                         trades=b.n_total, dd=b.dd_max, wins=b.n_windows_pos,
                         null_ceiling=round(nul.expR_min.max(), 4),
                         beats_null=bool(b.expR_min > nul.expR_min.max())))
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    print("\n" + "=" * 104)
    print("C. SURVIVORS — positive in all 3 windows, >=300 trades, outlier-proof, beats its "
          "own cell's null")
    print("=" * 104)
    surv = []
    for (sym, tf), g in res.groupby(["symbol", "tf"]):
        ceil = g[g.is_null].expR_min.max()
        s = g[(~g.is_null) & (g.expR_min > 0) & (g.n_windows_pos == 3) & (g.n_total >= 300)
              & (g.expR_ex_top_min > 0) & (g.expR_min > ceil)]
        surv.append(s)
    surv = pd.concat(surv, ignore_index=True) if surv else pd.DataFrame()
    real_n = int((~res.is_null).sum())
    print(f"  {len(surv)} of {real_n} real configs   (chance predicts ~{real_n*0.5**3:.0f} on the "
          f"regime test alone)")
    cols = ["symbol", "tf", "tag", "n_total", "expR_min", "expR_mean", "expR_ex_top_min",
            "medR_min", "dd_max"]
    print(surv.sort_values("expR_min", ascending=False)[cols].to_string(index=False)
          if len(surv) else "  NONE")

    print("\n" + "=" * 104)
    print("D. CROSS-INSTRUMENT — does any entry work on BOTH BTC and gold at the same timeframe?")
    print("=" * 104)
    x = (res[~res.is_null].pivot_table(index=["entry", "mgmt", "tf"], columns="symbol",
                                       values="expR_min"))
    if {"BTCUSDT", "XAUUSD"}.issubset(x.columns):
        x = x.dropna()
        both = x[(x.BTCUSDT > 0) & (x.XAUUSD > 0)]
        print(f"  positive on both: {len(both)} of {len(x)}  "
              f"(chance ~{len(x)//4})")
        print(both.sort_values("XAUUSD", ascending=False).head(15).round(4).to_string()
              if len(both) else "  NONE")
    print("\nsaved -> experiments/results_phase5_tf_instrument.csv")
    return res


if __name__ == "__main__":
    main()
