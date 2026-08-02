"""
PHASE 10 — EXIT DIAGNOSTIC: how much of each winner are we actually capturing?

Before building exit machinery, measure whether there is anything to capture. Two different questions,
both of which "extract the max juice" confuses into one:

  1. CAPTURE   while the trade was open, how far did it go in our favour (MFE) versus what we banked?
               capture = realised_R / MFE_R. Low capture = we sat through a move and gave it back.
  2. CONTINUATION  after we exited, did the move keep going? Measured as the extra R available in the
               next 30/60/120 bars beyond the exit price. High continuation = we left early.

These have opposite fixes. Low capture says trail tighter or take profit sooner; high continuation
says hold longer or trail looser. Optimising one without measuring the other is how exit "tuning"
turns into curve-fitting.

The NULL entry is measured alongside, because if a random entry shows the same capture and
continuation profile, then the numbers describe BTC's path structure rather than anything about the
strategy — and no exit rule built on them is an edge.

Note the theory: with a zero-edge entry, optional stopping says no exit rule changes expectancy
(minus costs). Exits can still change the SHAPE of the distribution -- fewer, larger winners -- which
is what this phase is really about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data import get_ohlcv
from quant.indicators import add_emas, add_sr_zones, add_zigzag_structure
from experiments.mgmt_sweep import EntryBundle
from experiments.search_harness import CFG_LEVERED, SPREAD_BTC
from experiments.phase9_long_fair_test import WINDOWS

TF = "15m"
HORIZONS = [30, 60, 120]

CONFIGS = [
    ("wt53_short", "wt_cross_short@v1", {"wt_level": 53.0}),
    ("wt53_long", "wt_cross_long@v1", {"wt_level": 53.0}),
    ("vwapz2_long", "vwap_rev_long@v1", {"vr_sigma": 2.0}),
    ("NULL_short", "time_entry_short@v1", {"te_every": 60}),
    ("NULL_long", "time_entry_long@v1", {"te_every": 60}),
]
MGMT = [("atr@v1", "rr@v1", {"stop_atr_mult": 2.5, "tp_rr": 1.5}, "rr1.5"),
        ("atr@v1", "rr@v1", {"stop_atr_mult": 2.5, "tp_rr": 3.0}, "rr3"),
        ("atr@v1", "trailing@v1", {"stop_atr_mult": 2.5, "trail_pct": 1.0}, "trail1%")]


def excursions(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Per trade: MFE and MAE while open, plus how much MORE was available after the exit."""
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    n = len(df)
    rows = []
    for t in trades.itertuples():
        a, b = int(t.entry_i), int(t.exit_i)
        if b < 0 or b <= a:
            continue
        risk = abs(t.stop_price - t.entry_price)
        if not np.isfinite(risk) or risk <= 0:
            continue
        sgn = 1.0 if t.side == "long" else -1.0
        # while open (bars after entry, up to and including the exit bar)
        seg_hi, seg_lo = hi[a + 1:b + 1], lo[a + 1:b + 1]
        if not len(seg_hi):
            continue
        best = seg_hi.max() if sgn > 0 else seg_lo.min()
        worst = seg_lo.min() if sgn > 0 else seg_hi.max()
        mfe = sgn * (best - t.entry_price) / risk
        mae = sgn * (worst - t.entry_price) / risk
        realised = sgn * (t.exit_price - t.entry_price) / risk
        rec = dict(trade_R=realised, mfe_R=mfe, mae_R=mae, reason=t.close_reason,
                   bars=int(b - a))
        # what was still on the table AFTER the exit
        for h in HORIZONS:
            e = min(b + h, n - 1)
            if e <= b:
                rec[f"cont{h}"] = np.nan
                continue
            fh, fl = hi[b + 1:e + 1], lo[b + 1:e + 1]
            if not len(fh):
                rec[f"cont{h}"] = np.nan
                continue
            far = fh.max() if sgn > 0 else fl.min()
            rec[f"cont{h}"] = sgn * (far - t.exit_price) / risk
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    cfg = {**CFG_LEVERED, "spread": SPREAD_BTC}
    frames = []
    for tag, skey, ep in CONFIGS:
        for stop, tp, mp, mtag in MGMT:
            for wname, (a, b) in WINDOWS.items():
                d = get_ohlcv("BTCUSDT", TF, start=a, end=b, source="binance", tz="UTC")
                d = add_emas(d, [50, 200]).reset_index(drop=True)
                res = EntryBundle(d, skey, ep).run(stop, tp, mp, cfg)
                if not len(res.trades):
                    continue
                # `run` returns a SimResult; excursions only needs high/low, and the
                # stop/TP prepare steps never reorder or drop rows, so `d` aligns.
                e = excursions(d, res.trades)
                e["cfg"], e["mgmt"], e["window"] = tag, mtag, wname
                frames.append(e)
        print(f"  done {tag}", flush=True)
    E = pd.concat(frames, ignore_index=True)
    E["is_null"] = E.cfg.str.startswith("NULL")
    E["won"] = E.trade_R > 0
    E.to_csv("experiments/results_phase10_excursions.csv", index=False)

    print("\n" + "=" * 100)
    print("A. CAPTURE ON WINNING TRADES — how much of the favourable move did we bank?")
    print("=" * 100)
    w = E[E.won]
    g = (w.groupby(["cfg", "mgmt"])
          .agg(n=("trade_R", "size"), realised=("trade_R", "median"),
               mfe=("mfe_R", "median"),
               capture=("trade_R", lambda s: np.nan)).reset_index())
    cap = w.assign(c=w.trade_R / w.mfe_R.replace(0, np.nan)).groupby(["cfg", "mgmt"]).c.median()
    g["capture"] = g.set_index(["cfg", "mgmt"]).index.map(cap).values
    print(g.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("B. CONTINUATION AFTER EXIT — extra R still available once we were out")
    print("   (measured on WINNERS; positive = the move kept going without us)")
    print("=" * 100)
    c = (w.groupby(["cfg", "mgmt"])
          .agg(**{f"cont{h}": (f"cont{h}", "median") for h in HORIZONS}).reset_index())
    print(c.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("C. THE SAME FOR THE NULL ENTRY — is any of this specific to the strategy?")
    print("=" * 100)
    comp = (E[E.won].assign(c=lambda x: x.trade_R / x.mfe_R.replace(0, np.nan))
              .groupby(["is_null", "mgmt"])
              .agg(n=("trade_R", "size"), capture=("c", "median"), mfe=("mfe_R", "median"),
                   **{f"cont{h}": (f"cont{h}", "median") for h in HORIZONS}).reset_index())
    comp["is_null"] = comp.is_null.map({False: "strategy", True: "NULL"})
    print(comp.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("D. WHERE THE MONEY GOES — MFE reached vs what each exit type banked")
    print("=" * 100)
    r = (E.groupby("reason")
          .agg(n=("trade_R", "size"), median_R=("trade_R", "median"),
               median_mfe=("mfe_R", "median"), median_mae=("mae_R", "median"),
               median_bars=("bars", "median")).reset_index()
          .sort_values("n", ascending=False))
    print(r.round(3).to_string(index=False))

    print("\n" + "=" * 100)
    print("E. THE PRIZE — how many trades reached a given R but did not close there?")
    print("=" * 100)
    rows = []
    for thr in (1.0, 1.5, 2.0, 3.0, 5.0):
        reached = E[E.mfe_R >= thr]
        rows.append(dict(mfe_at_least=thr, trades=len(reached),
                         pct_of_all=round(100 * len(reached) / len(E), 1),
                         median_realised=round(float(reached.trade_R.median()), 3),
                         gave_back=round(float((reached.mfe_R - reached.trade_R).median()), 3),
                         pct_ending_negative=round(100 * float((reached.trade_R < 0).mean()), 1)))
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nsaved -> experiments/results_phase10_excursions.csv")
    return E


if __name__ == "__main__":
    main()
