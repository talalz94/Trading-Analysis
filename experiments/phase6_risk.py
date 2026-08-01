"""
PHASE 6 — the risk layer, and whether it can do anything at all.

Daily loss caps, max-trades-per-day and consecutive-loss lockouts all rest on one assumption: that
losses CLUSTER. If trade outcomes are independent, a lockout cannot improve expectancy — it only
removes exposure, scaling both the wins and the losses down together. That is a testable property,
and testing it is much cheaper (and much more honest) than sweeping lockout lengths and picking the
one that happened to look best.

Part 1 measures it directly: autocorrelation of per-trade R, and P(loss | previous loss) against the
unconditional P(loss), across several configs and windows.
Part 2 sweeps the risk controls on the leading configs and reports what they actually change.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.mgmt_sweep import EntryBundle
from experiments.phase3_management import ENTRIES
from experiments.search_harness import CFG_LEVERED, SPREAD_BTC, make_windows, r_stats


def trade_R(run) -> np.ndarray:
    t = run.trades
    if not len(t):
        return np.array([])
    risk = (t["stop_price"] - t["entry_price"]).abs() * t["qty"]
    R = (t["pnl"] / risk).replace([np.inf, -np.inf], np.nan).dropna()
    return R.to_numpy(float)


def clustering_stats(R: np.ndarray) -> dict:
    if len(R) < 100:
        return {}
    loss = R <= 0
    p = loss.mean()
    prev = loss[:-1]
    nxt = loss[1:]
    p_ll = nxt[prev].mean() if prev.any() else np.nan
    p_lw = nxt[~prev].mean() if (~prev).any() else np.nan
    out = dict(n=len(R), p_loss=round(float(p), 4),
               p_loss_after_loss=round(float(p_ll), 4),
               p_loss_after_win=round(float(p_lw), 4),
               excess=round(float(p_ll - p), 4))
    for lag in (1, 2, 3, 5):
        x, y = R[:-lag], R[lag:]
        out[f"acf{lag}"] = round(float(np.corrcoef(x, y)[0, 1]), 4) if len(x) > 10 else np.nan
    return out


def main():
    w = make_windows("BTCUSDT", "1m")
    cfg = {**CFG_LEVERED, "spread": SPREAD_BTC}
    pd.set_option("display.width", 250)

    print("=" * 100)
    print("PART 1 — do losses CLUSTER? (if not, no lockout or loss cap can add expectancy)")
    print("=" * 100)
    rows = []
    for etag, skey, ep, _ in ENTRIES:
        for wname, d in w.items():
            b = EntryBundle(d, skey, ep)
            run = b.run("atr@v1", "rr@v1", {"stop_atr_mult": 1.5, "tp_rr": 2.0}, cfg)
            st = clustering_stats(trade_R(run))
            if st:
                rows.append(dict(entry=etag, window=wname, **st))
    cl = pd.DataFrame(rows)
    agg = (cl.groupby("entry")
             .agg(windows=("n", "size"), trades=("n", "sum"), p_loss=("p_loss", "mean"),
                  p_loss_after_loss=("p_loss_after_loss", "mean"), excess=("excess", "mean"),
                  acf1=("acf1", "mean"), acf2=("acf2", "mean"), acf3=("acf3", "mean"),
                  acf5=("acf5", "mean"))
             .reset_index().sort_values("excess", ascending=False))
    print(agg.round(4).to_string(index=False))
    n_obs = len(cl)
    se = float(np.nanmean([1 / np.sqrt(x) for x in cl["n"]]))
    print(f"\n  approx standard error on each autocorrelation ~ {se:.4f} "
          f"(1/sqrt(trades), averaged over {n_obs} config-windows)")
    print("  |acf| below that band, and `excess` near 0, means outcomes are effectively independent")
    cl.to_csv("experiments/results_phase6_clustering.csv", index=False)

    print("\n" + "=" * 100)
    print("PART 2 — risk controls on the leading configs: what do they actually change?")
    print("=" * 100)
    risk_grid = [("none", {}),
                 ("daily_cap_1%", {"daily_loss_limit": 1.0}),
                 ("daily_cap_2%", {"daily_loss_limit": 2.0}),
                 ("daily_cap_4%", {"daily_loss_limit": 4.0}),
                 ("risk_0.25%", {"sizing_value": 0.25}),
                 ("risk_0.5%", {"sizing_value": 0.5}),
                 ("max_1_open", {"max_open_trades": 1})]
    rows = []
    for etag, skey, ep, _ in ENTRIES[:6]:
        bundles = {k: EntryBundle(d, skey, ep) for k, d in w.items()}
        for rname, extra in risk_grid:
            per = {}
            for k, b in bundles.items():
                run = b.run("atr@v1", "rr@v1", {"stop_atr_mult": 1.5, "tp_rr": 2.0},
                            {**cfg, **extra})
                per[k] = r_stats(run)
            exps = [v["expR"] for v in per.values() if v["expR"] == v["expR"]]
            rows.append(dict(entry=etag, risk=rname,
                             trades=sum(v["n"] for v in per.values()),
                             expR_min=round(min(exps), 4) if exps else np.nan,
                             ret_min=round(min(v["ret"] for v in per.values()), 2),
                             dd_max=round(max(v["dd"] for v in per.values()), 2)))
    rk = pd.DataFrame(rows)
    print(rk.pivot_table(index="risk", values=["expR_min", "dd_max", "ret_min"],
                         aggfunc="median").round(4).to_string())
    print("\n  per-entry detail:")
    print(rk.pivot_table(index="entry", columns="risk", values="expR_min").round(4).to_string())
    print("\n  drawdown (max over windows):")
    print(rk.pivot_table(index="entry", columns="risk", values="dd_max").round(2).to_string())
    rk.to_csv("experiments/results_phase6_risk.csv", index=False)
    print("\nsaved -> experiments/results_phase6_{clustering,risk}.csv")


if __name__ == "__main__":
    main()
