"""
PHASE 6F — is the expectancy distinguishable from zero AT ALL?

The trim-top-K gate showed the Phase-3 candidates are carried by well under 1% of their trades. When
a sample mean depends on a handful of observations from a fat right tail, the usual "it was positive
in all three windows" reasoning is worthless: three windows of a lottery are still a lottery. The
right tool is a confidence interval on the mean that respects the actual shape of the distribution.

Two bootstraps, because they answer different objections:
  * IID       — resample trades with replacement. Asks: given this distribution of outcomes, how
                often would the mean come out positive?
  * BLOCK     — resample contiguous blocks of trades (stationary/moving-block). Also preserves any
                serial dependence, so clustered winners cannot be broken up into independent luck.

A config is only worth further work if the LOWER bound stays above zero — and the NULL entry has to
fail the same test, or the test is not discriminating.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.mgmt_sweep import EntryBundle
from experiments.phase6_verify import CANDIDATES, R_of
from experiments.search_harness import CFG_LEVERED, SPREAD_BTC, make_windows

N_BOOT = 20000
BLOCK = 25


def boot_ci(R: np.ndarray, n_boot: int = N_BOOT, block: int = 0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(R)
    if n < 50:
        return {}
    if block <= 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        means = R[idx].mean(axis=1)
    else:
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(n_boot, nb))
        offs = np.arange(block)
        idx = (starts[:, :, None] + offs[None, None, :]) % n      # circular blocks
        means = R[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    return dict(mean=float(R.mean()), lo5=float(np.percentile(means, 5)),
                lo1=float(np.percentile(means, 1)),
                hi95=float(np.percentile(means, 95)),
                p_neg=float((means <= 0).mean()))


def main():
    w = make_windows("BTCUSDT", "1m")
    base = {**CFG_LEVERED, "spread": SPREAD_BTC}
    pd.set_option("display.width", 260)
    rows = []
    for tag, skey, ep, stop, tp, mp, ts in CANDIDATES:
        allR = []
        for k, d in w.items():
            run = EntryBundle(d, skey, ep).run(stop, tp, mp,
                                               {**base, "max_bars_in_trade": int(ts)})
            R = R_of(run)
            allR.append(R)
            for name, blk in (("iid", 0), ("block", BLOCK)):
                ci = boot_ci(R, block=blk, seed=hash((tag, k, name)) % 10_000)
                if ci:
                    rows.append(dict(cfg=tag, window=k, boot=name, n=len(R),
                                     **{q: round(v, 4) for q, v in ci.items()}))
        pooled = np.concatenate(allR)
        for name, blk in (("iid", 0), ("block", BLOCK)):
            ci = boot_ci(pooled, block=blk, seed=hash((tag, "pooled", name)) % 10_000)
            rows.append(dict(cfg=tag, window="POOLED", boot=name, n=len(pooled),
                             **{q: round(v, 4) for q, v in ci.items()}))
    res = pd.DataFrame(rows)
    res.to_csv("experiments/results_phase6_bootstrap.csv", index=False)

    print("=" * 100)
    print(f"BOOTSTRAP CONFIDENCE INTERVALS ON expR  ({N_BOOT:,} resamples, block={BLOCK})")
    print("=" * 100)
    print("\nPOOLED over all three windows:")
    p = res[res.window == "POOLED"]
    print(p[["cfg", "boot", "n", "mean", "lo1", "lo5", "hi95", "p_neg"]].to_string(index=False))

    print("\nPer window (5th percentile of the bootstrap mean — the lower bound):")
    piv = res[res.window != "POOLED"].pivot_table(index=["cfg", "boot"], columns="window",
                                                  values="lo5")
    print(piv.round(4).to_string())

    print("\n" + "=" * 100)
    print("VERDICT — lower bound > 0 in EVERY window, on BOTH bootstraps?")
    print("=" * 100)
    v = (res[res.window != "POOLED"].groupby("cfg")
            .agg(min_lo5=("lo5", "min"), min_lo1=("lo1", "min"),
                 max_p_neg=("p_neg", "max")).reset_index())
    v["survives"] = v.min_lo5 > 0
    print(v.round(4).to_string(index=False))
    print("\n  `max_p_neg` = the chance this config's mean comes out NEGATIVE on a resample.")
    print("  Compare every row against the NULL rows: a gate the metronome also passes is not a gate.")
    print("\nsaved -> experiments/results_phase6_bootstrap.csv")


if __name__ == "__main__":
    main()
