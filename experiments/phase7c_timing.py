"""
PHASE 7c — when does a compressed range actually release?

The specific claim to test: a quiet weekend range breaks when the US comes in. That is checkable
directly rather than by proxy — find every compression EPISODE, find the bar where it ends, and look
at the clock.

An episode = a contiguous run of bars whose trailing compression percentile is below `QUIET_Q`. It
"releases" at the first later bar whose percentile exceeds `LOUD_Q`. Recorded per episode: when it
started, when it released, how long it waited, and how far price travelled over the release.

The instance that prompted this (Saturday 2026-08-01) is printed alongside, so a single vivid case
can be seen against the base rate rather than standing in for it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

QUIET_Q, LOUD_Q = 0.2, 0.6
MAX_WAIT = 720          # give an episode 12 hours to release before calling it unresolved
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def episodes(d: pd.DataFrame) -> pd.DataFrame:
    pct = d["cmp_pct"].to_numpy(float)
    c = d["close"].to_numpy(float)
    hi, lo = d["high"].to_numpy(float), d["low"].to_numpy(float)
    t = d["t"].to_numpy()
    quiet = np.nan_to_num(pct, nan=1.0) <= QUIET_Q
    loud = np.nan_to_num(pct, nan=0.0) >= LOUD_Q
    start = quiet & ~np.r_[False, quiet[:-1]]                 # first bar of a quiet run
    idx = np.flatnonzero(start)
    rows = []
    n = len(d)
    for i in idx:
        j = i
        while j < n - 1 and not loud[j]:                       # walk to the release
            j += 1
            if j - i > MAX_WAIT:
                break
        if j - i > MAX_WAIT or j >= n - 1:
            continue
        k = min(j + 120, n - 1)
        travel_bps = 1e4 * (hi[j:k].max() - lo[j:k].min()) / c[j]
        signed_bps = 1e4 * (c[k] - c[j]) / c[j]
        ts, tr = pd.Timestamp(t[i]), pd.Timestamp(t[j])
        rows.append(dict(start=ts, release=tr, wait_bars=int(j - i),
                         start_dow=ts.day_name()[:3], start_hour=ts.hour,
                         rel_dow=tr.day_name()[:3], rel_hour=tr.hour,
                         travel_bps=round(travel_bps, 1), signed_bps=round(signed_bps, 1)))
    return pd.DataFrame(rows)


def main():
    from quant.data import get_ohlcv
    from experiments.phase7_compression import prepare, SPANS
    pd.set_option("display.width", 250)
    eps = []
    for k, (a, b) in SPANS.items():
        d = prepare(get_ohlcv("BTCUSDT", "1m", start=a, end=b, source="binance", tz="UTC"))
        e = episodes(d)
        e["window"] = k
        eps.append(e)
        print(f"  {k}: {len(e)} compression episodes", flush=True)
    E = pd.concat(eps, ignore_index=True)
    E.to_csv("experiments/results_phase7c_episodes.csv", index=False)

    print("\n" + "=" * 92)
    print(f"{len(E):,} compression episodes (quiet<= {QUIET_Q}, released at >= {LOUD_Q})")
    print("=" * 92)
    print(f"  median wait to release : {E.wait_bars.median():.0f} bars "
          f"({E.wait_bars.median()/60:.1f} h)")
    print(f"  median travel over release (120 bars): {E.travel_bps.median():.1f} bps")
    print(f"  median |signed| move   : {E.signed_bps.abs().median():.1f} bps")
    print(f"  share that move UP     : {100*(E.signed_bps > 0).mean():.1f}%   "
          f"<- 50% means the release direction is a coin flip")

    print("\nA. WHEN DOES THE RELEASE HAPPEN? (UTC hour of release)")
    h = (E.groupby("rel_hour").agg(episodes=("travel_bps", "size"),
                                   med_travel=("travel_bps", "median"),
                                   med_wait=("wait_bars", "median")).reset_index())
    h["share_%"] = (100 * h.episodes / h.episodes.sum()).round(1)
    print(h.to_string(index=False))
    peak = h.loc[h.episodes.idxmax()]
    print(f"\n  busiest release hour: {int(peak.rel_hour):02d}:00 UTC "
          f"({peak['share_%']}% of releases)   US cash open = 13:30 UTC")

    print("\nB. WEEKEND COMPRESSION — does it wait for the weekday open?")
    E["start_weekend"] = E.start_dow.isin(["Sat", "Sun"])
    wk = E[E.start_weekend]
    print(f"  episodes starting on a weekend: {len(wk):,} of {len(E):,} "
          f"({100*len(wk)/len(E):.1f}%)")
    print("\n  where do WEEKEND-started episodes release?")
    t = (wk.groupby("rel_dow").agg(n=("travel_bps", "size"),
                                   med_travel=("travel_bps", "median"))
         .reindex(DOW).dropna())
    t["share_%"] = (100 * t.n / t.n.sum()).round(1)
    print(t.to_string())
    same = 100 * (wk.rel_dow.isin(["Sat", "Sun"])).mean()
    print(f"\n  {same:.1f}% of weekend compressions release WHILE IT IS STILL THE WEEKEND")

    print("\nC. TRAVEL BY START DAY vs RELEASE DAY (median bps over the release)")
    print(E.pivot_table(index="start_dow", columns="rel_dow", values="travel_bps",
                        aggfunc="median").reindex(DOW)[
        [c for c in DOW if c in E.rel_dow.unique()]].round(0).to_string())

    print("\nD. DOES A LONGER WAIT MEAN A BIGGER MOVE? (the 'coiled spring' claim)")
    E["wait_bucket"] = pd.cut(E.wait_bars, [0, 30, 60, 120, 240, 480, 720],
                              labels=["<30", "30-60", "60-120", "120-240", "240-480", "480+"])
    print(E.groupby("wait_bucket", observed=True)
           .agg(n=("travel_bps", "size"), med_travel=("travel_bps", "median"),
                med_abs_move=("signed_bps", lambda s: s.abs().median())).round(1).to_string())

    print("\nE. THE INSTANCE THAT PROMPTED THIS — Saturday 2026-08-01")
    inst = E[(E.start >= "2026-08-01") & (E.start < "2026-08-02")]
    if len(inst):
        print(inst[["start", "release", "wait_bars", "rel_hour", "travel_bps",
                    "signed_bps"]].to_string(index=False))
        big = inst.travel_bps.max()
        pctile = 100 * (E.travel_bps < big).mean()
        print(f"\n  its biggest release travelled {big:.0f} bps = the "
              f"{pctile:.1f}th percentile of all {len(E):,} episodes")
    else:
        print("  (no episode recorded that day at these thresholds)")
    print("\nsaved -> experiments/results_phase7c_episodes.csv")


if __name__ == "__main__":
    main()
