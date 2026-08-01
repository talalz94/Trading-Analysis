"""
PHASE 4 — hybrid / conditional strategies: does ANY regime make a dead entry work?

This is the "works in certain conditions / timeframes / sessions" idea, tested properly. The trap it
has to avoid is obvious once stated: almost any filter improves a losing rule, because cutting trades
cuts cost drag, and cutting the *right* trades in-sample is what overfitting looks like.

So every conditioner is applied to the real entries **and to the null entry**, and the number that
matters is the LIFT RELATIVE TO THE NULL's lift under the same conditioner:

    lift(entry, cond) = expR_min(entry, cond) - expR_min(entry, none)
    verdict           = lift(real, cond) - lift(NULL, cond)

If a filter helps the metronome as much as it helps the signal, it is removing costly bars, not
finding good ones — and it will not survive contact with new data.

Conditioners: volatility terciles, trend/chop by efficiency ratio, volatility-shock windows (the
announcement proxy — no calendar is available), position in the recent range, the three trading
sessions, higher-timeframe EMA bias, and a few combinations.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from quant.strategies.conditions import above_mtf, below_mtf
from quant.strategies.regimes import range_position, trendiness, vol_regime, vol_shock

from experiments.mgmt_sweep import sweep
from experiments.phase3_management import ENTRIES
from experiments.search_harness import make_windows

MTF = ("5min", [200])
SESSIONS = {"asia": list(range(0, 8)), "eu": list(range(8, 16)), "us": list(range(16, 24))}


def conditioners() -> list:
    """(name, Cond or None, extra entry params)."""
    out = [("none", None, {})]
    for m in ("high", "mid", "low"):
        out.append((f"vol_{m}", vol_regime(m), {}))
    for m in ("trending", "choppy"):
        out.append((m, trendiness(m), {}))
    out += [("shock_now", vol_shock(within=0), {}),
            ("post_shock60", vol_shock(within=60), {}),
            ("no_shock60", ~vol_shock(within=60), {}),
            ("range_upper", range_position("upper"), {}),
            ("range_lower", range_position("lower"), {}),
            ("vol_high+trending", vol_regime("high") & trendiness("trending"), {}),
            ("no_shock+trending", (~vol_shock(within=60)) & trendiness("trending"), {}),
            ("vol_low+choppy", vol_regime("low") & trendiness("choppy"), {})]
    for name, hrs in SESSIONS.items():
        out.append((f"sess_{name}", None, {"hours": hrs}))
    # higher-timeframe bias: trade only WITH, or only AGAINST, the 5m 200-EMA
    out.append(("htf_with", None, {"_htf": "with", "mtf_emas": MTF}))
    out.append(("htf_against", None, {"_htf": "against", "mtf_emas": MTF}))
    return out


def build_entries() -> list:
    """Every (entry x conditioner) pair, as `mgmt_sweep.sweep` entry tuples."""
    out = []
    for etag, skey, eparams, _ in ENTRIES:
        side = "long" if skey.endswith("long@v1") else "short"
        for cname, cond, extra in conditioners():
            p = {**eparams, **{k: v for k, v in extra.items() if not k.startswith("_")}}
            c = cond
            if extra.get("_htf"):
                with_trend = above_mtf(*MTF[0:1], MTF[1][0]) if side == "long" \
                    else below_mtf(MTF[0], MTF[1][0])
                c = with_trend if extra["_htf"] == "with" else ~with_trend
            out.append((f"{etag}@{cname}", skey, p, c))
    return out


MGMT = [dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 1.5, "tp_rr": 2.0},
             tag="atr1.5/rr2"),
        dict(stop="atr@v1", tp="rr@v1", params={"stop_atr_mult": 2.5, "tp_rr": 1.5},
             cfg={"max_bars_in_trade": 60}, tag="atr2.5/rr1.5/t60")]


def main():
    w = make_windows("BTCUSDT", "1m")
    entries = build_entries()
    print(f"Phase 4: {len(entries)} entry x conditioner pairs x {len(MGMT)} management "
          f"= {len(entries)*len(MGMT)} configs x {len(w)} windows", flush=True)
    t0 = time.time()
    res = sweep(w, entries, MGMT, verbose=False)
    res[["base", "cond"]] = res.entry.str.split("@", n=1, expand=True)
    res.to_csv("experiments/results_phase4_conditioners.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"done in {time.time()-t0:.0f}s")
    if res.error.astype(bool).any():
        print("ERRORS:", res[res.error.astype(bool)].error.value_counts().head().to_dict())

    base = (res[res.cond == "none"].set_index(["base", "mgmt"]).expR_min)
    res["lift"] = res.apply(
        lambda r: r.expR_min - base.get((r.base, r.mgmt), np.nan), axis=1)
    is_null = res.base.str.startswith("NULL")

    print("\n" + "=" * 100)
    print("A. RAW LIFT per conditioner (median over entries+management) — the naive view")
    print("=" * 100)
    raw = (res[res.cond != "none"].groupby("cond")
              .agg(configs=("lift", "size"), med_lift=("lift", "median"),
                   med_expR=("expR_min", "median"), med_trades=("n_total", "median"))
              .reset_index().sort_values("med_lift", ascending=False))
    print(raw.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("B. NULL-ADJUSTED LIFT — real entries' lift MINUS the null entries' lift.")
    print("   Positive = the conditioner finds good bars. ~0 = it only removes costly bars.")
    print("=" * 100)
    nl = res[is_null & (res.cond != "none")].groupby("cond").lift.median()
    rl = res[~is_null & (res.cond != "none")].groupby("cond").lift.median()
    adj = (pd.DataFrame({"real_lift": rl, "null_lift": nl})
             .assign(net=lambda d: d.real_lift - d.null_lift)
             .reset_index().sort_values("net", ascending=False))
    print(adj.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("C. BEST CONFIGS OVERALL")
    print("=" * 100)
    cols = ["tag", "n_total", "expR_min", "expR_mean", "expR_ex_top_min", "medR_min",
            "n_windows_pos", "dd_max", "lift"]
    print(res.head(20)[cols].to_string(index=False))

    real = res[~is_null]
    null_ceiling = res[is_null].expR_min.max()
    surv = real[(real.expR_min > 0) & (real.n_windows_pos == 3) & (real.n_total >= 300)
                & (real.expR_ex_top_min > 0) & (real.expR_min > null_ceiling)]
    print("\n" + "=" * 100)
    print(f"SURVIVORS (all 3 regimes positive, >=300 trades, outlier-proof, beats the null "
          f"ceiling {null_ceiling:.4f}): {len(surv)} of {len(real)}")
    print(f"chance predicts ~{len(real) * 0.5 ** 3:.0f} on the regime test alone")
    print("=" * 100)
    print(surv[cols].to_string(index=False) if len(surv) else "  NONE")
    print("\nsaved -> experiments/results_phase4_conditioners.csv")
    return res


if __name__ == "__main__":
    main()
