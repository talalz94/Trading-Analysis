"""
PHASE 7 — the compression -> expansion theory, screened before any backtest.

The claim has TWO independent halves and they must not be tested together:

  (1) VOLATILITY   after a very quiet stretch, the next move is unusually big.
  (2) DIRECTION    the way price leaves the range (here: crossing the 200-EMA) says which way it goes.

A breakout strategy only needs (1). It is a straddle: you take both sides and let the size of the
expansion pay for the whipsaws. So (1) is screened as forward *absolute* travel and (2) as forward
*signed* return, separately, each differenced against the within-day-shuffled null.

Note what the null does here. Shuffling bars inside a day destroys volatility CLUSTERING, so if
compression genuinely forecasts expansion the real series must beat the null on (1) by a wide margin.
On (2) the null is the usual guard against the screen measuring itself.

Part C answers the weekend / US-open sub-question directly: when does compression occur, and when
does the expansion actually arrive?
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                              # noqa: E402
from quant.indicators import add_atr, add_compression, add_emas   # noqa: E402
from experiments.null_control import shuffle_within_day       # noqa: E402

SPANS = {"w1": ("2025-01-01", "2025-07-29"), "w2": ("2025-07-29", "2026-01-28"),
         "w3": ("2026-01-28", "2026-08-01")}
HORIZONS = [30, 60, 120, 240]
SHORT, LONG = 60, 480


def _fwd_extreme(arr: np.ndarray, h: int, how: str) -> np.ndarray:
    """max/min of `arr` over the NEXT h bars (strictly after the current one)."""
    rev = pd.Series(arr[::-1])
    r = (rev.rolling(h, min_periods=h).max() if how == "max"
         else rev.rolling(h, min_periods=h).min())
    m = r.to_numpy()[::-1]                    # m[i] = extreme over [i, i+h-1]
    return np.r_[m[1:], np.nan]               # shift to [i+1, i+h]


def prepare(d: pd.DataFrame) -> pd.DataFrame:
    d = add_compression(d, short=SHORT, long=LONG)
    d = add_atr(d, 14)
    d = add_emas(d, [200])
    return d.reset_index(drop=True)


def expansion_table(d: pd.DataFrame) -> pd.DataFrame:
    """Forward absolute travel, by compression decile. Deciles are of `cmp_pct`, which is already a
    trailing percentile rank, so decile 0 = the quietest bars this instrument has recently seen."""
    hi, lo, c = (d["high"].to_numpy(float), d["low"].to_numpy(float), d["close"].to_numpy(float))
    atr = d["atr_14"].to_numpy(float)
    pct = d["cmp_pct"].to_numpy(float)
    dec = np.floor(np.clip(pct, 0, 0.999) * 10)
    rows = []
    for h in HORIZONS:
        fh, fl = _fwd_extreme(hi, h, "max"), _fwd_extreme(lo, h, "min")
        travel = (fh - fl) / atr                      # total range covered, in ATR units
        move = np.abs(np.r_[c[h:], np.full(h, np.nan)] - c) / atr   # net |move|
        ok = np.isfinite(travel) & np.isfinite(dec) & np.isfinite(move)
        for dd in range(10):
            m = ok & (dec == dd)
            if m.sum() < 200:
                continue
            rows.append(dict(h=h, decile=int(dd), n=int(m.sum()),
                             travel=round(float(np.median(travel[m])), 3),
                             net_move=round(float(np.median(move[m])), 3)))
    return pd.DataFrame(rows)


def direction_table(d: pd.DataFrame, quiet_q: float = 0.2) -> pd.DataFrame:
    """After a QUIET stretch, does crossing the 200-EMA predict the SIGN of the next move?"""
    c = d["close"].to_numpy(float)
    e = d["ema_200"].to_numpy(float)
    atr = d["atr_14"].to_numpy(float)
    pct = d["cmp_pct"].to_numpy(float)
    prev = np.r_[np.nan, c[:-1]]
    pe = np.r_[np.nan, e[:-1]]
    with np.errstate(invalid="ignore"):
        up = (c > e) & (prev <= pe)
        dn = (c < e) & (prev >= pe)
        quiet = pct <= quiet_q
    rows = []
    for h in HORIZONS:
        fwd = (np.r_[c[h:], np.full(h, np.nan)] - c) / atr
        for name, mask, sign in [("cross_up|quiet", up & quiet, +1),
                                 ("cross_dn|quiet", dn & quiet, -1),
                                 ("cross_up|any", up, +1), ("cross_dn|any", dn, -1)]:
            m = mask & np.isfinite(fwd)
            if m.sum() < 100:
                continue
            rows.append(dict(h=h, event=name, n=int(m.sum()),
                             signed=round(float(sign * np.mean(fwd[m])), 4),
                             hit=round(float(100 * np.mean(sign * fwd[m] > 0)), 1)))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 250)
    real, null = {}, {}
    for k, (a, b) in SPANS.items():
        d = get_ohlcv("BTCUSDT", "1m", start=a, end=b, source="binance", tz="UTC")
        real[k] = prepare(d.copy())
        null[k] = prepare(shuffle_within_day(d.copy(), seed=hash(k) % 1000))
        print(f"  prepared {k}: {len(d):,} bars", flush=True)

    print("\n" + "=" * 96)
    print("A. DOES COMPRESSION PREDICT EXPANSION?  median forward travel (high-low) in ATR units")
    print("   decile 0 = quietest. REAL vs within-day-shuffled NULL (which has no vol clustering).")
    print("=" * 96)
    R = pd.concat([expansion_table(v) for v in real.values()]).groupby(["h", "decile"]).median()
    N = pd.concat([expansion_table(v) for v in null.values()]).groupby(["h", "decile"]).median()
    cmp_ = R[["travel"]].rename(columns={"travel": "real"}).join(
        N[["travel"]].rename(columns={"travel": "null"}))
    cmp_["ratio"] = (cmp_["real"] / cmp_["null"]).round(3)
    print(cmp_.reset_index().pivot(index="decile", columns="h").round(3).to_string())

    print("\n  Same, as NET |move| rather than total travel:")
    Rm = pd.concat([expansion_table(v) for v in real.values()]).groupby(["h", "decile"]).median()
    print(Rm.reset_index().pivot(index="decile", columns="h", values="net_move").round(3).to_string())

    print("\n" + "=" * 96)
    print("B. DOES THE 200-EMA CROSS PREDICT DIRECTION AFTER A QUIET STRETCH?")
    print("   `signed` = mean forward return in the traded direction, ATR units. ~0 = no edge.")
    print("=" * 96)
    Rd = pd.concat([direction_table(v) for v in real.values()]).groupby(["h", "event"]).agg(
        n=("n", "sum"), signed=("signed", "median"), hit=("hit", "median")).reset_index()
    Nd = pd.concat([direction_table(v) for v in null.values()]).groupby(["h", "event"]).agg(
        null_signed=("signed", "median")).reset_index()
    m = Rd.merge(Nd, on=["h", "event"])
    m["excess"] = (m.signed - m.null_signed).round(4)
    print(m.to_string(index=False))

    print("\n" + "=" * 96)
    print("C. WHEN does compression happen, and WHEN does the expansion arrive? (weekday / hour)")
    print("=" * 96)
    allr = pd.concat(real.values(), ignore_index=True)
    allr["dow"] = allr["t"].dt.day_name().str[:3]
    allr["hour"] = allr["t"].dt.hour
    allr["quiet"] = allr["cmp_pct"] <= 0.2
    fh = _fwd_extreme(allr["high"].to_numpy(float), 120, "max")
    fl = _fwd_extreme(allr["low"].to_numpy(float), 120, "min")
    allr["fwd_travel"] = (fh - fl) / allr["atr_14"].to_numpy(float)
    order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_d = (allr.groupby("dow").agg(pct_quiet=("quiet", "mean"),
                                    med_cmp=("cmp_pct", "median"),
                                    travel_if_quiet=("fwd_travel",
                                                     lambda s: np.nan))
            .reindex(order))
    by_d["pct_quiet"] = (100 * allr.groupby("dow")["quiet"].mean()).reindex(order).round(1)
    by_d["travel_if_quiet"] = (allr[allr.quiet].groupby("dow")["fwd_travel"]
                               .median().reindex(order).round(3))
    by_d["travel_all"] = allr.groupby("dow")["fwd_travel"].median().reindex(order).round(3)
    by_d["med_cmp"] = by_d["med_cmp"].round(3)
    print("\n  by weekday:")
    print(by_d.to_string())

    hr = pd.DataFrame({
        "pct_quiet": (100 * allr.groupby("hour")["quiet"].mean()).round(1),
        "travel_if_quiet": allr[allr.quiet].groupby("hour")["fwd_travel"].median().round(3),
        "travel_all": allr.groupby("hour")["fwd_travel"].median().round(3)})
    print("\n  by UTC hour (US cash open = 13:30 UTC summer / 14:30 winter):")
    print(hr.to_string())

    print("\n  weekend-quiet -> where does the expansion land?")
    wk = allr[allr.dow.isin(["Sat", "Sun"]) & allr.quiet]
    print(f"    weekend quiet bars: {len(wk):,}   median fwd travel {wk.fwd_travel.median():.3f} ATR")
    wd = allr[~allr.dow.isin(["Sat", "Sun"]) & allr.quiet]
    print(f"    weekday quiet bars: {len(wd):,}   median fwd travel {wd.fwd_travel.median():.3f} ATR")
    allr.to_pickle("experiments/_phase7_prepared.pkl")
    print("\nsaved prepared frame -> experiments/_phase7_prepared.pkl")


if __name__ == "__main__":
    main()
