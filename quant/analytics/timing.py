"""
Time-of-day / day-of-week structure — and, crucially, whether it PERSISTS.

Two kinds of measurement live here:

* **Direction-agnostic market structure** — `efficiency_by_block` (Kaufman efficiency ratio: is this
  hour a clean one-way move or chop?) and `ema_followthrough` (after a price-vs-EMA cross, how far
  does price travel in the signal's direction, in ATR units?). Neither depends on being long or
  short, so they are not confounded by the market's drift over the sample.
* **Persistence** — `persistence` and `walkforward_block`. A bucket ranking measured in-sample is
  nearly always non-flat; the only question that matters is whether the SAME buckets stay good in the
  next period. If the rank correlation is ~0 and walk-forward blocking wins about half the time, the
  structure is noise and filtering on it is curve-fitting.

Use them together: measure, then prove it persists before you trade it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..indicators import add_atr
from ..indicators.overlays import ema as _ema

# UTC session labels for crypto (24/7); handy for reading hour tables.
SESSIONS = {**{h: "Asia" for h in range(0, 7)}, 7: "Asia/EU",
            **{h: "EU" for h in range(8, 13)}, **{h: "EU/US" for h in range(13, 17)},
            **{h: "US" for h in range(17, 22)}, **{h: "late US" for h in range(22, 24)}}


def efficiency_by_block(df: pd.DataFrame, *, freq: str = "h", time_col: str = "t",
                        price_col: str = "close") -> pd.DataFrame:
    """Kaufman efficiency ratio within each `freq` block: |net move| / total path travelled.

    1.0 = a perfectly clean directional move; ~0 = pure chop. Returns one row per block with the
    block's start time, its `hour`/`weekday`, and `er`.
    """
    x = df[[time_col, price_col]].copy()
    x["blk"] = x[time_col].dt.floor(freq)
    x["absdiff"] = x[price_col].diff().abs()
    g = x.groupby("blk")
    net = (g[price_col].last() - g[price_col].first()).abs()
    path = g["absdiff"].sum().replace(0, np.nan)
    out = pd.DataFrame({"er": (net / path).dropna()})
    out["hour"] = out.index.hour
    out["weekday"] = out.index.weekday
    return out.reset_index().rename(columns={"blk": time_col})


def ema_followthrough(df: pd.DataFrame, *, period: int = 50, fwd: int = 30,
                      time_col: str = "t") -> pd.DataFrame:
    """Direction-agnostic EMA signal quality.

    On every close-vs-EMA cross (either way), measure how far price moves IN THE SIGNAL'S DIRECTION
    over the next `fwd` bars, normalised by ATR. Positive = crosses follow through at that time;
    negative = they mean-revert. Returns one row per signal with `hour`, `weekday`, `ft`.
    """
    x = df.reset_index(drop=True)
    e = _ema(x["close"], period).to_numpy(dtype=float)
    c = x["close"].to_numpy(dtype=float)
    up = (c > e) & np.r_[False, c[:-1] <= e[:-1]]
    dn = (c < e) & np.r_[False, c[:-1] >= e[:-1]]
    atr = add_atr(x, 14)["atr_14"].to_numpy(dtype=float)
    fwd_move = np.r_[c[fwd:], np.full(fwd, np.nan)] - c
    with np.errstate(invalid="ignore", divide="ignore"):
        ft = np.where(up, fwd_move, np.where(dn, -fwd_move, np.nan)) / atr
    out = pd.DataFrame({time_col: x[time_col], "hour": x[time_col].dt.hour,
                        "weekday": x[time_col].dt.weekday, "ft": ft})
    return out.dropna(subset=["ft"])


def bucket_by_period(obs: pd.DataFrame, *, bucket: str = "hour", value: str = "ft",
                     freq: str = "Q", time_col: str = "t") -> pd.DataFrame:
    """Pivot observations into a [bucket x period] table of means — the input to `persistence`."""
    x = obs.copy()
    x["_period"] = x[time_col].dt.to_period(freq).astype(str)
    return x.pivot_table(index=bucket, columns="_period", values=value, aggfunc="mean")


def persistence(table: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation of a [bucket x period] table between CONSECUTIVE periods.

    This is the test that decides whether a time-of-day/weekday effect is tradeable. Mean rho near 0
    (or negative) means the ranking reshuffles every period — the structure is noise.
    """
    from scipy.stats import spearmanr
    cols = list(table.columns)
    rows = []
    for a, b in zip(cols[:-1], cols[1:]):
        rho, p = spearmanr(table[a], table[b])
        rows.append({"train": a, "test": b, "rho": round(float(rho), 3), "p_value": round(float(p), 3)})
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["mean_rho"] = float(out["rho"].mean())
    return out


def walkforward_block(obs: pd.DataFrame, table: pd.DataFrame, *, bucket: str = "hour",
                      value: str = "ft", n_block: int = 6, freq: str = "Q",
                      time_col: str = "t") -> pd.DataFrame:
    """Pick the worst `n_block` buckets in each period, then score the NEXT period with them removed.

    `improved` should be true far more often than half the time for blocking to be worth doing.
    """
    x = obs.copy()
    x["_period"] = x[time_col].dt.to_period(freq).astype(str)
    cols = list(table.columns)
    rows = []
    for a, b in zip(cols[:-1], cols[1:]):
        worst = list(table[a].nsmallest(n_block).index)
        nxt = x[x["_period"] == b]
        if not len(nxt):
            continue
        kept = nxt[~nxt[bucket].isin(worst)][value].mean()
        allb = nxt[value].mean()
        rows.append({"train": a, "test": b, "blocked": worst,
                     "test_all": round(float(allb), 4), "test_kept": round(float(kept), 4),
                     "improved": bool(kept > allb)})
    out = pd.DataFrame(rows)
    if len(out):
        out.attrs["hit_rate"] = float(out["improved"].mean())
    return out


def trade_buckets(trades: pd.DataFrame, *, bucket: str = "hour") -> pd.DataFrame:
    """Per-bucket expectancy in R for a trade table (needs entry/stop/qty to derive risk)."""
    t = trades.copy()
    if not len(t):
        return pd.DataFrame()
    t["risk"] = (t["stop_price"] - t["entry_price"]).abs() * t["qty"]
    t["R"] = t["pnl"] / t["risk"]
    et = pd.to_datetime(t["entry_time"])
    t[bucket] = et.dt.hour if bucket == "hour" else et.dt.weekday
    g = t.groupby(bucket)["R"]
    return pd.DataFrame({"expectancy_R": g.mean().round(4), "trades": g.size(),
                         "total_R": g.sum().round(2)})
