"""
Trade attribution: slice performance by time-of-day, weekday, month, market session, and
market regime (volatility / trend). Answers "when / in what conditions does this strategy work?".

Each function takes the engine's `trades` DataFrame (columns: entry_time, exit_time, side, pnl,
return_pct, entry_i, ...) and returns a tidy per-bucket stats table sorted by the bucket.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..signals.time_filters import SESSIONS_UTC


def _trade_stats(pnl: np.ndarray) -> dict:
    n = pnl.size
    if n == 0:
        return {"n_trades": 0, "win_rate_pct": 0.0, "total_pnl": 0.0,
                "avg_pnl": 0.0, "profit_factor": 0.0}
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)
    return {
        "n_trades": int(n),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "total_pnl": float(pnl.sum()),
        "avg_pnl": float(pnl.mean()),
        "profit_factor": float(pf),
    }


def bucket_stats(trades: pd.DataFrame, key: pd.Series, *, label: str = "bucket") -> pd.DataFrame:
    """Group trades by `key` and compute per-bucket stats."""
    if trades.empty:
        return pd.DataFrame(columns=[label, "n_trades", "win_rate_pct", "total_pnl",
                                     "avg_pnl", "profit_factor"])
    pnl = trades["pnl"].to_numpy(np.float64)
    k = np.asarray(key)
    rows = []
    for b in pd.unique(k):
        rows.append({label: b, **_trade_stats(pnl[k == b])})
    out = pd.DataFrame(rows).sort_values(label).reset_index(drop=True)
    return out


def _entry_local(trades: pd.DataFrame, tz: Optional[str]) -> pd.Series:
    t = pd.to_datetime(trades["entry_time"])
    if tz is not None and t.dt.tz is not None:
        t = t.dt.tz_convert(tz)
    return t


def by_hour(trades, *, tz: Optional[str] = None) -> pd.DataFrame:
    return bucket_stats(trades, _entry_local(trades, tz).dt.hour, label="hour")


def by_weekday(trades, *, tz: Optional[str] = None) -> pd.DataFrame:
    t = _entry_local(trades, tz)
    names = t.dt.strftime("%a")  # Mon, Tue, ...
    out = bucket_stats(trades, t.dt.weekday, label="weekday")
    if not out.empty:
        out["day"] = out["weekday"].map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
    return out


def by_month(trades, *, tz: Optional[str] = None) -> pd.DataFrame:
    return bucket_stats(trades, _entry_local(trades, tz).dt.month, label="month")


def by_session(trades, *, tz: Optional[str] = None) -> pd.DataFrame:
    """Per-session stats (a trade can belong to more than one overlapping session)."""
    if trades.empty:
        return pd.DataFrame(columns=["session", "n_trades", "win_rate_pct", "total_pnl",
                                     "avg_pnl", "profit_factor"])
    h = _entry_local(trades, tz).dt.hour.to_numpy()
    pnl = trades["pnl"].to_numpy(np.float64)
    rows = []
    for name, (start, end) in SESSIONS_UTC.items():
        mask = (h >= start) & (h < end) if start <= end else (h >= start) | (h < end)
        rows.append({"session": name, **_trade_stats(pnl[mask])})
    return pd.DataFrame(rows)


def by_regime(trades: pd.DataFrame, df: pd.DataFrame, *, ema_period: int = 200,
              atr_period: int = 14, vol_buckets: int = 3) -> pd.DataFrame:
    """Attribution by market regime at entry: trend (price vs EMA) x volatility (ATR% tercile).

    `df` must be the SAME frame the backtest ran on (trades carry integer `entry_i` into it).
    """
    from ..indicators import atr, ema
    if trades.empty:
        return pd.DataFrame(columns=["regime", "n_trades", "win_rate_pct", "total_pnl",
                                     "avg_pnl", "profit_factor"])
    close = df["close"].to_numpy(np.float64)
    e = ema(df["close"], ema_period).to_numpy(np.float64)
    a = (atr(df, atr_period) / df["close"]).to_numpy(np.float64)

    ei = trades["entry_i"].to_numpy(np.int64)
    ei = np.clip(ei, 0, len(df) - 1)
    trend = np.where(close[ei] >= e[ei], "uptrend", "downtrend")

    vol_at = a[ei]
    finite = vol_at[np.isfinite(vol_at)]
    if finite.size >= vol_buckets:
        qs = np.nanquantile(finite, np.linspace(0, 1, vol_buckets + 1))
        labels = ["lowvol", "midvol", "highvol"][:vol_buckets]
        vol = pd.cut(vol_at, bins=np.unique(qs), labels=labels[:len(np.unique(qs)) - 1],
                     include_lowest=True).astype(object)
        vol = pd.Series(vol).fillna("lowvol").to_numpy()
    else:
        vol = np.full(len(ei), "n/a", dtype=object)

    regime = np.array([f"{t}/{v}" for t, v in zip(trend, vol)], dtype=object)
    return bucket_stats(trades, regime, label="regime")


def monthly_returns(equity_curve: pd.DataFrame, *, time_col: str = "t") -> pd.DataFrame:
    """Month-by-month return % from the equity curve (pivot: year x month)."""
    eq = equity_curve[[time_col, "equity"]].dropna().set_index(time_col)["equity"]
    if eq.empty:
        return pd.DataFrame()
    m = eq.resample("ME").last()
    start = eq.iloc[0]
    prev = m.shift(1)
    prev.iloc[0] = start
    ret = (m / prev - 1.0) * 100.0
    out = pd.DataFrame({"year": ret.index.year, "month": ret.index.month, "return_pct": ret.values})
    return out.pivot(index="year", columns="month", values="return_pct")
