"""
Reference levels traders actually watch, plus VWAP and volume features.

All strictly causal. The previous day's high is known the moment that day ends, so it is valid for
every bar of the next day; a *running* session extreme is an expanding max/min over bars seen so far,
never the session's final value.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

SESSION_BOUNDS = {"asia": (0, 8), "eu": (8, 16), "us": (16, 24)}   # UTC blocks for a 24/7 market


def _session_key(t: pd.Series) -> pd.Series:
    """Integer session id: yyyymmdd * 10 + block. Integers rather than formatted strings because this
    runs over hundreds of thousands of bars in the sweeps and strftime dominates the cost."""
    h = t.dt.hour
    blk = np.where(h < 8, 0, np.where(h < 16, 1, 2))
    ymd = t.dt.year * 10000 + t.dt.month * 100 + t.dt.day
    return pd.Series(ymd.to_numpy() * 10 + blk, index=t.index)


def add_prev_day_levels(df: pd.DataFrame, *, time_col: str = "t", prefix: str = "pd") -> pd.DataFrame:
    """Previous CALENDAR day's completed high / low / close / mid, carried across the next day."""
    out = df.copy()
    day = out[time_col].dt.floor("D")
    agg = out.groupby(day).agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    prev = agg.shift(1)
    out[f"{prefix}_high"] = day.map(prev["h"]).to_numpy()
    out[f"{prefix}_low"] = day.map(prev["l"]).to_numpy()
    out[f"{prefix}_close"] = day.map(prev["c"]).to_numpy()
    out[f"{prefix}_mid"] = (out[f"{prefix}_high"] + out[f"{prefix}_low"]) / 2.0
    return out


def add_week_levels(df: pd.DataFrame, *, time_col: str = "t", prefix: str = "wk") -> pd.DataFrame:
    """Previous ISO week's completed high / low."""
    out = df.copy()
    wk = out[time_col].dt.to_period("W").astype(str)
    agg = out.groupby(wk).agg(h=("high", "max"), l=("low", "min"))
    order = list(dict.fromkeys(wk))
    ph = {c: agg["h"].get(p, np.nan) for c, p in zip(order[1:], order[:-1])}
    pl = {c: agg["l"].get(p, np.nan) for c, p in zip(order[1:], order[:-1])}
    out[f"{prefix}_high"] = wk.map(ph).to_numpy()
    out[f"{prefix}_low"] = wk.map(pl).to_numpy()
    return out


def add_session_levels(df: pd.DataFrame, *, time_col: str = "t", prefix: str = "sess",
                       or_bars: int = 30) -> pd.DataFrame:
    """Session structure, all causal.

    Adds `{prefix}_` + block / run_high / run_low / prev_high / prev_low / open / bar /
    or_high / or_low, where:
      block      0 Asia (0-8 UTC) | 1 EU (8-16) | 2 US (16-24)
      run_*      RUNNING extreme of the session so far (expanding)
      prev_*     the PREVIOUS session's completed extremes
      bar        bar index within the session (0-based)
      or_*       OPENING RANGE over the first `or_bars`, exposed only AFTER the range completes
    """
    out = df.copy()
    key = _session_key(out[time_col])
    out[f"{prefix}_id"] = key
    out[f"{prefix}_block"] = np.where(out[time_col].dt.hour < 8, 0,
                                      np.where(out[time_col].dt.hour < 16, 1, 2))
    g = out.groupby(key, sort=False)
    out[f"{prefix}_run_high"] = g["high"].cummax()
    out[f"{prefix}_run_low"] = g["low"].cummin()
    out[f"{prefix}_open"] = g["open"].transform("first")
    out[f"{prefix}_bar"] = g.cumcount()
    done = g.agg(h=("high", "max"), l=("low", "min"))
    order = list(dict.fromkeys(key))
    ph = {c: done["h"].get(p, np.nan) for c, p in zip(order[1:], order[:-1])}
    pl = {c: done["l"].get(p, np.nan) for c, p in zip(order[1:], order[:-1])}
    out[f"{prefix}_prev_high"] = key.map(ph).to_numpy()
    out[f"{prefix}_prev_low"] = key.map(pl).to_numpy()
    # opening range: extremes of the first `or_bars`, only visible once the range has completed
    inside = out[f"{prefix}_bar"] < int(or_bars)
    orh = out["high"].where(inside).groupby(key).cummax().groupby(key).ffill()
    orl = out["low"].where(inside).groupby(key).cummin().groupby(key).ffill()
    after = out[f"{prefix}_bar"] >= int(or_bars)
    out[f"{prefix}_or_high"] = orh.where(after).to_numpy()
    out[f"{prefix}_or_low"] = orl.where(after).to_numpy()
    return out


def add_vwap(df: pd.DataFrame, *, time_col: str = "t", anchor: str = "session",
             bands: tuple = (1.0, 2.0), prefix: str = "vwap") -> pd.DataFrame:
    """Session- or day-anchored VWAP with standard-deviation bands. Expanding, so fully causal.

    `anchor`: 'session' (Asia/EU/US blocks) | 'day'. Adds the VWAP, `_sd`, `_u<b>`/`_l<b>` bands and
    `_z` (close distance from VWAP in sd units).
    """
    out = df.copy()
    key = (_session_key(out[time_col]) if anchor == "session"
           else (out[time_col].dt.year * 10000 + out[time_col].dt.month * 100
                 + out[time_col].dt.day))
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    vol = out["volume"].astype(float).replace(0, np.nan).fillna(1.0)
    frame = pd.DataFrame({"k": key.to_numpy(), "pv": (tp * vol).to_numpy(),
                          "v": vol.to_numpy(), "pv2": (tp * tp * vol).to_numpy()})
    g = frame.groupby("k")
    cpv, cv, cpv2 = g["pv"].cumsum(), g["v"].cumsum(), g["pv2"].cumsum()
    vw = cpv / cv
    sd = np.sqrt(np.clip((cpv2 / cv) - vw ** 2, 0, None))
    out[prefix] = vw.to_numpy()
    out[f"{prefix}_sd"] = sd.to_numpy()
    for b in bands:
        out[f"{prefix}_u{b:g}"] = (vw + b * sd).to_numpy()
        out[f"{prefix}_l{b:g}"] = (vw - b * sd).to_numpy()
    sd_safe = sd.replace(0, np.nan).to_numpy()
    out[f"{prefix}_z"] = (out["close"].to_numpy() - vw.to_numpy()) / sd_safe
    return out


def add_volume_features(df: pd.DataFrame, *, time_col: str = "t", lookback: int = 50,
                        prefix: str = "vol") -> pd.DataFrame:
    """Volume features — unused anywhere in this project until now.

    Adds `{prefix}_z` (z-score vs the trailing window), `{prefix}_ratio` (vs trailing median) and
    `{prefix}_rel_tod` (vs the expanding median for this clock minute, so time-of-day seasonality in
    volume does not masquerade as a spike).
    """
    out = df.copy()
    v = out["volume"].astype(float)
    m = v.rolling(lookback, min_periods=lookback // 2).mean()
    s = v.rolling(lookback, min_periods=lookback // 2).std(ddof=0).replace(0, np.nan)
    out[f"{prefix}_z"] = ((v - m) / s).to_numpy()
    med = v.rolling(lookback, min_periods=lookback // 2).median().replace(0, np.nan)
    out[f"{prefix}_ratio"] = (v / med).to_numpy()
    tod = out[time_col].dt.hour * 60 + out[time_col].dt.minute
    exp_med = v.groupby(tod).transform(lambda x: x.shift(1).expanding(min_periods=5).median())
    out[f"{prefix}_rel_tod"] = (v / exp_med.replace(0, np.nan)).to_numpy()
    return out


def add_round_levels(df: pd.DataFrame, step: float = 1000.0, *, prefix: str = "rnd") -> pd.DataFrame:
    """Nearest round numbers above/below the close (psychological levels)."""
    out = df.copy()
    c = out["close"].to_numpy(dtype=float)
    out[f"{prefix}_below"] = np.floor(c / step) * step
    out[f"{prefix}_above"] = np.ceil(c / step) * step
    return out


def level_events(df: pd.DataFrame, level_col: str, *, tol_pct: float = 0.05,
                 side: str = "above", sweep_bars: int = 2) -> Dict[str, np.ndarray]:
    """Classify each bar's interaction with a level.

    `side='above'` = level sits overhead (resistance-like); `'below'` = underneath (support-like).

      break  close through the level
      fade   reached it (within tol) but closed back on the original side -> it held
      sweep  the wick pierced it and the close came back -> failed break / liquidity grab
      retest reached it again 2..20 bars after a break
    """
    lv = df[level_col].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    tol = tol_pct / 100.0
    ok = np.isfinite(lv)
    with np.errstate(invalid="ignore"):
        if side == "above":
            reached = hi >= lv * (1 - tol)
            pierced = hi > lv
            broke = c > lv
        else:
            reached = lo <= lv * (1 + tol)
            pierced = lo < lv
            broke = c < lv
        brk = broke & ok
        fade = reached & ~broke & ok
        sweep = pierced & ~broke & ok
    first_break = brk & ~np.r_[False, brk[:-1]]
    idx = np.arange(len(lv))
    last_break = np.where(first_break, idx, -10 ** 6)
    last_break = np.maximum.accumulate(last_break)
    since = idx - last_break
    retest = reached & ok & (since >= sweep_bars) & (since <= 20)
    return {"break": brk, "first_break": first_break, "fade": fade, "sweep": sweep,
            "retest": retest}
