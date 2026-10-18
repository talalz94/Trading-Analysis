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


# ══ PERIOD LEVELS AND MARKET SESSIONS -- what a chart draws, as tables ═══════════════════════════
PERIODS = {"D": "day", "W": "week", "M": "month", "Y": "year"}
_PERIOD_STEP = {"D": pd.DateOffset(days=1), "W": pd.DateOffset(weeks=1),
                "M": pd.DateOffset(months=1), "Y": pd.DateOffset(years=1)}


def _period_start(local: pd.Series, period: str) -> pd.Series:
    """The wall-clock start of each bar's period: midnight, Monday, the 1st, 1 January."""
    if period == "D":
        return local.dt.floor("D")
    return local.dt.to_period({"W": "W-SUN", "M": "M", "Y": "Y"}[period]).dt.start_time


def period_levels(frame: pd.DataFrame, period: str = "D", *, tz: str = "UTC",
                  time_col: str = "open_time") -> pd.DataFrame:
    """The PREVIOUS period's high and low, as the stretch of time they apply to.

    One row per period P found in `frame`: [start, end) is P in UTC (its boundaries are midnight,
    Monday, the 1st or 1 January in `tz`, DST included), and high / low are those of the period
    before it -- complete, so known from the first moment of P. That is the whole causality story:
    nothing in a row is unknown at its `start`.

    The period before P is the previous one THAT HAS BARS: Monday's "previous day" on a market
    shut at the weekend is Friday, as on every trading platform. A period the frame holds only
    part of (it starts mid-way) is never used as a source -- its high would be the high of the
    part loaded -- so give the frame a full period of history before the first one you draw.

    Columns: start, end, high, low, source_start (when the period the levels come from began).
    """
    if period not in PERIODS:
        raise ValueError(f"period must be one of {list(PERIODS)}, got {period!r}")
    cols = ["start", "end", "high", "low", "source_start"]
    if frame is None or len(frame) < 2:
        return pd.DataFrame(columns=cols)
    t = pd.to_datetime(frame[time_col], utc=True).reset_index(drop=True)
    local = t.dt.tz_convert(tz).dt.tz_localize(None)
    key = _period_start(local, period)
    g = pd.DataFrame({"k": key.to_numpy(),
                      "h": pd.to_numeric(frame["high"]).to_numpy(float),
                      "l": pd.to_numeric(frame["low"]).to_numpy(float)}).groupby("k", sort=True)
    agg = g.agg(h=("h", "max"), l=("l", "min"))
    bar = t.diff().median()
    first_whole = (local.iloc[0] - agg.index[0]) <= bar         # loaded from the period's start?
    src = pd.DataFrame({"high": agg["h"].shift(1).to_numpy(),
                        "low": agg["l"].shift(1).to_numpy(),
                        "source_start": pd.Series(agg.index).shift(1).to_numpy()},
                       index=agg.index)
    if not first_whole and len(src) > 1:
        src.iloc[1, :2] = np.nan                                # its source was a partial period
    src = src.iloc[1:].dropna(subset=["high", "low"])
    if not len(src):
        return pd.DataFrame(columns=cols)

    def utc(naive) -> pd.Series:
        return (pd.Series(pd.to_datetime(naive))
                .dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
                .dt.tz_convert("UTC"))

    start = utc(src.index)
    end = utc(pd.DatetimeIndex(src.index) + _PERIOD_STEP[period])
    out = pd.DataFrame({"start": start.to_numpy(), "end": end.to_numpy(),
                        "high": src["high"].to_numpy(), "low": src["low"].to_numpy(),
                        "source_start": utc(src["source_start"]).to_numpy()}, columns=cols)
    return out.dropna(subset=["start", "end"]).reset_index(drop=True)


def session_windows(sessions: Dict[str, tuple], start, end, *,
                    weekdays=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Every session's open and close in UTC between `start` and `end`.

    `sessions` = {name: (tz, "HH:MM" open, "HH:MM" close)} in each market's OWN clock, so daylight
    saving is that market's: London opens at 08:00 London time in summer and winter alike, which is
    07:00 or 08:00 UTC. A close earlier than the open runs past midnight. `weekdays` are the LOCAL
    days a session opens (Monday = 0), so Sydney's Monday session -- Sunday evening in UTC -- is
    there and its Saturday is not.

    Columns: name, open, close (UTC), tz, local_open, local_close.
    """
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    lo = lo.tz_localize("UTC") if lo.tzinfo is None else lo.tz_convert("UTC")
    hi = hi.tz_localize("UTC") if hi.tzinfo is None else hi.tz_convert("UTC")

    def at(day, hhmm, tz):
        return (pd.Timestamp(f"{day} {hhmm}")
                .tz_localize(tz, ambiguous=False, nonexistent="shift_forward").tz_convert("UTC"))

    rows = []
    for name, (tz, a, b) in sessions.items():
        days = pd.date_range((lo - pd.Timedelta(days=2)).tz_convert(tz).date(),
                             (hi + pd.Timedelta(days=1)).tz_convert(tz).date(), freq="D")
        for d in days:
            if d.weekday() not in weekdays:
                continue
            o, c = at(d.date(), a, tz), at(d.date(), b, tz)
            if c <= o:                                          # runs past local midnight
                c = at((d + pd.Timedelta(days=1)).date(), b, tz)
            if c > lo and o < hi:
                rows.append(dict(name=name, open=o, close=c, tz=tz, local_open=a, local_close=b))
    out = pd.DataFrame(rows, columns=["name", "open", "close", "tz", "local_open", "local_close"])
    return out.sort_values(["open", "name"]).reset_index(drop=True)


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
