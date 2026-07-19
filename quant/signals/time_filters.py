"""
Time-of-day / session / weekday signal filters.

All operate on a tz-aware time column (default 't') and return boolean numpy arrays, so they
compose with the other primitives via all_of/any_of. Sessions are evaluated in the timezone of
the `t` column — localize your data to the market tz (or pass tz) for session-accurate filters.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

# Common FX/metals sessions in UTC (approx.; adjust for DST as needed).
SESSIONS_UTC = {
    "london": (7, 16),
    "newyork": (12, 21),
    "tokyo": (0, 9),
    "sydney": (21, 6),   # wraps midnight
}


def _local(df: pd.DataFrame, time_col: str, tz: Optional[str]) -> pd.Series:
    t = df[time_col]
    if tz is not None:
        t = t.dt.tz_convert(tz)
    return t


def hour_between(df: pd.DataFrame, start_hour: int, end_hour: int, *,
                 time_col: str = "t", tz: Optional[str] = None) -> np.ndarray:
    """True where local hour is in [start_hour, end_hour). Handles windows that wrap midnight."""
    h = _local(df, time_col, tz).dt.hour.to_numpy()
    if start_hour <= end_hour:
        return (h >= start_hour) & (h < end_hour)
    return (h >= start_hour) | (h < end_hour)


def in_session(df: pd.DataFrame, session: str, *, time_col: str = "t",
               tz: Optional[str] = None) -> np.ndarray:
    s = session.lower().strip()
    if s not in SESSIONS_UTC:
        raise ValueError(f"Unknown session '{session}'. Known: {sorted(SESSIONS_UTC)}")
    start, end = SESSIONS_UTC[s]
    return hour_between(df, start, end, time_col=time_col, tz=tz)


def weekday_in(df: pd.DataFrame, days: Iterable[int], *, time_col: str = "t",
               tz: Optional[str] = None) -> np.ndarray:
    """days: 0=Mon .. 6=Sun."""
    wd = _local(df, time_col, tz).dt.weekday.to_numpy()
    allowed = set(int(d) for d in days)
    return np.isin(wd, list(allowed))


def not_weekend(df: pd.DataFrame, *, time_col: str = "t", tz: Optional[str] = None) -> np.ndarray:
    return weekday_in(df, [0, 1, 2, 3, 4], time_col=time_col, tz=tz)


def between_times(df: pd.DataFrame, start: str, end: str, *, time_col: str = "t",
                  tz: Optional[str] = None) -> np.ndarray:
    """True where local time-of-day is in [start, end), e.g. start='13:30', end='20:00'."""
    t = _local(df, time_col, tz)
    minutes = t.dt.hour.to_numpy() * 60 + t.dt.minute.to_numpy()
    sh, sm = [int(x) for x in start.split(":")]
    eh, em = [int(x) for x in end.split(":")]
    smin, emin = sh * 60 + sm, eh * 60 + em
    if smin <= emin:
        return (minutes >= smin) & (minutes < emin)
    return (minutes >= smin) | (minutes < emin)
