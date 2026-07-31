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


def _as_int_set(v, what: str, lo: int, hi: int) -> list:
    """Accept a single int or an iterable of ints; validate the range. `block_hours=12` is a natural
    thing to type, so don't make it an error."""
    if v is None:
        return []
    vals = [v] if isinstance(v, (int, np.integer)) else list(v)
    out = []
    for x in vals:
        i = int(x)
        if not (lo <= i <= hi):
            raise ValueError(f"{what} value {i} out of range {lo}..{hi}")
        out.append(i)
    return out


def hour_in(df: pd.DataFrame, hours, *, time_col: str = "t",
            tz: Optional[str] = None) -> np.ndarray:
    """True where the local hour is any of `hours` (0-23) — an arbitrary, non-contiguous set.

    Negate it to BLOCK those hours: `~hour_in(df, [12, 16, 18])`, which `hour_between` cannot
    express because it only takes one contiguous window. A single int is accepted too.
    """
    h = _local(df, time_col, tz).dt.hour.to_numpy()
    return np.isin(h, _as_int_set(hours, "hour", 0, 23))


def normalize_time_ranges(ranges) -> list:
    """Accept either a single ("HH:MM", "HH:MM") pair or a sequence of them, and always return a
    list of pairs. Writing one range without the outer tuple is the natural thing to type, so both
    forms are supported:

        ("13:30", "14:15")                          -> [("13:30", "14:15")]
        (("13:30", "14:15"), ("20:00", "20:30"))    -> both ranges
    """
    if ranges is None:
        return []
    seq = list(ranges)
    if not seq:
        return []
    # a bare pair of "HH:MM" strings
    if len(seq) == 2 and all(isinstance(x, str) for x in seq):
        seq = [(seq[0], seq[1])]
    out = []
    for i, r in enumerate(seq):
        if isinstance(r, str) or len(tuple(r)) != 2:
            raise ValueError(
                f"time range #{i} must be a ('HH:MM', 'HH:MM') pair, got {r!r}. "
                f"For a single range write (('13:30','14:15'),) or ('13:30','14:15').")
        out.append((str(tuple(r)[0]), str(tuple(r)[1])))
    return out


def in_any_times(df: pd.DataFrame, ranges, *, time_col: str = "t",
                 tz: Optional[str] = None) -> np.ndarray:
    """True where local time-of-day falls in ANY of the ("HH:MM", "HH:MM") ranges. Minute-level,
    so it can express windows like 13:30-14:15. Negate to block them.

    Accepts one range or many (see `normalize_time_ranges`)."""
    out = None
    for start, end in normalize_time_ranges(ranges):
        m = between_times(df, start, end, time_col=time_col, tz=tz)
        out = m if out is None else (out | m)
    return out if out is not None else np.zeros(len(df), dtype=bool)


def weekday_in(df: pd.DataFrame, days, *, time_col: str = "t",
               tz: Optional[str] = None) -> np.ndarray:
    """days: 0=Mon .. 6=Sun. A single int is accepted as well as an iterable."""
    wd = _local(df, time_col, tz).dt.weekday.to_numpy()
    return np.isin(wd, _as_int_set(days, "weekday", 0, 6))


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
