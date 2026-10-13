"""
The calendar on a bar grid -- for research filters and for strategies.

Every column is measured from the bar's CLOSE, the moment a signal on that bar is decided.

WHAT IS NOT LOOKAHEAD HERE, AND WHAT WOULD BE. A scheduled release is on the calendar days or
weeks before it happens, so "CPI is in 40 minutes" was knowable 40 minutes before CPI: the
schedule columns below use only the time of the release. Its ACTUAL is a different matter -- it
exists from the release onward -- so `news_last_surprise_z` takes a release only from the first bar
that CLOSES after it. Unscheduled items (a speech added that morning) are the one soft spot: the
calendar does not say when they were added. Filter on level >= high to keep to scheduled data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.data.base import TIME_COL
from quant.data.timeframe import bar_interval

from .calendar import select

_TIMED = ("exact", "official", "typical")


def _ns(t) -> np.ndarray:
    """Times -> int64 nanoseconds since the epoch, UTC."""
    s = pd.to_datetime(pd.Series(t), utc=True).dt.tz_convert(None)
    return s.to_numpy("datetime64[ns]").astype(np.int64)


def event_times(events: pd.DataFrame, coverage=None) -> np.ndarray:
    """Sorted, unique release instants (int64 ns, UTC) of the timed events kept by `coverage`."""
    ev = events if coverage is None else select(events, coverage)
    ev = ev[ev["time_q"].isin(_TIMED)]
    return np.unique(_ns(ev["t"]))


def _closes(df: pd.DataFrame, time_col: str) -> np.ndarray:
    return _ns(pd.to_datetime(df[time_col], utc=True) + bar_interval(df, time_col=time_col))


def news_features(df: pd.DataFrame, events: pd.DataFrame, *, coverage=None,
                  time_col: str = TIME_COL) -> pd.DataFrame:
    """-> a frame on df's index:
         news_next_min       minutes from the bar's close to the next covered release (inf: none)
         news_prev_min       minutes since the last covered release at or before the close
         news_day            the bar's New York date has a covered timed release
         news_last_surprise_z  the latest released surprise (z, see values.surprise_z) among
                             covered events, from the first bar closing after it
    """
    close = _closes(df, time_col)
    ts = event_times(events, coverage)
    nxt = np.searchsorted(ts, close, side="right")            # first release AFTER the close
    prv = nxt - 1
    nm = np.full(len(close), np.inf)
    ok = nxt < len(ts)
    nm[ok] = (ts[nxt[ok]] - close[ok]) / 6e10
    pm = np.full(len(close), np.inf)
    ok = prv >= 0
    pm[ok] = (close[ok] - ts[prv[ok]]) / 6e10

    day = pd.to_datetime(df[time_col], utc=True).dt.tz_convert("America/New_York").dt.date
    ev_days = set(pd.to_datetime(ts, utc=True).tz_convert("America/New_York").date)
    nd = day.map(lambda d: d in ev_days).to_numpy(bool)

    ev = events if coverage is None else select(events, coverage)
    ev = ev[ev["time_q"].isin(_TIMED) & ev["surprise_z"].notna()].sort_values("t")
    sz = np.full(len(close), np.nan)
    if len(ev):
        et = _ns(ev["t"])
        # a release is known from the first bar whose close is AFTER it (strictly)
        j = np.searchsorted(et, close, side="left") - 1
        ok = j >= 0
        sz[ok] = ev["surprise_z"].to_numpy(float)[j[ok]]
    return pd.DataFrame({"news_next_min": nm, "news_prev_min": pm, "news_day": nd,
                         "news_last_surprise_z": sz}, index=df.index)


def news_blackout(df: pd.DataFrame, events: pd.DataFrame, *, before: float = 30,
                  after: float = 30, coverage=None, time_col: str = TIME_COL) -> np.ndarray:
    """True on bars whose close is within `before` minutes before a covered release or `after`
    minutes after it -- the usual "no new trades around the news" rule."""
    f = news_features(df, events, coverage=coverage, time_col=time_col)
    return ((f["news_next_min"] <= before) | (f["news_prev_min"] <= after)).to_numpy(bool)
