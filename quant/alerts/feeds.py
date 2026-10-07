"""
Closed 5-minute bars, straight from each provider -- no parquet research cache in the loop.

Every market is fetched at 5m and every alert timeframe is resampled from it, so a market costs one
small request however many timeframes watch it. Only CLOSED bars are returned: a bar whose end lies
in the future is the forming candle, and an alert read off it would repaint.

A provider call can hang (a Dukascopy request once stalled 20 minutes with no error), and a thread
cannot be killed. So each call runs in a shared pool with a timeout, and a market whose previous
call is STILL running is skipped rather than given a second one -- at most one stuck thread per
market, never a pile of them.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from typing import Dict, Optional

import pandas as pd

from quant.data.base import TIME_COL

BAR = pd.Timedelta(minutes=5)
COLS = [TIME_COL, "open", "high", "low", "close", "volume"]
_BINANCE = "https://api.binance.com/api/v3/klines"


def _closed(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df = df[df[TIME_COL] + BAR <= now]
    return df[COLS].drop_duplicates(TIME_COL, keep="last").sort_values(TIME_COL).reset_index(drop=True)


def binance_5m(symbol: str, start: pd.Timestamp, now: pd.Timestamp, *, timeout: float = 15.0):
    import requests
    rows, t = [], int(start.timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    while t < end:
        r = requests.get(_BINANCE, params=dict(symbol=symbol, interval="5m", startTime=t,
                                               endTime=end, limit=1000), timeout=timeout)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        t = int(batch[-1][0]) + 300_000
        if len(batch) < 1000:
            break
    df = pd.DataFrame([r[:6] for r in rows], columns=COLS)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL].astype("int64"), unit="ms", utc=True)
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    return _closed(df, now)


def _quiet(record):                                  # dukascopy_python logs every page it reads,
    return record.levelno >= logging.WARNING        # in LOCAL time, and resets its level per call


logging.getLogger("DUKASCRIPT").addFilter(_quiet)


def dukascopy_5m(symbol: str, start: pd.Timestamp, now: pd.Timestamp):
    import dukascopy_python as d
    from quant.data.dukascopy import _instrument, _normalize
    raw = d.fetch(_instrument(symbol), d.INTERVAL_MIN_5, d.OFFER_SIDE_BID,
                  start.to_pydatetime().astimezone(timezone.utc),
                  now.to_pydatetime().astimezone(timezone.utc), max_retries=2)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COLS)
    return _closed(_normalize(raw), now)


_FETCHERS = {"binance": binance_5m, "dukascopy": dukascopy_5m}


class Feed:
    """fetch(markets, starts, now) -> {market name: closed 5m bars, or the Exception it raised}."""

    def __init__(self, timeout: float = 90.0, workers: int = 4):
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alerts-feed")
        self._inflight: Dict[str, object] = {}
        self._lock = threading.Lock()

    def fetch(self, markets, starts: Dict[str, pd.Timestamp], now: pd.Timestamp):
        futs, out = {}, {}
        for m in markets:
            with self._lock:
                prev = self._inflight.get(m.name)
                if prev is not None and not prev.done():
                    out[m.name] = TimeoutError("previous request is still hanging")
                    continue
                f = self._pool.submit(_FETCHERS[m.source], m.symbol, starts[m.name], now)
                self._inflight[m.name] = f
            futs[m.name] = f
        for name, f in futs.items():
            try:
                out[name] = f.result(timeout=self.timeout)
            except Exception as e:                    # TimeoutError included: the thread lingers,
                out[name] = e                         # and the next tick will not stack another
        return out


class FakeFeed:
    """For tests and dry runs: serves bars from a {market: 5m frame} dict as if live at `now`."""

    def __init__(self, bars: Dict[str, pd.DataFrame], fail: Optional[set] = None):
        self.bars, self.fail, self.calls = bars, set(fail or ()), []

    def fetch(self, markets, starts, now):
        out = {}
        for m in markets:
            self.calls.append((m.name, starts[m.name], now))
            if m.name in self.fail:
                out[m.name] = ConnectionError("fake outage")
                continue
            df = _closed(self.bars[m.name], now)
            df = df[df[TIME_COL] >= starts[m.name]].reset_index(drop=True)
            out[m.name] = df if len(df) else pd.DataFrame(columns=COLS)   # untyped, as the
        return out                                                        # real feeds send it
