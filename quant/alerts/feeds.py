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
# Binance refuses some servers outright -- every US one, and many datacenter ranges -- with HTTP
# 451 "restricted location" (or 403). Its market-data-only host serves the same public /klines; if
# the main host refuses, that one is tried, and whichever answered is remembered for the next call.
_BINANCE_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
_binance_host = [0]
_log = logging.getLogger("quant.alerts")


def _closed(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df = df[df[TIME_COL] + BAR <= now]
    return df[COLS].drop_duplicates(TIME_COL, keep="last").sort_values(TIME_COL).reset_index(drop=True)


def _binance_klines(params: dict, timeout: float):
    import requests
    first = _binance_host[0]
    refused = None
    for k in [first] + [i for i in range(len(_BINANCE_HOSTS)) if i != first]:
        r = requests.get(_BINANCE_HOSTS[k] + "/api/v3/klines", params=params, timeout=timeout)
        if r.status_code in (403, 451):              # "restricted location": try the other host
            refused = refused or r
            continue
        r.raise_for_status()
        if k != first:
            _log.info("Binance: %s refused this server (HTTP %s); using %s", _BINANCE_HOSTS[first],
                      refused.status_code if refused is not None else "?", _BINANCE_HOSTS[k])
            _binance_host[0] = k
        return r.json()
    raise requests.HTTPError(f"Binance refuses this server on every host (HTTP "
                             f"{refused.status_code}): move BTC to source='dukascopy', "
                             f"symbol='BTC/USD' -- docs/ALERTS_CLOUD.md", response=refused)


def binance_5m(symbol: str, start: pd.Timestamp, now: pd.Timestamp, *, timeout: float = 15.0):
    rows, t = [], int(start.timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    while t < end:
        batch = _binance_klines(dict(symbol=symbol, interval="5m", startTime=t, endTime=end,
                                     limit=1000), timeout)
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
                out[name] = f.result(timeout=self.timeout_for(starts[name], now))
            except TimeoutError:                     # the thread lingers, and the next tick
                out[name] = TimeoutError(            # will not stack another beside it
                    f"no answer in {self.timeout_for(starts[name], now):.0f} s")
            except Exception as e:
                out[name] = e
        return out

    def timeout_for(self, start, now) -> float:
        """A poll of a few new bars gets `timeout`; a first backfill of months gets longer. With
        everything switched on, BTC's first download is ~42k bars, about 43 Binance requests, and a
        flat 90 s cut it off on EVERY retry, since each retry asks for the same months again."""
        bars = max(0.0, (now - start) / pd.Timedelta(minutes=5))
        return self.timeout + 5.0 * bars / 1000


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
