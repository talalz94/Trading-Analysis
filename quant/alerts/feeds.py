"""
Closed 5-minute bars, straight from each provider -- no parquet research cache in the loop.

Every market is fetched at 5m and every alert timeframe is resampled from it, so a market costs one
small request however many timeframes watch it. A market some 1m alert watches is ALSO fetched at 1m
(`base="1m"`), into a store of its own -- only that alert's few hours of warm-up, never the months
the 5m store keeps. Only CLOSED bars are returned: a bar whose end lies in the future is the forming
candle, and an alert read off it would repaint.

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
BASE_BAR = {"1m": pd.Timedelta(minutes=1), "5m": BAR}
COLS = [TIME_COL, "open", "high", "low", "close", "volume"]
# Binance refuses some servers outright -- every US one, and many datacenter ranges -- with HTTP
# 451 "restricted location" (or 403). Its market-data-only host serves the same public /klines; if
# the main host refuses, that one is tried, and whichever answered is remembered for the next call.
_BINANCE_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
_binance_host = [0]
_log = logging.getLogger("quant.alerts")


def _closed(df: pd.DataFrame, now: pd.Timestamp, bar: pd.Timedelta = BAR) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=COLS)
    df = df[df[TIME_COL] + bar <= now]
    return df[COLS].drop_duplicates(TIME_COL, keep="last").sort_values(TIME_COL).reset_index(drop=True)


_tls = threading.local()
_ctx: list = []
_ctx_lock = threading.Lock()


def _tls_context():
    """The certificate store, loaded ONCE per process (see session())."""
    with _ctx_lock:
        if not _ctx:
            from requests.utils import DEFAULT_CA_BUNDLE_PATH
            from urllib3.util.ssl_ import create_urllib3_context
            c = create_urllib3_context()
            c.load_verify_locations(DEFAULT_CA_BUNDLE_PATH)
            _ctx.append(c)
    return _ctx[0]


def session():
    """This thread's requests.Session for the candle feeds: connections kept open, and verified
    with ONE certificate store per process. requests hands urllib3 the CA file for every new
    connection, and OpenSSL 3 parses it each time -- 4 s of CPU on the PC, ~2 s on the server,
    twice per Dukascopy fetch (the library opens a connection per page). That was most of a
    Dukascopy fetch's 11-33 s, and fetching 1m bars every minute would have spent a quarter of
    the server's CPU on it. Now a new connection costs ~0.05 s, and a wrong host, a self-signed,
    an expired and an untrusted certificate are still refused (checked, Oct 2026). No cookies
    are kept: the requests are exactly the ones made before, on a kept connection."""
    s = getattr(_tls, "session", None)
    if s is None:
        import http.cookiejar
        import requests
        from requests.adapters import HTTPAdapter

        class _Https(HTTPAdapter):
            def init_poolmanager(self, *a, **kw):
                kw["ssl_context"] = _tls_context()
                super().init_poolmanager(*a, **kw)

            def cert_verify(self, conn, url, verify, cert):
                super().cert_verify(conn, url, verify, cert)
                if verify is True:                   # already in the shared context: loading the
                    conn.ca_certs = None             # file again is the cost being avoided

        s = requests.Session()
        s.mount("https://", _Https())
        s.cookies.set_policy(http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
        _tls.session = s
    return s


class _DukascopyRequests:
    """Stands in for `requests` inside dukascopy_python, which calls requests.get() for every page:
    through session(), and with a timeout, which the library never sets (a request once hung 20
    minutes)."""

    @staticmethod
    def get(*a, **kw):
        kw.setdefault("timeout", 60)
        return session().get(*a, **kw)


def _binance_klines(params: dict, timeout: float):
    import requests
    first = _binance_host[0]
    refused = None
    for k in [first] + [i for i in range(len(_BINANCE_HOSTS)) if i != first]:
        r = session().get(_BINANCE_HOSTS[k] + "/api/v3/klines", params=params, timeout=timeout)
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


def binance_bars(symbol: str, start: pd.Timestamp, now: pd.Timestamp, *, base: str = "5m",
                 timeout: float = 15.0):
    rows, t = [], int(start.timestamp() * 1000)
    end = int(now.timestamp() * 1000)
    step = int(BASE_BAR[base].total_seconds() * 1000)
    while t < end:
        batch = _binance_klines(dict(symbol=symbol, interval=base, startTime=t, endTime=end,
                                     limit=1000), timeout)
        if not batch:
            break
        rows += batch
        t = int(batch[-1][0]) + step
        if len(batch) < 1000:
            break
    df = pd.DataFrame([r[:6] for r in rows], columns=COLS)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL].astype("int64"), unit="ms", utc=True)
    for c in COLS[1:]:
        df[c] = df[c].astype(float)
    return _closed(df, now, BASE_BAR[base])


def binance_5m(symbol: str, start: pd.Timestamp, now: pd.Timestamp, *, timeout: float = 15.0):
    return binance_bars(symbol, start, now, base="5m", timeout=timeout)


def binance_1m(symbol: str, start: pd.Timestamp, now: pd.Timestamp, *, timeout: float = 15.0):
    return binance_bars(symbol, start, now, base="1m", timeout=timeout)


def _quiet(record):                                  # dukascopy_python logs every page it reads,
    return record.levelno >= logging.WARNING        # in LOCAL time, and resets its level per call


logging.getLogger("DUKASCRIPT").addFilter(_quiet)


def dukascopy_bars(symbol: str, start: pd.Timestamp, now: pd.Timestamp, *, base: str = "5m"):
    import dukascopy_python as d
    from quant.data.dukascopy import _instrument, _normalize
    d.requests = _DukascopyRequests                  # its pages through session() (see there)
    interval = {"1m": d.INTERVAL_MIN_1, "5m": d.INTERVAL_MIN_5}[base]
    raw = d.fetch(_instrument(symbol), interval, d.OFFER_SIDE_BID,
                  start.to_pydatetime().astimezone(timezone.utc),
                  now.to_pydatetime().astimezone(timezone.utc), max_retries=2)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=COLS)
    return _closed(_normalize(raw), now, BASE_BAR[base])


def dukascopy_5m(symbol: str, start: pd.Timestamp, now: pd.Timestamp):
    return dukascopy_bars(symbol, start, now, base="5m")


def dukascopy_1m(symbol: str, start: pd.Timestamp, now: pd.Timestamp):
    return dukascopy_bars(symbol, start, now, base="1m")


_FETCHERS = {("binance", "5m"): binance_5m, ("dukascopy", "5m"): dukascopy_5m,
             ("binance", "1m"): binance_1m, ("dukascopy", "1m"): dukascopy_1m}


class Feed:
    """fetch(markets, starts, now, base="5m") -> {market name: closed bars of `base`, or the
    Exception it raised}."""

    def __init__(self, timeout: float = 90.0, workers: int = 4):
        self.timeout = timeout
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="alerts-feed")
        self._inflight: Dict[str, object] = {}
        self._lock = threading.Lock()

    def fetch(self, markets, starts: Dict[str, pd.Timestamp], now: pd.Timestamp,
              base: str = "5m"):
        futs, out = {}, {}
        for m in markets:
            key = m.name if base == "5m" else f"{m.name}@{base}"
            with self._lock:
                prev = self._inflight.get(key)
                if prev is not None and not prev.done():
                    out[m.name] = TimeoutError("previous request is still hanging")
                    continue
                f = self._pool.submit(_FETCHERS[m.source, base], m.symbol, starts[m.name], now)
                self._inflight[key] = f
            futs[m.name] = f
        for name, f in futs.items():
            wait = self.timeout_for(starts[name], now, base)
            try:
                out[name] = f.result(timeout=wait)
            except TimeoutError:                     # the thread lingers, and the next tick
                out[name] = TimeoutError(            # will not stack another beside it
                    f"no answer in {wait:.0f} s")
            except Exception as e:
                out[name] = e
        return out

    def timeout_for(self, start, now, base: str = "5m") -> float:
        """A poll of a few new bars gets `timeout`; a first backfill of months gets longer. With
        everything switched on, BTC's first download is ~42k bars, about 43 Binance requests, and a
        flat 90 s cut it off on EVERY retry, since each retry asks for the same months again."""
        bars = max(0.0, (now - start) / BASE_BAR[base])
        return self.timeout + 5.0 * bars / 1000


class FakeFeed:
    """For tests and dry runs: serves bars from a {market: 5m frame} dict as if live at `now`, and
    1m bars from `bars_1m` the same way."""

    def __init__(self, bars: Dict[str, pd.DataFrame], fail: Optional[set] = None,
                 bars_1m: Optional[Dict[str, pd.DataFrame]] = None):
        self.bars, self.fail, self.calls = bars, set(fail or ()), []
        self.bars_1m = bars_1m or {}

    def fetch(self, markets, starts, now, base: str = "5m"):
        out = {}
        src = self.bars if base == "5m" else self.bars_1m
        for m in markets:
            self.calls.append((m.name, starts[m.name], now) + (() if base == "5m" else (base,)))
            if m.name in self.fail:
                out[m.name] = ConnectionError("fake outage")
                continue
            df = _closed(src[m.name], now, BASE_BAR[base])
            df = df[df[TIME_COL] >= starts[m.name]].reset_index(drop=True)
            out[m.name] = df if len(df) else pd.DataFrame(columns=COLS)   # untyped, as the
        return out                                                        # real feeds send it
