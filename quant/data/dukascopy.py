"""
Dukascopy data source — free, no account, true spot/CFD **XAU/USD** (and FX) at 1m from 2003.

Uses the `dukascopy-python` package (install: `pip install dukascopy-python`). Data is fetched per
missing range and cached incrementally like the Binance provider (same parquet cache layout).

    from quant.data import get_ohlcv
    df = get_ohlcv("XAUUSD", "1m", start="2020-01-01", end="2024-12-31", source="dukascopy")

Notes: Dukascopy quotes bid/ask; this adapter uses the BID side by default (set offer="ask" or
"mid" via the source). Weekend gaps are normal for spot metals/FX. Bulk multi-year backfills pull
many files — the incremental cache means you only pay that once.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from .base import TIME_COL, validate_ohlcv
from .cache import incremental_fetch
from .store import cache_path
from ..logging_utils import get_logger

_log = get_logger("quant.data.dukascopy")

# Common symbol -> Dukascopy instrument code. Unmapped 6-char symbols fall back to XXX/YYY.
_SYMBOL_MAP = {
    "XAUUSD": "XAU/USD", "XAGUSD": "XAG/USD",
    "EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD", "USDCHF": "USD/CHF",
}


def _instrument(symbol: str) -> str:
    s = symbol.upper().replace("/", "")
    if s in _SYMBOL_MAP:
        return _SYMBOL_MAP[s]
    if len(s) == 6:
        return f"{s[:3]}/{s[3:]}"
    raise ValueError(f"Don't know the Dukascopy instrument for '{symbol}'. Add it to _SYMBOL_MAP.")


def _interval(interval: str):
    import dukascopy_python as d
    m = {
        "1m": d.INTERVAL_MIN_1, "5m": d.INTERVAL_MIN_5, "10m": d.INTERVAL_MIN_10,
        "15m": d.INTERVAL_MIN_15, "30m": d.INTERVAL_MIN_30,
        "1h": d.INTERVAL_HOUR_1, "4h": d.INTERVAL_HOUR_4, "1d": d.INTERVAL_DAY_1,
    }
    if interval not in m:
        raise ValueError(f"Unsupported Dukascopy interval '{interval}'. Use one of {sorted(m)}.")
    return m[interval]


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dukascopy_python result to the tidy OHLCV contract."""
    df = raw.copy()
    if df.index.name is not None or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    # find the timestamp column
    tcol = next((c for c in ("timestamp", "time", "date", "datetime", "index", TIME_COL)
                 if c in df.columns), df.columns[0])
    df = df.rename(columns={tcol: TIME_COL})
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    keep = [TIME_COL, "open", "high", "low", "close", "volume"]
    return df[[c for c in keep if c in df.columns]]


class DukascopySource:
    name = "dukascopy"

    def __init__(self, market: str = "cfd", cache_dir: Optional[str] = None, offer: str = "bid"):
        self.market = market
        self.cache_dir = cache_dir
        self.offer = offer

    def _raw_fetch(self, instrument, interval, offer_side):
        import dukascopy_python as d

        def fetch(s: pd.Timestamp, e: pd.Timestamp) -> pd.DataFrame:
            start = s.to_pydatetime().astimezone(timezone.utc)
            end = e.to_pydatetime().astimezone(timezone.utc)
            raw = d.fetch(instrument, interval, offer_side, start, end)
            if raw is None or len(raw) == 0:
                return pd.DataFrame()
            return _normalize(raw)
        return fetch

    def fetch(self, symbol: str, interval: str, start: str, end: Optional[str] = None,
              **kwargs) -> pd.DataFrame:
        import dukascopy_python as d
        instrument = _instrument(symbol)
        dk_interval = _interval(interval)   # map "1m" -> dukascopy INTERVAL_MIN_1
        offer_side = d.OFFER_SIDE_ASK if self.offer == "ask" else d.OFFER_SIDE_BID
        path = cache_path(symbol, interval, source=self.name, market=self.market,
                          data_dir=self.cache_dir)
        merged = incremental_fetch(self._raw_fetch(instrument, dk_interval, offer_side), path,
                                   start, end, interval, source=self.name, logger=_log)
        if merged.empty:
            return merged
        m = merged[TIME_COL] >= pd.Timestamp(start, tz="UTC")
        if end is not None:
            m &= merged[TIME_COL] <= pd.Timestamp(end, tz="UTC")
        return validate_ohlcv(merged[m].reset_index(drop=True), source=self.name)
