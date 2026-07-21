"""Data-source interface and the canonical OHLCV contract."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd

# Canonical tidy-OHLCV schema every source must return.
TIME_COL = "open_time"                       # tz-aware UTC
OHLCV_COLS = ["open", "high", "low", "close", "volume"]
BASE_COLS = [TIME_COL, *OHLCV_COLS]


@runtime_checkable
class DataSource(Protocol):
    """A price-history provider (Binance, OANDA, metals API, CSV, ...).

    Implementations must return a DataFrame with at least BASE_COLS, `open_time`
    tz-aware UTC, sorted ascending and de-duplicated on open_time. They should cache
    locally and fetch only missing ranges (incremental).
    """

    name: str

    def fetch(
        self,
        symbol: str,
        interval: str,
        start: str,
        end: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame: ...


def validate_ohlcv(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """Assert the OHLCV contract and return a normalized frame."""
    missing = [c for c in BASE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{source}] OHLCV frame missing columns: {missing}")
    out = df.copy()
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], utc=True)
    out = out.drop_duplicates(TIME_COL, keep="last").sort_values(TIME_COL).reset_index(drop=True)
    return out
