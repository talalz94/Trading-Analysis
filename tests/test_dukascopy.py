"""Dukascopy provider: symbol mapping, normalization, and incremental caching (mocked fetch)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

dk = pytest.importorskip("dukascopy_python")

from quant.data.binance import get_source            # noqa: E402
from quant.data.dukascopy import DukascopySource, _instrument, _normalize  # noqa: E402


def _fake_bars(start, end):
    idx = pd.to_datetime(pd.date_range(start, end, freq="1min"), utc=True)
    base = 2000.0 + np.arange(len(idx)) * 0.01
    return pd.DataFrame({"open": base, "high": base + 1, "low": base - 1,
                        "close": base, "volume": 1.0}, index=idx)


def test_instrument_map():
    assert _instrument("XAUUSD") == "XAU/USD"
    assert _instrument("EURUSD") == "EUR/USD"
    assert _instrument("xau/usd") == "XAU/USD"


def test_registered():
    src = get_source("dukascopy")
    assert src.name == "dukascopy"


def test_normalize_contract():
    out = _normalize(_fake_bars("2024-01-01", "2024-01-01 00:10"))
    assert list(out.columns[:1]) == ["open_time"]
    assert {"open", "high", "low", "close", "volume"}.issubset(out.columns)
    assert out["open_time"].dt.tz is not None


def test_fetch_incremental(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(instrument, interval, offer, start, end, **kw):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))   # start/end already tz-aware
        return _fake_bars(start, end)

    monkeypatch.setattr(dk, "fetch", fake_fetch)
    src = DukascopySource(cache_dir=str(tmp_path), market="cfd")

    df1 = src.fetch("XAUUSD", "1m", start="2024-01-01 00:00", end="2024-01-01 01:00")
    assert len(df1) > 0
    assert {"open_time", "open", "high", "low", "close", "volume"}.issubset(df1.columns)
    n1 = len(calls)

    # extend the window: only the new tail should be fetched (incremental)
    df2 = src.fetch("XAUUSD", "1m", start="2024-01-01 00:00", end="2024-01-01 02:00")
    assert len(df2) > len(df1)
    tail_calls = calls[n1:]
    assert tail_calls, "expected an incremental fetch for the extended tail"
    # the incremental fetch must start at/after the previous end (not refetch from scratch)
    assert all(s >= pd.Timestamp("2024-01-01 00:59", tz="UTC") for s, _ in tail_calls)
