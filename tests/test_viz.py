"""Viz smoke tests: responsive chart constructs and (when available) actually resamples."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.viz import ResearchChart, equity_chart, price_chart, _HAVE_RESAMPLER


def _df(n=20000):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"t": t, "open_time": t, "open": close, "high": close * 1.001,
                        "low": close * 0.999, "close": close, "volume": rng.random(n)})


def test_research_chart_builds_and_resamples():
    df = _df()
    ch = ResearchChart(df, candles=True)
    ch.add_ema(50)
    fig = ch.show()
    assert len(fig.data) >= 2  # close line + ema + candles
    if _HAVE_RESAMPLER:
        # line traces should be decimated to far fewer than n points
        line_lens = [len(t.x) for t in fig.data if t.type in ("scattergl", "scatter") and t.x is not None]
        assert line_lens and max(line_lens) < len(df)


def test_price_chart_with_trades():
    df = _df(5000)
    trades = pd.DataFrame({
        "side": ["long", "long"], "entry_time": [df["t"].iloc[100], df["t"].iloc[200]],
        "exit_time": [df["t"].iloc[150], df["t"].iloc[250]],
        "entry_price": [df["close"].iloc[100], df["close"].iloc[200]],
        "exit_price": [df["close"].iloc[150], df["close"].iloc[250]],
        "pnl": [5.0, -3.0],
    })
    fig = price_chart(df, emas=[20, 50], trades=trades, candles=True)
    assert len(fig.data) >= 3


def test_equity_chart_builds():
    df = _df(8000)
    eq = pd.DataFrame({"t": df["t"], "equity": np.linspace(10000, 11000, len(df)),
                      "drawdown": np.zeros(len(df))})
    fig = equity_chart(eq)
    assert len(fig.data) >= 1
