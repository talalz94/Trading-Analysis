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


def _mtf_fixture():
    import numpy as np, pandas as pd
    n = 900
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": idx, "t": idx, "open": np.r_[close[0], close[:-1]],
                         "high": close + 0.3, "low": close - 0.3, "close": close, "volume": 1.0})


def test_mtf_ema_lines_drawn_whether_or_not_the_frame_already_has_them():
    """Regression: when the scenario's `mtf_emas` param already built `{tf}__ema_{p}`, the chart used
    to re-run build_mtf, hit pandas' _x/_y suffixing, and silently draw NOTHING."""
    import numpy as np
    from quant import scenarios as sc
    from quant.viz import scenario_figure
    df = _mtf_fixture()
    base = {"grad_max": None, "prev_low_n": 20, "confirm_red": 2}
    for extra in ({}, {"mtf_emas": ("5min", [20, 50])}):
        run = sc.run_pieces(df, "support_break_short@v1.2", "leg_high@v1", "rr@v1",
                            params={**base, **extra}, cfg={"spread": 0.0})
        fig = scenario_figure(run, mtf_ema=("5min", [20, 50]))
        lines = [t for t in fig.data if t.name and "5min" in t.name]
        assert len(lines) == 2, f"expected 2 HTF EMA lines, got {[t.name for t in lines]}"
        for t in lines:
            v = np.asarray(t.y, dtype=float)
            assert np.isfinite(v).sum() > 100, f"{t.name} is empty/all-NaN"


def test_mtf_ema_missing_column_raises_instead_of_silently_skipping():
    """A column that cannot be built must be loud, not an invisible line."""
    import pytest as _pytest
    from quant.viz.studies import _add_mtf_emas
    import plotly.graph_objects as go
    df = _mtf_fixture()
    fig = go.Figure()
    with _pytest.raises(Exception):
        _add_mtf_emas(fig, df.drop(columns=["open_time"]), df["t"], ("5min", 20), None)
