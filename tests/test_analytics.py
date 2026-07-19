"""Tests for analytics attribution + reporting."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import analytics as A
from quant import reporting as R
from quant.engine import BacktestConfig
from quant.strategies import EmaRibbon


def _df(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"t": t, "open_time": t, "open": close, "high": close * 1.001,
                        "low": close * 0.999, "close": close, "volume": rng.random(n)})


def _res():
    df = _df()
    cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                         sl_mode="entry_pct", sl_value=0.6, tp_mode="rr", tp_value=2.0,
                         sizing_mode="risk_pct_equity", sizing_value=1.0)
    return df, EmaRibbon(fast=20, slow=100, confirm_n=3).backtest(df, cfg)


def test_attribution_tables():
    df, res = _res()
    for fn in (A.by_hour, A.by_weekday, A.by_month, A.by_session):
        t = fn(res.trades)
        assert set(["n_trades", "win_rate_pct", "total_pnl"]).issubset(t.columns)
        if not res.trades.empty:
            assert t["n_trades"].sum() >= len(res.trades) or fn is A.by_session
    reg = A.by_regime(res.trades, df)
    assert "regime" in reg.columns


def test_monthly_returns():
    df, res = _res()
    mr = A.monthly_returns(res.equity_curve)
    assert not mr.empty


def test_summary_and_html(tmp_path):
    df, res = _res()
    s = R.summary(res, df=df)
    assert "headline" in s and "by_hour" in s
    out = R.to_html(res, str(tmp_path / "r.html"), df=df, price_df=df)
    assert (tmp_path / "r.html").exists()
    assert (tmp_path / "r.html").stat().st_size > 1000
