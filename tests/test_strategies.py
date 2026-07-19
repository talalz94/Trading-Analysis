"""Smoke tests: every strategy prepares, builds signals, and backtests without error."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.engine import BacktestConfig
from quant.strategies import (
    EmaRibbon, HeikinAshiTrend, KeyLevelBounce, MacdTrend, MtfTrend,
    RsiReversal, SupertrendFlip, REGISTRY,
)


@pytest.fixture(scope="module")
def gold_df():
    rng = np.random.default_rng(0)
    n = 6000
    ret = rng.normal(0, 0.001, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    ot = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": ot, "t": ot, "open": open_, "high": high,
                        "low": low, "close": close, "volume": rng.random(n)})


def _cfg():
    return BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                          sl_mode="entry_pct", sl_value=0.6, tp_mode="rr", tp_value=2.0,
                          sizing_mode="risk_pct_equity", sizing_value=1.0)


STRATS = [
    EmaRibbon(fast=20, slow=100, confirm_n=3),
    RsiReversal(period=14, oversold=30, long_consec=3),
    MacdTrend(),
    HeikinAshiTrend(n_consec=3),
    SupertrendFlip(period=10, multiplier=3.0),
    MtfTrend(fast_1m=20, trend_5m_fast=20, trend_5m_slow=50, mom_15m=20),
    KeyLevelBounce(left=8, right=8, near_pct=0.2),
]


@pytest.mark.parametrize("strat", STRATS, ids=[s.name for s in STRATS])
def test_strategy_runs(gold_df, strat):
    res = strat.backtest(gold_df, _cfg())
    assert res.equity_curve.shape[0] == len(gold_df)
    assert set(["total_return_pct", "sharpe", "max_drawdown_pct"]).issubset(res.stats)
    if not res.trades.empty:
        assert res.trades["exit_time"].notna().all()


def test_registry_complete():
    assert set(REGISTRY) == {
        "ema_ribbon", "rsi_reversal", "macd_trend", "heikin_ashi",
        "supertrend", "mtf_trend", "key_level",
    }


def test_time_filter_reduces_entries(gold_df):
    base = EmaRibbon(fast=20, slow=100, confirm_n=3).backtest(gold_df, _cfg())
    filt = EmaRibbon(fast=20, slow=100, confirm_n=3, hours=(8, 12)).backtest(gold_df, _cfg())
    assert filt.stats["num_trades"] <= base.stats["num_trades"]
