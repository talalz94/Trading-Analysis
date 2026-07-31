"""Smoke tests: every strategy prepares, builds signals, and backtests without error."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.engine import BacktestConfig
from quant.strategies import (
    EmaCross, EmaRibbon, HeikinAshiTrend, KeyLevelBounce, MacdTrend, MtfTrend,
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
    EmaCross(ema_period=50, entry_mode="full_candle", confirm_n=2, exit_mode="below_ema", exit_ema=20),
    EmaCross(ema_period=50, entry_mode="cross", use_heikin_ashi=True, exit_mode="ha_flip"),
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
        "ema_ribbon", "ema_cross", "rsi_reversal", "macd_trend", "heikin_ashi",
        "supertrend", "mtf_trend", "key_level",
    }


def test_time_filter_reduces_entries(gold_df):
    base = EmaRibbon(fast=20, slow=100, confirm_n=3).backtest(gold_df, _cfg())
    filt = EmaRibbon(fast=20, slow=100, confirm_n=3, hours=(8, 12)).backtest(gold_df, _cfg())
    assert filt.stats["num_trades"] <= base.stats["num_trades"]


@pytest.mark.parametrize("use_ha", [False, True], ids=["real_candles", "heikin_ashi"])
def test_confirm_n_and_color_are_independent(gold_df, use_ha):
    """confirm_n (bars above/below the EMA) and confirm_color (consecutive green/red) are
    independent AND-ed toggles, on either real or Heikin Ashi candles: every entry must satisfy
    BOTH — confirm_n consecutive bars with low > EMA, and confirm_n consecutive green candles."""
    strat = EmaCross(ema_period=50, entry_mode="full_candle", confirm_n=3, confirm_color=True,
                      use_heikin_ashi=use_ha, allow_short=False, one_entry_per_cross=True)
    prep = strat.prepare(gold_df.copy())
    sig = strat.build_signals(prep)
    assert sig.entry_long.sum() > 0, "test setup should produce at least one entry"

    lo_col = "ha_low" if use_ha else "low"
    o_col, c_col = ("ha_open", "ha_close") if use_ha else ("open", "close")
    ema_col = strat._entry_col
    n = strat.confirm_n

    idxs = np.where(sig.entry_long)[0]
    for i in idxs:
        window = prep.iloc[i - n + 1: i + 1]
        assert (window[lo_col] > window[ema_col]).all(), f"bar {i}: not all {n} bars above EMA"
        assert (window[c_col] > window[o_col]).all(), f"bar {i}: not all {n} bars green"
