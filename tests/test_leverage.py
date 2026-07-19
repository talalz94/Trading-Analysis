"""Exness-style leverage/margin: lot sizing, free-margin gate, and stop-out liquidation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.engine import BacktestConfig, Signals, run_backtest


def _frame(close):
    n = len(close)
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({"t": t, "open": close, "high": close * 1.0002,
                        "low": close * 0.9998, "close": close})


def _enter_once_hold(n, at=1):
    el = np.zeros(n, bool); el[at] = True      # one long entry, then hold
    xl = np.zeros(n, bool)                      # no rule exit
    return Signals(entry_long=el, exit_long=xl)


def test_stop_out_liquidates_on_crash():
    # flat then a crash; a big leveraged long must be liquidated (margin_call) before the end.
    close = np.concatenate([np.full(50, 100.0), np.linspace(100.0, 80.0, 50)])  # -20%
    df = _frame(close)
    cfg = BacktestConfig(initial_cash=1000, fee_bps=0, slippage_bps=0, exit_enabled=True,
                         sizing_mode="lots", sizing_value=1.0, contract_size=100, leverage=100,
                         margin_enabled=True, stop_out_level=50.0, allow_rule_close=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 1
    tr = res.trades.iloc[0]
    assert tr["close_reason"] == "margin_call"
    assert tr["exit_i"] < len(df) - 1                  # liquidated mid-crash, not forced at end
    # equity at stop-out: 1000 + (px-100)*100 <= 0.5 * (100*100/100=100) -> px ~ 90.5
    assert 88 <= tr["exit_price"] <= 92


def test_free_margin_gate_blocks_oversized():
    df = _frame(np.full(60, 100.0))
    # 1 lot * 100 * price 100 = 10,000 notional; leverage 10 -> needs 1,000 margin; acct only 50.
    cfg = BacktestConfig(initial_cash=50, exit_enabled=True, sizing_mode="lots", sizing_value=1.0,
                         contract_size=100, leverage=10, margin_enabled=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 0                         # can't afford the margin


def test_lots_sizing_quantity():
    df = _frame(np.full(60, 100.0))
    cfg = BacktestConfig(initial_cash=100_000, exit_enabled=True, sizing_mode="lots",
                         sizing_value=0.5, contract_size=100, leverage=100, margin_enabled=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 1
    assert abs(res.trades.iloc[0]["qty"] - 0.5 * 100) < 1e-9   # lots * contract_size = 50 units


def test_leverage_allows_notional_above_cash():
    df = _frame(np.full(60, 100.0))
    # $1,000 account, 1:100 leverage, 2 lots * 100 * 100 = 20,000 notional (20x the cash) is fine.
    cfg = BacktestConfig(initial_cash=1000, exit_enabled=True, sizing_mode="lots", sizing_value=2.0,
                         contract_size=100, leverage=100, margin_enabled=True)
    res = run_backtest(df, _enter_once_hold(len(df)), cfg)
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["qty"] == 200
