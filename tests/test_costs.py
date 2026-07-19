"""Exness-style cost model: spread (bid/ask), volume-dependent spread, per-lot commission."""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant.engine import BacktestConfig, Signals, run_backtest


def _flat(n=20, price=100.0):
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    c = np.full(n, price)
    return pd.DataFrame({"t": t, "open": c, "high": c, "low": c, "close": c})


def _enter_then_exit(n, enter=1, exit_=2):
    el = np.zeros(n, bool); el[enter] = True
    xl = np.zeros(n, bool); xl[exit_] = True
    return Signals(entry_long=el, exit_long=xl)


def test_spread_round_trip_cost_exact():
    # constant price, 1 unit, spread 0.5 -> buy at 100.25, sell at 99.75 -> pnl = -0.5
    df = _flat()
    cfg = BacktestConfig(initial_cash=10_000, fee_bps=0, slippage_bps=0, spread=0.5,
                         exit_enabled=True, sizing_mode="lots", sizing_value=1.0, contract_size=1,
                         leverage=100, margin_enabled=True, allow_rule_close=True)
    res = run_backtest(df, _enter_then_exit(len(df)), cfg)
    assert len(res.trades) == 1
    assert abs(res.trades.iloc[0]["pnl"] - (-0.5)) < 1e-9


def test_commission_per_lot_exact():
    # 2 lots * $3.5/lot/side * 2 sides = $14 commission, no spread/fees, flat price.
    df = _flat()
    cfg = BacktestConfig(initial_cash=100_000, fee_bps=0, slippage_bps=0, spread=0.0,
                         commission_per_lot=3.5, exit_enabled=True, sizing_mode="lots",
                         sizing_value=2.0, contract_size=100, leverage=100, margin_enabled=True,
                         allow_rule_close=True)
    res = run_backtest(df, _enter_then_exit(len(df)), cfg)
    assert abs(res.trades.iloc[0]["pnl"] - (-14.0)) < 1e-9


def test_spread_cost_scales_with_lots():
    # EUR/USD-style: spread 1.2 pips = 0.00012 price; cost = spread * units.
    # 1 std lot (100k units) -> $12 ; 0.1 lot (10k units) -> $1.20 (constant spread width).
    df = _flat(price=1.10)  # price level doesn't affect the spread cost
    def cost(lots):
        cfg = BacktestConfig(initial_cash=1_000_000, fee_bps=0, slippage_bps=0, spread=0.00012,
                             exit_enabled=True, sizing_mode="lots", sizing_value=lots,
                             contract_size=100_000, leverage=100, margin_enabled=True,
                             allow_rule_close=True)
        return run_backtest(df, _enter_then_exit(len(df)), cfg).trades.iloc[0]["pnl"]
    assert abs(cost(1.0) - (-12.0)) < 1e-6
    assert abs(cost(0.1) - (-1.20)) < 1e-6


def test_zero_costs_no_effect():
    # spread/commission = 0 must equal the plain fee/slippage path (golden-consistent).
    df = _flat()
    a = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                       sizing_mode="cash", cash_per_trade=1000)
    b = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, spread=0.0,
                       commission_per_lot=0.0, exit_enabled=True, sizing_mode="cash", cash_per_trade=1000)
    ra = run_backtest(df, _enter_then_exit(len(df)), a)
    rb = run_backtest(df, _enter_then_exit(len(df)), b)
    assert abs(ra.stats["final_cash"] - rb.stats["final_cash"]) < 1e-9
