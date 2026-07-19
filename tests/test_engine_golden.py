"""
Golden regression tests for the engine — self-contained (no legacy dependency).

The golden numbers were captured from the engine AFTER it was validated at exact parity
against the legacy simulator (per-trade PnL diff 0.0) for these exact scenarios. They lock
the behaviour in going forward: signal-only, SL+rr TP, partial ladders (entry_pct + rr), and
ref_col structure stops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quant import signals as S
from quant.engine import BacktestConfig, Signals, run_backtest
from quant.engine.config import TakeProfit
from quant.indicators import add_emas


def _synth_a(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.001, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.0006, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return add_emas(pd.DataFrame({"t": t, "open": open_, "high": high, "low": low, "close": close}), [20])


def _synth_b(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.0012, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.0007, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.0007, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return add_emas(pd.DataFrame({"t": t, "open": open_, "high": high, "low": low, "close": close}), [20])


def _sig(df):
    return Signals(entry_long=S.cross_up(df, "close", "ema_20"),
                  exit_long=S.cross_down(df, "close", "ema_20"))


def _check(res, n_trades, final_cash):
    assert len(res.trades) == n_trades, f"trades {len(res.trades)} != {n_trades}"
    assert abs(res.stats["final_cash"] - final_cash) < 1e-3, res.stats["final_cash"]
    assert res.trades["exit_time"].notna().all()


def test_golden_signal_only():
    df = _synth_a(3000, 0)
    cfg = BacktestConfig(initial_cash=10_000, fee_bps=10, slippage_bps=1, exit_enabled=False)
    _check(run_backtest(df, _sig(df), cfg), 200, 5187.067737)


def test_golden_sl_and_rr_tp():
    df = _synth_a(3000, 3)
    cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                         sl_mode="entry_pct", sl_value=0.5, tp_mode="rr", tp_value=2.0,
                         sizing_mode="risk_pct_equity", sizing_value=1.0, allow_rule_close=False)
    _check(run_backtest(df, _sig(df), cfg), 49, 9398.038573)


def test_golden_ladder_entry_pct_breakeven():
    df = _synth_b(4000, 2)
    cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                         sl_mode="entry_pct", sl_value=0.6, sizing_mode="risk_pct_equity", sizing_value=1.0,
                         take_profits=(TakeProfit("entry_pct", 0.4, 50, "breakeven"),
                                       TakeProfit("entry_pct", 1.0, 100)),
                         allow_rule_close=False)
    _check(run_backtest(df, _sig(df), cfg), 101, 7926.780413)


def test_golden_ladder_rr():
    df = _synth_b(4000, 4)
    cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                         sl_mode="entry_pct", sl_value=0.6, sizing_mode="risk_pct_equity", sizing_value=1.0,
                         take_profits=(TakeProfit("rr", 1.0, 50), TakeProfit("rr", 2.5, 100)),
                         allow_rule_close=False)
    _check(run_backtest(df, _sig(df), cfg), 60, 8670.953423)


def test_golden_ref_col_stop():
    df = _synth_b(4000, 5)
    df["swing_low"] = df["low"].rolling(30, min_periods=1).min()
    cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                         sl_mode="ref_col", sl_buffer_pct=0.05, sl_fallback_mode="entry_pct",
                         sl_fallback_value=0.8, sl_ref_long_col="swing_low",
                         sizing_mode="risk_pct_equity", sizing_value=1.0,
                         take_profits=(TakeProfit("rr", 2.0, 100),), allow_rule_close=True)
    _check(run_backtest(df, _sig(df), cfg), 266, 6936.231323)
