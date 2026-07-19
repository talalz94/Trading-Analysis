"""
Numerical parity: the new numba engine must reproduce the legacy simulator exactly
(for the supported feature subset) on a synthetic, hermetic price series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import signals as S
from quant.engine import BacktestConfig, Signals, run_backtest
from quant.indicators import add_emas

legacy = pytest.importorskip("simulation.simulator")
from simulation.simulator import (  # noqa: E402
    Strategy, SimConfig, run_simulation,
    TradeExitConfig, StopLossConfig, PositionSizingConfig, TakeProfitConfig,
)
from simulation.rules import Rule, ALL  # noqa: E402


def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.001, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0006, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0006, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({"t": t, "open": open_, "high": high, "low": low, "close": close})
    df = add_emas(df, [20, 50])
    df["__open"] = S.cross_up(df, "close", "ema_20")
    df["__close"] = S.cross_down(df, "close", "ema_20")
    return df


def _legacy(df, lcfg):
    strat = Strategy(
        open_rules_long=ALL(Rule("open", lambda c: c.flag("__open"))),
        close_rules_long=ALL(Rule("close", lambda c: c.flag("__close"))),
    )
    return run_simulation(df, strat, lcfg, time_col="t", price_col="close")


def _sig(df):
    return Signals(entry_long=S.cross_up(df, "close", "ema_20"),
                  exit_long=S.cross_down(df, "close", "ema_20"))


def _assert_parity(lres, nres, atol=1e-6):
    lpnls = np.array(sorted(t.pnl for t in lres.trades if t.pnl is not None))
    npnls = np.array(sorted(nres.trades["pnl"].to_numpy()))
    assert len(lpnls) == len(npnls), f"trade count {len(lpnls)} vs {len(npnls)}"
    if len(lpnls):
        assert np.allclose(lpnls, npnls, atol=atol)
    assert abs(lres.stats["final_cash"] - nres.stats["final_cash"]) < 1e-4
    assert abs(lres.stats["max_drawdown_pct"] - nres.stats["max_drawdown_pct"]) < 1e-4


def test_parity_signal_only():
    df = _synthetic()
    lcfg = SimConfig(initial_cash=10_000, max_open_trades=1, fee_bps=10, slippage_bps=1,
                     progress=False, progress_bar=False, log_level="ERROR")
    ncfg = BacktestConfig(initial_cash=10_000, max_open_trades=1, fee_bps=10, slippage_bps=1,
                          exit_enabled=False)
    _assert_parity(_legacy(df, lcfg), run_backtest(df, _sig(df), ncfg))


def test_parity_exit_engine():
    df = _synthetic(seed=3)
    exit_cfg = TradeExitConfig(
        enabled=True,
        stop_loss=StopLossConfig(mode="entry_pct", value=0.5),
        sizing=PositionSizingConfig(mode="risk_pct_equity", value=1.0),
        take_profits=(TakeProfitConfig(label="TP1", mode="rr", value=2.0, close_pct=100.0),),
        intrabar_priority="stop_first", allow_rule_close=False,
    )
    lcfg = SimConfig(initial_cash=10_000, max_open_trades=1, fee_bps=8, slippage_bps=1.5,
                     progress=False, progress_bar=False, log_level="ERROR", exit=exit_cfg)
    ncfg = BacktestConfig(initial_cash=10_000, max_open_trades=1, fee_bps=8, slippage_bps=1.5,
                          exit_enabled=True, sl_mode="entry_pct", sl_value=0.5,
                          tp_mode="rr", tp_value=2.0, sizing_mode="risk_pct_equity",
                          sizing_value=1.0, allow_rule_close=False)
    _assert_parity(_legacy(df, lcfg), run_backtest(df, _sig(df), ncfg), atol=1e-4)
