"""
Parity + behavior for the extended exit model: laddered/partial take-profits with stop
movement, and ref_col structure stops — validated against the legacy simulator. Trailing
stops (which legacy lacks) get a behavioral test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant import signals as S
from quant.engine import BacktestConfig, Signals, run_backtest
from quant.engine.config import TakeProfit
from quant.indicators import add_emas

pytest.importorskip("simulation.simulator")
from simulation.simulator import (  # noqa: E402
    Strategy, SimConfig, run_simulation,
    TradeExitConfig, StopLossConfig, PositionSizingConfig, TakeProfitConfig,
)
from simulation.rules import Rule, ALL  # noqa: E402


def _synthetic(n=4000, seed=1):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0, 0.0012, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    high = close * (1.0 + np.abs(rng.normal(0, 0.0007, n)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.0007, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    t = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    df = pd.DataFrame({"t": t, "open": open_, "high": high, "low": low, "close": close})
    df = add_emas(df, [20])
    df["swing_low"] = df["low"].rolling(30, min_periods=1).min()
    df["__open"] = S.cross_up(df, "close", "ema_20")
    df["__close"] = S.cross_down(df, "close", "ema_20")
    return df


def _legacy(df, exit_cfg, fee=8, slip=1.5):
    strat = Strategy(open_rules_long=ALL(Rule("o", lambda c: c.flag("__open"))),
                     close_rules_long=ALL(Rule("c", lambda c: c.flag("__close"))))
    cfg = SimConfig(initial_cash=10_000, max_open_trades=1, fee_bps=fee, slippage_bps=slip,
                    progress=False, progress_bar=False, log_level="ERROR", exit=exit_cfg)
    return run_simulation(df, strat, cfg, time_col="t", price_col="close")


def _sig(df):
    return Signals(entry_long=S.cross_up(df, "close", "ema_20"),
                  exit_long=S.cross_down(df, "close", "ema_20"))


def _pnls(res_or_trades):
    if hasattr(res_or_trades, "trades") and isinstance(res_or_trades.trades, list):
        return np.array(sorted(t.pnl for t in res_or_trades.trades if t.pnl is not None))
    return np.array(sorted(res_or_trades.trades["pnl"].to_numpy()))


def test_parity_partial_ladder_entry_pct_with_stop_move():
    # entry_pct TPs don't depend on the stop, so legacy's per-bar recompute matches our
    # fixed-at-entry levels even with a breakeven stop-move. Validates partial + stop-move.
    df = _synthetic(seed=2)
    legacy_exit = TradeExitConfig(
        enabled=True,
        stop_loss=StopLossConfig(mode="entry_pct", value=0.6),
        sizing=PositionSizingConfig(mode="risk_pct_equity", value=1.0),
        take_profits=(
            TakeProfitConfig(label="TP1", mode="entry_pct", value=0.4, close_pct=50.0, move_stop_mode="breakeven"),
            TakeProfitConfig(label="TP2", mode="entry_pct", value=1.0, close_pct=100.0),
        ),
        intrabar_priority="stop_first", allow_rule_close=False,
    )
    lres = _legacy(df, legacy_exit)

    ncfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
        sl_mode="entry_pct", sl_value=0.6,
        sizing_mode="risk_pct_equity", sizing_value=1.0,
        take_profits=(
            TakeProfit(mode="entry_pct", value=0.4, close_pct=50.0, move_stop_mode="breakeven"),
            TakeProfit(mode="entry_pct", value=1.0, close_pct=100.0),
        ),
        allow_rule_close=False, intrabar_priority="stop_first",
    )
    nres = run_backtest(df, _sig(df), ncfg)

    lp, npnl = _pnls(lres), _pnls(nres)
    assert len(lp) == len(npnl), f"{len(lp)} vs {len(npnl)}"
    assert np.allclose(lp, npnl, atol=1e-4)
    assert abs(lres.stats["final_cash"] - nres.stats["final_cash"]) < 1e-3


def test_parity_partial_ladder_rr_no_stop_move():
    # rr TPs with a CONSTANT stop (no stop-move) -> legacy's per-bar rr recompute equals
    # our fixed-at-entry rr levels. Validates rr math + partial closes.
    df = _synthetic(seed=4)
    legacy_exit = TradeExitConfig(
        enabled=True,
        stop_loss=StopLossConfig(mode="entry_pct", value=0.6),
        sizing=PositionSizingConfig(mode="risk_pct_equity", value=1.0),
        take_profits=(
            TakeProfitConfig(label="TP1", mode="rr", value=1.0, close_pct=50.0),
            TakeProfitConfig(label="TP2", mode="rr", value=2.5, close_pct=100.0),
        ),
        intrabar_priority="stop_first", allow_rule_close=False,
    )
    lres = _legacy(df, legacy_exit)

    ncfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
        sl_mode="entry_pct", sl_value=0.6,
        sizing_mode="risk_pct_equity", sizing_value=1.0,
        take_profits=(
            TakeProfit(mode="rr", value=1.0, close_pct=50.0),
            TakeProfit(mode="rr", value=2.5, close_pct=100.0),
        ),
        allow_rule_close=False, intrabar_priority="stop_first",
    )
    nres = run_backtest(df, _sig(df), ncfg)

    lp, npnl = _pnls(lres), _pnls(nres)
    assert len(lp) == len(npnl), f"{len(lp)} vs {len(npnl)}"
    assert np.allclose(lp, npnl, atol=1e-4)
    assert abs(lres.stats["final_cash"] - nres.stats["final_cash"]) < 1e-3


def test_parity_ref_col_structure_stop():
    df = _synthetic(seed=5)
    legacy_exit = TradeExitConfig(
        enabled=True,
        stop_loss=StopLossConfig(mode="ref_col", ref_col="swing_low", buffer_pct=0.05,
                                 max_ref_risk_pct=None, fallback_mode="entry_pct", fallback_value=0.8),
        sizing=PositionSizingConfig(mode="risk_pct_equity", value=1.0),
        take_profits=(TakeProfitConfig(label="TP1", mode="rr", value=2.0, close_pct=100.0),),
        intrabar_priority="stop_first", allow_rule_close=True,
    )
    lres = _legacy(df, legacy_exit)

    ncfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
        sl_mode="ref_col", sl_value=0.0, sl_buffer_pct=0.05, sl_max_ref_risk_pct=0.0,
        sl_fallback_mode="entry_pct", sl_fallback_value=0.8, sl_ref_long_col="swing_low",
        sizing_mode="risk_pct_equity", sizing_value=1.0,
        take_profits=(TakeProfit(mode="rr", value=2.0, close_pct=100.0),),
        allow_rule_close=True, intrabar_priority="stop_first",
    )
    nres = run_backtest(df, _sig(df), ncfg)

    lp, npnl = _pnls(lres), _pnls(nres)
    assert len(lp) == len(npnl), f"{len(lp)} vs {len(npnl)}"
    assert np.allclose(lp, npnl, atol=1e-4)


def test_trailing_stop_behaves():
    df = _synthetic(seed=7)
    cfg = BacktestConfig(
        initial_cash=10_000, fee_bps=5, slippage_bps=1.0, exit_enabled=True,
        sl_mode="entry_pct", sl_value=1.0, trail_mode="pct", trail_value=0.5,
        sizing_mode="risk_pct_equity", sizing_value=1.0, allow_rule_close=False,
    )
    res = run_backtest(df, _sig(df), cfg)
    # Trades close, and with a tight trailing stop most exits are stops (not forced-at-end).
    assert len(res.trades) > 0
    reasons = set(res.trades["close_reason"])
    assert "stop_loss" in reasons
    assert res.trades["exit_time"].notna().all()
