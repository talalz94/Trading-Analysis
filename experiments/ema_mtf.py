"""
Experiment: best multi-timeframe EMA combination.

Idea: "Which EMA periods across timeframes give the most profitable trend strategy?"
Searches the 1m entry EMA + the 5m trend EMduo + the 15m momentum EMA of `MtfTrend`
(1m entry trigger, 5m trend filter, 15m momentum filter) and ranks by Sharpe.

Run:  python -m experiments.ema_mtf
"""
from __future__ import annotations

from quant.engine import BacktestConfig
from quant.strategies import MtfTrend

from .base import Experiment

DESCRIPTION = (
    "Find the most profitable multi-timeframe EMA combination for a trend strategy that enters "
    "on a 1-minute EMA cross, confirmed by a 5-minute EMA trend and 15-minute momentum. "
    "Objective: maximize Sharpe (risk-based sizing, RR take-profit)."
)


def build() -> Experiment:
    base_cfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
        sl_mode="entry_pct", sl_value=0.6, tp_mode="rr", tp_value=2.0,
        sizing_mode="risk_pct_equity", sizing_value=1.0,
    )
    return Experiment(
        name="ema_mtf",
        description=DESCRIPTION,
        strategy_cls=MtfTrend,
        base_cfg=base_cfg,
        symbol="PAXGUSDT", tf="1m", start="2025-06-01", end="2026-05-31",
        strategy_space={
            "fast_1m": [20, 50, 100],
            "trend_5m_fast": [20, 50],
            "trend_5m_slow": [100, 200],
            "mom_15m": [50, 100],
        },
        metric="sharpe", direction="max", min_trades=20,
        valid_fn=lambda p: p["trend_5m_fast"] < p["trend_5m_slow"],
    )


if __name__ == "__main__":
    build().run()
