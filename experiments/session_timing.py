"""
Experiment: what time of day / session does this strategy work best?

Idea: "When is this strategy most profitable?" Holds the strategy fixed and sweeps the trading
session (and, optionally, hour windows), ranking each by total PnL / Sharpe. Uses the universal
time filter every Strategy inherits (`session` / `hours`).

Run:  python -m experiments.session_timing
"""
from __future__ import annotations

from quant.engine import BacktestConfig
from quant.strategies import EmaRibbon

from .base import Experiment

DESCRIPTION = (
    "Hold an EMA-ribbon strategy fixed and find the market session in which it performs best. "
    "Sessions: none (24h), london, newyork, tokyo, sydney. Objective: maximize total return."
)


def build() -> Experiment:
    base_cfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
        sl_mode="entry_pct", sl_value=0.6, tp_mode="rr", tp_value=2.0,
        sizing_mode="risk_pct_equity", sizing_value=1.0,
    )
    return Experiment(
        name="session_timing",
        description=DESCRIPTION,
        strategy_cls=EmaRibbon,
        base_cfg=base_cfg,
        symbol="PAXGUSDT", tf="1m", start="2025-06-01", end="2026-05-31",
        strategy_space={
            "fast": [50],
            "slow": [200],
            "confirm_n": [5],
            "session": [None, "london", "newyork", "tokyo", "sydney"],
        },
        metric="total_return_pct", direction="max", min_trades=10,
    )


if __name__ == "__main__":
    build().run()
