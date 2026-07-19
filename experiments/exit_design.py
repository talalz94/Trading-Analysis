"""
Experiment: best stop-loss type/value and take-profit.

Idea: "What exit maximizes risk-adjusted return for this entry?" Holds the entry strategy fixed
and sweeps EXECUTION-CONFIG fields (stop-loss %, take-profit R-multiple) — signals are computed
once and only the numba kernel re-runs per exit config, so this is very fast.

Run:  python -m experiments.exit_design
"""
from __future__ import annotations

from quant.engine import BacktestConfig
from quant.strategies import EmaRibbon

from .base import Experiment

DESCRIPTION = (
    "Hold an EMA-ribbon entry fixed and find the stop-loss (% from entry) and take-profit "
    "(R multiple) that maximize Sharpe. Demonstrates sweeping execution-config fields (not just "
    "strategy params). Extendable to sl_mode='ref_col' structure stops or trailing."
)


def build() -> Experiment:
    base_cfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
        sl_mode="entry_pct", sizing_mode="risk_pct_equity", sizing_value=1.0,
        allow_rule_close=False,   # judge the EXIT model, not the rule exit
    )
    return Experiment(
        name="exit_design",
        description=DESCRIPTION,
        strategy_cls=EmaRibbon,
        base_cfg=base_cfg,
        symbol="PAXGUSDT", tf="1m", start="2025-06-01", end="2026-05-31",
        strategy_space={"fast": [50], "slow": [200], "confirm_n": [5]},
        cfg_space={
            "sl_value": [0.3, 0.5, 0.75, 1.0, 1.5],
            "tp_value": [1.0, 1.5, 2.0, 3.0],
            "tp_mode": ["rr"],
        },
        metric="sharpe", direction="max", min_trades=20,
    )


if __name__ == "__main__":
    build().run()
