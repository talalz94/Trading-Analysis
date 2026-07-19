"""Simulation engine: numba position/PnL kernel + Python wrapper."""
from __future__ import annotations

from .config import BacktestConfig, Signals, TakeProfit
from .run import SimResult, run_backtest

__all__ = ["BacktestConfig", "Signals", "TakeProfit", "SimResult", "run_backtest"]
