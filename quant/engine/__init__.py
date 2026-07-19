"""Simulation engine: numba position/PnL kernel + Python wrapper."""
from __future__ import annotations

from .config import BacktestConfig, Signals
from .run import SimResult, run_backtest

__all__ = ["BacktestConfig", "Signals", "SimResult", "run_backtest"]
