"""
Bayesian parameter search via Optuna (optional dependency).

Grid search is exhaustive but wasteful for large spaces; Optuna samples intelligently. Install
with `pip install optuna`. Each param spec is (low, high) for int/float ranges or a list for
categoricals.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import pandas as pd

from ..engine import BacktestConfig
from ..logging_utils import get_logger
from ..strategies.base import Strategy

_log = get_logger("quant.optimize.search")


def _suggest(trial, name, spec):
    if isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
        lo, hi = spec
        if isinstance(lo, int) and isinstance(hi, int):
            return trial.suggest_int(name, lo, hi)
        return trial.suggest_float(name, float(lo), float(hi))
    return trial.suggest_categorical(name, list(spec))


def optuna_search(
    base_df: pd.DataFrame,
    strategy_cls,
    space: Dict[str, Sequence],
    cfg: BacktestConfig,
    *,
    metric: str = "sharpe",
    direction: str = "maximize",
    n_trials: int = 100,
    valid_fn: Optional[Callable[[dict], bool]] = None,
    time_col: str = "t",
    price_col: str = "close",
    show_progress_bar: bool = True,
):
    """Run an Optuna study and return it (`study.best_params`, `study.best_value`, `study.trials`)."""
    try:
        import optuna
    except Exception as e:  # pragma: no cover
        raise ImportError("optuna_search requires optuna. Install with: pip install optuna") from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {k: _suggest(trial, k, spec) for k, spec in space.items()}
        if valid_fn is not None and not valid_fn(params):
            raise optuna.TrialPruned()
        res = strategy_cls(**params).backtest(base_df, cfg, time_col=time_col, price_col=price_col)
        val = res.stats.get(metric)
        if val is None:
            raise optuna.TrialPruned()
        return float(val)

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    _log.info("optuna: best %s=%.4f params=%s", metric, study.best_value, study.best_params)
    return study
