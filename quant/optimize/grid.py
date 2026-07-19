"""
Parameter sweep runner.

Given a strategy class and a parameter grid, run a backtest for every combination and
return a tidy results DataFrame (one row per combo, with stats). Designed for the
vectorized-signals + numba-engine model where each backtest is ~milliseconds.

Efficiency:
  - Indicator columns are precomputed ONCE into a shared frame (each unique column once),
    so combos differing only in non-indicator params don't recompute indicators.
  - Optional bounded parallelism via joblib (threading backend shares the frame in memory;
    the numba kernel releases the GIL). `n_jobs` defaults to (cpu_count - 2) so the machine
    stays usable during large sweeps.
"""
from __future__ import annotations

import itertools
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from ..analytics.fast import fast_stats, periods_per_year
from ..engine import BacktestConfig
from ..engine.run import invoke_kernel
from ..logging_utils import get_logger, maybe_tqdm
from ..strategies.base import Strategy

_log = get_logger("quant.optimize")


def expand_grid(grid: Dict[str, Sequence]) -> List[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def default_n_jobs() -> int:
    return max(1, (os.cpu_count() or 2) - 2)


def _precompute_columns(base: pd.DataFrame, strategy_cls: Type[Strategy],
                        combos: List[dict]) -> pd.DataFrame:
    """Materialize the union of indicator columns all combos need, each computed once."""
    prepared = base.copy()
    seen = set(prepared.columns)
    for combo in combos:
        pdf = strategy_cls(**combo).prepare(base)
        new = [c for c in pdf.columns if c not in seen]
        for c in new:
            prepared[c] = pdf[c].to_numpy()
            seen.add(c)
    return prepared


def run_grid(
    base_df: pd.DataFrame,
    strategy_cls: Type[Strategy],
    grid: Dict[str, Sequence],
    cfg: BacktestConfig,
    *,
    valid_fn: Optional[Callable[[dict], bool]] = None,
    keep_stats: Optional[Sequence[str]] = None,
    n_jobs: Optional[int] = None,
    time_col: str = "t",
    price_col: str = "close",
    progress: bool = True,
) -> pd.DataFrame:
    combos = expand_grid(grid)
    if valid_fn is not None:
        combos = [c for c in combos if valid_fn(c)]
    if not combos:
        raise ValueError("Empty parameter grid after filtering.")

    prepared = _precompute_columns(base_df, strategy_cls, combos)
    n_jobs = default_n_jobs() if n_jobs is None else max(1, n_jobs)

    # Precompute the invariant inputs ONCE (shared across all combos).
    n = len(prepared)
    close = prepared[price_col].to_numpy(np.float64)
    high = prepared["high"].to_numpy(np.float64) if "high" in prepared else close
    low = prepared["low"].to_numpy(np.float64) if "low" in prepared else close
    open_ = prepared["open"].to_numpy(np.float64) if "open" in prepared else close
    ppy = periods_per_year(prepared[time_col])
    ic = float(cfg.initial_cash)

    _log.info("sweep: %d combos | n_jobs=%d | bars=%d", len(combos), n_jobs, n)

    def _one(combo: dict) -> dict:
        # Fast path: kernel + array-native stats, no per-combo DataFrame construction.
        signals = strategy_cls(**combo).build_signals(prepared)
        el, xl, es, xs = signals.as_u8(n)
        out = invoke_kernel(open_, high, low, close, el, xl, es, xs, cfg, df=prepared)
        stats = fast_stats(out[0], out[9], out[11],
                          initial_cash=ic, final_cash=out[13], ppy=ppy)
        if keep_stats is not None:
            stats = {k: stats.get(k) for k in keep_stats}
        row = dict(combo)
        row.update(stats)
        return row

    t0 = time.perf_counter()
    rows: List[dict]
    if n_jobs == 1:
        bar = maybe_tqdm(progress, total=len(combos), desc="sweep", unit="combo")
        rows = []
        for combo in combos:
            rows.append(_one(combo))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs, backend="threading", batch_size="auto")(
            delayed(_one)(combo) for combo in combos
        )
    elapsed = time.perf_counter() - t0

    df = pd.DataFrame(rows)
    _log.info("sweep done | %d combos in %.2fs (%.1f combos/s)",
              len(combos), elapsed, len(combos) / max(elapsed, 1e-9))
    return df
