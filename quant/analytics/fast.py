"""
Array-native fast metrics for parameter sweeps.

Computes ranking metrics directly from the kernel's numpy outputs — no DataFrame
construction, and the bar frequency (periods/year) is computed once by the caller and
reused across all combos. This drops per-combo cost to roughly the kernel time.
"""
from __future__ import annotations

from typing import Dict

import math

import numpy as np
import pandas as pd

from ..engine.kernel import equity_stats


def periods_per_year(t_series: pd.Series) -> float:
    """Infer annualization factor from the median bar spacing (computed once per df)."""
    if t_series is None or len(t_series) < 3:
        return 0.0
    dt = t_series.diff().dropna()
    if dt.empty:
        return 0.0
    med = dt.median().total_seconds()
    return (365.25 * 24 * 3600) / med if med > 0 else 0.0


def fast_stats(
    side: np.ndarray,
    pnl: np.ndarray,
    equity: np.ndarray,
    *,
    initial_cash: float,
    final_cash: float,
    ppy: float,
) -> Dict[str, float]:
    total_return_pct = (final_cash / initial_cash - 1.0) * 100.0 if initial_cash > 0 else 0.0

    # Single compiled pass: drawdown + return moments.
    max_dd_frac, sum_r, sum_r2, sum_dr, sum_dr2, n_r, n_dr = equity_stats(
        np.ascontiguousarray(equity, dtype=np.float64))
    max_dd = float(max_dd_frac * 100.0)

    sharpe = sortino = 0.0
    if n_r > 1 and ppy > 0:
        ann = math.sqrt(ppy)
        mu = sum_r / n_r
        var = (sum_r2 - sum_r * sum_r / n_r) / (n_r - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd > 0:
            sharpe = float(mu / sd * ann)
        if n_dr > 1:
            dvar = (sum_dr2 - sum_dr * sum_dr / n_dr) / (n_dr - 1)
            dsd = math.sqrt(dvar) if dvar > 0 else 0.0
            if dsd > 0:
                sortino = float(mu / dsd * ann)

    n = pnl.size
    if n == 0:
        return {
            "num_trades": 0.0, "win_rate_pct": 0.0, "total_return_pct": total_return_pct,
            "profit_factor": 0.0, "expectancy_per_trade": 0.0, "sharpe": sharpe,
            "sortino": sortino, "max_drawdown_pct": max_dd, "final_cash": float(final_cash),
        }

    win = pnl > 0
    gp = float(pnl[win].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = gp / gl if gl > 0 else (np.inf if gp > 0 else 0.0)
    return {
        "num_trades": float(n),
        "win_rate_pct": float(win.mean() * 100.0),
        "total_return_pct": float(total_return_pct),
        "profit_factor": float(pf),
        "expectancy_per_trade": float(pnl.mean()),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd,
        "long_trades": float((side == 1).sum()),
        "short_trades": float((side == -1).sum()),
        "final_cash": float(final_cash),
    }
