"""
Performance metrics.

`compute_stats` takes the trades + equity-curve frames produced by the engine and returns
a flat dict of research metrics: returns, win/loss, profit factor, expectancy, drawdown,
recovery, and risk-adjusted ratios (Sharpe / Sortino / Calmar) computed from the per-bar
equity curve and annualized from the inferred bar frequency.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _periods_per_year(t: pd.Series) -> float:
    if t is None or len(t) < 3:
        return 0.0
    dt = pd.to_datetime(pd.Series(t)).diff().dropna()
    if dt.empty:
        return 0.0
    med = dt.median().total_seconds()
    if med <= 0:
        return 0.0
    return (365.25 * 24 * 3600) / med


def _max_consecutive(mask: np.ndarray) -> int:
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return int(best)


def compute_stats(
    trades: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    initial_cash: float,
    final_cash: float,
) -> Dict[str, float]:
    max_dd = float(equity_curve["drawdown"].max() * 100.0) if not equity_curve.empty else 0.0
    total_return_pct = float((final_cash / initial_cash - 1.0) * 100.0) if initial_cash > 0 else 0.0

    # Risk-adjusted ratios from the equity curve.
    sharpe = sortino = calmar = 0.0
    if not equity_curve.empty and "equity" in equity_curve:
        eq = equity_curve["equity"].to_numpy(np.float64)
        ret = np.diff(eq) / np.where(eq[:-1] == 0, np.nan, eq[:-1])
        ret = ret[np.isfinite(ret)]
        if ret.size > 1:
            ppy = _periods_per_year(equity_curve.get("t"))
            ann = np.sqrt(ppy) if ppy > 0 else 0.0
            mu = ret.mean()
            sd = ret.std(ddof=1)
            if sd > 0:
                sharpe = float(mu / sd * ann)
            downside = ret[ret < 0]
            dsd = downside.std(ddof=1) if downside.size > 1 else 0.0
            if dsd > 0:
                sortino = float(mu / dsd * ann)
            ann_return = mu * ppy if ppy > 0 else 0.0
            if max_dd > 0:
                calmar = float(ann_return / (max_dd / 100.0))

    base = {
        "initial_cash": float(initial_cash),
        "final_cash": float(final_cash),
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_dd,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }

    closed = trades[trades["pnl"].notna()] if not trades.empty else trades
    n = len(closed)
    if n == 0:
        base.update({
            "total_pnl": 0.0, "num_trades": 0.0, "num_winners": 0.0, "num_losers": 0.0,
            "win_rate_pct": 0.0, "loss_rate_pct": 0.0, "profit_factor": 0.0,
            "expectancy_per_trade": 0.0, "total_fees": 0.0,
        })
        return base

    pnl = closed["pnl"].to_numpy(np.float64)
    ret = closed["return_pct"].to_numpy(np.float64)
    bars = closed["bars_held"].to_numpy(np.float64)
    fees = closed["fees"].to_numpy(np.float64)

    win = pnl > 0
    loss = pnl < 0
    gross_profit = float(pnl[win].sum()) if win.any() else 0.0
    gross_loss = float(-pnl[loss].sum()) if loss.any() else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
    total_pnl = float(pnl.sum())
    recovery = (total_pnl / abs(max_dd / 100.0 * initial_cash)) if max_dd > 0 else (np.inf if total_pnl > 0 else 0.0)

    def _side(mask, prefix):
        p = pnl[mask]
        if p.size == 0:
            return {f"{prefix}_trades": 0.0, f"{prefix}_pnl": 0.0, f"{prefix}_win_rate_pct": 0.0}
        return {
            f"{prefix}_trades": float(p.size),
            f"{prefix}_pnl": float(p.sum()),
            f"{prefix}_win_rate_pct": float((p > 0).mean() * 100.0),
        }

    sides = closed["side"].to_numpy()
    base.update({
        "total_pnl": total_pnl,
        "num_trades": float(n),
        "num_winners": float(win.sum()),
        "num_losers": float(loss.sum()),
        "num_breakeven": float((pnl == 0).sum()),
        "win_rate_pct": float(win.mean() * 100.0),
        "loss_rate_pct": float(loss.mean() * 100.0),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(profit_factor),
        "avg_pnl": float(pnl.mean()),
        "median_pnl": float(np.median(pnl)),
        "avg_winner": float(pnl[win].mean()) if win.any() else 0.0,
        "avg_loser": float(pnl[loss].mean()) if loss.any() else 0.0,
        "largest_winner": float(pnl[win].max()) if win.any() else 0.0,
        "largest_loser": float(pnl[loss].min()) if loss.any() else 0.0,
        "expectancy_per_trade": float(pnl.mean()),
        "expectancy_pct_initial_cash": float(pnl.mean() / initial_cash * 100.0) if initial_cash > 0 else 0.0,
        "avg_return_pct": float(np.nanmean(ret)) if ret.size else 0.0,
        "avg_bars_held": float(np.nanmean(bars)) if bars.size else 0.0,
        "max_consecutive_wins": float(_max_consecutive(win)),
        "max_consecutive_losses": float(_max_consecutive(loss)),
        "recovery_factor": float(recovery),
        "total_fees": float(fees.sum()),
        "avg_fee_per_trade": float(fees.mean()),
    })
    base.update(_side(sides == "long", "long"))
    base.update(_side(sides == "short", "short"))
    return base
