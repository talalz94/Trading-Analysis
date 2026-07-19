"""Python wrapper around the numba kernel: DataFrame + Signals -> SimResult."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .config import BacktestConfig, Signals
from .kernel import run_kernel

_log = get_logger("quant.engine")

REASON_NAMES = {0: "signal", 1: "stop_loss", 2: "take_profit", 3: "forced_close_end", 4: "margin_call"}


def _ref_arr(df, col, n):
    if col is not None and df is not None and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
    return np.full(n, np.nan, dtype=np.float64)


def invoke_kernel(open_, high, low, close, el, xl, es, xs, cfg: BacktestConfig, df=None):
    """Assemble all kernel inputs from a config (+df for ref_col stops) and run it.

    Shared by run_backtest and the sweep runner so both go through one code path.
    """
    n = close.shape[0]
    n_tp, tp_modes, tp_values, tp_close, tp_mv_modes, tp_mv_values = cfg.tp_arrays()
    sl_ref_long = _ref_arr(df, cfg.sl_ref_long_col, n)
    sl_ref_short = _ref_arr(df, cfg.sl_ref_short_col, n)
    return run_kernel(
        open_, high, low, close, el, xl, es, xs,
        sl_ref_long, sl_ref_short,
        tp_modes, tp_values, tp_close, tp_mv_modes, tp_mv_values, n_tp,
        **cfg.scalar_args(),
    )


@dataclass
class SimResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    stats: Dict[str, float]
    elapsed_s: float


def run_backtest(
    df: pd.DataFrame,
    signals: Signals,
    cfg: BacktestConfig,
    *,
    time_col: str = "t",
    price_col: str = "close",
) -> SimResult:
    if df.empty:
        raise ValueError("Empty df")
    for c in (time_col, price_col):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in df.")

    n = len(df)
    close = pd.to_numeric(df[price_col], errors="coerce").to_numpy(np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(np.float64) if "high" in df else close
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(np.float64) if "low" in df else close
    open_ = pd.to_numeric(df["open"], errors="coerce").to_numpy(np.float64) if "open" in df else close

    el, xl, es, xs = signals.as_u8(n)

    t0 = time.perf_counter()
    (t_side, t_entry_i, t_exit_i, t_entry_px, t_exit_px, t_qty,
     t_gross, t_entry_fee, t_exit_fee, t_pnl, t_reason,
     equity_curve, pos_count, final_cash) = invoke_kernel(
        open_, high, low, close, el, xl, es, xs, cfg, df=df
    )
    elapsed = time.perf_counter() - t0

    # Keep time as a pandas Series (tz-aware). Never call .to_numpy() on a tz-aware column
    # for the full series — that materializes a slow 500k-element object array of Timestamps.
    t_series = df[time_col].reset_index(drop=True)
    trades = _build_trades_df(
        t_series, t_side, t_entry_i, t_exit_i, t_entry_px, t_exit_px,
        t_qty, t_gross, t_entry_fee, t_exit_fee, t_pnl, t_reason,
    )

    eq = pd.DataFrame({"equity": equity_curve, "open_trades": pos_count})
    eq.insert(0, "t", t_series)   # direct Series assign preserves tz and is fast
    peak = np.maximum.accumulate(np.where(np.isnan(equity_curve), -np.inf, equity_curve))
    with np.errstate(invalid="ignore", divide="ignore"):
        eq["drawdown"] = np.where(peak > 0, (peak - equity_curve) / peak, 0.0)

    from ..analytics.metrics import compute_stats
    stats = compute_stats(trades, eq, initial_cash=float(cfg.initial_cash),
                          final_cash=float(final_cash))
    stats["n_bars"] = float(n)
    stats["engine_elapsed_s"] = float(elapsed)

    _log.info("backtest done | bars=%s trades=%s final_cash=%.2f return=%.2f%% dd=%.2f%% | %.1f ms",
              f"{n:,}", f"{len(trades):,}", final_cash,
              stats.get("total_return_pct", 0.0), stats.get("max_drawdown_pct", 0.0),
              elapsed * 1000)

    return SimResult(trades=trades, equity_curve=eq, stats=stats, elapsed_s=elapsed)


def _build_trades_df(t_series, side, entry_i, exit_i, entry_px, exit_px,
                     qty, gross, entry_fee, exit_fee, pnl, reason) -> pd.DataFrame:
    if len(side) == 0:
        return pd.DataFrame(columns=[
            "side", "entry_time", "exit_time", "entry_i", "exit_i",
            "entry_price", "exit_price", "qty", "gross_pnl", "fees", "pnl",
            "return_pct", "bars_held", "close_reason",
        ])
    n = len(t_series)
    # Positional selection of only the trade bars (few thousand) — cheap even if tz-aware.
    entry_time = t_series.iloc[entry_i].to_numpy()
    exit_time = t_series.iloc[np.clip(exit_i, 0, n - 1)].to_numpy()
    exit_time = np.where(exit_i >= 0, exit_time, np.datetime64("NaT"))
    notional = entry_px * qty
    with np.errstate(invalid="ignore", divide="ignore"):
        return_pct = np.where(notional > 0, pnl / notional * 100.0, np.nan)
    df = pd.DataFrame({
        "side": np.where(side == 1, "long", "short"),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_i": entry_i,
        "exit_i": exit_i,
        "entry_price": entry_px,
        "exit_price": exit_px,
        "qty": qty,
        "gross_pnl": gross,
        "fees": entry_fee + exit_fee,
        "pnl": pnl,
        "return_pct": return_pct,
        "bars_held": np.where(exit_i >= 0, exit_i - entry_i, 0),
        "close_reason": pd.Series(reason).map(REASON_NAMES).to_numpy(),
    })
    return df
