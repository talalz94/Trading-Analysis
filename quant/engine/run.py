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

REASON_NAMES = {0: "signal", 1: "stop_loss", 2: "take_profit", 3: "forced_close_end",
                4: "margin_call", 5: "alt_exit"}


def _ref_arr(df, col, n):
    if col is not None and df is not None and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(np.float64)
    return np.full(n, np.nan, dtype=np.float64)


def _leg_args(signals, xl, xs, n):
    """Build the kernel's leg inputs. Single-leg (the default) reuses xl/xs and never consults the
    leg arrays; multi-leg pulls per-rule exit columns + per-bar entry-leg indices off `signals`."""
    ell = getattr(signals, "exit_legs_long", None)
    if ell is None:
        # single leg: entry_leg unused, exit_legs is a 1-row view of xl/xs (kept valid but unused)
        return (np.zeros(n, np.int64), np.zeros(n, np.int64),
                xl.reshape(1, n), xs.reshape(1, n), 1)
    els = signals.exit_legs_short
    n_legs = ell.shape[0]
    return (np.ascontiguousarray(signals.entry_leg_long).astype(np.int64),
            np.ascontiguousarray(signals.entry_leg_short).astype(np.int64),
            np.ascontiguousarray(ell).astype(np.uint8),
            np.ascontiguousarray(els).astype(np.uint8), n_legs)


_SL_MODES = {"none": 0, "entry_pct": 1, "price_abs": 2, "ref_col": 3}
_FB_MODES = {"entry_pct": 1, "price_abs": 2}


def _alt_exit_args(signals, n):
    """Secondary-exit inputs. Returns all-zero arrays + flags 0 (the byte-identical default path)
    unless the signals carry at least one `exit_alt_*` column."""
    z = np.zeros(n, np.uint8)
    if signals is None:
        return z, z, z, z, 0, 0, 0
    al = getattr(signals, "exit_alt_long", None)
    as_ = getattr(signals, "exit_alt_short", None)
    if al is None and as_ is None:
        return z, z, z, z, 0, 0, 0

    def u8(a):
        return z if a is None else np.ascontiguousarray(np.asarray(a, bool)).astype(np.uint8)
    replace_tp = 1 if getattr(signals, "alt_replace_tp", False) else 0
    arm_on_tp = 1 if getattr(signals, "alt_arm_on_tp", False) else 0
    return (u8(getattr(signals, "exit_arm_long", None)), u8(getattr(signals, "exit_arm_short", None)),
            u8(al), u8(as_), 1, replace_tp, arm_on_tp)


def _no_leg_stops(n_legs):
    """Default per-leg-stop arrays (unused; the kernel takes the cfg stop path when flag == 0)."""
    z = max(n_legs, 1)
    return (0, np.zeros(z, np.int64), np.zeros(z, np.float64), np.zeros(z, np.float64),
            np.zeros(z, np.float64), np.zeros(z, np.int64), np.zeros(z, np.float64),
            np.zeros((z, 1), np.float64), np.zeros((z, 1), np.float64))


def _leg_stop_args(signals, df, n, n_legs):
    """Build per-leg stop inputs from `signals.stop_specs` (one StopSpec-or-None per leg). A None
    leg falls back to... just the same-shape zeros (mode 0 -> no stop). Ref columns are pulled off
    df into a [n_legs, n] table so ref_col stops work per rule."""
    specs = getattr(signals, "stop_specs", None)
    if not specs:
        return _no_leg_stops(n_legs)
    mode = np.zeros(n_legs, np.int64); value = np.zeros(n_legs, np.float64)
    buf = np.zeros(n_legs, np.float64); maxrisk = np.zeros(n_legs, np.float64)
    fb_mode = np.zeros(n_legs, np.int64); fb_value = np.zeros(n_legs, np.float64)
    ref_long = np.full((n_legs, n), np.nan); ref_short = np.full((n_legs, n), np.nan)
    for i, sp in enumerate(specs):
        if sp is None:
            continue
        mode[i] = _SL_MODES[sp.mode]; value[i] = float(sp.value)
        buf[i] = float(sp.buffer_pct); maxrisk[i] = float(sp.max_ref_risk_pct)
        fb_mode[i] = _FB_MODES[sp.fallback_mode]; fb_value[i] = float(sp.fallback_value)
        ref_long[i] = _ref_arr(df, sp.ref_col_long, n)
        ref_short[i] = _ref_arr(df, sp.ref_col_short, n)
    return (1, mode, value, buf, maxrisk, fb_mode, fb_value,
            np.ascontiguousarray(ref_long), np.ascontiguousarray(ref_short))


def invoke_kernel(open_, high, low, close, el, xl, es, xs, cfg: BacktestConfig, df=None,
                  signals=None):
    """Assemble all kernel inputs from a config (+df for ref_col stops) and run it.

    Shared by run_backtest and the sweep runner so both go through one code path.
    """
    n = close.shape[0]
    n_tp, tp_modes, tp_values, tp_close, tp_mv_modes, tp_mv_values = cfg.tp_arrays()
    sl_ref_long = _ref_arr(df, cfg.sl_ref_long_col, n)
    sl_ref_short = _ref_arr(df, cfg.sl_ref_short_col, n)
    ell_i, els_i, exit_legs_long, exit_legs_short, n_legs = _leg_args(signals, xl, xs, n)
    (per_leg_stops, leg_sl_mode, leg_sl_value, leg_sl_buffer, leg_sl_maxrisk,
     leg_sl_fb_mode, leg_sl_fb_value, leg_sl_ref_long, leg_sl_ref_short) = _leg_stop_args(
        signals, df, n, n_legs)
    arm_l, arm_s, alt_l, alt_s, alt_on, alt_rep, alt_tp = _alt_exit_args(signals, n)
    return run_kernel(
        open_, high, low, close, el, xl, es, xs,
        ell_i, els_i, exit_legs_long, exit_legs_short, n_legs,
        per_leg_stops, leg_sl_mode, leg_sl_value, leg_sl_buffer, leg_sl_maxrisk,
        leg_sl_fb_mode, leg_sl_fb_value, leg_sl_ref_long, leg_sl_ref_short,
        sl_ref_long, sl_ref_short,
        arm_l, arm_s, alt_l, alt_s, alt_on, alt_rep, alt_tp,
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
     equity_curve, pos_count, final_cash, t_stop) = invoke_kernel(
        open_, high, low, close, el, xl, es, xs, cfg, df=df, signals=signals
    )
    elapsed = time.perf_counter() - t0

    # Keep time as a pandas Series (tz-aware). Never call .to_numpy() on a tz-aware column
    # for the full series — that materializes a slow 500k-element object array of Timestamps.
    t_series = df[time_col].reset_index(drop=True)
    trades = _build_trades_df(
        t_series, t_side, t_entry_i, t_exit_i, t_entry_px, t_exit_px,
        t_qty, t_gross, t_entry_fee, t_exit_fee, t_pnl, t_reason, t_stop,
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
                     qty, gross, entry_fee, exit_fee, pnl, reason, stop=None) -> pd.DataFrame:
    if len(side) == 0:
        return pd.DataFrame(columns=[
            "side", "entry_time", "exit_time", "entry_i", "exit_i",
            "entry_price", "exit_price", "stop_price", "qty", "gross_pnl", "fees", "pnl",
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
        "stop_price": stop if stop is not None else np.nan,
        "qty": qty,
        "gross_pnl": gross,
        "fees": entry_fee + exit_fee,
        "pnl": pnl,
        "return_pct": return_pct,
        "bars_held": np.where(exit_i >= 0, exit_i - entry_i, 0),
        "close_reason": pd.Series(reason).map(REASON_NAMES).to_numpy(),
    })
    return df
