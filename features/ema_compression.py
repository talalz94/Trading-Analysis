from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EMACompressionSpec:
    name: str
    cols: Sequence[str]
    compress_thr: float = 0.10
    expand_thr: float = 0.15
    lookback: int = 20
    require_bullish_order: bool = True
    require_bearish_order: bool = False
    fresh_only: bool = False


def _rolling_any_bool(mask: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(mask).fillna(False).astype(bool)
    return (
        s.rolling(int(window), min_periods=1)
        .max()
        .astype(bool)
        .to_numpy()
    )


def add_ema_compression_features(
    df: pd.DataFrame,
    specs: Sequence[EMACompressionSpec],
) -> pd.DataFrame:
    """
    Adds EMA compression / expansion features.

    For each spec named, for example:
        ema_15m_50_100_150_comp

    Creates:
        ema_15m_50_100_150_comp__range_pct
        ema_15m_50_100_150_comp__compressed_now
        ema_15m_50_100_150_comp__compressed_recent
        ema_15m_50_100_150_comp__expanded_now
        ema_15m_50_100_150_comp__ordered_bullish
        ema_15m_50_100_150_comp__ordered_bearish
        ema_15m_50_100_150_comp__bull_breakout
        ema_15m_50_100_150_comp__bear_breakout
        ema_15m_50_100_150_comp__compress_thr
        ema_15m_50_100_150_comp__expand_thr
    """
    out = df.copy()
    new_cols = {}

    for spec in specs:
        missing = [c for c in spec.cols if c not in out.columns]
        if missing:
            raise KeyError(f"Missing EMA columns for {spec.name}: {missing}")

        cols = list(spec.cols)
        mat = out[cols].to_numpy(dtype=float)

        row_max = np.nanmax(mat, axis=1)
        row_min = np.nanmin(mat, axis=1)
        row_mean = np.nanmean(mat, axis=1)

        range_pct = np.where(
            np.abs(row_mean) > 1e-12,
            (row_max - row_min) / row_mean * 100.0,
            np.nan,
        )

        compressed_now = range_pct < float(spec.compress_thr)
        compressed_recent = _rolling_any_bool(compressed_now, int(spec.lookback))
        expanded_now = range_pct > float(spec.expand_thr)

        ordered_bullish = np.all(mat[:, :-1] > mat[:, 1:], axis=1)
        ordered_bearish = np.all(mat[:, :-1] < mat[:, 1:], axis=1)

        bull_breakout = compressed_recent & expanded_now
        bear_breakout = compressed_recent & expanded_now

        if spec.require_bullish_order:
            bull_breakout = bull_breakout & ordered_bullish

        if spec.require_bearish_order:
            bear_breakout = bear_breakout & ordered_bearish

        if spec.fresh_only:
            prev_bull = pd.Series(bull_breakout).shift(1).fillna(False).to_numpy(dtype=bool)
            prev_bear = pd.Series(bear_breakout).shift(1).fillna(False).to_numpy(dtype=bool)

            bull_breakout = bull_breakout & ~prev_bull
            bear_breakout = bear_breakout & ~prev_bear

        base = spec.name

        new_cols[f"{base}__range_pct"] = range_pct
        new_cols[f"{base}__compressed_now"] = compressed_now
        new_cols[f"{base}__compressed_recent"] = compressed_recent
        new_cols[f"{base}__expanded_now"] = expanded_now
        new_cols[f"{base}__ordered_bullish"] = ordered_bullish
        new_cols[f"{base}__ordered_bearish"] = ordered_bearish
        new_cols[f"{base}__bull_breakout"] = bull_breakout
        new_cols[f"{base}__bear_breakout"] = bear_breakout

        # Constant threshold lines for plotting.
        new_cols[f"{base}__compress_thr"] = np.full(len(out), float(spec.compress_thr))
        new_cols[f"{base}__expand_thr"] = np.full(len(out), float(spec.expand_thr))

    features = pd.DataFrame(new_cols, index=out.index)

    return pd.concat([out, features], axis=1).copy()

def range_slope_strong_for_n_tf_candles(
    c,
    col: str,
    tf_step: int,
    n: int,
    min_step_delta: float = 0.0,
    min_total_delta: float = 0.0,
) -> bool:
    """
    Checks EMA range % slope strength.

    col:
      EMA compression range column, e.g.
      "ema_15m_50_100_150_comp__range_pct"

    tf_step:
      1  = 1m timeframe on 1m base chart
      5  = 5m timeframe on 1m base chart
      15 = 15m timeframe on 1m base chart

    n:
      Number of timeframe candles to check.

    min_step_delta:
      Minimum increase required at each step.
      Example: 0.005 means each step must increase by at least 0.005 percentage points.

    min_total_delta:
      Minimum total increase from start to now.
      Example: 0.03 means range_pct must increase by at least 0.03 percentage points over the full window.
    """

    current = c.v(col, shift=0)
    start = c.v(col, shift=-(n * tf_step))

    if not np.isfinite(current) or not np.isfinite(start):
        return False

    total_delta = current - start

    if total_delta < min_total_delta:
        return False

    # Check each timeframe step.
    for k in range(n):
        cur_shift = -(k * tf_step)
        prev_shift = -((k + 1) * tf_step)

        cur = c.v(col, shift=cur_shift)
        prev = c.v(col, shift=prev_shift)

        if not np.isfinite(cur) or not np.isfinite(prev):
            return False

        step_delta = cur - prev

        if step_delta < min_step_delta:
            return False

    return True