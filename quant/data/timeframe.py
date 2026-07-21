"""
Multi-timeframe utilities: resample OHLCV to a higher timeframe and align higher-timeframe
features back onto the base (e.g. 1m) grid WITHOUT lookahead.

Anti-lookahead rule (critical): a higher-timeframe candle stamped at open_time T only *closes*
one interval later, so its indicator values must not be visible to base bars before that close.
`align_timeframes` shifts HTF feature columns by one HTF bar before the as-of merge, so a 5m
candle opened at 10:00 (closing 10:05) is only seen by 1m bars from 10:05 onward.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from .base import TIME_COL

CORE_OHLCV = {
    TIME_COL, "open", "high", "low", "close", "volume",
    "quote_volume", "num_trades", "taker_buy_base", "taker_buy_quote", "t",
}

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def resample_ohlcv(df: pd.DataFrame, rule: str, *, time_col: str = TIME_COL,
                   tz: Optional[str] = None) -> pd.DataFrame:
    """Resample a 1m (or finer) OHLCV frame to a coarser timeframe (e.g. '5min', '15min', '1h')."""
    agg = {k: v for k, v in _AGG.items() if k in df.columns}
    out = (df.set_index(time_col).resample(rule, label="left", closed="left")
           .agg(agg).dropna(subset=["open"]).reset_index())
    if "t" in df.columns or tz is not None:
        tzname = tz or (df["t"].dt.tz if "t" in df.columns else "UTC")
        out["t"] = out[time_col].dt.tz_convert(tzname) if out[time_col].dt.tz is not None else out[time_col]
    return out


def feature_columns(df: pd.DataFrame, exclude: Optional[Iterable[str]] = None) -> List[str]:
    ex = set(CORE_OHLCV)
    if exclude:
        ex.update(exclude)
    return [c for c in df.columns if c not in ex]


def align_timeframes(
    base_df: pd.DataFrame,
    others: Dict[str, pd.DataFrame],
    *,
    time_col: str = "t",
    shift_bars: int = 1,
) -> pd.DataFrame:
    """Merge HTF feature columns onto the base grid via as-of join, prefixed `{tf}__`.

    others: {timeframe_label: htf_df_with_features}. HTF feature columns are shifted by
    `shift_bars` HTF candles first (anti-lookahead). Base OHLCV stays unprefixed.
    """
    if time_col not in base_df.columns:
        raise ValueError(f"base_df must contain '{time_col}'")
    merged = base_df.sort_values(time_col).copy()

    for tf, d in others.items():
        d2 = d.sort_values(time_col).copy()
        feats = feature_columns(d2)
        if shift_bars > 0 and feats:
            d2[feats] = d2[feats].shift(shift_bars)
        keep = [time_col] + feats
        d2 = d2[keep].rename(columns={c: f"{tf}__{c}" for c in feats})
        merged = pd.merge_asof(merged.sort_values(time_col), d2.sort_values(time_col),
                               on=time_col, direction="backward", allow_exact_matches=True)
    return merged


def build_mtf(
    base_df: pd.DataFrame,
    builders: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]],
    *,
    time_col: str = "t",
    shift_bars: int = 1,
) -> pd.DataFrame:
    """Convenience: resample base to each timeframe, run a feature builder, align back.

    builders: {timeframe_rule: fn(htf_df) -> htf_df_with_feature_columns}. The timeframe
    rule (e.g. '5min') is also used as the column prefix.
    Example:
        build_mtf(df_1m, {"5min": lambda d: add_emas(d, [50, 200]),
                          "15min": lambda d: add_emas(d, [50])})
        # -> columns like '5min__ema_50', '15min__ema_50' on the 1m grid, lookahead-safe.
    """
    others = {}
    for rule, fn in builders.items():
        htf = resample_ohlcv(base_df, rule)
        others[rule] = fn(htf)
    return align_timeframes(base_df, others, time_col=time_col, shift_bars=shift_bars)
