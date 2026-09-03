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


# --------------------------------------------------------------------------------------------
# Multi-timeframe FRAMES (as opposed to multi-timeframe COLUMNS, which is what build_mtf gives)
# --------------------------------------------------------------------------------------------
# `build_mtf` folds higher-timeframe FEATURES onto the base grid, which is what a strategy needs.
# Drawing is a different problem: to plot a 4h pattern you need the 4h FRAME it was fitted on,
# because its slopes are per-4h-bar and its indices index that frame. So these return frames.

_TF_UNIT_MINUTES = {"m": 1, "min": 1, "h": 60, "d": 1440, "w": 10080}


def tf_minutes(tf: str) -> int:
    """Minutes in one bar of `tf` ('1m', '15m', '1h', '4h', '1d', '1w')."""
    s = str(tf).strip().lower()
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] == "."):
        i += 1
    n, unit = s[:i], s[i:]
    if not n or unit not in _TF_UNIT_MINUTES:
        raise ValueError(f"cannot parse timeframe {tf!r}; expected e.g. '15m', '1h', '4h', '1d'")
    return int(float(n) * _TF_UNIT_MINUTES[unit])


def tf_frames(symbol: str, tfs, *, view_start, end, warmup_bars: int = 800,
              source: str = "binance", market: str = "spot", tz: str = "UTC",
              **kwargs) -> Dict[str, pd.DataFrame]:
    """Load one OHLCV frame per timeframe, each with the SAME warm-up measured in BARS.

    Detection cares about bars, not about wall-clock: a pattern needs `min_span_bars` bars of its
    own timeframe, and the zigzag needs pivots before that. So a fixed calendar start gives 4h
    almost nothing while giving 1m far more than it can use. Each frame therefore starts at
    ``view_start - warmup_bars x <one bar of that timeframe>``.

    Frames are fetched rather than resampled from a finer one on purpose. Resampling is free, but
    it caps the coarse frame's history at the finer frame's start, and coarse timeframes are
    exactly the ones that need a long run-up: a 4h shape spans 100-200 4h bars, which is 17-33
    days for the shape alone. `get_ohlcv` caches incrementally, so the fetch is paid once.

    Returns ``{tf: DataFrame}`` ordered coarsest first, which is also the order they should be
    drawn in — coarse structure underneath, fine detail on top.
    """
    from .loader import get_ohlcv

    vs = pd.Timestamp(view_start)
    if vs.tzinfo is None:
        vs = vs.tz_localize("UTC")
    out: Dict[str, pd.DataFrame] = {}
    for tf in sorted(dict.fromkeys(tfs), key=tf_minutes, reverse=True):
        start = (vs - pd.Timedelta(minutes=tf_minutes(tf) * int(warmup_bars))).strftime("%Y-%m-%d")
        out[tf] = get_ohlcv(symbol, tf, start=start, end=end, source=source,
                            market=market, tz=tz, **kwargs).reset_index(drop=True)
    return out


def bar_interval(df: pd.DataFrame, *, time_col: str = "t") -> pd.Timedelta:
    """One bar's duration, measured from the frame itself rather than from a label.

    Used to stamp a higher-timeframe EVENT at the bar's CLOSE. Bars are stamped at their OPEN
    everywhere in this project, so an event read off bar `i` of a 4h frame is only knowable four
    hours after that timestamp; drawing it at the open would be a four-hour lookahead on a 15m
    chart. The median of the diffs is exact for a regular grid and survives gaps and DST.
    """
    if time_col not in df.columns or len(df) < 2:
        return pd.Timedelta(0)
    d = pd.to_datetime(df[time_col]).diff().dropna()
    return pd.Timedelta(d.median()) if len(d) else pd.Timedelta(0)
