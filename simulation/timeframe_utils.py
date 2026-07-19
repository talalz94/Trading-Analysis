from __future__ import annotations

from typing import Iterable, List, Optional
import pandas as pd


CORE_OHLCV_COLS = {
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "num_trades",
    "taker_buy_base",
    "taker_buy_quote",
    "t",
}


def feature_columns(df: pd.DataFrame, extra_exclude: Optional[Iterable[str]] = None) -> List[str]:
    """
    Returns columns that are indicator/feature columns, excluding raw OHLCV/time columns.
    """
    exclude = set(CORE_OHLCV_COLS)
    if extra_exclude:
        exclude.update(extra_exclude)

    return [c for c in df.columns if c not in exclude]


def shift_htf_features_to_closed_candle(
    df_feat: pd.DataFrame,
    shift_bars: int = 1,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Makes higher-timeframe feature data backtest-safe.

    Binance candles are timestamped by open_time. A 5m candle with t=10:00
    is not closed until 10:05. If we merge that unshifted row onto 1m bars
    between 10:00 and 10:04, we leak future information.

    This function shifts only feature/indicator columns by one HTF candle,
    leaving t/open_time/OHLCV untouched.

    Example:
      5m row t=10:05 receives indicator values from the 5m candle t=10:00,
      which closed at 10:05 and is now safe to use.
    """
    if shift_bars <= 0:
        return df_feat.copy()

    out = df_feat.copy()

    if feature_cols is None:
        feature_cols = feature_columns(out)

    if feature_cols:
        out[feature_cols] = out[feature_cols].shift(int(shift_bars))

    return out