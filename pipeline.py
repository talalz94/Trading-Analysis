from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from indicators_load import INDICATOR_REGISTRY


@dataclass
class IndicatorSpec:
    """
    name: indicator type key (e.g. "rsi_divergence")
    tag:  unique instance id (required for multi-config usage)
    cols: populated by pipeline with actual output columns created
    """
    name: str
    tag: str
    config: Dict[str, Any] = field(default_factory=dict)
    cols: List[str] = field(default_factory=list)


def apply_mas(df: pd.DataFrame, ma_windows: List[int]) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    created = []
    for w in ma_windows:
        c = f"MA{w}"
        out[c] = out["close"].rolling(window=w, min_periods=w).mean()
        created.append(c)
    return out, created


def build_feature_df(
    raw_df: pd.DataFrame,
    tz: str,
    ma_windows: Optional[List[int]] = None,
    indicators: Optional[List[IndicatorSpec]] = None,
) -> Tuple[pd.DataFrame, List[IndicatorSpec], List[str]]:
    """
    Returns:
      df_feat: raw_df + 't' + MAs + indicator columns (for ALL rows)
      indicators: same objects, but .cols filled with created columns
      ma_cols: created MA column names
    """
    if raw_df.empty:
        raise ValueError("No data to process.")

    ma_windows = ma_windows or []
    indicators = indicators or []

    # copy + timezone index column
    df = raw_df.copy()
    df["t"] = df["open_time"].dt.tz_convert(tz)

    # MAs are overlays but still useful as features
    df, ma_cols = apply_mas(df, ma_windows)

    # Apply indicators in order; each fills spec.cols with the created col names
    used_tags = set()
    for spec in indicators:
        if not spec.tag or not isinstance(spec.tag, str):
            raise ValueError("Each IndicatorSpec must have a unique non-empty string 'tag'.")
        if spec.tag in used_tags:
            raise ValueError(f"Duplicate IndicatorSpec.tag '{spec.tag}'. Tags must be unique.")
        used_tags.add(spec.tag)

        ind = INDICATOR_REGISTRY.get(spec.name)
        if ind is None:
            raise ValueError(f"Unknown indicator '{spec.name}'. Available: {sorted(INDICATOR_REGISTRY.keys())}")

        df, created_cols = ind.compute(df, spec.config, spec.tag)
        spec.cols = created_cols

    return df, indicators, ma_cols