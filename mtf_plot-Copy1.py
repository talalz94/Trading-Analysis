from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import pandas as pd

from pipeline import IndicatorSpec
from precomputed_factory import precomputed_from_indicator


def precomputed_from_timeframe(
    tf: str,
    specs: Iterable[IndicatorSpec],
    include: Optional[Iterable[str]] = None,
) -> List[IndicatorSpec]:
    """
    Converts a list of normal indicator specs for one timeframe into
    precomputed plot specs.

    Example:
      ind_5m = [
          IndicatorSpec("macd", tag="macd", config={...}),
          IndicatorSpec("rsi_divergence", tag="rsi14", config={...}),
      ]

      precomputed_from_timeframe("5m", ind_5m)

    No duplicate indicator definitions needed.
    """
    include_set = set(include) if include is not None else None

    out: List[IndicatorSpec] = []

    for spec in specs:
        if include_set is not None and spec.tag not in include_set and spec.name not in include_set:
            continue

        out.append(precomputed_from_indicator(tf, spec))

    return out


def make_plot_indicators(
    base_specs: Optional[Iterable[IndicatorSpec]] = None,
    mtf_specs: Optional[Dict[str, Iterable[IndicatorSpec]]] = None,
) -> List[IndicatorSpec]:
    """
    Builds one final indicators list for plotting.

    base_specs:
      Normal indicators computed directly on the plotted/base dataframe.

    mtf_specs:
      Dict of timeframe -> original specs.
      These become generic precomputed specs automatically.

    Example:
      plot_indicators = make_plot_indicators(
          base_specs=ind_1m,
          mtf_specs={
              "5m": ind_5m,
              "15m": ind_15m,
          }
      )
    """
    out: List[IndicatorSpec] = []

    if base_specs:
        out.extend(list(base_specs))

    if mtf_specs:
        for tf, specs in mtf_specs.items():
            out.extend(precomputed_from_timeframe(tf, specs))

    return out


def assert_plot_columns_exist(df: pd.DataFrame, indicators: Iterable[IndicatorSpec]) -> None:
    """
    Optional safety check before plotting.

    This catches the common mistake:
      using df1_feat instead of merged for higher-timeframe precomputed panels.
    """
    missing = []

    for spec in indicators:
        if spec.name != "precomputed":
            continue

        traces = spec.config.get("traces", [])
        for tr in traces:
            c = tr.get("col")
            if c and c not in df.columns:
                missing.append(c)

    if missing:
        htf_cols = [c for c in df.columns if "__" in c]
        raise KeyError(
            "Missing precomputed plot columns. "
            "This usually means you passed df_raw=df1_feat instead of df_raw=merged. "
            f"Missing columns: {missing}. "
            f"Available prefixed columns sample: {htf_cols[:40]}"
        )