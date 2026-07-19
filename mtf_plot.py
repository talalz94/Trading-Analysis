from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

import pandas as pd

from pipeline import IndicatorSpec
from precomputed_factory import precomputed_from_indicator


def _as_set(values: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if values is None:
        return None
    return {str(v).strip() for v in values}


def _match_spec(spec: IndicatorSpec, keys: Set[str]) -> bool:
    """
    Match by:
      - indicator name: "macd", "rsi_divergence"
      - tag: "macd_8_21_5", "rsi14"
      - name:tag: "macd:macd_8_21_5"
    """
    return (
        spec.name in keys
        or spec.tag in keys
        or f"{spec.name}:{spec.tag}" in keys
    )


def precomputed_from_timeframe(
    tf: str,
    specs: Iterable[IndicatorSpec],
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> List[IndicatorSpec]:
    """
    Convert normal indicator specs from one timeframe into precomputed plot specs.

    Example:
      precomputed_from_timeframe("5m", ind_5m, include=["rsi14", "macd_8_21_5"])
    """
    include_set = _as_set(include)
    exclude_set = _as_set(exclude) or set()

    out: List[IndicatorSpec] = []

    for spec in specs:
        if include_set is not None and not _match_spec(spec, include_set):
            continue

        if exclude_set and _match_spec(spec, exclude_set):
            continue

        out.append(precomputed_from_indicator(tf, spec))

    return out


def make_plot_indicators(
    base_specs: Optional[Iterable[IndicatorSpec]] = None,
    mtf_specs: Optional[Dict[str, Iterable[IndicatorSpec]]] = None,
    mtf_include: Optional[Dict[str, Iterable[str]]] = None,
    mtf_exclude: Optional[Dict[str, Iterable[str]]] = None,
) -> List[IndicatorSpec]:
    """
    Build final indicator list for plotting.

    base_specs:
      Indicators computed directly on the base dataframe, usually 1m.

    mtf_specs:
      Dict of timeframe -> indicator specs.
      These are converted into precomputed plot specs.

    mtf_include:
      Dict of timeframe -> indicators to include.
      Match by spec.name, spec.tag, or "name:tag".

    mtf_exclude:
      Dict of timeframe -> indicators to exclude.

    Example:
      plot_indicators = make_plot_indicators(
          base_specs=ind_1m,
          mtf_specs={"5m": ind_5m, "15m": ind_15m},
          mtf_include={
              "5m": ["rsi14", "macd_8_21_5"],
              "15m": ["rsi14", "macd_8_21_5"],
          }
      )
    """
    out: List[IndicatorSpec] = []

    if base_specs:
        out.extend(list(base_specs))

    if mtf_specs:
        mtf_include = mtf_include or {}
        mtf_exclude = mtf_exclude or {}

        for tf, specs in mtf_specs.items():
            out.extend(
                precomputed_from_timeframe(
                    tf=tf,
                    specs=specs,
                    include=mtf_include.get(tf),
                    exclude=mtf_exclude.get(tf),
                )
            )

    return out


def assert_plot_columns_exist(df: pd.DataFrame, indicators: Iterable[IndicatorSpec]) -> None:
    """
    Safety check before plotting.

    If this fails, you are likely passing df_raw=df1_feat instead of df_raw=merged.
    """
    missing = []

    for spec in indicators:
        if spec.name != "precomputed":
            continue

        for trace in spec.config.get("traces", []):
            col = trace.get("col")
            if col and col not in df.columns:
                missing.append(col)

    if missing:
        prefixed_cols = [c for c in df.columns if "__" in c]

        raise KeyError(
            "Missing precomputed plot columns. "
            "Most likely you passed df_raw=df1_feat instead of df_raw=merged, "
            "or the higher-timeframe indicators were not built/aligned. "
            f"Missing columns: {missing}. "
            f"Available prefixed columns sample: {prefixed_cols[:50]}"
        )