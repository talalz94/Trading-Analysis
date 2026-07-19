from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Optional, Union

import pandas as pd

from pipeline import IndicatorSpec
from precomputed_factory import precomputed_from_indicator


VisibleMode = Union[bool, Literal["legendonly"]]


@dataclass(frozen=True)
class PlotToggle:
    """
    Select one calculated indicator from one timeframe for plotting.

    tf:
      "1m", "5m", "15m", etc.

    key:
      matches IndicatorSpec.tag, IndicatorSpec.name, or "name:tag".

    visible:
      True          -> shown by default
      "legendonly"  -> added to plot but hidden; user can toggle from legend
      False         -> not plotted at all
    """
    tf: str
    key: str
    visible: VisibleMode = True
    title: Optional[str] = None


def _match_spec(spec: IndicatorSpec, key: str) -> bool:
    key = str(key).strip()

    return (
        spec.tag == key
        or spec.name == key
        or f"{spec.name}:{spec.tag}" == key
    )


def _find_spec(specs: Iterable[IndicatorSpec], key: str) -> IndicatorSpec:
    matches = [s for s in specs if _match_spec(s, key)]

    if not matches:
        available = [f"{s.name}:{s.tag}" for s in specs]
        raise KeyError(
            f"No indicator spec found for key='{key}'. "
            f"Available specs: {available}"
        )

    if len(matches) > 1:
        available = [f"{s.name}:{s.tag}" for s in matches]
        raise ValueError(
            f"Ambiguous plot key='{key}'. Matched multiple specs: {available}. "
            f"Use 'name:tag' to disambiguate."
        )

    return matches[0]


def make_plot_indicators_from_toggles(
    indicators_by_tf: Dict[str, Iterable[IndicatorSpec]],
    toggles: Iterable[PlotToggle],
) -> List[IndicatorSpec]:
    """
    Build plot indicators from explicit toggles.

    This fully separates:
      - indicators calculated for strategy
      - indicators shown on chart

    The returned specs are all precomputed, including the base timeframe.
    """
    out: List[IndicatorSpec] = []

    for toggle in toggles:
        if toggle.visible is False:
            continue

        if toggle.tf not in indicators_by_tf:
            raise KeyError(
                f"Timeframe '{toggle.tf}' not found in indicators_by_tf. "
                f"Available timeframes: {list(indicators_by_tf.keys())}"
            )

        spec = _find_spec(indicators_by_tf[toggle.tf], toggle.key)
        plot_spec = precomputed_from_indicator(
            tf=toggle.tf,
            spec=spec,
            title=toggle.title,
            visible=toggle.visible,
        )

        if plot_spec is not None:
            out.append(plot_spec)

    return out


def assert_plot_columns_exist(df: pd.DataFrame, indicators: Iterable[IndicatorSpec]) -> None:
    """
    Safety check before plotting.

    If this fails, usually:
      - you passed df_raw=df1_feat instead of df_raw=merged
      - or you asked to plot a timeframe/indicator that was not built/aligned
    """
    missing = []

    for spec in indicators:
        if spec.name != "precomputed":
            continue

        for trace in spec.config.get("traces", []):
            col = trace.get("col")
            optional = bool(trace.get("optional", False))

            if col and col not in df.columns and not optional:
                missing.append(col)

    if missing:
        prefixed_cols = [c for c in df.columns if "__" in c]

        raise KeyError(
            "Missing precomputed plot columns. "
            "Use df_raw=merged when plotting higher-timeframe indicators. "
            f"Missing columns: {missing}. "
            f"Available prefixed columns sample: {prefixed_cols[:80]}"
        )