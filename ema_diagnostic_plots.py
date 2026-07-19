from __future__ import annotations

from pipeline import IndicatorSpec


def ema_pair_spread_panel(
    name: str,
    title: str | None = None,
    mode: str = "pct",
    visible=True,
    show_zero_line: bool = True,
    dynamic_threshold_cols: list[str] | None = None,
) -> IndicatorSpec:
    """
    Panel for difference between two EMAs.

    mode="pct":
      plots {name}__pct

    mode="abs":
      plots {name}__abs
    """
    value_col = f"{name}__{mode}"

    traces = [
        {
            "kind": "line",
            "col": value_col,
            "name": title or value_col,
            "width": 2.0,
        }
    ]

    for c in dynamic_threshold_cols or []:
        traces.append({
            "kind": "line",
            "col": c,
            "name": c,
            "width": 1.2,
            "dash": "dot",
        })

    hlines = []
    if show_zero_line:
        hlines.append({"y": 0, "dash": "dot", "color": "rgba(0,0,0,0.35)"})

    return IndicatorSpec("precomputed", tag=f"{name}_{mode}_panel", config={
        "title": title or f"{name} {mode}",
        "is_overlay": False,
        "row_weight": 0.70,
        "visible": visible,
        "traces": traces,
        "hlines": hlines,
    })


def ema_group_spread_panel(
    name: str,
    title: str | None = None,
    metrics: tuple[str, ...] = ("range_pct", "std_pct"),
    visible=True,
    dynamic_threshold_cols: list[str] | None = None,
) -> IndicatorSpec:
    """
    Panel for how compressed/expanded a group of EMAs is.

    Good metrics:
      range_pct = (max EMA - min EMA) / mean EMA * 100
      std_pct   = standard deviation of EMA basket / mean EMA * 100
      mad_pct   = mean absolute deviation / mean EMA * 100
    """
    traces = []

    for metric in metrics:
        col = f"{name}__{metric}"
        traces.append({
            "kind": "line",
            "col": col,
            "name": col,
            "width": 2.0,
        })

    for c in dynamic_threshold_cols or []:
        traces.append({
            "kind": "line",
            "col": c,
            "name": c,
            "width": 1.2,
            "dash": "dot",
        })

    return IndicatorSpec("precomputed", tag=f"{name}_group_panel", config={
        "title": title or f"{name} EMA Spread",
        "is_overlay": False,
        "row_weight": 0.70,
        "visible": visible,
        "traces": traces,
        "hlines": [{"y": 0, "dash": "dot", "color": "rgba(0,0,0,0.25)"}],
    })


def ema_cut_through_panel(
    name: str,
    lookback: int = 10,
    title: str | None = None,
    visible=True,
) -> IndicatorSpec:
    """
    Panel for selected EMA cutting through a list of other EMAs.

    Plots:
      rank_pct: 0 to 1, where 1 means leader EMA is above all others.
      above_count_change_N: how many more EMAs the leader is above now vs N bars ago.
      cross_up_events_N: number of pairwise cross-up events in last N bars.
    """
    traces = [
        {
            "kind": "line",
            "col": f"{name}__rank_pct",
            "name": f"{name} rank pct",
            "width": 2.0,
        },
        {
            "kind": "bar",
            "col": f"{name}__above_count_change_{lookback}",
            "name": f"{name} above-count change {lookback}",
        },
        {
            "kind": "bar",
            "col": f"{name}__cross_up_events_{lookback}",
            "name": f"{name} cross-up events {lookback}",
        },
    ]

    return IndicatorSpec("precomputed", tag=f"{name}_cut_panel", config={
        "title": title or f"{name} Cut-Through Speed",
        "is_overlay": False,
        "row_weight": 0.75,
        "visible": visible,
        "traces": traces,
        "hlines": [
            {"y": 0, "dash": "dot", "color": "rgba(0,0,0,0.25)"},
            {"y": 1, "dash": "dot", "color": "rgba(0,0,0,0.25)"},
        ],
    })