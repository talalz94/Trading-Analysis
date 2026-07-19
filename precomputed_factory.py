from __future__ import annotations

from typing import Any
from pipeline import IndicatorSpec


def _p(tf: str, tag: str, col: str) -> str:
    return f"{tf}__{tag}__{col}"


def _periods(cfg: dict) -> list[int]:
    if "periods" in cfg:
        return [int(x) for x in cfg["periods"]]
    return [int(cfg.get("period", 20))]


def _visible(visible):
    """
    Plotly accepts:
      True
      "legendonly"
    We use False to mean do not add traces at all.
    """
    return visible


def precomputed_from_indicator(
    tf: str,
    spec: IndicatorSpec,
    title: str | None = None,
    visible=True,
) -> IndicatorSpec:
    """
    Converts a normal calculation IndicatorSpec into a generic precomputed plot spec.

    This does not recalculate the indicator. It only plots columns that already exist
    in the aligned dataframe.

    visible:
      True          -> shown by default
      "legendonly"  -> included but hidden by default, user can toggle from legend
      False         -> not included
    """
    if visible is False:
        return None  # handled by caller

    name = spec.name
    tag = spec.tag
    cfg = spec.config or {}

    # -------------------------
    # Moving average overlay
    # -------------------------
    if name == "moving_average":
        ma_type = str(cfg.get("type", "sma")).upper()
        periods = _periods(cfg)

        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} {ma_type}",
            "is_overlay": True,
            "row_weight": 0.0,
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "line",
                    "col": _p(tf, tag, f"{ma_type}_{p}"),
                    "name": f"{tf} {ma_type}({p})",
                    "width": 1.5,
                }
                for p in periods
            ],
        })

    # -------------------------
    # MACD
    # -------------------------
    if name == "macd":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} MACD",
            "is_overlay": False,
            "row_weight": 0.85,
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "bar",
                    "col": _p(tf, tag, "HIST"),
                    "name": f"{tf} MACD Hist",
                    "positive_negative_colors": True,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "MACD"),
                    "name": f"{tf} MACD",
                    "width": 1.8,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "SIGNAL"),
                    "name": f"{tf} Signal",
                    "width": 1.5,
                },
            ],
            "hlines": [{"y": 0, "dash": "dot"}],
        })

    # -------------------------
    # RSI divergence
    # -------------------------
    if name == "rsi_divergence":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} RSI",
            "is_overlay": False,
            "row_weight": 0.85,
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "line",
                    "col": _p(tf, tag, "RSI"),
                    "name": f"{tf} RSI",
                    "width": 2.0,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "BULL_RSI_LINE"),
                    "name": f"{tf} Bull Div",
                    "width": 2.5,
                    "dash": "dash",
                    "color": "rgba(0,160,0,0.95)",
                    "connectgaps": False,
                    "optional": True,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "BEAR_RSI_LINE"),
                    "name": f"{tf} Bear Div",
                    "width": 2.5,
                    "dash": "dash",
                    "color": "rgba(200,0,0,0.95)",
                    "connectgaps": False,
                    "optional": True,
                },
            ],
            "hlines": [
                {"y": 20, "dash": "dot", "color": "rgba(0,0,0,0.20)"},
                {"y": 30, "dash": "solid", "color": "rgba(0,0,0,0.45)"},
                {"y": 50, "dash": "dot", "color": "rgba(0,0,0,0.18)"},
                {"y": 70, "dash": "solid", "color": "rgba(0,0,0,0.45)"},
                {"y": 80, "dash": "dot", "color": "rgba(0,0,0,0.20)"},
            ],
        })

    # -------------------------
    # Stochastic
    # -------------------------
    if name == "stochastic":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} Stochastic",
            "is_overlay": False,
            "row_weight": 0.65,
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "line",
                    "col": _p(tf, tag, "K"),
                    "name": f"{tf} Stoch K",
                    "width": 2.0,
                    "optional": True,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "D"),
                    "name": f"{tf} Stoch D",
                    "width": 1.5,
                    "optional": True,
                },
            ],
            "hlines": [
                {"y": 20, "dash": "dot", "color": "rgba(0,0,0,0.35)"},
                {"y": 80, "dash": "dot", "color": "rgba(0,0,0,0.35)"},
                {"y": 50, "dash": "dot", "color": "rgba(0,0,0,0.20)"},
            ],
            "yaxis": {
                "range": [0, 100],
            },
        })

    # -------------------------
    # Momentum
    # -------------------------
    if name == "momentum":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} Momentum",
            "is_overlay": False,
            "row_weight": 0.75,
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "line",
                    "col": _p(tf, tag, "MOM"),
                    "name": f"{tf} Momentum",
                    "width": 1.8,
                },
            ],
            "hlines": [{"y": 0, "dash": "dot"}],
        })

    # -------------------------
    # Supertrend
    # -------------------------
    if name == "supertrend":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} Supertrend",
            "is_overlay": True,
            "row_weight": 0.0,
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "line",
                    "col": _p(tf, tag, "ST_BUY_LINE"),
                    "name": f"{tf} Supertrend Buy Line",
                    "width": 2.3,
                    "optional": True,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "ST_SELL_LINE"),
                    "name": f"{tf} Supertrend Sell Line",
                    "width": 2.3,
                    "optional": True,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "ST"),
                    "name": f"{tf} Supertrend Full Line",
                    "width": 1.4,
                    "optional": True,
                    "visible": "legendonly",
                },
            ],
        })

    # -------------------------
    # EMA Compression / Expansion
    # -------------------------
    if name == "ema_compression":
        """
        Plots EMA compression features created by add_ema_compression_features().

        Expected columns:
          {feature_name}__range_pct
          {feature_name}__compress_thr
          {feature_name}__expand_thr

        Example feature_name:
          ema_15m_50_100_150_comp
        """

        feature_name = str(cfg.get("feature_name", tag))

        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or cfg.get("title", f"{tf} EMA Compression"),
            "is_overlay": False,
            "row_weight": float(cfg.get("row_weight", 0.65)),
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "line",
                    "col": f"{feature_name}__range_pct",
                    "name": cfg.get("range_name", f"{tf} EMA range %"),
                    "width": float(cfg.get("range_width", 2.0)),
                    "optional": True,
                },
                {
                    "kind": "line",
                    "col": f"{feature_name}__compress_thr",
                    "name": cfg.get("compress_name", "Compression threshold"),
                    "width": float(cfg.get("threshold_width", 1.2)),
                    "dash": "dot",
                    "optional": True,
                },
                {
                    "kind": "line",
                    "col": f"{feature_name}__expand_thr",
                    "name": cfg.get("expand_name", "Expansion threshold"),
                    "width": float(cfg.get("threshold_width", 1.2)),
                    "dash": "dash",
                    "optional": True,
                },
            ],
        })
    # -------------------------
    # Market Structure: HH / HL / LH / LL
    # -------------------------
    if name == "market_structure":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} Market Structure",
            "is_overlay": True,
            "row_weight": 0.0,
            "visible": _visible(visible),
            "traces": [
                {
                    "kind": "markers",
                    "col": _p(tf, tag, "HH_PRICE"),
                    "name": f"{tf} HH",
                    "mode": "markers+text",
                    "text": "HH",
                    "marker_symbol": "triangle-up",
                    "marker_size": 11,
                    "marker_color": "rgba(0,160,0,0.95)",
                    "optional": True,
                },
                {
                    "kind": "markers",
                    "col": _p(tf, tag, "HL_PRICE"),
                    "name": f"{tf} HL",
                    "mode": "markers+text",
                    "text": "HL",
                    "marker_symbol": "circle",
                    "marker_size": 9,
                    "marker_color": "rgba(0,120,255,0.95)",
                    "optional": True,
                },
                {
                    "kind": "markers",
                    "col": _p(tf, tag, "LH_PRICE"),
                    "name": f"{tf} LH",
                    "mode": "markers+text",
                    "text": "LH",
                    "marker_symbol": "triangle-down",
                    "marker_size": 11,
                    "marker_color": "rgba(255,140,0,0.95)",
                    "optional": True,
                },
                {
                    "kind": "markers",
                    "col": _p(tf, tag, "LL_PRICE"),
                    "name": f"{tf} LL",
                    "mode": "markers+text",
                    "text": "LL",
                    "marker_symbol": "x",
                    "marker_size": 10,
                    "marker_color": "rgba(200,0,0,0.95)",
                    "optional": True,
                },
            ],
        })
    # -------------------------
    # Support / Resistance
    # -------------------------
    if name == "support_resistance":
        max_levels = int(cfg.get("max_levels", 1))
        plot_zones = bool(cfg.get("plot_zones", True))
    
        traces = []
    
        for k in range(1, max_levels + 1):
            traces.append({
                "kind": "line",
                "col": _p(tf, tag, f"S{k}"),
                "name": f"{tf} Support S{k}",
                "width": 2.4 if k == 1 else 1.4,
                "dash": "solid" if k == 1 else "dot",
                "color": "rgba(0,130,0,0.95)",
                "connectgaps": False,
                "optional": True,
            })
    
            traces.append({
                "kind": "line",
                "col": _p(tf, tag, f"R{k}"),
                "name": f"{tf} Resistance R{k}",
                "width": 2.4 if k == 1 else 1.4,
                "dash": "solid" if k == 1 else "dot",
                "color": "rgba(180,0,0,0.95)",
                "connectgaps": False,
                "optional": True,
            })
    
            if plot_zones:
                traces.extend([
                    {
                        "kind": "line",
                        "col": _p(tf, tag, f"S{k}_ZONE_LOW"),
                        "name": f"{tf} S{k} Zone Low",
                        "width": 0.9,
                        "dash": "dot",
                        "color": "rgba(0,130,0,0.35)",
                        "connectgaps": False,
                        "optional": True,
                        "visible": "legendonly",
                    },
                    {
                        "kind": "line",
                        "col": _p(tf, tag, f"S{k}_ZONE_HIGH"),
                        "name": f"{tf} S{k} Zone High",
                        "width": 0.9,
                        "dash": "dot",
                        "color": "rgba(0,130,0,0.35)",
                        "connectgaps": False,
                        "optional": True,
                        "visible": "legendonly",
                    },
                    {
                        "kind": "line",
                        "col": _p(tf, tag, f"R{k}_ZONE_LOW"),
                        "name": f"{tf} R{k} Zone Low",
                        "width": 0.9,
                        "dash": "dot",
                        "color": "rgba(180,0,0,0.35)",
                        "connectgaps": False,
                        "optional": True,
                        "visible": "legendonly",
                    },
                    {
                        "kind": "line",
                        "col": _p(tf, tag, f"R{k}_ZONE_HIGH"),
                        "name": f"{tf} R{k} Zone High",
                        "width": 0.9,
                        "dash": "dot",
                        "color": "rgba(180,0,0,0.35)",
                        "connectgaps": False,
                        "optional": True,
                        "visible": "legendonly",
                    },
                ])
    
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} Support / Resistance",
            "is_overlay": True,
            "row_weight": 0.0,
            "visible": _visible(visible),
            "traces": traces,
        })
    raise ValueError(
        f"No precomputed plotting preset found for indicator '{name}' with tag '{tag}'. "
        f"Add it to charting/precomputed_factory.py or use IndicatorSpec('precomputed', ...) manually."
    )