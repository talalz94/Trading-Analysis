from __future__ import annotations

from typing import Dict, Any
from pipeline import IndicatorSpec


def _p(tf: str, tag: str, col: str) -> str:
    return f"{tf}__{tag}__{col}"


def precomputed_from_indicator(
    tf: str,
    spec: IndicatorSpec,
    title: str | None = None,
) -> IndicatorSpec:
    """
    Converts a normal indicator spec into a generic precomputed plot spec.

    Example:
      precomputed_from_indicator(
          "5m",
          IndicatorSpec("macd", tag="macd", config={"fast": 12, "slow": 26, "signal": 9})
      )

    This expects the indicator columns to already exist in the aligned dataframe,
    e.g. from align_timeframes(...).
    """

    name = spec.name
    tag = spec.tag
    cfg = spec.config or {}

    label = title or f"{tf} {tag}"

    # -------------------------
    # MACD
    # -------------------------
    if name == "macd":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} MACD",
            "is_overlay": False,
            "row_weight": 0.85,
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
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "BEAR_RSI_LINE"),
                    "name": f"{tf} Bear Div",
                    "width": 2.5,
                    "dash": "dash",
                    "color": "rgba(200,0,0,0.95)",
                    "connectgaps": False,
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
            "row_weight": 0.85,
            "traces": [
                {
                    "kind": "line",
                    "col": _p(tf, tag, "K"),
                    "name": f"{tf} %K",
                    "width": 1.8,
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, "D"),
                    "name": f"{tf} %D",
                    "width": 1.5,
                },
            ],
            "hlines": [
                {"y": 20, "dash": "dot", "color": "rgba(0,0,0,0.25)"},
                {"y": 80, "dash": "dot", "color": "rgba(0,0,0,0.25)"},
            ],
        })

    # -------------------------
    # Volume MA
    # -------------------------
    if name == "volume_ma":
        ma_len = cfg.get("ma_length", cfg.get("length", 20))
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} Volume",
            "is_overlay": False,
            "row_weight": 0.75,
            "traces": [
                {
                    "kind": "bar",
                    "col": "volume" if tf in ("1m", "base") else _p(tf, tag, "VOLUME"),
                    "name": f"{tf} Volume",
                },
                {
                    "kind": "line",
                    "col": _p(tf, tag, f"VOL_MA_{ma_len}"),
                    "name": f"{tf} Vol MA({ma_len})",
                    "width": 1.8,
                },
            ],
        })

    # -------------------------
    # Momentum
    # -------------------------
    if name == "momentum":
        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} Momentum",
            "is_overlay": False,
            "row_weight": 0.75,
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
    # Moving average overlay
    # -------------------------
    if name == "moving_average":
        ma_type = str(cfg.get("type", "sma")).upper()
        periods = cfg.get("periods", [cfg.get("period", 20)])

        return IndicatorSpec("precomputed", tag=f"{tf}_{tag}_pre", config={
            "title": title or f"{tf} {ma_type}",
            "is_overlay": True,
            "row_weight": 0.0,
            "traces": [
                {
                    "kind": "line",
                    "col": _p(tf, tag, f"{ma_type}_{int(p)}"),
                    "name": f"{tf} {ma_type}({int(p)})",
                    "width": 1.8,
                }
                for p in periods
            ],
        })

    raise ValueError(
        f"No precomputed plotting preset found for indicator '{name}'. "
        f"Either add a preset in precomputed_factory.py or use IndicatorSpec('precomputed', ...) manually."
    )