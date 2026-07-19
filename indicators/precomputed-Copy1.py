from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Precomputed:
    """
    Generic plotter for already-computed/aligned indicator columns.

    It does not calculate anything. It only plots columns already present
    in the dataframe, such as:
      5m__macd__MACD
      5m__rsi14__RSI
      15m__st14__K
      15m__st14__D

    Config format:
      {
        "title": "5m MACD",
        "is_overlay": False,
        "row_weight": 0.85,
        "traces": [
          {"kind": "bar", "col": "5m__macd__HIST", "name": "5m MACD Hist", "positive_negative_colors": True},
          {"kind": "line", "col": "5m__macd__MACD", "name": "5m MACD"},
          {"kind": "line", "col": "5m__macd__SIGNAL", "name": "5m Signal"},
        ],
        "hlines": [
          {"y": 0, "dash": "dot"}
        ]
      }
    """

    name = "precomputed"

    # Actual overlay-ness is config-driven.
    is_overlay = False
    row_weight = 0.85

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        out = df.copy()

        traces = cfg.get("traces", [])
        cols = [t.get("col") for t in traces if t.get("col")]

        missing = [c for c in cols if c not in out.columns]
        if missing:
            available_hint = []
            for m in missing:
                root = m.split("__")[0]
                available_hint.extend([c for c in out.columns if root in c])
            raise KeyError(
                f"Precomputed indicator '{tag}' missing columns: {missing}. "
                f"Similar available columns: {sorted(set(available_hint))[:50]}"
            )

        return out, cols

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        traces = cfg.get("traces", [])

        target_row = price_row if bool(cfg.get("is_overlay", False)) else row

        for tr in traces:
            kind = str(tr.get("kind", "line")).lower().strip()
            c = tr["col"]
            name = tr.get("name", c)
            visible = tr.get("visible", True)

            hovertemplate = tr.get(
                "hovertemplate",
                f"<b>%{{x}}</b><br>{name}: %{{y:.6f}}<extra></extra>"
            )

            if kind == "bar":
                marker_color = tr.get("color", None)

                if bool(tr.get("positive_negative_colors", False)):
                    vals = pd.to_numeric(df[c], errors="coerce")
                    pos_color = tr.get("positive_color", "rgba(0,150,0,0.70)")
                    neg_color = tr.get("negative_color", "rgba(200,0,0,0.70)")
                    marker_color = np.where(vals >= 0, pos_color, neg_color)

                fig.add_trace(
                    go.Bar(
                        x=df["t"],
                        y=df[c],
                        name=name,
                        marker_color=marker_color,
                        opacity=tr.get("opacity", 1.0),
                        visible=visible,
                        hovertemplate=hovertemplate,
                    ),
                    row=target_row,
                    col=1,
                )

            elif kind == "line":
                fig.add_trace(
                    go.Scatter(
                        x=df["t"],
                        y=df[c],
                        mode="lines",
                        name=name,
                        line=dict(
                            width=tr.get("width", 1.8),
                            dash=tr.get("dash", "solid"),
                            color=tr.get("color", None),
                        ),
                        opacity=tr.get("opacity", 1.0),
                        visible=visible,
                        hovertemplate=hovertemplate,
                        connectgaps=bool(tr.get("connectgaps", False)),
                    ),
                    row=target_row,
                    col=1,
                )

            elif kind == "marker":
                fig.add_trace(
                    go.Scatter(
                        x=df["t"],
                        y=df[c],
                        mode="markers",
                        name=name,
                        marker=dict(
                            size=tr.get("size", 8),
                            symbol=tr.get("symbol", "circle"),
                            color=tr.get("color", None),
                        ),
                        visible=visible,
                        hovertemplate=hovertemplate,
                    ),
                    row=target_row,
                    col=1,
                )

            else:
                raise ValueError(f"Unsupported precomputed trace kind: {kind}")

        for h in cfg.get("hlines", []):
            fig.add_hline(
                y=h.get("y", 0),
                row=target_row,
                col=1,
                line_width=h.get("width", 1),
                line_dash=h.get("dash", "dot"),
                line_color=h.get("color", "rgba(0,0,0,0.35)"),
            )

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return cfg.get("title", tag)