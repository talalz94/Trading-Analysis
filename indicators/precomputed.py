from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class Precomputed:
    """
    Generic plotter for already-computed/aligned indicator columns.

    This lets you plot any indicator from any timeframe without recalculating it.

    Supports:
      visible=True          -> shown by default
      visible=False         -> not shown
      visible="legendonly"  -> hidden by default, user can toggle from legend

    Supported trace kinds:
      - "line"
      - "bar"
      - "marker" / "markers"
    """

    name = "precomputed"
    is_overlay = False
    row_weight = 0.85

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        out = df.copy()

        traces = cfg.get("traces", [])
        cols = []
        missing_required = []

        for tr in traces:
            c = tr.get("col")
            if not c:
                continue

            optional = bool(tr.get("optional", False))

            if c not in out.columns:
                if optional:
                    continue
                missing_required.append(c)
            else:
                cols.append(c)

        if missing_required:
            available_hint = []
            for m in missing_required:
                root = m.split("__")[0]
                available_hint.extend([c for c in out.columns if root in c])

            raise KeyError(
                f"Precomputed indicator '{tag}' missing required columns: {missing_required}. "
                f"Similar available columns: {sorted(set(available_hint))[:80]}"
            )

        return out, cols

    @staticmethod
    def add_traces(
        fig: go.Figure,
        df: pd.DataFrame,
        cfg: Dict[str, Any],
        tag: str,
        row: int,
        price_row: int,
    ) -> None:
        traces = cfg.get("traces", [])

        target_row = price_row if bool(cfg.get("is_overlay", False)) else row
        default_visible = cfg.get("visible", True)

        for tr in traces:
            c = tr.get("col")
            if not c or c not in df.columns:
                if bool(tr.get("optional", False)):
                    continue
                raise KeyError(f"Precomputed trace column '{c}' not found for '{tag}'.")

            kind = str(tr.get("kind", "line")).lower().strip()
            name = tr.get("name", c)
            visible = tr.get("visible", default_visible)

            # Allow visible=False to remove a trace fully.
            # Plotly accepts True and "legendonly".
            if visible is False:
                continue

            hovertemplate = tr.get(
                "hovertemplate",
                f"<b>%{{x}}</b><br>{name}: %{{y:.6f}}<extra></extra>",
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

            elif kind in ("marker", "markers"):
                # Supports both the old config names:
                #   size, symbol, color
                # and the new market-structure config names:
                #   marker_size, marker_symbol, marker_color
                marker = {
                    "size": tr.get("marker_size", tr.get("size", 8)),
                    "symbol": tr.get("marker_symbol", tr.get("symbol", "circle")),
                }

                marker_color = tr.get("marker_color", tr.get("color", None))
                if marker_color is not None:
                    marker["color"] = marker_color

                fig.add_trace(
                    go.Scatter(
                        x=df["t"],
                        y=df[c],
                        mode=tr.get("mode", "markers"),
                        name=name,
                        text=tr.get("text", None),
                        textposition=tr.get("textposition", "top center"),
                        marker=marker,
                        opacity=tr.get("opacity", 1.0),
                        visible=visible,
                        hovertemplate=hovertemplate,
                        connectgaps=bool(tr.get("connectgaps", False)),
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

        yaxis = cfg.get("yaxis")
        if isinstance(yaxis, dict) and not bool(cfg.get("is_overlay", False)):
            if "range" in yaxis:
                fig.update_yaxes(range=yaxis["range"], row=target_row, col=1)

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return cfg.get("title", tag)
