from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

class BollingerBands:
    name = "bollinger_bands"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        length = int(cfg.get("length", 20))
        stdev = float(cfg.get("stdev", 2.0))

        out = df.copy()
        mid_c = col(tag, "MID")
        up_c  = col(tag, "UP")
        lo_c  = col(tag, "LO")

        mid = out["close"].rolling(length, min_periods=length).mean()
        sd = out["close"].rolling(length, min_periods=length).std()
        out[mid_c] = mid
        out[up_c] = mid + stdev * sd
        out[lo_c] = mid - stdev * sd
        return out, [mid_c, up_c, lo_c]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        length = int(cfg.get("length", 20))
        stdev = float(cfg.get("stdev", 2.0))
        fill = bool(cfg.get("fill", True))

        mid_c = col(tag, "MID")
        up_c  = col(tag, "UP")
        lo_c  = col(tag, "LO")

        fig.add_trace(go.Scatter(x=df["t"], y=df[up_c], mode="lines", name=f"{tag}:BBUp({length},{stdev:g})",
                                 hovertemplate="<b>%{x}</b><br>BB Up: %{y:.6f}<extra></extra>"),
                      row=price_row, col=1)
        fig.add_trace(go.Scatter(x=df["t"], y=df[lo_c], mode="lines", name=f"{tag}:BBLo({length},{stdev:g})",
                                 hovertemplate="<b>%{x}</b><br>BB Lo: %{y:.6f}<extra></extra>"),
                      row=price_row, col=1)
        fig.add_trace(go.Scatter(x=df["t"], y=df[mid_c], mode="lines", name=f"{tag}:BBMid({length})",
                                 hovertemplate="<b>%{x}</b><br>BB Mid: %{y:.6f}<extra></extra>"),
                      row=price_row, col=1)

        if fill:
            fig.add_trace(go.Scatter(x=df["t"], y=df[up_c], mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"),
                          row=price_row, col=1)
            fig.add_trace(go.Scatter(x=df["t"], y=df[lo_c], mode="lines", fill="tonexty", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"),
                          row=price_row, col=1)

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return ""