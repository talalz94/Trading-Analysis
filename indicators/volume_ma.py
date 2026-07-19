from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"
    
class VolumeMA:
    name = "volume_ma"
    is_overlay = False
    row_weight = 1.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        length = int(cfg.get("ma_length", 20))
        out = df.copy()
        vma_c = col(tag, "VOL_MA")
        out[vma_c] = out["volume"].rolling(length, min_periods=length).mean()
        return out, [vma_c]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        length = int(cfg.get("ma_length", 20))
        vma_c = col(tag, "VOL_MA")

        fig.add_trace(
            go.Bar(x=df["t"], y=df["volume"], name=f"{tag}:Volume",
                   hovertemplate="<b>%{x}</b><br>Volume: %{y:,.4f}<extra></extra>"),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["t"], y=df[vma_c], mode="lines",
                       name=f"{tag}:VolMA({length})",
                       hovertemplate="<b>%{x}</b><br>VolMA: %{y:,.4f}<extra></extra>"),
            row=row, col=1
        )

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return f"{tag}:Volume"