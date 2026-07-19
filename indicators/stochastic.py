from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"
    
class Stochastic:
    name = "stochastic"
    is_overlay = False
    row_weight = 1.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        k_len = int(cfg.get("k_length", 14))
        d_len = int(cfg.get("d_length", 3))
        smooth = int(cfg.get("smooth", 3))

        out = df.copy()
        k_c = col(tag, "K")
        d_c = col(tag, "D")

        low_min = out["low"].rolling(k_len, min_periods=k_len).min()
        high_max = out["high"].rolling(k_len, min_periods=k_len).max()
        k_raw = 100 * (out["close"] - low_min) / (high_max - low_min)
        out[k_c] = k_raw.rolling(smooth, min_periods=smooth).mean()
        out[d_c] = out[k_c].rolling(d_len, min_periods=d_len).mean()
        return out, [k_c, d_c]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        show_levels = bool(cfg.get("show_levels", True))
        k_c = col(tag, "K")
        d_c = col(tag, "D")

        fig.add_trace(
            go.Scatter(x=df["t"], y=df[k_c], mode="lines", name=f"{tag}:%K",
                       hovertemplate="<b>%{x}</b><br>%K: %{y:.2f}<extra></extra>"),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["t"], y=df[d_c], mode="lines", name=f"{tag}:%D",
                       hovertemplate="<b>%{x}</b><br>%D: %{y:.2f}<extra></extra>"),
            row=row, col=1
        )
        if show_levels:
            for lvl in (20, 80):
                fig.add_hline(y=lvl, line_width=1, line_dash="dot", row=row, col=1)

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return f"{tag}:Stoch"