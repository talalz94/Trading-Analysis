from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

class Momentum:
    name = "momentum"
    is_overlay = False
    row_weight = 1.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        length = int(cfg.get("length", 10))
        mode = str(cfg.get("mode", "diff")).lower()  # "diff" or "roc"

        out = df.copy()
        mom_c = col(tag, "MOM")
        if mode == "roc":
            out[mom_c] = 100 * (out["close"] / out["close"].shift(length) - 1.0)
        else:
            out[mom_c] = out["close"] - out["close"].shift(length)
        return out, [mom_c]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        mom_c = col(tag, "MOM")
        fig.add_trace(
            go.Scatter(x=df["t"], y=df[mom_c], mode="lines", name=f"{tag}:Momentum",
                       hovertemplate="<b>%{x}</b><br>Momentum: %{y:.6f}<extra></extra>"),
            row=row, col=1
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot", row=row, col=1)

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return f"{tag}:Momentum"