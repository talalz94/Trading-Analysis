from __future__ import annotations
from typing import Any, Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


class MACD:
    name = "macd"
    is_overlay = False
    row_weight = 1.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        fast = int(cfg.get("fast", 12))
        slow = int(cfg.get("slow", 26))
        signal = int(cfg.get("signal", 9))

        out = df.copy()
        macd_c = col(tag, "MACD")
        sig_c  = col(tag, "SIGNAL")
        hist_c = col(tag, "HIST")

        ema_fast = _ema(out["close"], fast)
        ema_slow = _ema(out["close"], slow)
        out[macd_c] = ema_fast - ema_slow
        out[sig_c] = _ema(out[macd_c], signal)
        out[hist_c] = out[macd_c] - out[sig_c]
        return out, [macd_c, sig_c, hist_c]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        fast = int(cfg.get("fast", 12))
        slow = int(cfg.get("slow", 26))
        signal = int(cfg.get("signal", 9))

        macd_c = col(tag, "MACD")
        sig_c  = col(tag, "SIGNAL")
        hist_c = col(tag, "HIST")

        fig.add_trace(
            go.Bar(x=df["t"], y=df[hist_c], name=f"{tag}:Hist",
                   hovertemplate="<b>%{x}</b><br>Hist: %{y:.6f}<extra></extra>"),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["t"], y=df[macd_c], mode="lines",
                       name=f"{tag}:MACD({fast},{slow})",
                       hovertemplate="<b>%{x}</b><br>MACD: %{y:.6f}<extra></extra>"),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(x=df["t"], y=df[sig_c], mode="lines",
                       name=f"{tag}:Signal({signal})",
                       hovertemplate="<b>%{x}</b><br>Signal: %{y:.6f}<extra></extra>"),
            row=row, col=1
        )

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return f"{tag}:MACD"