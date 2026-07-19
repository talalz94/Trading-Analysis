from __future__ import annotations

from typing import Any, Dict, List, Tuple
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

class MovingAverage:
    """
    Moving average overlay indicator.

    Config examples:
      {"type": "sma", "period": 50}
      {"type": "ema", "period": 21}
      {"type": "ema", "periods": [20, 50, 200]}
      {"type": "ema", "period": 50, "source": "close"}

    Output columns:
      tag__SMA_50
      tag__EMA_21
      etc.
    """
    name = "moving_average"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def _periods(cfg: Dict[str, Any]) -> List[int]:
        if "periods" in cfg:
            return [int(x) for x in cfg["periods"]]
        return [int(cfg.get("period", 20))]

    @staticmethod
    def _ma_col(tag: str, ma_type: str, period: int) -> str:
        return col(tag, f"{ma_type.upper()}_{period}")

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        out = df.copy()

        ma_type = str(cfg.get("type", "sma")).lower().strip()
        source = str(cfg.get("source", "close")).strip()
        periods = MovingAverage._periods(cfg)

        if source not in out.columns:
            raise KeyError(f"MovingAverage source column '{source}' not found in df.")

        created: List[str] = []

        for p in periods:
            min_periods = int(cfg.get("min_periods", p))
            c = MovingAverage._ma_col(tag, ma_type, p)

            if ma_type in ("sma", "simple"):
                out[c] = out[source].rolling(window=p, min_periods=min_periods).mean()

            elif ma_type in ("ema", "exponential"):
                adjust = bool(cfg.get("adjust", False))
                out[c] = out[source].ewm(span=p, adjust=adjust, min_periods=min_periods).mean()

            else:
                raise ValueError("moving_average config 'type' must be one of: 'sma', 'ema'.")

            created.append(c)

        return out, created

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        ma_type = str(cfg.get("type", "sma")).lower().strip()
        periods = MovingAverage._periods(cfg)
        source = str(cfg.get("source", "close")).strip()

        for p in periods:
            c = MovingAverage._ma_col(tag, ma_type, p)
            label = f"{tag}:{ma_type.upper()}({p})"

            fig.add_trace(
                go.Scatter(
                    x=df["t"],
                    y=df[c],
                    mode="lines",
                    name=label,
                    hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.6f}}<extra></extra>",
                ),
                row=price_row,
                col=1,
            )

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return ""