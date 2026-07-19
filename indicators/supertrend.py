from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

def _atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.rolling(length, min_periods=length).mean()


def _supertrend_core(df: pd.DataFrame, length: int, multiplier: float) -> Tuple[pd.Series, pd.Series]:
    atr = _atr(df, length)
    hl2 = (df["high"] + df["low"]) / 2.0

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, len(df)):
        if pd.isna(final_upper.iloc[i-1]) or pd.isna(final_lower.iloc[i-1]):
            continue

        if (upper.iloc[i] < final_upper.iloc[i-1]) or (df["close"].iloc[i-1] > final_upper.iloc[i-1]):
            final_upper.iloc[i] = upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if (lower.iloc[i] > final_lower.iloc[i-1]) or (df["close"].iloc[i-1] < final_lower.iloc[i-1]):
            final_lower.iloc[i] = lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if pd.isna(final_upper.iloc[i]) or pd.isna(final_lower.iloc[i]):
            direction.iloc[i] = direction.iloc[i-1] if i > 0 else 1
            st.iloc[i] = np.nan
            continue

        prev_dir = direction.iloc[i-1] if i > 1 else 1

        if df["close"].iloc[i] > final_upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < final_lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_dir

        st.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return st, direction


def _marker_offset(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    span = int(cfg.get("marker_offset_window", 14))
    mult = float(cfg.get("marker_offset_mult", 1.2))
    rng = (df["high"] - df["low"]).rolling(span, min_periods=1).mean()
    return (rng * mult).replace(0, np.nan)


class Supertrend:
    name = "supertrend"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        length = int(cfg.get("length", 10))
        multiplier = float(cfg.get("multiplier", 3.0))

        out = df.copy()
        st, direction = _supertrend_core(out, length, multiplier)

        st_c = col(tag, "ST")
        dir_c = col(tag, "DIR")
        buy_c = col(tag, "BUY")
        sell_c = col(tag, "SELL")
        st_buy_line = col(tag, "ST_BUY_LINE")
        st_sell_line = col(tag, "ST_SELL_LINE")

        out[st_c] = st
        out[dir_c] = direction

        prev_dir = out[dir_c].shift(1)
        out[buy_c] = (out[dir_c] == 1) & (prev_dir == -1)
        out[sell_c] = (out[dir_c] == -1) & (prev_dir == 1)

        out[st_buy_line] = out[st_c].where(out[dir_c] == 1, np.nan)
        out[st_sell_line] = out[st_c].where(out[dir_c] == -1, np.nan)

        return out, [st_c, dir_c, buy_c, sell_c, st_buy_line, st_sell_line]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        length = int(cfg.get("length", 10))
        multiplier = float(cfg.get("multiplier", 3.0))

        st_buy_line = col(tag, "ST_BUY_LINE")
        st_sell_line = col(tag, "ST_SELL_LINE")
        buy_c = col(tag, "BUY")
        sell_c = col(tag, "SELL")

        # Two colored lines
        fig.add_trace(
            go.Scatter(
                x=df["t"], y=df[st_buy_line],
                mode="lines",
                name=f"ST Up({length},{multiplier:g})",
                line=dict(color="green", width=2.5),
                hovertemplate="<b>%{x}</b><br>ST Up: %{y:.6f}<extra></extra>",
            ),
            row=price_row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["t"], y=df[st_sell_line],
                mode="lines",
                name=f"ST Down({length},{multiplier:g})",
                line=dict(color="red", width=2.5),
                hovertemplate="<b>%{x}</b><br>ST Down: %{y:.6f}<extra></extra>",
            ),
            row=price_row, col=1
        )

        # Bigger, offset markers
        if bool(cfg.get("show_markers", True)):
            marker_size = int(cfg.get("marker_size", 18))
            off = _marker_offset(df, cfg)

            buy_idx = df.index[df[buy_c].fillna(False)]
            sell_idx = df.index[df[sell_c].fillna(False)]

            if len(buy_idx):
                y = (df.loc[buy_idx, "low"] - off.loc[buy_idx]).astype(float)
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[buy_idx, "t"],
                        y=y,
                        mode="markers",
                        name="ST Buy",
                        marker=dict(symbol="triangle-up", size=marker_size, color="green", line=dict(width=2)),
                        hovertemplate="<b>%{x}</b><br>Supertrend BUY<extra></extra>",
                    ),
                    row=price_row, col=1
                )

            if len(sell_idx):
                y = (df.loc[sell_idx, "high"] + off.loc[sell_idx]).astype(float)
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[sell_idx, "t"],
                        y=y,
                        mode="markers",
                        name="ST Sell",
                        marker=dict(symbol="triangle-down", size=marker_size, color="red", line=dict(width=2)),
                        hovertemplate="<b>%{x}</b><br>Supertrend SELL<extra></extra>",
                    ),
                    row=price_row, col=1
                )

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return ""