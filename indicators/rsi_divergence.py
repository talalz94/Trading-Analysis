from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

def _rsi(close: pd.Series, length: int) -> pd.Series:
    """
    Wilder RSI with proper warmup:
      - first RSI values are NaN until length bars are available
      - much more stable than EMA-from-start
    """
    close = close.astype(float)
    delta = close.diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    rsi = pd.Series(index=close.index, dtype=float)

    # Need at least length+1 prices to seed
    if len(close) < length + 1:
        return rsi

    # seed using SMA of first length periods (excluding first NaN)
    avg_gain = gain.iloc[1:length+1].mean()
    avg_loss = loss.iloc[1:length+1].mean()

    # first RSI value placed at index length
    if avg_loss == 0:
        rsi.iloc[length] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi.iloc[length] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder smoothing
    for i in range(length + 1, len(close)):
        avg_gain = (avg_gain * (length - 1) + gain.iloc[i]) / length
        avg_loss = (avg_loss * (length - 1) + loss.iloc[i]) / length

        if avg_loss == 0:
            rsi.iloc[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi.iloc[i] = 100.0 - (100.0 / (1.0 + rs))

    # warmup NaNs
    rsi.iloc[:length] = np.nan
    return rsi


def _find_pivots(values: np.ndarray, lb: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values)
    piv_hi = np.zeros(n, dtype=bool)
    piv_lo = np.zeros(n, dtype=bool)
    if n < (2 * lb + 1):
        return piv_hi, piv_lo
    for i in range(lb, n - lb):
        w = values[i - lb:i + lb + 1]
        v = values[i]
        if not np.isfinite(v):
            continue
        if v == np.nanmax(w):
            piv_hi[i] = True
        if v == np.nanmin(w):
            piv_lo[i] = True
    return piv_hi, piv_lo


def _zone_pass(r1: float, r2: float, os_level: float, ob_level: float, zone_mode: str, side: str) -> bool:
    zone_mode = zone_mode.lower().strip()
    if zone_mode == "none":
        return True

    if side == "bull":
        level = os_level
        if zone_mode == "cross":
            return (r1 <= level <= r2) or (r2 <= level <= r1)
        if zone_mode == "touch":
            return min(r1, r2) <= level
        if zone_mode == "both":
            return (r1 <= level) and (r2 <= level)
    else:
        level = ob_level
        if zone_mode == "cross":
            return (r1 >= level >= r2) or (r2 >= level >= r1)
        if zone_mode == "touch":
            return max(r1, r2) >= level
        if zone_mode == "both":
            return (r1 >= level) and (r2 >= level)

    return True


def _divergences(
    close: np.ndarray,
    rsi: np.ndarray,
    lb: int,
    min_rsi_delta: float,
    os_level: float,
    ob_level: float,
    zone_mode: str,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    piv_hi, piv_lo = _find_pivots(close, lb)
    low_idx = np.where(piv_lo)[0]
    high_idx = np.where(piv_hi)[0]

    bullish: List[Tuple[int, int]] = []
    bearish: List[Tuple[int, int]] = []

    for j in range(1, len(low_idx)):
        i1, i2 = low_idx[j - 1], low_idx[j]
        if not (np.isfinite(close[i1]) and np.isfinite(close[i2]) and np.isfinite(rsi[i1]) and np.isfinite(rsi[i2])):
            continue
        if (close[i2] < close[i1]) and (rsi[i2] > rsi[i1] + min_rsi_delta):
            if _zone_pass(rsi[i1], rsi[i2], os_level, ob_level, zone_mode, side="bull"):
                bullish.append((i1, i2))

    for j in range(1, len(high_idx)):
        i1, i2 = high_idx[j - 1], high_idx[j]
        if not (np.isfinite(close[i1]) and np.isfinite(close[i2]) and np.isfinite(rsi[i1]) and np.isfinite(rsi[i2])):
            continue
        if (close[i2] > close[i1]) and (rsi[i2] < rsi[i1] - min_rsi_delta):
            if _zone_pass(rsi[i1], rsi[i2], os_level, ob_level, zone_mode, side="bear"):
                bearish.append((i1, i2))

    return bullish, bearish


def _build_line_series(n: int, i1: int, y1: float, i2: int, y2: float) -> np.ndarray:
    out = np.full(n, np.nan, dtype=float)
    if i2 <= i1:
        return out
    xs = np.arange(i1, i2 + 1)
    ys = y1 + (y2 - y1) * (xs - i1) / (i2 - i1)
    out[i1 : i2 + 1] = ys
    return out


def _marker_offset(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    span = int(cfg.get("marker_offset_window", 14))
    mult = float(cfg.get("marker_offset_mult", 1.35))
    rng = (df["high"] - df["low"]).rolling(span, min_periods=1).mean()
    return (rng * mult).replace(0, np.nan)


def _axis_ref_for_row(row: int) -> Tuple[str, str]:
    suffix = "" if row == 1 else str(row)
    return f"x{suffix}", f"y{suffix}"


class RSI_Divergence:
    name = "rsi_divergence"
    is_overlay = False
    row_weight = 1.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        length = int(cfg.get("length", 14))
        pivot_lb = int(cfg.get("pivot_lookback", 5))
        min_rsi_delta = float(cfg.get("min_rsi_delta", 2.0))

        os_level = float(cfg.get("os_level", 20))
        ob_level = float(cfg.get("ob_level", 80))
        zone_mode = str(cfg.get("zone_mode", "cross"))

        out = df.copy()

        rsi_c = col(tag, "RSI")
        bull_c = col(tag, "BULL_DIV")
        bear_c = col(tag, "BEAR_DIV")
        bull_start = col(tag, "BULL_START_RSI")
        bear_start = col(tag, "BEAR_START_RSI")
        bull_line = col(tag, "BULL_RSI_LINE")
        bear_line = col(tag, "BEAR_RSI_LINE")

        out[rsi_c] = _rsi(out["close"], length)
        out[bull_c] = False
        out[bear_c] = False
        out[bull_start] = np.nan
        out[bear_start] = np.nan
        out[bull_line] = np.nan
        out[bear_line] = np.nan

        c = out["close"].to_numpy(dtype=float)
        r = out[rsi_c].to_numpy(dtype=float)

        bulls, bears = _divergences(c, r, pivot_lb, min_rsi_delta, os_level, ob_level, zone_mode)

        n = len(out)
        bull_line_arr = np.full(n, np.nan, dtype=float)
        bear_line_arr = np.full(n, np.nan, dtype=float)

        for i1, i2 in bulls:
            out.loc[out.index[i2], bull_c] = True
            out.loc[out.index[i2], bull_start] = out.loc[out.index[i1], rsi_c]
            seg = _build_line_series(n, i1, r[i1], i2, r[i2])
            bull_line_arr = np.where(np.isfinite(seg), seg, bull_line_arr)

        for i1, i2 in bears:
            out.loc[out.index[i2], bear_c] = True
            out.loc[out.index[i2], bear_start] = out.loc[out.index[i1], rsi_c]
            seg = _build_line_series(n, i1, r[i1], i2, r[i2])
            bear_line_arr = np.where(np.isfinite(seg), seg, bear_line_arr)

        out[bull_line] = bull_line_arr
        out[bear_line] = bear_line_arr

        return out, [rsi_c, bull_c, bear_c, bull_start, bear_start, bull_line, bear_line]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        # requested default: strong 30/70, light 20/80
        major_levels = cfg.get("major_levels", [30, 70])
        minor_levels = cfg.get("minor_levels", [20, 80])

        show_zone_shading = bool(cfg.get("show_zone_shading", True))
        show_labels = bool(cfg.get("show_div_labels", True))
        max_labels = int(cfg.get("max_div_labels", 8))

        mark_price = bool(cfg.get("mark_price", True))
        price_marker_size = int(cfg.get("price_marker_size", 20))

        label_font_size = int(cfg.get("label_font_size", 10))
        label_xshift = int(cfg.get("label_xshift", 10))
        label_yshift = int(cfg.get("label_yshift", 14))

        rsi_c = col(tag, "RSI")
        bull_c = col(tag, "BULL_DIV")
        bear_c = col(tag, "BEAR_DIV")
        bull_line = col(tag, "BULL_RSI_LINE")
        bear_line = col(tag, "BEAR_RSI_LINE")

        xref, yref = _axis_ref_for_row(row)

        # TradingView-ish bounds
        fig.update_yaxes(range=[0, 100], row=row, col=1, fixedrange=False)

        # Zone shading (based on minor levels, typically 20/80)
        if show_zone_shading and len(minor_levels) >= 2:
            lo = float(min(minor_levels))
            hi = float(max(minor_levels))
            fig.add_hrect(y0=0, y1=lo, row=row, col=1, fillcolor="rgba(0,160,0,0.06)", line_width=0)
            fig.add_hrect(y0=hi, y1=100, row=row, col=1, fillcolor="rgba(200,0,0,0.06)", line_width=0)

        # Major ref lines (30/70) darker
        for lvl in major_levels:
            fig.add_hline(y=float(lvl), row=row, col=1, line_width=2, line_dash="solid", line_color="rgba(0,0,0,0.55)")
        # Minor ref lines (20/80) lighter
        for lvl in minor_levels:
            fig.add_hline(y=float(lvl), row=row, col=1, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.25)")
        # Optional midline
        fig.add_hline(y=50, row=row, col=1, line_width=1, line_dash="dot", line_color="rgba(0,0,0,0.18)")

        # ✅ Continuous RSI base line (prevents the “broken line” look)
        fig.add_trace(
            go.Scatter(
                x=df["t"], y=df[rsi_c],
                mode="lines",
                name=f"{tag}:RSI",
                line=dict(width=2.2, color="rgba(40,70,200,0.9)"),
                hovertemplate="<b>%{x}</b><br>RSI: %{y:.2f}<extra></extra>",
            ),
            row=row, col=1
        )

        # Optional OS/OB colored overlays (do NOT break base line)
        if len(minor_levels) >= 2:
            lo = float(min(minor_levels))
            hi = float(max(minor_levels))
            rsi = df[rsi_c].astype(float)
            rsi_os = rsi.where(rsi <= lo, np.nan)
            rsi_ob = rsi.where(rsi >= hi, np.nan)

            fig.add_trace(
                go.Scatter(
                    x=df["t"], y=rsi_os,
                    mode="lines", showlegend=False,
                    connectgaps=False,
                    line=dict(width=2.6, color="rgba(0,140,0,0.95)"),
                    hovertemplate="<b>%{x}</b><br>RSI(OS): %{y:.2f}<extra></extra>",
                ),
                row=row, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=df["t"], y=rsi_ob,
                    mode="lines", showlegend=False,
                    connectgaps=False,
                    line=dict(width=2.6, color="rgba(200,0,0,0.95)"),
                    hovertemplate="<b>%{x}</b><br>RSI(OB): %{y:.2f}<extra></extra>",
                ),
                row=row, col=1
            )

        # ✅ Divergence lines (neat dashed)
        fig.add_trace(
            go.Scatter(
                x=df["t"], y=df[bull_line],
                mode="lines",
                showlegend=False,
                connectgaps=False,
                line=dict(width=3.2, dash="dash", color="rgba(0,160,0,0.95)"),
                hovertemplate="<b>%{x}</b><br>Bull div (RSI)<extra></extra>",
            ),
            row=row, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df["t"], y=df[bear_line],
                mode="lines",
                showlegend=False,
                connectgaps=False,
                line=dict(width=3.2, dash="dash", color="rgba(200,0,0,0.95)"),
                hovertemplate="<b>%{x}</b><br>Bear div (RSI)<extra></extra>",
            ),
            row=row, col=1
        )

        # Endpoint markers on RSI panel for readability
        bull_idx = df.index[df[bull_c].fillna(False)]
        bear_idx = df.index[df[bear_c].fillna(False)]

        if len(bull_idx):
            fig.add_trace(
                go.Scatter(
                    x=df.loc[bull_idx, "t"], y=df.loc[bull_idx, rsi_c],
                    mode="markers", showlegend=False,
                    marker=dict(symbol="circle", size=7, color="rgba(0,160,0,0.98)", line=dict(width=1)),
                    hovertemplate="<b>%{x}</b><br>Bull divergence<extra></extra>",
                ),
                row=row, col=1
            )
        if len(bear_idx):
            fig.add_trace(
                go.Scatter(
                    x=df.loc[bear_idx, "t"], y=df.loc[bear_idx, rsi_c],
                    mode="markers", showlegend=False,
                    marker=dict(symbol="circle", size=7, color="rgba(200,0,0,0.98)", line=dict(width=1)),
                    hovertemplate="<b>%{x}</b><br>Bear divergence<extra></extra>",
                ),
                row=row, col=1
            )

        # ✅ Labels as annotations (stable placement)
        if show_labels:
            bull_end = df.index[df[bull_line].notna() & df[bull_line].shift(-1).isna()][-max_labels:]
            bear_end = df.index[df[bear_line].notna() & df[bear_line].shift(-1).isna()][-max_labels:]

            for ix in bull_end:
                fig.add_annotation(
                    x=df.loc[ix, "t"], y=float(df.loc[ix, rsi_c]),
                    xref=xref, yref=yref,
                    text="Bull div",
                    showarrow=False,
                    xshift=label_xshift, yshift=label_yshift,
                    font=dict(size=label_font_size, color="rgba(0,120,0,1)"),
                )
            for ix in bear_end:
                fig.add_annotation(
                    x=df.loc[ix, "t"], y=float(df.loc[ix, rsi_c]),
                    xref=xref, yref=yref,
                    text="Bear div",
                    showarrow=False,
                    xshift=label_xshift, yshift=-label_yshift,
                    font=dict(size=label_font_size, color="rgba(160,0,0,1)"),
                )

        # Larger, offset markers on main price chart (unchanged logic, but clearer)
        if mark_price:
            off = _marker_offset(df, cfg)

            if len(bull_idx):
                y = (df.loc[bull_idx, "low"] - off.loc[bull_idx]).astype(float)
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[bull_idx, "t"], y=y,
                        mode="markers",
                        name=f"{tag}:BullDiv",
                        marker=dict(symbol="triangle-up", size=price_marker_size,
                                    color="rgba(0,160,0,0.98)", line=dict(width=2)),
                        hovertemplate="<b>%{x}</b><br>Bullish divergence<extra></extra>",
                    ),
                    row=price_row, col=1
                )

            if len(bear_idx):
                y = (df.loc[bear_idx, "high"] + off.loc[bear_idx]).astype(float)
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[bear_idx, "t"], y=y,
                        mode="markers",
                        name=f"{tag}:BearDiv",
                        marker=dict(symbol="triangle-down", size=price_marker_size,
                                    color="rgba(200,0,0,0.98)", line=dict(width=2)),
                        hovertemplate="<b>%{x}</b><br>Bearish divergence<extra></extra>",
                    ),
                    row=price_row, col=1
                )

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        length = int(cfg.get("length", 14))
        return f"{tag}:RSI({length})"