from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from enum import Enum


def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

class ChannelType(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass
class Channel:
    upper_slope: float
    upper_intercept: float
    lower_slope: float
    lower_intercept: float
    start_idx: int
    end_idx: int
    channel_type: ChannelType
    timeframe: str
    confidence: float
    is_active: bool = True
    broken_at: Optional[int] = None
    break_direction: Optional[str] = None  # 'upper' or 'lower'

    def upper(self, idx: np.ndarray) -> np.ndarray:
        return self.upper_slope * idx + self.upper_intercept

    def lower(self, idx: np.ndarray) -> np.ndarray:
        return self.lower_slope * idx + self.lower_intercept

    def mid(self, idx: np.ndarray) -> np.ndarray:
        return (self.upper(idx) + self.lower(idx)) / 2.0


class AutoTrendChannelsDetector:
    """
    Adapted from the uploaded implementation (swing points -> channel fit -> scoring -> dynamic updates). :contentReference[oaicite:1]{index=1}
    """

    def __init__(
        self,
        min_history_bars: int = 50,
        swing_detection_window: int = 5,
        max_channel_angle: float = 85.0,
        min_touches: int = 2,
        touch_tolerance: float = 0.015,
        enable_dynamic_updates: bool = True,
        max_points: int = 10,  # limit recent swing points for performance
    ):
        self.min_history_bars = int(min_history_bars)
        self.swing_detection_window = int(swing_detection_window)
        self.max_channel_angle = float(max_channel_angle)
        self.min_touches = int(min_touches)
        self.touch_tolerance = float(touch_tolerance)
        self.enable_dynamic_updates = bool(enable_dynamic_updates)
        self.max_points = int(max_points)

    def find_swing_highs(self, high: np.ndarray, window: int) -> np.ndarray:
        swing_highs = np.zeros(len(high), dtype=bool)
        for i in range(window, len(high) - window):
            is_high = True
            for j in range(-window, window + 1):
                if j == 0:
                    continue
                if high[i] <= high[i + j]:
                    is_high = False
                    break
            swing_highs[i] = is_high
        return swing_highs

    def find_swing_lows(self, low: np.ndarray, window: int) -> np.ndarray:
        swing_lows = np.zeros(len(low), dtype=bool)
        for i in range(window, len(low) - window):
            is_low = True
            for j in range(-window, window + 1):
                if j == 0:
                    continue
                if low[i] >= low[i + j]:
                    is_low = False
                    break
            swing_lows[i] = is_low
        return swing_lows

    def fit_line_through_points(self, x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        if x1 == x2:
            return 0.0, y1
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    def find_parallel_intercept(self, slope: float, indices: np.ndarray, values: np.ndarray, is_upper: bool) -> float:
        intercepts = values - slope * indices
        return float(np.max(intercepts) if is_upper else np.min(intercepts))

    def score_channel(
        self,
        upper_slope: float,
        upper_intercept: float,
        lower_slope: float,
        lower_intercept: float,
        high: np.ndarray,
        low: np.ndarray,
        swing_high_idx: np.ndarray,
        swing_high_val: np.ndarray,
        swing_low_idx: np.ndarray,
        swing_low_val: np.ndarray,
        start_idx: int,
        end_idx: int,
    ) -> Tuple[float, int]:
        indices = np.arange(start_idx, min(end_idx + 1, len(high)))
        if len(indices) == 0:
            return -np.inf, 0

        upper_line = upper_slope * indices + upper_intercept
        lower_line = lower_slope * indices + lower_intercept

        high_slice = high[start_idx : min(end_idx + 1, len(high))]
        low_slice = low[start_idx : min(end_idx + 1, len(low))]

        contained = np.sum(
            (low_slice >= lower_line * (1 - self.touch_tolerance))
            & (high_slice <= upper_line * (1 + self.touch_tolerance))
        )
        containment_ratio = contained / len(indices)

        valid_highs = (swing_high_idx >= start_idx) & (swing_high_idx <= end_idx)
        valid_lows = (swing_low_idx >= start_idx) & (swing_low_idx <= end_idx)

        fhi_i = swing_high_idx[valid_highs]
        fhi_v = swing_high_val[valid_highs]
        flo_i = swing_low_idx[valid_lows]
        flo_v = swing_low_val[valid_lows]

        upper_touches = 0
        if len(fhi_i) > 0:
            pred = upper_slope * fhi_i + upper_intercept
            dist = np.abs(fhi_v - pred)
            upper_touches = int(np.sum(dist <= (fhi_v * self.touch_tolerance)))

        lower_touches = 0
        if len(flo_i) > 0:
            pred = lower_slope * flo_i + lower_intercept
            dist = np.abs(flo_v - pred)
            lower_touches = int(np.sum(dist <= (flo_v * self.touch_tolerance)))

        touches = upper_touches + lower_touches
        score = containment_ratio * 100.0 + touches * 20.0
        return float(score), int(touches)

    def detect_channel_in_range(
        self,
        high: np.ndarray,
        low: np.ndarray,
        start_idx: int,
        end_idx: int,
        timeframe: str = "current",
    ) -> Optional[Channel]:
        if end_idx - start_idx < self.min_history_bars:
            return None

        swing_highs = self.find_swing_highs(high, self.swing_detection_window)
        swing_lows = self.find_swing_lows(low, self.swing_detection_window)

        swing_high_idx = np.where(swing_highs)[0]
        swing_low_idx = np.where(swing_lows)[0]

        swing_high_idx = swing_high_idx[(swing_high_idx >= start_idx) & (swing_high_idx <= end_idx)]
        swing_low_idx = swing_low_idx[(swing_low_idx >= start_idx) & (swing_low_idx <= end_idx)]

        if len(swing_high_idx) < 2 or len(swing_low_idx) < 2:
            return None

        swing_high_val = high[swing_high_idx]
        swing_low_val = low[swing_low_idx]

        # Limit to recent points for performance
        maxp = self.max_points
        recent_lows_idx = swing_low_idx[-maxp:] if len(swing_low_idx) > maxp else swing_low_idx
        recent_lows_val = swing_low_val[-maxp:] if len(swing_low_val) > maxp else swing_low_val
        recent_highs_idx = swing_high_idx[-maxp:] if len(swing_high_idx) > maxp else swing_high_idx
        recent_highs_val = swing_high_val[-maxp:] if len(swing_high_val) > maxp else swing_high_val

        best_score = -np.inf
        best_channel: Optional[Channel] = None

        # Ascending: draw through lows, parallel upper touches highs
        for i in range(len(recent_lows_idx) - 1):
            for j in range(i + 1, len(recent_lows_idx)):
                slope, lower_intercept = self.fit_line_through_points(
                    float(recent_lows_idx[i]), float(recent_lows_val[i]),
                    float(recent_lows_idx[j]), float(recent_lows_val[j]),
                )
                angle = abs(np.degrees(np.arctan(slope)))
                if slope <= 0 or angle > self.max_channel_angle:
                    continue

                upper_intercept = self.find_parallel_intercept(
                    slope, swing_high_idx.astype(float), swing_high_val.astype(float), is_upper=True
                )

                score, touches = self.score_channel(
                    slope, upper_intercept, slope, lower_intercept,
                    high, low,
                    swing_high_idx, swing_high_val,
                    swing_low_idx, swing_low_val,
                    start_idx, end_idx,
                )
                score += abs(slope) * 10.0

                if touches >= self.min_touches and score > best_score:
                    best_score = score
                    best_channel = Channel(
                        upper_slope=slope,
                        upper_intercept=upper_intercept,
                        lower_slope=slope,
                        lower_intercept=lower_intercept,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        channel_type=ChannelType.ASCENDING,
                        timeframe=timeframe,
                        confidence=touches / max(1, (len(swing_high_idx) + len(swing_low_idx))),
                    )

        # Descending: draw through highs, parallel lower touches lows
        for i in range(len(recent_highs_idx) - 1):
            for j in range(i + 1, len(recent_highs_idx)):
                slope, upper_intercept = self.fit_line_through_points(
                    float(recent_highs_idx[i]), float(recent_highs_val[i]),
                    float(recent_highs_idx[j]), float(recent_highs_val[j]),
                )
                angle = abs(np.degrees(np.arctan(slope)))
                if slope >= 0 or angle > self.max_channel_angle:
                    continue

                lower_intercept = self.find_parallel_intercept(
                    slope, swing_low_idx.astype(float), swing_low_val.astype(float), is_upper=False
                )

                score, touches = self.score_channel(
                    slope, upper_intercept, slope, lower_intercept,
                    high, low,
                    swing_high_idx, swing_high_val,
                    swing_low_idx, swing_low_val,
                    start_idx, end_idx,
                )
                score += abs(slope) * 10.0

                if touches >= self.min_touches and score > best_score:
                    best_score = score
                    best_channel = Channel(
                        upper_slope=slope,
                        upper_intercept=upper_intercept,
                        lower_slope=slope,
                        lower_intercept=lower_intercept,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        channel_type=ChannelType.DESCENDING,
                        timeframe=timeframe,
                        confidence=touches / max(1, (len(swing_high_idx) + len(swing_low_idx))),
                    )

        return best_channel

    def check_breakout(self, ch: Channel, high: np.ndarray, low: np.ndarray, idx: int) -> Tuple[bool, Optional[str]]:
        if idx >= len(high) or idx >= len(low):
            return False, None
        upper_v = ch.upper(np.array([idx], dtype=float))[0]
        lower_v = ch.lower(np.array([idx], dtype=float))[0]
        if high[idx] > upper_v * (1 + self.touch_tolerance):
            return True, "upper"
        if low[idx] < lower_v * (1 - self.touch_tolerance):
            return True, "lower"
        return False, None

    def detect_all_channels(self, high: np.ndarray, low: np.ndarray, timeframe: str = "current") -> List[Channel]:
        channels: List[Channel] = []
        search_start = 0

        while search_start < len(high) - self.min_history_bars:
            ch = self.detect_channel_in_range(high, low, search_start, len(high) - 1, timeframe)
            if ch is None:
                search_start += max(1, self.min_history_bars // 2)
                continue

            if self.enable_dynamic_updates:
                broke = False
                for idx in range(ch.start_idx + self.min_history_bars, len(high)):
                    is_broken, direction = self.check_breakout(ch, high, low, idx)
                    if is_broken:
                        ch.end_idx = idx
                        ch.is_active = False
                        ch.broken_at = idx
                        ch.break_direction = direction
                        channels.append(ch)
                        search_start = idx
                        broke = True
                        break
                if not broke:
                    channels.append(ch)
                    break
            else:
                channels.append(ch)
                break

        # mark last channel active if it reaches the end
        if channels:
            # If last was set inactive by breakout, the next loop would have found another or stopped.
            # In practice, final channel (if any) should be active to the end:
            channels[-1].is_active = True
            channels[-1].broken_at = None
            channels[-1].break_direction = None
            channels[-1].end_idx = min(channels[-1].end_idx, len(high) - 1)

        return channels


class TrendChannels:
    """
    Auto Trend Channels (FXSSI-style) adapted to the indicator framework. :contentReference[oaicite:2]{index=2}
    Produces piecewise channel lines over time + active-channel columns + breakout events.
    """
    name = "trend_channels"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        out = df.copy()

        timeframe = str(cfg.get("timeframe", "current"))

        det = AutoTrendChannelsDetector(
            min_history_bars=int(cfg.get("min_history_bars", 50)),
            swing_detection_window=int(cfg.get("swing_detection_window", 5)),
            max_channel_angle=float(cfg.get("max_channel_angle", 85.0)),
            min_touches=int(cfg.get("min_touches", 2)),
            touch_tolerance=float(cfg.get("touch_tolerance", 0.015)),
            enable_dynamic_updates=bool(cfg.get("enable_dynamic_updates", True)),
            max_points=int(cfg.get("max_points", 10)),
        )

        high = out["high"].to_numpy(dtype=float)
        low = out["low"].to_numpy(dtype=float)
        close = out["close"].to_numpy(dtype=float)

        channels = det.detect_all_channels(high, low, timeframe=timeframe)

        # columns
        up = col(tag, "UP_LINE")
        lo = col(tag, "LO_LINE")
        mid = col(tag, "MID_LINE")
        width = col(tag, "WIDTH")
        slope = col(tag, "SLOPE")
        up_int = col(tag, "UP_INTERCEPT")
        lo_int = col(tag, "LO_INTERCEPT")
        conf = col(tag, "CONFIDENCE")
        ch_id = col(tag, "CHANNEL_ID")
        direction = col(tag, "DIR")  # 1 asc, -1 desc, 0 none
        active = col(tag, "IS_ACTIVE")

        up_a = col(tag, "UP_ACTIVE")
        lo_a = col(tag, "LO_ACTIVE")
        mid_a = col(tag, "MID_ACTIVE")

        brk_evt = col(tag, "BREAK_EVENT")  # 1 upper, -1 lower
        brk_up = col(tag, "BREAK_UP")      # per bar
        brk_dn = col(tag, "BREAK_DN")      # per bar

        # init
        n = len(out)
        for c in [up, lo, mid, width, slope, up_int, lo_int, conf, ch_id, direction, up_a, lo_a, mid_a]:
            out[c] = np.nan
        out[active] = False
        out[brk_evt] = 0
        out[brk_up] = False
        out[brk_dn] = False

        tol = float(cfg.get("touch_tolerance", 0.015))

        # fill segments
        for k, ch in enumerate(channels, start=1):
            s = max(0, int(ch.start_idx))
            e = min(n - 1, int(ch.end_idx))
            if e <= s:
                continue

            idx = np.arange(s, e + 1, dtype=float)
            u = ch.upper(idx)
            l = ch.lower(idx)
            m = (u + l) / 2.0

            out.loc[out.index[s:e+1], up] = u
            out.loc[out.index[s:e+1], lo] = l
            out.loc[out.index[s:e+1], mid] = m
            out.loc[out.index[s:e+1], width] = (u - l)
            out.loc[out.index[s:e+1], slope] = ch.upper_slope
            out.loc[out.index[s:e+1], up_int] = ch.upper_intercept
            out.loc[out.index[s:e+1], lo_int] = ch.lower_intercept
            out.loc[out.index[s:e+1], conf] = ch.confidence
            out.loc[out.index[s:e+1], ch_id] = k
            out.loc[out.index[s:e+1], direction] = (1 if ch.channel_type == ChannelType.ASCENDING else -1)

            # bar-level breakout flags (within segment)
            seg_i = np.arange(s, e + 1, dtype=int)
            out.loc[out.index[s:e+1], brk_up] = high[seg_i] > (u * (1 + tol))
            out.loc[out.index[s:e+1], brk_dn] = low[seg_i] < (l * (1 - tol))

            if ch.is_active:
                out.loc[out.index[s:e+1], active] = True
                out.loc[out.index[s:e+1], up_a] = u
                out.loc[out.index[s:e+1], lo_a] = l
                out.loc[out.index[s:e+1], mid_a] = m

            if ch.broken_at is not None and 0 <= ch.broken_at < n:
                out.loc[out.index[ch.broken_at], brk_evt] = (1 if ch.break_direction == "upper" else -1)

        created = [
            up, lo, mid, width, slope, up_int, lo_int, conf, ch_id, direction, active,
            up_a, lo_a, mid_a,
            brk_evt, brk_up, brk_dn
        ]
        return out, created

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        up = col(tag, "UP_LINE")
        lo = col(tag, "LO_LINE")
        mid = col(tag, "MID_LINE")
        up_a = col(tag, "UP_ACTIVE")
        lo_a = col(tag, "LO_ACTIVE")
        mid_a = col(tag, "MID_ACTIVE")
        brk_evt = col(tag, "BREAK_EVENT")
        slope = col(tag, "SLOPE")
        up_int = col(tag, "UP_INTERCEPT")
        lo_int = col(tag, "LO_INTERCEPT")

        show_history = bool(cfg.get("show_history", True))
        show_mid = bool(cfg.get("show_mid", True))
        show_fill = bool(cfg.get("fill_active", True))
        show_breakouts = bool(cfg.get("show_breakout_markers", True))
        marker_size = int(cfg.get("breakout_marker_size", 14))
        projection_bars = int(cfg.get("projection_bars", 0))

        # History (thin)
        if show_history and up in df.columns and lo in df.columns:
            fig.add_trace(go.Scatter(
                x=df["t"], y=df[up], mode="lines",
                name=f"{tag}:Ch Up (hist)",
                line=dict(width=1.4, dash="dot", color="rgba(0,0,0,0.35)"),
                connectgaps=False,
                hovertemplate="<b>%{x}</b><br>Upper: %{y:.6f}<extra></extra>",
            ), row=price_row, col=1)

            fig.add_trace(go.Scatter(
                x=df["t"], y=df[lo], mode="lines",
                name=f"{tag}:Ch Lo (hist)",
                line=dict(width=1.4, dash="dot", color="rgba(0,0,0,0.35)"),
                connectgaps=False,
                hovertemplate="<b>%{x}</b><br>Lower: %{y:.6f}<extra></extra>",
            ), row=price_row, col=1)

            if show_mid and mid in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df[mid], mode="lines",
                    name=f"{tag}:Ch Mid (hist)",
                    line=dict(width=1.0, dash="dot", color="rgba(0,0,0,0.22)"),
                    connectgaps=False,
                    showlegend=False,
                    hoverinfo="skip",
                ), row=price_row, col=1)

        # Active (thick)
        if up_a in df.columns and lo_a in df.columns:
            fig.add_trace(go.Scatter(
                x=df["t"], y=df[up_a], mode="lines",
                name=f"{tag}:Ch Up",
                line=dict(width=2.8, color="rgba(0,0,0,0.65)"),
                connectgaps=False,
                hovertemplate="<b>%{x}</b><br>Upper: %{y:.6f}<extra></extra>",
            ), row=price_row, col=1)

            # lower first if we want fill to upper
            fig.add_trace(go.Scatter(
                x=df["t"], y=df[lo_a], mode="lines",
                name=f"{tag}:Ch Lo",
                line=dict(width=2.8, color="rgba(0,0,0,0.65)"),
                connectgaps=False,
                hovertemplate="<b>%{x}</b><br>Lower: %{y:.6f}<extra></extra>",
            ), row=price_row, col=1)

            if show_fill:
                # fill by duplicating upper with fill to nexty (order matters)
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df[up_a], mode="lines",
                    line=dict(width=0),
                    connectgaps=False,
                    showlegend=False,
                    hoverinfo="skip",
                ), row=price_row, col=1)

                fig.add_trace(go.Scatter(
                    x=df["t"], y=df[lo_a], mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    connectgaps=False,
                    showlegend=False,
                    hoverinfo="skip",
                ), row=price_row, col=1)

            if show_mid and mid_a in df.columns:
                fig.add_trace(go.Scatter(
                    x=df["t"], y=df[mid_a], mode="lines",
                    name=f"{tag}:Ch Mid",
                    line=dict(width=1.6, dash="dot", color="rgba(0,0,0,0.40)"),
                    connectgaps=False,
                    hovertemplate="<b>%{x}</b><br>Mid: %{y:.6f}<extra></extra>",
                ), row=price_row, col=1)

        # Breakout markers
        if show_breakouts and brk_evt in df.columns:
            up_idx = df.index[df[brk_evt] == 1]
            dn_idx = df.index[df[brk_evt] == -1]

            if len(up_idx):
                fig.add_trace(go.Scatter(
                    x=df.loc[up_idx, "t"],
                    y=(df.loc[up_idx, "high"] * 1.003).astype(float),
                    mode="markers",
                    name=f"{tag}:BreakUp",
                    marker=dict(symbol="star", size=marker_size, color="rgba(0,140,0,0.95)", line=dict(width=2)),
                    hovertemplate="<b>%{x}</b><br>Channel breakout UP<extra></extra>",
                ), row=price_row, col=1)

            if len(dn_idx):
                fig.add_trace(go.Scatter(
                    x=df.loc[dn_idx, "t"],
                    y=(df.loc[dn_idx, "low"] * 0.997).astype(float),
                    mode="markers",
                    name=f"{tag}:BreakDn",
                    marker=dict(symbol="star", size=marker_size, color="rgba(200,0,0,0.95)", line=dict(width=2)),
                    hovertemplate="<b>%{x}</b><br>Channel breakout DOWN<extra></extra>",
                ), row=price_row, col=1)

        # Projection (optional, uses last known slope/intercepts of active segment)
        if projection_bars > 0 and slope in df.columns and up_int in df.columns and lo_int in df.columns:
            # pick last non-NaN params
            last = df[[ "t", slope, up_int, lo_int]].dropna().tail(1)
            if len(last):
                m = float(last[slope].iloc[0])
                b_up = float(last[up_int].iloc[0])
                b_lo = float(last[lo_int].iloc[0])

                # estimate time step
                dt = df["t"].diff().median()
                if pd.isna(dt):
                    dt = pd.Timedelta(minutes=1)

                t0 = df["t"].iloc[-1]
                # index projection uses integer bar index extension; approximate by sequential bars
                # we use "bar number" proxy by using 0..proj and applying same slope in that local index space
                # projection is mainly visual; channel parameters are already in df for strategy use.
                xs = np.arange(1, projection_bars + 1, dtype=float)

                t_future = [t0 + dt * int(i) for i in xs]
                # approximate with last point index = 0
                y_up = b_up + m * (xs + 0.0)
                y_lo = b_lo + m * (xs + 0.0)

                fig.add_trace(go.Scatter(
                    x=t_future, y=y_up, mode="lines",
                    showlegend=False,
                    line=dict(width=2.0, dash="dash", color="rgba(0,0,0,0.45)"),
                    hoverinfo="skip",
                ), row=price_row, col=1)
                fig.add_trace(go.Scatter(
                    x=t_future, y=y_lo, mode="lines",
                    showlegend=False,
                    line=dict(width=2.0, dash="dash", color="rgba(0,0,0,0.45)"),
                    hoverinfo="skip",
                ), row=price_row, col=1)

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return ""