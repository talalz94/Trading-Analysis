from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def col(tag: str, base: str) -> str:
    return f"{tag}__{base}"

def _pivot_flags(values: np.ndarray, lb: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(values)
    hi = np.zeros(n, dtype=bool)
    lo = np.zeros(n, dtype=bool)
    if n < (2 * lb + 1):
        return hi, lo
    for i in range(lb, n - lb):
        w = values[i - lb:i + lb + 1]
        v = values[i]
        if not np.isfinite(v):
            continue
        if v == np.nanmax(w):
            hi[i] = True
        if v == np.nanmin(w):
            lo[i] = True
    return hi, lo


def _cluster_levels(prices: List[float], tol_pct: float) -> List[Tuple[float, int]]:
    """
    Cluster prices into levels within tolerance (percent).
    Returns list of (level_price, touch_count).
    """
    levels: List[Tuple[float, int]] = []
    for p in prices:
        if not np.isfinite(p) or p <= 0:
            continue
        found = False
        for i, (lvl, cnt) in enumerate(levels):
            if abs(p - lvl) / lvl <= tol_pct:
                # update level as running average
                new_lvl = (lvl * cnt + p) / (cnt + 1)
                levels[i] = (new_lvl, cnt + 1)
                found = True
                break
        if not found:
            levels.append((p, 1))
    return levels


class SupportResistance:
    """
    Support/Resistance via pivot clustering.
    Features:
      - nearest support below close
      - nearest resistance above close
      - distance pct to each
      - break flags (close crosses through)
    """
    name = "support_resistance"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: Dict[str, Any], tag: str) -> Tuple[pd.DataFrame, List[str]]:
        pivot_lb = int(cfg.get("pivot_lookback", 10))
        tol_pct = float(cfg.get("tolerance_pct", 0.003))  # 0.3%
        max_levels = int(cfg.get("max_levels", 8))
        min_touches = int(cfg.get("min_touches", 2))

        out = df.copy()

        sup_c = col(tag, "SUPPORT")
        res_c = col(tag, "RESISTANCE")
        dsup_c = col(tag, "DIST_SUP_PCT")
        dres_c = col(tag, "DIST_RES_PCT")
        brk_sup = col(tag, "BREAK_SUP")
        brk_res = col(tag, "BREAK_RES")

        out[sup_c] = np.nan
        out[res_c] = np.nan
        out[dsup_c] = np.nan
        out[dres_c] = np.nan
        out[brk_sup] = False
        out[brk_res] = False

        hi_flags, lo_flags = _pivot_flags(out["high"].to_numpy(float), pivot_lb)[0], _pivot_flags(out["low"].to_numpy(float), pivot_lb)[1]

        pivot_prices: List[float] = []
        pivot_prices.extend(out.loc[hi_flags, "high"].tolist())
        pivot_prices.extend(out.loc[lo_flags, "low"].tolist())

        clusters = _cluster_levels(pivot_prices, tol_pct)
        clusters = [(lvl, cnt) for (lvl, cnt) in clusters if cnt >= min_touches]
        clusters.sort(key=lambda x: (-x[1], x[0]))  # touches desc, then price
        clusters = clusters[:max_levels]
        levels = sorted([lvl for (lvl, _) in clusters])

        # nearest support/resistance per bar
        closes = out["close"].to_numpy(float)
        supports = np.full(len(out), np.nan, dtype=float)
        resistances = np.full(len(out), np.nan, dtype=float)

        for i, px in enumerate(closes):
            if not np.isfinite(px):
                continue
            # support = max(level <= px)
            s = [l for l in levels if l <= px]
            r = [l for l in levels if l >= px]
            supports[i] = max(s) if s else np.nan
            resistances[i] = min(r) if r else np.nan

        out[sup_c] = supports
        out[res_c] = resistances

        out[dsup_c] = (out["close"] / out[sup_c] - 1.0) * 100.0
        out[dres_c] = (out[res_c] / out["close"] - 1.0) * 100.0

        prev_close = out["close"].shift(1)
        out[brk_res] = out[res_c].notna() & prev_close.notna() & (prev_close <= out[res_c]) & (out["close"] > out[res_c])
        out[brk_sup] = out[sup_c].notna() & prev_close.notna() & (prev_close >= out[sup_c]) & (out["close"] < out[sup_c])

        # store levels for plotting (recompute deterministically in add_traces too)
        return out, [sup_c, res_c, dsup_c, dres_c, brk_sup, brk_res]

    @staticmethod
    def add_traces(fig: go.Figure, df: pd.DataFrame, cfg: Dict[str, Any], tag: str, row: int, price_row: int) -> None:
        pivot_lb = int(cfg.get("pivot_lookback", 10))
        tol_pct = float(cfg.get("tolerance_pct", 0.003))
        max_levels = int(cfg.get("max_levels", 8))
        min_touches = int(cfg.get("min_touches", 2))
        show_markers = bool(cfg.get("show_break_markers", True))
        marker_size = int(cfg.get("marker_size", 14))

        # recompute level list deterministically
        hi_flags, lo_flags = _pivot_flags(df["high"].to_numpy(float), pivot_lb)[0], _pivot_flags(df["low"].to_numpy(float), pivot_lb)[1]
        pivot_prices: List[float] = []
        pivot_prices.extend(df.loc[hi_flags, "high"].tolist())
        pivot_prices.extend(df.loc[lo_flags, "low"].tolist())

        clusters = _cluster_levels(pivot_prices, tol_pct)
        clusters = [(lvl, cnt) for (lvl, cnt) in clusters if cnt >= min_touches]
        clusters.sort(key=lambda x: (-x[1], x[0]))
        clusters = clusters[:max_levels]
        levels = sorted([lvl for (lvl, _) in clusters])

        # horizontal lines
        for lvl in levels:
            fig.add_hline(
                y=lvl, row=price_row, col=1,
                line_width=1.5, line_dash="dot",
                line_color="rgba(0,0,0,0.35)"
            )

        brk_sup = col(tag, "BREAK_SUP")
        brk_res = col(tag, "BREAK_RES")

        if show_markers and brk_sup in df.columns and brk_res in df.columns:
            sup_idx = df.index[df[brk_sup].fillna(False)]
            res_idx = df.index[df[brk_res].fillna(False)]

            if len(res_idx):
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[res_idx, "t"],
                        y=df.loc[res_idx, "high"] * 1.003,
                        mode="markers",
                        name=f"{tag}:ResBreak",
                        marker=dict(symbol="diamond", size=marker_size, line=dict(width=2)),
                        hovertemplate="<b>%{x}</b><br>Resistance Break<extra></extra>",
                    ),
                    row=price_row, col=1
                )

            if len(sup_idx):
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[sup_idx, "t"],
                        y=df.loc[sup_idx, "low"] * 0.997,
                        mode="markers",
                        name=f"{tag}:SupBreak",
                        marker=dict(symbol="diamond", size=marker_size, line=dict(width=2)),
                        hovertemplate="<b>%{x}</b><br>Support Break<extra></extra>",
                    ),
                    row=price_row, col=1
                )

    @staticmethod
    def yaxis_title(cfg: Dict[str, Any], tag: str) -> str:
        return ""