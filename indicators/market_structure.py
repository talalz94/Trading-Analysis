from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class MarketStructure:
    name = "market_structure"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def compute(df: pd.DataFrame, cfg: dict[str, Any], tag: str):
        """
        Detect confirmed swing highs/lows and classify:

          HH = Higher High
          LH = Lower High
          HL = Higher Low
          LL = Lower Low

        Important:
          Uses confirmation logic.
          A pivot needs `right` candles to confirm, so signals are marked
          on the confirmation candle, not the original pivot candle.
        """
        out = df.copy()

        high_col = cfg.get("high_col", "high")
        low_col = cfg.get("low_col", "low")

        left = int(cfg.get("left", 5))
        right = int(cfg.get("right", 2))
        min_swing_pct = float(cfg.get("min_swing_pct", 0.0))

        if high_col not in out.columns:
            raise KeyError(f"MarketStructure missing high_col='{high_col}'")

        if low_col not in out.columns:
            raise KeyError(f"MarketStructure missing low_col='{low_col}'")

        high = pd.to_numeric(out[high_col], errors="coerce")
        low = pd.to_numeric(out[low_col], errors="coerce")

        window = left + right + 1

        raw_pivot_high = high.eq(
            high.rolling(window=window, center=True, min_periods=window).max()
        )

        raw_pivot_low = low.eq(
            low.rolling(window=window, center=True, min_periods=window).min()
        )

        confirmed_high = raw_pivot_high.shift(right).fillna(False).astype(bool)
        confirmed_low = raw_pivot_low.shift(right).fillna(False).astype(bool)

        confirmed_high_price = high.shift(right).where(confirmed_high)
        confirmed_low_price = low.shift(right).where(confirmed_low)

        n = len(out)

        hh = np.zeros(n, dtype=bool)
        lh = np.zeros(n, dtype=bool)
        hl = np.zeros(n, dtype=bool)
        ll = np.zeros(n, dtype=bool)

        swing_high = confirmed_high.to_numpy(dtype=bool)
        swing_low = confirmed_low.to_numpy(dtype=bool)

        high_price_arr = confirmed_high_price.to_numpy(dtype=float)
        low_price_arr = confirmed_low_price.to_numpy(dtype=float)

        last_high = np.nan
        last_low = np.nan

        for i in range(n):
            hp = high_price_arr[i]

            if swing_high[i] and np.isfinite(hp):
                if np.isfinite(last_high):
                    threshold = abs(last_high) * (min_swing_pct / 100.0)

                    if hp > last_high + threshold:
                        hh[i] = True
                    elif hp < last_high - threshold:
                        lh[i] = True

                last_high = hp

            lp = low_price_arr[i]

            if swing_low[i] and np.isfinite(lp):
                if np.isfinite(last_low):
                    threshold = abs(last_low) * (min_swing_pct / 100.0)

                    if lp > last_low + threshold:
                        hl[i] = True
                    elif lp < last_low - threshold:
                        ll[i] = True

                last_low = lp

        created_cols = []

        def add_col(name: str, values):
            col = f"{tag}__{name}"
            out[col] = values
            created_cols.append(col)

        add_col("SWING_HIGH", swing_high)
        add_col("SWING_LOW", swing_low)

        add_col("SWING_HIGH_PRICE", confirmed_high_price.to_numpy(dtype=float))
        add_col("SWING_LOW_PRICE", confirmed_low_price.to_numpy(dtype=float))

        add_col("HH", hh)
        add_col("LH", lh)
        add_col("HL", hl)
        add_col("LL", ll)

        add_col("HH_PRICE", np.where(hh, high_price_arr, np.nan))
        add_col("LH_PRICE", np.where(lh, high_price_arr, np.nan))
        add_col("HL_PRICE", np.where(hl, low_price_arr, np.nan))
        add_col("LL_PRICE", np.where(ll, low_price_arr, np.nan))

        add_col(
            "LAST_SWING_HIGH",
            pd.Series(confirmed_high_price, index=out.index).ffill().to_numpy(dtype=float),
        )
        add_col(
            "LAST_SWING_LOW",
            pd.Series(confirmed_low_price, index=out.index).ffill().to_numpy(dtype=float),
        )

        add_col(
            "LAST_HH",
            pd.Series(np.where(hh, high_price_arr, np.nan), index=out.index).ffill().to_numpy(dtype=float),
        )
        add_col(
            "LAST_LH",
            pd.Series(np.where(lh, high_price_arr, np.nan), index=out.index).ffill().to_numpy(dtype=float),
        )
        add_col(
            "LAST_HL",
            pd.Series(np.where(hl, low_price_arr, np.nan), index=out.index).ffill().to_numpy(dtype=float),
        )
        add_col(
            "LAST_LL",
            pd.Series(np.where(ll, low_price_arr, np.nan), index=out.index).ffill().to_numpy(dtype=float),
        )

        event_code = np.full(n, "", dtype=object)
        event_code[hh] = "HH"
        event_code[lh] = "LH"
        event_code[hl] = "HL"
        event_code[ll] = "LL"

        last_event = pd.Series(event_code, index=out.index).replace("", np.nan).ffill().fillna("")

        add_col("LAST_EVENT_IS_HH", last_event.eq("HH").to_numpy(dtype=bool))
        add_col("LAST_EVENT_IS_LH", last_event.eq("LH").to_numpy(dtype=bool))
        add_col("LAST_EVENT_IS_HL", last_event.eq("HL").to_numpy(dtype=bool))
        add_col("LAST_EVENT_IS_LL", last_event.eq("LL").to_numpy(dtype=bool))

        return out.copy(), created_cols

    @staticmethod
    def add_traces(fig, df: pd.DataFrame, cfg: dict[str, Any], tag: str, row: int, price_row: int = 1):
        """
        Optional direct plotting support.
        If you are using precomputed_factory + plot_toggles, this may not be used.
        """
        import plotly.graph_objects as go

        def add_marker(col: str, name: str, text: str, symbol: str, color: str, size: int):
            if col not in df.columns:
                return

            fig.add_trace(
                go.Scatter(
                    x=df["t"],
                    y=df[col],
                    mode="markers+text",
                    name=name,
                    text=text,
                    textposition="top center",
                    marker=dict(
                        symbol=symbol,
                        size=size,
                        color=color,
                    ),
                    connectgaps=False,
                ),
                row=row,
                col=1,
            )

        add_marker(f"{tag}__HH_PRICE", "HH", "HH", "triangle-up", "rgba(0,160,0,0.95)", 11)
        add_marker(f"{tag}__HL_PRICE", "HL", "HL", "circle", "rgba(0,120,255,0.95)", 9)
        add_marker(f"{tag}__LH_PRICE", "LH", "LH", "triangle-down", "rgba(255,140,0,0.95)", 11)
        add_marker(f"{tag}__LL_PRICE", "LL", "LL", "x", "rgba(200,0,0,0.95)", 10)

    @staticmethod
    def yaxis_title(cfg: dict[str, Any], tag: str) -> str:
        return ""