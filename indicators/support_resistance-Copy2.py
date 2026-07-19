from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class SupportResistance:
    name = "support_resistance"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def _cluster_levels(
        pivots: list[tuple[int, float]],
        tolerance_pct: float,
        min_touches: int,
    ) -> list[dict]:
        """
        Clusters nearby pivot prices into zones.

        Returns clusters with:
          center
          zone_low
          zone_high
          touches
          last_i
        """
        clusters: list[dict] = []

        for idx, price in pivots:
            if not np.isfinite(price) or price <= 0:
                continue

            best_j = None
            best_dist = np.inf

            for j, cl in enumerate(clusters):
                center = cl["center"]
                dist_pct = abs(price - center) / center * 100.0

                if dist_pct <= tolerance_pct and dist_pct < best_dist:
                    best_j = j
                    best_dist = dist_pct

            if best_j is None:
                clusters.append({
                    "center": float(price),
                    "zone_low": float(price),
                    "zone_high": float(price),
                    "touches": 1,
                    "last_i": int(idx),
                    "prices": [float(price)],
                })
            else:
                cl = clusters[best_j]
                cl["prices"].append(float(price))
                cl["touches"] += 1
                cl["last_i"] = max(cl["last_i"], int(idx))
                cl["center"] = float(np.mean(cl["prices"]))
                cl["zone_low"] = float(min(cl["prices"]))
                cl["zone_high"] = float(max(cl["prices"]))

        return [
            cl for cl in clusters
            if int(cl["touches"]) >= int(min_touches)
        ]

    @staticmethod
    def compute(df: pd.DataFrame, cfg: dict[str, Any], tag: str):
        out = df.copy()

        high_col = cfg.get("high_col", "high")
        low_col = cfg.get("low_col", "low")
        close_col = cfg.get("close_col", "close")

        left = int(cfg.get("left", 5))
        right = int(cfg.get("right", 2))

        lookback_bars = int(cfg.get("lookback_bars", 500))
        tolerance_pct = float(cfg.get("tolerance_pct", 0.08))
        min_touches = int(cfg.get("min_touches", 2))
        max_levels = int(cfg.get("max_levels", 3))
        near_pct = float(cfg.get("near_pct", 0.10))

        if high_col not in out.columns:
            raise KeyError(f"SupportResistance missing high_col='{high_col}'")

        if low_col not in out.columns:
            raise KeyError(f"SupportResistance missing low_col='{low_col}'")

        if close_col not in out.columns:
            raise KeyError(f"SupportResistance missing close_col='{close_col}'")

        high = pd.to_numeric(out[high_col], errors="coerce")
        low = pd.to_numeric(out[low_col], errors="coerce")
        close = pd.to_numeric(out[close_col], errors="coerce")

        n = len(out)
        window = left + right + 1

        # Raw pivot is on actual pivot candle.
        raw_pivot_high = high.eq(
            high.rolling(window=window, center=True, min_periods=window).max()
        )
        raw_pivot_low = low.eq(
            low.rolling(window=window, center=True, min_periods=window).min()
        )

        # Rule-safe confirmation happens right candles later.
        confirmed_high = raw_pivot_high.shift(right).fillna(False).astype(bool)
        confirmed_low = raw_pivot_low.shift(right).fillna(False).astype(bool)

        confirmed_high_price = high.shift(right).where(confirmed_high)
        confirmed_low_price = low.shift(right).where(confirmed_low)

        resistance_pivots: list[tuple[int, float]] = []
        support_pivots: list[tuple[int, float]] = []

        data = {}

        for k in range(1, max_levels + 1):
            data[f"R{k}"] = np.full(n, np.nan)
            data[f"R{k}_ZONE_LOW"] = np.full(n, np.nan)
            data[f"R{k}_ZONE_HIGH"] = np.full(n, np.nan)
            data[f"R{k}_TOUCHES"] = np.full(n, np.nan)

            data[f"S{k}"] = np.full(n, np.nan)
            data[f"S{k}_ZONE_LOW"] = np.full(n, np.nan)
            data[f"S{k}_ZONE_HIGH"] = np.full(n, np.nan)
            data[f"S{k}_TOUCHES"] = np.full(n, np.nan)

        swing_high_arr = confirmed_high.to_numpy(dtype=bool)
        swing_low_arr = confirmed_low.to_numpy(dtype=bool)
        high_price_arr = confirmed_high_price.to_numpy(dtype=float)
        low_price_arr = confirmed_low_price.to_numpy(dtype=float)
        close_arr = close.to_numpy(dtype=float)

        for i in range(n):
            px = close_arr[i]

            if swing_high_arr[i] and np.isfinite(high_price_arr[i]):
                resistance_pivots.append((i, float(high_price_arr[i])))

            if swing_low_arr[i] and np.isfinite(low_price_arr[i]):
                support_pivots.append((i, float(low_price_arr[i])))

            cutoff = i - lookback_bars
            resistance_pivots = [(j, p) for j, p in resistance_pivots if j >= cutoff]
            support_pivots = [(j, p) for j, p in support_pivots if j >= cutoff]

            if not np.isfinite(px) or px <= 0:
                continue

            resistance_clusters = SupportResistance._cluster_levels(
                resistance_pivots,
                tolerance_pct=tolerance_pct,
                min_touches=min_touches,
            )

            support_clusters = SupportResistance._cluster_levels(
                support_pivots,
                tolerance_pct=tolerance_pct,
                min_touches=min_touches,
            )

            # Resistance = nearest clustered pivot-high zones above current price.
            resistances = [
                cl for cl in resistance_clusters
                if cl["center"] > px
            ]
            resistances = sorted(
                resistances,
                key=lambda cl: (
                    abs(cl["center"] - px),
                    -cl["touches"],
                    -cl["last_i"],
                )
            )

            # Support = nearest clustered pivot-low zones below current price.
            supports = [
                cl for cl in support_clusters
                if cl["center"] < px
            ]
            supports = sorted(
                supports,
                key=lambda cl: (
                    abs(px - cl["center"]),
                    -cl["touches"],
                    -cl["last_i"],
                )
            )

            for k, cl in enumerate(resistances[:max_levels], start=1):
                data[f"R{k}"][i] = cl["center"]
                data[f"R{k}_ZONE_LOW"][i] = cl["zone_low"]
                data[f"R{k}_ZONE_HIGH"][i] = cl["zone_high"]
                data[f"R{k}_TOUCHES"][i] = cl["touches"]

            for k, cl in enumerate(supports[:max_levels], start=1):
                data[f"S{k}"][i] = cl["center"]
                data[f"S{k}_ZONE_LOW"][i] = cl["zone_low"]
                data[f"S{k}_ZONE_HIGH"][i] = cl["zone_high"]
                data[f"S{k}_TOUCHES"][i] = cl["touches"]

        created_cols = []

        def add_col(name: str, values):
            col = f"{tag}__{name}"
            out[col] = values
            created_cols.append(col)

        # Raw confirmed pivot info.
        add_col("PIVOT_HIGH", swing_high_arr)
        add_col("PIVOT_LOW", swing_low_arr)
        add_col("PIVOT_HIGH_PRICE", high_price_arr)
        add_col("PIVOT_LOW_PRICE", low_price_arr)

        for name, values in data.items():
            add_col(name, values)

        # Convenience nearest levels.
        s1 = data["S1"]
        r1 = data["R1"]

        dist_to_s1 = np.where(
            np.isfinite(s1) & (close_arr > 0),
            (close_arr - s1) / close_arr * 100.0,
            np.nan,
        )

        dist_to_r1 = np.where(
            np.isfinite(r1) & (close_arr > 0),
            (r1 - close_arr) / close_arr * 100.0,
            np.nan,
        )

        add_col("NEAREST_SUPPORT", s1)
        add_col("NEAREST_RESISTANCE", r1)
        add_col("DIST_TO_SUPPORT_PCT", dist_to_s1)
        add_col("DIST_TO_RESISTANCE_PCT", dist_to_r1)

        add_col("NEAR_SUPPORT", np.isfinite(dist_to_s1) & (dist_to_s1 <= near_pct))
        add_col("NEAR_RESISTANCE", np.isfinite(dist_to_r1) & (dist_to_r1 <= near_pct))

        close_prev = pd.Series(close_arr).shift(1).to_numpy(dtype=float)
        s1_prev = pd.Series(s1).shift(1).to_numpy(dtype=float)
        r1_prev = pd.Series(r1).shift(1).to_numpy(dtype=float)

        break_above_r1 = (
            np.isfinite(r1)
            & np.isfinite(r1_prev)
            & (close_arr > r1)
            & (close_prev <= r1_prev)
        )

        break_below_s1 = (
            np.isfinite(s1)
            & np.isfinite(s1_prev)
            & (close_arr < s1)
            & (close_prev >= s1_prev)
        )

        add_col("BREAK_ABOVE_R1", break_above_r1)
        add_col("BREAK_BELOW_S1", break_below_s1)

        return out.copy(), created_cols

    @staticmethod
    def add_traces(fig, df: pd.DataFrame, cfg: dict[str, Any], tag: str, row: int, price_row: int = 1):
        return

    @staticmethod
    def yaxis_title(cfg: dict[str, Any], tag: str) -> str:
        return ""