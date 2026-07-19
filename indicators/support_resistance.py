from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class SupportResistance:
    """
    Robust horizontal support/resistance zones.

    Logic:
      - Confirm pivot highs/lows.
      - Cluster repeated touches near the same price.
      - Activate level only after min_touches.
      - Freeze the horizontal level after activation.
      - Keep plotting until the level is broken.
      - Stop plotting after break.
      - Avoid jagged nearest-level recalculation every candle.

    This is designed to produce chart levels like:
      resistance ceiling
      support floor
      repeated-touch zones
    """

    name = "support_resistance"
    is_overlay = True
    row_weight = 0.0

    @staticmethod
    def _pivot_high(high: pd.Series, left: int, right: int) -> pd.Series:
        window = left + right + 1
        return high.eq(
            high.rolling(window=window, center=True, min_periods=window).max()
        )

    @staticmethod
    def _pivot_low(low: pd.Series, left: int, right: int) -> pd.Series:
        window = left + right + 1
        return low.eq(
            low.rolling(window=window, center=True, min_periods=window).min()
        )

    @staticmethod
    def _level_price(prices: list[float], method: str) -> float:
        arr = np.asarray(prices, dtype=float)

        if method == "mean":
            return float(np.nanmean(arr))

        if method == "last":
            return float(arr[-1])

        # Best default for zones: median resists outlier wicks.
        return float(np.nanmedian(arr))

    @staticmethod
    def _new_cluster(
        cluster_id: int,
        mode: str,
        price: float,
        idx: int,
    ) -> dict:
        return {
            "id": int(cluster_id),
            "mode": mode,  # support / resistance

            "touches": 1,
            "touch_idxs": [int(idx)],
            "touch_prices": [float(price)],
            "last_touch_i": int(idx),

            "center": float(price),
            "zone_low": float(price),
            "zone_high": float(price),

            # Frozen once activated.
            "line_center": np.nan,
            "line_zone_low": np.nan,
            "line_zone_high": np.nan,

            "active": False,
            "broken": False,
            "formed_i": None,
            "broken_i": None,
        }

    @staticmethod
    def _update_cluster_stats(
        cl: dict,
        level_method: str,
        freeze_level_after_activation: bool,
    ) -> None:
        prices = np.asarray(cl["touch_prices"], dtype=float)

        cl["center"] = SupportResistance._level_price(cl["touch_prices"], level_method)
        cl["zone_low"] = float(np.nanmin(prices))
        cl["zone_high"] = float(np.nanmax(prices))
        cl["touches"] = len(cl["touch_prices"])
        cl["last_touch_i"] = int(max(cl["touch_idxs"]))

        # If not frozen, allow the level to update.
        # Default is frozen to avoid jagged lines.
        if cl["active"] and not freeze_level_after_activation:
            cl["line_center"] = cl["center"]
            cl["line_zone_low"] = cl["zone_low"]
            cl["line_zone_high"] = cl["zone_high"]

    @staticmethod
    def _add_touch(
        clusters: list[dict],
        price: float,
        idx: int,
        mode: str,
        tolerance_pct: float,
        min_touches: int,
        min_bars_between_touches: int,
        level_method: str,
        freeze_level_after_activation: bool,
        cluster_id_counter: list[int],
    ) -> None:
        if not np.isfinite(price) or price <= 0:
            return

        best_j = None
        best_dist = np.inf

        for j, cl in enumerate(clusters):
            if cl["mode"] != mode:
                continue

            if cl["broken"]:
                continue

            ref = cl["line_center"] if cl["active"] and np.isfinite(cl["line_center"]) else cl["center"]

            if not np.isfinite(ref) or ref <= 0:
                continue

            dist_pct = abs(price - ref) / ref * 100.0

            if dist_pct <= tolerance_pct and dist_pct < best_dist:
                best_j = j
                best_dist = dist_pct

        if best_j is None:
            cluster_id_counter[0] += 1
            clusters.append(
                SupportResistance._new_cluster(
                    cluster_id=cluster_id_counter[0],
                    mode=mode,
                    price=price,
                    idx=idx,
                )
            )
            return

        cl = clusters[best_j]

        # Do not count repeated micro touches too close in time.
        if int(idx) - int(cl["last_touch_i"]) < int(min_bars_between_touches):
            return

        cl["touch_prices"].append(float(price))
        cl["touch_idxs"].append(int(idx))

        SupportResistance._update_cluster_stats(
            cl,
            level_method=level_method,
            freeze_level_after_activation=freeze_level_after_activation,
        )

        # Activate the horizontal level once enough true touches exist.
        if (not cl["active"]) and cl["touches"] >= int(min_touches):
            cl["active"] = True
            cl["formed_i"] = int(idx)

            # Freeze line at activation price/zone.
            cl["line_center"] = cl["center"]
            cl["line_zone_low"] = cl["zone_low"]
            cl["line_zone_high"] = cl["zone_high"]

    @staticmethod
    def _prune_clusters(
        clusters: list[dict],
        cutoff_i: int,
        max_clusters: int,
        expire_active_on_lookback: bool,
        level_method: str,
        freeze_level_after_activation: bool,
    ) -> list[dict]:
        kept = []

        for cl in clusters:
            # Active unbroken levels remain until broken by price.
            if cl["active"] and not cl["broken"] and not expire_active_on_lookback:
                kept.append(cl)
                continue

            idxs = np.asarray(cl["touch_idxs"], dtype=int)
            prices = np.asarray(cl["touch_prices"], dtype=float)

            mask = idxs >= int(cutoff_i)

            if not mask.any():
                continue

            cl2 = dict(cl)
            cl2["touch_idxs"] = idxs[mask].astype(int).tolist()
            cl2["touch_prices"] = prices[mask].astype(float).tolist()

            SupportResistance._update_cluster_stats(
                cl2,
                level_method=level_method,
                freeze_level_after_activation=freeze_level_after_activation,
            )

            kept.append(cl2)

        # Keep active levels first, then strongest/recent candidates.
        kept = sorted(
            kept,
            key=lambda x: (
                int(bool(x["active"] and not x["broken"])),
                int(x["touches"]),
                int(x["last_touch_i"]),
            ),
            reverse=True,
        )

        return kept[: int(max_clusters)]

    @staticmethod
    def _break_active_levels(
        clusters: list[dict],
        close_px: float,
        high_px: float,
        low_px: float,
        idx: int,
        breakout_basis: str,
        breakout_buffer_pct: float,
    ) -> None:
        if breakout_basis == "wick":
            resistance_break_px = high_px
            support_break_px = low_px
        else:
            # Best default: close-based break avoids wick fakeouts.
            resistance_break_px = close_px
            support_break_px = close_px

        for cl in clusters:
            if not cl["active"] or cl["broken"]:
                continue

            buffer = breakout_buffer_pct / 100.0

            if cl["mode"] == "resistance":
                break_price = cl["line_zone_high"] * (1.0 + buffer)

                if np.isfinite(resistance_break_px) and resistance_break_px > break_price:
                    cl["active"] = False
                    cl["broken"] = True
                    cl["broken_i"] = int(idx)

            elif cl["mode"] == "support":
                break_price = cl["line_zone_low"] * (1.0 - buffer)

                if np.isfinite(support_break_px) and support_break_px < break_price:
                    cl["active"] = False
                    cl["broken"] = True
                    cl["broken_i"] = int(idx)

    @staticmethod
    def _select_active_levels(
        clusters: list[dict],
        px: float,
        mode: str,
        max_levels: int,
        selection: str,
        overlap_pct: float,
    ) -> list[dict]:
        if not np.isfinite(px) or px <= 0:
            return []

        out = []

        overlap = overlap_pct / 100.0

        for cl in clusters:
            if cl["mode"] != mode:
                continue

            if not cl["active"] or cl["broken"]:
                continue

            center = cl["line_center"]

            if not np.isfinite(center):
                continue

            if mode == "resistance":
                # Resistance should generally be above price, but allow slight overlap.
                if center >= px * (1.0 - overlap):
                    out.append(cl)

            elif mode == "support":
                # Support should generally be below price, but allow slight overlap.
                if center <= px * (1.0 + overlap):
                    out.append(cl)

        if selection == "nearest":
            out = sorted(
                out,
                key=lambda cl: (
                    abs(cl["line_center"] - px),
                    -int(cl["touches"]),
                    -int(cl["last_touch_i"]),
                ),
            )

        elif selection == "recent":
            out = sorted(
                out,
                key=lambda cl: (
                    -int(cl["last_touch_i"]),
                    abs(cl["line_center"] - px),
                    -int(cl["touches"]),
                ),
            )

        else:
            # Best default for robust lines: strongest first.
            out = sorted(
                out,
                key=lambda cl: (
                    -int(cl["touches"]),
                    -int(cl["last_touch_i"]),
                    abs(cl["line_center"] - px),
                ),
            )

        return out[: int(max_levels)]

    @staticmethod
    def compute(df: pd.DataFrame, cfg: dict[str, Any], tag: str):
        out = df.copy()

        high_col = cfg.get("high_col", "high")
        low_col = cfg.get("low_col", "low")
        close_col = cfg.get("close_col", "close")

        left = int(cfg.get("left", 10))
        right = int(cfg.get("right", 5))

        lookback_bars = int(cfg.get("lookback_bars", 2000))
        tolerance_pct = float(cfg.get("tolerance_pct", 0.10))
        min_touches = int(cfg.get("min_touches", 3))
        min_bars_between_touches = int(cfg.get("min_bars_between_touches", 25))

        max_levels = int(cfg.get("max_levels", 1))
        max_clusters = int(cfg.get("max_clusters", 80))

        selection = str(cfg.get("selection", "strongest")).lower().strip()
        level_method = str(cfg.get("level_method", "median")).lower().strip()

        breakout_basis = str(cfg.get("breakout_basis", "close")).lower().strip()
        breakout_buffer_pct = float(cfg.get("breakout_buffer_pct", 0.03))

        freeze_level_after_activation = bool(cfg.get("freeze_level_after_activation", True))
        expire_active_on_lookback = bool(cfg.get("expire_active_on_lookback", False))
        overlap_pct = float(cfg.get("overlap_pct", 0.02))

        near_pct = float(cfg.get("near_pct", tolerance_pct))

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

        raw_high = SupportResistance._pivot_high(high, left, right)
        raw_low = SupportResistance._pivot_low(low, left, right)

        # Confirmed pivots appear after `right` bars.
        # This avoids lookahead for rule columns.
        confirmed_high = raw_high.shift(right).fillna(False).astype(bool)
        confirmed_low = raw_low.shift(right).fillna(False).astype(bool)

        confirmed_high_price = high.shift(right).where(confirmed_high)
        confirmed_low_price = low.shift(right).where(confirmed_low)

        ch_arr = confirmed_high.to_numpy(dtype=bool)
        cl_arr = confirmed_low.to_numpy(dtype=bool)

        ch_price_arr = confirmed_high_price.to_numpy(dtype=float)
        cl_price_arr = confirmed_low_price.to_numpy(dtype=float)

        close_arr = close.to_numpy(dtype=float)
        high_arr = high.to_numpy(dtype=float)
        low_arr = low.to_numpy(dtype=float)

        data: dict[str, np.ndarray] = {}

        for k in range(1, max_levels + 1):
            for prefix in ("S", "R"):
                data[f"{prefix}{k}"] = np.full(n, np.nan)
                data[f"{prefix}{k}_ZONE_LOW"] = np.full(n, np.nan)
                data[f"{prefix}{k}_ZONE_HIGH"] = np.full(n, np.nan)
                data[f"{prefix}{k}_TOUCHES"] = np.full(n, np.nan)
                data[f"{prefix}{k}_ID"] = np.full(n, np.nan)

        clusters: list[dict] = []
        cluster_id_counter = [0]

        for i in range(n):
            px = close_arr[i]
            hi = high_arr[i]
            lo = low_arr[i]

            # First add newly confirmed touches.
            if ch_arr[i] and np.isfinite(ch_price_arr[i]):
                SupportResistance._add_touch(
                    clusters=clusters,
                    price=float(ch_price_arr[i]),
                    idx=i,
                    mode="resistance",
                    tolerance_pct=tolerance_pct,
                    min_touches=min_touches,
                    min_bars_between_touches=min_bars_between_touches,
                    level_method=level_method,
                    freeze_level_after_activation=freeze_level_after_activation,
                    cluster_id_counter=cluster_id_counter,
                )

            if cl_arr[i] and np.isfinite(cl_price_arr[i]):
                SupportResistance._add_touch(
                    clusters=clusters,
                    price=float(cl_price_arr[i]),
                    idx=i,
                    mode="support",
                    tolerance_pct=tolerance_pct,
                    min_touches=min_touches,
                    min_bars_between_touches=min_bars_between_touches,
                    level_method=level_method,
                    freeze_level_after_activation=freeze_level_after_activation,
                    cluster_id_counter=cluster_id_counter,
                )

            # Then invalidate active levels if broken.
            SupportResistance._break_active_levels(
                clusters=clusters,
                close_px=px,
                high_px=hi,
                low_px=lo,
                idx=i,
                breakout_basis=breakout_basis,
                breakout_buffer_pct=breakout_buffer_pct,
            )

            # Occasionally prune old unformed/broken clutter.
            if i % 100 == 0:
                clusters = SupportResistance._prune_clusters(
                    clusters=clusters,
                    cutoff_i=i - lookback_bars,
                    max_clusters=max_clusters,
                    expire_active_on_lookback=expire_active_on_lookback,
                    level_method=level_method,
                    freeze_level_after_activation=freeze_level_after_activation,
                )

            supports = SupportResistance._select_active_levels(
                clusters=clusters,
                px=px,
                mode="support",
                max_levels=max_levels,
                selection=selection,
                overlap_pct=overlap_pct,
            )

            resistances = SupportResistance._select_active_levels(
                clusters=clusters,
                px=px,
                mode="resistance",
                max_levels=max_levels,
                selection=selection,
                overlap_pct=overlap_pct,
            )

            for k, clx in enumerate(supports, start=1):
                data[f"S{k}"][i] = clx["line_center"]
                data[f"S{k}_ZONE_LOW"][i] = clx["line_zone_low"]
                data[f"S{k}_ZONE_HIGH"][i] = clx["line_zone_high"]
                data[f"S{k}_TOUCHES"][i] = clx["touches"]
                data[f"S{k}_ID"][i] = clx["id"]

            for k, clx in enumerate(resistances, start=1):
                data[f"R{k}"][i] = clx["line_center"]
                data[f"R{k}_ZONE_LOW"][i] = clx["line_zone_low"]
                data[f"R{k}_ZONE_HIGH"][i] = clx["line_zone_high"]
                data[f"R{k}_TOUCHES"][i] = clx["touches"]
                data[f"R{k}_ID"][i] = clx["id"]

        # Prevent Plotly from drawing diagonal connectors when a level changes.
        for k in range(1, max_levels + 1):
            for prefix in ("S", "R"):
                id_col = data[f"{prefix}{k}_ID"]
                prev_id = pd.Series(id_col).shift(1).to_numpy(dtype=float)

                changed = (
                    np.isfinite(id_col)
                    & np.isfinite(prev_id)
                    & (id_col != prev_id)
                )

                for suffix in ("", "_ZONE_LOW", "_ZONE_HIGH", "_TOUCHES"):
                    arr = data[f"{prefix}{k}{suffix}"]
                    arr[changed] = np.nan
                    data[f"{prefix}{k}{suffix}"] = arr

        created_cols = []

        def add_col(name: str, values):
            col = f"{tag}__{name}"
            out[col] = values
            created_cols.append(col)

        add_col("PIVOT_HIGH", ch_arr)
        add_col("PIVOT_LOW", cl_arr)
        add_col("PIVOT_HIGH_PRICE", ch_price_arr)
        add_col("PIVOT_LOW_PRICE", cl_price_arr)

        for name, values in data.items():
            add_col(name, values)

        s1 = data["S1"]
        r1 = data["R1"]
        s1_high = data["S1_ZONE_HIGH"]
        s1_low = data["S1_ZONE_LOW"]
        r1_high = data["R1_ZONE_HIGH"]
        r1_low = data["R1_ZONE_LOW"]

        dist_to_s1 = np.where(
            np.isfinite(s1) & np.isfinite(close_arr) & (close_arr > 0),
            (close_arr - s1) / close_arr * 100.0,
            np.nan,
        )

        dist_to_r1 = np.where(
            np.isfinite(r1) & np.isfinite(close_arr) & (close_arr > 0),
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
        r1_high_prev = pd.Series(r1_high).shift(1).to_numpy(dtype=float)
        s1_low_prev = pd.Series(s1_low).shift(1).to_numpy(dtype=float)

        breakout_buffer = breakout_buffer_pct / 100.0

        break_above_r1 = (
            np.isfinite(r1_high_prev)
            & np.isfinite(close_arr)
            & np.isfinite(close_prev)
            & (close_arr > r1_high_prev * (1.0 + breakout_buffer))
            & (close_prev <= r1_high_prev * (1.0 + breakout_buffer))
        )

        break_below_s1 = (
            np.isfinite(s1_low_prev)
            & np.isfinite(close_arr)
            & np.isfinite(close_prev)
            & (close_arr < s1_low_prev * (1.0 - breakout_buffer))
            & (close_prev >= s1_low_prev * (1.0 - breakout_buffer))
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