from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


def _as_windows(windows) -> tuple[int, ...]:
    """
    Allows:
      windows=(3, 5, 10)
      windows=range(1, 31)
      windows=10 -> interpreted as (10,)
    """
    if isinstance(windows, int):
        return (int(windows),)

    return tuple(sorted({int(w) for w in windows if int(w) > 0}))


def _rolling_count_bool(mask: pd.Series, window: int) -> pd.Series:
    return (
        mask.fillna(False)
        .astype(bool)
        .rolling(int(window), min_periods=int(window))
        .sum()
    )


def _rolling_any_bool(mask: pd.Series, window: int) -> pd.Series:
    return _rolling_count_bool(mask, window).gt(0)


def _active_cross_state(cross_up: np.ndarray, cross_down: np.ndarray, lookback: int) -> np.ndarray:
    """
    True when:
      - latest relevant cross was an up-cross
      - it happened within lookback candles
      - no newer opposite cross happened after it
    """
    n = len(cross_up)
    idx = np.arange(n)

    last_up = np.maximum.accumulate(np.where(cross_up, idx, -1))
    last_down = np.maximum.accumulate(np.where(cross_down, idx, -1))

    return (
        (last_up >= 0)
        & ((idx - last_up) < int(lookback))
        & (last_up > last_down)
    )


def add_stochastic_signal_features(
    df: pd.DataFrame,
    tag: str = "st14",
    k_col: str | None = None,
    d_col: str | None = None,
    levels: Sequence[float] = (20, 50, 80),
    windows: Sequence[int] | range | int = (3, 5, 10, 15),
    source: str = "K",

    # Control how many features are created.
    add_state_cols: bool = True,
    add_count_cols: bool = True,
    add_all_cols: bool = True,
    add_prev_cols: bool = True,
    add_cross_cols: bool = True,
    add_active_cols: bool = True,
) -> pd.DataFrame:
    """
    Adds reusable stochastic rule features.

    IMPORTANT:
      Apply this BEFORE align_timeframes(), separately on each timeframe df.

    Example outputs:
      st14__K_ABOVE_20
      st14__K_BELOW_20
      st14__K_CROSS_UP_20
      st14__K_CROSS_DOWN_20

      st14__K_ABOVE_20_COUNT_5
      st14__K_BELOW_20_COUNT_5

      st14__K_PREV_BELOW_20_COUNT_5
      st14__K_PREV_BELOW_20_ALL_5

      st14__K_CROSS_UP_20_LAST_5
      st14__K_CROSS_UP_20_ACTIVE_5
      st14__K_CROSS_DOWN_20_ACTIVE_5

    Meaning:
      CROSS_UP_20_LAST_5:
        cross happened at least once in the last 5 candles.

      CROSS_UP_20_ACTIVE_5:
        crossed up within the last 5 candles and has not crossed back down since.

      PREV_BELOW_20_COUNT_5:
        how many of the previous 5 candles, excluding current, were below 20.
    """
    windows = _as_windows(windows)

    if k_col is None:
        k_col = f"{tag}__K"

    if d_col is None:
        d_col = f"{tag}__D"

    source = source.upper().strip()

    if source == "K":
        src_col = k_col
        src_name = "K"
    elif source == "D":
        src_col = d_col
        src_name = "D"
    else:
        raise ValueError("source must be 'K' or 'D'.")

    if src_col not in df.columns:
        raise KeyError(
            f"Missing stochastic source column: {src_col}. "
            f"Available stochastic columns: {[c for c in df.columns if tag in c]}"
        )

    s = pd.to_numeric(df[src_col], errors="coerce")

    # Store all new columns here, then concat once.
    new_cols: dict[str, object] = {}

    for level in levels:
        level_int = int(level)

        above = s > level
        below = s < level

        cross_up = above & (s.shift(1) <= level)
        cross_down = below & (s.shift(1) >= level)

        base = f"{tag}__{src_name}"

        if add_state_cols:
            new_cols[f"{base}_ABOVE_{level_int}"] = above.to_numpy(dtype=bool)
            new_cols[f"{base}_BELOW_{level_int}"] = below.to_numpy(dtype=bool)

        if add_cross_cols:
            new_cols[f"{base}_CROSS_UP_{level_int}"] = cross_up.to_numpy(dtype=bool)
            new_cols[f"{base}_CROSS_DOWN_{level_int}"] = cross_down.to_numpy(dtype=bool)

        cross_up_np = cross_up.fillna(False).to_numpy(dtype=bool)
        cross_down_np = cross_down.fillna(False).to_numpy(dtype=bool)

        for w in windows:
            w = int(w)

            if add_count_cols or add_all_cols:
                above_count = _rolling_count_bool(above, w)
                below_count = _rolling_count_bool(below, w)

                if add_count_cols:
                    new_cols[f"{base}_ABOVE_{level_int}_COUNT_{w}"] = above_count.to_numpy()
                    new_cols[f"{base}_BELOW_{level_int}_COUNT_{w}"] = below_count.to_numpy()

                if add_all_cols:
                    new_cols[f"{base}_ABOVE_{level_int}_ALL_{w}"] = above_count.eq(w).to_numpy(dtype=bool)
                    new_cols[f"{base}_BELOW_{level_int}_ALL_{w}"] = below_count.eq(w).to_numpy(dtype=bool)

            if add_prev_cols:
                prev_above_count = _rolling_count_bool(above.shift(1), w)
                prev_below_count = _rolling_count_bool(below.shift(1), w)

                new_cols[f"{base}_PREV_ABOVE_{level_int}_COUNT_{w}"] = prev_above_count.to_numpy()
                new_cols[f"{base}_PREV_BELOW_{level_int}_COUNT_{w}"] = prev_below_count.to_numpy()

                new_cols[f"{base}_PREV_ABOVE_{level_int}_ALL_{w}"] = prev_above_count.eq(w).to_numpy(dtype=bool)
                new_cols[f"{base}_PREV_BELOW_{level_int}_ALL_{w}"] = prev_below_count.eq(w).to_numpy(dtype=bool)

            if add_cross_cols:
                new_cols[f"{base}_CROSS_UP_{level_int}_LAST_{w}"] = (
                    _rolling_any_bool(cross_up, w).to_numpy(dtype=bool)
                )
                new_cols[f"{base}_CROSS_DOWN_{level_int}_LAST_{w}"] = (
                    _rolling_any_bool(cross_down, w).to_numpy(dtype=bool)
                )

            if add_active_cols:
                new_cols[f"{base}_CROSS_UP_{level_int}_ACTIVE_{w}"] = _active_cross_state(
                    cross_up_np,
                    cross_down_np,
                    lookback=w,
                )

                new_cols[f"{base}_CROSS_DOWN_{level_int}_ACTIVE_{w}"] = _active_cross_state(
                    cross_down_np,
                    cross_up_np,
                    lookback=w,
                )

    features = pd.DataFrame(new_cols, index=df.index)

    # Single concat avoids dataframe fragmentation.
    out = pd.concat([df, features], axis=1)

    # Copy once to de-fragment memory.
    return out.copy()