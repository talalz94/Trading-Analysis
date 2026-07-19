from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmaPairSpreadSpec:
    """
    Difference between two EMA columns.

    For long-style interpretation:
      left > right is bullish.
    """
    name: str
    left: str
    right: str


@dataclass(frozen=True)
class EmaGroupSpreadSpec:
    """
    Measures how close/far a basket of EMAs is.

    Useful for detecting:
      - compression / chop
      - expansion / trending conditions
    """
    name: str
    cols: Sequence[str]


@dataclass(frozen=True)
class EmaCutThroughSpec:
    """
    Measures how fast one EMA cuts through a group of other EMAs.

    Example:
      leader = 1m EMA50
      others = [1m EMA100, 1m EMA150, 5m EMA100, 15m EMA100]
    """
    name: str
    leader: str
    others: Sequence[str]
    lookbacks: Tuple[int, ...] = (3, 5, 10, 15)


def _safe_pct(num, den):
    return np.where(np.abs(den) > 1e-12, num / den * 100.0, np.nan)


def add_ema_diagnostics(
    df: pd.DataFrame,
    pair_specs: Sequence[EmaPairSpreadSpec] = (),
    group_specs: Sequence[EmaGroupSpreadSpec] = (),
    cut_specs: Sequence[EmaCutThroughSpec] = (),
    dynamic_windows: Tuple[int, ...] = (1000,),
    dynamic_quantiles: Tuple[float, ...] = (0.70, 0.80, 0.85),
    min_periods_ratio: float = 0.20,
) -> pd.DataFrame:
    """
    Adds EMA diagnostic features to an already aligned dataframe.

    Pair spread features:
      {name}__abs
      {name}__pct
      {name}__pct_q80_w1000

    Group spread features:
      {name}__range_abs
      {name}__range_pct
      {name}__std_pct
      {name}__mad_pct

    Cut-through features:
      {name}__above_count
      {name}__below_count
      {name}__rank_pct
      {name}__is_top
      {name}__is_bottom
      {name}__above_count_change_10
      {name}__cross_up_events_10
      {name}__cross_down_events_10
      {name}__cut_up_speed_10
    """
    out = df.copy()

    # ------------------------------------------------------------
    # 1) Pair spreads: EMA A vs EMA B
    # ------------------------------------------------------------
    for spec in pair_specs:
        if spec.left not in out.columns:
            raise KeyError(f"Missing left EMA column: {spec.left}")
        if spec.right not in out.columns:
            raise KeyError(f"Missing right EMA column: {spec.right}")

        left = pd.to_numeric(out[spec.left], errors="coerce")
        right = pd.to_numeric(out[spec.right], errors="coerce")

        abs_col = f"{spec.name}__abs"
        pct_col = f"{spec.name}__pct"

        out[abs_col] = left - right
        out[pct_col] = _safe_pct(left - right, right)

        for window in dynamic_windows:
            window = int(window)
            min_periods = max(50, int(window * min_periods_ratio))

            for q in dynamic_quantiles:
                q_int = int(round(q * 100))
                q_col = f"{pct_col}_q{q_int}_w{window}"

                out[q_col] = (
                    out[pct_col]
                    .rolling(window, min_periods=min_periods)
                    .quantile(q)
                    .shift(1)
                )

    # ------------------------------------------------------------
    # 2) Group spreads: how close/far EMA basket is
    # ------------------------------------------------------------
    for spec in group_specs:
        missing = [c for c in spec.cols if c not in out.columns]
        if missing:
            raise KeyError(f"Missing EMA group columns for {spec.name}: {missing}")

        mat = out[list(spec.cols)].apply(pd.to_numeric, errors="coerce")

        row_max = mat.max(axis=1)
        row_min = mat.min(axis=1)
        row_mean = mat.mean(axis=1)
        row_std = mat.std(axis=1)

        range_abs_col = f"{spec.name}__range_abs"
        range_pct_col = f"{spec.name}__range_pct"
        std_pct_col = f"{spec.name}__std_pct"
        mad_pct_col = f"{spec.name}__mad_pct"

        out[range_abs_col] = row_max - row_min
        out[range_pct_col] = _safe_pct(row_max - row_min, row_mean)
        out[std_pct_col] = _safe_pct(row_std, row_mean)

        mad_abs = mat.sub(row_mean, axis=0).abs().mean(axis=1)
        out[mad_pct_col] = _safe_pct(mad_abs, row_mean)

        for base_col in [range_pct_col, std_pct_col, mad_pct_col]:
            for window in dynamic_windows:
                window = int(window)
                min_periods = max(50, int(window * min_periods_ratio))

                for q in dynamic_quantiles:
                    q_int = int(round(q * 100))
                    q_col = f"{base_col}_q{q_int}_w{window}"

                    out[q_col] = (
                        out[base_col]
                        .rolling(window, min_periods=min_periods)
                        .quantile(q)
                        .shift(1)
                    )

    # ------------------------------------------------------------
    # 3) Cut-through speed: selected EMA moving through others
    # ------------------------------------------------------------
    for spec in cut_specs:
        if spec.leader not in out.columns:
            raise KeyError(f"Missing leader EMA column: {spec.leader}")

        missing = [c for c in spec.others if c not in out.columns]
        if missing:
            raise KeyError(f"Missing cut-through comparison columns for {spec.name}: {missing}")

        leader = pd.to_numeric(out[spec.leader], errors="coerce")
        others = out[list(spec.others)].apply(pd.to_numeric, errors="coerce")

        above_matrix = others.lt(leader, axis=0)
        below_matrix = others.gt(leader, axis=0)

        above_count_col = f"{spec.name}__above_count"
        below_count_col = f"{spec.name}__below_count"
        rank_pct_col = f"{spec.name}__rank_pct"
        is_top_col = f"{spec.name}__is_top"
        is_bottom_col = f"{spec.name}__is_bottom"

        n_others = len(spec.others)

        out[above_count_col] = above_matrix.sum(axis=1)
        out[below_count_col] = below_matrix.sum(axis=1)
        out[rank_pct_col] = out[above_count_col] / max(n_others, 1)

        out[is_top_col] = out[above_count_col] == n_others
        out[is_bottom_col] = out[below_count_col] == n_others

        # Pairwise cross events.
        cross_up_events = pd.DataFrame(index=out.index)
        cross_down_events = pd.DataFrame(index=out.index)

        for other_col in spec.others:
            other = pd.to_numeric(out[other_col], errors="coerce")

            clean = other_col.replace("__", "_")

            cross_up_events[clean] = (leader > other) & (leader.shift(1) <= other.shift(1))
            cross_down_events[clean] = (leader < other) & (leader.shift(1) >= other.shift(1))

        for lb in spec.lookbacks:
            lb = int(lb)

            above_change_col = f"{spec.name}__above_count_change_{lb}"
            cross_up_count_col = f"{spec.name}__cross_up_events_{lb}"
            cross_down_count_col = f"{spec.name}__cross_down_events_{lb}"
            cut_speed_col = f"{spec.name}__cut_up_speed_{lb}"
            became_top_col = f"{spec.name}__became_top_in_{lb}"

            out[above_change_col] = out[above_count_col] - out[above_count_col].shift(lb)

            out[cross_up_count_col] = (
                cross_up_events
                .rolling(lb, min_periods=1)
                .sum()
                .sum(axis=1)
            )

            out[cross_down_count_col] = (
                cross_down_events
                .rolling(lb, min_periods=1)
                .sum()
                .sum(axis=1)
            )

            out[cut_speed_col] = out[above_change_col] / lb

            out[became_top_col] = (
                out[is_top_col]
                & (~out[is_top_col].shift(lb).fillna(False).astype(bool))
            )

    return out