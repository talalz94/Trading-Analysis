from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import re
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CrossThroughSpec:
    """
    Precomputes how many refs a leader crossed through in the recent lookback.

    Example:
      leader = 1m EMA50
      refs   = [1m EMA100, 1m EMA150, 5m EMA100, 15m EMA100]
      direction = "up"

    Output:
      {name}__count
    """
    name: str
    leader: str
    refs: Sequence[str]
    direction: str = "up"              # "up" or "down"
    lookback: int = 10
    include_current: bool = True
    require_current_side: bool = True
    add_debug_cols: bool = False       # optional per-ref booleans


def _rolling_any_bool(arr: np.ndarray, window: int, include_current: bool = True) -> np.ndarray:
    """
    Fast rolling any for boolean arrays.

    include_current=True:
      at row i checks [i-window+1 ... i]

    include_current=False:
      at row i checks [i-window ... i-1]
    """
    arr = np.asarray(arr, dtype=np.int8)
    n = len(arr)

    if n == 0:
        return np.array([], dtype=bool)

    window = int(window)
    if window <= 0:
        return np.zeros(n, dtype=bool)

    cs = np.concatenate(([0], np.cumsum(arr, dtype=np.int64)))
    idx = np.arange(n)

    if include_current:
        start = np.maximum(0, idx - window + 1)
        end = idx + 1
    else:
        start = np.maximum(0, idx - window)
        end = idx

    counts = cs[end] - cs[start]
    return counts > 0


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", s.replace("__", "_"))


def add_cross_through_features(
    df: pd.DataFrame,
    specs: Sequence[CrossThroughSpec],
) -> pd.DataFrame:
    """
    Vectorized cross-through feature generation.

    This replaces slow runtime rules like:
      c.crossed_up_through_at_least(...)

    with a fast rule:
      c.gte("some_name__count", min_crosses)
    """
    out = df.copy()
    n = len(out)

    for spec in specs:
        direction = spec.direction.lower().strip()
        lookback = int(spec.lookback)

        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'.")

        if spec.leader not in out.columns:
            raise KeyError(f"Missing leader column: {spec.leader}")

        refs = []
        for ref in spec.refs:
            if ref == spec.leader:
                continue
            if ref not in out.columns:
                raise KeyError(f"Missing ref column: {ref}")
            refs.append(ref)

        leader = pd.to_numeric(out[spec.leader], errors="coerce").to_numpy(dtype=float)

        prev_leader = np.empty(n, dtype=float)
        prev_leader[0] = np.nan
        prev_leader[1:] = leader[:-1]

        count = np.zeros(n, dtype=np.int16)

        for ref_col in refs:
            ref = pd.to_numeric(out[ref_col], errors="coerce").to_numpy(dtype=float)

            prev_ref = np.empty(n, dtype=float)
            prev_ref[0] = np.nan
            prev_ref[1:] = ref[:-1]

            if direction == "up":
                cross_event = (leader > ref) & (prev_leader <= prev_ref)

                if spec.require_current_side:
                    current_side_ok = leader > ref
                else:
                    current_side_ok = np.ones(n, dtype=bool)

            else:
                cross_event = (leader < ref) & (prev_leader >= prev_ref)

                if spec.require_current_side:
                    current_side_ok = leader < ref
                else:
                    current_side_ok = np.ones(n, dtype=bool)

            recent_cross = _rolling_any_bool(
                cross_event,
                window=lookback,
                include_current=spec.include_current,
            )

            passed = recent_cross & current_side_ok
            count += passed.astype(np.int16)

            if spec.add_debug_cols:
                debug_col = f"{spec.name}__{direction}__{_safe_name(ref_col)}"
                out[debug_col] = passed

        out[f"{spec.name}__count"] = count

    return out