from __future__ import annotations

from typing import Callable, Union, Any
import pandas as pd

Ref = Union[str, int, float, Callable[[pd.DataFrame], Any]]


def _as_series(df: pd.DataFrame, ref: Ref) -> pd.Series:
    if isinstance(ref, str):
        return pd.to_numeric(df[ref], errors="coerce")

    if callable(ref):
        out = ref(df)
        if isinstance(out, pd.Series):
            return pd.to_numeric(out.reindex(df.index), errors="coerce")
        return pd.Series(float(out), index=df.index)

    return pd.Series(float(ref), index=df.index)


def _compare(left: pd.Series, op: str, right: pd.Series) -> pd.Series:
    op = op.strip()

    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == "==":
        return left == right
    if op == "!=":
        return left != right

    raise ValueError(f"Unsupported operator: {op}")


def add_last_n_compare(
    df: pd.DataFrame,
    left: Ref,
    op: str,
    right: Ref,
    n: int,
    out_col: str,
    include_current: bool = True,
) -> pd.DataFrame:
    """
    Adds boolean column:
      condition is true for last n bars.

    If include_current=True:
      current + previous n-1 bars.

    If include_current=False:
      previous n completed bars only.
    """
    out = df.copy()

    left_s = _as_series(out, left)
    right_s = _as_series(out, right)

    cond = _compare(left_s, op, right_s).fillna(False)

    if not include_current:
        cond = cond.shift(1).fillna(False)

    out[out_col] = (
        cond.astype(int)
        .rolling(int(n), min_periods=int(n))
        .sum()
        .eq(int(n))
        .fillna(False)
    )

    return out


def add_cross_compare(
    df: pd.DataFrame,
    left: Ref,
    direction: str,
    right: Ref,
    out_col: str,
    lookback: int = 1,
    inclusive: bool = True,
) -> pd.DataFrame:
    """
    Adds dynamic cross boolean column.

    direction:
      "up"   => left crosses above right
      "down" => left crosses below right
    """
    out = df.copy()

    left_s = _as_series(out, left)
    right_s = _as_series(out, right)

    diff = left_s - right_s

    direction = direction.lower().strip()

    if direction == "up":
        current = diff.ge(0) if inclusive else diff.gt(0)
        previous = diff.shift(1).lt(0).rolling(int(lookback), min_periods=1).max().fillna(False).astype(bool)

    elif direction == "down":
        current = diff.le(0) if inclusive else diff.lt(0)
        previous = diff.shift(1).gt(0).rolling(int(lookback), min_periods=1).max().fillna(False).astype(bool)

    else:
        raise ValueError("direction must be 'up' or 'down'.")

    out[out_col] = (current & previous).fillna(False)
    return out