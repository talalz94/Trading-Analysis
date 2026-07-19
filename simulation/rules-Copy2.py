from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Union, Optional
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Ctx:
    df: pd.DataFrame
    i: int

    @property
    def row(self) -> pd.Series:
        return self.df.iloc[self.i]

    def v(self, col: str, shift: int = 0) -> float:
        j = self.i + shift
        if j < 0 or j >= len(self.df):
            return np.nan
        return self.df[col].iloc[j]

    def is_finite(self, col: str, shift: int = 0) -> bool:
        return bool(np.isfinite(self.v(col, shift)))

    # ---------------------------
    # Level-based helpers (existing)
    # ---------------------------
    def cross_up(self, col: str, level: float, lookback: int = 1, inclusive: bool = True) -> bool:
        cur = self.v(col, 0)
        if not np.isfinite(cur):
            return False
        cur_ok = (cur >= level) if inclusive else (cur > level)
        if not cur_ok:
            return False
        for k in range(1, lookback + 1):
            prev = self.v(col, -k)
            if np.isfinite(prev) and prev < level:
                return True
        return False

    def cross_down(self, col: str, level: float, lookback: int = 1, inclusive: bool = True) -> bool:
        cur = self.v(col, 0)
        if not np.isfinite(cur):
            return False
        cur_ok = (cur <= level) if inclusive else (cur < level)
        if not cur_ok:
            return False
        for k in range(1, lookback + 1):
            prev = self.v(col, -k)
            if np.isfinite(prev) and prev > level:
                return True
        return False

    def prev_all_below(self, col: str, level: float, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(1, n + 1):
            prev = self.v(col, -k)
            if not np.isfinite(prev) or not (prev < level):
                return False
        return True

    def prev_all_above(self, col: str, level: float, n: int) -> bool:
        if n <= 0:
            return True
        for k in range(1, n + 1):
            prev = self.v(col, -k)
            if not np.isfinite(prev) or not (prev > level):
                return False
        return True

    # ---------------------------
    # ✅ Pair-based helpers (Option B)
    # e.g. close vs MA50, K vs D, etc.
    # ---------------------------
    def cross_up_pair(self, a: str, b: str, lookback: int = 1) -> bool:
        """
        True if a is currently > b AND within lookback bars there exists a bar where a < b.
        Example: cross up MA => a='close', b='MA50'
        """
        cur_a, cur_b = self.v(a, 0), self.v(b, 0)
        if not (np.isfinite(cur_a) and np.isfinite(cur_b)):
            return False
        if not (cur_a > cur_b):
            return False
        for k in range(1, lookback + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if np.isfinite(pa) and np.isfinite(pb) and (pa < pb):
                return True
        return False

    def cross_down_pair(self, a: str, b: str, lookback: int = 1) -> bool:
        """
        True if a is currently < b AND within lookback bars there exists a bar where a > b.
        """
        cur_a, cur_b = self.v(a, 0), self.v(b, 0)
        if not (np.isfinite(cur_a) and np.isfinite(cur_b)):
            return False
        if not (cur_a < cur_b):
            return False
        for k in range(1, lookback + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if np.isfinite(pa) and np.isfinite(pb) and (pa > pb):
                return True
        return False

    def prev_all_below_pair(self, a: str, b: str, n: int) -> bool:
        """
        True if for the previous n candles: a < b.
        Example: previous 5 closes below MA50.
        """
        if n <= 0:
            return True
        for k in range(1, n + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if not (np.isfinite(pa) and np.isfinite(pb) and pa < pb):
                return False
        return True

    def prev_all_above_pair(self, a: str, b: str, n: int) -> bool:
        """
        True if for the previous n candles: a > b.
        """
        if n <= 0:
            return True
        for k in range(1, n + 1):
            pa, pb = self.v(a, -k), self.v(b, -k)
            if not (np.isfinite(pa) and np.isfinite(pb) and pa > pb):
                return False
        return True

@dataclass(frozen=True)
class Rule:
    name: str
    fn: Callable[..., bool]
    desc: str = ""

    def eval(self, ctx: Ctx) -> Tuple[bool, List[str]]:
        # Backwards compatible: allow fn(row) or fn(ctx)
        try:
            ok = bool(self.fn(ctx))
        except TypeError:
            ok = bool(self.fn(ctx.row))
        return ok, ([self.name] if ok else [])


@dataclass
class RuleGroup:
    mode: str
    items: Sequence[Union["Rule", "RuleGroup"]]
    name: str = ""

    def eval(self, ctx: Ctx) -> Tuple[bool, List[str]]:
        mode = self.mode.lower().strip()
        if mode not in ("all", "any"):
            raise ValueError("RuleGroup.mode must be 'all' or 'any'")

        reasons: List[str] = []
        results: List[bool] = []

        for it in self.items:
            ok, rs = it.eval(ctx)  # type: ignore
            results.append(ok)
            if ok:
                reasons.extend(rs)

        if mode == "all":
            ok_all = all(results) if results else True
            return ok_all, reasons if ok_all else []
        else:
            ok_any = any(results) if results else False
            return ok_any, reasons if ok_any else []


def ALL(*items: Union[Rule, RuleGroup], name: str = "") -> RuleGroup:
    return RuleGroup(mode="all", items=list(items), name=name)

def ANY(*items: Union[Rule, RuleGroup], name: str = "") -> RuleGroup:
    return RuleGroup(mode="any", items=list(items), name=name)