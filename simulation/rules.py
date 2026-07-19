from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from simulation.context_mixins import RuleContextMixin


@dataclass(frozen=True)
class Ctx(RuleContextMixin):
    df: pd.DataFrame
    i: int

    @property
    def row(self) -> pd.Series:
        return self.df.iloc[self.i]

    def v(self, col: str, shift: int = 0):
        j = self.i + shift
        if j < 0 or j >= len(self.df):
            return np.nan
        return self.df[col].iloc[j]


@dataclass(frozen=True)
class Rule:
    name: str
    fn: Callable[..., bool]
    desc: str = ""

    def eval(self, ctx: Ctx) -> Tuple[bool, List[str]]:
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

        ok_any = any(results) if results else False
        return ok_any, reasons if ok_any else []


def ALL(*items: Union[Rule, RuleGroup], name: str = "") -> RuleGroup:
    return RuleGroup(mode="all", items=list(items), name=name)


def ANY(*items: Union[Rule, RuleGroup], name: str = "") -> RuleGroup:
    return RuleGroup(mode="any", items=list(items), name=name)