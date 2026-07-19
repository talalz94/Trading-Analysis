from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple, Union
import pandas as pd


@dataclass(frozen=True)
class Rule:
    name: str
    fn: Callable[[pd.Series], bool]
    desc: str = ""

    def eval(self, row: pd.Series) -> Tuple[bool, List[str]]:
        ok = bool(self.fn(row))
        return ok, ([self.name] if ok else [])


@dataclass
class RuleGroup:
    """
    mode:
      - "all": every item must be True
      - "any": at least one item must be True
    """
    mode: str
    items: Sequence[Union["Rule", "RuleGroup"]]
    name: str = ""

    def eval(self, row: pd.Series) -> Tuple[bool, List[str]]:
        mode = self.mode.lower().strip()
        if mode not in ("all", "any"):
            raise ValueError("RuleGroup.mode must be 'all' or 'any'")

        reasons: List[str] = []
        results: List[bool] = []

        for it in self.items:
            ok, rs = it.eval(row)  # type: ignore
            results.append(ok)
            if ok:
                reasons.extend(rs)

        if mode == "all":
            ok_all = all(results) if results else True
            return ok_all, reasons if ok_all else []
        else:
            ok_any = any(results) if results else False
            return ok_any, reasons if ok_any else []


# Convenience helpers
def ALL(*items: Union[Rule, RuleGroup], name: str = "") -> RuleGroup:
    return RuleGroup(mode="all", items=list(items), name=name)

def ANY(*items: Union[Rule, RuleGroup], name: str = "") -> RuleGroup:
    return RuleGroup(mode="any", items=list(items), name=name)