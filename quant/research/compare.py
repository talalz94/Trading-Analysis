"""
TWO RUNS SIDE BY SIDE — the table an A/B study always ends up wanting.

A filter or a confirmation can only ever REMOVE trades or move them later, so the totals alone
cannot tell you whether it removed losers or removed trades at random. Both views are needed: the
summary to see the direction, the per-trade diff to see which rows actually changed.

Kept here rather than in a notebook cell for the reason the package docstring gives: a helper
defined in a cell is code you scroll past and eventually edit by accident.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd

__all__ = ["compare_runs", "trade_diff", "run_arms", "Arms"]

_COLS = ["side", "trigger", "R", "close_reason"]


def _trades(run: Any) -> pd.DataFrame:
    return run if isinstance(run, pd.DataFrame) else run.trades


def compare_runs(**runs: Any) -> pd.DataFrame:
    """One row per run. Pass labs or trade frames by keyword; the order is the column order."""
    rows = []
    for name, run in runs.items():
        t = _trades(run)
        rows.append({"run": name.replace("_", " "), "trades": len(t),
                     "total R": round(float(t["R"].sum()), 2) if len(t) else 0.0,
                     "wins": int((t["R"] > 0).sum()) if len(t) else 0,
                     "R / trade": round(float(t["R"].mean()), 3) if len(t) else 0.0})
    return pd.DataFrame(rows)


def trade_diff(**runs: Any) -> pd.DataFrame:
    """One row per entry time, one block of columns per run — so a row present in one and blank in
    the other is exactly the trade the change added or removed."""
    out = None
    for name, run in runs.items():
        t = _trades(run).copy()
        t.index = pd.Index(t["entry_time"].astype(str).str[:16], name="entry")
        part = t[[c for c in _COLS if c in t.columns]].add_suffix(f"  {name}")
        out = part if out is None else out.join(part, how="outer")
    return out.sort_index() if out is not None else pd.DataFrame()


@dataclass
class Arms:
    """Every arm of an A/B, run. `labs[name]` is that arm's lab; `.table()` and `.diff()` read them."""
    labs: Dict[str, Any]

    def table(self) -> pd.DataFrame:
        t = compare_runs(**self.labs)
        # the exit reasons are what an exit rule changes, so count them per arm
        why = pd.DataFrame({k: _trades(v)["close_reason"].value_counts()
                            for k, v in self.labs.items()}).T.fillna(0).astype(int)
        return t.join(why.reset_index(drop=True))

    def diff(self) -> pd.DataFrame:
        """One row per entry: side and trigger once, then each arm's R and how the trade ended. A
        blank R is a trade that arm did not take -- an exit that closes one trade early frees the
        slot, so an arm can take a trade the others could not."""
        out, meta = None, []
        for name, run in self.labs.items():
            t = _trades(run).copy()
            t.index = pd.Index(t["entry_time"].astype(str).str[:16], name="entry")
            meta.append(t[[c for c in ("side", "trigger") if c in t.columns]])
            part = pd.DataFrame({f"{name} R": t["R"].round(2),
                                 f"{name} exit": t["close_reason"].map(_EXIT).fillna(
                                     t["close_reason"])})
            out = part if out is None else out.join(part, how="outer")
        if out is None:
            return pd.DataFrame()
        m = pd.concat(meta)
        return m[~m.index.duplicated()].join(out, how="right").sort_index()


_EXIT = {"take_profit": "target", "stop_loss": "stop", "forced_close_end": "still open",
         "tp_moved": "moved target", "fvg_exit": "fvg exit", "time_stop": "time"}


def run_arms(cfg: Any, arms: Optional[Dict[str, dict]] = None,
             run: Optional[Callable] = None, base: Any = None) -> Arms:
    """Run `cfg` once per arm, each with ONLY that arm's settings changed (`variant`), so every
    column of the result differs from the config by exactly what its arm names.

    `arms` defaults to the config's own `COMPARE` -- `{"fvg exit": dict(USE_FVG_EXIT=True), ...}` --
    so which A/B a notebook shows is decided where every other setting is, not in a cell.

    `base` is a lab already built from `cfg`: when every arm changes trading settings only, the arms
    re-trade it instead of loading again.
    """
    from .fvg_lab import fvg_lab
    from .variant import variant
    arms = arms if arms is not None else getattr(cfg, "COMPARE", None)
    if not arms:
        # most likely an editor tab saved over the config before COMPARE was added to it
        raise AttributeError(
            "this config has no COMPARE setting. Add one, e.g. "
            'COMPARE = {"neither": dict(USE_TP_MOVE=False), "move target": dict(USE_TP_MOVE=True)}')
    if run is None:
        from .fvg_lab import RETRADE_KEYS
        if all(set(over) <= RETRADE_KEYS for over in arms.values()):
            # trading settings only: load and fit ONCE, then one sim per arm
            base = base if base is not None else fvg_lab(cfg)
            return Arms(labs={name: base.retrade(**over) for name, over in arms.items()})
        run = fvg_lab
    return Arms(labs={name: run(variant(cfg, **over)) for name, over in arms.items()})
