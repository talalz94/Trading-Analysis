"""
TWO RUNS SIDE BY SIDE — the table an A/B study always ends up wanting.

A filter or a confirmation can only ever REMOVE trades or move them later, so the totals alone
cannot tell you whether it removed losers or removed trades at random. Both views are needed: the
summary to see the direction, the per-trade diff to see which rows actually changed.

Kept here rather than in a notebook cell for the reason the package docstring gives: a helper
defined in a cell is code you scroll past and eventually edit by accident.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["compare_runs", "trade_diff"]

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
