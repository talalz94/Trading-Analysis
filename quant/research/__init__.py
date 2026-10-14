"""
NOTEBOOK-FACING ORCHESTRATION — the code a notebook would otherwise have to spell out.

A notebook here is meant to read as **settings, one call, a chart**. Anything longer than that puts
machinery in front of the person using it: a helper defined in a cell is code you have to scroll
past, re-read to check it still matches the library, and accidentally edit. So the wiring lives
here instead, where it is imported, tested and versioned like everything else.

WHAT BELONGS IN THIS PACKAGE
    Loading the frames a study needs, assembling the indicator columns onto one DataFrame, running
    a scenario the way a notebook wants it run, and handing back an object with the tables and the
    figure already built.

WHAT DOES NOT
    * Strategy logic. An entry rule is a `Scenario` in `quant/scenarios/`, a condition is a `Cond`
      in `quant/strategies/`. If it decides WHEN to trade, it does not go here.
    * Indicator maths. That is `quant/indicators/`.
    * Drawing. That is `quant/viz/`. This package CALLS `pattern_figure`; it does not draw.
    * Experiment logic — sweeps, window scoring, survivor counting. That is `experiments/`, kept
      out of the installed package on purpose (CLAUDE.md section 14).
    * SETTINGS. Every value comes from the caller's config object. Nothing here states a default
      for anything a notebook config owns, because a default stated in two places is the bug that
      cost `TF`, `LINE_TFS` and `WARMUP_START` in commit 8a8b8de.

THE CONTRACT WITH A CONFIG
    These helpers take the config MODULE (`import nb19_config as CFG`) and read attributes off it.
    They never import a config themselves and never write one back.
"""
from __future__ import annotations

from .fvg_lab import FvgLab, fvg_lab
from .variant import variant
from .compare import Arms, compare_runs, run_arms, trade_diff
from .rejected import rejected_setups
from .market_view import MarketView, market_view

__all__ = ["FvgLab", "fvg_lab", "variant", "compare_runs", "trade_diff", "run_arms", "Arms",
           "rejected_setups", "MarketView", "market_view"]
