"""
Versioned, composable trading scenarios.

A scenario run = Scenario (entry logic) × StopMethod × TPMethod × params, bundled into a `Preset`
(JSON) that reproduces the backtest on any symbol / timeframe. Importing this package registers the
shipped scenarios / stop methods / TP methods into the registries.

    from quant.scenarios import run_pieces, SCENARIOS, STOP_METHODS, TP_METHODS
    r = run_pieces(df, "support_break_short@v1.0", "swing_high@v1", "rr@v1",
                   params={"tp_rr": 2.0}, cfg={"spread": 0.25})
    print(r.stats); r.trades

See `support_break.py` (scenario family), `stops.py`, `takeprofits.py` for the versioned templates,
and `notebooks/10_scenario_lab.ipynb` for the fully-annotated chart + variant sweep.
"""
from __future__ import annotations

from .base import (Preset, Scenario, ScenarioRun, SecondaryMethod, StopMethod, TPMethod, compose,
                   merged_params, register_scenario, register_secondary, register_stop, register_tp,
                   resolve_scenario, resolve_secondary, resolve_secondaries, resolve_stop, resolve_tp,
                   run, run_pieces,
                   SCENARIOS, SECONDARIES, STOP_METHODS, TP_METHODS)
# Import the template modules for their registration side effects.
from . import stops as _stops          # noqa: F401
from . import takeprofits as _tps      # noqa: F401
from . import secondaries as _sec      # noqa: F401
from . import support_break as _sb     # noqa: F401
from . import resistance_break as _rb  # noqa: F401
from . import entries as _entries      # noqa: F401

__all__ = [
    "Scenario", "StopMethod", "TPMethod", "SecondaryMethod", "Preset", "ScenarioRun",
    "SCENARIOS", "STOP_METHODS", "TP_METHODS", "SECONDARIES",
    "register_scenario", "register_stop", "register_tp", "register_secondary",
    "resolve_scenario", "resolve_stop", "resolve_tp", "resolve_secondary", "resolve_secondaries",
    "compose", "merged_params", "run", "run_pieces",
]
