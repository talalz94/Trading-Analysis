"""
Experiments / inference layer — search for the best settings given an idea.

Sits on top of the `quant` core and never modifies it. Each experiment declares a search space +
objective + description, runs via `Experiment.run()`, and writes a self-contained results folder
(`results.csv`, `best.json`, `report.md`). See README.md and base.Experiment.
"""
from __future__ import annotations

from .base import Experiment

__all__ = ["Experiment"]
