"""Parameter optimization: grid sweeps (Bayesian search planned)."""
from __future__ import annotations

from .grid import default_n_jobs, expand_grid, run_grid

__all__ = ["run_grid", "expand_grid", "default_n_jobs"]
