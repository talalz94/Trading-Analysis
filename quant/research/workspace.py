"""
STRATEGIES AS FILES, NOTEBOOKS AS VIEWS.

Before this, every idea became a new notebook with a copied config: 18 -> 19 -> 20 -> 21, each file
a few hundred lines, each held equal to its parent by a test that broke whenever its owner edited
it. The layout now separates the three things those files mixed:

    notebooks/strategies/<name>.py   WHAT is traded -- entries, exits, costs, the gaps the rules
                                     read, and COMPARE (the questions this strategy is asking)
    notebooks/lab_config.py          WHICH strategy, WHEN (the window) and HOW it is drawn
    notebooks/evaluate_config.py     WHICH strategy, and over which quarters it is judged

`lab.ipynb` and `evaluate.ipynb` read those and state nothing. A new idea is a toggle in a strategy
file and an arm in its COMPARE, or a copied strategy file -- never a new notebook.

EACH SETTING LIVES IN EXACTLY ONE FILE. The merge refuses a setting found in both a strategy and a
view config: two copies of a value is how a change "didn't take" -- the bug the old copy tests were
fighting from the other side.

Files are EXECUTED fresh on every call (runpy), not imported, so an edited strategy is picked up
without a kernel restart or an importlib.reload -- and a stale module can never answer instead.
"""
from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

__all__ = ["NOTEBOOKS", "STRATEGIES", "load_settings", "list_strategies", "load_strategy",
           "lab_config", "evaluate_config", "market_view_config"]

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
STRATEGIES = NOTEBOOKS / "strategies"

PathLike = Union[str, Path]


def load_settings(path: PathLike) -> Dict[str, object]:
    """The UPPER_CASE names a settings file defines, by executing it."""
    ns = runpy.run_path(str(path))
    return {k: v for k, v in ns.items() if k.isupper()}


def list_strategies(folder: PathLike = STRATEGIES) -> List[str]:
    """Every strategy file's name, for `STRATEGY = "..."`."""
    return sorted(p.stem for p in Path(folder).glob("*.py") if not p.stem.startswith("_"))


def load_strategy(name: str, folder: PathLike = STRATEGIES) -> Dict[str, object]:
    path = Path(folder) / f"{name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"no strategy {name!r} in {folder}; have "
                                f"{list_strategies(folder)}")
    return load_settings(path)


def _merge(view: Dict[str, object], folder: PathLike, source: PathLike) -> SimpleNamespace:
    if "STRATEGY" not in view:
        raise KeyError(f"{Path(source).name} must say which strategy: STRATEGY = \"...\" "
                       f"(one of {list_strategies(folder)})")
    strat = load_strategy(str(view["STRATEGY"]), folder)
    both = sorted((set(view) & set(strat)) - {"STRATEGY"})
    if both:
        raise ValueError(
            f"{both} set in BOTH {Path(source).name} and strategies/{view['STRATEGY']}.py. Each "
            f"setting lives in one file: what is traded in the strategy, when and how it is "
            f"drawn in the view.")
    return SimpleNamespace(**strat, **view)


def lab_config(path: PathLike = NOTEBOOKS / "lab_config.py",
               folder: PathLike = STRATEGIES) -> SimpleNamespace:
    """The lab's settings: the strategy file `lab_config.STRATEGY` names, plus the window and the
    drawing settings from `lab_config.py`. Hand it to `fvg_lab`, `run_arms`, `variant`."""
    return _merge(load_settings(path), folder, path)


def market_view_config(path: PathLike = NOTEBOOKS / "market_view_config.py") -> SimpleNamespace:
    """The market view's settings. No strategy file: it draws and never trades."""
    return SimpleNamespace(**load_settings(path))


def evaluate_config(path: PathLike = NOTEBOOKS / "evaluate_config.py",
                    folder: PathLike = STRATEGIES) -> SimpleNamespace:
    """The evaluation's settings: the strategy, plus which quarters train and which judge."""
    return _merge(load_settings(path), folder, path)
