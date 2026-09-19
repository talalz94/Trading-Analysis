"""
ONE SETTING CHANGED — a config object you can hand straight back to a lab.

Every comparison a notebook wants to make is "the same study with one thing different": with the
confirmation and without it, one timeframe's wait at 1 and at 3. Doing that in a cell means writing
a loop and a copy, which is the machinery `quant/research/` exists to keep out of notebooks.

It DEEP-COPIES, so a dict setting (`TL_CLOSE_N`, `COSTS`) cannot be mutated through the copy and
change the original underneath a later run -- the failure mode that makes an A/B silently compare a
thing against itself.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

__all__ = ["variant"]


def variant(cfg: Any, **overrides: Any) -> SimpleNamespace:
    """The same settings with `overrides` applied, as a new object. `cfg` is left alone."""
    out = {k: copy.deepcopy(v) for k, v in vars(cfg).items() if k.isupper()}
    unknown = [k for k in overrides if k not in out]
    if unknown:
        # a typo'd setting name would otherwise be added silently and do nothing, and the A/B would
        # report "no difference" -- which is indistinguishable from the real answer
        raise KeyError(f"not settings on this config: {sorted(unknown)}")
    out.update({k: copy.deepcopy(v) for k, v in overrides.items()})
    return SimpleNamespace(**out)
