"""
"There is an unfilled gap behind this move" as a `Cond` you can AND onto any entry.

The idea being tested: a trendline break that leaves an unmitigated fair value gap behind it is
suspect, because price tends to come back and take the gap before going anywhere. If that is true,
the break is worth skipping rather than trading.

TWO READINGS, BOTH BUILT, because they are opposite filters and only measurement can separate them:

    mode="retrace"   (default)  UNFINISHED BUSINESS BEHIND YOU.
                     A bullish gap sits BELOW price and is filled by a move DOWN, so an unmitigated
                     bullish gap is a downward magnet -- and it is a LONG breakout that dies when
                     price turns round to take it. Longs are blocked by bullish gaps, shorts by
                     bearish ones.

    mode="barrier"   AN OPPOSING IMPULSE AHEAD OF YOU.
                     A bearish gap is where sellers displaced through; buying into it is buying
                     into their zone. Longs are blocked by bearish gaps, shorts by bullish ones.

They are exact mirrors, so a result that looks good under one and bad under the other is a real
signal about which story holds; one that looks the same under both is noise.

WHICH GAPS COUNT. All four gates are optional and all are scale-free:

    max_age_bars    the gap must have formed within this many bars -- IN ITS OWN TIMEFRAME'S BARS,
                    so 20 on a 1h layer is 20 hours, not 20 chart bars. A gap is live from the bar
                    it was confirmed on, so "formed at or before the trigger candle" is automatic.
    min_size_atr    ignore gaps smaller than this (see the caveat below)
    max_dist_atr    ignore gaps price has already walked away from -- a magnet 10 ATR off is not
                    pulling this trade anywhere
    timeframes      which layers may veto; None means every one on the frame

CAVEAT, stated plainly: the per-bar columns describe the NEAREST live gap of each direction, not all
of them, so `min_size_atr` and `max_dist_atr` here are tested against that nearest one. If the
nearest gap is tiny and a large one sits just behind it, this will not see the large one. The way to
get that exactly right is to drop insignificant gaps at DETECTION -- `add_fvgs(min_size_atr=...)`
removes them from the count entirely, so "nearest" becomes "nearest that mattered". That is why the
default here is 0: filter once, at the source, and this gate is then a no-op refinement rather than
a second approximate one.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .conditions import Cond

VETO_MODES = ("retrace", "barrier")

# which gap direction threatens which side, per mode
_BLOCKED_BY = {
    "retrace": {"long": "bull", "short": "bear"},
    "barrier": {"long": "bear", "short": "bull"},
}


def fvg_layers(df: pd.DataFrame, *, prefix: str = "fvg",
               timeframes: Optional[Iterable[str]] = None) -> list:
    """The `{tf}__` prefixes carrying FVG columns on this frame, base first then coarsest-first.

    `timeframes` restricts it; names are matched as written ("15m") and the chart's own layer is
    the empty string, which `None` and `""` both select.
    """
    from ..data.timeframe import tf_minutes

    out = [""] if f"{prefix}_bull_n" in df.columns else []
    extra = sorted({c.split("__", 1)[0] for c in df.columns if c.endswith(f"__{prefix}_bull_n")},
                   key=lambda k: -tf_minutes(k))
    out += [f"{k}__" for k in extra]
    if timeframes is None:
        return out
    want = {("" if t in (None, "", "chart") else str(t)) for t in timeframes}
    return [p for p in out if (p[:-2] if p.endswith("__") else "") in want]


def fvg_blocks(side: str, *, mode: str = "retrace", prefix: str = "fvg",
               max_age_bars: int = 0, min_size_atr: float = 0.0, max_dist_atr: float = 0.0,
               timeframes: Optional[Sequence[str]] = None) -> Cond:
    """TRUE on bars where an unmitigated gap should stop a `side` trade being opened.

    `side` is the TRADE's direction ("long"/"short"); which gap direction threatens it is decided
    by `mode`. Every gate left at 0 is off, so the bare call is "any live gap of the threatening
    direction, on any timeframe present".
    """
    side = str(side).lower()
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")
    if mode not in VETO_MODES:
        raise ValueError(f"mode must be one of {VETO_MODES}, got {mode!r}")
    which = _BLOCKED_BY[mode][side]

    def fn(df: pd.DataFrame) -> np.ndarray:
        layers = fvg_layers(df, prefix=prefix, timeframes=timeframes)
        if not layers:
            raise KeyError(
                f"fvg_blocks: no {prefix}_* columns on the frame. Run mtf_fvgs (or add_fvgs) first "
                f"— and if you passed `timeframes`, check the names match the ones loaded.")
        out = np.zeros(len(df), bool)
        for p in layers:
            n = df[f"{p}{prefix}_{which}_n"].to_numpy()
            hit = n > 0
            # the gates describe the NEAREST live gap of this direction -- see the module docstring
            if max_age_bars > 0:
                hit &= df[f"{p}{prefix}_{which}_age"].to_numpy(float) <= max_age_bars
            if min_size_atr > 0:
                hit &= df[f"{p}{prefix}_{which}_size_atr"].to_numpy(float) >= min_size_atr
            if max_dist_atr > 0:
                hit &= df[f"{p}{prefix}_{which}_dist_atr"].to_numpy(float) <= max_dist_atr
            out |= np.nan_to_num(hit.astype(float), nan=0.0).astype(bool)
        return out

    tfs = "any tf" if timeframes is None else "+".join(map(str, timeframes))
    gates = ",".join(f"{k}={v}" for k, v in
                     (("age", max_age_bars), ("size", min_size_atr), ("dist", max_dist_atr)) if v)
    return Cond(fn, f"fvg_blocks({side},{mode},{which},{tfs}{',' + gates if gates else ''})")


def fvg_clear(side: str, **kwargs) -> Cond:
    """The ANDable one: TRUE where no blocking gap is present, so the trade MAY be taken.

    This is what goes in `entry_filter=` beside the rulespec rules — `entry_filter` is a "may
    enter" condition, and handing it the veto itself would trade exactly the setups you wanted to
    avoid.
    """
    c = fvg_blocks(side, **kwargs)
    out = ~c
    out.name = f"fvg_clear{c.name[len('fvg_blocks'):]}"
    return out
