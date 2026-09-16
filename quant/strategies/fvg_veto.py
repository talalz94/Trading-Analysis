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

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

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
               timeframes: Optional[Sequence[str]] = None, which: Optional[str] = None) -> Cond:
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
    # `which` names the gap direction ABSOLUTELY ("bull"/"bear"); leave it None and `mode` derives
    # it from the trade's side, which is what "opposite"/"same" mean in a rule sentence.
    if which is None:
        which = _BLOCKED_BY[mode][side]
    elif which not in ("bull", "bear"):
        raise ValueError(f"which must be 'bull' or 'bear', got {which!r}")

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


# =================================================================================================
# A ONE-LINE LANGUAGE FOR GAP VETOES, so a config can say the rule instead of spelling out a call.
# =================================================================================================
#
#     "short: opposite FVG 15m within 20 candles min 0.8 ATR near 3 ATR"
#     "long: no bull fvg within 40 candles"
#     "opposite fvg 1h+4h min 1.0 atr"
#
# Order does not matter and filler words ("no", "fvg", "candles", "bars", "the") are ignored, the
# same deal as `quant.strategies.rulespec`. Every clause is optional except the DIRECTION.
#
#     DIRECTION   opposite | against  the gap whose FILL would move price AGAINST the trade.
#                                      For a SHORT that is a BEARISH gap (it sits ABOVE and is
#                                      filled by price rising, which is the sweep that kills the
#                                      short); for a LONG it is a BULLISH gap below. Say it as
#                                      "the gap that would take me out".
#                 same | with          the mirror: the gap on the other side of that argument.
#                 bull | bear          absolute, if you would rather say it outright
#     TIMEFRAME   15m  ·  1h+4h  ·  chart  ·  any        (default: every layer loaded)
#     within <n>  the gap formed within n bars OF ITS OWN TIMEFRAME       (default: no age limit)
#     min <x>     the gap is at least x ATR tall                          (default: 0, no limit)
#     near <x>    price is within x ATR of it                             (default: 0, no limit)
#
# A rule BLOCKS. `build_fvg_rules(specs, side)` returns the ANDed "may enter" condition, so it goes
# straight into `entry_filter=` beside the rulespec rules.

_FVG_DIR = {"opposite": None, "against": None, "same": "same", "with": "same",
            "bull": "bull", "bullish": "bull", "bear": "bear", "bearish": "bear"}
_FVG_FILLER = {"no", "a", "an", "the", "fvg", "fvgs", "gap", "gaps", "candles", "candle",
               "bars", "bar", "atr", "is", "if", "there"}


@dataclass(frozen=True)
class FvgRule:
    """One parsed sentence."""
    direction: Optional[str]                 # "bull" | "bear" | "same" | None (= opposite)
    timeframes: Optional[Tuple[str, ...]] = None
    within: int = 0
    min_atr: float = 0.0
    near_atr: float = 0.0
    sides: Tuple[str, ...] = ("long", "short")
    text: str = ""


def parse_fvg_rule(text: str) -> FvgRule:
    """One sentence -> one `FvgRule`. Raises `ValueError` naming the problem."""
    raw = str(text).strip()
    body, sides = raw, ("long", "short")
    m = re.match(r"^\s*(long|short)\s*:\s*(.*)$", raw, re.I)
    if m:
        sides, body = (m.group(1).lower(),), m.group(2)

    direction = "MISSING"
    tfs: List[str] = []
    within, min_atr, near_atr = 0, 0.0, 0.0
    toks = [t for t in re.split(r"[\s,]+", body.strip()) if t]
    i = 0
    while i < len(toks):
        low = toks[i].lower().strip(".")
        if low in _FVG_FILLER:
            i += 1
            continue
        if low in _FVG_DIR:
            direction = _FVG_DIR[low]
            i += 1
            continue
        if low in ("within", "last", "past") and i + 1 < len(toks):
            within = int(float(toks[i + 1]))
            i += 2
            continue
        if low in ("min", "bigger", "over", "at") and i + 1 < len(toks):
            min_atr = float(toks[i + 1])
            i += 2
            continue
        if low in ("near", "within_atr", "closer") and i + 1 < len(toks):
            near_atr = float(toks[i + 1])
            i += 2
            continue
        if low in ("any", "all"):
            tfs = []
            i += 1
            continue
        # anything left that looks like a timeframe list: "15m", "1h+4h", "chart"
        parts = [q for q in re.split(r"[+/]", toks[i]) if q]
        if all(re.fullmatch(r"(chart|\d+(m|h|d))", q, re.I) for q in parts):
            tfs += [q.lower() for q in parts]
            i += 1
            continue
        raise ValueError(f"fvg rule {raw!r}: cannot read {toks[i]!r}. Expected a direction "
                         f"(opposite/same/bull/bear), a timeframe (15m, 1h+4h, chart, any), or "
                         f"`within N` / `min X` / `near X`")

    if direction == "MISSING":
        raise ValueError(f"fvg rule {raw!r}: no direction. Say `opposite`, `same`, `bull` or `bear`")
    return FvgRule(direction=direction, timeframes=tuple(tfs) or None, within=within,
                   min_atr=min_atr, near_atr=near_atr, sides=tuple(sides), text=raw)


def fvg_rule_cond(spec, side: str, **kwargs) -> Cond:
    """One `FvgRule` (or its sentence) -> the BLOCKING condition for `side`."""
    r = spec if isinstance(spec, FvgRule) else parse_fvg_rule(spec)
    if r.direction in (None, "same"):
        mode, which = ("retrace" if r.direction is None else "barrier"), None
    else:
        mode, which = "retrace", r.direction
    return fvg_blocks(side, mode=mode, which=which, timeframes=r.timeframes,
                      max_age_bars=r.within, min_size_atr=r.min_atr,
                      max_dist_atr=r.near_atr, **kwargs)


def build_fvg_rules(specs: Optional[Sequence], side: str, **kwargs) -> Optional[Cond]:
    """Every rule that applies to `side`, as ONE "may enter" condition, or None if there are none.

    None means "no rule for this side", NOT "nothing passes" -- the same trap `build_rules` avoids.
    Rules are ANDed as BLOCKS, so the allow condition is `~(block1 | block2 | ...)`: a trade is
    taken when none of the sentences objects.
    """
    parsed = [sp if isinstance(sp, FvgRule) else parse_fvg_rule(sp) for sp in (specs or [])]
    mine = [sp for sp in parsed if side in sp.sides]
    if not mine:
        return None
    blocks = [fvg_rule_cond(sp, side, **kwargs) for sp in mine]
    out = blocks[0]
    for b in blocks[1:]:
        out = out | b
    clear = ~out
    clear.name = "fvg_rules(" + " | ".join(sp.text or "?" for sp in mine) + ")"
    return clear


def describe_fvg_rules(specs: Optional[Sequence]) -> "pd.DataFrame":
    """What each sentence parsed to -- the cheapest way to catch one that does not say what you
    meant before it silently removes every entry."""
    rows = []
    for sp in (specs or []):
        r = sp if isinstance(sp, FvgRule) else parse_fvg_rule(sp)
        # the RESOLVED gap direction per side, because "opposite" is the one word here anyone
        # could read two ways, and reading it backwards inverts the filter
        res = {sd: (r.direction if r.direction in ("bull", "bear")
                    else _BLOCKED_BY["barrier" if r.direction == "same" else "retrace"][sd])
               for sd in ("long", "short")}
        rows.append(dict(rule=r.text, says=("opposite" if r.direction is None else r.direction),
                         blocks_long_on=res["long"] + "ish gap",
                         blocks_short_on=res["short"] + "ish gap",
                         timeframes="any" if not r.timeframes else "+".join(r.timeframes),
                         within_bars=r.within or "-", min_atr=r.min_atr or "-",
                         near_atr=r.near_atr or "-", applies_to="+".join(r.sides)))
    return pd.DataFrame(rows, columns=["rule", "says", "blocks_long_on", "blocks_short_on",
                                       "timeframes", "within_bars", "min_atr", "near_atr",
                                       "applies_to"])
