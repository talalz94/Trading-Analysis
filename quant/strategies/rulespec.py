"""
A one-line language for "n candles, against this level, doing this".

    "2 candles EMA_5m_200 close above"
    "3 candles EMA_15m_50 formed fully below red"
    "1 candle SUP_15m touched"

The Cond algebra in `conditions.py` can already express all of these, but only as code, and a
notebook config is not the place for code — `(full_above(ema(200), n=3) & red(3))` is three ideas
spelled as one expression, and swapping a timeframe means editing it rather than a token. This
parses the sentence instead, so a strategy stays a list of strings you can read down a column.

GRAMMAR — order does not matter, and filler words ("candles", "the", "is") are ignored:

    <n>          how many CONSECUTIVE candles, ending at the bar being tested
    <reference>  EMA_<tf>_<period>  ·  RES_<tf> / SUP_<tf>  ·  or any column already on the frame
    <relation>   close | full (a.k.a. "formed fully") | touch
    <side>       above | below      — omitted for `touch`, where it means nothing
    <colour>     red | green | NA

`long:` / `short:` at the front restricts a rule to that side; without it the rule applies to both.

WHAT THE RELATIONS MEAN, precisely:

    close   the candle's CLOSE is above/below the level
    full    the WHOLE candle is, wicks included (low > level, or high < level)
    touch   the candle's range CONTAINS the level (low <= level <= high)

`full` is much stricter than `close` and they are not interchangeable: a candle can close below a
support and still have its high above it. `touch` is neither — it is the candle straddling the
level, which is what "respected it" looks like.

Colour is literal: `red` means red, on both sides. A rule that should flip with the trade's
direction is the entry family's own `tl_dir_n`, not this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .conditions import Cond

_REL = {"close": "close", "closed": "close", "closes": "close", "closing": "close",
        "full": "full", "fully": "full", "formed": "full", "forms": "full", "wholly": "full",
        "touch": "touch", "touched": "touch", "touches": "touch", "touching": "touch"}
_SIDE = {"above": "above", "over": "above", "up": "above",
         "below": "below", "under": "below", "down": "below"}
_COLOUR = {"red": "red", "bear": "red", "bearish": "red",
           "green": "green", "bull": "green", "bullish": "green",
           "na": None, "any": None, "none": None, "either": None}
_FILLER = {"candle", "candles", "bar", "bars", "the", "is", "are", "of", "a", "an", "and",
           "that", "which", "then", "with"}

_EMA_RE = re.compile(r"^ema[_\-]([0-9a-z]+)[_\-](\d+)$", re.I)
_LINE_RE = re.compile(r"^(res|sup|resistance|support)[_\-]([0-9a-z]+)$", re.I)


@dataclass(frozen=True)
class RuleSpec:
    """One parsed sentence."""
    n: int
    ref: str
    rel: str                      # "close" | "full" | "touch"
    side: Optional[str] = None    # "above" | "below" | None (touch, or unspecified)
    colour: Optional[str] = None  # "red" | "green" | None
    sides: Tuple[str, ...] = ("long", "short")
    text: str = ""

    def __str__(self) -> str:
        return self.text or (f"{self.n} candles {self.ref} {self.rel} "
                             f"{self.side or ''} {self.colour or ''}".strip())


def parse_rule(text: str) -> RuleSpec:
    """One sentence -> one `RuleSpec`. Raises `ValueError` naming the problem, not a KeyError."""
    raw = str(text).strip()
    body, sides = raw, ("long", "short")
    m = re.match(r"^\s*(long|short)\s*:\s*(.*)$", raw, re.I)
    if m:
        sides, body = (m.group(1).lower(),), m.group(2)

    n = ref = None
    rel = side = colour = None
    seen_colour = False
    for tok in re.split(r"[\s,]+", body.strip()):
        if not tok:
            continue
        low = tok.lower().strip(".")
        if low in _FILLER:
            continue
        if n is None and low.isdigit():
            n = int(low)
            continue
        if low in _REL:
            rel = _REL[low]
            continue
        if low in _SIDE:
            side = _SIDE[low]
            continue
        if low in _COLOUR:
            colour, seen_colour = _COLOUR[low], True
            continue
        if ref is not None:
            raise ValueError(f"rule {raw!r}: two references, {ref!r} and {tok!r} — expected one")
        ref = tok

    if ref is None:
        raise ValueError(f"rule {raw!r}: no reference. Expected something like EMA_5m_200, "
                         f"RES_15m, or a column already on the frame")
    if rel is None:
        raise ValueError(f"rule {raw!r}: no relation. Expected one of close / full / touch")
    if rel != "touch" and side is None:
        raise ValueError(f"rule {raw!r}: {rel!r} needs `above` or `below`")
    if rel == "touch" and side is not None:
        # a touch is the candle straddling the level; "above" would be a different test entirely
        raise ValueError(f"rule {raw!r}: `touch` takes no side — a touch is neither above nor below")
    _ = seen_colour
    return RuleSpec(n=max(1, int(n or 1)), ref=ref, rel=rel, side=side, colour=colour,
                    sides=tuple(sides), text=raw)


def resolve_ref(ref: str, *, chart_tf: Optional[str] = None) -> str:
    """A reference -> the column it names. `EMA_5m_200` on a 5m chart is `ema_200`, on a 1m chart
    `5m__ema_200`; `RES_15m` is `15m__tl_res`. Anything else is taken as a column name as written."""
    m = _EMA_RE.match(ref)
    if m:
        tf, per = m.group(1), int(m.group(2))
        return f"ema_{per}" if (chart_tf and tf.lower() == chart_tf.lower()) else f"{tf}__ema_{per}"
    m = _LINE_RE.match(ref)
    if m:
        kind = "res" if m.group(1).lower().startswith("res") else "sup"
        tf = m.group(2)
        return f"tl_{kind}" if (chart_tf and tf.lower() == chart_tf.lower()) else f"{tf}__tl_{kind}"
    return ref


def required_emas(specs: Iterable, *, chart_tf: Optional[str] = None) -> Dict[str, List[int]]:
    """`{timeframe: [periods]}` the rules need, so a caller can add exactly those columns.

    The chart's own timeframe comes back under the key `""`.
    """
    out: Dict[str, List[int]] = {}
    for sp in specs:
        sp = sp if isinstance(sp, RuleSpec) else parse_rule(sp)
        m = _EMA_RE.match(sp.ref)
        if not m:
            continue
        tf, per = m.group(1), int(m.group(2))
        key = "" if (chart_tf and tf.lower() == chart_tf.lower()) else tf
        if per not in out.setdefault(key, []):
            out[key].append(per)
    return out


def _runs(flag: np.ndarray, n: int) -> np.ndarray:
    """True where `flag` has held for the last `n` bars INCLUDING this one."""
    if n <= 1:
        return flag.copy()
    out = flag.copy()
    for j in range(1, n):
        out[j:] &= flag[:-j]
        out[:j] = False
    return out


def rule_cond(spec, *, chart_tf: Optional[str] = None) -> Cond:
    """One `RuleSpec` (or its sentence) -> a `Cond` you can AND onto anything."""
    sp = spec if isinstance(spec, RuleSpec) else parse_rule(spec)
    col = resolve_ref(sp.ref, chart_tf=chart_tf)

    def fn(df: pd.DataFrame) -> np.ndarray:
        if col not in df.columns:
            raise KeyError(f"rule {sp.text or sp!s}: column {col!r} is not on the frame. "
                           f"Add it first — `required_emas()` says which EMAs a rule set needs.")
        lv = df[col].to_numpy(float)
        o = df["open"].to_numpy(float)
        h = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        c = df["close"].to_numpy(float)
        if sp.rel == "close":
            ok = (c > lv) if sp.side == "above" else (c < lv)
        elif sp.rel == "full":
            ok = (lo > lv) if sp.side == "above" else (h < lv)
        else:
            ok = (lo <= lv) & (lv <= h)
        if sp.colour == "red":
            ok &= c < o
        elif sp.colour == "green":
            ok &= c > o
        return _runs(ok & np.isfinite(lv), sp.n)

    return Cond(fn, str(sp))


def build_rules(specs: Optional[Sequence], side: str, *, chart_tf: Optional[str] = None,
                combine: str = "all") -> Optional[Cond]:
    """Every rule that applies to `side`, joined by `combine` ("all" | "any").

    Returns None when no rule applies, which callers should read as "no extra condition" rather
    than as "nothing passes" — an empty `all` is vacuously true and an empty `any` is not, and
    silently choosing either would be a trap.
    """
    parsed = [sp if isinstance(sp, RuleSpec) else parse_rule(sp) for sp in (specs or [])]
    mine = [sp for sp in parsed if side in sp.sides]
    if not mine:
        return None
    conds = [rule_cond(sp, chart_tf=chart_tf) for sp in mine]
    out = conds[0]
    for c in conds[1:]:
        out = (out & c) if str(combine).lower() == "all" else (out | c)
    return out


def describe_rules(specs: Optional[Sequence], *, chart_tf: Optional[str] = None) -> pd.DataFrame:
    """What each sentence parsed to — the cheapest way to catch a rule that does not say what you
    meant, before it silently filters every entry away."""
    rows = []
    for sp in (specs or []):
        p = sp if isinstance(sp, RuleSpec) else parse_rule(sp)
        rows.append(dict(rule=p.text or str(p), n=p.n, reference=p.ref,
                         column=resolve_ref(p.ref, chart_tf=chart_tf), relation=p.rel,
                         side=p.side or "-", colour=p.colour or "any",
                         applies_to="+".join(p.sides)))
    return pd.DataFrame(rows, columns=["rule", "n", "reference", "column", "relation", "side",
                                       "colour", "applies_to"])
