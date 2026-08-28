"""
Classical chart-pattern recognition — double tops, head & shoulders, wedges, triangles, flags.

These are geometric shapes over SWING PIVOTS rather than over individual candles, which makes
causality the whole problem. Two rules are enforced everywhere:

1. **A pivot does not exist until it is confirmed.** Pivots come from the ATR-adaptive zigzag
   (`add_zigzag_structure`), which emits a swing only once price has retraced a volatility-scaled
   threshold from the extreme — so the pivot is known at its CONFIRMATION bar, never at the extreme
   itself. Every shape test uses only pivots confirmed at or before the bar being evaluated.
2. **A pattern is not tradable until its boundary breaks.** Detection is two-stage: a candidate is
   registered when its pivots are in place, and the `pat_*` signal fires only on the bar where price
   CLOSES through the boundary. Drawing the shape earlier is fine for a chart — and
   `chart_pattern_shapes` does exactly that — but trading it earlier is lookahead.

**Trendlines are FITTED, not assumed.** A line through two pivots is not a trendline: two points
always define a line exactly, so every pivot pair would "fit" and the resulting shapes are arbitrary.
Here each boundary is a least-squares fit through the pivots in the window, and a pivot only counts
as a TOUCH if it sits within `line_tol_atr` of that line. A shape is rejected unless both boundaries
carry enough touches, which is what makes the detected patterns correspond to what a person would
draw. Boundaries are sloping where the shape says they should be, so a triangle confirms on a break
of its rising/falling line rather than of a horizontal proxy.

Sensitivity is configurable **per pattern** through `PatternSpec`, with `sensitivity` presets
('major' | 'normal' | 'minor') as a coarse dial and `settings={name: {...}}` for fine control.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .structure import add_zigzag_structure

PATTERN_DIRECTION: Dict[str, int] = {
    "double_top": -1, "double_bottom": +1,
    "triple_top": -1, "triple_bottom": +1,
    "head_shoulders": -1, "inverse_head_shoulders": +1,
    "rising_wedge": -1, "falling_wedge": +1,
    "ascending_triangle": +1, "descending_triangle": -1,
    "symmetrical_triangle": 0,
    "rectangle_bull": +1, "rectangle_bear": -1,
    "bull_flag": +1, "bear_flag": -1,
    "bull_pennant": +1, "bear_pennant": -1,
    "broadening_top": -1, "broadening_bottom": +1,
}
ALL_PATTERNS: List[str] = list(PATTERN_DIRECTION)


# ---------------------------------------------------------------------------
# per-pattern configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PatternSpec:
    """How strict one pattern is. Every field is per-pattern overridable.

    `min_touches_*`  how many pivots each boundary must pass through. THREE is the meaningful
                     minimum: two points define a line exactly, so a 2-touch "trendline" carries no
                     information and is what made the old detector produce arbitrary shapes.
    `min_span_bars`  how wide the shape must be, IN BARS. Absolute on purpose: a triangle is a
                     triangle whether you loaded two days or two years, so scaling this with the
                     size of the loaded window would make the same setting mean different things.
    `span_cap_frac`  never demand more than this fraction of the loaded chart, so a short load can
                     still find something instead of silently returning nothing.
    `min_height_atr` reject shapes shorter than this many ATR — noise rather than structure.
    `line_tol_atr`   how far off the fitted line a pivot may sit and still count as a touch.
    `flat_tol_atr`   a boundary is "flat" if it moves less than this many ATR across the whole span.
    `level_tol`      equal-highs / equal-lows tolerance, as a fraction of the shape's own height
                     (double and triple tops, head-and-shoulders shoulder symmetry).
    `slope_sym_max`  how UNEQUAL the two boundary slopes may be and the shape still count as
                     SYMMETRICAL: the steeper divided by the shallower. A symmetrical triangle is
                     defined by two roughly equal and opposite slopes, and without this test the only
                     requirements were "upper falls, lower rises, converging" — which a line falling
                     at -12.9 against one rising at 0.76 satisfies. Measured on BTC 1m over five
                     days, ratios reached 17:1 and the median was 3.3, which is why the drawn
                     triangles did not look symmetrical. A shape that converges but fails this is not
                     reclassified: it is not flat-topped or flat-bottomed either, so it is not a
                     classical named pattern and is dropped.
    """
    min_touches_upper: int = 3
    min_touches_lower: int = 3
    min_span_bars: int = 80
    span_cap_frac: float = 0.25
    min_height_atr: float = 3.0
    line_tol_atr: float = 1.0
    flat_tol_atr: float = 1.0
    level_tol: float = 0.12
    slope_sym_max: float = 2.5


# Shapes built from discrete extremes (tops, bottoms, H&S) are defined by their pivots rather than
# by two fitted lines, so they ask for fewer touches on the "boundary" side.
_LEVEL_SPEC = PatternSpec(min_touches_upper=2, min_touches_lower=2)

DEFAULT_SPECS: Dict[str, PatternSpec] = {
    **{n: PatternSpec() for n in ALL_PATTERNS},
    "double_top": _LEVEL_SPEC, "double_bottom": _LEVEL_SPEC,
    "triple_top": _LEVEL_SPEC, "triple_bottom": _LEVEL_SPEC,
    "head_shoulders": _LEVEL_SPEC, "inverse_head_shoulders": _LEVEL_SPEC,
    # flags and pennants are by definition short consolidations after a pole
    "bull_flag": PatternSpec(min_span_bars=25),
    "bear_flag": PatternSpec(min_span_bars=25),
    "bull_pennant": PatternSpec(min_span_bars=25),
    "bear_pennant": PatternSpec(min_span_bars=25),
}

#: coarse dial. Multipliers on span/height, and an offset on the touch requirement.
SENSITIVITY_PRESETS: Dict[str, dict] = {
    "major":  dict(touch_delta=+1, span_mult=1.5, height_mult=1.4),
    "normal": dict(touch_delta=0,  span_mult=1.0, height_mult=1.0),
    "minor":  dict(touch_delta=-1, span_mult=0.35, height_mult=0.5),
}


#: Per-pattern detector calibration from the Phase-15 sweep (108 settings x 9 windows x 13 patterns,
#: BTCUSDT 15m 2022-2026), re-run on the causal detector.
#:
#: WHAT THESE ARE: settings that produce well-formed shapes in usable numbers — at least ten
#: detections per window in every one of the nine windows, at the strictest geometry that still
#: clears that floor. The quality knobs (`min_touches=3`, `line_tol_atr=0.6`) come from a scan pooled
#: over all thirteen patterns, and both land where the prior argument says they should: two points
#: define a line exactly, so three touches is the meaningful minimum.
#:
#: WHAT THESE ARE NOT: settings that predict direction, and they must not be read that way. They were
#: chosen deliberately WITHOUT looking at forward returns, because selecting on the score is not
#: defensible here — re-running the identical grid after fixing a lookahead moved the best-scoring
#: cell for 13 of 13 patterns, including patterns whose rank correlation across all 108 settings was
#: 0.98-0.999. At these settings the measured directional hit rate is **0.48, below a coin flip**.
#: What a confirmed pattern does predict is EXPANSION: 18-42% more forward range than normal, against
#: a shuffled null at 0.97. Use these to SEE the shapes and to size risk; see `docs/WHAT_WORKS.md`
#: before trading one.
TUNED_SPECS: Dict[str, dict] = {
    # atr_mult 5.0 — the level shapes, which need big, clean pivots
    "head_shoulders":         dict(atr_mult=5.0, min_span_bars=100),
    "inverse_head_shoulders": dict(atr_mult=5.0, min_span_bars=100),
    "triple_top":             dict(atr_mult=5.0, min_span_bars=100),
    "triple_bottom":          dict(atr_mult=5.0, min_span_bars=100),
    "double_top":             dict(atr_mult=5.0, min_span_bars=100),
    "double_bottom":          dict(atr_mult=5.0, min_span_bars=100),
    # atr_mult 3.5 — wedges and the symmetrical triangle, which need a long span to be real
    "rising_wedge":           dict(atr_mult=3.5, min_span_bars=200),
    "falling_wedge":          dict(atr_mult=3.5, min_span_bars=200),
    "symmetrical_triangle":   dict(atr_mult=3.5, min_span_bars=200),
    # atr_mult 2.5 — the rarer shapes, which need a finer zigzag to appear at all
    "ascending_triangle":     dict(atr_mult=2.5, min_span_bars=40),
    "descending_triangle":    dict(atr_mult=2.5, min_span_bars=40),
    "broadening_top":         dict(atr_mult=2.5, min_span_bars=100),
    "broadening_bottom":      dict(atr_mult=2.5, min_span_bars=100),
}
#: shared by every entry above — kept separate so the table shows only what actually differs
TUNED_COMMON: dict = dict(min_touches_upper=3, min_touches_lower=3,
                          min_height_atr=4.0, line_tol_atr=0.6)


def tuned_specs(patterns: Optional[Sequence[str]] = None) -> Dict[str, dict]:
    """`TUNED_SPECS` merged with `TUNED_COMMON`, optionally restricted to `patterns`."""
    out = {k: {**TUNED_COMMON, **v} for k, v in TUNED_SPECS.items()}
    if patterns is not None:
        missing = set(patterns) - set(out)
        if missing:
            raise KeyError(f"no tuned settings for {sorted(missing)}; "
                           f"available: {sorted(out)}")
        out = {k: v for k, v in out.items() if k in patterns}
    return out


def resolve_specs(sensitivity: str = "normal",
                  settings: Optional[Dict[str, dict]] = None) -> Dict[str, PatternSpec]:
    """`DEFAULT_SPECS` -> `sensitivity` preset -> per-pattern `settings` override (most specific wins)."""
    if sensitivity not in SENSITIVITY_PRESETS:
        raise KeyError(f"unknown sensitivity {sensitivity!r}; "
                       f"use one of {sorted(SENSITIVITY_PRESETS)}")
    pre = SENSITIVITY_PRESETS[sensitivity]
    out: Dict[str, PatternSpec] = {}
    for name, base in DEFAULT_SPECS.items():
        s = _dc_replace(
            base,
            min_touches_upper=max(2, base.min_touches_upper + pre["touch_delta"]),
            min_touches_lower=max(2, base.min_touches_lower + pre["touch_delta"]),
            min_span_bars=int(base.min_span_bars * pre["span_mult"]),
            min_height_atr=base.min_height_atr * pre["height_mult"])
        over = (settings or {}).get(name)
        if over:
            bad = set(over) - {f.name for f in PatternSpec.__dataclass_fields__.values()}
            if bad:
                raise KeyError(f"unknown PatternSpec field(s) {sorted(bad)} for {name!r}")
            s = _dc_replace(s, **over)
        out[name] = s
    return out


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
@dataclass
class _Pivot:
    confirm_i: int      # bar at which this pivot became known
    i: int              # bar of the extreme itself
    price: float
    kind: int           # +1 high, -1 low


@dataclass
class _Fit:
    """A fitted boundary line plus how well the pivots actually sit on it.

    `pivots` are the INLIERS — the ones within tolerance of the line, i.e. the actual touches. The
    window they were chosen from holds others that sit inside the shape and touch nothing; keeping
    only the inliers here is what lets a chart mark real touches instead of every nearby swing.
    """
    slope: float
    intercept: float
    touches: int
    max_dev: float
    pivots: Tuple["_Pivot", ...] = ()

    def at(self, x: float) -> float:
        return self.slope * x + self.intercept


@dataclass
class _Candidate:
    name: str
    direction: int          # +1 break up, -1 break down
    b_slope: float          # the BOUNDARY that must break (sloping for triangles/wedges)
    b_intercept: float
    stop: float             # protective level on the far side of the shape
    height: float           # pattern height, for the measured move
    formed_i: int           # bar the shape was complete and known
    expires_i: int
    upper: Optional[_Fit] = None
    lower: Optional[_Fit] = None
    pivots: List[_Pivot] = field(default_factory=list)

    def boundary_at(self, x: float) -> float:
        return self.b_slope * x + self.b_intercept

    @property
    def span(self) -> int:
        return (max(p.i for p in self.pivots) - min(p.i for p in self.pivots)) if self.pivots else 0

    @property
    def score(self) -> Tuple[int, int, float]:
        t = (self.upper.touches if self.upper else 0) + (self.lower.touches if self.lower else 0)
        dev = max((self.upper.max_dev if self.upper else 0.0),
                  (self.lower.max_dev if self.lower else 0.0))
        return (t, self.span, -dev)


def _pivots_from_zigzag(df: pd.DataFrame, prefix: str = "zz") -> List[_Pivot]:
    lab = df[f"{prefix}_label"].to_numpy(object)
    price = df[f"{prefix}_price"].to_numpy(float)
    piv_i = df[f"{prefix}_pivot_i"].to_numpy(float)
    out: List[_Pivot] = []
    for i in np.flatnonzero(pd.notna(lab)):
        L = lab[i]
        if L is None or (isinstance(L, float) and np.isnan(L)):
            continue
        kind = +1 if L in ("HH", "LH") else -1
        pi = int(piv_i[i]) if np.isfinite(piv_i[i]) else int(i)
        out.append(_Pivot(confirm_i=int(i), i=pi, price=float(price[i]), kind=kind))
    return out


def _best_line(pivots: List[_Pivot], tol: float, min_touches: int, side: int) -> Optional[_Fit]:
    """The best TRENDLINE through a set of pivots — a line several of them sit on, not all of them.

    Fitting through every pivot in a window is wrong and was the second half of the original bug: a
    window spans more than one structure, so the least-squares line through all of its highs misses
    every one of them (measured deviations of 40-600 against a tolerance of 24). A real trendline
    touches a SUBSET — the swings that define the boundary — while the others sit inside the shape.

    So: RANSAC. Every pair of pivots proposes a line, the pivots within `tol` of it are its inliers,
    and the best-supported proposal wins; the returned line is then refitted through those inliers.

    `side` makes it an ENVELOPE, which is what "resistance" and "support" actually mean: for an upper
    line (+1) no pivot may sit more than `tol` ABOVE it, for a lower line (-1) none more than `tol`
    below. Without that constraint a line can cut straight through the middle of the swings.
    """
    n = len(pivots)
    if n < max(2, min_touches):
        return None
    x = np.array([p.i for p in pivots], float)
    y = np.array([p.price for p in pivots], float)
    total_span = max(x.max() - x.min(), 1.0)

    # Every pivot pair proposes a line. Done as one vectorised batch rather than a Python double
    # loop: this runs ~20k times per detection pass and the loop version made a parameter sweep
    # take hours.
    ii, jj = np.triu_indices(n, k=1)
    dx = x[jj] - x[ii]
    good = dx > 0
    if not good.any():
        return None
    ii, jj, dx = ii[good], jj[good], dx[good]
    slope = (y[jj] - y[ii]) / dx
    intercept = y[ii] - slope * x[ii]
    resid = y[None, :] - (slope[:, None] * x[None, :] + intercept[:, None])   # (pairs, pivots)

    # envelope: for an upper line nothing may poke more than `tol` above it, and vice versa —
    # which is what "resistance" and "support" actually mean
    if side > 0:
        keep = ~(resid > tol).any(axis=1)
    elif side < 0:
        keep = ~(resid < -tol).any(axis=1)
    else:
        keep = np.ones(len(slope), bool)
    if not keep.any():
        return None

    inl = np.abs(resid) <= tol
    counts = inl.sum(axis=1)
    xs = np.where(inl, x[None, :], np.nan)
    with np.errstate(invalid="ignore"):
        spans = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)
    keep &= counts >= min_touches
    keep &= spans >= 0.5 * total_span        # touches must cover the window, not cluster
    if not keep.any():
        return None

    order = np.lexsort((spans[keep], counts[keep]))[::-1]         # most touches, then widest
    pick = np.flatnonzero(keep)[order[0]]
    sel = inl[pick]
    xi, yi = x[sel], y[sel]
    if sel.sum() > 2:
        sl2, ic2 = np.polyfit(xi, yi, 1)
    else:
        sl2, ic2 = slope[pick], intercept[pick]
    dev = float(np.abs(yi - (sl2 * xi + ic2)).max())
    return _Fit(float(sl2), float(ic2), int(sel.sum()), dev,
                tuple(p for p, is_touch in zip(pivots, sel) if is_touch))


def _touch_pivots(cand: "_Candidate") -> List[_Pivot]:
    """The pivots a shape is genuinely built on, in time order.

    For a trendline shape those are the fitted boundaries' inliers. Level shapes (double and triple
    tops, head-and-shoulders) have no fitted lines — their defining extremes ARE the touches.
    """
    seen = {}
    for fit in (cand.upper, cand.lower):
        for q in (fit.pivots if fit else ()):
            seen[(q.i, q.price)] = q
    if not seen:
        for q in cand.pivots:
            seen[(q.i, q.price)] = q
    return sorted(seen.values(), key=lambda q: q.i)


def _level_ok(a: float, b: float, tol: float, scale: float) -> bool:
    return abs(a - b) <= tol * max(scale, 1e-12)


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
def _shapes_from_window(pv: List[_Pivot], *, specs: Dict[str, PatternSpec], atr: float,
                        n_bars: int, names: Sequence[str]) -> List[_Candidate]:
    """Fit both boundaries over one pivot window and emit whichever shapes the fit supports."""
    highs = [p for p in pv if p.kind > 0]
    lows = [p for p in pv if p.kind < 0]
    if len(highs) < 2 or len(lows) < 2 or atr <= 0:
        return []
    formed = max(p.confirm_i for p in pv)
    span = max(p.i for p in pv) - min(p.i for p in pv)
    x0, x1 = min(p.i for p in pv), max(p.i for p in pv)
    out: List[_Candidate] = []

    def gate(spec: PatternSpec, height: float, pts: Sequence[_Pivot]) -> bool:
        # measure the SHAPE, not the search window that contained it -- a head-and-shoulders built
        # from five pivots inside a twenty-pivot window is as wide as its own five pivots.
        w = max(p.i for p in pts) - min(p.i for p in pts)
        need = min(spec.min_span_bars, max(1, int(spec.span_cap_frac * n_bars)))
        return w >= need and height >= spec.min_height_atr * atr

    # ---- fitted-boundary shapes: triangles, wedges, rectangles, broadening ----------
    tol_ref = max((specs[n].line_tol_atr for n in names if n in specs), default=0.6) * atr
    need_up = min((specs[n].min_touches_upper for n in names if n in specs), default=3)
    need_lo = min((specs[n].min_touches_lower for n in names if n in specs), default=3)
    up = _best_line(highs, tol_ref, need_up, +1)
    lo = _best_line(lows, tol_ref, need_lo, -1)
    if up is not None and lo is not None:
        w0 = up.at(x0) - lo.at(x0)
        w1 = up.at(x1) - lo.at(x1)
        converging = w1 < w0
        diverging = w1 > w0
        # The measured move is the pattern's widest part, but the boundaries are EXTRAPOLATED to the
        # window edges, and two steep converging lines can be metres apart there while price never
        # spanned anything like it. Clamp to the range price actually covered, or a triangle projects
        # a target thousands of points away.
        observed = max(p.price for p in pv) - min(p.price for p in pv)
        height = float(min(max(w0, w1, 0.0), observed))

        def flat(f: _Fit, spec: PatternSpec) -> bool:
            return abs(f.slope) * max(span, 1) <= spec.flat_tol_atr * atr

        def rises(f: _Fit, spec: PatternSpec) -> bool:
            return f.slope * max(span, 1) > spec.flat_tol_atr * atr

        def falls(f: _Fit, spec: PatternSpec) -> bool:
            return f.slope * max(span, 1) < -spec.flat_tol_atr * atr

        def symmetrical(spec: PatternSpec) -> bool:
            """Are the two boundaries close enough to equal-and-opposite to be called symmetrical?"""
            a, b = abs(up.slope), abs(lo.slope)
            lo_s, hi_s = min(a, b), max(a, b)
            if lo_s <= 1e-12:
                return False                      # one boundary flat: ascending/descending, not this
            return hi_s / lo_s <= spec.slope_sym_max

        def touches_ok(spec: PatternSpec) -> bool:
            return (up.touches >= spec.min_touches_upper
                    and lo.touches >= spec.min_touches_lower
                    and up.max_dev <= spec.line_tol_atr * atr
                    and lo.max_dev <= spec.line_tol_atr * atr)

        # The protective level has to be somewhere price ACTUALLY traded. Using the opposite
        # boundary extrapolated to the window end puts the stop at the apex, where converging lines
        # nearly meet -- measured 43 points of risk against a 2,328-point target. The last opposite
        # swing is a real level and keeps the risk proportionate to the shape.
        stop_up = min(p.price for p in lows)      # protective level for a BULLISH break
        stop_dn = max(p.price for p in highs)     # ... and for a bearish one

        def emit(name: str, direction: int, boundary: _Fit, stop: float):
            spec = specs.get(name)
            if spec is None or name not in names:
                return
            if not touches_ok(spec) or not gate(spec, height, pv):
                return
            out.append(_Candidate(name, direction, boundary.slope, boundary.intercept, stop,
                                  height, formed, 0, upper=up, lower=lo, pivots=list(pv)))

        s_asc = specs.get("ascending_triangle", PatternSpec())
        if flat(up, s_asc) and rises(lo, s_asc):
            emit("ascending_triangle", +1, up, stop_up)
        elif flat(lo, s_asc) and falls(up, s_asc):
            emit("descending_triangle", -1, lo, stop_dn)
        elif falls(up, s_asc) and rises(lo, s_asc) and converging:
            # converging with opposite slopes is necessary but NOT sufficient — the slopes also have
            # to be comparable, or a near-flat line paired with a steep one gets called symmetrical
            if symmetrical(specs.get("symmetrical_triangle", PatternSpec())):
                emit("symmetrical_triangle", +1, up, stop_up)
                emit("symmetrical_triangle", -1, lo, stop_dn)
        elif rises(up, s_asc) and rises(lo, s_asc) and converging:
            emit("rising_wedge", -1, lo, stop_dn)
        elif falls(up, s_asc) and falls(lo, s_asc) and converging:
            emit("falling_wedge", +1, up, stop_up)
        elif flat(up, s_asc) and flat(lo, s_asc):
            emit("rectangle_bull", +1, up, stop_up)
            emit("rectangle_bear", -1, lo, stop_dn)
        elif rises(up, s_asc) and falls(lo, s_asc) and diverging:
            emit("broadening_top", -1, lo, stop_dn)
            emit("broadening_bottom", +1, up, stop_up)

    # ---- level shapes: double / triple tops and bottoms, head & shoulders -----------
    scale = max(max(p.price for p in pv) - min(p.price for p in pv), 1e-12)

    def emit_level(name: str, direction: int, neck: float, stop: float, height: float,
                   pts: List[_Pivot]):
        spec = specs.get(name)
        if spec is None or name not in names or not gate(spec, height, pts):
            return
        out.append(_Candidate(name, direction, 0.0, neck, stop, height, formed, 0,
                              pivots=sorted(pts, key=lambda q: q.i)))

    s_lv = specs.get("double_top", _LEVEL_SPEC)
    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]
    if _level_ok(h1.price, h2.price, s_lv.level_tol, scale) and h1.i < h2.i:
        mid = [p for p in lows if h1.i < p.i < h2.i]
        if mid:
            neck = min(p.price for p in mid)
            top = max(h1.price, h2.price)
            emit_level("double_top", -1, neck, top, top - neck, [h1, mid[-1], h2])
    if _level_ok(l1.price, l2.price, s_lv.level_tol, scale) and l1.i < l2.i:
        mid = [p for p in highs if l1.i < p.i < l2.i]
        if mid:
            neck = max(p.price for p in mid)
            bot = min(l1.price, l2.price)
            emit_level("double_bottom", +1, neck, bot, neck - bot, [l1, mid[-1], l2])
    if len(highs) >= 3:
        a, b, c = highs[-3], highs[-2], highs[-1]
        if (_level_ok(a.price, b.price, s_lv.level_tol, scale)
                and _level_ok(b.price, c.price, s_lv.level_tol, scale)):
            mid = [p for p in lows if a.i < p.i < c.i]
            if mid:
                neck = min(p.price for p in mid)
                top = max(a.price, b.price, c.price)
                emit_level("triple_top", -1, neck, top, top - neck, [a, b, c] + mid[-2:])
    if len(lows) >= 3:
        a, b, c = lows[-3], lows[-2], lows[-1]
        if (_level_ok(a.price, b.price, s_lv.level_tol, scale)
                and _level_ok(b.price, c.price, s_lv.level_tol, scale)):
            mid = [p for p in highs if a.i < p.i < c.i]
            if mid:
                neck = max(p.price for p in mid)
                bot = min(a.price, b.price, c.price)
                emit_level("triple_bottom", +1, neck, bot, neck - bot, [a, b, c] + mid[-2:])
    if len(highs) >= 3 and len(lows) >= 2:
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        troughs = [p for p in lows if ls.i < p.i < rs.i]
        s_hs = specs.get("head_shoulders", _LEVEL_SPEC)
        if (head.price > ls.price and head.price > rs.price
                and _level_ok(ls.price, rs.price, s_hs.level_tol * 1.5, scale)
                and len(troughs) >= 2):
            neck = float(np.mean([p.price for p in troughs[-2:]]))
            emit_level("head_shoulders", -1, neck, head.price, head.price - neck,
                       [ls] + troughs[-2:] + [head, rs])
    if len(lows) >= 3 and len(highs) >= 2:
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        peaks = [p for p in highs if ls.i < p.i < rs.i]
        s_hs = specs.get("inverse_head_shoulders", _LEVEL_SPEC)
        if (head.price < ls.price and head.price < rs.price
                and _level_ok(ls.price, rs.price, s_hs.level_tol * 1.5, scale)
                and len(peaks) >= 2):
            neck = float(np.mean([p.price for p in peaks[-2:]]))
            emit_level("inverse_head_shoulders", +1, neck, head.price, neck - head.price,
                       [ls] + peaks[-2:] + [head, rs])
    return out


def _detect(pivots: List[_Pivot], upto: int, *, specs: Dict[str, PatternSpec], atr: float,
            n_bars: int, names: Sequence[str], min_pivots: int,
            lookback_pivots: int, pivot_step: int = 1) -> List[_Candidate]:
    """Search several pivot windows ending at the newest pivot and keep the best shape per name.

    Trying a range of window sizes is what lets a large structure be found: the old code looked only
    at the last two highs and two lows, which can never span more than a handful of swings.
    """
    best: Dict[Tuple[str, int], _Candidate] = {}
    for k in range(min_pivots, lookback_pivots + 1, pivot_step):
        if k > upto + 1:
            break
        window = pivots[upto + 1 - k: upto + 1]
        for cand in _shapes_from_window(window, specs=specs, atr=atr, n_bars=n_bars, names=names):
            key = (cand.name, cand.direction)
            if key not in best or cand.score > best[key].score:
                best[key] = cand
    return list(best.values())


def _flagpole(pv: List[_Pivot], cand: _Candidate, *, pole_mult: float) -> Optional[str]:
    """A consolidation preceded by a sharp move is a flag/pennant rather than a plain range."""
    if len(pv) < 4:
        return None
    start = min(p.i for p in cand.pivots) if cand.pivots else min(p.i for p in pv[-4:])
    pole = [p for p in pv if p.i < start]
    if len(pole) < 2:
        return None
    move = pole[-1].price - pole[-2].price
    if abs(move) < pole_mult * cand.height:
        return None
    return "bull" if move > 0 else "bear"


def _dedupe(rows: List[dict], *, overlap_frac: float) -> List[dict]:
    """Collapse detections of the SAME pattern that describe the same stretch of price action.

    CAUSAL, and that is the whole subtlety. Detections are resolved in CONFIRMATION ORDER, and a new
    one is dropped when it overlaps something already kept. Ranking the full set by quality instead —
    keeping the best-formed shape wherever it sits — lets a pattern confirming at bar 5,000 delete one
    that already fired at bar 2,997, which makes a past signal depend on future bars. It is also the
    wrong semantics for a trader, who has already acted on the earlier signal and cannot un-take it.
    """
    kept: List[dict] = []
    for r in sorted(rows, key=lambda z: (z["confirm_i"], -z["touches"], -z["span_bars"])):
        a0, a1 = r["start_i"], r["confirm_i"]
        clash = False
        for k in kept:
            if k["pattern"] != r["pattern"]:
                continue
            b0, b1 = k["start_i"], k["confirm_i"]
            inter = max(0, min(a1, b1) - max(a0, b0))
            shorter = max(1, min(a1 - a0, b1 - b0))
            if inter / shorter > overlap_frac:
                clash = True
                break
        if not clash:
            kept.append(r)
    return kept                      # already in confirmation order


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def add_chart_patterns(df: pd.DataFrame, *, sensitivity: str = "normal",
                       settings: Optional[Dict[str, dict]] = None,
                       atr_mult: float = 3.5, atr_period: int = 14,
                       min_pivots: int = 4, lookback_pivots: int = 24, pivot_step: int = 2,
                       max_wait: int = 150,
                       pole_mult: float = 1.5, overlap_frac: float = 0.5,
                       patterns: Optional[Sequence[str]] = None,
                       prefix: str = "pat",
                       _collect: Optional[List[dict]] = None,
                       **removed) -> pd.DataFrame:
    """Detect classical chart patterns and publish their entry / stop / target levels.

    Adds, all NaN or False except on the bar a pattern CONFIRMS:
      `{prefix}_<name>`  one boolean column per pattern
      `{prefix}_signal`  any pattern confirmed on this bar
      `{prefix}_dir`     +1 / -1 for the confirmed pattern
      `{prefix}_name`    its name
      `{prefix}_entry`   the breakout price — the boundary's value AT the break bar, so a sloping
                         triangle line is honoured rather than flattened to a horizontal proxy
      `{prefix}_stop`    protective level on the far side of the shape
      `{prefix}_target`  measured move: entry +/- the pattern height
    plus `_ff` forward-filled copies of entry/stop/target/dir for the engine's `sl_mode='ref_col'`.

    Sensitivity: `sensitivity` is the coarse dial ('major' | 'normal' | 'minor'), `settings` the
    per-pattern override, e.g. `settings={'ascending_triangle': dict(min_touches_upper=4)}`. See
    `PatternSpec` for every field. `atr_mult` sizes the zigzag (larger = fewer, more structural
    pivots); `min_pivots`/`lookback_pivots` bound the window search; `max_wait` is how long a formed
    shape waits for its break; `overlap_frac` is how much two same-name detections may overlap
    before the weaker is dropped.
    """
    if removed:
        # These knobs were replaced when trendlines became fitted rather than assumed. Say so
        # explicitly: the bare "unexpected keyword argument" is a confusing thing to hit from a
        # notebook that a kernel is still holding an older copy of.
        moved = {"tol": "level_tol (via settings={name: {...}}) — the old meaning was the "
                        "equal-highs tolerance",
                 "min_height_frac": "min_height_atr (via settings=)",
                 "min_span_frac": "min_span_bars / span_cap_frac (via settings=)"}
        bits = [f"{k!r} -> {moved.get(k, 'no longer used')}" for k in sorted(removed)]
        raise TypeError(
            "add_chart_patterns: " + "; ".join(bits)
            + ". Strictness is now per-pattern: sensitivity='major'|'normal'|'minor' plus "
              "settings={'ascending_triangle': dict(min_touches_upper=4), ...}. "
              "If this came from notebook 14, reload it from disk — the kernel is holding an "
              "older copy of the cell.")
    names = list(patterns) if patterns is not None else ALL_PATTERNS
    unknown = [p for p in names if p not in PATTERN_DIRECTION]
    if unknown:
        raise KeyError(f"unknown chart pattern(s) {unknown}; available: {ALL_PATTERNS}")
    specs = resolve_specs(sensitivity, settings)

    out = df.copy()
    if "zz_label" not in out.columns:
        out = add_zigzag_structure(out, atr_mult=atr_mult, atr_period=atr_period)
    from .volatility import atr as _atr
    atr_s = _atr(out, atr_period).to_numpy(float)
    close = out["close"].to_numpy(float)
    n = len(out)

    pivots = _pivots_from_zigzag(out)
    hits = {name: np.zeros(n, bool) for name in names}
    e_entry = np.full(n, np.nan)
    e_stop = np.full(n, np.nan)
    e_target = np.full(n, np.nan)
    e_dir = np.zeros(n, np.int8)
    e_name = np.empty(n, dtype=object)

    live: List[_Candidate] = []
    rows: List[dict] = []
    pi = 0
    for i in range(n):
        # 1) register candidates from every pivot CONFIRMED on this bar
        while pi < len(pivots) and pivots[pi].confirm_i == i:
            a = atr_s[i] if np.isfinite(atr_s[i]) else 0.0
            for cand in _detect(pivots, pi, specs=specs, atr=a, n_bars=n, names=names,
                                min_pivots=min_pivots, lookback_pivots=lookback_pivots,
                                pivot_step=pivot_step):
                pole = _flagpole(pivots[:pi + 1], cand, pole_mult=pole_mult)
                if pole and cand.name in ("symmetrical_triangle", "rectangle_bull",
                                          "rectangle_bear"):
                    # A flag/pennant continues the pole, so ONLY the candidate already registered on
                    # that side qualifies. Relabelling the opposite-side one would keep its boundary
                    # and stop, leaving the stop on the wrong side of entry.
                    if cand.direction != (+1 if pole == "bull" else -1):
                        continue
                    is_pennant = cand.name == "symmetrical_triangle"
                    cand.name = f"{pole}_pennant" if is_pennant else f"{pole}_flag"
                if cand.name not in hits:
                    continue
                if not np.isfinite(cand.height) or cand.height <= 0:
                    continue
                cand.expires_i = i + max_wait
                live.append(cand)
            pi += 1

        # 2) a live shape confirms when price CLOSES through its boundary
        still: List[_Candidate] = []
        for cand in live:
            if i > cand.expires_i:
                continue
            if i <= cand.formed_i:
                still.append(cand)
                continue
            neck = cand.boundary_at(i)
            broke = close[i] > neck if cand.direction > 0 else close[i] < neck
            if not broke:
                still.append(cand)
                continue
            # risk must be defined: the stop sits on the far side of the entry. These levels drive
            # real stop-losses, so an ill-formed shape is dropped rather than patched.
            if cand.direction > 0 and not cand.stop < neck:
                continue
            if cand.direction < 0 and not cand.stop > neck:
                continue
            pv = cand.pivots
            rows.append(dict(
                confirm_i=i, formed_i=cand.formed_i, start_i=min(q.i for q in pv),
                pattern=cand.name, direction=cand.direction,
                entry=neck, stop=cand.stop,
                target=(neck + cand.height if cand.direction > 0 else neck - cand.height),
                height=cand.height, span_bars=cand.span,
                touches=(cand.upper.touches if cand.upper else 0)
                        + (cand.lower.touches if cand.lower else 0),
                touches_upper=cand.upper.touches if cand.upper else 0,
                touches_lower=cand.lower.touches if cand.lower else 0,
                upper_slope=cand.upper.slope if cand.upper else np.nan,
                upper_intercept=cand.upper.intercept if cand.upper else np.nan,
                lower_slope=cand.lower.slope if cand.lower else np.nan,
                lower_intercept=cand.lower.intercept if cand.lower else np.nan,
                boundary_slope=cand.b_slope, boundary_intercept=cand.b_intercept,
                pivot_i=[q.i for q in pv], pivot_price=[q.price for q in pv],
                # the pivots the boundaries were actually FITTED THROUGH. `pivot_i` is the whole
                # search window and is roughly twice as long — drawing that as "touches" put a dot
                # on every nearby swing, which is what made the chart look noisy.
                touch_i=[q.i for q in _touch_pivots(cand)],
                touch_price=[q.price for q in _touch_pivots(cand)],
                hi_i=[q.i for q in pv if q.kind > 0],
                hi_price=[q.price for q in pv if q.kind > 0],
                lo_i=[q.i for q in pv if q.kind < 0],
                lo_price=[q.price for q in pv if q.kind < 0]))
        live = still

    rows = _dedupe(rows, overlap_frac=overlap_frac)
    for r in rows:
        i = r["confirm_i"]
        hits[r["pattern"]][i] = True
        e_entry[i] = r["entry"]
        e_stop[i] = r["stop"]
        e_target[i] = r["target"]
        e_dir[i] = r["direction"]
        e_name[i] = r["pattern"]

    for name in names:
        out[f"{prefix}_{name}"] = hits[name]
    out[f"{prefix}_signal"] = (np.any(np.column_stack([hits[k] for k in names]), axis=1)
                               if names else False)
    out[f"{prefix}_dir"] = e_dir
    out[f"{prefix}_name"] = e_name
    out[f"{prefix}_entry"] = e_entry
    out[f"{prefix}_stop"] = e_stop
    out[f"{prefix}_target"] = e_target
    for c in ("entry", "stop", "target", "dir"):
        src = out[f"{prefix}_{c}"]
        out[f"{prefix}_{c}_ff"] = src.replace(0, np.nan).ffill() if c == "dir" else src.ffill()
    if _collect is not None:
        _collect.extend(rows)
    return out


def chart_pattern_shapes(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """The GEOMETRY of every confirmed pattern — the FITTED boundaries plus their touch pivots.

    `add_chart_patterns` returns per-bar columns, which is what a strategy needs but not enough to
    DRAW a triangle: for that you need the lines themselves. This returns one row per pattern with
    `upper_slope`/`upper_intercept`, `lower_slope`/`lower_intercept` (evaluate at any bar index to
    get the boundary there), the touch counts, `start_i`, `span_bars`, and entry / stop / target.

    Accepts the same arguments as `add_chart_patterns`.
    """
    shapes: List[dict] = []
    add_chart_patterns(df, _collect=shapes, **kwargs)
    cols = ["confirm_i", "formed_i", "start_i", "pattern", "direction", "entry", "stop", "target",
            "height", "span_bars", "touches", "touches_upper", "touches_lower",
            "upper_slope", "upper_intercept", "lower_slope", "lower_intercept",
            "boundary_slope", "boundary_intercept",
            "pivot_i", "pivot_price", "touch_i", "touch_price",
            "hi_i", "hi_price", "lo_i", "lo_price"]
    out = pd.DataFrame(shapes)
    if not len(out):
        return pd.DataFrame(columns=cols)
    if "t" in df.columns:
        tv = df["t"].to_numpy()
        out["t"] = tv[out["confirm_i"].to_numpy(int)]
        out["start_t"] = tv[out["start_i"].to_numpy(int)]
        out["pivot_t"] = [list(tv[np.asarray(v, int)]) for v in out["pivot_i"]]
        out["touch_t"] = [list(tv[np.asarray(v, int)]) for v in out["touch_i"]]
    return out


def chart_pattern_shapes_tuned(df: pd.DataFrame, *, specs: Optional[Dict[str, dict]] = None,
                               patterns: Optional[Sequence[str]] = None,
                               **kwargs) -> pd.DataFrame:
    """Shape geometry with PER-PATTERN settings, including a per-pattern `atr_mult`.

    `atr_mult` sizes the zigzag, so it is a property of the whole detection pass rather than of one
    pattern — patterns calibrated to different values genuinely cannot come out of a single call, and
    that is why `chart_pattern_shapes` alone cannot draw `TUNED_SPECS`. This groups the patterns by
    `atr_mult`, runs one pass per group and concatenates the results. Patterns sharing an `atr_mult`
    still share a pass, so the cost is one detection per DISTINCT value — three for `TUNED_SPECS`,
    not thirteen.
    """
    spec = tuned_specs(patterns) if specs is None else {
        k: dict(v) for k, v in specs.items() if patterns is None or k in patterns}
    groups: Dict[float, Dict[str, dict]] = {}
    for name, cfg in spec.items():
        cfg = dict(cfg)
        groups.setdefault(float(cfg.pop("atr_mult", 3.5)), {})[name] = cfg
    frames = [chart_pattern_shapes(df, patterns=list(per), atr_mult=am, settings=per, **kwargs)
              for am, per in groups.items()]
    full = [f for f in frames if len(f)]
    if not full:
        return frames[0] if frames else pd.DataFrame()
    out = pd.concat(full, ignore_index=True)
    return out.sort_values("confirm_i").reset_index(drop=True)


def chart_pattern_table(df: pd.DataFrame, *, prefix: str = "pat") -> pd.DataFrame:
    """Long-form table of confirmed patterns — for charting and inspection."""
    m = df[f"{prefix}_signal"].to_numpy(bool)
    idx = np.flatnonzero(m)
    if not len(idx):
        return pd.DataFrame(columns=["i", "t", "pattern", "direction", "entry", "stop", "target"])
    tcol = "t" if "t" in df.columns else None
    return pd.DataFrame({
        "i": idx,
        "t": (df[tcol].to_numpy()[idx] if tcol else idx),
        "pattern": df[f"{prefix}_name"].to_numpy(object)[idx],
        "direction": df[f"{prefix}_dir"].to_numpy()[idx],
        "entry": df[f"{prefix}_entry"].to_numpy()[idx],
        "stop": df[f"{prefix}_stop"].to_numpy()[idx],
        "target": df[f"{prefix}_target"].to_numpy()[idx],
    })
