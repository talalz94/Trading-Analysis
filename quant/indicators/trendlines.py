"""
Standalone TRENDLINES — the single sloping line price keeps respecting, and the break of it.

A chart pattern is a closed shape made of two boundaries. A trendline is the simpler and more common
object: one line under a rise (support) or over a fall (resistance), touched repeatedly, meaningful
precisely because price keeps turning at it — and informative when price finally closes through.

Everything here obeys the same two rules as `chart_patterns`, for the same reasons:

1. **A line does not exist until its pivots are confirmed.** Pivots come from the ATR-adaptive
   zigzag, so each is known at its confirmation bar rather than at its extreme. A line is published
   from the bar its last defining pivot confirmed — never from the bar it was "really" drawable, which
   is only knowable afterwards.
2. **The line is FITTED, not assumed.** `_best_line` runs RANSAC over pivot pairs, counts pivots
   within `line_tol_atr` as touches, and imposes an ENVELOPE so nothing pokes through the side the
   line is supposed to bound. A line through two points is not a trendline — two points always fit
   exactly — which is the single most common way this is got wrong.

The break is the event. `tl_break_dn` fires on the bar price CLOSES below an active support line,
`tl_break_up` on a close above an active resistance line. A line is retired once broken: the level
that just failed is no longer the line price is respecting.

Read `docs/WHAT_WORKS.md` before trading any of it. Nothing in this project has shown a directional
edge, chart patterns included, and a trendline break is a close cousin of the level-break family that
was measured and found to carry none.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .chart_patterns import _best_line, _Fit, _Pivot, _pivots_from_zigzag
from .structure import add_zigzag_structure


@dataclass
class _Line:
    """One fitted trendline while it is live."""
    kind: int                 # +1 resistance (over highs), -1 support (under lows)
    slope: float
    intercept: float
    touches: int
    start_i: int              # first touch pivot
    born_i: int               # bar the line became KNOWN (last pivot's confirmation)
    end_i: int = -1           # bar it broke, or where it was last valid
    broke: bool = False
    touch_i: tuple = ()
    touch_price: tuple = ()

    def at(self, x: float) -> float:
        return self.slope * x + self.intercept


def _fit_side(pivots: List[_Pivot], kind: int, *, tol: float, min_touches: int,
              min_span_bars: int) -> Optional[_Fit]:
    side = [p for p in pivots if p.kind == kind]
    if len(side) < min_touches:
        return None
    fit = _best_line(side, tol, min_touches, kind)
    if fit is None or len(fit.pivots) < min_touches:
        return None
    if fit.pivots[-1].i - fit.pivots[0].i < min_span_bars:
        return None
    return fit


def add_trendlines(df: pd.DataFrame, *,
                   atr_mult: float = 3.5, atr_period: int = 14,
                   min_touches: int = 3, lookback_pivots: int = 16,
                   line_tol_atr: float = 0.6, min_span_bars: int = 40,
                   max_age_bars: int = 500, break_buffer_atr: float = 0.0,
                   max_distance_atr: float = 8.0,
                   prefix: str = "tl") -> pd.DataFrame:
    """Fit the live support and resistance trendlines bar by bar and flag their breaks.

    Adds, per bar:
      `{prefix}_sup` / `{prefix}_res`             the line's price here, NaN when none is active
      `{prefix}_sup_slope` / `{prefix}_res_slope` its slope, in price per bar
      `{prefix}_sup_touches` / `{prefix}_res_touches`
      `{prefix}_break_dn`                          close went through an active SUPPORT line
      `{prefix}_break_up`                          close went through an active RESISTANCE line
      `{prefix}_break`                             either
      `{prefix}_break_dir`                         +1 up, -1 down, 0 none

    Parameters
    ----------
    atr_mult
        Zigzag swing size — how big a move must be to count as a pivot at all. THE dial for how many
        lines you get: lower finds more and smaller structure, higher keeps only major swings.
    min_touches
        Pivots the line must pass through. **Three is the meaningful minimum**: two points define a
        line exactly, so a 2-touch line always fits and tells you nothing.
    lookback_pivots
        How many recent pivots a line may be fitted over — its memory. Larger admits longer lines.
    line_tol_atr
        How far off the line a pivot may sit and still count as a touch, in ATR.
    min_span_bars
        A line must cover at least this many bars between its first and last touch.
    max_age_bars
        Retire a line this long after it was born even if unbroken; an ancient line is not the one
        price is currently respecting.
    max_distance_atr
        Retire a line once price sits further than this many ATR from it, and stop drawing it there.
        Age alone does not catch a steeply sloped line: the confirmation lag alone can carry one a
        long way, because a pivot's extreme sits well before the bar that confirms it. Lines used to
        end as far as 119 ATR from price, which is what made them look like rays flying off the
        chart; the furthest is now 11.8. Lower it for tighter lines, raise it for longer ones.
    break_buffer_atr
        Require the close to clear the line by this much ATR before calling it a break. 0 = a plain
        close through. Raise it to filter marginal pokes.
    """
    out = df.reset_index(drop=True).copy()
    if f"zz_label" not in out.columns:
        out = add_zigzag_structure(out, atr_mult=atr_mult, atr_period=atr_period)
    from .volatility import atr as _atr
    atr_s = _atr(out, atr_period).to_numpy(float)
    close = out["close"].to_numpy(float)
    n = len(out)

    pivots = _pivots_from_zigzag(out)
    sup_px = np.full(n, np.nan); res_px = np.full(n, np.nan)
    sup_sl = np.full(n, np.nan); res_sl = np.full(n, np.nan)
    sup_tc = np.zeros(n, np.int16); res_tc = np.zeros(n, np.int16)
    brk_dn = np.zeros(n, bool); brk_up = np.zeros(n, bool)

    live: dict = {+1: None, -1: None}
    retired: dict = {+1: set(), -1: set()}   # touch-sets that have already had their chance
    lines: List[_Line] = []
    pi = 0
    for i in range(n):
        # 1) refit from the pivots confirmed on this bar — only those, so the line is causal
        while pi < len(pivots) and pivots[pi].confirm_i == i:
            a = atr_s[i] if np.isfinite(atr_s[i]) else 0.0
            if a > 0:
                window = pivots[max(0, pi + 1 - lookback_pivots):pi + 1]
                for kind in (+1, -1):
                    fit = _fit_side(window, kind, tol=line_tol_atr * a,
                                    min_touches=min_touches, min_span_bars=min_span_bars)
                    if fit is None:
                        continue
                    # A line that has already been broken or retired does not come back. Refitting
                    # the same pivots one bar later resurrected it, which is why consecutive lines
                    # kept sharing a start_i and differing in slope by under 5% — the same line over
                    # and over. "A broken line is retired" has to mean it stays retired.
                    sig = tuple(p.i for p in fit.pivots)
                    if sig in retired[kind]:
                        continue
                    # A line that is ALREADY far from price when it is published is stale on arrival:
                    # a pivot's extreme sits well before the bar that confirms it, so a steep fit can
                    # be a long way off by the time it is knowable. Reject it here rather than
                    # accepting and retiring it a bar later — accepting it EVICTS the live incumbent,
                    # which is what collapsed the share of bars carrying a line from 52% to 3%.
                    if abs(close[i] - (fit.slope * i + fit.intercept)) > max_distance_atr * a:
                        continue
                    cur = live[kind]
                    # keep the incumbent unless the new fit is better supported; churning between
                    # equally-good lines every pivot would make the "current" line meaningless
                    if cur is not None and not cur.broke and fit.touches <= cur.touches:
                        continue
                    # A line that picks up another touch is the SAME line, refined — not a new one.
                    # Replacing it outright reset `born_i` and emitted a fresh record, which is why
                    # 16% of lines lived under 10 bars and a fifth of consecutive pairs differed in
                    # slope by less than 5%. Update in place when the new fit sits within tolerance
                    # of the old one across the span they share, so line IDENTITY survives a refit.
                    if cur is not None and not cur.broke:
                        span_x = (cur.start_i, i)
                        same = all(abs(fit.slope * x + fit.intercept - cur.at(x))
                                   <= line_tol_atr * a for x in span_x)
                        if same:
                            cur.slope, cur.intercept = fit.slope, fit.intercept
                            cur.touches = fit.touches
                            cur.touch_i = tuple(p.i for p in fit.pivots)
                            cur.touch_price = tuple(p.price for p in fit.pivots)
                            cur.start_i = min(cur.start_i, fit.pivots[0].i)
                            continue
                    ln = _Line(kind, fit.slope, fit.intercept, fit.touches,
                               fit.pivots[0].i, i,
                               touch_i=tuple(p.i for p in fit.pivots),
                               touch_price=tuple(p.price for p in fit.pivots))
                    # The outgoing line keeps whatever end it earned in the publish step below. It is
                    # NOT closed at `i`: several pivots can confirm on the same bar, so a line could
                    # be born and superseded before ever publishing, and stamping `end_i = i` gave it
                    # a 343-bar drawn span it had never been valid over — the worst sat 71 ATR from
                    # price. A line that never published keeps end_i < 0 and is dropped entirely.
                    live[kind] = ln
                    lines.append(ln)
            pi += 1

        # 2) publish, then test the break — in that order, so the bar that breaks a line still
        #    reports the line it broke rather than a blank
        a = atr_s[i] if np.isfinite(atr_s[i]) else 0.0
        buf = break_buffer_atr * a
        # A support line CANNOT sit above a resistance line — that inverts what the two mean, and
        # lets one bar break both at once. Rare (1 bar in ~156,000) but nonsense when it happens, so
        # the weaker line gives way: fewer touches loses, and the younger loses a tie.
        up_ln, dn_ln = live[+1], live[-1]
        if up_ln is not None and dn_ln is not None and dn_ln.at(i) > up_ln.at(i):
            drop = min((dn_ln, up_ln), key=lambda L: (L.touches, -L.born_i))
            retired[drop.kind].add(tuple(drop.touch_i))
            live[drop.kind] = None
        for kind in (+1, -1):
            ln = live[kind]
            if ln is None:
                continue
            if i - ln.born_i > max_age_bars:
                retired[kind].add(tuple(ln.touch_i))
                live[kind] = None            # keeps the end it earned; see below
                continue
            y = ln.at(i)
            # RELEVANCE. A support line price has left 30 ATR behind is not a line price is
            # respecting, it is a ray heading off the chart. Age alone does not catch it, because a
            # steeply sloped line diverges fast — and the confirmation lag alone can carry one a long
            # way, since a pivot's extreme sits well before the bar that confirms it.
            if a > 0 and abs(close[i] - y) > max_distance_atr * a:
                retired[kind].add(tuple(ln.touch_i))
                live[kind] = None            # retired WITHOUT extending its drawn span to here
                continue
            # the line is live and near price on this bar, so this is the furthest it has earned the
            # right to be drawn to. Anything past here was never validated against anything.
            ln.end_i = i
            if kind < 0:
                sup_px[i], sup_sl[i], sup_tc[i] = y, ln.slope, ln.touches
                if close[i] < y - buf:
                    brk_dn[i] = True
                    ln.broke, ln.end_i = True, i
                    retired[kind].add(tuple(ln.touch_i))
                    live[kind] = None
            else:
                res_px[i], res_sl[i], res_tc[i] = y, ln.slope, ln.touches
                if close[i] > y + buf:
                    brk_up[i] = True
                    ln.broke, ln.end_i = True, i
                    retired[kind].add(tuple(ln.touch_i))
                    live[kind] = None

    # A line that never published — born and superseded on the same bar, or never once found within
    # distance of price — has no span it was ever valid over, so it is not a line. Drop it rather
    # than inventing an endpoint for it.
    lines = [ln for ln in lines if ln.end_i >= 0]
    out.attrs[f"{prefix}_lines"] = lines

    out[f"{prefix}_sup"] = sup_px
    out[f"{prefix}_res"] = res_px
    out[f"{prefix}_sup_slope"] = sup_sl
    out[f"{prefix}_res_slope"] = res_sl
    out[f"{prefix}_sup_touches"] = sup_tc
    out[f"{prefix}_res_touches"] = res_tc
    out[f"{prefix}_break_dn"] = brk_dn
    out[f"{prefix}_break_up"] = brk_up
    out[f"{prefix}_break"] = brk_dn | brk_up
    out[f"{prefix}_break_dir"] = np.where(brk_up, 1, np.where(brk_dn, -1, 0)).astype(np.int8)
    return out


def trendline_table(df: pd.DataFrame, *, prefix: str = "tl", **kwargs) -> pd.DataFrame:
    """One row per fitted trendline — for drawing, and for inspecting what was found.

    Pass a frame already through `add_trendlines` (the lines ride along in `df.attrs`), or a raw
    frame plus the same keyword arguments and it will run the detection for you.
    """
    lines = df.attrs.get(f"{prefix}_lines")
    if lines is None:
        df = add_trendlines(df, prefix=prefix, **kwargs)
        lines = df.attrs[f"{prefix}_lines"]
    cols = ["id", "kind", "side", "slope", "intercept", "touches", "start_i", "born_i", "end_i",
            "broke", "y_start", "y_end", "touch_i", "touch_price"]
    if not lines:
        return pd.DataFrame(columns=cols)
    rows = [dict(kind=ln.kind, side="resistance" if ln.kind > 0 else "support",
                 slope=ln.slope, intercept=ln.intercept, touches=ln.touches,
                 start_i=ln.start_i, born_i=ln.born_i, end_i=ln.end_i, broke=ln.broke,
                 y_start=ln.at(ln.start_i), y_end=ln.at(ln.end_i),
                 touch_i=list(ln.touch_i), touch_price=list(ln.touch_price))
            for ln in lines]
    out = pd.DataFrame(rows)
    # A STABLE NAME per line: R0, R1... for resistance and S0, S1... for support, numbered in the
    # order the fitter published them. Everything that has to agree about "which line was that"
    # -- the chart's hover, the entry that traded it, the dropdown that isolates it -- derives the
    # name here rather than each inventing its own, so they cannot drift apart.
    out["id"] = [f"{'R' if k > 0 else 'S'}{n}"
                 for k, n in zip(out["kind"], out.groupby("kind").cumcount())]
    out = out[[c for c in cols if c in out.columns]
              + [c for c in out.columns if c not in cols]]
    if "t" in df.columns:
        tv = df["t"].to_numpy()
        out["born_t"] = tv[out["born_i"].to_numpy(int)]
        out["end_t"] = tv[out["end_i"].to_numpy(int)]
        out["touch_t"] = [list(tv[np.asarray(v, int)]) for v in out["touch_i"]]
    return out


def add_trendline_ids(df: pd.DataFrame, *, prefix: str = "tl") -> pd.DataFrame:
    """Add `{prefix}_res_n` / `{prefix}_sup_n`: WHICH line is live on each bar, or -1 for none.

    `{prefix}_res` says what the level is; this says which line it belongs to. Only one line of a
    kind is ever live, so one column per kind is enough. The number is the `id`'s digits, so bar
    `n == 3` on the resistance side is the line `trendline_table` calls `R3` -- which is how a trade
    can name the exact line it broke instead of only the timeframe it came from.
    """
    out = df if f"{prefix}_res" in df.columns else add_trendlines(df, prefix=prefix)
    lines = out.attrs.get(f"{prefix}_lines") or []
    for kind, nm in ((+1, "res"), (-1, "sup")):
        ids = np.full(len(out), -1, dtype=np.int32)
        for n, ln in enumerate(l for l in lines if l.kind == kind):
            ids[int(ln.born_i):int(ln.end_i) + 1] = n
        out[f"{prefix}_{nm}_n"] = ids
    return out


def mtf_trendlines(base_df: pd.DataFrame, frames: Optional[Dict[str, pd.DataFrame]] = None, *,
                   time_col: str = "t", prefix: str = "tl", **kwargs) -> pd.DataFrame:
    """Support and resistance lines from SEVERAL timeframes, all on the base frame's grid.

    Fits lines on `base_df` itself and on every frame in `frames`, then merges the higher-timeframe
    columns back with the project's usual anti-lookahead shift: a 1h line read off the bar stamped
    12:00 is only knowable when that bar CLOSES at 13:00, so `align_timeframes(shift_bars=1)` makes
    it visible to base bars from 13:00 onward and no earlier.

    Each frame's candle-colour runs (`up_run` / `dn_run`) come along too, so a rule can ask for
    three red 15m candles rather than three red bars of whatever the chart happens to be.

    Columns are the ones `add_trendlines` produces, prefixed by timeframe for the extra frames:
    `tl_res` / `tl_sup` for the base, `1h__tl_res` / `1h__tl_sup` and so on for the rest. Load the
    frames with `quant.data.tf_frames` so each has its history measured in ITS OWN bars — a 1h line
    needs `min_span_bars` 1h bars before it can exist at all.

    `frames` may be None or empty, in which case this is just `add_trendlines`.
    """
    from ..data.timeframe import align_timeframes

    from .candles import add_candle_runs

    out = add_candle_runs(add_trendline_ids(add_trendlines(base_df, prefix=prefix, **kwargs),
                                            prefix=prefix))
    if not frames:
        return out
    # each timeframe's OWN candle runs ride along, so "three red 15m candles" is a question the base
    # grid can answer. Counted on the 15m frame and merged, never recounted here — on a 5m grid a
    # run of 15m reds would otherwise be measured in 5m bars.
    others = {tf: add_candle_runs(add_trendline_ids(
        add_trendlines(f.reset_index(drop=True), prefix=prefix, **kwargs), prefix=prefix))
        for tf, f in frames.items()}
    merged = align_timeframes(out, others, time_col=time_col, shift_bars=1)
    # the shift turns the int columns into floats with a leading NaN; -1 means "no line" everywhere
    for c in [c for c in merged.columns if c.endswith(("_res_n", "_sup_n"))]:
        merged[c] = merged[c].fillna(-1).astype(np.int32)
    for c in [c for c in merged.columns if c.endswith(("up_run", "dn_run"))]:
        merged[c] = merged[c].fillna(0).astype(np.int32)
    return merged


def trendline_level_cols(df: pd.DataFrame, side: str, *, prefix: str = "tl") -> List[str]:
    """Every `{tf}__{prefix}_{res|sup}` column present, base frame first then coarsest-first.

    `side` is "long" (resistance lines, broken upward) or "short" (support lines, broken downward).
    """
    from ..data.timeframe import tf_minutes

    which = "res" if side == "long" else "sup"
    base = f"{prefix}_{which}"
    cols = [base] if base in df.columns else []
    extra = [c for c in df.columns if c.endswith(f"__{base}")]

    def _key(c):
        try:
            return tf_minutes(c.split("__", 1)[0])
        except ValueError:
            return 10 ** 9

    return cols + sorted(extra, key=_key, reverse=True)   # coarsest first: structure before detail
