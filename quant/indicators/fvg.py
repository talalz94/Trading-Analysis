"""
FAIR VALUE GAPS — the three-candle imbalance, detected causally and tracked to its death.

A fair value gap is a price band that one side of the market skipped. Take three consecutive
candles. If the third candle's LOW is still above the first candle's HIGH, then no trade happened
anywhere between them: the band `[high[i-2], low[i]]` was jumped over by the displacement in the
middle. That is a BULLISH gap. Mirror it — the third candle's HIGH below the first's LOW — and you
have a BEARISH gap spanning `[high[i], low[i-2]]`.

    bullish:  low[i]  > high[i-2]        band = (high[i-2], low[i])      sits BELOW price
    bearish:  high[i] < low[i-2]         band = (high[i],  low[i-2])     sits ABOVE price

WHEN IT EXISTS. The gap is knowable at the CLOSE of bar `i`, because it reads bar `i`'s own high and
low, and it is published from bar `i` — never from the middle candle, which is where it visually
appears to be. Drawing it at `i-1` would be a two-bar lookahead. `tests/test_fvg.py` checks this by
truncation, the same test the other detectors get: run on the first k bars and every value already
published must be unchanged.

HOW IT DIES. Price coming back into the band "mitigates" it, and there are three defensible
thresholds for when it stops counting. All three are tracked for every gap, always — the choice is
only which one the live-count columns and the veto use:

    touch   price trades into the band at all                (earliest; one wick kills a gap)
    ce      price reaches the band's MIDPOINT                (default: "consequent encroachment")
    full    price trades through the far edge                (latest; a half-eaten gap stays live)

`ce` is the default because the other two are the ends of a spectrum: a single tick through the near
edge is not a retracement, and waiting for a complete fill keeps a gap alive long after the market
has plainly been back.

WHICH ONES MATTER. Every timeframe produces gaps constantly — most are one tick of noise between two
ordinary candles. Two scale-free gates, both in ATR so they mean the same thing at 60,000 and at
600, and neither an absolute number that would quietly become a date filter:

    min_size_atr   the band must be this tall
    min_disp_atr   the MIDDLE candle's body must be this large -- a real gap comes from
                   displacement, not from a drift where two neighbours happened not to overlap

THESE DEFAULTS ARE CONVENTIONS, NOT MEASUREMENTS. The three-candle definition is standard and the
ATR gates are the usual practitioner filter, but nothing here has been shown to predict anything on
this data. See `docs/WHAT_WORKS.md` for how that has gone for every other pattern family.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from numba import njit
except Exception:  # pragma: no cover
    def njit(*a, **k):
        def w(f):
            return f
        return w(a[0]) if a and callable(a[0]) else w


MITIGATION_MODES = ("touch", "ce", "full")


# ---------------------------------------------------------------------------------------------
# detection + mitigation
# ---------------------------------------------------------------------------------------------
@njit(cache=True)
def _mitigate(born, kind, top, bot, high, low, max_age):
    """First bar at which each gap is touched / half-eaten / fully filled. -1 for "not yet".

    A forward scan per gap rather than a range-min structure. The scan stops at `max_age` bars,
    which bounds the work AND is the same cut the caller wants anyway: a gap older than that is not
    a trade consideration. With `max_age <= 0` it runs to the end of the frame, which is exact but
    costs O(bars) for every gap that never mitigates.
    """
    m = born.shape[0]
    n = high.shape[0]
    t_i = np.full(m, -1, np.int64)
    c_i = np.full(m, -1, np.int64)
    f_i = np.full(m, -1, np.int64)
    for g in range(m):
        b = born[g]
        hi = top[g]
        lo = bot[g]
        mid = 0.5 * (hi + lo)
        stop = n if max_age <= 0 else min(n, b + max_age + 1)
        if kind[g] > 0:
            # a BULLISH gap sits below price; it is eaten from above, by lows coming down
            for j in range(b + 1, stop):
                if t_i[g] < 0 and low[j] <= hi:
                    t_i[g] = j
                if c_i[g] < 0 and low[j] <= mid:
                    c_i[g] = j
                if low[j] <= lo:
                    f_i[g] = j
                    break
        else:
            # a BEARISH gap sits above price; it is eaten from below, by highs coming up
            for j in range(b + 1, stop):
                if t_i[g] < 0 and high[j] >= lo:
                    t_i[g] = j
                if c_i[g] < 0 and high[j] >= mid:
                    c_i[g] = j
                if high[j] >= hi:
                    f_i[g] = j
                    break
    return t_i, c_i, f_i


@njit(cache=True)
def _live_state(born, dead, kind, top, bot, close, atr, n, max_age):
    """Per-bar view of every gap alive at that bar: how many, and the nearest one's geometry.

    One pass with a compacted active list. A gap enters at `born`, leaves at `dead` (or at
    `born + max_age`), and "nearest" is measured from the close to the near edge of the band, in
    ATR, so the number means the same thing on any instrument.

    Returns, per bar and per direction: count, nearest top/bot, its age in bars, its size in ATR,
    and its distance in ATR.
    """
    cnt = np.zeros((2, n), np.int32)
    n_top = np.full((2, n), np.nan)
    n_bot = np.full((2, n), np.nan)
    n_age = np.full((2, n), np.nan)
    n_size = np.full((2, n), np.nan)
    n_dist = np.full((2, n), np.nan)

    m = born.shape[0]
    act = np.empty(m, np.int64)
    n_act = 0
    nxt = 0
    for i in range(n):
        while nxt < m and born[nxt] <= i:
            act[n_act] = nxt
            n_act += 1
            nxt += 1
        # drop the dead ones, keeping the survivors packed at the front
        k = 0
        for a in range(n_act):
            g = act[a]
            d = dead[g]
            if d >= 0 and i >= d:
                continue
            if max_age > 0 and i - born[g] > max_age:
                continue
            act[k] = g
            k += 1
        n_act = k

        a_i = atr[i]
        if a_i <= 0.0 or not np.isfinite(a_i):
            a_i = np.nan
        c_i = close[i]
        # SCALARS, not two-element arrays. numba allocates on every `np.full` inside a loop, and at
        # one bar per iteration that was two heap allocations per bar -- 460k of them on a 1m year,
        # and the single largest cost in this function.
        best0 = np.inf
        best1 = np.inf
        bg0 = -1
        bg1 = -1
        for a in range(n_act):
            g = act[a]
            # distance to the NEAR edge: 0 while price is inside the band
            if c_i > top[g]:
                dist = c_i - top[g]
            elif c_i < bot[g]:
                dist = bot[g] - c_i
            else:
                dist = 0.0
            if kind[g] > 0:
                cnt[0, i] += 1
                if dist < best0:
                    best0 = dist
                    bg0 = g
            else:
                cnt[1, i] += 1
                if dist < best1:
                    best1 = dist
                    bg1 = g
        if bg0 >= 0:
            n_top[0, i] = top[bg0]
            n_bot[0, i] = bot[bg0]
            n_age[0, i] = i - born[bg0]
            n_size[0, i] = (top[bg0] - bot[bg0]) / a_i
            n_dist[0, i] = best0 / a_i
        if bg1 >= 0:
            n_top[1, i] = top[bg1]
            n_bot[1, i] = bot[bg1]
            n_age[1, i] = i - born[bg1]
            n_size[1, i] = (top[bg1] - bot[bg1]) / a_i
            n_dist[1, i] = best1 / a_i
    return cnt, n_top, n_bot, n_age, n_size, n_dist


def add_fvgs(df: pd.DataFrame, *, prefix: str = "fvg", mitigation: str = "ce",
             min_size_atr: float = 0.25, min_disp_atr: float = 0.0, min_size_pct: float = 0.0,
             atr_period: int = 14, max_age_bars: int = 500,
             time_col: str = "t") -> pd.DataFrame:
    """Detect fair value gaps and publish a per-bar view of the ones still live.

    The gaps themselves ride along in `df.attrs[f"{prefix}_gaps"]` as a DataFrame — one row each,
    which is what `fvg_table` hands back for drawing. What lands in COLUMNS is the part a signal can
    use, and it is deliberately not "the level":

        {prefix}_bull_n / {prefix}_bear_n     how many gaps of that direction are live here
        {prefix}_bull_top / _bot              the NEAREST live one's band
        {prefix}_bull_age                     how many bars ago it formed
        {prefix}_bull_size_atr                how tall it is, in ATR
        {prefix}_bull_dist_atr                how far price is from its near edge, in ATR (0 inside)

    Counts and distances rather than fixed-n predicates, for the reason `add_candle_runs` gives: one
    column then serves every threshold a rule might want, AND a 15m count can be merged onto a 5m
    grid, where recomputing it would count 5m bars.

    `mitigation` is which of touch / ce / full retires a gap for the COUNT columns. All three are
    recorded per gap regardless, so changing your mind costs a column lookup, not a re-run.

    `max_age_bars` bounds both the mitigation scan and how long a gap can stay live; 0 means no
    limit, which is exact but makes the scan O(bars) for every gap that never mitigates.
    """
    if mitigation not in MITIGATION_MODES:
        raise ValueError(f"mitigation must be one of {MITIGATION_MODES}, got {mitigation!r}")

    out = df.copy()
    n = len(out)
    h = out["high"].to_numpy(np.float64)
    lo = out["low"].to_numpy(np.float64)
    o = out["open"].to_numpy(np.float64)
    c = out["close"].to_numpy(np.float64)

    # ATR is the single most expensive call in here, so reuse one the frame already carries --
    # `add_atr` names it `atr_{period}` and a scenario has usually added it before we run.
    _ac = f"atr_{int(atr_period)}"
    if _ac in out.columns:
        a = out[_ac].to_numpy(np.float64)
    else:
        from .volatility import atr as _atr
        a = _atr(out, period=atr_period).to_numpy(np.float64)

    if n < 3:
        gaps = pd.DataFrame(columns=["kind", "born_i", "top", "bot", "touch_i", "ce_i", "full_i"])
        return _attach(out, gaps, prefix, n, np.zeros((2, n), np.int32),
                       *(np.full((2, n), np.nan) for _ in range(5)))

    # ---- detect: whole-series, no loop. Index i is the THIRD candle, which is where the gap
    # becomes knowable; `i-1` is the displacement candle and `i-2` the one being gapped away from.
    i = np.arange(2, n)
    bull = lo[2:] > h[:-2]
    bear = h[2:] < lo[:-2]
    kind = np.r_[np.ones(bull.sum(), np.int64), -np.ones(bear.sum(), np.int64)]
    born = np.r_[i[bull], i[bear]]
    # `top` is always the HIGHER edge. For a bullish gap that is the third candle's low and the
    # floor is the first candle's high; for a bearish gap it is the other way round, and writing
    # them in candle order rather than in price order silently inverts every bearish band.
    top = np.r_[lo[2:][bull], lo[:-2][bear]]
    bot = np.r_[h[:-2][bull], h[2:][bear]]

    if len(born):
        order = np.argsort(born, kind="stable")
        kind, born, top, bot = kind[order], born[order], top[order], bot[order]

        # ---- significance. Both gates are ATR-relative so they carry across instruments; an
        # absolute size would silently become a filter on WHEN, not on WHAT.
        size = top - bot
        av = a[born]
        keep = np.isfinite(av) & (av > 0)
        if min_size_atr > 0:
            keep &= size >= min_size_atr * av
        if min_size_pct > 0:
            keep &= size >= (min_size_pct / 100.0) * c[born]
        if min_disp_atr > 0:
            body = np.abs(c[born - 1] - o[born - 1])     # the MIDDLE candle did the displacing
            keep &= body >= min_disp_atr * av
        kind, born, top, bot = kind[keep], born[keep], top[keep], bot[keep]

    if not len(born):
        gaps = pd.DataFrame(columns=["kind", "born_i", "top", "bot", "touch_i", "ce_i", "full_i"])
        return _attach(out, gaps, prefix, n, np.zeros((2, n), np.int32),
                       *(np.full((2, n), np.nan) for _ in range(5)))

    t_i, c_i, f_i = _mitigate(born, kind, top, bot, h, lo, int(max_age_bars))
    dead = {"touch": t_i, "ce": c_i, "full": f_i}[mitigation]
    st = _live_state(born, dead, kind, top, bot, c, a, n, int(max_age_bars))

    gaps = pd.DataFrame({"kind": kind, "born_i": born, "top": top, "bot": bot,
                         "touch_i": t_i, "ce_i": c_i, "full_i": f_i})
    gaps["side"] = np.where(gaps["kind"] > 0, "bullish", "bearish")
    gaps["mid"] = 0.5 * (gaps["top"] + gaps["bot"])
    gaps["size"] = gaps["top"] - gaps["bot"]
    gaps["size_atr"] = gaps["size"] / a[born]
    gaps["dead_i"] = dead
    # A STABLE NAME per gap, for the same reason trendline_table has one: the chart's hover, the
    # dropdown that isolates it and the trade that was vetoed by it must all call it the same
    # thing, so they derive the name from here rather than each inventing its own. FB/FS rather
    # than B/S because a bare S is already a support trendline.
    gaps["id"] = (pd.Series(np.where(kind > 0, "FB", "FS"))
                  + gaps.groupby("kind").cumcount().astype(str))
    if time_col in out.columns:
        # via DatetimeIndex.take, NOT by indexing `.to_numpy()`. A tz-aware column comes back from
        # to_numpy() as an OBJECT array of Timestamps, and converting 58,776 of those back to
        # datetime64 cost 2.6 s -- more than the detection, the mitigation scan and the per-bar
        # pass put together.
        ti = pd.DatetimeIndex(out[time_col])
        gaps["born_t"] = ti.take(born)
        for col in ("touch_i", "ce_i", "full_i", "dead_i"):
            v = gaps[col].to_numpy(np.int64)
            gaps[col.replace("_i", "_t")] = pd.Series(ti.take(np.clip(v, 0, n - 1))).where(v >= 0)
    return _attach(out, gaps, prefix, n, *st)


def _attach(out, gaps, prefix, n, cnt, n_top, n_bot, n_age, n_size, n_dist):
    """Write the per-bar columns and park the gap list in attrs."""
    for s, nm in ((0, "bull"), (1, "bear")):
        out[f"{prefix}_{nm}_n"] = np.asarray(cnt[s], np.int32)
        out[f"{prefix}_{nm}_top"] = n_top[s]
        out[f"{prefix}_{nm}_bot"] = n_bot[s]
        out[f"{prefix}_{nm}_age"] = n_age[s]
        out[f"{prefix}_{nm}_size_atr"] = n_size[s]
        out[f"{prefix}_{nm}_dist_atr"] = n_dist[s]
    out.attrs[f"{prefix}_gaps"] = gaps
    return out


def fvg_table(df: pd.DataFrame, *, prefix: str = "fvg", **kwargs) -> pd.DataFrame:
    """One row per gap — for drawing, and for seeing what was actually found.

    Pass a frame already through `add_fvgs` (the gaps ride along in `df.attrs`), or a raw frame plus
    the same keyword arguments and the detection runs for you.
    """
    gaps = df.attrs.get(f"{prefix}_gaps")
    if gaps is None:
        gaps = add_fvgs(df, prefix=prefix, **kwargs).attrs[f"{prefix}_gaps"]
    return gaps.copy()


def mtf_fvgs(base_df: pd.DataFrame, frames: Optional[Dict[str, pd.DataFrame]] = None, *,
             time_col: str = "t", prefix: str = "fvg", **kwargs) -> pd.DataFrame:
    """Fair value gaps from SEVERAL timeframes, all on the base frame's grid.

    Detects on `base_df` and on every frame in `frames`, then merges the higher-timeframe columns
    with the project's usual anti-lookahead shift: a 1h gap confirmed on the bar stamped 12:00 is
    only knowable when that bar CLOSES at 13:00, so `align_timeframes(shift_bars=1)` makes it
    visible to base bars from 13:00 onward and no earlier.

    Columns are `fvg_bull_n` for the base and `1h__fvg_bull_n` and so on for the rest. Each frame's
    own gap table is kept in `attrs[f"{tf}__{prefix}_gaps"]` so the chart can draw every layer.

    Load the frames with `quant.data.tf_frames`: a 4h gap needs 4h bars, and resampling a short base
    frame cannot produce them.
    """
    from ..data.timeframe import align_timeframes

    out = add_fvgs(base_df, prefix=prefix, time_col=time_col, **kwargs)
    if not frames:
        return out
    base_gaps = out.attrs.get(f"{prefix}_gaps")
    others = {tf: add_fvgs(f.reset_index(drop=True), prefix=prefix, time_col=time_col, **kwargs)
              for tf, f in frames.items()}
    merged = align_timeframes(out, others, time_col=time_col, shift_bars=1)
    # the shift turns the count columns into floats with a leading NaN; 0 means "none live"
    for c in [c for c in merged.columns if c.endswith(("_bull_n", "_bear_n"))]:
        merged[c] = merged[c].fillna(0).astype(np.int32)
    merged.attrs[f"{prefix}_gaps"] = base_gaps
    for tf, f in others.items():
        merged.attrs[f"{tf}__{prefix}_gaps"] = f.attrs.get(f"{prefix}_gaps")
    return merged


def fvg_count_cols(df: pd.DataFrame, side: str, *, prefix: str = "fvg") -> List[str]:
    """Every `{tf}__{prefix}_{bull|bear}_n` column present, base frame first then coarsest-first.

    `side` is "bull" or "bear" — the gap's own direction, not a trade's.
    """
    from ..data.timeframe import tf_minutes

    base = f"{prefix}_{side}_n"
    cols = [base] if base in df.columns else []
    extra = [c for c in df.columns if c.endswith(f"__{base}")]

    def _key(c):
        try:
            return tf_minutes(c.split("__", 1)[0])
        except ValueError:
            return 10 ** 9

    return cols + sorted(extra, key=_key, reverse=True)   # coarsest first: structure before detail
