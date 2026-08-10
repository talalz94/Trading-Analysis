"""
Classical chart-pattern recognition — double tops, head & shoulders, wedges, triangles, flags.

These are geometric shapes over SWING PIVOTS rather than over individual candles, which makes
causality the whole problem. Two rules are enforced everywhere:

1. **A pivot does not exist until it is confirmed.** Pivots come from the ATR-adaptive zigzag
   (`add_zigzag_structure`), which emits a swing only once price has retraced a volatility-scaled
   threshold from the extreme — so the pivot is known at its CONFIRMATION bar, never at the extreme
   itself. Every shape test uses only pivots confirmed at or before the bar being evaluated.
2. **A pattern is not tradable until its neckline breaks.** Detection happens in two stages: a
   candidate shape is registered when its pivots are in place, then the signal fires on the bar
   where price CLOSES through the neckline (or trendline). Drawing the shape earlier is fine for a
   chart; trading it earlier is lookahead, and it is the single most common bug in pattern libraries.

Each confirmed pattern publishes the three levels a trade actually needs — `pat_entry`,
`pat_stop`, `pat_target` — computed by the standard measured-move rule (project the pattern's height
from the breakout point). They are forward-filled into `pat_stop_ff` / `pat_target_ff` so they can be
handed straight to `BacktestConfig(sl_mode='ref_col', ...)` or a price-absolute take-profit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

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


@dataclass
class _Pivot:
    confirm_i: int      # bar at which this pivot became known
    i: int              # bar of the extreme itself
    price: float
    kind: int           # +1 high, -1 low


@dataclass
class _Candidate:
    name: str
    direction: int          # +1 break up, -1 break down
    neckline: float         # the level whose break confirms it
    stop: float             # protective level on the far side of the shape
    height: float           # pattern height, for the measured move
    formed_i: int           # bar the shape was complete and known
    expires_i: int
    pivots: List[_Pivot] = field(default_factory=list)


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


def _close(a: float, b: float, tol: float, scale: float) -> bool:
    """Two pivot prices are 'the same level' within `tol` (a fraction of the pattern's own scale)."""
    return abs(a - b) <= tol * max(scale, 1e-12)


def _slope(p0: _Pivot, p1: _Pivot) -> float:
    dt = max(p1.i - p0.i, 1)
    return (p1.price - p0.price) / dt


def _detect(pv: List[_Pivot], *, tol: float, min_height_frac: float,
            atr: float) -> List[_Candidate]:
    """Try every shape against the most recent pivots. `pv` is oldest -> newest, all CONFIRMED."""
    out: List[_Candidate] = []
    if len(pv) < 4:
        return out
    last = pv[-1]
    formed = last.confirm_i
    highs = [p for p in pv if p.kind > 0]
    lows = [p for p in pv if p.kind < 0]
    if len(highs) < 2 or len(lows) < 2:
        return out
    scale = max(max(p.price for p in pv) - min(p.price for p in pv), 1e-12)
    if scale < min_height_frac * atr:
        return out                                   # too small to be a pattern, just noise

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    # ---- double top / bottom: two equal extremes with a trough/peak between -------------
    if _close(h1.price, h2.price, tol, scale) and h1.i < h2.i:
        mid = [p for p in lows if h1.i < p.i < h2.i]
        if mid:
            neck = min(p.price for p in mid)
            top = max(h1.price, h2.price)
            out.append(_Candidate("double_top", -1, neck, top, top - neck, formed, 0,
                                  pivots=[h1] + mid[-1:] + [h2]))
    if _close(l1.price, l2.price, tol, scale) and l1.i < l2.i:
        mid = [p for p in highs if l1.i < p.i < l2.i]
        if mid:
            neck = max(p.price for p in mid)
            bot = min(l1.price, l2.price)
            out.append(_Candidate("double_bottom", +1, neck, bot, neck - bot, formed, 0,
                                  pivots=[l1] + mid[-1:] + [l2]))

    # ---- triple top / bottom --------------------------------------------------------
    if len(highs) >= 3:
        a, b, c = highs[-3], highs[-2], highs[-1]
        if _close(a.price, b.price, tol, scale) and _close(b.price, c.price, tol, scale):
            mid = [p for p in lows if a.i < p.i < c.i]
            if mid:
                neck = min(p.price for p in mid)
                top = max(a.price, b.price, c.price)
                out.append(_Candidate("triple_top", -1, neck, top, top - neck, formed, 0,
                                      pivots=[a, b, c] + mid[-2:]))
    if len(lows) >= 3:
        a, b, c = lows[-3], lows[-2], lows[-1]
        if _close(a.price, b.price, tol, scale) and _close(b.price, c.price, tol, scale):
            mid = [p for p in highs if a.i < p.i < c.i]
            if mid:
                neck = max(p.price for p in mid)
                bot = min(a.price, b.price, c.price)
                out.append(_Candidate("triple_bottom", +1, neck, bot, neck - bot, formed, 0,
                                      pivots=[a, b, c] + mid[-2:]))

    # ---- head & shoulders: 3 highs, middle highest, shoulders roughly level ----------
    if len(highs) >= 3 and len(lows) >= 2:
        ls, head, rs = highs[-3], highs[-2], highs[-1]
        troughs = [p for p in lows if ls.i < p.i < rs.i]
        if (head.price > ls.price and head.price > rs.price
                and _close(ls.price, rs.price, tol * 1.5, scale) and len(troughs) >= 2):
            neck = float(np.mean([p.price for p in troughs[-2:]]))
            out.append(_Candidate("head_shoulders", -1, neck, head.price,
                                  head.price - neck, formed, 0,
                                  pivots=[ls] + troughs[-2:] + [head, rs]))
    if len(lows) >= 3 and len(highs) >= 2:
        ls, head, rs = lows[-3], lows[-2], lows[-1]
        peaks = [p for p in highs if ls.i < p.i < rs.i]
        if (head.price < ls.price and head.price < rs.price
                and _close(ls.price, rs.price, tol * 1.5, scale) and len(peaks) >= 2):
            neck = float(np.mean([p.price for p in peaks[-2:]]))
            out.append(_Candidate("inverse_head_shoulders", +1, neck, head.price,
                                  neck - head.price, formed, 0,
                                  pivots=[ls] + peaks[-2:] + [head, rs]))

    # ---- triangles, wedges, rectangles: compare the two trendlines -------------------
    if len(highs) >= 2 and len(lows) >= 2:
        sh = _slope(h1, h2)                      # slope of the upper line
        sl = _slope(l1, l2)                      # slope of the lower line
        span = max(max(h2.i, l2.i) - min(h1.i, l1.i), 1)
        flat = tol * scale / span                # "flat" means slope smaller than this
        hi_lvl, lo_lvl = max(h1.price, h2.price), min(l1.price, l2.price)
        height = hi_lvl - lo_lvl
        converging = (h2.price - l2.price) < (h1.price - l1.price)

        if abs(sh) <= flat and sl > flat:
            out.append(_Candidate("ascending_triangle", +1, max(h1.price, h2.price),
                                  l2.price, height, formed, 0, pivots=[h1, h2, l1, l2]))
        elif abs(sl) <= flat and sh < -flat:
            out.append(_Candidate("descending_triangle", -1, min(l1.price, l2.price),
                                  h2.price, height, formed, 0, pivots=[h1, h2, l1, l2]))
        elif sh < -flat and sl > flat and converging:
            # symmetrical: breaks either way, registered on both sides
            out.append(_Candidate("symmetrical_triangle", +1, h2.price, l2.price,
                                  height, formed, 0, pivots=[h1, h2, l1, l2]))
            out.append(_Candidate("symmetrical_triangle", -1, l2.price, h2.price,
                                  height, formed, 0, pivots=[h1, h2, l1, l2]))
        elif sh > flat and sl > flat and converging:
            out.append(_Candidate("rising_wedge", -1, min(l1.price, l2.price),
                                  max(h1.price, h2.price), height, formed, 0,
                                  pivots=[h1, h2, l1, l2]))
        elif sh < -flat and sl < -flat and converging:
            out.append(_Candidate("falling_wedge", +1, max(h1.price, h2.price),
                                  min(l1.price, l2.price), height, formed, 0,
                                  pivots=[h1, h2, l1, l2]))
        elif abs(sh) <= flat and abs(sl) <= flat:
            out.append(_Candidate("rectangle_bull", +1, hi_lvl, lo_lvl, height, formed, 0,
                                  pivots=[h1, h2, l1, l2]))
            out.append(_Candidate("rectangle_bear", -1, lo_lvl, hi_lvl, height, formed, 0,
                                  pivots=[h1, h2, l1, l2]))
        elif sh > flat and sl < -flat:
            out.append(_Candidate("broadening_top", -1, min(l1.price, l2.price),
                                  max(h1.price, h2.price), height, formed, 0,
                                  pivots=[h1, h2, l1, l2]))
            out.append(_Candidate("broadening_bottom", +1, max(h1.price, h2.price),
                                  min(l1.price, l2.price), height, formed, 0,
                                  pivots=[h1, h2, l1, l2]))
    return out


def _flagpole(pv: List[_Pivot], cand: _Candidate, *, pole_mult: float) -> Optional[str]:
    """A consolidation preceded by a sharp move is a flag/pennant rather than a plain range."""
    if len(pv) < 4:
        return None
    start = min(p.i for p in pv[-4:])
    pole = [p for p in pv if p.i < start]
    if len(pole) < 2:
        return None
    move = pole[-1].price - pole[-2].price
    if abs(move) < pole_mult * cand.height:
        return None
    return "bull" if move > 0 else "bear"


def add_chart_patterns(df: pd.DataFrame, *, atr_mult: float = 2.0, atr_period: int = 14,
                       tol: float = 0.12, lookback_pivots: int = 6, max_wait: int = 150,
                       min_height_frac: float = 1.0, pole_mult: float = 1.5,
                       patterns: Optional[Sequence[str]] = None,
                       prefix: str = "pat",
                       _collect: Optional[List[dict]] = None) -> pd.DataFrame:
    """Detect classical chart patterns and publish their entry / stop / target levels.

    Adds, all NaN or False except on the bar a pattern CONFIRMS:
      `{prefix}_<name>`  one boolean column per pattern
      `{prefix}_signal`  any pattern confirmed on this bar
      `{prefix}_dir`     +1 / -1 for the confirmed pattern
      `{prefix}_name`    its name
      `{prefix}_entry`   the breakout price (the neckline)
      `{prefix}_stop`    protective level on the far side of the shape
      `{prefix}_target`  measured move: entry +/- the pattern height
    plus `_ff` forward-filled copies of entry/stop/target/dir, so a stop or target can be read
    per-bar by the engine after the trade opens.

    Parameters: `atr_mult`/`atr_period` size the zigzag that produces the pivots; `tol` is how close
    two pivots must be to count as the same level (as a fraction of the pattern's own height);
    `lookback_pivots` is how many recent pivots a shape may span; `max_wait` is how many bars a
    formed shape stays live waiting for its break; `min_height_frac` rejects shapes smaller than
    that multiple of ATR as noise; `pole_mult` is how large the prior move must be relative to the
    consolidation for it to be reclassified as a flag or pennant.
    """
    names = list(patterns) if patterns is not None else ALL_PATTERNS
    unknown = [p for p in names if p not in PATTERN_DIRECTION]
    if unknown:
        raise KeyError(f"unknown chart pattern(s) {unknown}; available: {ALL_PATTERNS}")

    out = df.copy()
    if f"zz_label" not in out.columns:
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
    confirmed: List[dict] = []
    pi = 0
    for i in range(n):
        # 1) register candidates from every pivot CONFIRMED on this bar
        while pi < len(pivots) and pivots[pi].confirm_i == i:
            window = pivots[max(0, pi + 1 - lookback_pivots): pi + 1]
            a = atr_s[i] if np.isfinite(atr_s[i]) else 0.0
            for cand in _detect(window, tol=tol, min_height_frac=min_height_frac, atr=a):
                pole = _flagpole(window, cand, pole_mult=pole_mult)
                if pole and cand.name in ("symmetrical_triangle", "rectangle_bull",
                                          "rectangle_bear"):
                    # A flag/pennant continues the pole, so ONLY the candidate already registered
                    # on that side qualifies. Relabelling the opposite-side candidate would keep
                    # its neckline and stop, leaving the stop on the wrong side of entry.
                    if cand.direction != (+1 if pole == "bull" else -1):
                        continue
                    is_pennant = cand.name == "symmetrical_triangle"
                    cand.name = f"{pole}_pennant" if is_pennant else f"{pole}_flag"
                if cand.name not in hits:
                    continue
                # invariant: risk must be defined -- the stop sits on the far side of the entry.
                # These levels drive real stop-losses, so an ill-formed shape is dropped, not fixed.
                if cand.direction > 0 and not (cand.stop < cand.neckline):
                    continue
                if cand.direction < 0 and not (cand.stop > cand.neckline):
                    continue
                if not np.isfinite(cand.height) or cand.height <= 0:
                    continue
                cand.expires_i = i + max_wait
                live.append(cand)
            pi += 1

        # 2) a live shape confirms when price CLOSES through its neckline
        still: List[_Candidate] = []
        for cand in live:
            if i > cand.expires_i:
                continue
            if i <= cand.formed_i:
                still.append(cand)
                continue
            broke = (close[i] > cand.neckline if cand.direction > 0
                     else close[i] < cand.neckline)
            if not broke:
                still.append(cand)
                continue
            hits[cand.name][i] = True
            pv = sorted(cand.pivots, key=lambda q: q.i)
            confirmed.append(dict(
                confirm_i=i, formed_i=cand.formed_i, pattern=cand.name,
                direction=cand.direction, entry=cand.neckline, stop=cand.stop,
                target=(cand.neckline + cand.height if cand.direction > 0
                        else cand.neckline - cand.height),
                pivot_i=[q.i for q in pv], pivot_price=[q.price for q in pv],
                hi_i=[q.i for q in pv if q.kind > 0],
                hi_price=[q.price for q in pv if q.kind > 0],
                lo_i=[q.i for q in pv if q.kind < 0],
                lo_price=[q.price for q in pv if q.kind < 0]))
            e_entry[i] = cand.neckline
            e_stop[i] = cand.stop
            e_target[i] = (cand.neckline + cand.height if cand.direction > 0
                           else cand.neckline - cand.height)
            e_dir[i] = cand.direction
            e_name[i] = cand.name
        live = still

    for name in names:
        out[f"{prefix}_{name}"] = hits[name]
    out[f"{prefix}_signal"] = np.any(np.column_stack([hits[k] for k in names]), axis=1) \
        if names else False
    out[f"{prefix}_dir"] = e_dir
    out[f"{prefix}_name"] = e_name
    out[f"{prefix}_entry"] = e_entry
    out[f"{prefix}_stop"] = e_stop
    out[f"{prefix}_target"] = e_target
    for c in ("entry", "stop", "target", "dir"):
        src = out[f"{prefix}_{c}"]
        out[f"{prefix}_{c}_ff"] = src.replace(0, np.nan).ffill() if c == "dir" else src.ffill()
    if _collect is not None:
        _collect.extend(confirmed)
    return out


def chart_pattern_shapes(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """The GEOMETRY of every confirmed pattern — the pivot coordinates that make up the shape.

    `add_chart_patterns` returns per-bar columns, which is what a strategy needs but not enough to
    DRAW a triangle: for that you need the pivots the two trendlines pass through. This returns one
    row per confirmed pattern with `pivot_i`/`pivot_price` (the full outline, in time order) plus
    `hi_*` and `lo_*` (the upper and lower trendline points), alongside entry / stop / target.

    Accepts the same tuning arguments as `add_chart_patterns`.
    """
    shapes: List[dict] = []
    add_chart_patterns(df, _collect=shapes, **kwargs)
    out = pd.DataFrame(shapes)
    if not len(out):
        return pd.DataFrame(columns=["confirm_i", "formed_i", "pattern", "direction", "entry",
                                     "stop", "target", "pivot_i", "pivot_price", "hi_i",
                                     "hi_price", "lo_i", "lo_price"])
    if "t" in df.columns:
        tv = df["t"].to_numpy()
        out["t"] = tv[out["confirm_i"].to_numpy(int)]
        out["pivot_t"] = [list(tv[np.asarray(v, int)]) for v in out["pivot_i"]]
        out["hi_t"] = [list(tv[np.asarray(v, int)]) if len(v) else [] for v in out["hi_i"]]
        out["lo_t"] = [list(tv[np.asarray(v, int)]) if len(v) else [] for v in out["lo_i"]]
    return out


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
