"""
SHAPE-QUALITY AUDIT — does each detected pattern actually look like what it claims to be?

Three detector bugs in a row were found by LOOKING at a rendered chart, with the whole test suite
green throughout: touch dots that were not touches, "symmetrical" triangles with 17:1 slopes, and a
parallel channel emitted as a wedge. Every existing test asserts an INVARIANT — causality, subset
relations, monotonicity — and none asserts that a shape resembles its own definition. This is that
missing check, made repeatable.

Six checks. Each has a pass criterion derived from the pattern's own definition, so a violation is a
bug rather than a matter of taste:

  1. CONTAINMENT     what fraction of the bars a shape spans stay inside its boundaries. A channel or
                     triangle that price keeps escaping is not one. This is the general form of "that
                     triangle looks wrong", and it is the check that would have caught the asymmetric
                     ones without anybody opening a chart.
  2. CONTRACT        each shape must satisfy its OWN definition: a channel parallel within
                     `width_tol_frac`, a wedge narrowing by at least `converge_min_frac`, an ascending
                     triangle actually flat-topped, a symmetrical triangle within `slope_sym_max`.
                     These must hold for 100% of detections; anything less is a classification bug.
  3. TOUCH QUALITY   residual of each touch from its line against the tolerance that admitted it, and
                     whether the touches SPAN the shape or cluster at one end.
  4. RISK GEOMETRY   stop and target on the correct sides of entry, and a sane reward-to-risk. An
                     apex stop once produced 43 points of risk against a 2,328-point target.
  5. OVERLAP         two shapes of the same name describing the same price action. De-duplication
                     lives here, and so did the lookahead.
  6. CAUSALITY       truncation on REAL data at many points. The suite only truncates a synthetic
                     random walk, which need not exercise the same paths.

Everything runs at the shipped `TUNED_SPECS` over the balanced nine-window cycle, so the report
describes what the library actually does by default rather than a hand-picked configuration.
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from quant.data import get_ohlcv                                            # noqa: E402
from quant.indicators import add_atr, add_chart_patterns, add_trendlines    # noqa: E402
from quant.indicators import chart_pattern_shapes_tuned, trendline_table    # noqa: E402
from quant.indicators import tuned_specs                                    # noqa: E402
from quant.indicators.chart_patterns import PatternSpec, PATTERN_DIRECTION  # noqa: E402
from experiments.phase9_long_fair_test import WINDOWS                       # noqa: E402

TF = "15m"
OUT = "experiments/results_pattern_quality_audit.csv"
TRUNC_POINTS = 8            # truncation boundaries per window for the causality check
CONTAIN_MIN = 0.80          # a shape should hold this share of its bars inside its boundaries
RR_BOUNDS = (0.2, 20.0)     # reward-to-risk outside this is a geometry failure

_DEF = PatternSpec()
FLAT_TOL, LINE_TOL = _DEF.flat_tol_atr, 0.6      # LINE_TOL is TUNED_COMMON's value
SYM_MAX, WIDTH_TOL, CONV_MIN = _DEF.slope_sym_max, _DEF.width_tol_frac, _DEF.converge_min_frac


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _load(key: str, symbol: str = "BTCUSDT", tf: str = TF,
          source: str = "binance") -> pd.DataFrame:
    a, b = WINDOWS[key]
    d = get_ohlcv(symbol, tf, start=a, end=b, source=source, tz="UTC")
    return add_atr(d, 14).reset_index(drop=True)


def _line(slope: float, icept: float, x: np.ndarray) -> np.ndarray:
    return slope * x + icept


def _has_lines(r) -> bool:
    """Level shapes (H&S, double/triple tops) have no fitted boundaries, only a neckline."""
    return np.isfinite(r.upper_slope) and np.isfinite(r.lower_slope)


def _pivot_frame(r):
    """The x-range the DETECTOR validated over: the span of the pivot window, not the drawn range.

    This distinction matters and the first run of this audit got it wrong. A shape is CLASSIFIED from
    its pivots, but it CONFIRMS when price later breaks out — up to `max_wait` (150) bars past the
    last pivot. Checking the contract over the drawn range therefore extrapolates the boundaries well
    beyond where they were fitted, and reports drift that the detector never claimed was absent.
    The contract belongs in the detector's frame; the extrapolation is measured separately below.
    """
    pi = np.asarray(r.pivot_i, float)
    return float(pi.min()), float(pi.max())


def _width_at(r, x: float) -> float:
    return (_line(r.upper_slope, r.upper_intercept, x)
            - _line(r.lower_slope, r.lower_intercept, x))


def _width_change(r, x0=None, x1=None) -> float:
    """Fractional change in the gap between the boundaries. <0 narrows. Detector frame by default."""
    if x0 is None:
        x0, x1 = _pivot_frame(r)
    w0, w1 = _width_at(r, x0), _width_at(r, x1)
    return float((w1 - w0) / max(abs(w0), abs(w1), 1e-12))


def _slope_ratio(r) -> float:
    a, b = abs(r.upper_slope), abs(r.lower_slope)
    lo, hi = min(a, b), max(a, b)
    return float(hi / lo) if lo > 1e-12 else np.inf


def _is_flat(slope: float, span: int, atr: float) -> bool:
    return abs(slope) * max(span, 1) <= FLAT_TOL * atr


# ---------------------------------------------------------------------------
# checks 1-5, all from one detection pass per window
# ---------------------------------------------------------------------------
def audit_shapes(d: pd.DataFrame, window: str) -> List[dict]:
    hi = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    atr_arr = d["atr_14"].to_numpy(float)
    atr_med = float(np.nanmedian(atr_arr))
    shapes = chart_pattern_shapes_tuned(d)
    rows: List[dict] = []
    for r in shapes.itertuples():
        a, b = int(r.start_i), int(r.confirm_i)
        span = max(b - a, 1)
        atr_here = atr_arr[b] if np.isfinite(atr_arr[b]) else atr_med
        rec: Dict[str, object] = dict(
            window=window, pattern=r.pattern, direction=int(r.direction),
            start_i=a, confirm_i=b, span_bars=int(r.span_bars),
            touches_upper=int(r.touches_upper), touches_lower=int(r.touches_lower),
            n_touch_drawn=len(r.touch_i), has_lines=_has_lines(r))

        # ---- 1. containment -------------------------------------------------
        if _has_lines(r) and b > a:
            x = np.arange(a, b + 1)
            up = _line(r.upper_slope, r.upper_intercept, x)
            dn = _line(r.lower_slope, r.lower_intercept, x)
            tol = LINE_TOL * atr_here
            inside = (hi[a:b + 1] <= up + tol) & (lo[a:b + 1] >= dn - tol)
            rec["containment"] = float(inside.mean())
        else:
            rec["containment"] = np.nan

        # ---- 2. definitional contract --------------------------------------
        ok, why = True, ""
        if _has_lines(r):
            x0, x1 = _pivot_frame(r)
            span = max(int(x1 - x0), 1)
            f_i = int(r.formed_i)
            atr_here = atr_arr[f_i] if np.isfinite(atr_arr[f_i]) else atr_med
            wc = _width_change(r)
            sr = _slope_ratio(r)
            # how far the shape was EXTRAPOLATED past its last pivot to reach the break, and what
            # that did to its width. This is what the chart actually draws, so a large number here
            # means the drawn object drifts from the geometry that was validated.
            rec["extrap_bars"] = int(b - x1)
            rec["drift_to_break"] = (_width_change(r, x1, float(b))
                                     if b > x1 else 0.0)
            rec["width_change"] = wc
            rec["slope_ratio"] = sr
            fu = _is_flat(r.upper_slope, span, atr_here)
            fl = _is_flat(r.lower_slope, span, atr_here)
            p = r.pattern
            if p.startswith("channel"):
                ok, why = abs(wc) <= WIDTH_TOL + 1e-9, f"channel width drifted {wc:+.3f}"
            elif p in ("symmetrical_triangle", "broadening_top", "broadening_bottom",
                       "bull_pennant", "bear_pennant"):
                ok, why = sr <= SYM_MAX + 1e-9, f"slope ratio {sr:.2f} > {SYM_MAX}"
                if ok and p.startswith("broadening"):
                    ok, why = wc >= CONV_MIN - 1e-9, f"broadening did not widen ({wc:+.3f})"
                elif ok and p == "symmetrical_triangle":
                    ok, why = wc <= -CONV_MIN + 1e-9, f"triangle did not narrow ({wc:+.3f})"
            elif p in ("rising_wedge", "falling_wedge"):
                same = np.sign(r.upper_slope) == np.sign(r.lower_slope)
                ok = same and wc <= -CONV_MIN + 1e-9
                why = f"wedge slopes same-sign={same} width {wc:+.3f}"
            elif p == "ascending_triangle":
                ok, why = fu and not fl, "ascending triangle is not flat-topped"
            elif p == "descending_triangle":
                ok, why = fl and not fu, "descending triangle is not flat-bottomed"
            elif p.startswith("rectangle"):
                ok, why = fu and fl, "rectangle boundary is not flat"
        else:
            rec["width_change"] = np.nan
            rec["slope_ratio"] = np.nan
            rec["extrap_bars"] = int(b - max(np.asarray(r.pivot_i, float).max(), a))
            rec["drift_to_break"] = np.nan
        rec["contract_ok"] = bool(ok)
        rec["contract_why"] = "" if ok else why

        # ---- 3. touch quality ----------------------------------------------
        ti = np.asarray(r.touch_i, float)
        tp = np.asarray(r.touch_price, float)
        if _has_lines(r) and len(ti):
            du = np.abs(tp - _line(r.upper_slope, r.upper_intercept, ti))
            dl = np.abs(tp - _line(r.lower_slope, r.lower_intercept, ti))
            resid = np.minimum(du, dl) / max(atr_here, 1e-12)
            rec["touch_resid_med"] = float(np.median(resid))
            rec["touch_resid_max"] = float(resid.max())
            rec["touch_span_frac"] = float((ti.max() - ti.min()) / span) if len(ti) > 1 else 0.0
        else:
            rec["touch_resid_med"] = rec["touch_resid_max"] = np.nan
            rec["touch_span_frac"] = np.nan

        # ---- 4. risk geometry ----------------------------------------------
        e, s_, t_ = float(r.entry), float(r.stop), float(r.target)
        good_sides = ((s_ < e < t_) if r.direction > 0 else
                      (s_ > e > t_) if r.direction < 0 else True)
        risk = abs(e - s_)
        rr = abs(t_ - e) / risk if risk > 1e-12 else np.inf
        rec["rr"] = float(rr)
        rec["risk_sides_ok"] = bool(good_sides and np.isfinite([e, s_, t_]).all())
        rec["rr_ok"] = bool(RR_BOUNDS[0] <= rr <= RR_BOUNDS[1])
        rows.append(rec)
    return rows


def audit_overlap(rows: List[dict]) -> pd.DataFrame:
    """Two shapes of the SAME name covering the same stretch — what de-duplication exists to stop."""
    df = pd.DataFrame(rows)
    out = []
    if not len(df):
        return pd.DataFrame(columns=["window", "pattern", "n", "overlapping_pairs", "worst_frac"])
    for (w, p), g in df.groupby(["window", "pattern"]):
        iv = g[["start_i", "confirm_i"]].to_numpy()
        pairs, worst = 0, 0.0
        for i in range(len(iv)):
            for j in range(i + 1, len(iv)):
                a0, a1 = iv[i]
                b0, b1 = iv[j]
                inter = max(0, min(a1, b1) - max(a0, b0))
                shorter = max(1, min(a1 - a0, b1 - b0))
                f = inter / shorter
                if f > 0.5:
                    pairs += 1
                worst = max(worst, f)
        out.append(dict(window=w, pattern=p, n=len(g), overlapping_pairs=pairs,
                        worst_frac=round(worst, 3)))
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# check 6 — causality on real data
# ---------------------------------------------------------------------------
def audit_causality(d: pd.DataFrame, window: str) -> List[dict]:
    names = list(tuned_specs())
    spec = tuned_specs()
    groups: Dict[float, Dict[str, dict]] = {}
    for nm, cfg in spec.items():
        c = dict(cfg)
        groups.setdefault(float(c.pop("atr_mult")), {})[nm] = c

    def detect(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        cols: Dict[str, np.ndarray] = {}
        for am, per in groups.items():
            o = add_chart_patterns(frame, patterns=list(per), atr_mult=am, settings=per)
            for nm in per:
                cols[nm] = o[f"pat_{nm}"].to_numpy(bool)
        return cols

    full = detect(d)
    n = len(d)
    out = []
    for k in np.linspace(n * 0.25, n * 0.95, TRUNC_POINTS).astype(int):
        part = detect(d.iloc[:k].copy())
        for nm in names:
            mism = int((full[nm][:k] != part[nm]).sum())
            out.append(dict(window=window, k=int(k), pattern=nm, mismatches=mism,
                            fires_full=int(full[nm][:k].sum()), fires_trunc=int(part[nm].sum())))
    return out


def audit_dots(d: pd.DataFrame, window: str) -> List[dict]:
    """Does every drawn TOUCH marker actually sit on the thing it claims to touch?

    This check did not exist, and its absence is why three defects reached a rendered chart. A
    trendline shape's dots must lie on one of its two fitted boundaries. A LEVEL shape's dots must lie
    on the extreme the pattern is named for — a triple top marked its two intervening TROUGHS as
    touches, putting 33-40% of its dots on the low of a candle.
    """
    hi = d["high"].to_numpy(float)
    lo = d["low"].to_numpy(float)
    atr_arr = d["atr_14"].to_numpy(float)
    atr_med = float(np.nanmedian(atr_arr))
    shapes = chart_pattern_shapes_tuned(d)
    rows = []
    for r in shapes.itertuples():
        # the ATR the FIT used, not the one at confirmation. Measuring a touch against a later,
        # smaller ATR reported 8 dots as off their line when 0 of 10,297 actually are — the same
        # denominator mistake this audit already made once with the contract frame.
        f_i = int(r.formed_i)
        atr_here = atr_arr[f_i] if np.isfinite(atr_arr[f_i]) else atr_med
        off_line = wrong_side = 0
        for i, px in zip(r.touch_i, r.touch_price):
            i = int(i)
            if _has_lines(r):
                du = abs(px - _line(r.upper_slope, r.upper_intercept, i))
                dl = abs(px - _line(r.lower_slope, r.lower_intercept, i))
                if min(du, dl) > LINE_TOL * atr_here * 2.0:
                    off_line += 1
            else:
                # a top's touches are highs, a bottom's are lows
                near_hi = abs(px - hi[i]) < abs(px - lo[i])
                want_hi = int(r.direction) < 0
                if near_hi != want_hi:
                    wrong_side += 1
        rows.append(dict(window=window, pattern=r.pattern, dots=len(r.touch_i),
                         dots_off_line=off_line, dots_wrong_side=wrong_side,
                         # only meaningful where the pattern claims ONE level. A head-and-shoulders
                         # has no single level — the head is supposed to stand clear of the
                         # shoulders — so reporting its full spread just looks like a failure.
                         level_spread_atr=(
                             float((max(r.touch_price) - min(r.touch_price)) / max(atr_here, 1e-9))
                             if (not _has_lines(r) and len(r.touch_price) > 1
                                 and "head_shoulders" not in r.pattern) else np.nan)))
    return rows


# ---------------------------------------------------------------------------
# trendlines
# ---------------------------------------------------------------------------
def audit_trendlines(d: pd.DataFrame, window: str) -> dict:
    o = add_trendlines(d)
    tl = trendline_table(o)
    atr_med = float(np.nanmedian(d["atr_14"].to_numpy(float)))
    lo_px, hi_px = float(d["low"].min()), float(d["high"].max())
    rng = hi_px - lo_px
    resid, off_chart, bad_touch = [], 0, 0
    for r in tl.itertuples():
        ti = np.asarray(r.touch_i, float)
        tp = np.asarray(r.touch_price, float)
        if len(ti):
            resid.append(float(np.median(np.abs(tp - (r.slope * ti + r.intercept)) / atr_med)))
        if len(ti) != r.touches:
            bad_touch += 1
        if not (lo_px - 3 * rng < r.y_end < hi_px + 3 * rng):
            off_chart += 1
    # DISTANCE FROM PRICE. A support line price has left 100 ATR behind is a ray heading off the
    # chart, not support. Nothing checked this, and the old bound here (endpoint within 3x the whole
    # price range) was so loose that lines 119 ATR away passed it.
    c_all = o["close"].to_numpy(float)
    d_start, d_end, far = [], [], 0
    atr_all = o["atr_14"].to_numpy(float) if "atr_14" in o.columns else None
    for r in tl.itertuples():
        s_i, e_i = int(r.start_i), int(r.end_i)
        # the LOCAL ATR, because that is what the retirement rule compares against. Using the window
        # median flagged 38 lines as far from price when by the rule's own measure none are.
        a_s = (atr_all[e_i] if atr_all is not None and np.isfinite(atr_all[e_i]) and atr_all[e_i] > 0
               else atr_med)
        ds = abs((r.slope * s_i + r.intercept) - c_all[s_i]) / a_s
        de = abs((r.slope * e_i + r.intercept) - c_all[e_i]) / a_s
        d_start.append(ds); d_end.append(de)
        if de > 20.0:
            far += 1
    # TOUCHES INSIDE THE DRAWN SPAN. Lines were drawn from `born_i`, which is after every pivot they
    # are fitted through, so all 36 lines in the first sample had their dots outside the segment.
    outside = sum(1 for r in tl.itertuples()
                  if len(r.touch_i) and not (int(r.start_i) <= min(r.touch_i)
                                             and max(r.touch_i) <= int(r.end_i)))
    c = o["close"].to_numpy(float)
    dn = o["tl_break_dn"].to_numpy(bool)
    up = o["tl_break_up"].to_numpy(bool)
    sup = o["tl_sup"].to_numpy(float)
    res = o["tl_res"].to_numpy(float)
    wrong = int((c[dn] >= sup[dn]).sum() + (c[up] <= res[up]).sum())
    return dict(window=window, lines=len(tl), broke=int(tl.broke.sum()) if len(tl) else 0,
                dist_start_med=float(np.median(d_start)) if d_start else np.nan,
                dist_end_med=float(np.median(d_end)) if d_end else np.nan,
                dist_end_max=float(max(d_end)) if d_end else np.nan,
                lines_far_from_price=far, touches_outside_span=outside,
                median_touches=float(tl.touches.median()) if len(tl) else np.nan,
                touch_resid_med=float(np.median(resid)) if resid else np.nan,
                off_chart_endpoints=off_chart, touch_count_mismatch=bad_touch,
                breaks_on_wrong_side=wrong, both_ways_same_bar=int((dn & up).sum()),
                pct_bars_sup=round(100 * np.isfinite(sup).mean(), 1),
                pct_bars_res=round(100 * np.isfinite(res).mean(), 1))


# ---------------------------------------------------------------------------
def main(windows=None, trunc_points=None, skip_causality=False,
         symbol: str = "BTCUSDT", tf: str = TF, source: str = "binance", out: str = OUT):
    """Audit `symbol` over `windows`.

    The instrument is a parameter because the detector claims to be SCALE-FREE — everything is
    ratios and ATR multiples — so gold near $2,000 and BTC near $60,000 should behave identically.
    Running both is the cheapest available test of that claim.
    """
    global TRUNC_POINTS
    if trunc_points is not None:
        TRUNC_POINTS = trunc_points
    keys = list(windows or WINDOWS)
    t0 = time.time()
    print(f"AUDIT: {symbol} {tf} over {len(keys)} windows, {TRUNC_POINTS} truncation points",
          flush=True)
    shape_rows, caus_rows, tl_rows, dot_rows = [], [], [], []
    for wi, k in enumerate(keys, 1):
        try:
            d = _load(k, symbol=symbol, tf=tf, source=source)
        except Exception as e:                      # a window with no data must not kill the run
            print(f"  [{wi}/{len(keys)}] {k} SKIPPED ({type(e).__name__}: {e})", flush=True)
            continue
        if len(d) < 500:
            print(f"  [{wi}/{len(keys)}] {k} SKIPPED (only {len(d)} bars)", flush=True)
            continue
        shape_rows += audit_shapes(d, k)
        dot_rows += audit_dots(d, k)
        tl_rows.append(audit_trendlines(d, k))
        if not skip_causality:
            caus_rows += audit_causality(d, k)
        print(f"  [{wi}/{len(keys)}] {k} done ({time.time()-t0:.0f}s)", flush=True)

    if not shape_rows:
        print("\n  NO DATA — every window was skipped; nothing to audit")
        return
    shapes = pd.DataFrame(shape_rows)
    shapes.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    tl = pd.DataFrame(tl_rows)
    caus = pd.DataFrame(caus_rows)

    def hdr(s):
        print("\n" + "=" * 108); print(s); print("=" * 108)

    hdr("1. CONTAINMENT — share of a shape's bars that stay INSIDE its boundaries\n"
        f"   two-boundary shapes only; a shape below {CONTAIN_MIN:.2f} does not describe its price action")
    tb = shapes[shapes.has_lines]
    if len(tb):
        g = (tb.groupby("pattern")
               .agg(n=("containment", "size"), median=("containment", "median"),
                    p10=("containment", lambda s: float(np.nanpercentile(s, 10))),
                    worst=("containment", "min"),
                    pct_below=("containment", lambda s: round(100 * (s < CONTAIN_MIN).mean(), 1)))
               .reset_index().sort_values("median"))
        print(g.round(3).to_string(index=False))

    hdr("2. DEFINITIONAL CONTRACT — does each shape satisfy its OWN definition?\n"
        "   this must be 100%. anything less is a classification bug, not a matter of degree")
    g2 = (shapes.groupby("pattern")
            .agg(n=("contract_ok", "size"), pass_pct=("contract_ok", lambda s: round(100 * s.mean(), 1)))
            .reset_index().sort_values("pass_pct"))
    print(g2.to_string(index=False))
    bad = shapes[~shapes.contract_ok]
    if len(bad):
        print(f"\n   {len(bad)} VIOLATIONS — examples:")
        print(bad[["window", "pattern", "start_i", "confirm_i", "contract_why"]]
              .head(15).to_string(index=False))
    else:
        print("\n   no violations")

    hdr("3. TOUCH QUALITY — residual vs the 0.6 ATR tolerance that admitted the touch,\n"
        "   and whether touches SPAN the shape (1.0) or cluster at one end (0.0)")
    if len(tb):
        g3 = (tb.groupby("pattern")
                .agg(n=("touch_resid_med", "size"), resid_med=("touch_resid_med", "median"),
                     resid_max=("touch_resid_max", "max"),
                     span_frac=("touch_span_frac", "median"),
                     drawn=("n_touch_drawn", "median"))
                .reset_index().sort_values("resid_med", ascending=False))
        print(g3.round(3).to_string(index=False))

    hdr("3b. EXTRAPOLATION — how far past its last pivot a shape is drawn to reach its break,\n"
        "   and how much the boundary gap drifts over that stretch. The geometry was validated at\n"
        "   the pivots; this is the part the chart draws on trust.")
    if len(tb):
        g3b = (tb.groupby("pattern")
                 .agg(n=("extrap_bars", "size"), extrap_med=("extrap_bars", "median"),
                      extrap_max=("extrap_bars", "max"),
                      drift_med=("drift_to_break", lambda s: float(np.nanmedian(np.abs(s)))),
                      drift_p90=("drift_to_break",
                                 lambda s: float(np.nanpercentile(np.abs(s), 90))))
                 .reset_index().sort_values("drift_p90", ascending=False))
        print(g3b.round(3).to_string(index=False))

    hdr("4. RISK GEOMETRY — stop/entry/target on the right sides, reward-to-risk sane")
    g4 = (shapes.groupby("pattern")
            .agg(n=("rr", "size"), sides_ok=("risk_sides_ok", lambda s: round(100 * s.mean(), 1)),
                 rr_med=("rr", "median"), rr_max=("rr", "max"),
                 rr_ok=("rr_ok", lambda s: round(100 * s.mean(), 1)))
            .reset_index().sort_values("sides_ok"))
    print(g4.round(2).to_string(index=False))

    hdr("5. OVERLAP — same-name shapes covering the same price action (de-dup should prevent this)")
    ov = audit_overlap(shape_rows)
    if len(ov):
        tot = ov.groupby("pattern").agg(n=("n", "sum"), overlapping_pairs=("overlapping_pairs", "sum"),
                                        worst_frac=("worst_frac", "max")).reset_index()
        print(tot.sort_values("overlapping_pairs", ascending=False).to_string(index=False))

    if not skip_causality and len(caus):
        hdr(f"6. CAUSALITY ON REAL DATA — {TRUNC_POINTS} truncation points x {len(keys)} windows\n"
            "   running the detector on the first k bars must reproduce the full run's first k")
        g6 = (caus.groupby("pattern")
                .agg(checks=("mismatches", "size"), total_mismatches=("mismatches", "sum"),
                     fires=("fires_full", "sum")).reset_index()
                .sort_values("total_mismatches", ascending=False))
        print(g6.to_string(index=False))
        tot = int(caus.mismatches.sum())
        print(f"\n   TOTAL MISMATCHES: {tot}   -> {'CAUSAL' if tot == 0 else 'LOOKAHEAD PRESENT'}")

    hdr("6b. DRAWN DOTS — does every touch marker sit on the thing it claims to touch?")
    dots = pd.DataFrame(dot_rows)
    if len(dots):
        g6b = (dots.groupby("pattern")
                   .agg(shapes=("dots", "size"), dots=("dots", "sum"),
                        off_line=("dots_off_line", "sum"),
                        wrong_side=("dots_wrong_side", "sum"),
                        worst_level_spread_atr=("level_spread_atr",
                            lambda v: float(np.nanmax(v)) if v.notna().any() else np.nan))
                   .reset_index().sort_values(["wrong_side", "off_line"], ascending=False))
        print(g6b.round(2).to_string(index=False))
        print("   trendline shapes: a dot must lie on one of the two fitted boundaries.")
        print("   level shapes: a TOP's dots must be highs and a BOTTOM's must be lows.")
        print(f"   dots off their line: {int(dots.dots_off_line.sum())} | "
              f"wrong side of the candle: {int(dots.dots_wrong_side.sum())} | "
              f"of {int(dots.dots.sum())} drawn")

    hdr("7. TRENDLINES")
    print(tl.to_string(index=False))
    print("\n   invariants that must be zero: "
          f"lines far from price {int(tl.lines_far_from_price.sum())}, "
          f"touches outside the drawn span {int(tl.touches_outside_span.sum())}, "
          f"off-chart endpoints {int(tl.off_chart_endpoints.sum())}, "
          f"touch-count mismatches {int(tl.touch_count_mismatch.sum())}, "
          f"breaks on the wrong side {int(tl.breaks_on_wrong_side.sum())}, "
          f"both-ways-same-bar {int(tl.both_ways_same_bar.sum())}")

    hdr("VERDICT")
    fails = []
    if len(bad):
        fails.append(f"{len(bad)} definitional-contract violations")
    if len(tb) and (tb.containment < CONTAIN_MIN).mean() > 0.25:
        fails.append(f"{100*(tb.containment < CONTAIN_MIN).mean():.0f}% of shapes below "
                     f"{CONTAIN_MIN} containment")
    if not skip_causality and len(caus) and caus.mismatches.sum():
        fails.append(f"{int(caus.mismatches.sum())} causality mismatches")
    if not shapes.risk_sides_ok.all():
        fails.append(f"{int((~shapes.risk_sides_ok).sum())} shapes with stop/target on the wrong side")
    # `rr_ok` was computed by check 4 from the first run but never consulted here, so extreme
    # reward-to-risk geometry could not fail the audit. Found by reading the audit's own output.
    if not shapes.rr_ok.all():
        n_bad = int((~shapes.rr_ok).sum())
        fails.append(f"{n_bad} shapes with reward-to-risk outside {RR_BOUNDS} "
                     f"(worst {shapes.rr.max():.1f})")
    # A converging shape drawn far past its last pivot stops describing its price action: for the
    # triangles, containment below 0.80 goes from 0% on short draws to 32.7% on long ones.
    tri = shapes[shapes.pattern.isin(("ascending_triangle", "descending_triangle",
                                      "symmetrical_triangle"))]
    if len(tri) and (tri.containment < CONTAIN_MIN).mean() > 0.10:
        fails.append(f"{100*(tri.containment < CONTAIN_MIN).mean():.0f}% of triangles below "
                     f"{CONTAIN_MIN} containment (extrapolation past the last pivot)")
    if len(dots):
        if int(dots.dots_off_line.sum()):
            fails.append(f"{int(dots.dots_off_line.sum())} drawn dots do not sit on their boundary")
        if int(dots.dots_wrong_side.sum()):
            fails.append(f"{int(dots.dots_wrong_side.sum())} drawn dots on the wrong side of the "
                         f"candle (a top marking a low, or vice versa)")
    for c in ("off_chart_endpoints", "touch_count_mismatch", "breaks_on_wrong_side",
              "both_ways_same_bar", "lines_far_from_price", "touches_outside_span"):
        if int(tl[c].sum()):
            fails.append(f"trendlines: {int(tl[c].sum())} {c}")
    print("  " + ("\n  ".join(f"FAIL: {f}" for f in fails) if fails else "PASS — all checks clean"))
    print(f"\n  {len(shapes)} shapes audited over {len(keys)} windows in {time.time()-t0:.0f}s")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
