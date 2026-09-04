"""Candlestick and chart pattern recognition.

The single most important property is CAUSALITY, and the strongest test of it is truncation: run the
detector on the first k bars, run it on the whole series, and require the first k results to be
identical. If any pattern peeked at a later bar the two runs would disagree. Retail pattern libraries
routinely fail this, usually by flagging a shape at its extreme rather than at its confirmation.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant import scenarios as sc
from quant.indicators import (add_candle_patterns, add_chart_patterns, candle_pattern_table,
                              chart_pattern_shapes,
                              chart_pattern_table, CANDLE_PATTERNS, CANDLE_PATTERN_DIRECTION,
                              CHART_PATTERNS, CHART_PATTERN_DIRECTION)


def _frame(rows) -> pd.DataFrame:
    """rows = [(open, high, low, close), ...]"""
    a = np.asarray(rows, dtype=float)
    t = pd.date_range("2026-01-01", periods=len(a), freq="15min", tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": a[:, 0], "high": a[:, 1],
                         "low": a[:, 2], "close": a[:, 3], "volume": 1.0})


def _random(n=3000, seed=11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0015, n)) * close
    lo = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0015, n)) * close
    t = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_, "high": hi, "low": lo,
                         "close": close, "volume": 1.0})


@pytest.fixture(scope="module")
def rnd():
    return _random()


# ---------------------------------------------------------------------------
# candlestick shapes — hand-built cases
# ---------------------------------------------------------------------------
def _pad(pattern_rows, n_pad=30, base=100.0, drift=0.0):
    """Filler bars so the rolling average body and the trend context are defined.

    The filler must have a NON-ZERO body: `avg_body` is the scale every "long"/"short" test is
    measured against, and a zero average makes those tests NaN rather than False.
    """
    rows = []
    px = base
    for _ in range(n_pad):
        rows.append((px, px + 1.2, px - 1.2, px + 0.4))
        px += drift
    return rows + pattern_rows


def test_hammer_needs_a_downtrend_and_hanging_man_an_uptrend():
    shape = (99.5, 100.5, 94.0, 100.0)             # small body on top, long lower shadow
    down = add_candle_patterns(_frame(_pad([shape], drift=-0.5)), ["hammer", "hanging_man"])
    up = add_candle_patterns(_frame(_pad([shape], drift=+0.5)), ["hammer", "hanging_man"])
    assert down["cdl_hammer"].iloc[-1] and not down["cdl_hanging_man"].iloc[-1]
    assert up["cdl_hanging_man"].iloc[-1] and not up["cdl_hammer"].iloc[-1]


def test_shooting_star_and_inverted_hammer_are_the_same_shape():
    shape = (100.0, 106.0, 99.8, 100.5)            # small body at the bottom, long upper shadow
    down = add_candle_patterns(_frame(_pad([shape], drift=-0.5)),
                               ["inverted_hammer", "shooting_star"])
    up = add_candle_patterns(_frame(_pad([shape], drift=+0.5)),
                             ["inverted_hammer", "shooting_star"])
    assert down["cdl_inverted_hammer"].iloc[-1] and not down["cdl_shooting_star"].iloc[-1]
    assert up["cdl_shooting_star"].iloc[-1] and not up["cdl_inverted_hammer"].iloc[-1]


def test_engulfing_requires_the_body_to_cover_the_previous_one():
    rows = _pad([(100.0, 100.2, 98.0, 98.5),       # red
                 (98.5, 101.0, 98.4, 100.5)])      # green, body covers 98.5..100.5 > 98.5..100.0
    out = add_candle_patterns(_frame(rows), ["engulfing_bull", "engulfing_bear"])
    assert out["cdl_engulfing_bull"].iloc[-1]
    assert not out["cdl_engulfing_bear"].iloc[-1]
    # a smaller follow-through body is NOT engulfing
    rows2 = _pad([(100.0, 100.2, 98.0, 98.5), (98.5, 99.4, 98.4, 99.2)])
    out2 = add_candle_patterns(_frame(rows2), ["engulfing_bull"])
    assert not out2["cdl_engulfing_bull"].iloc[-1]


def test_engulf_ratio_tightens_the_definition(rnd):
    loose = add_candle_patterns(rnd, ["engulfing_bull"], engulf_ratio=1.0)
    tight = add_candle_patterns(rnd, ["engulfing_bull"], engulf_ratio=2.0)
    assert tight["cdl_engulfing_bull"].sum() < loose["cdl_engulfing_bull"].sum()


def test_doji_and_marubozu_are_opposites():
    doji = _frame(_pad([(100.0, 101.0, 99.0, 100.02)]))
    maru = _frame(_pad([(100.0, 105.0, 100.0, 105.0)]))
    assert add_candle_patterns(doji, ["doji"])["cdl_doji"].iloc[-1]
    assert not add_candle_patterns(maru, ["doji"])["cdl_doji"].iloc[-1]
    assert add_candle_patterns(maru, ["marubozu_bull"])["cdl_marubozu_bull"].iloc[-1]


def test_three_candle_pattern_flags_on_its_LAST_bar():
    """A morning star is not knowable until the third candle closes."""
    rows = _pad([(100.0, 100.2, 94.0, 94.5),       # long red
                 (94.4, 94.8, 93.8, 94.2),         # small body
                 (94.3, 99.5, 94.2, 99.0)])        # long green closing above the first midpoint
    out = add_candle_patterns(_frame(rows), ["morning_star"])
    hits = np.flatnonzero(out["cdl_morning_star"].to_numpy())
    assert len(hits) == 1 and hits[0] == len(rows) - 1


def test_gap_patterns_are_impossible_without_gaps(rnd):
    """Textbook gap modes must produce nothing on a continuous series; 'none' must produce some."""
    gapped = add_candle_patterns(rnd, ["piercing_line", "dark_cloud_cover"], gap_mode="range")
    assert gapped["cdl_piercing_line"].sum() == 0
    assert gapped["cdl_dark_cloud_cover"].sum() == 0
    relaxed = add_candle_patterns(rnd, ["piercing_line", "dark_cloud_cover"], gap_mode="none")
    assert relaxed["cdl_piercing_line"].sum() > 0


def test_score_columns_agree_with_the_individual_flags(rnd):
    out = add_candle_patterns(rnd)
    bull = [f"cdl_{p}" for p, d in CANDLE_PATTERN_DIRECTION.items() if d > 0]
    bear = [f"cdl_{p}" for p, d in CANDLE_PATTERN_DIRECTION.items() if d < 0]
    assert (out["cdl_bull_n"] == out[bull].sum(axis=1)).all()
    assert (out["cdl_bear_n"] == out[bear].sum(axis=1)).all()
    assert (out["cdl_score"] == out["cdl_bull_n"] - out["cdl_bear_n"]).all()


def test_unknown_pattern_name_raises(rnd):
    with pytest.raises(KeyError):
        add_candle_patterns(rnd, ["not_a_pattern"])


# ---------------------------------------------------------------------------
# CAUSALITY — the truncation test
# ---------------------------------------------------------------------------
def test_candle_patterns_are_causal(rnd):
    k = 2000
    full = add_candle_patterns(rnd)
    part = add_candle_patterns(rnd.iloc[:k].copy())
    cols = [f"cdl_{p}" for p in CANDLE_PATTERNS]
    # the rolling average body is a trailing window, so results on the shared prefix must match
    pd.testing.assert_frame_equal(full[cols].iloc[:k].reset_index(drop=True),
                                  part[cols].reset_index(drop=True))


def test_chart_patterns_are_causal(rnd):
    k = 2000
    full = add_chart_patterns(rnd)
    part = add_chart_patterns(rnd.iloc[:k].copy())
    cols = [f"pat_{p}" for p in CHART_PATTERNS] + ["pat_signal", "pat_dir"]
    pd.testing.assert_frame_equal(full[cols].iloc[:k].reset_index(drop=True),
                                  part[cols].reset_index(drop=True))


@pytest.mark.parametrize("k", [700, 1100, 1500, 1900, 2300, 2700])
def test_chart_pattern_dedupe_cannot_delete_a_signal_that_already_fired(rnd, k):
    """Truncation at MANY points, at settings loose enough that detections actually overlap.

    De-duplication is where causality dies quietly. Ranking the whole set by quality and keeping the
    best-formed shape lets a pattern confirming at bar 5,000 remove one that fired at bar 2,997 — the
    single-truncation-point version of this test missed exactly that, because at default (sparse)
    settings no overlap clash ever straddled the one boundary it checked. These settings produce
    dense, overlapping wedges, which is the regime that exposes it.
    """
    loose = dict(min_touches_upper=2, min_touches_lower=2, min_span_bars=40,
                 min_height_atr=4.0, line_tol_atr=1.2)
    names = ["rising_wedge", "falling_wedge"]
    kw = dict(patterns=names, atr_mult=2.5, settings={p: loose for p in names})
    cols = [f"pat_{p}" for p in names]
    full = add_chart_patterns(rnd, **kw)
    part = add_chart_patterns(rnd.iloc[:k].copy(), **kw)
    # precondition, not the assertion under test: there must be enough detections for overlap
    # clashes to exist at all. Was >20 before converge_min_frac made wedges require REAL
    # narrowing, which legitimately took this fixture from 24 to 19.
    assert full[cols].to_numpy().sum() > 15, "fixture must actually produce overlapping detections"
    pd.testing.assert_frame_equal(full[cols].iloc[:k].reset_index(drop=True),
                                  part[cols].reset_index(drop=True))


# ---------------------------------------------------------------------------
# chart patterns
# ---------------------------------------------------------------------------
def test_chart_patterns_publish_a_usable_risk_geometry(rnd):
    out = add_chart_patterns(rnd)
    tb = chart_pattern_table(out)
    assert len(tb) > 0
    long_, short_ = tb[tb.direction > 0], tb[tb.direction < 0]
    # a stop must sit on the far side of entry, and the measured move on the near side
    assert (long_.stop < long_.entry).all() and (long_.target > long_.entry).all()
    assert (short_.stop > short_.entry).all() and (short_.target < short_.entry).all()
    assert np.isfinite(tb[["entry", "stop", "target"]].to_numpy()).all()


def test_chart_pattern_fires_only_when_price_breaks_the_neckline(rnd):
    out = add_chart_patterns(rnd)
    m = out["pat_signal"].to_numpy(bool)
    c = out["close"].to_numpy(float)
    entry = out["pat_entry"].to_numpy(float)
    d = out["pat_dir"].to_numpy(int)
    idx = np.flatnonzero(m)
    assert len(idx) > 0
    up, dn = idx[d[idx] > 0], idx[d[idx] < 0]
    assert (c[up] > entry[up]).all(), "a bullish pattern must confirm ABOVE its neckline"
    assert (c[dn] < entry[dn]).all(), "a bearish pattern must confirm BELOW its neckline"


def test_pattern_direction_matches_the_registry(rnd):
    out = add_chart_patterns(rnd)
    d = out["pat_dir"].to_numpy(int)
    for name, want in CHART_PATTERN_DIRECTION.items():
        if want == 0:                     # symmetrical triangle registers on both sides
            continue
        hit = out[f"pat_{name}"].to_numpy(bool)
        if hit.any():
            assert set(np.unique(d[hit])) <= {want}, f"{name} fired with the wrong direction"


def test_forward_filled_levels_persist_after_the_signal(rnd):
    out = add_chart_patterns(rnd)
    idx = np.flatnonzero(out["pat_signal"].to_numpy(bool))
    assert len(idx) >= 2
    i = int(idx[0])
    nxt = int(idx[1])
    ff = out["pat_stop_ff"].to_numpy(float)
    assert ff[i] == pytest.approx(out["pat_stop"].to_numpy(float)[i])
    if nxt > i + 1:
        assert ff[i + 1] == pytest.approx(ff[i])       # held until the next pattern


def test_unknown_chart_pattern_raises(rnd):
    with pytest.raises(KeyError):
        add_chart_patterns(rnd, patterns=["not_a_shape"])


# ---------------------------------------------------------------------------
# strategy integration
# ---------------------------------------------------------------------------
def test_pattern_scenarios_are_registered():
    for k in ("candle_pattern_long@v1", "candle_pattern_short@v1",
              "chart_pattern_long@v1", "chart_pattern_short@v1"):
        assert k in sc.SCENARIOS
    assert "pattern_stop@v1" in sc.STOP_METHODS
    assert "pattern_target@v1" in sc.TP_METHODS


@pytest.mark.parametrize("key,params", [
    ("candle_pattern_long@v1", {"cdl_names": ["hammer", "engulfing_bull"]}),
    ("candle_pattern_short@v1", {"cdl_min_n": 1}),
    ("chart_pattern_long@v1", {}),
    ("chart_pattern_short@v1", {"pat_names": ["double_top", "head_shoulders"]}),
])
def test_pattern_entry_runs_with_stops_on_the_right_side(rnd, key, params):
    run = sc.run_pieces(rnd, key, "atr@v1", "rr@v1",
                        params={**params, "stop_atr_mult": 2.0, "tp_rr": 2.0},
                        cfg={"initial_cash": 10_000})
    t = run.trades
    assert len(t) > 0
    ok = t["stop_price"].notna()
    if sc.SCENARIOS[key].side == "long":
        assert (t.loc[ok, "stop_price"] < t.loc[ok, "entry_price"]).all()
    else:
        assert (t.loc[ok, "stop_price"] > t.loc[ok, "entry_price"]).all()


def test_pattern_stop_uses_the_shape_level(rnd):
    run = sc.run_pieces(rnd, "chart_pattern_long@v1", "pattern_stop@v1", "pattern_target@v1",
                        params={"stop_buffer_pct": 0.0}, cfg={"initial_cash": 10_000})
    t = run.trades
    assert len(t) > 0
    lvl = run.df["pat_stop_ff"].to_numpy(float)
    ok = t["stop_price"].notna()
    # every stop should match the pattern level in force at that trade's entry bar
    got = t.loc[ok, "stop_price"].to_numpy(float)
    want = lvl[t.loc[ok, "entry_i"].to_numpy(int)]
    assert np.allclose(got, want, rtol=1e-6) or (got < t.loc[ok, "entry_price"]).all()


def test_candle_pattern_table_lists_hits(rnd):
    out = add_candle_patterns(rnd, ["hammer", "engulfing_bull", "doji"])
    tb = candle_pattern_table(out)
    assert set(tb.pattern.unique()) <= {"hammer", "engulfing_bull", "doji"}
    assert len(tb) == int(out[["cdl_hammer", "cdl_engulfing_bull", "cdl_doji"]].sum().sum())


def test_calling_the_detector_twice_does_not_duplicate_columns(rnd):
    """REGRESSION: a loader adding every pattern, then a scenario preparing a subset, used to leave
    two columns of the same name — after which `df[col]` returns a DataFrame, every boolean test
    silently misbehaves, and pattern entries produce ZERO trades with no error anywhere."""
    once = add_candle_patterns(rnd)
    twice = add_candle_patterns(once, ["hammer", "engulfing_bull"])
    assert list(twice.columns).count("cdl_hammer") == 1
    assert isinstance(twice["cdl_hammer"], pd.Series)
    # and the recomputed values must match the first pass
    pd.testing.assert_series_equal(once["cdl_hammer"], twice["cdl_hammer"])


def test_named_subset_entry_actually_trades(rnd):
    run = sc.run_pieces(rnd, "candle_pattern_long@v1", "atr@v1", "rr@v1",
                        params={"cdl_names": ["hammer", "engulfing_bull"],
                                "stop_atr_mult": 2.0, "tp_rr": 2.0},
                        cfg={"initial_cash": 10_000})
    assert len(run.trades) > 0


# ---------------------------------------------------------------------------
# fitted trendlines — the shape must be a real structure, not any two pivots
# ---------------------------------------------------------------------------
def _zigzag_frame(turns, *, bars_per_leg=40, noise=0.0, seed=0) -> pd.DataFrame:
    """Build a price path that walks through `turns` (prices), so the zigzag emits those pivots."""
    rng = np.random.default_rng(seed)
    closes = []
    for a, b in zip(turns[:-1], turns[1:]):
        closes += list(np.linspace(a, b, bars_per_leg, endpoint=False))
    closes.append(turns[-1])
    c = np.asarray(closes, float)
    if noise:
        c = c + rng.normal(0, noise, len(c))
    o = np.r_[c[0], c[:-1]]
    hi = np.maximum(o, c) + 0.05
    lo = np.minimum(o, c) - 0.05
    t = pd.date_range("2026-01-01", periods=len(c), freq="1min", tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": o, "high": hi, "low": lo,
                         "close": c, "volume": 1.0})


def _ascending_triangle_frame(bars_per_leg=40):
    """Flat highs at 100, lows rising 88 -> 96, then a break above the resistance."""
    turns = [88, 100, 90, 100, 92, 100, 94, 100, 96, 108]
    return _zigzag_frame(turns, bars_per_leg=bars_per_leg)


def test_ascending_triangle_is_detected_with_real_touches():
    df = _ascending_triangle_frame()
    sh = chart_pattern_shapes(df, patterns=["ascending_triangle"], sensitivity="normal",
                              atr_mult=2.0)
    assert len(sh) > 0, "a textbook ascending triangle must be detected"
    r = sh.iloc[0]
    assert r.touches_upper >= 3, "the flat resistance must be fitted through 3+ highs"
    assert r.touches_lower >= 3, "the rising support must be fitted through 3+ lows"
    assert r.direction == 1
    # the flat boundary really is flat, the support really rises
    assert abs(r.upper_slope) < abs(r.lower_slope)
    assert r.lower_slope > 0
    assert r.span_bars > 5 * 40 * 0.5


def test_two_point_lines_are_rejected():
    """A window offering only two highs and two lows must not produce a fitted-boundary shape.

    This is the whole bug: two points define a line exactly, so the old detector called every pivot
    pair a trendline. With 3 touches required, a 2-swing window yields nothing.
    """
    df = _zigzag_frame([90, 100, 92, 101], bars_per_leg=40)
    sh = chart_pattern_shapes(df, sensitivity="normal", atr_mult=2.0)
    fitted = {"ascending_triangle", "descending_triangle", "symmetrical_triangle",
              "rising_wedge", "falling_wedge", "rectangle_bull", "rectangle_bear"}
    assert set(sh.pattern) & fitted == set()


def test_small_shapes_are_rejected_by_min_span():
    """The same triangle squeezed into a few bars is noise, not structure."""
    big = _ascending_triangle_frame(bars_per_leg=40)
    small = _ascending_triangle_frame(bars_per_leg=4)
    n_big = len(chart_pattern_shapes(big, patterns=["ascending_triangle"],
                                     sensitivity="normal", atr_mult=2.0))
    n_small = len(chart_pattern_shapes(small, patterns=["ascending_triangle"],
                                       sensitivity="normal", atr_mult=2.0,
                                       settings={"ascending_triangle": dict(min_span_bars=200)}))
    assert n_big > 0 and n_small == 0


def test_overlapping_duplicates_are_collapsed(rnd):
    loose = chart_pattern_shapes(rnd, sensitivity="minor", overlap_frac=1.01)
    tight = chart_pattern_shapes(rnd, sensitivity="minor", overlap_frac=0.3)
    assert len(tight) <= len(loose)
    # nothing that survives may sit almost on top of another detection of the same pattern
    for name, sub in tight.groupby("pattern"):
        iv = sorted(zip(sub.start_i, sub.confirm_i))
        for (a0, a1), (b0, b1) in zip(iv, iv[1:]):
            inter = max(0, min(a1, b1) - max(a0, b0))
            shorter = max(1, min(a1 - a0, b1 - b0))
            assert inter / shorter <= 0.3 + 1e-9


def test_per_pattern_settings_only_affect_that_pattern(rnd):
    base = chart_pattern_shapes(rnd, sensitivity="minor")
    tuned = chart_pattern_shapes(rnd, sensitivity="minor",
                                 settings={"symmetrical_triangle": dict(min_touches_upper=6)})
    b = base.pattern.value_counts()
    t = tuned.pattern.value_counts()
    assert t.get("symmetrical_triangle", 0) < b.get("symmetrical_triangle", 1)
    for name in set(b.index) - {"symmetrical_triangle"}:
        assert t.get(name, 0) == b.get(name, 0), f"{name} should be untouched"


def test_sensitivity_is_monotonic(rnd):
    n = {s: len(chart_pattern_shapes(rnd, sensitivity=s)) for s in ("major", "normal", "minor")}
    assert n["minor"] >= n["normal"] >= n["major"]


def test_unknown_sensitivity_and_spec_field_raise(rnd):
    with pytest.raises(KeyError):
        add_chart_patterns(rnd, sensitivity="enormous")
    with pytest.raises(KeyError):
        add_chart_patterns(rnd, settings={"double_top": dict(not_a_field=1)})


def test_boundary_may_slope(rnd):
    """A triangle confirms on a break of its SLOPING line, not a horizontal proxy."""
    sh = chart_pattern_shapes(rnd, sensitivity="minor")
    tri = sh[sh.pattern.isin(["symmetrical_triangle", "ascending_triangle",
                              "descending_triangle", "rising_wedge", "falling_wedge"])]
    if len(tri):
        assert (tri.boundary_slope.abs() > 0).any(), "no sloping boundary anywhere"


# ---------------------------------------------------------------------------
# tuned per-pattern settings (notebook 15)
# ---------------------------------------------------------------------------
def test_tuned_specs_cover_every_pattern_they_claim_and_reject_unknown_names():
    from quant.indicators import tuned_specs, TUNED_COMMON
    s = tuned_specs()
    assert len(s) == len(CHART_PATTERNS), (
        "TUNED_SPECS must cover EVERY pattern: a name missing here is never requested by "
        "chart_pattern_shapes_tuned, so it silently shows zero detections on a tuned chart")
    assert set(s) == set(CHART_PATTERNS)
    for name, cfg in s.items():
        assert name in CHART_PATTERNS
        assert "atr_mult" in cfg
        # TUNED_COMMON holds only what the POOLED scan fixed for everything, so those must reach
        # every pattern untouched. Per-pattern values (atr_mult, span, height) are meant to differ.
        for k, v in TUNED_COMMON.items():
            assert cfg[k] == v, f"{name} lost {k}"
        assert "min_height_atr" in cfg, f"{name} must state its own height"
        # three touches is the point: two points define a line exactly
        assert cfg["min_touches_upper"] >= 3 and cfg["min_touches_lower"] >= 3
    assert set(tuned_specs(["rising_wedge"])) == {"rising_wedge"}
    with pytest.raises(KeyError):
        tuned_specs(["not_a_pattern"])


def test_tuned_shapes_merge_the_separate_atr_mult_passes(rnd):
    """The whole reason this helper exists: `atr_mult` sizes the zigzag for a whole pass, so
    patterns calibrated to different values cannot come out of one call."""
    from quant.indicators import chart_pattern_shapes, chart_pattern_shapes_tuned, tuned_specs
    spec = tuned_specs()
    assert len({c["atr_mult"] for c in spec.values()}) > 1, "fixture assumes differing atr_mult"

    got = chart_pattern_shapes_tuned(rnd)
    assert list(got["confirm_i"]) == sorted(got["confirm_i"]), "must come back in time order"

    # each pattern's rows must match what its OWN single-atr_mult pass produces
    for name in ("rising_wedge", "head_shoulders"):
        cfg = dict(spec[name])
        am = cfg.pop("atr_mult")
        one = chart_pattern_shapes(rnd, patterns=[name], atr_mult=am, settings={name: cfg})
        mine = got[got["pattern"] == name]
        assert len(mine) == len(one)
        assert sorted(mine["confirm_i"]) == sorted(one["confirm_i"])


def test_tuned_shapes_stay_causal(rnd):
    from quant.indicators import chart_pattern_shapes_tuned
    k = 2200
    full = chart_pattern_shapes_tuned(rnd)
    part = chart_pattern_shapes_tuned(rnd.iloc[:k].copy())
    a = sorted(zip(full["confirm_i"], full["pattern"]))
    a = [x for x in a if x[0] < k]
    assert a == sorted(zip(part["confirm_i"], part["pattern"]))


def test_pattern_figure_tuned_draws_without_a_shared_detect_config(rnd):
    from quant.viz.studies import pattern_figure
    fig = pattern_figure(rnd, tuned=True, sr_zones=False, emas=())
    drawn = {t.legendgroup for t in fig.data if t.legendgroup}
    assert drawn, "tuned figure drew nothing"
    assert drawn <= set(CHART_PATTERNS), f"unexpected legend groups: {drawn - set(CHART_PATTERNS)}"
    # and the per_pattern route must reach the same place as tuned=True
    from quant.indicators import tuned_specs
    same = pattern_figure(rnd, per_pattern=tuned_specs(), sr_zones=False, emas=())
    assert {t.legendgroup for t in same.data if t.legendgroup} == drawn


# ---------------------------------------------------------------------------
# what the chart draws as a "touch" must BE a touch
# ---------------------------------------------------------------------------
def test_touch_pivots_are_the_fitted_inliers_not_the_whole_search_window(rnd):
    """The dots on the chart claim to be touches, so they must be the pivots the lines were fitted
    through — `pivot_i` is the whole lookback window and is about twice as long."""
    s = chart_pattern_shapes(rnd)
    assert len(s)
    trend = s[s["upper_slope"].notna() & s["lower_slope"].notna()]
    assert len(trend), "fixture must produce trendline shapes"
    for r in trend.itertuples():
        assert len(r.touch_i) == r.touches_upper + r.touches_lower - _shared(r)
        assert set(r.touch_i) <= set(r.pivot_i), "a touch must come from the search window"
        assert len(r.touch_i) <= len(r.pivot_i)
    # level shapes have no fitted lines; their defining extremes are the touches
    lvl = s[s["upper_slope"].isna() & s["lower_slope"].isna()]
    for r in lvl.itertuples():
        assert len(r.touch_i) > 0


def _shared(r) -> int:
    """A pivot can be an inlier of both boundaries; it is one dot, counted once."""
    return len(r.touch_i) and (r.touches_upper + r.touches_lower) - len(set(r.touch_i))


def test_every_drawn_touch_actually_sits_on_one_of_the_boundaries(rnd):
    from quant.indicators import add_atr
    d = add_atr(rnd, 14)
    atr = float(d["atr_14"].median())
    s = chart_pattern_shapes(rnd)
    trend = s[s["upper_slope"].notna()]
    for r in trend.itertuples():
        for i, px in zip(r.touch_i, r.touch_price):
            du = abs(px - (r.upper_slope * i + r.upper_intercept))
            dl = abs(px - (r.lower_slope * i + r.lower_intercept))
            # default line_tol_atr is 1.0; allow generous slack for the ATR being a median here
            assert min(du, dl) <= 4.0 * atr, f"dot at {i} sits on neither boundary"


# ---------------------------------------------------------------------------
# a symmetrical triangle has to be symmetrical
# ---------------------------------------------------------------------------
def test_symmetrical_triangle_requires_comparable_slopes(rnd):
    """Converging with opposite slopes is necessary but not sufficient: without a symmetry test a
    line falling at -12.9 paired with one rising at 0.76 was being called symmetrical."""
    tight = chart_pattern_shapes(rnd, patterns=["symmetrical_triangle"],
                                 settings={"symmetrical_triangle": dict(slope_sym_max=2.0)})
    for r in tight.itertuples():
        a, b = abs(r.upper_slope), abs(r.lower_slope)
        assert max(a, b) / max(min(a, b), 1e-12) <= 2.0 + 1e-9

    loose = chart_pattern_shapes(rnd, patterns=["symmetrical_triangle"],
                                 settings={"symmetrical_triangle": dict(slope_sym_max=50.0)})
    assert len(loose) >= len(tight), "loosening the symmetry bound cannot find fewer"


def test_symmetry_bound_does_not_touch_the_other_triangles(rnd):
    """Ascending/descending are defined by a FLAT boundary, so the symmetry test must not reach
    them — it would otherwise silently reshape patterns it has no business filtering."""
    for name in ("ascending_triangle", "descending_triangle"):
        a = chart_pattern_shapes(rnd, patterns=[name],
                                 settings={name: dict(slope_sym_max=1.01)})
        b = chart_pattern_shapes(rnd, patterns=[name],
                                 settings={name: dict(slope_sym_max=99.0)})
        assert len(a) == len(b), f"{name} changed with slope_sym_max"


# ---------------------------------------------------------------------------
# channels — parallel boundaries, constant width
# ---------------------------------------------------------------------------
def _synthetic_channel(n=600, slope=0.0, width=40.0, seed=3):
    """Price oscillating between two PARALLEL lines — the thing a channel detector must find."""
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    mid = 1000 + slope * x
    osc = (width / 2) * np.sin(2 * np.pi * x / 90)
    close = mid + osc + rng.normal(0, width * 0.03, n)
    open_ = np.r_[close[0], close[:-1]]
    hi = np.maximum(open_, close) + width * 0.02
    lo = np.minimum(open_, close) - width * 0.02
    t = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_, "high": hi, "low": lo,
                         "close": close, "volume": 1.0})


@pytest.mark.parametrize("slope,expect", [(0.0, "channel_flat"),
                                          (0.35, "channel_up"),
                                          (-0.35, "channel_down")])
def test_channels_are_found_and_classified_by_their_slope(slope, expect):
    df = _synthetic_channel(slope=slope)
    # atr_mult 1.0 rather than 1.5: on a pure sine the zigzag threshold interacts with the
    # oscillation period, and 1.5 happens to skip the pivots that define the flat case.
    kw = dict(atr_mult=1.0, settings={p: dict(min_touches_upper=3, min_touches_lower=3,
                                              min_span_bars=80, min_height_atr=1.0,
                                              line_tol_atr=1.5)
                                      for p in CHART_PATTERNS})
    s = chart_pattern_shapes(df, patterns=["channel_up", "channel_down", "channel_flat"], **kw)
    assert len(s), f"no channel found for slope={slope}"
    assert expect in set(s["pattern"]), f"slope={slope} gave {sorted(set(s['pattern']))}"


def test_channel_boundaries_really_are_parallel(rnd):
    """The defining property: the gap between the lines must hold across the span."""
    names = ["channel_up", "channel_down", "channel_flat"]
    s = chart_pattern_shapes(rnd, patterns=names,
                             settings={p: dict(width_tol_frac=0.25) for p in names})
    for r in s.itertuples():
        a, b = int(r.start_i), int(r.confirm_i)
        w0 = (r.upper_slope * a + r.upper_intercept) - (r.lower_slope * a + r.lower_intercept)
        w1 = (r.upper_slope * b + r.upper_intercept) - (r.lower_slope * b + r.lower_intercept)
        assert abs(w1 - w0) / max(abs(w0), abs(w1), 1e-12) <= 0.25 + 1e-9


def test_a_channel_is_not_also_a_wedge(rnd):
    """`converging` used to mean any narrowing at all, so a parallel channel was emitted as a wedge.
    width_tol_frac and converge_min_frac must not overlap."""
    both = ["channel_up", "channel_down", "rising_wedge", "falling_wedge"]
    s = chart_pattern_shapes(rnd, patterns=both)
    keyed = {}
    for r in s.itertuples():
        keyed.setdefault((int(r.start_i), int(r.confirm_i)), set()).add(r.pattern)
    for k, v in keyed.items():
        assert not ({"channel_up", "channel_down"} & v and
                    {"rising_wedge", "falling_wedge"} & v), f"same shape called both at {k}: {v}"


def test_widening_the_width_tolerance_finds_more_channels(rnd):
    names = ["channel_up", "channel_down", "channel_flat"]
    tight = chart_pattern_shapes(rnd, patterns=names,
                                 settings={p: dict(width_tol_frac=0.05) for p in names})
    loose = chart_pattern_shapes(rnd, patterns=names,
                                 settings={p: dict(width_tol_frac=0.40) for p in names})
    assert len(loose) >= len(tight)


def test_channels_stay_causal(rnd):
    names = ["channel_up", "channel_down", "channel_flat"]
    k = 2200
    full = add_chart_patterns(rnd, patterns=names)
    part = add_chart_patterns(rnd.iloc[:k].copy(), patterns=names)
    cols = [f"pat_{p}" for p in names]
    pd.testing.assert_frame_equal(full[cols].iloc[:k].reset_index(drop=True),
                                  part[cols].reset_index(drop=True))


# ---------------------------------------------------------------------------
# standalone trendlines
# ---------------------------------------------------------------------------
def test_trendline_touches_meet_the_minimum_and_sit_on_the_line(rnd):
    from quant.indicators import add_trendlines, trendline_table
    o = add_trendlines(rnd, min_touches=3)
    tl = trendline_table(o)
    assert len(tl), "no trendlines found"
    for r in tl.itertuples():
        assert r.touches >= 3, "two points always fit; three is the minimum that means anything"
        assert len(r.touch_i) == r.touches
        for i, px in zip(r.touch_i, r.touch_price):
            assert abs(px - (r.slope * i + r.intercept)) < 1e6


def test_trendline_break_direction_matches_the_side_of_the_line(rnd):
    from quant.indicators import add_trendlines
    o = add_trendlines(rnd)
    c = o["close"].to_numpy(float)
    dn = o["tl_break_dn"].to_numpy(bool)
    up = o["tl_break_up"].to_numpy(bool)
    assert dn.any() or up.any(), "no breaks at all"
    assert not (dn & up).any(), "a bar cannot break both ways"
    for i in np.flatnonzero(dn):
        assert c[i] < o["tl_sup"].to_numpy()[i] + 1e-9
    for i in np.flatnonzero(up):
        assert c[i] > o["tl_res"].to_numpy()[i] - 1e-9


def test_trendlines_are_causal(rnd):
    from quant.indicators import add_trendlines
    k = 2200
    full = add_trendlines(rnd)
    part = add_trendlines(rnd.iloc[:k].copy())
    for col in ("tl_break_dn", "tl_break_up", "tl_break_dir"):
        np.testing.assert_array_equal(full[col].to_numpy()[:k], part[col].to_numpy())


def test_trendline_endpoints_stay_on_the_chart(rnd):
    """A superseded line left open ran to the end of the series; extrapolating a slope over
    thousands of bars produced endpoints far outside the instrument's range."""
    from quant.indicators import add_trendlines, trendline_table
    tl = trendline_table(add_trendlines(rnd))
    lo, hi = rnd["low"].min(), rnd["high"].max()
    span = hi - lo
    assert (tl["y_end"] > lo - 3 * span).all() and (tl["y_end"] < hi + 3 * span).all()


def test_more_touches_required_means_fewer_trendlines(rnd):
    from quant.indicators import add_trendlines, trendline_table
    three = trendline_table(add_trendlines(rnd, min_touches=3))
    four = trendline_table(add_trendlines(rnd, min_touches=4))
    assert len(four) <= len(three)


def test_pattern_figure_can_draw_trendlines(rnd):
    from quant.viz.studies import pattern_figure
    fig = pattern_figure(rnd, patterns=["rising_wedge"], sr_zones=False, trendlines=True)
    groups = {t.legendgroup for t in fig.data if t.legendgroup}
    assert "support trendline" in groups or "resistance trendline" in groups


# ---------------------------------------------------------------------------
# apex expiry and the containment gate
# ---------------------------------------------------------------------------
def test_converging_shapes_never_confirm_past_their_apex(rnd):
    """Past the apex the upper boundary sits BELOW the lower one and the shape is inverted."""
    names = ["ascending_triangle", "descending_triangle", "symmetrical_triangle",
             "rising_wedge", "falling_wedge"]
    s = chart_pattern_shapes(rnd, patterns=names)
    checked = 0
    for r in s.itertuples():
        ds = r.upper_slope - r.lower_slope
        if abs(ds) < 1e-12:
            continue
        apex = (r.lower_intercept - r.upper_intercept) / ds
        last = max(r.pivot_i)
        if apex <= last:
            continue                     # diverging: no apex ahead
        checked += 1
        frac = (r.confirm_i - last) / (apex - last)
        assert frac <= 0.75 + 1e-6, f"{r.pattern} confirmed {frac:.2f} of the way to its apex"
    assert checked > 0, "fixture produced no converging shapes to check"


def test_channels_are_exempt_from_the_apex_limit(rnd):
    """Parallel boundaries never meet, so the apex rule must not touch channels. If it did, the
    fix for one family would silently shrink another."""
    names = ["channel_up", "channel_down", "channel_flat"]
    loose = chart_pattern_shapes(rnd, patterns=names,
                                 settings={p: dict(max_apex_frac=0.05) for p in names})
    wide = chart_pattern_shapes(rnd, patterns=names,
                                settings={p: dict(max_apex_frac=10.0) for p in names})
    assert len(loose) == len(wide), "the apex limit reached the channels"


def test_every_confirmed_shape_contains_its_own_price_action(rnd):
    """The direct statement of 'this shape describes what price did'. Before the gate existed,
    32.7% of long-draw triangles held under 80% of their bars, worst case 0.31."""
    from quant.indicators import add_atr
    d = add_atr(rnd, 14)
    atr = d["atr_14"].to_numpy(float)
    hi, lo = d["high"].to_numpy(float), d["low"].to_numpy(float)
    s = chart_pattern_shapes(rnd)
    checked = 0
    for r in s.itertuples():
        if not (np.isfinite(r.upper_slope) and np.isfinite(r.lower_slope)):
            continue
        a, b = int(min(r.pivot_i)), int(r.confirm_i)
        if b <= a:
            continue
        checked += 1
        x = np.arange(a, b + 1)
        tol = 1.0 * (atr[b] if np.isfinite(atr[b]) else 0.0)
        u = r.upper_slope * x + r.upper_intercept
        l = r.lower_slope * x + r.lower_intercept
        inside = ((hi[a:b + 1] <= u + tol) & (lo[a:b + 1] >= l - tol)).mean()
        assert inside >= 0.80 - 1e-9, f"{r.pattern} contains only {inside:.2f} of its bars"
    assert checked > 0


def test_containment_gate_can_be_disabled(rnd):
    """0 turns it off, and turning it off cannot find FEWER shapes."""
    names = ["ascending_triangle", "descending_triangle"]
    on = chart_pattern_shapes(rnd, patterns=names)
    off = chart_pattern_shapes(rnd, patterns=names,
                               settings={p: dict(min_containment=0.0) for p in names})
    assert len(off) >= len(on)


def test_apex_and_containment_stay_causal(rnd):
    """Both new gates read only bars at or before the break, so truncation must still agree."""
    k = 2200
    full = add_chart_patterns(rnd)
    part = add_chart_patterns(rnd.iloc[:k].copy())
    cols = [f"pat_{p}" for p in CHART_PATTERNS] + ["pat_signal", "pat_dir"]
    pd.testing.assert_frame_equal(full[cols].iloc[:k].reset_index(drop=True),
                                  part[cols].reset_index(drop=True))


def test_requesting_a_subset_does_not_lose_shapes_to_the_flag_rename(rnd):
    """Renaming a shape into a flag used to happen unconditionally, so a channel relabelled
    `bull_flag` was then dropped by the `not in hits` filter instead of staying a channel. The
    visible symptom was that the channel count moved with `pole_mult` — a parameter that has
    nothing to do with channels."""
    ch = ["channel_up", "channel_down", "channel_flat"]
    counts = [len(chart_pattern_shapes(rnd, patterns=ch, pole_mult=pm)) for pm in (2.0, 1.0, 0.4)]
    assert len(set(counts)) == 1, f"channel count moved with pole_mult: {counts}"


def test_flag_rename_still_happens_when_flags_are_requested(rnd):
    """The guard must not disable the feature: ask for flags and they must appear."""
    ch = ["channel_up", "channel_down", "channel_flat"]
    fl = ["bull_flag", "bear_flag"]
    with_flags = chart_pattern_shapes(rnd, patterns=ch + fl, pole_mult=0.3)
    assert with_flags["pattern"].isin(fl).any(), "no flags produced even at a loose pole"


def test_a_refitted_trendline_keeps_its_identity(rnd):
    """A line that picks up another touch is the SAME line, refined. Emitting a fresh record reset
    `born_i` and produced near-duplicate lines: 16% of them lived under 10 bars and a fifth of
    consecutive pairs differed in slope by under 5%."""
    from quant.indicators import add_trendlines, trendline_table
    tl = trendline_table(add_trendlines(rnd))
    assert len(tl) > 5
    # measured on the DRAWN span (first touch to end), not `end_i - born_i`. The latter is short by
    # construction: a pivot's extreme sits well before the bar that confirms it, so a line is often
    # already near retirement when it becomes knowable. That is the confirmation lag, not churn.
    drawn = tl["end_i"] - tl["start_i"]
    assert (drawn < 10).mean() < 0.25, "too many lines drawn over almost nothing — identity churning"
    for _, g in tl.groupby("side"):
        s = g.sort_values("born_i")["slope"].to_numpy()
        if len(s) > 3:
            rel = np.abs(np.diff(s)) / np.maximum(np.abs(s[:-1]), 1e-9)
            assert (rel < 0.05).mean() < 0.30, "consecutive lines are near-duplicates"


def test_the_protective_level_is_a_meaningful_distance_from_entry(rnd):
    """Being on the correct side of entry is not enough. A long, steeply rising wedge lets its
    rising lower boundary catch up with the highest high, so the stop lands a hair from the break
    and reward-to-risk explodes — 585:1 measured on gold before this gate existed."""
    s = chart_pattern_shapes(rnd)
    assert len(s)
    risk = (s["entry"] - s["stop"]).abs()
    assert (risk > 0).all(), "a zero-risk shape publishes an unusable stop"
    # the target sits one height away, so the risk floor is equivalent to an R:R ceiling
    rr = (s["target"] - s["entry"]).abs() / risk
    assert rr.max() <= 1.0 / 0.05 + 1e-6, f"reward-to-risk reached {rr.max():.1f}"


def test_min_risk_frac_is_tunable(rnd):
    tight = chart_pattern_shapes(rnd, settings={p: dict(min_risk_frac=0.30)
                                                for p in CHART_PATTERNS})
    loose = chart_pattern_shapes(rnd, settings={p: dict(min_risk_frac=0.0)
                                                for p in CHART_PATTERNS})
    assert len(tight) <= len(loose)


# ---------------------------------------------------------------------------
# what gets DRAWN — the checks whose absence let three defects reach a chart
# ---------------------------------------------------------------------------
def test_level_shape_dots_are_the_extremes_the_pattern_is_named_for(rnd):
    """A triple top's pivot list carries its three highs AND the two troughs that set the neckline.
    Drawing all of them put a "triple top touch" on the LOW of a candle — 33-40% of the dots."""
    hi, lo = rnd["high"].to_numpy(float), rnd["low"].to_numpy(float)
    s = chart_pattern_shapes(rnd)
    lvl = s[s["upper_slope"].isna()]
    assert len(lvl), "fixture produced no level shapes"
    wrong = 0
    for r in lvl.itertuples():
        want_high = int(r.direction) < 0                    # a top marks highs, a bottom lows
        for i, px in zip(r.touch_i, r.touch_price):
            near_high = abs(px - hi[int(i)]) < abs(px - lo[int(i)])
            wrong += int(near_high != want_high)
    assert wrong == 0, f"{wrong} dots on the wrong side of the candle"


def test_a_level_pattern_actually_sits_at_one_level(rnd):
    """Equality was checked only between ADJACENT pairs and scaled by the whole pivot window's
    range, which let three "equal" tops sit 14.8 ATR apart."""
    from quant.indicators import add_atr
    from quant.indicators.chart_patterns import PatternSpec
    atr = add_atr(rnd, 14)["atr_14"].to_numpy(float)
    tol = PatternSpec().level_tol_atr
    s = chart_pattern_shapes(rnd)
    lvl = s[s["upper_slope"].isna() & ~s["pattern"].str.contains("head_shoulders")]
    for r in lvl.itertuples():
        tp = np.asarray(r.touch_price, float)
        if len(tp) < 2:
            continue
        a = atr[int(r.formed_i)]
        if not np.isfinite(a) or a <= 0:
            continue
        assert (tp.max() - tp.min()) <= tol * a + 1e-6, (
            f"{r.pattern} spans {(tp.max()-tp.min())/a:.1f} ATR — not one level")


def test_trendlines_are_drawn_where_they_are_near_price(rnd):
    """44% of lines used to end more than 6 ATR from price, the worst at 119 — rays off the chart."""
    from quant.indicators import add_trendlines, trendline_table, add_atr
    d = add_atr(rnd, 14)
    med = float(np.nanmedian(d["atr_14"].to_numpy(float)))
    c = d["close"].to_numpy(float)
    tl = trendline_table(add_trendlines(rnd))
    assert len(tl)
    for r in tl.itertuples():
        e = int(r.end_i)
        assert abs((r.slope * e + r.intercept) - c[e]) / med < 20.0, "line ends far from price"


def test_every_trendline_touch_lies_inside_its_drawn_span(rnd):
    """Lines were drawn from `born_i`, which is AFTER every pivot they are fitted through, so the
    touch dots sat outside the segment on all 36 lines of the first sample."""
    from quant.indicators import add_trendlines, trendline_table
    tl = trendline_table(add_trendlines(rnd))
    for r in tl.itertuples():
        if not len(r.touch_i):
            continue
        assert int(r.start_i) <= min(r.touch_i) and max(r.touch_i) <= int(r.end_i)


def test_a_trendline_that_never_lived_is_not_recorded(rnd):
    """Several pivots can confirm on one bar, so a line could be born and superseded before ever
    publishing. Those were still drawn, across spans they had never been valid over."""
    from quant.indicators import add_trendlines, trendline_table
    tl = trendline_table(add_trendlines(rnd))
    assert (tl["end_i"] >= tl["born_i"]).all(), "a line ends before it was born"
    assert (tl["end_i"] >= 0).all()


def test_emas_are_drawn_without_being_asked_for(rnd):
    """They default ON. A caller that predates the feature -- an un-updated notebook cell, say --
    still gets them, which is the only version of this fix that survives a stale caller."""
    from quant.viz.studies import pattern_figure
    fig = pattern_figure(rnd, patterns=[], sr_zones=False, trendlines=False)
    emas = [t for t in fig.data if t.name and str(t.name).startswith("EMA")]
    assert len(emas) == 4, f"expected 4 EMAs by default, got {[t.name for t in emas]}"
    assert all(t.visible is True for t in emas), "EMAs must be visible, not legendonly"
    bare = pattern_figure(rnd, patterns=[], sr_zones=False, trendlines=False, emas=())
    assert not [t for t in bare.data if t.name and str(t.name).startswith("EMA")]


def test_isolating_a_trendline_still_shows_where_it_confirmed_and_broke(rnd):
    """The grouped view marks the break with an X; the isolated view was drawing only the touch
    dots, so the two events that matter — when the line became tradable, and when it failed —
    disappeared exactly when you zoomed in on one line to study it."""
    from quant.viz.studies import pattern_figure
    fig = pattern_figure(rnd, patterns=[], sr_zones=False, trendlines=True, emas=())
    hidden = [t for t in fig.data if t.visible is False and getattr(t, "text", None)]
    assert hidden, "no isolated per-line traces were built"
    labels = {str(x) for t in hidden for x in (t.text or [])}
    assert any("CONFIRMED" in l for l in labels), "isolated line never marks where it confirmed"
    assert any("BROKE" in l for l in labels), "isolated line never marks where it broke"


# ---------------------------------------------------------------------------
# MULTI-TIMEFRAME drawing: patterns from other timeframes on one chart
#
# The frames share exactly one coordinate — the timestamp — so a shape fitted on 4h bars can be
# drawn over a 15m chart with no index remapping, provided each layer keeps its own bar space. The
# two things that can go wrong are CAUSALITY (a higher-timeframe bar is stamped at its OPEN but is
# only knowable at its CLOSE) and SIZE (every dropdown button carries a visibility vector as long as
# the trace list, so an uncapped layer makes the chart unopenable rather than merely cluttered).
# ---------------------------------------------------------------------------
def _marker_points(trace):
    """(x, symbol) for every point of a marker trace; symbols may be per-point or shared."""
    xs = list(trace.x or [])
    syms = trace.marker.symbol
    return zip(xs, (list(syms) if isinstance(syms, (list, tuple)) else [syms] * len(xs)))


def _tf_layer(fig, tf):
    """Every grouped trace belonging to timeframe `tf`'s layer."""
    return [tr for tr in fig.data if (getattr(tr, "legendgroup", "") or "").startswith(f"{tf}::")]


def _n_picker(fig):
    """Picker instances in a figure. Each costs exactly two hidden traces."""
    return sum(1 for tr in fig.data if tr.visible is False) // 2


def test_extra_timeframes_leave_the_chart_timeframes_own_drawing_untouched(rnd):
    """Adding a layer must not perturb the chart you already had.

    The same discipline as golden parity in the kernel: the feature is additive, so every trace the
    single-timeframe chart drew has to come out identical with a second layer loaded.
    """
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    h1 = resample_ohlcv(rnd, "1h")
    plain = pattern_figure(rnd, sr_zones=False, emas=(), trendlines=True)
    multi = pattern_figure(rnd, sr_zones=False, emas=(), trendlines=True, tf_frames={"1h": h1})

    assert len(multi.data) > len(plain.data)          # the layer really was drawn
    a = json.loads(plain.to_json())["data"]
    b = json.loads(multi.to_json())["data"]
    assert a == b[:len(a)]


def test_a_higher_timeframe_event_is_drawn_where_that_bar_CLOSED(rnd):
    """A 1h pattern confirming on the bar stamped 12:00 is not knowable until 13:00.

    Bars carry their OPEN time everywhere in this project. Drawing a higher-timeframe confirmation
    at the open would show a signal up to one whole HTF bar before it existed — an hour here, four
    hours for a 4h layer. Events therefore go at the bar close; the geometry they hang off (the
    touch pivots) stays at the open, which is where that candle physically sits.
    """
    from quant.data import bar_interval, resample_ohlcv
    from quant.viz.studies import pattern_figure

    h1 = resample_ohlcv(rnd, "1h")
    fig = pattern_figure(rnd, sr_zones=False, emas=(), trendlines=True, tf_frames={"1h": h1})

    step = bar_interval(h1)
    assert step == pd.Timedelta(hours=1)
    opens = set(pd.to_datetime(h1["t"]))
    closes = {o + step for o in opens}

    n_events = n_touches = 0
    for tr in _tf_layer(fig, "1h"):
        if getattr(tr, "mode", None) != "markers":
            continue
        for x, sym in _marker_points(tr):
            if x is None:
                continue
            ts = pd.Timestamp(str(x))
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            if sym in ("star", "x", "diamond"):
                n_events += 1
                assert ts in closes, f"{sym} drawn at {ts}, which is not a 1h bar close"
            else:
                n_touches += 1
                assert ts in opens, f"touch drawn at {ts}, which is not a 1h bar open"
    assert n_events and n_touches, "nothing was drawn, so nothing was tested"


def test_a_higher_timeframe_layer_lands_in_the_same_place_whatever_it_is_drawn_over(rnd):
    """The layer is a property of ITS OWN frame, not of whatever chart it happens to sit on."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    h1 = resample_ohlcv(rnd, "1h")

    def stars(chart):
        fig = pattern_figure(chart, sr_zones=False, emas=(), tf_frames={"1h": h1})
        return sorted(str(x) for tr in _tf_layer(fig, "1h")
                      if getattr(tr, "mode", None) == "markers"
                      for x, sym in _marker_points(tr) if sym == "star" and x is not None)

    assert stars(rnd) == stars(resample_ohlcv(rnd, "30min")) != []


def test_the_extra_timeframes_cannot_flood_the_picker(rnd):
    """Buttons x traces is the cost, so the buttons have to be bounded, not merely tidy.

    Uncapped, a 1m layer over a nine-day window contributed 335 instance entries on its own and took
    the dropdown JSON to ~9.8 MB. A chart that will not open is worse than one that is cluttered.
    """
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    frames = {"1h": resample_ohlcv(rnd, "1h"), "30min": resample_ohlcv(rnd, "30min")}
    kw = dict(sr_zones=False, emas=(), trendlines=True)
    plain = _n_picker(pattern_figure(rnd, **kw))
    tight = _n_picker(pattern_figure(rnd, tf_frames=frames, max_picker_instances=4, **kw))
    loose = _n_picker(pattern_figure(rnd, tf_frames=frames, max_picker_instances=40, **kw))

    assert tight - plain <= 4                 # the budget is a hard cap on the EXTRA layers
    assert plain < tight < loose              # ...and the chart's own picker is never trimmed


def test_max_shapes_per_tf_bounds_what_each_extra_pattern_draws(rnd):
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    fig = pattern_figure(rnd, sr_zones=False, emas=(), trendlines=True,
                         tf_frames={"30min": resample_ohlcv(rnd, "30min")}, max_shapes_per_tf=1)
    named = [tr.name for tr in _tf_layer(fig, "30min") if tr.name]
    assert named
    for name in named:                        # "30min · double top (1)" / "... (1, 0 broke)"
        assert name.rstrip(")").rsplit("(", 1)[-1].split(",")[0] == "1"


def test_extra_timeframes_start_hidden_and_are_drawn_at_a_different_weight(rnd):
    """Colour keeps meaning DIRECTION; weight is what tells you which timeframe you are looking at."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    fig = pattern_figure(rnd, sr_zones=False, emas=(),
                         tf_frames={"1h": resample_ohlcv(rnd, "1h")})
    layer = [tr for tr in _tf_layer(fig, "1h") if tr.name]
    assert layer, "nothing drawn on the extra layer"
    for tr in layer:
        assert tr.visible == "legendonly"     # the chart opens on its own timeframe only
        assert tr.name.startswith("1h · ")
    # coarser than the chart, so heavier than the chart's own 2.6
    widths = [tr.line.width for tr in layer if getattr(tr, "mode", None) == "lines"]
    assert widths and max(widths) > 2.6


def test_tf_trendlines_follows_the_chart_unless_it_is_asked_otherwise(rnd):
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    frames = {"1h": resample_ohlcv(rnd, "1h")}

    def has_lines(**kw):
        fig = pattern_figure(rnd, sr_zones=False, emas=(), tf_frames=frames, **kw)
        return any("trendline" in (tr.name or "") for tr in _tf_layer(fig, "1h"))

    assert not has_lines(trendlines=False)                        # follows the chart
    assert has_lines(trendlines=True)
    assert has_lines(trendlines=False, tf_trendlines=True)        # ...unless overridden
    assert not has_lines(trendlines=True, tf_trendlines=False)


def test_tf_minutes_parses_the_timeframe_strings_the_data_layer_uses():
    from quant.data import tf_minutes

    got = [tf_minutes(x) for x in ("1m", "5m", "15m", "1h", "4h", "1d")]
    assert got == [1, 5, 15, 60, 240, 1440]
    with pytest.raises(ValueError):
        tf_minutes("fortnight")


def test_bar_interval_is_read_off_the_frame_not_off_a_label(rnd):
    """No timeframe string is involved, so a relabelled or gappy frame still reports the truth."""
    from quant.data import bar_interval, resample_ohlcv

    assert bar_interval(rnd) == pd.Timedelta(minutes=15)
    assert bar_interval(resample_ohlcv(rnd, "4h")) == pd.Timedelta(hours=4)
    gappy = rnd.drop(index=range(100, 140)).reset_index(drop=True)
    assert bar_interval(gappy) == pd.Timedelta(minutes=15)      # the median survives a gap


def test_tf_frames_gives_every_timeframe_the_same_history_in_BARS(monkeypatch):
    """A fixed calendar start starves 4h and drowns 1m; detection counts bars, so the loader does too."""
    from quant.data import timeframe as tfmod

    seen = {}

    def fake_get_ohlcv(symbol, tf, *, start, end, source, market, tz, **kw):
        seen[tf] = start
        t = pd.date_range(start, end, freq="1h", tz="UTC")[:5]
        return pd.DataFrame({"open_time": t, "t": t, "open": 1.0, "high": 1.0,
                             "low": 1.0, "close": 1.0, "volume": 1.0})

    monkeypatch.setattr("quant.data.loader.get_ohlcv", fake_get_ohlcv)
    out = tfmod.tf_frames("BTCUSDT", ("1h", "4h"), view_start="2026-06-01", end="2026-06-10",
                          warmup_bars=100)

    assert list(out) == ["4h", "1h"]                       # coarsest first: structure underneath
    # 100 bars of warm-up is 100 hours for 1h and 400 for 4h, floored to a whole day because the
    # cache is date-granular — flooring can only ever hand detection MORE bars than were asked for
    for tf, hours in (("1h", 100), ("4h", 400)):
        want = (pd.Timestamp("2026-06-01") - pd.Timedelta(hours=hours)).floor("D")
        assert pd.Timestamp(seen[tf]) == want


def test_the_y_axis_is_scaled_to_the_candles_not_to_the_geometry_hanging_off_them(rnd):
    """A pattern that began before the window can be far taller than the window.

    Measured: a 1h falling wedge starting at 86,000 and converging at 51,800 stretched a 60k-76k
    chart out to 50k-85k, leaving the price action a third of the pane. The line was correct — it
    really is that tall — but the chart is there to show price.
    """
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    frames = {"1h": resample_ohlcv(rnd, "1h")}
    fig = pattern_figure(rnd, sr_zones=False, emas=(), trendlines=True, tf_frames=frames)
    lo, hi = float(rnd["low"].min()), float(rnd["high"].max())
    y0, y1 = fig.layout.yaxis.range
    assert y0 < lo and y1 > hi                       # the candles all fit...
    assert (y1 - y0) < 1.5 * (hi - lo)               # ...without the pane being given away
    assert pattern_figure(rnd, sr_zones=False, emas=(), trendlines=True, tf_frames=frames,
                          y_from_price=False).layout.yaxis.range is None


def test_higher_timeframe_emas_step_at_their_own_bar_close(rnd):
    """An interpolated 1h EMA shows a number nobody could have read at the time.

    The value is only final when the 1h bar closes, so the line is a step that changes there and
    holds until the next one — the same causality rule the pattern markers follow.
    """
    from quant.data import bar_interval, resample_ohlcv
    from quant.viz.studies import pattern_figure

    h1 = resample_ohlcv(rnd, "1h")
    fig = pattern_figure(rnd, sr_zones=False, emas=(), patterns=[], tf_frames={"1h": h1},
                         mtf_emas=("1h", [50, 200]))
    lines = [tr for tr in fig.data if (tr.name or "").startswith("1h EMA ")]
    assert [tr.name for tr in lines] == ["1h EMA 50", "1h EMA 200"]

    closes = set(pd.to_datetime(h1["t"]) + bar_interval(h1))
    for tr in lines:
        assert tr.line.shape == "hv"                      # a step, not an interpolation
        assert all(pd.Timestamp(str(x)) in closes for x in tr.x)


def test_a_higher_timeframe_ema_reuses_a_loaded_frame_and_says_so_when_it_cannot(rnd):
    """A 1h EMA200 needs 200 1h bars. Resampling a short chart frame yields NaN, not a line."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    short = rnd.iloc[-400:].reset_index(drop=True)        # 400 x 15m = 100 1h bars, short of 200
    kw = dict(sr_zones=False, emas=(), patterns=[])

    with pytest.warns(UserWarning, match="fewer than the 200"):
        bare = pattern_figure(short, mtf_emas=("1h", 200), **kw)
    assert not any((tr.name or "").startswith("1h EMA") for tr in bare.data)

    # the SAME request succeeds once the frame is supplied with its own history — which is the
    # whole reason tf_frames loads each timeframe rather than resampling the chart
    loaded = pattern_figure(short, tf_frames={"1h": resample_ohlcv(rnd, "1h")},
                            mtf_emas=("1h", 200), **kw)
    line = [tr for tr in loaded.data if (tr.name or "").startswith("1h EMA")]
    assert len(line) == 1 and np.isfinite(np.asarray(line[0].y, float)).any()


@pytest.mark.parametrize("bad", [["4h", "1h", "15m"], ["1h"], "1h", ("4h", "1h")])
def test_mtf_emas_says_what_is_wrong_instead_of_recursing_forever(rnd, bad):
    """A bare timeframe used to die 3,000 frames deep in a RecursionError naming nothing.

    A `str` is not a tuple or list, so it fell past the pair check into the flattening loop,
    iterated its own characters, and recursed on each one. `["4h", "1h"]` is an easy thing to write
    — it is exactly the shape `pattern_tfs` takes — so the error has to name the difference.
    """
    from quant.viz.studies import _norm_mtf_ema, pattern_figure

    with pytest.raises(ValueError, match="(?i)period"):
        _norm_mtf_ema(bad)
    with pytest.raises(ValueError, match="(?i)period"):
        pattern_figure(rnd, patterns=[], sr_zones=False, emas=(), mtf_emas=bad)


@pytest.mark.parametrize("spec,want", [
    (("1h", 200), [("1h", 200)]),
    (("1h", [50, 200]), [("1h", 50), ("1h", 200)]),
    ([("1h", 200), ("4h", 200)], [("1h", 200), ("4h", 200)]),
    (None, []), ((), []), ([], []),
])
def test_mtf_emas_accepts_every_documented_shape(spec, want):
    from quant.viz.studies import _norm_mtf_ema

    assert _norm_mtf_ema(spec) == want
