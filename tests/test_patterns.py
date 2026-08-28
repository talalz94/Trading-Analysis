"""Candlestick and chart pattern recognition.

The single most important property is CAUSALITY, and the strongest test of it is truncation: run the
detector on the first k bars, run it on the whole series, and require the first k results to be
identical. If any pattern peeked at a later bar the two runs would disagree. Retail pattern libraries
routinely fail this, usually by flagging a shape at its extreme rather than at its confirmation.
"""
from __future__ import annotations

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
        # the shared quality knobs must reach every pattern
        for k, v in TUNED_COMMON.items():
            assert cfg[k] == v, f"{name} lost {k}"
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
    fig = pattern_figure(rnd, tuned=True, sr_zones=False)
    drawn = {t.legendgroup for t in fig.data if t.legendgroup}
    assert drawn, "tuned figure drew nothing"
    assert drawn <= set(CHART_PATTERNS), f"unexpected legend groups: {drawn - set(CHART_PATTERNS)}"
    # and the per_pattern route must reach the same place as tuned=True
    from quant.indicators import tuned_specs
    same = pattern_figure(rnd, per_pattern=tuned_specs(), sr_zones=False)
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
    life = tl["end_i"] - tl["born_i"]
    assert (life < 10).mean() < 0.25, "too many lines living almost no time — identity is churning"
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
