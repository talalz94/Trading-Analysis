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
    assert full[cols].to_numpy().sum() > 20, "fixture must actually produce overlapping detections"
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
