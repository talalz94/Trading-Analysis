"""
Fair value gaps: the geometry, when it is knowable, when it dies, and the veto built on it.

The three-candle definition is simple enough that the interesting failures are all about ORIENTATION
and TIME -- which edge is the top, which bar the gap is published from, and whether a higher
timeframe's gap can be seen before its bar closed. So that is what most of this file is about.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.indicators import add_fvgs, fvg_table, mtf_fvgs
from quant.strategies import fvg_blocks, fvg_clear


def _f(o, h, l, c, freq="5min"):
    n = len(c)
    t = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": np.asarray(o, float),
                         "high": np.asarray(h, float), "low": np.asarray(l, float),
                         "close": np.asarray(c, float), "volume": 1.0})


def _bull(extra_bars=3):
    """c0 tops at 10, c1 displaces up, c2 bottoms at 12 -> a bullish gap of 10..12."""
    o = [9, 11, 13] + [13] * extra_bars
    h = [10, 20, 22] + [22] * extra_bars
    l = [8, 10, 12] + [12] * extra_bars
    c = [10, 19, 21] + [21] * extra_bars
    return _f(o, h, l, c)


# ---------------------------------------------------------------------------
# the geometry
# ---------------------------------------------------------------------------
def test_a_bullish_gap_is_the_band_between_the_first_high_and_the_third_low():
    g = fvg_table(add_fvgs(_bull(), min_size_atr=0, atr_period=2))
    assert len(g) == 1
    r = g.iloc[0]
    assert r.kind == 1 and r.side == "bullish"
    assert (r.bot, r.top) == (10.0, 12.0)
    assert r.mid == 11.0


def test_a_bearish_gap_is_the_mirror_image_and_top_is_still_the_higher_edge():
    """The bug this replaces: `top` and `bot` were written in CANDLE order rather than PRICE order,
    so every bearish band came out upside down -- and an inverted band inverts its own mitigation
    test, its midpoint and the box drawn for it."""
    d = _f([21, 19, 13, 13, 13], [22, 20, 12, 12, 12], [20, 10, 8, 8, 8], [21, 11, 9, 9, 9])
    g = fvg_table(add_fvgs(d, min_size_atr=0, atr_period=2))
    assert len(g) == 1
    r = g.iloc[0]
    assert r.kind == -1 and r.side == "bearish"
    assert (r.bot, r.top) == (12.0, 20.0)
    assert r.bot < r.top, "top must always be the higher edge"


def test_the_gap_is_published_from_the_third_candle_not_the_middle_one():
    """It reads the third candle's own low, so it cannot be known before that candle closes. Drawing
    it where it visually sits -- around the middle candle -- would be a two-bar lookahead."""
    g = fvg_table(add_fvgs(_bull(), min_size_atr=0, atr_period=2))
    assert int(g.born_i.iloc[0]) == 2


def test_overlapping_candles_are_not_a_gap():
    d = _f([9, 11, 13], [10, 20, 22], [8, 10, 9.5], [10, 19, 21])
    assert not len(fvg_table(add_fvgs(d, min_size_atr=0, atr_period=2)))


# ---------------------------------------------------------------------------
# causality — the test every detector in this project gets
# ---------------------------------------------------------------------------
def test_causality_by_truncation():
    """Run on the first k bars and every gap already published must be identical.

    A detector that fails this is reading its own future: the gap list from the whole frame would
    disagree with what you could actually have seen at bar k.
    """
    rng = np.random.default_rng(11)
    n = 1200
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 25, n))
    d = _f(open_, np.maximum(open_, close) + wick, np.minimum(open_, close) - wick, close,
           freq="1min")

    full = fvg_table(add_fvgs(d))
    k = 800
    part = fvg_table(add_fvgs(d.iloc[:k].reset_index(drop=True)))
    assert len(part), "the fixture produced no gaps at all — it is not testing anything"

    a = full[full.born_i < k]
    assert len(a) == len(part), "truncating changed WHICH gaps exist"
    for col in ("kind", "born_i", "top", "bot"):
        assert np.allclose(a[col].to_numpy(float), part[col].to_numpy(float)), col
    # a mitigation that had already happened before bar k must also be unchanged
    m = a["dead_i"].to_numpy(np.int64)
    p = part["dead_i"].to_numpy(np.int64)
    both = (m >= 0) & (m < k)
    assert np.array_equal(m[both], p[both]), "a mitigation moved when the future was removed"


# ---------------------------------------------------------------------------
# how it dies
# ---------------------------------------------------------------------------
def test_touch_ce_and_full_are_three_different_questions():
    # band 10..12, midpoint 11. bar 3 dips to 11.5, bar 4 to 10.8, bar 5 to 9.5
    d = _bull(extra_bars=3)
    d.loc[3, "low"], d.loc[4, "low"], d.loc[5, "low"] = 11.5, 10.8, 9.5
    g = fvg_table(add_fvgs(d, min_size_atr=0, atr_period=2))
    r = g.iloc[0]
    assert (int(r.touch_i), int(r.ce_i), int(r.full_i)) == (3, 4, 5)


@pytest.mark.parametrize("mode,dead", [("touch", 3), ("ce", 4), ("full", 5)])
def test_the_mitigation_mode_picks_which_one_retires_the_gap(mode, dead):
    d = _bull(extra_bars=3)
    d.loc[3, "low"], d.loc[4, "low"], d.loc[5, "low"] = 11.5, 10.8, 9.5
    g = fvg_table(add_fvgs(d, mitigation=mode, min_size_atr=0, atr_period=2))
    assert int(g.dead_i.iloc[0]) == dead
    # all three are recorded whatever the mode is, so changing your mind costs a column lookup
    assert (int(g.touch_i.iloc[0]), int(g.ce_i.iloc[0]), int(g.full_i.iloc[0])) == (3, 4, 5)


def test_an_unmitigated_gap_reports_minus_one_rather_than_a_bar():
    g = fvg_table(add_fvgs(_bull(), min_size_atr=0, atr_period=2))
    assert int(g.dead_i.iloc[0]) == -1


def test_a_bad_mitigation_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="mitigation"):
        add_fvgs(_bull(), mitigation="halfway")


# ---------------------------------------------------------------------------
# which ones count
# ---------------------------------------------------------------------------
def test_the_size_gate_is_in_atr_so_it_carries_across_instruments():
    d = _bull()
    big = fvg_table(add_fvgs(d, min_size_atr=0.1, atr_period=2))
    small = fvg_table(add_fvgs(d, min_size_atr=50.0, atr_period=2))
    assert len(big) == 1 and len(small) == 0


def test_the_displacement_gate_looks_at_the_middle_candle():
    """The gap is made by the candle between the two that define it, so that is the body to test."""
    d = _bull()
    assert len(fvg_table(add_fvgs(d, min_size_atr=0, min_disp_atr=0.1, atr_period=2))) == 1
    d2 = d.copy()
    d2.loc[1, "open"], d2.loc[1, "close"] = 19.0, 19.1      # same gap, no displacement
    assert len(fvg_table(add_fvgs(d2, min_size_atr=0, min_disp_atr=1.0, atr_period=2))) == 0


def test_a_frame_too_short_to_hold_three_candles_returns_nothing_rather_than_raising():
    out = add_fvgs(_f([1, 2], [2, 3], [0, 1], [1, 2]), min_size_atr=0)
    assert not len(fvg_table(out)) and len(out) == 2
    assert out["fvg_bull_n"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# the per-bar columns
# ---------------------------------------------------------------------------
def test_the_live_count_starts_at_the_gaps_own_bar_and_stops_when_it_dies():
    d = _bull(extra_bars=3)
    d.loc[4, "low"] = 10.8                                   # reaches the midpoint at bar 4
    out = add_fvgs(d, mitigation="ce", min_size_atr=0, atr_period=2)
    assert out["fvg_bull_n"].tolist() == [0, 0, 1, 1, 0, 0]
    assert out["fvg_bear_n"].sum() == 0


def test_the_nearest_gap_columns_describe_that_gap():
    out = add_fvgs(_bull(), min_size_atr=0, atr_period=2)
    i = 2
    assert out["fvg_bull_top"].iloc[i] == 12.0
    assert out["fvg_bull_bot"].iloc[i] == 10.0
    assert out["fvg_bull_age"].iloc[i] == 0
    # price closed at 21, 9 above the top edge; distance is to the NEAR edge, in ATR
    assert out["fvg_bull_dist_atr"].iloc[i] > 0


def test_distance_is_zero_while_price_is_inside_the_band():
    d = _bull(extra_bars=1)
    d.loc[3, ["open", "high", "low", "close"]] = [11.5, 11.8, 11.2, 11.4]   # inside 10..12
    out = add_fvgs(d, mitigation="full", min_size_atr=0, atr_period=2)
    assert out["fvg_bull_dist_atr"].iloc[3] == 0.0


# ---------------------------------------------------------------------------
# several timeframes at once
# ---------------------------------------------------------------------------
def _walk(n=2000, seed=3, freq="1min"):
    rng = np.random.default_rng(seed)
    close = 30000 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 25, n))
    return _f(open_, np.maximum(open_, close) + wick, np.minimum(open_, close) - wick, close,
              freq=freq)


def test_mtf_fvgs_puts_every_timeframes_count_on_the_base_grid():
    from quant.data import resample_ohlcv

    base = _walk()
    frames = {"5m": resample_ohlcv(base, "5min"), "15m": resample_ohlcv(base, "15min")}
    out = mtf_fvgs(base, frames)
    assert len(out) == len(base)
    for c in ("fvg_bull_n", "5m__fvg_bull_n", "15m__fvg_bear_n"):
        assert c in out.columns
    assert out["5m__fvg_bull_n"].dtype == np.int32, "a shifted count must come back an int"
    assert out.attrs.get("5m__fvg_gaps") is not None, "each layer's table must ride along"


def test_a_higher_timeframes_gap_is_invisible_until_its_bar_closed():
    """A 15m gap confirmed on the bar stamped 12:00 is knowable at 12:15, not at 12:00."""
    from quant.data import resample_ohlcv

    base = _walk()
    frames = {"15m": resample_ohlcv(base, "15min")}
    full = mtf_fvgs(base, frames)
    k = 1400
    part = mtf_fvgs(
        base.iloc[:k].reset_index(drop=True),
        {"15m": frames["15m"][pd.to_datetime(frames["15m"]["t"]) < base["t"].iloc[k]]
         .reset_index(drop=True)})
    a = full["15m__fvg_bull_n"].to_numpy()[:k]
    b = part["15m__fvg_bull_n"].to_numpy()
    assert a.sum() > 0, "the fixture produced no 15m gaps — it is not testing anything"
    assert np.array_equal(a, b), "a 15m count changed when the future was removed"


# ---------------------------------------------------------------------------
# the veto
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vframe():
    from quant.data import resample_ohlcv

    base = _walk(n=3000, seed=8)
    return mtf_fvgs(base, {"5m": resample_ohlcv(base, "5min")}, min_size_atr=0.2)


def test_the_two_modes_are_exact_mirrors(vframe):
    """Not a style point: if they ever stop being mirrors, one of them has its direction wrong, and
    a veto pointing the wrong way is worse than no veto."""
    for a, b in (("long", "short"), ("short", "long")):
        x = fvg_blocks(a, mode="retrace")(vframe)
        y = fvg_blocks(b, mode="barrier")(vframe)
        assert np.array_equal(x, y), f"{a}/retrace should equal {b}/barrier"


def test_retrace_blocks_a_long_with_an_unfilled_bullish_gap_below(vframe):
    """With no `timeframes` given, every loaded layer may veto -- so it is the OR across them."""
    blocked = fvg_blocks("long", mode="retrace")(vframe)
    want = (vframe["fvg_bull_n"].to_numpy() > 0) | (vframe["5m__fvg_bull_n"].to_numpy() > 0)
    assert np.array_equal(blocked, want)


def test_every_gate_can_only_narrow_the_veto(vframe):
    wide = fvg_blocks("long", mode="retrace")(vframe)
    for gate in (dict(max_age_bars=5), dict(min_size_atr=2.0), dict(max_dist_atr=0.1)):
        tight = fvg_blocks("long", mode="retrace", **gate)(vframe)
        assert (tight <= wide).all(), f"{gate} let through a bar the ungated veto blocked"


def test_restricting_the_timeframes_ignores_the_others(vframe):
    only5 = fvg_blocks("long", mode="retrace", timeframes=("5m",))(vframe)
    assert np.array_equal(only5, vframe["5m__fvg_bull_n"].to_numpy() > 0)
    chart = fvg_blocks("long", mode="retrace", timeframes=("chart",))(vframe)
    assert np.array_equal(chart, vframe["fvg_bull_n"].to_numpy() > 0)


def test_fvg_clear_is_the_negation_and_is_what_an_entry_filter_wants(vframe):
    """`entry_filter` is a MAY-ENTER condition. Handing it the veto itself would trade exactly the
    setups it was built to avoid, and nothing would look wrong."""
    assert np.array_equal(fvg_clear("long")(vframe), ~fvg_blocks("long")(vframe))


def test_a_frame_without_gap_columns_says_so_rather_than_raising_a_keyerror():
    with pytest.raises(KeyError, match="mtf_fvgs"):
        fvg_blocks("long")(_walk(n=50))


@pytest.mark.parametrize("bad", [dict(side="sideways"), dict(side="long", mode="sideways")])
def test_a_bad_argument_is_refused_at_build_time_not_at_call_time(bad):
    with pytest.raises(ValueError):
        fvg_blocks(**bad)


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def test_a_whole_layer_of_gaps_costs_one_trace_pair():
    """The cap that matters. Every picker button carries a visibility vector as long as the trace
    list, so a trace per box would make the cost boxes x buttons."""
    from quant.viz.studies import pattern_figure

    d = _walk(n=2000, freq="5min")
    fig = pattern_figure(d, patterns=[], sr_zones=False, chart_tf="5m", fvgs=True,
                         max_fvg_picker=0, emas=())
    boxes = [t for t in fig.data if getattr(t, "fill", None) == "toself"]
    assert len(boxes) == 2, [t.name for t in boxes]     # one bullish, one bearish
    n_drawn = sum(1 for v in boxes[0].x if v is None)
    assert n_drawn > 5, "the fixture drew almost nothing — it is not testing the bulk path"


def test_turning_fvgs_on_leaves_every_existing_trace_exactly_as_it_was():
    from quant.viz.studies import pattern_figure

    d = _walk(n=2000, freq="5min")
    kw = dict(patterns=[], sr_zones=False, trendlines=True, chart_tf="5m", emas=(50,))
    off = pattern_figure(d, **kw)
    on = pattern_figure(d, fvgs=True, **kw)
    # every trace the gap layer adds is tagged meta="fvg" -- one reliable handle, rather than
    # guessing from names that a future rename would quietly break
    keep = [t.name for t in on.data if getattr(t, "meta", None) != "fvg"]
    assert keep == [t.name for t in off.data]


def test_the_picker_cap_bounds_the_dropdown_not_the_drawing():
    """Trimming only the menu is what once left a legend reading (11) beside a dropdown of 9."""
    from quant.viz.studies import pattern_figure

    d = _walk(n=2000, freq="5min")
    fig = pattern_figure(d, patterns=[], sr_zones=False, chart_tf="5m", fvgs=True,
                         max_fvg_picker=3, emas=())
    named = [t for t in fig.data if (t.legendgroup or "").startswith("i#")
             and getattr(t, "fill", None) == "toself"]
    assert len(named) <= 6, "more isolatable gaps than the cap allows"
    bulk = [t for t in fig.data if getattr(t, "fill", None) == "toself"
            and (t.name or "").find("FVG") >= 0]
    assert len(bulk) == 2, "the bulk layer must be drawn in full regardless of the cap"
    # and the count in the legend is the number DRAWN, not the number listed
    assert "(" in bulk[0].name and int(bulk[0].name.split("(")[1].split(",")[0]) >= len(named)


# ---------------------------------------------------------------------------
# the one-line gap-veto language
# ---------------------------------------------------------------------------
def test_a_gap_sentence_parses_the_way_it_reads():
    from quant.strategies import parse_fvg_rule

    r = parse_fvg_rule("short: opposite FVG 15m+1h within 20 candles min 0.8 ATR near 3 ATR")
    assert r.direction is None                      # None == "opposite", resolved per side
    assert r.timeframes == ("15m", "1h")
    assert (r.within, r.min_atr, r.near_atr) == (20, 0.8, 3.0)
    assert r.sides == ("short",)


@pytest.mark.parametrize("text", [
    "opposite fvg 15m within 20 min 0.8",
    "within 20 opposite min 0.8 15m",                 # order does not matter
    "no opposite fvg the 15m within 20 candles min 0.8 atr",   # filler ignored
])
def test_word_order_and_filler_do_not_change_a_gap_rule(text):
    from quant.strategies import parse_fvg_rule

    r = parse_fvg_rule(text)
    assert (r.direction, r.timeframes, r.within, r.min_atr) == (None, ("15m",), 20, 0.8)


def test_opposite_means_the_gap_whose_fill_moves_price_against_the_trade(vframe):
    """The one word here anyone could read two ways, and reading it backwards inverts the filter.

    "Opposite" is NOT "a bullish gap when I am short". It is the gap whose FILL would move price
    AGAINST the trade: for a SHORT that is a BEARISH gap, which sits ABOVE and is filled by price
    RISING -- the sweep that kills the short. For a LONG it is a BULLISH gap below.
    """
    from quant.strategies import fvg_blocks, fvg_rule_cond

    assert np.array_equal(fvg_rule_cond("opposite fvg", "short")(vframe),
                          fvg_blocks("short", which="bear")(vframe))
    assert np.array_equal(fvg_rule_cond("opposite fvg", "long")(vframe),
                          fvg_blocks("long", which="bull")(vframe))
    # "same" is the exact mirror on both sides
    assert np.array_equal(fvg_rule_cond("same fvg", "short")(vframe),
                          fvg_blocks("short", which="bull")(vframe))
    assert np.array_equal(fvg_rule_cond("same fvg", "long")(vframe),
                          fvg_blocks("long", which="bear")(vframe))


def test_describe_spells_out_which_gap_each_sentence_blocks_on():
    """Left to a docstring, "opposite" is something you have to reason about every time."""
    from quant.strategies import describe_fvg_rules

    row = describe_fvg_rules(["opposite fvg 15m within 20"]).iloc[0]
    assert row.blocks_short_on == "bearish gap"
    assert row.blocks_long_on == "bullish gap"


def test_an_absolute_direction_ignores_the_side(vframe):
    from quant.strategies import fvg_rule_cond

    a = fvg_rule_cond("bear fvg", "long")(vframe)
    b = fvg_rule_cond("bear fvg", "short")(vframe)
    assert np.array_equal(a, b), "bear should mean bear whichever way the trade points"


def test_build_fvg_rules_returns_the_MAY_ENTER_condition(vframe):
    from quant.strategies import build_fvg_rules, fvg_rule_cond

    r = ["opposite fvg"]
    allow = build_fvg_rules(r, "short")(vframe)
    block = fvg_rule_cond(r[0], "short")(vframe)
    assert np.array_equal(allow, ~block), "entry_filter takes a MAY-ENTER condition"


def test_two_gap_rules_block_if_EITHER_objects(vframe):
    from quant.strategies import build_fvg_rules, fvg_rule_cond

    r = ["opposite fvg", "same fvg"]
    allow = build_fvg_rules(r, "short")(vframe)
    a = fvg_rule_cond(r[0], "short")(vframe)
    b = fvg_rule_cond(r[1], "short")(vframe)
    assert np.array_equal(allow, ~(a | b))


def test_no_gap_rule_for_a_side_returns_none_rather_than_guessing():
    from quant.strategies import build_fvg_rules

    assert build_fvg_rules([], "long") is None
    assert build_fvg_rules(["short: opposite fvg"], "long") is None


@pytest.mark.parametrize("bad,why", [
    ("15m within 20", "no direction"),
    ("opposite fvg 15m within", "cannot read"),
    ("opposite fvg sideways", "cannot read"),
])
def test_a_bad_gap_sentence_says_what_is_wrong_with_it(bad, why):
    from quant.strategies import parse_fvg_rule

    with pytest.raises(ValueError, match=why):
        parse_fvg_rule(bad)
