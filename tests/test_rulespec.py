"""
The one-line rule language: "2 candles EMA_5m_200 close above".

The Cond algebra could already express every one of these; the point of the language is that a
notebook config is not the place for code. So the tests are mostly about the two ways a sentence
can betray you — parsing to something you did not mean, or resolving to a column that is not there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.rulespec import (build_rules, describe_rules, parse_rule, required_emas,
                                       resolve_ref, rule_cond)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def test_the_sentence_parses_the_way_it_reads():
    p = parse_rule("2 candles EMA_5m_200 close above")
    assert (p.n, p.ref, p.rel, p.side, p.colour) == (2, "EMA_5m_200", "close", "above", None)
    assert p.sides == ("long", "short")


@pytest.mark.parametrize("text", [
    "3 candles EMA_5m_200 formed fully below red",
    "EMA_5m_200 3 fully formed below red",              # order does not matter
    "red 3 the EMA_5m_200 is formed fully below",       # ...and filler words are ignored
])
def test_word_order_and_filler_do_not_change_the_meaning(text):
    p = parse_rule(text)
    assert (p.n, p.ref, p.rel, p.side, p.colour) == (3, "EMA_5m_200", "full", "below", "red")


@pytest.mark.parametrize("word,rel", [("close", "close"), ("closed", "close"),
                                      ("full", "full"), ("fully", "full"), ("formed", "full"),
                                      ("touch", "touch"), ("touched", "touch")])
def test_every_spelling_of_a_relation(word, rel):
    side = "" if rel == "touch" else "above"
    assert parse_rule(f"1 candle EMA_5m_50 {word} {side}").rel == rel


@pytest.mark.parametrize("word,colour", [("red", "red"), ("bearish", "red"), ("green", "green"),
                                         ("bull", "green"), ("NA", None), ("any", None)])
def test_colour_words(word, colour):
    assert parse_rule(f"1 EMA_5m_50 close above {word}").colour == colour


def test_a_side_prefix_restricts_the_rule():
    assert parse_rule("long: 2 EMA_5m_200 close above").sides == ("long",)
    assert parse_rule("short: 2 EMA_5m_200 close below").sides == ("short",)
    assert parse_rule("2 EMA_5m_200 close above").sides == ("long", "short")


@pytest.mark.parametrize("bad,why", [
    ("2 candles close above", "no reference"),
    ("2 candles EMA_5m_200 above", "no relation"),
    ("2 candles EMA_5m_200 close", "needs"),
    ("2 candles EMA_5m_200 touched above", "takes no side"),
    ("2 candles EMA_5m_200 EMA_15m_50 close above", "two references"),
])
def test_a_bad_sentence_says_what_is_wrong_with_it(bad, why):
    with pytest.raises(ValueError, match=why):
        parse_rule(bad)


def test_a_touch_cannot_take_a_side():
    """A touch is the candle STRADDLING the level. "touched above" would be a different test, and
    silently picking one of the two readings is worse than refusing."""
    assert parse_rule("1 SUP_15m touched").side is None
    with pytest.raises(ValueError):
        parse_rule("1 SUP_15m touched below")


# ---------------------------------------------------------------------------
# resolving a reference to a column
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ref,chart_tf,col", [
    ("EMA_5m_200", "5m", "ema_200"),            # the chart's own timeframe is unprefixed
    ("EMA_5m_200", "1m", "5m__ema_200"),
    ("EMA_15m_50", "5m", "15m__ema_50"),
    ("RES_5m", "5m", "tl_res"),
    ("SUP_15m", "5m", "15m__tl_sup"),
    ("resistance_1h", "5m", "1h__tl_res"),
    ("some_column", "5m", "some_column"),       # escape hatch: anything already on the frame
])
def test_a_reference_names_a_column(ref, chart_tf, col):
    assert resolve_ref(ref, chart_tf=chart_tf) == col


def test_required_emas_says_exactly_which_columns_to_add():
    r = ["2 EMA_5m_200 close above", "3 EMA_15m_50 close below", "1 EMA_5m_20 touched",
         "1 SUP_15m touched"]
    assert required_emas(r, chart_tf="5m") == {"": [200, 20], "15m": [50]}


def test_a_missing_column_names_itself_rather_than_raising_a_keyerror():
    d = pd.DataFrame({"open": [1.0], "high": [2.0], "low": [0.0], "close": [1.5]})
    with pytest.raises(KeyError, match="ema_200"):
        rule_cond("2 EMA_5m_200 close above", chart_tf="5m")(d)


# ---------------------------------------------------------------------------
# what the relations actually test
# ---------------------------------------------------------------------------
def _frame(o, h, lo, c, lvl=100.0):
    return pd.DataFrame({"open": np.array(o, float), "high": np.array(h, float),
                         "low": np.array(lo, float), "close": np.array(c, float),
                         "ema_50": lvl})


def test_close_full_and_touch_are_three_different_questions():
    # one candle: closes above 100, but its LOW is below it, and its range contains it
    d = _frame([99], [104], [98], [102])
    assert rule_cond("1 EMA_5m_50 close above", chart_tf="5m")(d)[0]
    assert not rule_cond("1 EMA_5m_50 full above", chart_tf="5m")(d)[0], \
        "the low is under the level, so the candle is not wholly above it"
    assert rule_cond("1 EMA_5m_50 touched", chart_tf="5m")(d)[0]

    # wholly above: low > level
    d2 = _frame([101], [105], [100.5], [104])
    assert rule_cond("1 EMA_5m_50 full above", chart_tf="5m")(d2)[0]
    assert not rule_cond("1 EMA_5m_50 touched", chart_tf="5m")(d2)[0]


def test_colour_is_tested_on_the_same_candle():
    green = _frame([99], [104], [98], [102])        # closes above, and is green
    red = _frame([103], [104], [98], [102])         # closes above, but is RED
    assert rule_cond("1 EMA_5m_50 close above green", chart_tf="5m")(green)[0]
    assert not rule_cond("1 EMA_5m_50 close above green", chart_tf="5m")(red)[0]
    assert rule_cond("1 EMA_5m_50 close above red", chart_tf="5m")(red)[0]
    assert rule_cond("1 EMA_5m_50 close above", chart_tf="5m")(red)[0], "NA means either colour"


def test_n_means_n_consecutive_candles_ending_here():
    #        0     1     2     3     4
    c = [102, 102, 99, 102, 102]
    d = _frame([101] * 5, [104] * 5, [98] * 5, c)
    got = rule_cond("2 EMA_5m_50 close above", chart_tf="5m")(d)
    assert list(got) == [False, True, False, False, True]
    got3 = rule_cond("3 EMA_5m_50 close above", chart_tf="5m")(d)
    assert not got3.any(), "there is never a run of three"


def test_a_bar_with_no_level_can_never_satisfy_a_rule():
    d = _frame([99], [104], [98], [102])
    d["ema_50"] = np.nan
    assert not rule_cond("1 EMA_5m_50 close above", chart_tf="5m")(d).any()


# ---------------------------------------------------------------------------
# combining
# ---------------------------------------------------------------------------
def test_rules_are_joined_by_all_or_any():
    # bar 0 closes ABOVE 100 and straddles it; bar 1 closes BELOW it but still straddles it
    d = _frame([99, 99], [104, 104], [98, 98], [102, 97])
    both = ["1 EMA_5m_50 close above", "1 EMA_5m_50 touched"]
    assert list(build_rules(both, "long", chart_tf="5m", combine="all")(d)) == [True, False]
    assert list(build_rules(both, "long", chart_tf="5m", combine="any")(d)) == [True, True]


def test_a_side_only_sees_its_own_rules():
    r = ["long: 1 EMA_5m_50 close above", "short: 1 EMA_5m_50 close below"]
    assert build_rules(r, "long", chart_tf="5m").name == "long: 1 EMA_5m_50 close above"
    assert build_rules(r, "short", chart_tf="5m").name == "short: 1 EMA_5m_50 close below"


def test_no_applicable_rule_returns_none_rather_than_guessing():
    """An empty `all` is vacuously true and an empty `any` is false; silently picking either would
    turn "I wrote no rules for shorts" into "no short may ever trade"."""
    assert build_rules([], "long") is None
    assert build_rules(["long: 1 EMA_5m_50 close above"], "short", chart_tf="5m") is None


def test_describe_shows_what_each_sentence_became():
    t = describe_rules(["long: 3 EMA_15m_50 formed fully below red"], chart_tf="5m")
    row = t.iloc[0]
    assert (row.n, row.column, row.relation, row.side, row.colour, row.applies_to) == \
        (3, "15m__ema_50", "full", "below", "red", "long")


# ---------------------------------------------------------------------------
# the multi-timeframe EMA columns the rules refer to
# ---------------------------------------------------------------------------
def _bars(n=900, seed=5, freq="1min"):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    t = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": close, "high": close * 1.001,
                         "low": close * 0.999, "close": close, "volume": 1.0})


def test_add_mtf_emas_puts_each_timeframes_ema_on_the_base_grid():
    from quant.data import resample_ohlcv
    from quant.indicators import add_mtf_emas

    base = _bars()
    frames = {"5m": resample_ohlcv(base, "5min")}
    out = add_mtf_emas(base, frames, {"": [20], "5m": [50]})
    assert "ema_20" in out.columns and "5m__ema_50" in out.columns
    assert len(out) == len(base)
    # the 5m EMA changes once per 5m bar, so it is far flatter than the base one
    assert out["5m__ema_50"].diff().ne(0).sum() < out["ema_20"].diff().ne(0).sum()


def test_a_higher_timeframe_ema_is_invisible_until_that_bar_closed():
    from quant.data import resample_ohlcv
    from quant.indicators import add_mtf_emas

    base = _bars()
    frames = {"5m": resample_ohlcv(base, "5min")}
    full = add_mtf_emas(base, frames, {"5m": [50]})
    k = 600
    part = add_mtf_emas(
        base.iloc[:k].reset_index(drop=True),
        {"5m": frames["5m"][pd.to_datetime(frames["5m"]["t"]) < base["t"].iloc[k]]
         .reset_index(drop=True)}, {"5m": [50]})
    a, b = full["5m__ema_50"].to_numpy()[:k], part["5m__ema_50"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.any() and np.allclose(a[ok], b[ok]), "an EMA changed when the future was removed"


def test_a_long_period_needs_its_own_frame_and_says_so():
    """Resampling a short base frame yields NaN instead of an EMA; refusing beats that."""
    from quant.indicators import add_mtf_emas

    with pytest.raises(KeyError, match="no frame for"):
        add_mtf_emas(_bars(), {}, {"1h": [200]})
