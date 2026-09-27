"""
S/R ZONES — the three corrections of 2026-09-27, and the knob that chooses between tuned presets.

Every one of these was found on a real chart before it was written down:
  * a zone could grow wider than `cluster_pct` (its mean moved as pivots joined, so it crept);
  * crossing a zone did nothing -- 48 of 54 zones on one chart had been broken and held beyond;
  * eviction kept old heavily-touched zones and threw away new ones next to price, so on a
    trending instrument the table filled with levels a median 67% away.

The legacy behaviour is kept on its ORIGINAL code path, so strategies and scenarios whose results
are recorded in docs/ are unchanged; `sensitivity=None` reaches it from the chart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.indicators import add_sr_zones, sr_params, sr_zone_table, SR_LEVELS, SR_TUNED
from quant.indicators.structure import _confirmed_pivots, _sr_core, _nearest_tuned_tf


def _frame(n=3000, seed=3):
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    close = 30000 + 900 * np.sin(2 * np.pi * i / 300) + np.cumsum(rng.normal(0, 9, n))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 12, n))
    t = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"t": t, "open_time": t, "open": open_,
                         "high": np.maximum(open_, close) + wick,
                         "low": np.minimum(open_, close) - wick, "close": close, "volume": 1.0})


# ---------------------------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------------------------
def test_the_core_reproduces_the_legacy_table_when_both_corrections_are_off():
    """The new engine and the legacy loop are the same algorithm once the corrections are off and
    nothing is evicted (eviction is the one rule that deliberately changed). If they disagree here,
    the core is wrong -- not merely different."""
    df = _frame()
    leg = sr_zone_table(df, 10, 10, cluster_pct=0.15, max_zones=100000)
    conf = _confirmed_pivots(df, 10, 10).to_numpy(float)
    zo, st, en, px, tc, *_ = _sr_core(conf, df["close"].to_numpy(float), 0.0015, 100000, 0, False)
    core = pd.DataFrame({"zone": zo, "start_i": st, "end_i": en, "price": px, "touches": tc})
    k = ["zone", "start_i", "end_i", "touches"]
    a = leg.sort_values(k).reset_index(drop=True)
    b = core.sort_values(k).reset_index(drop=True)
    assert len(a) == len(b) and (a[k].to_numpy() == b[k].to_numpy()).all()
    assert np.allclose(a["price"], b["price"])


def test_the_legacy_path_is_what_existing_callers_still_get():
    """Nothing that called these before passes the new keywords, so nothing may change for it."""
    df = _frame()
    assert "lo" not in sr_zone_table(df).columns, "the default must take the ORIGINAL code path"
    a = add_sr_zones(df)
    b = add_sr_zones(df, break_bars=0, anchor=False)
    for c in ("sr_support", "sr_resistance", "sr_support_touches", "sr_resistance_touches"):
        assert a[c].equals(b[c])


def test_an_anchored_zone_is_never_wider_than_cluster_pct():
    """The legacy test compared a new pivot against the zone's RUNNING MEAN, which moves as pivots
    join -- measured on BTC 5m at ~63k, 0.15% is ~94 points and zones spanned 149-157."""
    df = _frame(6000)
    wide = sr_zone_table(df, 10, 10, cluster_pct=0.15, max_zones=40, break_bars=10**9)
    tight = sr_zone_table(df, 10, 10, cluster_pct=0.15, max_zones=40, anchor=True)
    w0 = ((wide["hi"] - wide["lo"]) / wide["price"] * 100).max()
    w1 = ((tight["hi"] - tight["lo"]) / tight["price"] * 100).max()
    assert w0 > 0.15, "the fixture must show the drift, or this test proves nothing"
    assert w1 <= 0.15 + 1e-9, f"an anchored zone spans {w1:.3f}% against a 0.15% limit"


def _cross(side_first: float, n_beyond: int, reclaim_after: int | None = None):
    """A zone at 100 made of two pivots, price above it, then `n_beyond` closes below it -- with an
    optional close back above after `reclaim_after` of them."""
    closes = [110.0] * 5 + [100.0] + [110.0] * 5 + [100.0] + [110.0] * 5
    pivots = [np.nan] * len(closes)
    pivots[5], pivots[11] = 100.0, 100.0
    seq = [90.0] * n_beyond
    if reclaim_after is not None:
        seq = [90.0] * reclaim_after + [110.0] + [90.0] * (n_beyond - reclaim_after)
    closes += seq + [90.0] * 3
    pivots += [np.nan] * (len(seq) + 3)
    return np.asarray(pivots, float), np.asarray(closes, float), len(closes)


def test_a_zone_price_breaks_and_holds_beyond_is_retired():
    piv, c, n = _cross(110.0, 4)
    zo, st, en, *_ = _sr_core(piv, c, 0.0015, 40, 3, True)
    assert en.max() < n - 1, "a zone broken and held beyond for 3 closes must be retired"
    zo0, st0, en0, *_ = _sr_core(piv, c, 0.0015, 40, 0, True)
    assert en0.max() == n - 1, "and with break_bars=0 it must live to the end, as before"


def test_a_close_back_on_the_original_side_resets_the_count():
    """A wick-and-reclaim is not a break. Without the reset, 'three closes beyond' would quietly
    mean 'three closes beyond at any time', which a chopping market satisfies by accident."""
    piv, c, n = _cross(110.0, 4, reclaim_after=2)
    zo, st, en, *_ = _sr_core(piv, c, 0.0015, 40, 3, True)
    # closes: 90 90 (beyond) | 110 (reclaim, index 19) | 90 90 90 ...
    #   with the reset    -> count restarts at 20, reaches 3 at index 22
    #   without it        -> 17, 18, 20 are three closes beyond, retired at index 20
    assert en[zo == zo[0]].max() == 22, "the reclaim must RESET the count, not pause it"


def test_eviction_drops_the_zone_farthest_from_price():
    """The legacy rule kept the most-touched zones, so old strong levels held every slot and each
    new zone next to price was evicted as it formed."""
    far = [1000.0, 1000.0, 1000.0, 1000.0]                 # a strong zone, far from price
    near = [100.0, 100.6]                                   # two separate zones near price
    piv = np.array(far + [np.nan] + near + [np.nan] * 3, float)
    c = np.array([100.0] * len(piv), float)
    zo, st, en, px, tc, *_ = _sr_core(piv, c, 0.0001, 2, 0, True)
    live_at_end = set(np.round(px[en == len(c) - 1], 2))
    assert live_at_end == {100.0, 100.6}, f"kept {live_at_end}: the far zone should have gone"


def test_zone_ids_are_unique_even_under_eviction():
    """The legacy table used the slot index as the id, so an evicted zone and its replacement
    shared one, and the chart's one-line-per-zone step silently dropped the evicted zone."""
    df = _frame(6000)
    t = sr_zone_table(df, 5, 5, cluster_pct=0.05, max_zones=8, anchor=True)
    assert len(t["zone"].unique()) > 8, "the fixture must force evictions"
    # Contiguity alone would NOT catch slot reuse -- the evicted zone ends at i-1 and its
    # replacement starts at i. What gives it away is the PRICE: a revision nudges a zone within
    # its band, a different zone under the same id jumps.
    for z, g in t.sort_values("start_i").groupby("zone"):
        px = g["price"].to_numpy()
        assert (np.abs(np.diff(px)) <= 0.0005 * px[1:] + 1e-9).all(),             f"zone {z} jumps -- two different zones are sharing one id"


def test_the_strategy_path_and_the_chart_path_agree():
    """ONE engine behind both, so the level a strategy reads is the level the chart draws. The
    legacy pair were documented as mirrors but evicted with different tie-breaks."""
    df = _frame(4000)
    kw = dict(cluster_pct=0.15, max_zones=12, break_bars=3, anchor=True)
    z = add_sr_zones(df, 10, 10, **kw)
    t = sr_zone_table(df, 10, 10, **kw)
    c = df["close"].to_numpy()
    for i in range(200, len(df), 97):
        live = t[(t.start_i <= i) & (t.end_i >= i)]
        below = live.price[live.price <= c[i]]
        want = below.max() if len(below) else np.nan
        got = z["sr_support"].iloc[i]
        assert (np.isnan(want) and np.isnan(got)) or np.isclose(want, got), i


# ---------------------------------------------------------------------------------------------
# the knob
# ---------------------------------------------------------------------------------------------
def test_every_tuned_timeframe_has_all_three_levels_and_every_key():
    keys = {"left", "right", "cluster_pct", "min_touches", "break_bars", "anchor"}
    for tf, levels in SR_TUNED.items():
        assert set(levels) == set(SR_LEVELS), tf
        for lvl, prm in levels.items():
            assert set(prm) == keys, (tf, lvl)


def test_none_is_legacy_and_an_explicit_setting_always_wins():
    leg = sr_params("5m", None)
    assert leg["break_bars"] == 0 and leg["anchor"] is False, "None must reach the original path"
    assert sr_params("1h", "tight", cluster_pct=0.3)["cluster_pct"] == 0.3
    assert sr_params("1h", "tight", cluster_pct=None) == sr_params("1h", "tight"), \
        "None means 'not stated', and must never blank a tuned value"
    with pytest.raises(ValueError):
        sr_params("5m", "medium")


def test_an_untuned_timeframe_borrows_the_nearest_tuned_one():
    assert _nearest_tuned_tf("1m") == "5m"
    assert _nearest_tuned_tf("2h") in ("1h", "4h")
    assert _nearest_tuned_tf("1d") == "4h"


def test_the_tuned_table_is_exactly_what_the_study_chose():
    """"Do not hand-edit without re-running the study" is a comment until a test makes it true. The
    table and its evidence file must agree to the value, so an edit to one without the other fails
    here instead of quietly detaching a number from the measurement behind it."""
    import pathlib

    f = pathlib.Path(__file__).parent.parent / "experiments" / "results_sr_zones_picks.csv"
    pk = pd.read_csv(f)
    for r in pk.itertuples():
        want = dict(left=r.lr, right=r.lr, cluster_pct=r.cluster, min_touches=r.min_t,
                    break_bars=r.break_bars, anchor=True)
        got = SR_TUNED[r.tf][r.level]
        assert got == want, f"{r.tf}/{r.level}: table {got} != study {want}"


def test_the_knob_turns_clutter_up_in_order_and_normal_beats_the_old_defaults():
    """What the knob MEANS, checked against the measurement: tight < normal < loose in lines drawn,
    and "normal" draws fewer than the settings it replaced on every timeframe."""
    import pathlib

    pk = pd.read_csv(pathlib.Path(__file__).parent.parent / "experiments"
                     / "results_sr_zones_picks.csv")
    for tf, g in pk.groupby("tf"):
        n = g.set_index("level")["lines"]
        assert n["tight"] < n["normal"] < n["loose"], f"{tf}: {dict(n)}"
        assert n["normal"] < g["legacy_lines"].iloc[0], f"{tf}: normal is not less cluttered"
