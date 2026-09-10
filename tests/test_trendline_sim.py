"""Multi-timeframe trendline entries, and the trade overlay that draws them.

The rule is "a fitted line from ANY timeframe gives way". Two things have to hold for that to mean
anything: a higher timeframe's level must not be visible before its bar closed, and the same
confirmation rule must behave the same way on every timeframe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _frame(n=3600, seed=7, freq="1min"):
    """A trend that swings, then reverses hard — geometry a trendline fitter can actually find.

    A random walk is the wrong fixture here. At this length it grew a line on two of six
    (timeframe x side) combinations and never once put a whole candle past one, so the properties
    under test were mostly untested. This is a rising channel with a dozen clean swings, broken by
    a sharp decline, which gives every timeframe both a support and a resistance line and produces
    breaks of both kinds.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    turn = int(n * 0.7)
    trend = np.where(i < turn, 30000 + 2.2 * i, 30000 + 2.2 * turn - 6.0 * (i - turn))
    swing = 260 * np.sin(2 * np.pi * i / 240)
    close = trend + swing + rng.normal(0, 12, n)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, 18, n))
    hi = np.maximum(open_, close) + wick
    lo = np.minimum(open_, close) - wick
    t = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t, "open": open_, "high": hi, "low": lo,
                         "close": close, "volume": 1.0})


@pytest.fixture(scope="module")
def mtf():
    from quant.data import resample_ohlcv
    from quant.indicators.trendlines import mtf_trendlines

    base = _frame()
    frames = {"5min": resample_ohlcv(base, "5min"), "15min": resample_ohlcv(base, "15min")}
    return base, frames, mtf_trendlines(base, frames, min_span_bars=20, max_age_bars=400)


def test_mtf_trendlines_puts_every_timeframes_level_on_the_base_grid(mtf):
    base, frames, D = mtf
    assert len(D) == len(base)
    for col in ("tl_res", "tl_sup", "5min__tl_res", "15min__tl_sup"):
        assert col in D.columns
    assert np.isfinite(D["tl_res"].to_numpy(float)).any(), "no base-timeframe line at all"


def test_a_higher_timeframe_level_is_invisible_until_its_bar_closed(mtf):
    """The whole point of the merge. A 15m line read off the bar stamped 12:00 is knowable at 12:15.

    Checked by truncation, the same test the detectors get: rebuild everything from the first k bars
    only, and every value that was already published must be unchanged. If any higher-timeframe
    level leaked backwards, the truncated run would disagree.
    """
    from quant.data import resample_ohlcv
    from quant.indicators.trendlines import mtf_trendlines

    base, frames, D = mtf
    k = 1800
    part = mtf_trendlines(base.iloc[:k].reset_index(drop=True),
                          {tf: f[pd.to_datetime(f["t"]) < base["t"].iloc[k]].reset_index(drop=True)
                           for tf, f in frames.items()},
                          min_span_bars=20, max_age_bars=400)
    for col in ("5min__tl_res", "5min__tl_sup", "15min__tl_res", "15min__tl_sup"):
        a = D[col].to_numpy(float)[:k]
        b = part[col].to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both]), f"{col} changed when the future was removed"


def test_trendline_level_cols_lists_the_base_frame_first_then_coarsest(mtf):
    from quant.indicators.trendlines import trendline_level_cols

    _, _, D = mtf
    assert trendline_level_cols(D, "long") == ["tl_res", "15min__tl_res", "5min__tl_res"]
    assert trendline_level_cols(D, "short") == ["tl_sup", "15min__tl_sup", "5min__tl_sup"]


def test_the_break_rule_behaves_the_same_on_every_timeframe(mtf):
    """A line is RETIRED the moment it breaks, so a second confirmation bar would be comparing
    itself against a level that no longer exists. Measured before the fix: the base timeframe fired
    ZERO times at n=2 while the merged higher ones, whose values persist across their own bar,
    fired hundreds. The entry holds the last published value for the length of the confirmation
    window, which is a past value and so still causal.
    """
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    tested = 0
    for pref in ("", "5min__", "15min__"):
        for side in ("long", "short"):
            base = dict(tl_prefixes=(pref,), tl_full_m=0, tl_touch_k=0, tl_combine="all")
            one = _tl_entry(dict(base, tl_close_n=1), side)(D).sum()
            two = _tl_entry(dict(base, tl_close_n=2), side)(D).sum()
            if not one:
                continue          # this synthetic frame grew no line of that kind — nothing to test
            tested += 1
            assert two > 0, f"{pref or 'base'} {side} fires at n=1 but not n=2 — the level is not held"
            assert two <= one, "asking for more confirmation cannot produce MORE entries"
    assert tested >= 2, "too few live lines to have tested anything"


def test_the_level_is_held_across_the_confirmation_window():
    """The exact bug, on a hand-built frame so no fitter is involved.

    `add_trendlines` RETIRES a line the moment price closes through it, so the level column goes NaN
    at the break bar. Asking for two closes beyond the line then asks bar 2 to compare itself
    against nothing, and `NaN` comparisons are False — so the rule fired ZERO times on the base
    timeframe while the merged higher ones, whose values persist across their own bar, fired
    hundreds. Holding the last published value for the length of the window fixes it and stays
    causal, because the held value is one the frame had already published.
    """
    from quant.scenarios.entries import _tl_entry

    n = 12
    close = np.array([100.0] * 6 + [105.0] * 6)      # breaks up at bar 6 and stays there
    lvl = np.array([101.0] * 7 + [np.nan] * 5)       # ...and the line retires the moment it does
    d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
                      "open": close, "high": close, "low": close, "close": close,
                      "volume": 1.0, "tl_res": lvl})
    p = dict(tl_prefixes=("",), tl_full_m=0, tl_touch_k=0, tl_combine="all")

    one = _tl_entry(dict(p, tl_close_n=1), "long")(d)
    assert one.sum() == 1 and one[6], "the break bar itself must fire at n=1"

    two = _tl_entry(dict(p, tl_close_n=2), "long")(d)
    assert two[7], "the second close beyond a line that just retired must still confirm"

    # the hold is exactly as long as the window asks for, never longer
    assert not _tl_entry(dict(p, tl_close_n=2), "long")(
        d.assign(close=np.r_[close[:7], [100.0] * 5])).any(), "price came back — nothing should fire"


def test_long_takes_resistance_and_short_takes_support(mtf):
    """A long breaks UP through resistance; a short breaks DOWN through support. Never the reverse."""
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=1, tl_full_m=0, tl_touch_k=0, tl_combine="all")
    for side, w, cmp in (("long", "res", np.greater), ("short", "sup", np.less)):
        fired = _tl_entry(p, side)(D)
        assert fired.any()
        c = D["close"].to_numpy(float)
        beyond = np.zeros(len(D), bool)
        for pref in ("", "5min__"):
            lvl = D[f"{pref}tl_{w}"].to_numpy(float)
            beyond |= np.isfinite(lvl) & cmp(c, lvl)
        assert (fired & ~beyond).sum() == 0, f"{side} fired without price being through a {w} line"


def test_a_full_candle_beyond_is_stricter_than_a_close_beyond(mtf):
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    seen = 0
    for side in ("long", "short"):
        base = dict(tl_prefixes=("", "5min__", "15min__"), tl_touch_k=0, tl_combine="all")
        closes = _tl_entry(dict(base, tl_close_n=1, tl_full_m=0), side)(D).sum()
        fulls = _tl_entry(dict(base, tl_close_n=0, tl_full_m=1), side)(D).sum()
        assert fulls <= closes, "a whole candle past the line cannot be commoner than a close past it"
        seen += int(fulls > 0)
    assert seen, "no side ever put a whole candle past a line, so nothing was compared"


def test_the_scenarios_run_both_ways_with_the_stop_on_the_right_side(mtf):
    from quant.scenarios import run_pieces

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__", "15min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    cfg = dict(initial_cash=10_000, exit_enabled=False, sizing_mode="risk_pct_equity",
               sizing_value=1.0, max_open_trades=1)
    for side in ("long", "short"):
        run = run_pieces(D, f"mtf_trendline_{side}@v1", "high_of_n@v1", "be_ladder@v1",
                         params=p, cfg=cfg)
        tr = run.trades
        assert len(tr), f"{side} produced no trades"
        assert (tr["side"] == side).all()
        ok = (tr.stop_price < tr.entry_price) if side == "long" else (tr.stop_price > tr.entry_price)
        assert ok.all(), f"a {side}'s stop must sit on the losing side of entry"


def test_each_trade_is_labelled_with_the_timeframe_that_fired(mtf):
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__", "15min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    run = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=dict(initial_cash=10_000, exit_enabled=False))
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    out = trendline_trigger(run.df, run.trades, p, prefix_names=names)
    assert "trigger" in out.columns and len(out) == len(run.trades)
    assert (out["trigger"] != "?").all(), "a trade fired without any timeframe claiming it"
    assert set("+".join(out["trigger"]).split("+")) <= set(names.values())


def test_the_chart_draws_entries_stops_targets_and_exits(mtf):
    from quant.scenarios import run_pieces
    from quant.viz.studies import pattern_figure

    _, frames, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    tr = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                    params=p, cfg=dict(initial_cash=10_000, exit_enabled=False)).trades
    assert len(tr)

    bare = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=False)
    fig = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=False,
                         trades=tr, trade_tp_rr=3.0)
    names = [t.name for t in fig.data if t.name]
    assert "trade stop" in names and "trade target" in names
    assert any(n.startswith("short entries") for n in names)
    assert any(n.startswith("exits") for n in names)
    # the overlay is additive: the chart underneath is untouched
    assert len(fig.data) > len(bare.data)

    tgt = [t for t in fig.data if t.name == "trade target"][0]
    stp = [t for t in fig.data if t.name == "trade stop"][0]
    ys, yt = np.array(stp.y, float), np.array(tgt.y, float)
    m = np.isfinite(ys) & np.isfinite(yt)
    assert (ys[m] > yt[m]).all(), "a short's stop must be drawn above its target"


# ---------------------------------------------------------------------------
# IDs: one name per line, per pattern and per trade, used by the hover, the legend, the dropdown
# and the trade that broke it — all derived in one place so they cannot drift apart.
# ---------------------------------------------------------------------------
def test_every_line_gets_a_stable_name(mtf):
    from quant.indicators.trendlines import trendline_table

    base, frames, _ = mtf
    tbl = trendline_table(frames["5min"], min_span_bars=20, max_age_bars=400)
    assert len(tbl) and "id" in tbl.columns
    assert tbl["id"].is_unique
    # R for resistance and S for support, numbered from zero within each kind
    for kind, letter in ((+1, "R"), (-1, "S")):
        got = list(tbl.loc[tbl["kind"] == kind, "id"])
        assert got == [f"{letter}{i}" for i in range(len(got))]


def test_the_per_bar_id_column_agrees_with_the_table(mtf):
    """`tl_res_n` on a bar must be the number of the line whose level `tl_res` reports there."""
    from quant.indicators.trendlines import add_trendline_ids, add_trendlines, trendline_table

    base, _, _ = mtf
    d = add_trendline_ids(add_trendlines(base, min_span_bars=20, max_age_bars=400))
    tbl = trendline_table(d)
    for kind, nm in ((+1, "res"), (-1, "sup")):
        rows = tbl[tbl["kind"] == kind].reset_index(drop=True)
        ids = d[f"tl_{nm}_n"].to_numpy()
        for k, r in rows.iterrows():
            live = ids[int(r.born_i):int(r.end_i) + 1]
            assert (live == k).all(), f"{r.id} does not own its own bars"
        assert (ids[np.isfinite(d[f"tl_{nm}"].to_numpy(float)) & (ids < 0)]).size == 0


def test_a_trade_names_the_exact_line_it_broke(mtf):
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__", "15min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    run = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=dict(initial_cash=10_000, exit_enabled=False))
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    out = trendline_trigger(run.df, run.trades, p, prefix_names=names)
    assert len(out) and "cause" in out.columns
    # never "?" — the level is held across the confirmation window and so is the id beside it
    assert (out["cause"] != "?").all(), "a trade could not name the line it broke"
    for c in out["cause"]:
        for part in c.split("+"):
            tf, ident = part.split("·")
            assert tf in names.values() and ident[0] == "S" and ident[1:].isdigit()


def test_the_chart_can_isolate_one_trade_together_with_its_line(mtf):
    """The question the trade picker exists to answer: show me #23 and what caused it."""
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger
    from quant.viz.studies import pattern_figure

    _, frames, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    run = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=dict(initial_cash=10_000, exit_enabled=False))
    tr = trendline_trigger(run.df, run.trades, p,
                           prefix_names={"": "1m", "5min__": "5m"})
    risk = (tr.entry_price - tr.stop_price).abs()
    tr = tr.assign(R=tr.pnl / (risk * tr.qty))

    fig = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=True,
                         tf_frames=frames, chart_tf="1m", trades=tr, trade_tp_rr=3.0,
                         max_picker_instances=6, max_shapes_per_tf=4)

    rows = [c for b in fig.layout.updatemenus[-2].buttons
            for c in (b.args[1] or {}).get("updatemenus[4].buttons", [])
            if c["label"].startswith("#")]
    assert rows, "no trade rows in the picker"
    for c in rows:                                  # the label carries id, side, line, R, outcome
        assert "·" in c["label"] and "R " in c["label"]

    seen_with_line = 0
    for c in rows:
        vis = c["args"][0]["visible"]
        on = [i for i, v in enumerate(vis) if v is True]
        # price, the trade's FOUR traces (entry / stop / target / markers, one colour each) and,
        # when its line survived the caps, that line's two
        assert 5 <= len(on) <= 7
        names = [str(getattr(fig.data[i], "hovertemplate", None)
                     or getattr(fig.data[i], "name", None) or "") for i in on]
        # the three levels are separate traces now, each naming itself
        assert sum(any(w in nm.upper() for w in ("ENTRY", "STOP", "TARGET")) for nm in names) >= 3
        seen_with_line += len(on) == 7
    assert seen_with_line, "no trade was shown together with the line it broke"


def test_a_line_a_trade_broke_survives_the_caps(mtf):
    """The caps keep the chart openable, but never at the price of hiding a traded line."""
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger
    from quant.viz.studies import pattern_figure

    _, frames, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    run = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=dict(initial_cash=10_000, exit_enabled=False))
    tr = trendline_trigger(run.df, run.trades, p, prefix_names={"": "1m", "5min__": "5m"})
    tr = tr.assign(R=0.0)

    # caps as tight as they go
    fig = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=True,
                         tf_frames=frames, chart_tf="1m", trades=tr, trade_tp_rr=3.0,
                         max_picker_instances=1, max_shapes_per_tf=1)
    drawn = set()
    for t_ in fig.data:
        for cd in (t_.customdata if t_.customdata is not None else []):
            v = cd if isinstance(cd, str) else (cd[0] if len(cd) else "")
            if isinstance(v, str) and "·" in v:
                drawn.add(v)
    wanted = {x for c in tr["cause"] for x in str(c).split("+") if x != "?"}
    assert wanted and wanted <= drawn, f"the caps hid a traded line: {sorted(wanted - drawn)}"


# ---------------------------------------------------------------------------
# Per-timeframe wait, and telling the three trade levels apart on the chart
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec,pref,want", [
    (4, "5min__", 4),                                   # a plain number applies everywhere
    ({"5m": 2, "15m": 3}, "5min__", 2),                 # ...looked up by timeframe NAME
    ({"5m": 2, "15m": 3}, "15min__", 3),
    ({"5min__": 7}, "5min__", 7),                       # ...or by raw column prefix
    ({"5m": 2, "*": 9}, "1h__", 9),                     # "*" is the fallback
    ({"5m": 2}, "1h__", 0),                             # ...and without one, off
])
def test_a_wait_can_be_set_per_timeframe(spec, pref, want):
    from quant.scenarios.entries import tl_per_tf

    p = dict(tl_tf_names={"": "1m", "5min__": "5m", "15min__": "15m", "1h__": "1h"},
             tl_close_n=spec)
    assert tl_per_tf(p, "tl_close_n", pref) == want


def test_each_timeframe_waits_its_own_number_of_candles(mtf):
    """Two 15m closes is half an hour and two 1m closes is two minutes, so one number cannot fit."""
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    base = dict(tl_prefixes=("", "5min__"), tl_tf_names=names, tl_full_m=0, tl_touch_k=0,
                tl_combine="all")

    fast = _tl_entry(dict(base, tl_close_n={"1m": 1, "5m": 1}), "long")(D).sum()
    slow = _tl_entry(dict(base, tl_close_n={"1m": 1, "5m": 4}), "long")(D).sum()
    assert 0 < slow < fast, "asking 5m to wait longer must cut its entries and nothing else's"

    # ...and it really is only 5m that changed
    only_1m = dict(base, tl_prefixes=("",))
    a = _tl_entry(dict(only_1m, tl_close_n={"1m": 1, "5m": 1}), "long")(D).sum()
    b = _tl_entry(dict(only_1m, tl_close_n={"1m": 1, "5m": 4}), "long")(D).sum()
    assert a == b


def test_the_three_trade_levels_are_drawn_in_three_colours(mtf):
    """A losing trade used to draw entry, stop and target as one red trace.

    Two red lines with nothing to tell them apart, and the reader reasonably concluded the stop was
    on the wrong side of the entry. Each level is its own trace in its own colour now, matching the
    grouped view: grey entry, red stop, green target.
    """
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger
    from quant.viz.studies import pattern_figure

    _, frames, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    run = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=dict(initial_cash=10_000, exit_enabled=False))
    tr = trendline_trigger(run.df, run.trades, p, prefix_names={"": "1m", "5min__": "5m"})
    tr = tr.assign(R=0.0)
    fig = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=True,
                         tf_frames=frames, chart_tf="1m", trades=tr, trade_tp_rr=3.0,
                         max_picker_instances=6, max_shapes_per_tf=4)

    rows = [c for b in fig.layout.updatemenus[-2].buttons
            for c in (b.args[1] or {}).get("updatemenus[4].buttons", [])
            if c["label"].startswith("#")]
    assert rows
    for c in rows:
        on = [i for i, v in enumerate(c["args"][0]["visible"]) if v is True]
        levels = [fig.data[i] for i in on
                  if "ENTRY" in str(getattr(fig.data[i], "hovertemplate", "")).upper()
                  or "STOP" in str(getattr(fig.data[i], "hovertemplate", "")).upper()
                  or "TARGET" in str(getattr(fig.data[i], "hovertemplate", "")).upper()]
        assert len(levels) == 3, "entry, stop and target must each be their own trace"
        assert len({lv.line.color for lv in levels}) == 3, "and each its own colour"
        by = {("stop" if "STOP" in str(lv.hovertemplate).upper() else
               "target" if "TARGET" in str(lv.hovertemplate).upper() else "entry"): lv
              for lv in levels}
        assert by["stop"].line.color == "#ef5350" and by["target"].line.color == "#26a69a"


def test_a_stop_that_walked_to_breakeven_says_so(mtf):
    """Otherwise the exit sits at the entry level, nowhere near the red line, and reads as a bug."""
    import numpy as np
    from quant.viz.studies import pattern_figure

    _, _, D = mtf
    t = pd.to_datetime(D["t"])
    # one short whose stop was hit a hair from ENTRY, not from the stop it was given
    tr = pd.DataFrame([dict(side="short", entry_time=t.iloc[100], exit_time=t.iloc[200],
                            entry_i=100, exit_i=200, entry_price=100.0, exit_price=100.2,
                            stop_price=110.0, qty=1.0, pnl=-0.2, fees=0.0, bars_held=100,
                            close_reason="stop_loss", R=-0.02, cause="1m·S0")])
    fig = pattern_figure(D, patterns=[], sr_zones=False, emas=(), trendlines=False,
                         chart_tf="1m", trades=tr, trade_tp_rr=3.0)
    rows = [c for b in fig.layout.updatemenus[-2].buttons
            for c in (b.args[1] or {}).get("updatemenus[4].buttons", [])
            if c["label"].startswith("#")]
    assert rows and all("sl@be" in c["label"] for c in rows)
    txt = " ".join(str(t_.text) for t_ in fig.data if getattr(t_, "text", None) is not None)
    assert "breakeven" in txt


def test_a_per_timeframe_wait_survives_a_whole_run_and_a_preset(mtf):
    """The dict has to reach the kernel through `run_pieces`, and through Preset JSON."""
    import json

    from quant.scenarios import run_pieces
    from quant.scenarios.base import Preset

    _, _, D = mtf
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    p = dict(tl_prefixes=tuple(names), tl_tf_names=names,
             tl_close_n={"1m": 2, "5m": 4, "*": 3}, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0)
    cfg = dict(initial_cash=10_000, exit_enabled=False)

    run = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                     params=p, cfg=cfg)
    assert len(run.trades), "a dict wait produced no trades at all"

    # ...and asking every timeframe to wait longer must not produce MORE of them
    slower = dict(p, tl_close_n={"1m": 5, "5m": 6, "*": 6})
    assert len(run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                          params=slower, cfg=cfg).trades) <= len(run.trades)

    # a Preset is JSON, and a dict param has to come back a dict rather than a string
    ps = Preset(name="t", scenario="mtf_trendline_short@v1", stop="high_of_n@v1",
                tp="be_ladder@v1", params=p, cfg=cfg)
    back = Preset.from_dict(json.loads(json.dumps(ps.to_dict())))
    assert back.params["tl_close_n"] == {"1m": 2, "5m": 4, "*": 3}


# ---------------------------------------------------------------------------
# Candle COLOUR as an entry condition — a different question from "closed beyond the line"
# ---------------------------------------------------------------------------
def test_candle_runs_count_consecutive_colours_and_a_doji_ends_both():
    from quant.indicators.candles import add_candle_runs

    d = pd.DataFrame({"open": [1, 1, 1, 2, 1, 1, 1], "close": [2, 2, 0, 2, 2, 2, 0]})
    r = add_candle_runs(d)                       # G G R - G G R
    assert list(r["up_run"]) == [1, 2, 0, 0, 1, 2, 0]
    assert list(r["dn_run"]) == [0, 0, 1, 0, 0, 0, 1]


def test_each_timeframes_runs_are_counted_on_its_own_candles(mtf):
    """Counted on the 15m frame and merged — on a 5m grid a run of 15m reds would otherwise be
    measured in 5m bars, which is a different and much easier condition."""
    from quant.indicators.candles import add_candle_runs

    base, frames, D = mtf
    for pref, f in (("", base), ("5min__", frames["5min"]), ("15min__", frames["15min"])):
        assert f"{pref}up_run" in D.columns and f"{pref}dn_run" in D.columns
    # the 15m run can only be as long as the 15m frame's own run ever gets
    own = add_candle_runs(frames["15min"])
    assert D["15min__dn_run"].max() <= own["dn_run"].max()
    # ...and it is strictly coarser than the base frame's, because it changes once per 15m bar
    assert D["15min__dn_run"].diff().ne(0).sum() < D["dn_run"].diff().ne(0).sum()


def test_a_higher_timeframes_run_is_invisible_until_that_bar_closed(mtf):
    """Same anti-lookahead rule the levels obey: the run you can act on is the last CLOSED one."""
    from quant.data import resample_ohlcv
    from quant.indicators.candles import add_candle_runs
    from quant.indicators.trendlines import mtf_trendlines

    base, frames, _ = mtf
    k = 1800
    part = mtf_trendlines(base.iloc[:k].reset_index(drop=True),
                          {tf: f[pd.to_datetime(f["t"]) < base["t"].iloc[k]].reset_index(drop=True)
                           for tf, f in frames.items()},
                          min_span_bars=20, max_age_bars=400)
    full = mtf_trendlines(base, frames, min_span_bars=20, max_age_bars=400)
    for c in ("5min__up_run", "15min__dn_run"):
        assert (full[c].to_numpy()[:k] == part[c].to_numpy()).all(), \
            f"{c} changed when the future was removed"


def test_asking_for_n_candles_of_the_right_colour_only_removes_entries(mtf):
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    base = dict(tl_prefixes=tuple(names), tl_tf_names=names, tl_close_n=2, tl_full_m=0,
                tl_touch_k=0, tl_combine="all")
    counts = [_tl_entry(dict(base, tl_dir_n=n), "short")(D).sum() for n in (0, 1, 2, 3)]
    assert counts[0] > 0
    assert counts == sorted(counts, reverse=True), "more candles demanded cannot mean more entries"
    assert counts[-1] < counts[0], "the colour test did nothing at all"


def test_the_colour_can_be_read_off_the_line_s_timeframe_or_the_chart_s(mtf):
    """`src='line'` asks for three red 15m candles; `src='chart'` for three red execution bars."""
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    base = dict(tl_prefixes=("5min__", "15min__"), tl_tf_names=names, tl_close_n=2,
                tl_full_m=0, tl_touch_k=0, tl_combine="all", tl_dir_n=3)
    chart = _tl_entry(dict(base, tl_dir_src="chart"), "short")(D).sum()
    line = _tl_entry(dict(base, tl_dir_src="line"), "short")(D).sum()
    assert chart != line, "the two sources cannot be the same condition"
    # a run of the LINE's own candles is a slower clock, so it is the looser of the two
    assert line >= chart


def test_the_colour_count_can_differ_per_timeframe(mtf):
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    base = dict(tl_prefixes=("", "5min__"), tl_tf_names=names, tl_close_n=1, tl_full_m=0,
                tl_touch_k=0, tl_combine="all")
    loose = _tl_entry(dict(base, tl_dir_n={"1m": 1, "5m": 1}), "short")(D).sum()
    tight = _tl_entry(dict(base, tl_dir_n={"1m": 1, "5m": 4}), "short")(D).sum()
    assert 0 < tight < loose


def test_a_green_close_below_support_passes_the_close_rule_and_fails_the_colour_rule():
    """The whole reason the two conditions are separate."""
    from quant.scenarios.entries import _tl_entry

    n = 10
    # every bar closes below the line, but they are GREEN (close above their own open)
    d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
                      "open": 90.0, "close": 95.0, "high": 96.0, "low": 89.0, "volume": 1.0,
                      "tl_sup": 100.0})
    from quant.indicators.candles import add_candle_runs
    d = add_candle_runs(d)
    p = dict(tl_prefixes=("",), tl_close_n=2, tl_full_m=0, tl_touch_k=0, tl_combine="all")

    assert _tl_entry(p, "short")(d).any(), "two closes below the line should fire"
    assert not _tl_entry(dict(p, tl_dir_n=2), "short")(d).any(), \
        "...but not once two RED candles are also required"
