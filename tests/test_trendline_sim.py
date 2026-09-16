"""Multi-timeframe trendline entries, and the trade overlay that draws them.

The rule is "a fitted line from ANY timeframe gives way". Two things have to hold for that to mean
anything: a higher timeframe's level must not be visible before its bar closed, and the same
confirmation rule must behave the same way on every timeframe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest



def _layout_patch(button):
    """A button's LAYOUT patch, or {}.

    `method="update"` puts it in args[1]; `method="restyle"` puts TRACE INDICES there instead (the
    EMAs on/off control does), so this cannot just index and call .get.
    """
    args = list(getattr(button, "args", None) or ())
    return args[1] if len(args) > 1 and isinstance(args[1], dict) else {}


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

    rows = [c for m in fig.layout.updatemenus for b in (m.buttons or ())
            for c in _layout_patch(b).get("updatemenus[4].buttons", [])
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

    rows = [c for m in fig.layout.updatemenus for b in (m.buttons or ())
            for c in _layout_patch(b).get("updatemenus[4].buttons", [])
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
    rows = [c for m in fig.layout.updatemenus for b in (m.buttons or ())
            for c in _layout_patch(b).get("updatemenus[4].buttons", [])
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


# ---------------------------------------------------------------------------
# n candles of the right colour, FULLY clear of a line that has been extended
# ---------------------------------------------------------------------------
def test_a_held_line_is_carried_along_its_own_slope():
    """A rising support keeps rising. Holding the last value flat quietly makes the test easier on
    one side and harder on the other, over exactly the bars a confirmation rule spans."""
    from quant.scenarios.entries import _held_level

    d = pd.DataFrame({"tl_sup": [100.0, 101.0, 102.0, np.nan, np.nan, np.nan],
                      "tl_sup_slope": [1.0, 1.0, 1.0, np.nan, np.nan, np.nan]})
    flat = _held_level(d, "tl_sup", 3)
    ext = _held_level(d, "tl_sup", 3, slope_col="tl_sup_slope")
    assert list(flat) == [100, 101, 102, 102, 102, 102]
    assert list(ext) == [100, 101, 102, 103, 104, 105]
    # the hold is bounded, and a published value is never overwritten
    assert not np.isfinite(_held_level(d, "tl_sup", 1, slope_col="tl_sup_slope")[-1])
    assert list(ext[:3]) == [100, 101, 102]


def test_the_extension_only_ever_reads_values_already_published():
    """Causality by truncation, the same test the levels themselves get."""
    from quant.scenarios.entries import _held_level

    n = 60
    rng = np.random.default_rng(3)
    lvl = pd.Series(np.where(rng.random(n) < 0.4, rng.normal(100, 2, n), np.nan))
    d = pd.DataFrame({"tl_sup": lvl, "tl_sup_slope": 0.5})
    full = _held_level(d, "tl_sup", 4, slope_col="tl_sup_slope")
    for k in (20, 35, 50):
        part = _held_level(d.iloc[:k].reset_index(drop=True), "tl_sup", 4,
                           slope_col="tl_sup_slope")
        a, b = full[:k], part
        both = np.isfinite(a) & np.isfinite(b)
        assert np.allclose(a[both], b[both]), f"the extension changed when the future was removed"


def test_a_candle_must_be_both_the_right_colour_and_wholly_clear_of_the_line():
    from quant.scenarios.entries import _tl_entry
    from quant.indicators.candles import add_candle_runs

    def frame(open_, close, high, low):
        d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=len(close), freq="1min",
                                             tz="UTC"),
                          "open": open_, "close": close, "high": high, "low": low,
                          "volume": 1.0, "tl_sup": 100.0, "tl_sup_slope": 0.0})
        return add_candle_runs(d)

    p = dict(tl_prefixes=("",), tl_close_n=0, tl_full_m=0, tl_touch_k=0, tl_dir_n=0,
             tl_combine="all", tl_dir_full_n=2)

    # red, and the whole candle below 100 -> fires
    good = frame([98.0] * 4, [95.0] * 4, [99.0] * 4, [94.0] * 4)
    assert _tl_entry(p, "short")(good).any()

    # red, but the WICK pokes back above the line -> does not
    wick = frame([98.0] * 4, [95.0] * 4, [101.0] * 4, [94.0] * 4)
    assert not _tl_entry(p, "short")(wick).any()

    # wholly below, but GREEN -> does not
    green = frame([94.0] * 4, [97.0] * 4, [98.0] * 4, [93.0] * 4)
    assert not _tl_entry(p, "short")(green).any()


def test_the_conjoined_knob_matches_the_two_separate_ones_under_all_and_not_under_any(mtf):
    """Which is the reason it exists: as ONE condition it survives `tl_combine='any'`."""
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    base = dict(tl_prefixes=tuple(names), tl_tf_names=names, tl_close_n=2, tl_touch_k=0)

    joint = _tl_entry(dict(base, tl_dir_full_n=2, tl_combine="all"), "short")(D)
    split = _tl_entry(dict(base, tl_full_m=2, tl_dir_n=2, tl_combine="all"), "short")(D)
    assert (joint == split).all(), "under 'all' the two spellings are the same conjunction"

    joint_any = _tl_entry(dict(base, tl_dir_full_n=2, tl_combine="any"), "short")(D).sum()
    split_any = _tl_entry(dict(base, tl_full_m=2, tl_dir_n=2, tl_combine="any"), "short")(D).sum()
    assert joint_any < split_any, "under 'any' they must differ — that is the point"


def test_demanding_more_such_candles_only_removes_entries(mtf):
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    names = {"": "1m", "5min__": "5m", "15min__": "15m"}
    base = dict(tl_prefixes=tuple(names), tl_tf_names=names, tl_close_n=2, tl_full_m=0,
                tl_touch_k=0, tl_dir_n=0, tl_combine="all")
    counts = [_tl_entry(dict(base, tl_dir_full_n=n), "short")(D).sum() for n in (0, 1, 2, 3)]
    assert counts[0] > 0 and counts == sorted(counts, reverse=True) and counts[-1] < counts[0]
    # ...and it is per timeframe like every other knob
    a = _tl_entry(dict(base, tl_dir_full_n={"1m": 1, "5m": 1, "15m": 1}), "short")(D).sum()
    b = _tl_entry(dict(base, tl_dir_full_n={"1m": 1, "5m": 4, "15m": 4}), "short")(D).sum()
    assert b < a


# ---------------------------------------------------------------------------
# The trade arms ITSELF: price kept coming back to where it opened, so stop aiming
# at the target and take what is there. The level is the trade's OWN entry, which no
# global signal array can express — hence two per-trade counters in the kernel.
# ---------------------------------------------------------------------------
def _self_arm(highs, lows, closes, *, side="long", entry_bar=0, touch_n=0, wrong_m=0,
              alt_bars=(), min_pct=0.0, replace_tp=True):
    """One trade, hand-built bars, zero costs — so "in profit" is purely the price."""
    from quant.engine import BacktestConfig
    from quant.engine.config import Signals
    from quant.engine.run import run_backtest

    n = len(closes)
    d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
                      "open": np.asarray(closes, float), "high": np.asarray(highs, float),
                      "low": np.asarray(lows, float), "close": np.asarray(closes, float),
                      "volume": 1.0})
    z = np.zeros(n, bool)
    ent = z.copy()
    ent[entry_bar] = True
    alt = z.copy()
    for b in alt_bars:
        alt[b] = True
    kw = dict(entry_long=z.copy(), entry_short=z.copy(), exit_long=z.copy(), exit_short=z.copy(),
              arm_touch_n=touch_n, arm_wrong_m=wrong_m, alt_min_pct=min_pct,
              alt_replace_tp=replace_tp)
    if side == "long":
        kw.update(entry_long=ent, exit_arm_long=z.copy(), exit_alt_long=alt)
    else:
        kw.update(entry_short=ent, exit_arm_short=z.copy(), exit_alt_short=alt)
    cfg = BacktestConfig(initial_cash=10_000, exit_enabled=True, allow_short=(side == "short"),
                         sl_mode="entry_pct", sl_value=50.0,      # stop far away, never hit
                         sizing_mode="cash", cash_per_trade=1_000)
    return run_backtest(d, Signals(**kw), cfg).trades


def test_the_arming_counters_are_off_by_default_and_change_nothing():
    """Byte-identical when neither count is set — the whole block is skipped."""
    c = [100, 102, 101, 103, 100, 104, 99, 105]
    a = _self_arm(c, c, c, alt_bars=(5,), touch_n=0, wrong_m=0)
    b = _self_arm(c, c, c, alt_bars=(5,))
    assert a.to_dict() == b.to_dict()
    # nothing armed it, so the alt signal at bar 5 did nothing and the trade ran to the end
    assert len(a) == 1 and a.close_reason.iloc[0] == "forced_close_end"
    # ...and switching the counter on is what changes that
    on = _self_arm(c, c, c, alt_bars=(5,), touch_n=1)
    assert on.close_reason.iloc[0] == "alt_exit"


def test_a_trade_arms_once_price_has_come_back_to_its_entry_n_times():
    # long entered at bar 0 at 100; bars 2 and 4 trade back THROUGH 100
    hi = [100, 105, 101, 106, 101, 107]
    lo = [100, 103, 99, 104, 99, 105]
    cl = [100, 104, 100, 105, 100, 106]

    early = _self_arm(hi, lo, cl, touch_n=2, alt_bars=(3,))     # bar 3: one touch so far
    assert early.close_reason.iloc[0] != "alt_exit", "armed before the second touch"

    late = _self_arm(hi, lo, cl, touch_n=2, alt_bars=(5,))      # bar 5: two touches are in
    assert late.close_reason.iloc[0] == "alt_exit"
    assert late.pnl.iloc[0] > 0, "the in-profit gate must still hold"


def test_a_trade_arms_on_candles_formed_wholly_on_its_losing_side():
    # long at 100; bars 1 and 2 form ENTIRELY below it (high < 100), bar 3 recovers
    hi = [100, 99, 98, 104, 106]
    lo = [100, 97, 96, 100, 105]
    cl = [100, 98, 97, 103, 105]
    assert _self_arm(hi, lo, cl, wrong_m=3, alt_bars=(4,)).close_reason.iloc[0] != "alt_exit"
    assert _self_arm(hi, lo, cl, wrong_m=2, alt_bars=(4,)).close_reason.iloc[0] == "alt_exit"


def test_a_short_counts_its_losing_side_the_other_way():
    # short at 100; bars 1-2 form entirely ABOVE it, which is the losing side for a short
    hi = [100, 104, 105, 100, 98]
    lo = [100, 101, 102, 99, 97]
    cl = [100, 103, 104, 99, 97]
    tr = _self_arm(hi, lo, cl, side="short", wrong_m=2, alt_bars=(4,))
    assert tr.close_reason.iloc[0] == "alt_exit" and tr.pnl.iloc[0] > 0


def test_a_second_trade_does_not_inherit_the_first_trades_counts():
    """The counters are per trade, and a fresh trade starts at zero."""
    from quant.engine import BacktestConfig
    from quant.engine.config import Signals
    from quant.engine.run import run_backtest

    # long at 100: bar 2 touches it, bar 3 touches again -> armed, and bar 3's alt closes it in
    # profit. A second trade opens at bar 4 (at 107) and by bar 5 has seen ONE touch, not two.
    hi = [100, 105, 101, 105, 108, 108]
    lo = [100, 103, 99, 99, 106, 106]
    cl = [100, 104, 100, 103, 107, 107]
    n = len(cl)
    d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC"),
                      "open": np.asarray(cl, float), "high": np.asarray(hi, float),
                      "low": np.asarray(lo, float), "close": np.asarray(cl, float), "volume": 1.0})
    z = np.zeros(n, bool)
    ent = z.copy()
    ent[0] = True
    ent[4] = True                                   # a second trade, on a clean slate
    alt = z.copy()
    alt[3] = True
    alt[5] = True
    sig = Signals(entry_long=ent, entry_short=z.copy(), exit_long=z.copy(), exit_short=z.copy(),
                  exit_arm_long=z.copy(), exit_alt_long=alt, arm_touch_n=2, alt_replace_tp=True)
    cfg = BacktestConfig(initial_cash=10_000, exit_enabled=True, sl_mode="entry_pct",
                         sl_value=50.0, sizing_mode="cash", cash_per_trade=1_000,
                         max_open_trades=1)
    tr = run_backtest(d, sig, cfg).trades
    assert len(tr) == 2
    # the first armed on its own two touches (bars 2 and 3) and closed at bar 3
    assert tr.close_reason.iloc[0] == "alt_exit" and tr.exit_i.iloc[0] == 3
    # the second opened at bar 4 with a clean slate, so bar 5's alt signal finds it UNarmed
    assert tr.close_reason.iloc[1] != "alt_exit", "the second trade inherited the first's count"


def test_in_profit_must_mean_after_costs():
    """Measured on BTC 5m: three of ten bail-outs moved price the right way and still lost money,
    one of them 0.92 in price against 15.39 of fees. `alt_min_pct` is the gate that fixes it."""
    hi = [100, 105, 101, 106, 100.4]
    lo = [100, 103, 99, 104, 100.0]
    cl = [100, 104, 100, 105, 100.2]              # bar 4 is only +0.2% up on the entry

    loose = _self_arm(hi, lo, cl, touch_n=2, alt_bars=(4,), min_pct=0.0)
    assert loose.close_reason.iloc[0] == "alt_exit", "any favourable move is enough by default"

    tight = _self_arm(hi, lo, cl, touch_n=2, alt_bars=(4,), min_pct=0.5)
    assert tight.close_reason.iloc[0] != "alt_exit", "+0.2% must not clear a 0.5% gate"


def test_the_secondary_is_reachable_by_name_and_replaces_the_target(mtf):
    from quant.scenarios import run_pieces

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=2, tl_full_m=0, tl_touch_k=0,
             tl_combine="all", stop_n=20, be_at_rr=1.0, tp_rr=3.0,
             sec_touch_n=1, sec_wrong_m=1, sec_opp_n=1)
    cfg = dict(initial_cash=10_000, exit_enabled=False, max_open_trades=1)
    plain = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                       params=p, cfg=cfg).trades
    bail = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                      secondary="entry_retest_bail@v1", params=p, cfg=cfg).trades
    assert len(bail) and (bail.close_reason == "alt_exit").any(), "the bail-out never fired"
    assert not (plain.close_reason == "alt_exit").any(), "it fired without being asked for"
    # every bail-out is a favourable price move, which is the engine's gate
    a = bail[bail.close_reason == "alt_exit"]
    sgn = np.where(a.side == "long", 1.0, -1.0)
    assert (sgn * (a.exit_price - a.entry_price) > 0).all()


def test_the_enabled_emas_stay_visible_while_one_trade_is_isolated(mtf):
    """Isolating a trade must not take away the context you judge it against."""
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

    fig = pattern_figure(D, patterns=[], sr_zones=False, trendlines=True, tf_frames=frames,
                         chart_tf="1m", emas=(20, 50), mtf_emas=[("5min", 50)],
                         trades=tr, trade_tp_rr=3.0, max_picker_instances=6, max_shapes_per_tf=4)
    ema = [i for i, t_ in enumerate(fig.data)
           if (t_.legendgroup or "").startswith(("ema", "mtfema"))]
    assert len(ema) == 3, [fig.data[i].name for i in ema]

    rows = [c for m in fig.layout.updatemenus for b in (m.buttons or ())
            for c in _layout_patch(b).get("updatemenus[4].buttons", [])
            if c["label"].startswith("#")]
    assert rows
    for c in rows:
        # The button must not ADDRESS an EMA trace at all. Forcing them to True kept them on the
        # chart but also turned them back on after you had switched them off -- and the same vector
        # did it to the gap and level layers. A button that does not address a trace leaves it
        # exactly as you set it, which is the only version of "stays visible" worth having.
        args = c["args"]
        assert len(args) > 2, "the trade button still carries a whole-figure visibility vector"
        assert not (set(args[2]) & set(ema)), f"{c['label']} reaches an EMA trace"


# ---------------------------------------------------------------------------
# EMAs on the chart: whose bars they are computed from, and turning them off
# ---------------------------------------------------------------------------
def test_an_ema_finer_than_the_chart_is_refused_rather_than_resampled(mtf):
    """The bug this replaces: a 1m EMA asked for on a 5m chart came back as the 5m EMA.

    `resample_ohlcv` only works DOWNWARD. Upsampling a 5m frame to 1m returns the same 11,521 rows,
    so the "1m" EMA200 was identical to the 5m one on 100% of bars — 310 points from the truth on
    BTC, with nothing on the chart to say so. Silence is the whole problem, so this refuses.
    """
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base, _, _ = mtf
    d5 = resample_ohlcv(base, "5min")
    with pytest.raises(ValueError, match="FINER"):
        pattern_figure(d5, patterns=[], sr_zones=False, chart_tf="5m", mtf_emas=[("1m", 50)])

    # given the real bars, it is a different line
    fig = pattern_figure(d5, patterns=[], sr_zones=False, chart_tf="5m", emas=(50,),
                         mtf_emas=[("1m", 50)], ema_frames={"1m": base})
    y = {t.name: pd.Series(np.asarray(t.y, float), index=pd.to_datetime(t.x))
         for t in fig.data if t.name and "EMA" in t.name}
    # each EMA rides its OWN frame's timestamps, so line them up on time rather than on position
    j = y["EMA 50"].reindex(y["1m EMA 50"].index).ffill().to_numpy()
    k = y["1m EMA 50"].to_numpy()
    ok = np.isfinite(j) & np.isfinite(k)
    assert ok.any() and not np.allclose(j[ok], k[ok]), "the 1m EMA is still the chart's own"


def test_a_coarser_ema_still_resamples_because_that_direction_is_sound(mtf):
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base, _, _ = mtf
    d5 = resample_ohlcv(base, "5min")
    fig = pattern_figure(d5, patterns=[], sr_zones=False, chart_tf="5m", mtf_emas=[("15m", 50)])
    assert any(t.name == "15m EMA 50" for t in fig.data)


def test_mtf_ema_timeframes_names_the_frames_a_spec_needs():
    from quant.viz import mtf_ema_timeframes

    spec = [("1m", 20, 200), ("5m", 50), ("15m", 200)]
    assert mtf_ema_timeframes(spec, chart_tf="5m") == ["1m", "15m"]
    assert mtf_ema_timeframes(spec) == ["1m", "5m", "15m"]


def test_emas_on_false_hides_every_ema_without_removing_any_of_them(mtf):
    """The ask was one switch for all of them that still leaves you picking individuals — so they
    are drawn and listed as usual, just `legendonly`. One click each brings back the two you want."""
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base, _, _ = mtf
    d5 = resample_ohlcv(base, "5min")
    kw = dict(patterns=[], sr_zones=False, chart_tf="5m", emas=(20, 50), mtf_emas=[("15m", 50)])
    on = pattern_figure(d5, **kw)
    off = pattern_figure(d5, emas_on=False, **kw)
    names = lambda f: [t.name for t in f.data if t.name and "EMA" in t.name]
    assert names(on) == names(off) and len(names(on)) == 3
    assert all(t.visible is True for t in on.data if t.name and "EMA" in t.name)
    assert all(t.visible == "legendonly" for t in off.data if t.name and "EMA" in t.name)


def test_the_chart_carries_an_emas_on_off_control(mtf):
    from quant.data import resample_ohlcv
    from quant.viz.studies import pattern_figure

    base, _, _ = mtf
    d5 = resample_ohlcv(base, "5min")
    fig = pattern_figure(d5, patterns=[], sr_zones=False, chart_tf="5m", emas=(20, 50))
    btn = [m for m in (fig.layout.updatemenus or ()) if m.type == "buttons"]
    assert len(btn) == 1, "no EMA on/off control on a chart that has EMAs"
    assert [b.label for b in btn[0].buttons] == ["EMAs on (2)", "off"]
    # it must restyle ONLY the EMA traces, or it would disturb the trade you are looking at
    ema_ix = [i for i, t in enumerate(fig.data) if (t.legendgroup or "").startswith("ema")]
    assert list(btn[0].buttons[0].args[1]) == ema_ix


def test_a_legend_double_click_cannot_wipe_an_isolated_trade(mtf):
    """Plotly's default is `toggleothers`: a double-click ISOLATES that entry and hides the rest —
    so double-clicking an EMA while a trade was picked took the trade and its trendline with it."""
    from quant.viz.studies import pattern_figure

    _, frames, D = mtf
    fig = pattern_figure(D, patterns=[], sr_zones=False, trendlines=True, tf_frames=frames,
                         chart_tf="1m", emas=(20,))
    assert fig.layout.legend.itemdoubleclick == "toggle"


def test_one_line_can_be_traded_twice_and_tl_once_per_line_stops_it(mtf):
    """A steeply falling resistance keeps dropping, so a chart-timeframe close can clear it again
    hours later at a much lower level while the line itself is still live -- its own timeframe never
    closed through it. Two real breaks, but both carry the same name on the chart, which reads as a
    duplicate. Off by default because they ARE separate; on when a line should be spent once."""
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_tf_names={"": "1m", "5min__": "5m"},
             tl_close_n=1, tl_full_m=0, tl_touch_k=0, tl_combine="all",
             stop_n=20, be_at_rr=0, tp_rr=3.0)
    cfg = dict(initial_cash=10_000, exit_enabled=False, max_open_trades=1)

    def _causes(once):
        out = []
        for side in ("long", "short"):
            r = run_pieces(D, f"mtf_trendline_{side}@v1", "high_of_n@v1", "be_ladder@v1",
                           params=dict(p, tl_once_per_line=once), cfg=cfg)
            t = trendline_trigger(r.df, r.trades, p, prefix_names=p["tl_tf_names"])
            out += list(t["cause"])
        return out

    many, once = _causes(False), _causes(True)
    assert len(many) >= len(once)
    assert len(set(once)) == len(once), "a line still fired twice with tl_once_per_line on"
    assert set(once) <= set(many), "the option invented a trade rather than removing one"


def test_tl_once_per_line_is_off_by_default(mtf):
    from quant.scenarios.entries import _MTF_TL_DEFAULTS

    assert _MTF_TL_DEFAULTS["tl_once_per_line"] is False


def test_tl_confirm_n_waits_for_a_candle_that_repeats_the_move(mtf):
    """A break candle that is huge next to its neighbours and closes a whole new body beyond the
    line looks decisive and is exactly the shape that gets swept -- entering on its close is
    entering at the worst price. The confirmation asks the market to do it again: within n chart
    candles, one of the TRADE'S OWN COLOUR must close beyond the WHOLE trigger candle."""
    from quant.scenarios.entries import _tl_entry

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=1, tl_full_m=0, tl_touch_k=0,
             tl_combine="all")
    plain = _tl_entry(p, "short")(D)
    waited = _tl_entry(dict(p, tl_confirm_n=3), "short")(D)
    assert plain.any(), "the fixture produced no breaks at all"
    assert waited.sum() <= plain.sum(), "confirmation can only remove or move entries"

    o = D["open"].to_numpy(float); c = D["close"].to_numpy(float); lo = D["low"].to_numpy(float)
    for j in np.flatnonzero(waited):
        assert c[j] < o[j], "a short confirmation candle must be RED"
        prior = np.flatnonzero(plain[max(0, j - 3):j]) + max(0, j - 3)
        assert len(prior), f"bar {j} confirmed nothing within the window"
        assert any(c[j] < lo[i] for i in prior), "it must close below the WHOLE trigger candle"


def test_tl_confirm_n_is_off_by_default_and_changes_nothing_at_zero(mtf):
    from quant.scenarios.entries import _MTF_TL_DEFAULTS, _tl_entry

    assert _MTF_TL_DEFAULTS["tl_confirm_n"] == 0
    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_close_n=1, tl_combine="all")
    assert np.array_equal(_tl_entry(p, "short")(D), _tl_entry(dict(p, tl_confirm_n=0), "short")(D))


def test_a_confirmed_trade_still_names_the_line_it_broke(mtf):
    """The line is retired at the break, so a DELAYED entry lands on a bar where its id is already
    gone -- every confirmed trade was labelled `15m.?` until the id was held for the trigger's own
    window PLUS the confirmation wait."""
    from quant.scenarios import run_pieces
    from quant.scenarios.entries import trendline_trigger

    _, _, D = mtf
    p = dict(tl_prefixes=("", "5min__"), tl_tf_names={"": "1m", "5min__": "5m"},
             tl_close_n=1, tl_full_m=0, tl_touch_k=0, tl_combine="all", tl_confirm_n=3,
             stop_n=20, be_at_rr=0, tp_rr=3.0)
    r = run_pieces(D, "mtf_trendline_short@v1", "high_of_n@v1", "be_ladder@v1",
                   params=p, cfg=dict(initial_cash=10_000, exit_enabled=False))
    t = trendline_trigger(r.df, r.trades, p, prefix_names=p["tl_tf_names"])
    if len(t):
        assert not any("?" in str(c) for c in t["cause"]), list(t["cause"])
