"""
exness/trendlines_mtf.py on the stand-in runtime (tests/indie_standin.py), held to the research
code: each timeframe's lines are quant.indicators.add_trendlines on that timeframe's own candles,
with the email alerts' settings for it; the lines drawn on the chart start at the first touch and
end where the research line ends; and the early marks are the alerts' early-break rule (a chart
candle closing through a higher timeframe's LIVE line, once per line), worked out here
independently from the research lines.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import indie_standin as rt
from quant.alerts.detectors import _tf_x
from quant.indicators import add_trendlines

ROOT = Path(__file__).resolve().parents[1]
MTF = ROOT / "exness" / "trendlines_mtf.py"
DATA = ROOT / "tests" / "data"
ONE_MIN = dict(line_tol_atr=0.4, wick_tol_atr=0.3, max_distance_atr=20, prefer_newer=True,
               break_sustain_bars=1, min_span_frac=1 / 3)
STRICT = dict(line_tol_atr=0.3, wick_tol_atr=0.15, max_distance_atr=8.0, prefer_newer=False,
              break_sustain_bars=0, min_span_frac=0.5)


def _frame(bars, minutes, chart_minutes):
    if minutes == chart_minutes:
        return bars.reset_index(drop=True)
    f = rt.htf_bars(bars, minutes)
    last_close = bars["open_time"].iloc[-1].timestamp() + chart_minutes * 60
    keep = (f["open_time"].astype("int64") // 10**9 + minutes * 60) <= last_close
    return f[keep].reset_index(drop=True)


def _research(bars, minutes, chart_minutes):
    f = _frame(bars, minutes, chart_minutes)
    d = add_trendlines(f[["open", "high", "low", "close"]].copy(), min_touches=3, atr_mult=3.5,
                       fit="strict", **(ONE_MIN if minutes == 1 else STRICT))
    return f, d


def _slots(cm):
    return [(s, m) for s, m in enumerate([cm, 15, 60, 240]) if s == 0 or m > cm]


def _secs(f):
    return (f["open_time"].astype("int64") // 10**9).to_numpy(float)


CASES = [("usdjpy_5m_2026-10-22.csv", 300), ("gold_1m_2026-10-22.csv", 60),
         ("btc_1m_2026-10-24.csv", 60)]


@pytest.fixture(scope="module")
def runs():
    out = {}
    mp = pytest.MonkeyPatch()
    mod = rt.load(MTF, mp)
    for name, secs in CASES:
        bars = rt.read_bars(DATA / name)
        out[name] = (bars, secs, rt.run(mod, bars, bar_secs=secs, max_kept=10_000,
                                        marks_kept=10_000))
    yield out
    mp.undo()


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_each_timeframes_lines_and_breaks_are_add_trendlines_on_its_own_candles(runs, name):
    bars, secs, ctx = runs[name]
    cm = secs // 60
    for s, m in _slots(cm):
        f, d = _research(bars, m, cm)
        outs = np.array(ctx._requests[s].outputs, dtype=float)
        assert len(outs) == len(f)
        h = np.arange(len(f), dtype=float)
        for side, (lvl, brk) in enumerate([("tl_res", "tl_break_up"), ("tl_sup", "tl_break_dn")]):
            b = side * 8
            on = (outs[:, b] == 1.0) | (outs[:, b + 2] == 1.0)   # a line has a price on the
            level = np.where(on, outs[:, b + 6] * h + outs[:, b + 7], np.nan)   # candle it breaks
            np.testing.assert_array_equal(level, d[lvl].to_numpy(), err_msg=f"{m}m {lvl}")
            np.testing.assert_array_equal(outs[:, b + 2] == 1.0, d[brk].to_numpy())
        np.testing.assert_array_equal(outs[:, 16], h)
        np.testing.assert_array_equal(outs[:, 17], _secs(f))


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_the_lines_drawn_are_the_research_lines_from_first_touch_to_end(runs, name):
    bars, secs, ctx = runs[name]
    cm = secs // 60
    total = 0
    for s, m in _slots(cm):
        f, d = _research(bars, m, cm)
        t = _secs(f)
        n, M = len(f), m * 60

        def at(i, ln):
            return (t[i] + M - secs, ln.slope * i + ln.intercept)

        want = sorted(at(ln.start_i, ln) + at(ln.end_i, ln) + (bool(ln.broke),)
                      for ln in d.attrs["tl_lines"] if ln.end_i >= 0 and (ln.broke or
                                                                          ln.end_i < n - 1))
        got = sorted((o.point_a.time, o.point_a.price, o.point_b.time, o.point_b.price,
                      bool(br)) for o, sl, br in zip(ctx._old.get(), ctx._old_slot.get(),
                                                      ctx._old_broke.get()) if sl == s)
        assert got == want, f"{m}m"
        live = sorted(at(ln.start_i, ln) for ln in d.attrs["tl_lines"]
                      if ln.end_i == n - 1 and not ln.broke)
        drawn = sorted((ctx._seg.get()[k].point_a.time, ctx._seg.get()[k].point_a.price)
                       for k in (2 * s, 2 * s + 1) if ctx._cur.get()[k] >= 0)
        assert drawn == live, f"{m}m live"
        total += len(want) + len(live)
    assert total >= (20 if name.startswith("usdjpy") else 1)


def _early_reference(bars, secs, s, m):
    """The alerts' early-break rule, from the research lines: the first chart close beyond a
    higher timeframe's line while it is LIVE -- from its own candle's close, until the close
    of the candle that broke it or (dropped) the candle that dropped it."""
    cm = secs // 60
    f, d = _research(bars, m, cm)
    t = _secs(f)
    n, M = len(f), m * 60
    close_t = _secs(bars) + secs
    x = _tf_x((t * 1e9).astype(np.int64), M * 10**9, (close_t * 1e9).astype(np.int64))
    c = bars["close"].to_numpy(float)
    out = set()
    for ln in d.attrs["tl_lines"]:
        if ln.end_i < 0:
            continue
        seen = t[ln.born_i] + M
        if ln.broke:
            until = t[ln.end_i] + M
        elif ln.end_i < n - 1:
            until = t[ln.end_i + 1] + M
        else:
            until = np.inf
        live = (close_t > seen) & (close_t < until)
        level = ln.slope * x + ln.intercept
        up = ln.kind > 0
        beyond = live & ((c > level) if up else (c < level))
        if beyond.any():
            out.add((s, 0 if up else 1, float(_secs(bars)[int(np.argmax(beyond))])))
    return out


@pytest.mark.parametrize("name", [c[0] for c in CASES])
def test_early_marks_are_the_alerts_early_break_rule(runs, name):
    bars, secs, ctx = runs[name]
    cm = secs // 60
    want = set()
    for s, m in _slots(cm):
        if s:
            want |= _early_reference(bars, secs, s, m)
    got = {(k // 2, k % 2, t) for k, t in zip(ctx._mk_k.get(), ctx._mk_t.get())}
    assert got == want
    if name.startswith("usdjpy"):
        # the alerts' own fixture: USDJPY's 1h support, closed through by the 03:05 5m candle
        at = pd.Timestamp("2026-10-22 03:00", tz="UTC").timestamp()
        assert (2, 1, at) in got
        mk = [mk for mk, k, t in zip(ctx._marks.get(), ctx._mk_k.get(), ctx._mk_t.get())
              if k == 5 and t == at][0]
        assert mk.text == "1h early down" and mk.position.price < bars["low"].iloc[
            int(np.searchsorted(_secs(bars), at))]
        assert len(got) >= 5


def test_thickness_names_the_timeframe_and_tags_the_higher_ones(runs, monkeypatch):
    bars, secs, ctx = runs["usdjpy_5m_2026-10-22.csv"]
    widths = set()

    def spy(c):
        for k in range(8):
            if c._cur.get()[k] >= 0:
                widths.add((k // 2, c._seg.get()[k].line_width))
    rt.run(rt.load(MTF, monkeypatch), bars, bar_secs=secs, spy=spy)
    assert all(w == s + 1 for s, w in widths)                # each timeframe its own width
    assert {s for s, w in widths} >= {0, 1, 2}               # (a month holds no 4h line)
    for k in range(8):
        if ctx._cur.get()[k] >= 0:
            s, side = divmod(k, 2)
            seg = ctx._seg.get()[k]
            assert seg.line_width == s + 1 and seg.line_style == "solid"
            assert seg.color.name == ("red", "green")[side]
            if s:
                tag = ctx._tag.get()[k]
                assert tag.text == ["", "15m", "1h", "4h"][s] + (" R", " S")[side]
                assert (tag.position.time, tag.position.price) == (seg.point_b.time,
                                                                    seg.point_b.price)
    for o, br in zip(ctx._old.get(), ctx._old_broke.get()):
        assert o.line_width == 1 and o.line_style == ("dotted", "solid")[br]
    drawn = ctx.chart.items(rt.LabelAbs)
    assert {d.text for d in drawn} >= {"1h early down"}


@pytest.mark.parametrize("variant", [dict(live_merge="forming", realtime_last=60),
                                     dict(backfill_first=True), dict(lookahead=True),
                                     dict(realtime_last=400, seam="redo"),      # Exness's seam,
                                     dict(realtime_last=400, seam="again"),     # both endings
                                     dict(realtime_last=40, rerun=range(0, 3000, 7))])
def test_a_forming_candle_or_a_rerun_changes_nothing(monkeypatch, variant):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")[:3000]
    mod = rt.load(MTF, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300, max_kept=5)
    ctx = rt.run(mod, bars, bar_secs=300, max_kept=5, **variant)
    assert rt.state(ctx)[1] == rt.state(plain)[1]
    assert ctx._mk_t.get() == plain._mk_t.get()
    if variant.get("seam"):                       # its own state too, after the seam
        assert rt.sec_state(ctx) == rt.sec_state(plain)
    assert ctx.chart.peak <= 90 and ctx.chart.forming == 0


def test_between_closes_a_line_moves_on_in_time_and_holds_across_a_weekend(monkeypatch):
    # the coordinate a chart close is judged at equals the alerts' _tf_x on every 5m candle of
    # a month of USDJPY -- weekends included, where it must hold -- and never reads as late
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    ctx = rt.run(rt.load(MTF, monkeypatch), bars.iloc[:300], bar_secs=300)
    close = _secs(bars) + 300
    for s, m in [(2, 60), (3, 240)]:
        f = _frame(bars, m, 5)
        fo = _secs(f)
        want = _tf_x((fo * 1e9).astype(np.int64), m * 60 * 10**9, (close * 1e9).astype(np.int64))
        j = np.searchsorted(fo + m * 60, close, side="right") - 1
        gaps = 0
        for k in np.flatnonzero(j >= 0):
            ctx._lj.get()[s] = int(j[k])
            ctx._ltj.get()[s] = fo[j[k]]
            ctx._prev_t.set(close[k - 1] - 300 if k else np.nan)
            got = ctx._x(s, close[k])
            assert got == pytest.approx(want[k], abs=1e-9), (m, bars["open_time"].iloc[k])
            gaps += k and close[k] - close[k - 1] > 3600
        assert gaps >= 3                                          # weekends were crossed


def test_timeframes_switch_off_and_old_lines_are_bounded(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(MTF, monkeypatch)
    ctx = rt.run(mod, bars, bar_secs=300, max_kept=3, marks_kept=4)
    assert all(ctx._old_slot.get().count(s) <= 3 for s in range(4))
    assert len(ctx._marks.get()) == 4
    assert len(ctx.chart.items(rt.LineSegment)) == len(ctx._old.get()) + sum(
        ctx._cur.get()[k] >= 0 for k in range(8))
    ctx = rt.run(mod, bars, bar_secs=300, show_15m=False, show_4h=False, early=False)
    assert set(ctx._old_slot.get()) <= {0, 2} and not ctx._marks.get()
    one_h = rt.htf_bars(bars, 60)                      # on a 1h chart only 4h is above it
    ctx = rt.run(mod, one_h, bar_secs=3600)
    assert set(ctx._old_slot.get()) <= {0, 3}


def test_the_fitter_is_trendlines_pys_own_copied_unchanged():
    # Indie has no imports between files, so trendlines_mtf.py carries its own copy of the
    # fitter: it must be exness/trendlines.py's, but for the drawing calls. After changing the
    # fitter there, run `python scripts/sync_exness_fitter.py`
    import importlib.util
    spec = importlib.util.spec_from_file_location("sync", ROOT / "scripts" / "sync_exness_fitter.py")
    sync = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sync)
    want = sync.fitter_block((ROOT / "exness" / "trendlines.py").read_text(encoding="utf-8"))
    assert sync.current_block(MTF.read_text(encoding="utf-8")) == want


@pytest.mark.parametrize("seam", ["redo", "again"])
def test_the_chart_running_before_its_timeframes_draws_the_same(monkeypatch, seam):
    # Exness does not say whether, at a shared close, the chart or the higher timeframe updates
    # first. Chart first, it is handed the candle closing now before that timeframe has finished
    # it (after a load, the version computed at load): a state at its own close is taken only if
    # its close IS the chart's close, else a chart candle later. Every line, end and early mark is
    # the clean run's; only a live line's tip can sit a candle further on
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")[:3000]
    mod = rt.load(MTF, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300, max_kept=10_000, marks_kept=10_000)
    ctx = rt.run(mod, bars, bar_secs=300, max_kept=10_000, marks_kept=10_000, realtime_last=400,
                 seam=seam, main_first=True)

    def drawn(c):
        old = sorted(zip(map(repr, map(rt.plain, c._old.get())), c._old_slot.get(),
                         c._old_broke.get()))
        live = sorted((k, rt.plain(c._seg.get()[k])[1:3]) for k in range(8) if c._cur.get()[k] >= 0)
        return old, live, list(zip(c._mk_k.get(), c._mk_t.get()))
    assert drawn(ctx) == drawn(plain)
    assert ctx.chart.peak <= 90 and ctx.chart.forming == 0


@pytest.mark.parametrize("k0, what", [(1933, "15m resistance broken by the 30 Sep 23:45 candle"),
                                      (4801, "1h support broken by the 14 Oct 23:00 candle")])
def test_a_break_candle_loaded_half_way_is_not_taken_unfinished(monkeypatch, k0, what):
    # The chart goes live 5 minutes into the candle that BREAKS a line; after those 5 minutes the
    # line still held. Chart first at that candle's close, it is handed the version computed at
    # load: taken as it is, the line would stay live, then end as DROPPED, never as broken. Its
    # close is not the chart's close at that instant, so it waits a chart candle for the final one
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")[:k0 + 60]
    mod = rt.load(MTF, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300, max_kept=10_000, marks_kept=10_000)
    ctx = rt.run(mod, bars, bar_secs=300, max_kept=10_000, marks_kept=10_000, realtime_last=60,
                 seam="redo", main_first=True)
    assert 1 in plain._old_broke.get()
    assert list(zip(ctx._old_slot.get(), ctx._old_broke.get())) == \
        list(zip(plain._old_slot.get(), plain._old_broke.get()))
    assert [rt.plain(o) for o in ctx._old.get()] == [rt.plain(o) for o in plain._old.get()]
