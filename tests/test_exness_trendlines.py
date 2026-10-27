"""
exness/trendlines.py -- the strict trendlines behind the email alerts, ported to Indie for the
Exness web terminal -- run here on a stand-in for the `indie` runtime, candle by candle the way the
terminal runs it, and held to the research code (quant.indicators.add_trendlines) on real candles:
the same line price on every candle, the same breaks on the same candles, and the same lines drawn
from the same first touch to the same end. Indie cannot run on this PC, so this proves the LOGIC;
whether Exness accepts the syntax is checked by pasting it in.

The stand-in models the runtime rules Exness enforced on the FVG indicator (tests/test_exness_fvg.py):
every update of a candle starts from the Vars and the chart as they were before it, a closed candle
can be run again, erasing a drawing the rollback removed is an error, and an update may make at
most 100 drawing changes. TimeFrame offers to_minutes only, as Exness's does.
"""
import copy
import importlib.util
import itertools
import sys
import types
from math import isnan
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.indicators import add_trendlines

ROOT = Path(__file__).resolve().parents[1]
INDIE_FILE = ROOT / "exness" / "trendlines.py"
DATA = ROOT / "tests" / "data"
_ids = itertools.count(1)

# the email alerts' settings (alerts/alerts_config_trendlines.py): STRICT_1M and STRICT
ONE_MIN = dict(line_tol_atr=0.4, wick_tol_atr=0.3, max_distance_atr=20, prefer_newer=True,
               break_sustain_bars=1, min_span_frac=1 / 3)
STRICT = dict(line_tol_atr=0.3, wick_tol_atr=0.15, max_distance_atr=8.0, prefer_newer=False,
              break_sustain_bars=0, min_span_frac=0.5)


# ── a stand-in for the parts of `indie` the indicator uses ─────────────────────────────────────
class Color:
    def __init__(self, name, alpha=1.0):
        self.name, self.alpha = name, alpha

    def __call__(self, alpha):
        return Color(self.name, alpha)


class Pos:
    def __init__(self, time, price, offset=0.0):
        self.time, self.price = time, price


class Segment:
    """A drawing is known to the chart by an id it carries, so a Var's copy is the same drawing."""
    def __init__(self, point_a, point_b, color=None, line_style="solid", line_width=1):
        self.point_a, self.point_b = point_a, point_b
        self.color = color if color is not None else Color("yellow")
        self.line_style, self.line_width = line_style, line_width
        self.did = next(_ids)


class Var:
    def __init__(self, init):
        self.work = copy.deepcopy(init)

    def get(self):
        return self.work

    def set(self, v):
        self.work = copy.deepcopy(v)


class Series:
    def __init__(self, ctx, name):
        self.ctx, self.name = ctx, name

    def __getitem__(self, k):
        j = self.ctx.bar_index - k
        return float(self.ctx.data[self.name][j]) if j >= 0 else float("nan")


class RemovedDrawing(RuntimeError):
    pass


class Chart:
    def __init__(self):
        self.on = {}
        self.changes = self.peak = self.forming = 0

    def _count(self):
        self.changes += 1
        self.peak = max(self.peak, self.changes)

    def draw(self, obj):
        self.on[obj.did] = copy.deepcopy(obj)
        self._count()

    def erase(self, obj):
        if obj.did not in self.on:
            raise RemovedDrawing("cannot erase: this drawing was removed because the candle "
                                 "value changed and your script no longer draws it")
        del self.on[obj.did]
        self._count()


class Context:
    def new_var(self, init):
        v = Var(init)
        self._vars.append(v)
        return v


DEFAULTS = {}


def _param(kind):
    def deco(pid, default, **kw):
        DEFAULTS[pid] = default
        return lambda cls: cls
    return deco


def load_indicator(monkeypatch):
    indie = types.ModuleType("indie")
    indie.indicator = lambda *a, **k: (lambda cls: cls)
    indie.param = types.SimpleNamespace(**{k: _param(k) for k in ("str", "float", "int", "bool",
                                                                  "color")})
    indie.color = types.SimpleNamespace(GREEN=Color("green"), RED=Color("red"),
                                        GRAY=Color("gray"))
    indie.line_style = types.SimpleNamespace(SOLID="solid", DOTTED="dotted", DASHED="dashed")
    indie.MainContext = Context
    draws = types.ModuleType("indie.drawings")
    draws.LineSegment, draws.AbsolutePosition = Segment, Pos
    monkeypatch.setitem(sys.modules, "indie", indie)
    monkeypatch.setitem(sys.modules, "indie.drawings", draws)
    spec = importlib.util.spec_from_file_location("exness_trendlines", INDIE_FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Main


def run(monkeypatch, bars, bar_secs=60, realtime_last=0, rerun=(), corrected=None, spy=False,
        **settings):
    """Feed `bars` candle by candle, the way Exness does (see test_exness_fvg.run). Returns the
    context, the line price of each side on every candle and the breaks."""
    Main = load_indicator(monkeypatch)
    kw = dict(DEFAULTS, **settings)
    final = corrected if corrected is not None else bars
    ctx = Main.__new__(Main)
    ctx._vars = []
    ctx.time_frame = types.SimpleNamespace(to_minutes=lambda: bar_secs // 60)
    ctx.chart = Chart()
    full = {c: bars[c].to_numpy(float).copy() for c in ("open", "high", "low", "close")}
    fixed = {c: final[c].to_numpy(float) for c in ("open", "high", "low", "close")}
    full["time"] = (bars["open_time"].astype("int64") // 10**9).to_numpy(float)
    ctx.data = {k: v.copy() for k, v in full.items()}
    for name in ("open", "high", "low", "close", "time"):
        setattr(ctx, name, Series(ctx, name))
    ctx.bar_index = 0
    Main.__init__(ctx, **kw)
    if spy:                                        # the ATR of every candle and every swing found
        ctx.atrs, ctx.swings = {}, []
        atr, add = ctx._atr, ctx._add_pivot

        def _atr():
            ctx.atrs[ctx.bar_index] = atr()
            return ctx.atrs[ctx.bar_index]

        def _add(at, price, kind):
            ctx.swings.append((ctx.bar_index, at, price, kind))
            add(at, price, kind)
        ctx._atr, ctx._add_pivot = _atr, _add
    n = len(bars)
    level = np.full((n, 2), np.nan)
    brk = np.zeros((n, 2), bool)
    rerun = set(rerun)
    state = {}

    def update(closed, first):
        if not first:
            for v in ctx._vars:
                v.work = copy.deepcopy(state[id(v)])
            ctx.chart.on = copy.deepcopy(state["chart"])
        ctx.chart.changes = 0
        ctx.is_closed_bar = closed
        ctx.calc()
        if not closed:
            ctx.chart.forming += ctx.chart.changes

    for k in range(n):
        ctx.bar_index = k
        live = k >= n - realtime_last
        if live or k in rerun:
            state = {id(v): copy.deepcopy(v.work) for v in ctx._vars}
            state["chart"] = copy.deepcopy(ctx.chart.on)
        first = True
        if live:                                   # three unfinished updates, then the close
            for frac in (0.3, 0.6, 0.9):
                for c in ("high", "low", "close"):
                    ctx.data[c][k] = ctx.data["open"][k] + frac * (full[c][k] - full["open"][k])
                update(False, first)
                first = False
            for c in ("high", "low", "close"):
                ctx.data[c][k] = full[c][k]
        update(True, first)
        if k in rerun:                             # the closed candle runs again, maybe corrected
            for c in ("open", "high", "low", "close"):
                ctx.data[c][k] = fixed[c][k]
            update(True, False)
        level[k] = ctx._level.get()
        brk[k] = ctx._brk.get()
    return ctx, level, brk


def _bars(name):
    b = pd.read_csv(DATA / name, comment="#")
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    return b


def research(bars, settings):
    return add_trendlines(bars[["open", "high", "low", "close"]].copy(),
                          **dict(dict(min_touches=3, atr_mult=3.5, fit="strict"), **settings))


# Settings "custom" with short swings and a far max distance: lines whose touches span under 40
# candles are refused, and lines live long enough to be retired at 500 candles old
CUSTOM = dict(preset="custom", swing_size=1.5, min_touches=2, touch_tol=0.5, wick_tol=0.3,
              max_distance=50.0, hold=0, span=0.3, prefer_newer=False)
CUSTOM_RESEARCH = dict(atr_mult=1.5, min_touches=2, line_tol_atr=0.5, wick_tol_atr=0.3,
                       max_distance_atr=50.0, break_sustain_bars=0, min_span_frac=0.3)

CASES = [("gold_1m_2026-10-21.csv", 60, ONE_MIN, {}), ("gold_1m_2026-10-22.csv", 60, ONE_MIN, {}),
         ("btc_1m_2026-10-24.csv", 60, ONE_MIN, {}), ("usdjpy_5m_2026-10-22.csv", 300, STRICT, {}),
         ("usdjpy_5m_2026-10-22.csv", 60, ONE_MIN, {}),  # 6,300 candles as "1m": 60 lines, 33 breaks
         ("usdjpy_5m_2026-10-22.csv", 300, CUSTOM_RESEARCH, CUSTOM)]      # 163 lines


@pytest.mark.parametrize("name, secs, settings, exness", CASES)
def test_the_exness_lines_are_the_research_lines_to_the_last_digit(monkeypatch, name, secs,
                                                                     settings, exness):
    # Settings "auto": a 1m chart takes the 1m alerts' values, any other the 5m-4h alerts'
    bars = _bars(name)
    ctx, level, brk = run(monkeypatch, bars, bar_secs=secs, max_kept=10_000, **exness)
    d = research(bars, settings)
    np.testing.assert_array_equal(level[:, 0], d["tl_res"].to_numpy())
    np.testing.assert_array_equal(level[:, 1], d["tl_sup"].to_numpy())
    np.testing.assert_array_equal(brk[:, 0], d["tl_break_up"].to_numpy())
    np.testing.assert_array_equal(brk[:, 1], d["tl_break_dn"].to_numpy())
    assert d["tl_break"].any() and np.isfinite(d["tl_sup"]).any()      # the case has content
    # every line the research code publishes is drawn, from its first touch to where it ended
    t = (bars["open_time"].astype("int64") // 10**9).to_numpy(float)
    want = sorted((t[ln.start_i], ln.at(ln.start_i), t[ln.end_i], ln.at(ln.end_i), ln.broke)
                  for ln in d.attrs["tl_lines"] if ln.end_i >= 0 and ln.end_i < len(bars) - 1
                  or ln.broke)
    old = [s for s in ctx.chart.on.values() if s.line_width == 1]       # ended lines
    got = sorted((s.point_a.time, s.point_a.price, s.point_b.time, s.point_b.price,
                  s.line_style == "solid") for s in old)
    live_end = [ln for ln in d.attrs["tl_lines"]
                if ln.end_i == len(bars) - 1 and not ln.broke]          # still live at the end
    assert got == want
    live = [s for s in ctx.chart.on.values() if s.line_width == 2]
    assert sorted((s.point_a.time, s.point_a.price) for s in live) == sorted(
        (t[ln.start_i], ln.at(ln.start_i)) for ln in live_end)


def _plain(v):
    if isinstance(v, Segment):                     # a drawing, by what it looks like
        return ("seg", v.point_a.time, v.point_a.price, v.point_b.time, v.point_b.price,
                v.color.name, v.color.alpha, v.line_style, v.line_width)
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, float) and isnan(v):
        return "nan"
    return v


def _state(ctx):
    """Everything the indicator remembers, and everything on the chart."""
    held = {k: _plain(v.get()) for k, v in vars(ctx).items()
            if k.startswith("_") and hasattr(v, "get") and not isinstance(v, dict)}
    return held, sorted(_plain(s) for s in ctx.chart.on.values())


@pytest.mark.parametrize("name, secs", [("usdjpy_5m_2026-10-22.csv", 300),     # hold 0
                                        ("gold_1m_2026-10-22.csv", 60)])       # 1m: hold 1
def test_a_live_candle_and_a_candle_run_twice_change_nothing_and_stay_under_the_limit(
        monkeypatch, name, secs):
    # three unfinished updates of each of the last 40 candles, and every 7th candle run again
    # after it closed (old lines trimmed to 5, so drawings are erased too): the state and the
    # chart are exactly the straight run's, no drawing is erased twice, nothing moves while a
    # candle forms, and no update comes near Exness's 100 drawing changes
    bars = _bars(name)[:2500]
    ctx0, level0, brk0 = run(monkeypatch, bars, bar_secs=secs, max_kept=5)
    ctx, level, brk = run(monkeypatch, bars, bar_secs=secs, realtime_last=40,
                          rerun=range(0, len(bars), 7), max_kept=5)
    np.testing.assert_array_equal(level, level0)
    np.testing.assert_array_equal(brk, brk0)
    assert _state(ctx) == _state(ctx0)
    assert ctx.chart.peak <= 90
    assert ctx.chart.forming == 0


def test_a_candle_corrected_after_it_closed_is_judged_on_its_corrected_value(monkeypatch):
    # Exness re-runs a closed candle whose value changed. Here every candle that broke a line
    # is corrected to a doji back on the line's own side, so the break never happened: the
    # result must be the run on the corrected candles, as if the first version had never been
    # seen -- the line is not left retired, and the chart has no trace of it
    from quant.indicators.volatility import atr
    bars = _bars("usdjpy_5m_2026-10-22.csv")[:2500]
    d = research(bars, STRICT)
    a = atr(bars, 14).to_numpy(float)
    fixed = bars.copy()
    ks = []
    for k in np.flatnonzero(d["tl_break_dn"].to_numpy() | d["tl_break_up"].to_numpy()):
        up = bool(d["tl_break_up"].iloc[k])
        line = (d["tl_res"] if up else d["tl_sup"]).iloc[k - 1]
        px = line - 0.5 * a[k] if up else line + 0.5 * a[k]
        fixed.loc[k, ["open", "high", "low", "close"]] = px
        ks.append(int(k))
    assert len(ks) >= 5
    assert not research(fixed, STRICT)["tl_break"].iloc[ks].any()        # the correction matters
    ctx0, want_level, want_brk = run(monkeypatch, fixed, bar_secs=300, max_kept=5)
    ctx, level, brk = run(monkeypatch, bars, bar_secs=300, rerun=ks, corrected=fixed, max_kept=5)
    np.testing.assert_array_equal(level[ks[0]:], want_level[ks[0]:])
    np.testing.assert_array_equal(brk[ks[0]:], want_brk[ks[0]:])
    assert _state(ctx) == _state(ctx0)


def test_presets_take_the_alerts_values_and_custom_takes_its_own(monkeypatch):
    bars = _bars("gold_1m_2026-10-21.csv")
    ctx, level_auto, _ = run(monkeypatch, bars)                      # a 1m chart
    assert (ctx._tol, ctx._wick, ctx._maxd, ctx._hold, ctx._newer) == (0.4, 0.3, 20.0, 1, True)
    assert ctx._span == 1 / 3
    ctx, level_1m, _ = run(monkeypatch, bars, bar_secs=300, preset="1m alerts")
    np.testing.assert_array_equal(level_1m, level_auto)              # named: same on any chart
    ctx, _, _ = run(monkeypatch, bars, bar_secs=300)
    assert (ctx._tol, ctx._wick, ctx._maxd, ctx._hold, ctx._newer, ctx._span) == \
        (0.3, 0.15, 8.0, 0, False, 0.5)
    ctx, level_c, _ = run(monkeypatch, bars, preset="custom", touch_tol=0.4, wick_tol=0.3,
                          max_distance=20.0, hold=1, span=1 / 3, prefer_newer=True)
    np.testing.assert_array_equal(level_c, level_auto)


def test_old_lines_are_capped_and_dropped_lines_can_be_hidden(monkeypatch):
    bars = _bars("usdjpy_5m_2026-10-22.csv")
    ctx, _, _ = run(monkeypatch, bars, bar_secs=300, max_kept=5)
    assert sum(s.line_width == 1 for s in ctx.chart.on.values()) == 5
    ctx, _, _ = run(monkeypatch, bars, bar_secs=300, keep_old=False)
    assert all(s.line_width == 2 for s in ctx.chart.on.values())
    ctx, level, _ = run(monkeypatch, bars, bar_secs=300, show_sup=False, show_res=False)
    assert not ctx.chart.on and np.isfinite(level).any()   # the lines exist, nothing is drawn


def _swings_research(bars, swing):
    from quant.indicators.chart_patterns import _pivots_from_zigzag
    from quant.indicators.structure import add_zigzag_structure
    z = add_zigzag_structure(bars[["open", "high", "low", "close"]].copy(), atr_mult=swing,
                             atr_period=14)
    return [(p.confirm_i, p.i, p.price, p.kind) for p in _pivots_from_zigzag(z)]


@pytest.mark.parametrize("name", ["usdjpy_5m_2026-10-22.csv", "btc_1m_2026-10-24.csv"])
def test_the_atr_and_the_swings_are_the_research_ones_on_every_candle(monkeypatch, name):
    # the ATR is computed here, not taken from the terminal: equal to pandas' to the last bit,
    # NaN on exactly the same first candles; every swing on the candle that confirms it
    from quant.indicators.volatility import atr
    bars = _bars(name)
    ctx, _, _ = run(monkeypatch, bars, bar_secs=300, spy=True)
    got = np.array([ctx.atrs[k] for k in range(len(bars))])
    np.testing.assert_array_equal(got, atr(bars, 14).to_numpy(float))
    assert ctx.swings == _swings_research(bars, 3.5)


def test_a_swing_is_confirmed_by_a_move_of_exactly_the_threshold(monkeypatch):
    # candles one point tall moving half a point a candle: every true range is exactly 1, so
    # the ATR is exactly 1 and the 3.5 x ATR threshold is met EXACTLY by a five-candle leg --
    # which the research code counts (>=) and a strict test (>) would not
    steps = [0.0] * 20                             # a flat start while the ATR warms up
    for leg, n in enumerate([5, 8, 5, 6, 5, 9, 5, 5, 7, 5, 6, 5] * 3):
        steps += [0.5 if leg % 2 == 0 else -0.5] * n
    mid = 100 + np.cumsum(steps)
    t = pd.date_range("2026-10-01", periods=len(mid), freq="5min", tz="UTC")
    bars = pd.DataFrame(dict(open_time=t, open=mid, high=mid + 0.5, low=mid - 0.5, close=mid))
    ctx, _, _ = run(monkeypatch, bars, bar_secs=300, spy=True)
    assert set(ctx.atrs[k] for k in range(14, len(bars))) == {1.0}
    want = _swings_research(bars, 3.5)
    assert len(want) >= 20 and ctx.swings == want


def test_the_deviations_are_summed_in_numpys_order(monkeypatch):
    # The closest fit breaks ties between equally-touched lines, and the research code sums
    # each line's deviations with numpy, whose order (and so whose last bit) changes at 8
    # values. Lines are fitted over the last 16 swings, so every row length up to 16 is checked
    # -- on values where a plain left-to-right sum gets a different last bit.
    Main = load_indicator(monkeypatch)
    rng = np.random.default_rng(7)
    plain_differs = 0
    for n in range(1, 17):
        dev = np.abs(rng.normal(size=(400, n))) * 10.0 ** rng.integers(-3, 4, size=(400, n))
        devs = np.where(rng.random((400, n)) < 0.7, dev, 0.0)
        want = devs.sum(axis=1)
        for r in range(len(devs)):
            row = [float(v) for v in devs[r]]
            assert Main._rowsum(None, row) == want[r]
            plain = 0.0
            for v in row:
                plain += v
            plain_differs += plain != want[r]
    assert plain_differs > 100


@pytest.mark.parametrize("res_touches, sup_touches, res_born, sup_born, close, dropped", [
    (4, 3, 10, 20, 99.5, 1),        # fewer touches loses: the support goes, the resistance holds
    (3, 4, 10, 20, 101.5, 0),       # ... and the mirror
    (3, 3, 10, 20, 99.5, 1),        # a tie: the YOUNGER goes (the support, drawn later)
    (3, 3, 20, 10, 101.5, 0),       # ... and the mirror
])
def test_a_support_above_a_resistance_drops_the_weaker_without_a_break(
        monkeypatch, res_touches, sup_touches, res_born, sup_born, close, dropped):
    # The research code's rule (add_trendlines: min by (touches, -born)), set up by hand because
    # on real candles it is ~1 candle in 156,000 -- none in ~46,000 candles under six settings
    ctx, _, _ = run(monkeypatch, _bars("usdjpy_5m_2026-10-22.csv")[:60], bar_secs=300)
    i = ctx.bar_index
    ctx.data["close"][i] = close
    ctx._on.set([True, True])
    ctx._slope.set([0.0, 0.0])
    ctx._icpt.set([100.0, 101.0])                  # the support sits a point ABOVE the resistance
    ctx._touches.set([res_touches, sup_touches])
    ctx._born.set([res_born, sup_born])
    ctx._start.set([0, 0])
    ctx._end.set([i - 1, i - 1])
    ctx._beyond.set([0, 0])
    ctx._sig.set(["res,", "sup,"])
    ctx._sfirst.set([0, 0])
    ctx._last.set([5, 5])
    retired = len(ctx._ret_sig.get())
    ctx._publish(i, 1.0)
    kept = 1 - dropped
    assert ctx._on.get() == [kept == 0, kept == 1]
    assert ctx._brk.get() == [False, False]        # dropped, not broken; the other one holds
    assert ctx._level.get()[kept] == [100.0, 101.0][kept] and isnan(ctx._level.get()[dropped])
    assert ctx._ret_sig.get()[retired:] == [["res,", "sup,"][dropped]]
