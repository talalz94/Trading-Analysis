"""
exness/fair_value_gaps.py -- the research FVG ported to Indie for the Exness web terminal -- run
here on a stand-in for the `indie` runtime, candle by candle the way the terminal runs it, and
held to the research code: the same gaps, born on the same candle, with the same band, dying on
the same candle. Indie cannot run on this PC, so this proves the LOGIC; whether Exness accepts the
syntax is checked by pasting it in.

The stand-in models the three runtime rules Exness enforced with an error each:
  - every update of a candle starts from the state AND the chart as they were before that candle
    (Var rolls back; a plain field does not), and a candle can be updated again after it closed;
  - erasing a drawing the rollback removed is an error ("cannot erase: this drawing was removed
    because the candle value changed");
  - at most 100 drawing changes per update.
"""
import copy
import importlib.util
import itertools
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant.indicators.fvg import add_fvgs
from quant.indicators.volatility import atr as wilder_atr

ROOT = Path(__file__).resolve().parents[1]
INDIE_FILE = ROOT / "exness" / "fair_value_gaps.py"
GOLD_1M = ROOT / "tests" / "data" / "gold_1m_2026-10-21.csv"
_ids = itertools.count(1)


# ── a stand-in for the parts of `indie` the indicator uses ─────────────────────────────────────
class Color:
    def __init__(self, name, alpha=1.0):
        self.name, self.alpha = name, alpha

    def __call__(self, alpha):
        return Color(self.name, alpha)


class Pos:
    def __init__(self, time, price, offset=0.0):
        self.time, self.price = time, price


class Rect:
    """A drawing is known to the chart by an id it carries, so a Var's copy is the same drawing."""
    def __init__(self, point_a, point_b, **kw):
        self.point_a, self.point_b = point_a, point_b
        self.__dict__.update(kw)
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
    """Counts changes per update (Exness refuses more than 100) and, like Exness, refuses to
    erase a drawing that is not on it."""
    def __init__(self):
        self.on = {}                     # drawing id -> its fields as drawn
        self.changes = 0
        self.peak = 0

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


def fake_indie():
    indie = types.ModuleType("indie")
    indie.indicator = lambda *a, **k: (lambda cls: cls)
    indie.param = types.SimpleNamespace(**{k: _param(k) for k in ("str", "float", "int", "bool",
                                                                  "color")})
    indie.color = types.SimpleNamespace(GREEN=Color("green"), RED=Color("red"),
                                        TRANSPARENT=Color("none", 0.0))
    indie.line_style = types.SimpleNamespace(SOLID="solid", DOTTED="dotted", DASHED="dashed")
    indie.MainContext = Context
    algos = types.ModuleType("indie.algorithms")

    class Atr:
        @staticmethod
        def new(length):
            return Series(Atr.ctx, f"atr{length}")
    algos.Atr = Atr
    draws = types.ModuleType("indie.drawings")
    draws.Rectangle, draws.AbsolutePosition = Rect, Pos
    return {"indie": indie, "indie.algorithms": algos, "indie.drawings": draws}


def load_indicator(monkeypatch, path=INDIE_FILE):
    mods = fake_indie()
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    spec = importlib.util.spec_from_file_location("exness_fvg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Main, mods["indie.algorithms"].Atr


def run(monkeypatch, bars, bar_secs=60, realtime_last=0, rerun=(), corrected=None,
        path=INDIE_FILE, **settings):
    """Feed `bars` to the indicator candle by candle, the way Exness does:
      - every update of a candle starts from the Vars and the chart as they were before it;
      - the last `realtime_last` candles arrive live: three unfinished updates, then the close;
      - candles in `rerun` are run AGAIN after they closed (Exness re-runs a candle whose value
        changed), with the values of `corrected` if given -- which then stand from there on."""
    Main, Atr = load_indicator(monkeypatch, path)
    kw = dict(DEFAULTS, **settings)
    final = corrected if corrected is not None else bars
    ctx = Main.__new__(Main)
    ctx._vars = []
    # what Exness's TimeFrame offers (its compiler listed __eq__ __le__ __lt__ to_minutes):
    # anything newer, like to_seconds, must fail here as it failed there
    ctx.time_frame = types.SimpleNamespace(to_minutes=lambda: bar_secs // 60)
    ctx.chart = Chart()
    full = {c: bars[c].to_numpy(float).copy() for c in ("open", "high", "low", "close")}
    fixed = {c: final[c].to_numpy(float) for c in ("open", "high", "low", "close")}
    full["time"] = (bars["open_time"].astype("int64") // 10**9).to_numpy(float)
    full[f"atr{kw['atr_len']}"] = wilder_atr(final, kw["atr_len"]).to_numpy(float)
    ctx.data = {k: v.copy() for k, v in full.items()}
    for name in ("open", "high", "low", "close", "time"):
        setattr(ctx, name, Series(ctx, name))
    Atr.ctx = ctx
    ctx.bar_index = 0
    Main.__init__(ctx, **kw)
    ctx.min_ahead = float("inf")          # the closest any open box's edge came to the live candle
    rerun = set(rerun)
    state = {}

    def update(closed, first):
        if not first:                     # every later update starts from before this candle:
            for v in ctx._vars:           # the Vars...
                v.work = copy.deepcopy(state[id(v)])
            ctx.chart.on = copy.deepcopy(state["chart"])       # ...and the chart
        ctx.chart.changes = 0
        ctx.is_closed_bar = closed
        ctx.calc()

    n = len(bars)
    for k in range(n):
        ctx.bar_index = k
        live = k >= n - realtime_last
        if live or k in rerun:            # a candle with more than one update: remember its start
            state = {id(v): copy.deepcopy(v.work) for v in ctx._vars}
            state["chart"] = copy.deepcopy(ctx.chart.on)
        first = True
        if live:                                               # a live candle: unfinished updates
            o = full["open"][k]
            for frac in (0.2, 0.6, 0.9):
                for c in ("high", "low", "close"):
                    ctx.data[c][k] = o + frac * (full[c][k] - o)
                update(False, first)
                first = False
            for c in ("high", "low", "close"):
                ctx.data[c][k] = full[c][k]
        update(True, first)
        if k in rerun:                                         # the closed candle, run again
            for c in ("open", "high", "low", "close"):
                ctx.data[c][k] = fixed[c][k]
            update(True, False)
        if ctx._box.get():                # after each closed candle: how far ahead of it
            ahead = min(b.point_b.time for b in ctx._box.get()) - (full["time"][k] + bar_secs)
            ctx.min_ahead = min(ctx.min_ahead, ahead)
    return ctx, full["time"]


def gaps_of(ctx, times):
    """What the indicator holds: (born index, kind, top, bot, end index or -1 while open)."""
    idx = {t: i for i, t in enumerate(times)}
    secs = ctx._bar_secs
    out = []
    for b, dead in [(b, True) for b in ctx._dbox.get()] + [(b, False) for b in ctx._box.get()]:
        born = idx[b.point_a.time - secs]
        kind = 1 if b.bg_color.name == "green" else -1
        end = idx[b.point_b.time - secs] if dead else -1
        out.append((born, kind, round(b.point_a.price, 6), round(b.point_b.price, 6), end))
    return sorted(out)


def research(bars, max_age=1000, mitigation="ce", min_size_atr=0.5, min_disp_atr=0.0):
    g = add_fvgs(bars.rename(columns={"open_time": "t"}), mitigation=mitigation,
                 min_size_atr=min_size_atr, min_disp_atr=min_disp_atr, max_age_bars=max_age,
                 atr_period=14).attrs["fvg_gaps"]
    end = np.where(g["end_reason"].isin(["filled", "expired"]), g["end_i"], -1)
    return sorted(zip(g["born_i"].astype(int), g["kind"].astype(int), g["top"].round(6),
                      g["bot"].round(6), end.astype(int)))


def on_chart_is_the_state(ctx):
    """Every box the state says is drawn is on the chart, once, and nothing else is."""
    want = {b.did for b, d in zip(ctx._box.get(), ctx._drawn.get()) if d}
    want |= {b.did for b, d in zip(ctx._dbox.get(), ctx._ddrawn.get()) if d}
    return set(ctx.chart.on) == want and not ctx._trash.get()


@pytest.fixture
def gold():
    b = pd.read_csv(GOLD_1M, comment="#")
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    b["volume"] = 1.0
    return b


def test_the_defaults_are_the_notebooks(monkeypatch):
    load_indicator(monkeypatch)
    assert DEFAULTS["mitigation"] == "ce" and DEFAULTS["min_size_atr"] == 0.5
    assert DEFAULTS["min_disp_atr"] == 0.0 and DEFAULTS["max_age"] == 1000
    assert DEFAULTS["atr_len"] == 14


@pytest.mark.parametrize("settings", [
    dict(),                                             # as shipped: ce, 0.5 ATR, 1000 candles
    dict(mitigation="touch", min_size_atr=0.25),
    dict(mitigation="full", min_size_atr=0.0, max_age=30),   # many gaps, and some expire
    dict(min_size_atr=0.0, min_disp_atr=1.0),           # the displacement gate does the cutting
])
def test_the_port_finds_exactly_the_research_gaps(monkeypatch, gold, settings):
    s = dict(keep_dead=True, max_dead=100000, max_live=100000, **settings)
    ctx, times = run(monkeypatch, gold, **s)
    want = research(gold, max_age=s.get("max_age", 1000), mitigation=s.get("mitigation", "ce"),
                    min_size_atr=s.get("min_size_atr", 0.5),
                    min_disp_atr=s.get("min_disp_atr", 0.0))
    assert len(want) >= 5
    assert gaps_of(ctx, times) == want
    assert on_chart_is_the_state(ctx)


def test_a_live_candle_changes_nothing_until_it_closes(monkeypatch, gold):
    s = dict(keep_dead=True, max_dead=100000, max_live=100000, mitigation="touch",
             min_size_atr=0.0)
    hist, t = run(monkeypatch, gold, **s)
    live, _ = run(monkeypatch, gold, realtime_last=60, **s)
    assert gaps_of(live, t) == gaps_of(hist, t)


def _corrected(gold):
    """The fixture with one candle's value changed after the fact so that it opens a gap."""
    fixed = gold.copy()
    lo, hi = gold["low"].to_numpy(), gold["high"].to_numpy()
    k = int(np.flatnonzero(lo[2:] <= hi[:-2])[40]) + 2          # a candle with no gap...
    fixed.loc[k, "low"] = hi[k - 2] + 0.5                       # ...given one
    fixed.loc[k, ["high", "close", "open"]] = fixed.loc[k, "low"] + np.array([1.0, 0.5, 0.5])
    return fixed, k


@pytest.mark.parametrize("keep_dead", [False, True])
def test_a_closed_candle_run_again_neither_breaks_nor_duplicates(monkeypatch, gold, keep_dead):
    # what Exness raised: "cannot erase: this drawing was removed because the candle value
    # changed". It re-runs a candle after it closed and the chart rolls back; state in plain
    # fields did not. Every 3rd candle is run twice as it was, and one is CORRECTED so that a gap
    # opens there that the first run did not see.
    s = dict(keep_dead=keep_dead, max_dead=100000, max_live=100000, mitigation="touch",
             min_size_atr=0.0)
    fixed, k = _corrected(gold)
    ctx, t = run(monkeypatch, gold, rerun=set(range(0, len(gold), 3)) | {k}, corrected=fixed, **s)
    assert on_chart_is_the_state(ctx)
    if keep_dead:
        want = research(fixed, mitigation="touch", min_size_atr=0.0)
        assert gaps_of(ctx, t) == want
        assert k in [g[0] for g in want]                   # the corrected candle's own gap


def test_open_gaps_reach_past_the_live_candle_and_dead_ones_leave_the_chart(monkeypatch, gold):
    ctx, times = run(monkeypatch, gold, mitigation="touch", min_size_atr=0.0)
    last_close, bar = times[-1] + 60, 60
    assert ctx._box.get() and not any(ctx._dirty.get())    # nothing owed on an ordinary chart
    assert ctx.min_ahead >= 2 * bar                    # after EVERY candle, past the next one
    for b in ctx._box.get():                            # ahead of the live candle, at most `reach`
        assert last_close + 2 * bar <= b.point_b.time <= last_close + 20 * bar
    assert on_chart_is_the_state(ctx) and len(ctx.chart.on) == len(ctx._box.get())


def test_too_many_open_gaps_drop_the_oldest(monkeypatch, gold):
    ctx, _ = run(monkeypatch, gold, mitigation="full", min_size_atr=0.0, max_live=3)
    assert len(ctx._box.get()) <= 3 and len(ctx.chart.on) <= 3
    assert ctx._born.get() == sorted(ctx._born.get())


def _staircase(steps=300):
    """`steps` candles each a dollar above the last -- a gap on every one -- then a crash candle
    through all of them, then a quiet tail."""
    t = pd.date_range("2026-10-21 00:00", periods=steps + 41, freq="1min", tz="UTC")
    p = np.r_[100 + np.arange(steps, dtype=float), [50.0] * 41]
    b = pd.DataFrame({"open_time": t, "open": p - 0.1, "high": p + 0.2, "low": p - 0.2,
                      "close": p + 0.1, "volume": 1.0})
    b.loc[steps, ["open", "high", "low", "close"]] = [p[steps - 1], p[steps - 1] + 0.2, 49.8, 50.1]
    return b


@pytest.mark.parametrize("keep_dead", [False, True])
def test_one_candle_filling_hundreds_of_gaps_never_breaks_exness_limit(monkeypatch, keep_dead):
    # what broke on Exness: more than 100 drawing changes in one update
    bars = _staircase()
    ctx, times = run(monkeypatch, bars, keep_dead=keep_dead, max_dead=100000, max_live=100000)
    assert ctx.chart.peak <= 90
    assert ctx._kind.get() == [-1]  # the crash filled every bullish gap AT ONCE; what is left is
                                    # the bearish gap the crash itself opened (research has it too)
    assert on_chart_is_the_state(ctx)                  # ...and the drawing caught up later
    if keep_dead:
        assert len(ctx._dbox.get()) > 250
        assert gaps_of(ctx, times) == research(bars)   # every box stops on the crash candle


def test_hundreds_of_open_gaps_stay_ahead_of_the_live_candle(monkeypatch):
    bars = _staircase().iloc[:300]                     # no crash: ~300 gaps stay open
    ctx, times = run(monkeypatch, bars, max_live=1000)
    assert len(ctx._box.get()) > 250 and ctx.chart.peak <= 90
    assert ctx.min_ahead >= 2 * 60                     # 300 boxes, and none ever fell behind
