"""
exness/fair_value_gaps.py -- the research FVG ported to Indie for the Exness web terminal -- run
here on a stand-in for the `indie` library, candle by candle the way the terminal runs it, and
held to the research code: the same gaps, born on the same candle, with the same band, dying on
the same candle. Indie cannot run on this PC, so this proves the LOGIC; whether Exness accepts the
syntax is checked by pasting it in.
"""
import importlib.util
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
    def __init__(self, point_a, point_b, **kw):
        self.point_a, self.point_b = point_a, point_b
        self.__dict__.update(kw)


class Series:
    def __init__(self, ctx, name):
        self.ctx, self.name = ctx, name

    def __getitem__(self, k):
        j = self.ctx.bar_index - k
        return float(self.ctx.data[self.name][j]) if j >= 0 else float("nan")


class Chart:
    def __init__(self):
        self.on = {}                     # id -> drawing currently on the chart

    def draw(self, obj):
        self.on[id(obj)] = obj

    def erase(self, obj):
        self.on.pop(id(obj), None)


class Context:
    pass


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


def load_indicator(monkeypatch):
    mods = fake_indie()
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    spec = importlib.util.spec_from_file_location("exness_fvg", INDIE_FILE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.Main, mods["indie.algorithms"].Atr


def run(monkeypatch, bars, bar_secs=60, realtime_last=0, **settings):
    """Feed `bars` to the indicator candle by candle. The last `realtime_last` candles arrive the
    way a live candle does: several unfinished updates, then the final one."""
    Main, Atr = load_indicator(monkeypatch)
    kw = dict(DEFAULTS, **settings)
    ctx = Main.__new__(Main)
    # what Exness's TimeFrame offers (its compiler listed __eq__ __le__ __lt__ to_minutes):
    # anything newer, like to_seconds, must fail here as it failed there
    ctx.time_frame = types.SimpleNamespace(to_minutes=lambda: bar_secs // 60)
    ctx.chart = Chart()
    full = {c: bars[c].to_numpy(float).copy() for c in ("open", "high", "low", "close")}
    full["time"] = (bars["open_time"].astype("int64") // 10**9).to_numpy(float)
    full[f"atr{kw['atr_len']}"] = wilder_atr(bars, kw["atr_len"]).to_numpy(float)
    ctx.data = {k: v.copy() for k, v in full.items()}
    for name in ("open", "high", "low", "close", "time"):
        setattr(ctx, name, Series(ctx, name))
    Atr.ctx = ctx
    ctx.bar_index = 0
    Main.__init__(ctx, **kw)
    n = len(bars)
    for k in range(n):
        ctx.bar_index, ctx.is_last_bar = k, k >= n - 1 - realtime_last
        if k >= n - realtime_last:                       # a live candle: unfinished updates
            o = full["open"][k]
            for frac in (0.2, 0.6, 0.9):
                ctx.data["high"][k] = o + frac * (full["high"][k] - o)
                ctx.data["low"][k] = o + frac * (full["low"][k] - o)
                ctx.data["close"][k] = o + frac * (full["close"][k] - o)
                ctx.is_closed_bar = False
                ctx.calc()
            for c in ("high", "low", "close"):
                ctx.data[c][k] = full[c][k]
        ctx.is_closed_bar = True
        ctx.calc()
    return ctx, full["time"]


def gaps_of(ctx, times):
    """What the indicator left: (born index, kind, top, bot, end index or -1 while open)."""
    idx = {t: i for i, t in enumerate(times)}
    secs = ctx._bar_secs
    out = []
    for b in list(ctx._dead) + list(ctx._box):
        born = idx[b.point_a.time - secs]
        kind = 1 if b.bg_color.name == "green" else -1
        dead = b in ctx._dead
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


def test_a_live_candle_changes_nothing_until_it_closes(monkeypatch, gold):
    s = dict(keep_dead=True, max_dead=100000, max_live=100000, mitigation="touch",
             min_size_atr=0.0)
    hist, t = run(monkeypatch, gold, **s)
    live, _ = run(monkeypatch, gold, realtime_last=60, **s)
    assert gaps_of(live, t) == gaps_of(hist, t)


def test_open_gaps_run_to_the_live_candle_and_dead_ones_leave_the_chart(monkeypatch, gold):
    ctx, times = run(monkeypatch, gold, mitigation="touch", min_size_atr=0.0)
    last_close = times[-1] + 60
    assert ctx._box and all(b.point_b.time == last_close for b in ctx._box)
    on = set(ctx.chart.on)
    assert on == {id(b) for b in ctx._box}             # keep_dead off: only the open gaps drawn


def test_too_many_open_gaps_drop_the_oldest(monkeypatch, gold):
    ctx, _ = run(monkeypatch, gold, mitigation="full", min_size_atr=0.0, max_live=3)
    assert len(ctx._box) <= 3 and len(ctx.chart.on) <= 3
    assert ctx._born == sorted(ctx._born)
