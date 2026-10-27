"""
A stand-in for the parts of Exness's Indie runtime the indicators in exness/ use, so they can run
here candle by candle and be held to the research code. Indie cannot run on this PC; this proves
the LOGIC, and pasting a file into the terminal proves the syntax.

What it models, each from the Indie docs or from what Exness did to the FVG indicator:
  - Var state rolled back before every update of a candle, the chart with it; a closed candle can
    be run again; erasing a drawing the rollback removed is an error (Exness, Oct 2026);
  - at most 100 drawing changes per update (counted, the tests require <= 90);
  - TimeFrame.from_str / to_minutes and NO to_seconds (Exness's Indie lacks it);
  - calc_on(SecClass, time_frame=...): the secondary context runs on its own candles, and a chart
    candle sees the latest one CLOSED by its own close (Indie's documented merge); a timeframe
    finer than the chart is refused unless `lower_ok` (Indie 5.14+). Two harsher variants test that
    the indicators guard themselves: `live_merge="forming"` hands the chart the FORMING candle's
    output while a chart candle forms, and `backfill_first` hands early chart candles the first
    higher-timeframe candle before it has closed (what the CSV docs say for candles before the
    first one);
    A third, `lookahead=True`, is Indie's documented calc_on(lookahead=True): candles merged by
    their OPENING time, so every chart candle sees the final values of the candle it is in;
  - @param_ref: a secondary context's __init__ gets the chart's settings by id.
"""
from __future__ import annotations

import copy
import importlib.util
import itertools
import sys
import types
from math import isnan, nan

import numpy as np
import pandas as pd

_ids = itertools.count(1)


class Color:
    def __init__(self, name, alpha=1.0):
        self.name, self.alpha = name, alpha

    def __call__(self, alpha):
        return Color(self.name, alpha)

    def __repr__(self):
        return f"{self.name}({self.alpha})"


def rgba(r, g, b, a=1.0):
    return Color(f"rgb({r},{g},{b})", a)


class AbsolutePosition:
    def __init__(self, time, price):
        self.time, self.price = float(time), float(price)


class RelativePosition:
    def __init__(self, vertical_anchor=None, horizontal_anchor=None, top_bottom_ratio=0.0,
                 left_right_ratio=0.0):
        self.va, self.ha = vertical_anchor, horizontal_anchor
        self.top_bottom_ratio, self.left_right_ratio = top_bottom_ratio, left_right_ratio


class _Drawing:
    def __init__(self):
        self.did = next(_ids)


class LineSegment(_Drawing):
    def __init__(self, point_a, point_b, color=None, line_style="solid", line_width=1):
        super().__init__()
        self.point_a, self.point_b = point_a, point_b
        self.color = color if color is not None else Color("yellow")
        self.line_style, self.line_width = line_style, line_width


class _Label(_Drawing):
    def __init__(self, text, position, bg_color=None, text_color=None, font_size=12):
        super().__init__()
        assert isinstance(text, str), "a label's text must be a str"
        self.text, self.position = text, position
        self.bg_color, self.text_color, self.font_size = bg_color, text_color, font_size


class LabelAbs(_Label):
    pass


class LabelRel(_Label):
    pass


class Var:
    def __init__(self, init):
        self.work = copy.deepcopy(init)

    def get(self):
        return self.work

    def set(self, v):
        self.work = copy.deepcopy(v)


class TimeFrame:
    """Exness's: from_str and to_minutes, and NO to_seconds."""
    UNITS = {"m": 1, "h": 60, "D": 1440, "W": 10080}

    def __init__(self, minutes):
        self._m = int(minutes)

    @staticmethod
    def from_str(s):
        return TimeFrame(int(s[:-1] or 1) * TimeFrame.UNITS[s[-1]])

    def to_minutes(self):
        return self._m


class Series:
    def __init__(self, ctx, name):
        self.ctx, self.name = ctx, name

    def __getitem__(self, k):
        j = self.ctx.bar_index - k
        return float(self.ctx.data[self.name][j]) if j >= 0 else nan


class MergedSeries:
    """What calc_on returns: element `idx` of the secondary context's output, as the chart candle
    `k` back sees it."""
    def __init__(self, main, req, idx):
        self.main, self.req, self.idx = main, req, idx

    def __getitem__(self, k):
        j = self.main.bar_index - k
        if j < 0:
            return nan
        got = self.req.merged.get(j)
        return nan if got is None else float(got[self.idx])


class RemovedDrawing(RuntimeError):
    pass


class LowerTimeFrame(RuntimeError):
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

    def items(self, kind):
        return [d for d in self.on.values() if isinstance(d, kind)]


class Context:
    def new_var(self, init):
        v = Var(init)
        self._vars.append(v)
        return v


class SecContext(Context):
    pass


class _Request:
    def __init__(self, cls, minutes):
        self.cls, self.minutes = cls, minutes
        self.ctx = None
        self.outputs = []             # one per CLOSED candle of its own timeframe
        self.merged = {}              # chart candle -> the output it sees
        self.next = 0                 # its next candle to run


class MainContext(Context):
    def calc_on(self, sec_context, exchange=None, ticker=None, time_frame=None, *, source=None,
                lookahead=False):
        m = (time_frame or self.time_frame).to_minutes()
        if m < self.time_frame.to_minutes() and not self._lower_ok:
            raise LowerTimeFrame("only time frames equal to or higher than the current one")
        req = _Request(sec_context, m)
        self._requests.append(req)
        n_out = self._arity(sec_context)
        res = tuple(MergedSeries(self, req, i) for i in range(n_out))
        return res if n_out > 1 else res[0]


DEFAULTS: dict = {}


def _param(kind):
    def deco(pid, default=None, **kw):
        DEFAULTS[pid] = default

        def apply(obj):
            return obj
        return apply
    return deco


def _param_ref(pid):
    def apply(cls):
        cls._refs = [pid] + list(getattr(cls, "_refs", []))
        return cls
    return apply


def fake_indie():
    indie = types.ModuleType("indie")
    indie.indicator = lambda *a, **k: (lambda cls: cls)
    indie.sec_context = lambda f: f
    indie.param = types.SimpleNamespace(**{k: _param(k) for k in (
        "str", "float", "int", "bool", "color", "time_frame")})
    indie.param_ref = _param_ref
    col = types.ModuleType("indie.color")
    for c in ("GREEN", "RED", "BLUE", "NAVY", "YELLOW"):     # proven: docs examples + the FVG
        setattr(col, c, Color(c.lower()))
    col.TRANSPARENT, col.rgba = Color("transparent", 0.0), rgba
    indie.color = col
    indie.line_style = types.SimpleNamespace(SOLID="solid", DOTTED="dotted", DASHED="dashed")
    indie.MainContext, indie.SecContext, indie.TimeFrame = MainContext, SecContext, TimeFrame
    draws = types.ModuleType("indie.drawings")
    draws.LineSegment, draws.LabelAbs, draws.LabelRel = LineSegment, LabelAbs, LabelRel
    draws.AbsolutePosition, draws.RelativePosition = AbsolutePosition, RelativePosition
    draws.vertical_anchor = types.SimpleNamespace(TOP="top", MIDDLE="middle", BOTTOM="bottom")
    draws.horizontal_anchor = types.SimpleNamespace(LEFT="left", CENTER="center", RIGHT="right")
    return indie, draws, col


def load(path, monkeypatch):
    """Import an exness/ file against the stand-in `indie`. Returns the module."""
    DEFAULTS.clear()
    indie, draws, col = fake_indie()
    monkeypatch.setitem(sys.modules, "indie", indie)
    monkeypatch.setitem(sys.modules, "indie.drawings", draws)
    monkeypatch.setitem(sys.modules, "indie.color", col)
    spec = importlib.util.spec_from_file_location(f"exness_{path.stem}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.DEFAULTS = dict(DEFAULTS)
    return m


def read_bars(path):
    b = pd.read_csv(path, comment="#")
    b["open_time"] = pd.to_datetime(b["open_time"], utc=True)
    return b[["open_time", "open", "high", "low", "close"]].reset_index(drop=True)


def htf_bars(bars, minutes):
    """The candles a `minutes` timeframe has over `bars`: UTC buckets, as quant.alerts.engine
    .tf_bars builds them (proved equal to the research resampler)."""
    key = bars["open_time"].dt.floor(f"{int(minutes)}min")
    g = bars.groupby(key, sort=True)
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last()})
    out.index.name = "open_time"
    return out.reset_index()


def _data(frame):
    d = {c: frame[c].to_numpy(float).copy() for c in ("open", "high", "low", "close")}
    d["time"] = (frame["open_time"].astype("int64") // 10**9).to_numpy(float)
    return d


def _setup(ctx, data, minutes):
    ctx._vars = []
    ctx.data = data
    ctx.time_frame = TimeFrame(minutes)
    for name in ("open", "high", "low", "close", "time"):
        setattr(ctx, name, Series(ctx, name))
    ctx.bar_index = 0
    ctx.is_closed_bar = True
    ctx.is_last_bar = False


def run(module, bars, bar_secs=60, *, frames=None, realtime_last=0, rerun=(), lower_ok=False,
        live_merge="closed", backfill_first=False, lookahead=False, spy=None, **settings):
    """Feed `bars` candle by candle the way Exness does. Every secondary context runs on its own
    timeframe's candles (from `frames` {minutes: DataFrame}, else bucketed from `bars`), each closed
    candle once, before the first chart candle that closes at or after it. `spy(ctx)` is called
    after every chart candle's last update. Returns the chart context."""
    Main = module.Main
    kw = dict(module.DEFAULTS, **settings)
    S = int(bar_secs)
    ctx = Main.__new__(Main)
    _setup(ctx, _data(bars), S // 60)
    ctx.chart = Chart()
    ctx._requests, ctx._lower_ok = [], lower_ok
    ctx._arity = lambda cls: _arity(cls, module, bars, S, frames)
    full = _data(bars)
    Main.__init__(ctx, **kw)
    for req in ctx._requests:                          # the secondary contexts' own candles
        f = (frames or {}).get(req.minutes)
        if f is None:
            if req.minutes < S // 60:
                raise ValueError(f"a {req.minutes}m frame is needed below the chart's timeframe")
            f = bars if req.minutes == S // 60 else htf_bars(bars, req.minutes)
        sec = req.cls.__new__(req.cls)
        _setup(sec, _data(f), req.minutes)
        req.ctx, req.close = sec, _data(f)["time"] + req.minutes * 60
        req.full = _data(f)
        req.cls.__init__(sec, **{r: kw[r] for r in getattr(req.cls, "_refs", [])})
        if backfill_first and len(f):
            _sec_closed(req, 0)
    n = len(bars)
    rerun = set(rerun)

    def snapshot():
        st = {id(v): copy.deepcopy(v.work) for v in ctx._vars}
        st["chart"] = copy.deepcopy(ctx.chart.on)
        for r in ctx._requests:
            st[id(r)] = ({id(v): copy.deepcopy(v.work) for v in r.ctx._vars}, len(r.outputs),
                         r.next, dict(r.merged))
        return st

    def restore(st):
        for v in ctx._vars:
            v.work = copy.deepcopy(st[id(v)])
        ctx.chart.on = copy.deepcopy(st["chart"])
        for r in ctx._requests:
            vs, n_out, nxt, merged = st[id(r)]
            for v in r.ctx._vars:
                v.work = copy.deepcopy(vs[id(v)])
            del r.outputs[n_out:]
            r.next, r.merged = nxt, dict(merged)

    def update(k, closed, first, state, live):
        if not first:
            restore(state)
        ctx.chart.changes = 0
        ctx.is_closed_bar = closed
        ctx.is_last_bar = live or k == n - 1          # live, every new candle is the last one
        close_t = full["time"][k] + S
        for r in ctx._requests:
            _advance(r, k, close_t, live and live_merge == "forming",
                     close_t - S if lookahead else None)
        ctx.calc()
        if not closed:
            ctx.chart.forming += ctx.chart.changes

    for k in range(n):
        ctx.bar_index = k
        live = k >= n - realtime_last
        state = snapshot() if (live or k in rerun) else None
        first = True
        if live:
            for frac in (0.3, 0.6, 0.9):
                for c in ("high", "low", "close"):
                    ctx.data[c][k] = full["open"][k] + frac * (full[c][k] - full["open"][k])
                update(k, False, first, state, live)
                first = False
            for c in ("high", "low", "close"):
                ctx.data[c][k] = full[c][k]
        update(k, True, first, state, live)
        if k in rerun:
            update(k, True, False, state, live)
        if spy is not None:
            spy(ctx)
    return ctx


def _sec_closed(req, h):
    sec = req.ctx
    sec.bar_index, sec.is_closed_bar = h, True
    sec.is_last_bar = h == len(req.close) - 1
    out = sec.calc()
    req.outputs.append(out if isinstance(out, tuple) else (out,))
    req.next = h + 1


def _advance(req, k, close_t, forming, opened_by=None):
    while req.next < len(req.close) and req.close[req.next] <= close_t:
        _sec_closed(req, req.next)
    while opened_by is not None and req.next < len(req.close) and             req.full["time"][req.next] <= opened_by:            # lookahead: by opening time
        _sec_closed(req, req.next)
    seen = req.outputs[-1] if req.outputs else None
    if forming and req.next < len(req.close) and req.full["time"][req.next] < close_t:
        # a live chart candle shown its own timeframe's FORMING candle, which updates too and
        # which Indie rolls back after
        sec, h = req.ctx, req.next
        saved = {id(v): copy.deepcopy(v.work) for v in sec._vars}
        sec.bar_index, sec.is_closed_bar = h, False
        out = sec.calc()
        for v in sec._vars:
            v.work = saved[id(v)]
        seen = out if isinstance(out, tuple) else (out,)
    req.merged[k] = seen


def _arity(cls, module, bars, S, frames):
    """How many series a secondary context returns: run a throwaway copy on one candle."""
    probe = cls.__new__(cls)
    _setup(probe, _data(bars.iloc[:1]), S // 60)
    kw = dict(module.DEFAULTS)
    cls.__init__(probe, **{r: kw[r] for r in getattr(cls, "_refs", [])})
    out = probe.calc()
    return len(out) if isinstance(out, tuple) else 1


def plain(v):
    """A drawing (or anything holding drawings) by what it looks like, for state comparisons."""
    if isinstance(v, LineSegment):
        return ("seg", v.point_a.time, v.point_a.price, v.point_b.time, v.point_b.price,
                repr(v.color), v.line_style, v.line_width)
    if isinstance(v, _Label):
        pos = v.position
        where = ((pos.time, pos.price) if isinstance(pos, AbsolutePosition)
                 else (pos.top_bottom_ratio, pos.left_right_ratio))
        return ("label", type(v).__name__, v.text, where, repr(v.bg_color), repr(v.text_color))
    if isinstance(v, list):
        return [plain(x) for x in v]
    if isinstance(v, float) and isnan(v):
        return "nan"
    return v


def state(ctx):
    """Everything a context remembers (its own Vars and its secondary contexts'), and the chart."""
    held = {k: plain(v.get()) for k, v in vars(ctx).items()
            if k.startswith("_") and hasattr(v, "get") and not isinstance(v, dict)}
    for i, r in enumerate(getattr(ctx, "_requests", [])):
        held[f"sec{i}"] = {k: plain(v.get()) for k, v in vars(r.ctx).items()
                           if k.startswith("_") and hasattr(v, "get") and not isinstance(v, dict)}
    return held, sorted(map(repr, (plain(d) for d in ctx.chart.on.values())))
