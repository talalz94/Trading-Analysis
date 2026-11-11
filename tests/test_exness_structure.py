"""
exness/structure.py on the stand-in runtime (tests/indie_standin.py), held to the research code:
every HH / HL / LH / LL label on every timeframe is quant.indicators.add_zigzag_structure's, on
that timeframe's own candles, and appears on the chart candle that CLOSES at or after the candle
confirming it -- never earlier, whatever the terminal does with a forming candle.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import indie_standin as rt
from quant.indicators import add_zigzag_structure, atr

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "exness" / "structure.py"
DATA = ROOT / "tests" / "data"
CODE = {"HH": 2, "LH": 1, "HL": -1, "LL": -2}
CASES = [("usdjpy_5m_2026-10-22.csv", 300), ("gold_1m_2026-10-21.csv", 60),
         ("btc_1m_2026-10-24.csv", 60)]


def _frame(bars, minutes, chart_minutes):
    """The candles a slot runs on: the chart's own, or complete higher-timeframe candles."""
    if minutes == chart_minutes:
        return bars
    f = rt.htf_bars(bars, minutes)
    last_close = bars["open_time"].iloc[-1].timestamp() + chart_minutes * 60
    return f[(f["open_time"].astype("int64") // 10**9 + minutes * 60) <= last_close]


def _research(bars, minutes, chart_minutes, swing=3.5):
    f = _frame(bars, minutes, chart_minutes).reset_index(drop=True)
    z = add_zigzag_structure(f[["open", "high", "low", "close"]], atr_mult=swing)
    t = (f["open_time"].astype("int64") // 10**9).to_numpy(float)
    rows = []
    for i in np.flatnonzero(z["zz_label"].notna().to_numpy()):
        rows.append(dict(code=CODE[z["zz_label"].iloc[i]], px=float(z["zz_price"].iloc[i]),
                         pt=t[int(z["zz_pivot_i"].iloc[i])], known=t[i] + minutes * 60))
    return rows


def _labels(ctx, s):
    sl = ctx._slot.get()
    return [(c, p, t) for k, (c, p, t) in enumerate(zip(ctx._code.get(), ctx._px.get(),
                                                       ctx._pt.get())) if sl[k] == s]


@pytest.mark.parametrize("name, secs", CASES)
def test_every_label_on_every_timeframe_is_the_research_swing(monkeypatch, name, secs):
    bars = rt.read_bars(DATA / name)
    cm = secs // 60
    seen = []
    ctx = rt.run(rt.load(STRUCT, monkeypatch), bars, bar_secs=secs, labels_kept=10_000,
                 spy=lambda c: seen.append([c._seen.get()[s] for s in range(5)]))
    close = (bars["open_time"].astype("int64") // 10**9).to_numpy(float) + secs
    for s, minutes in enumerate([cm, 5, 15, 60, 240]):
        if s and minutes <= cm:
            continue
        want = _research(bars, minutes, cm)
        assert _labels(ctx, s) == [(r["code"], r["px"], r["pt"]) for r in want], (s, minutes)
        # each appears on the first chart candle closing at or after its confirmation, not before
        known = np.array([r["known"] for r in want])
        count = np.searchsorted(np.sort(known), close, side="right")
        np.testing.assert_array_equal([row[s] for row in seen], count)
    assert len(_labels(ctx, 0)) >= 10
    if name.startswith("usdjpy"):                       # a month: every timeframe has swings
        assert all(len(_labels(ctx, s)) >= 2 for s in (0, 2, 3, 4))   # 5m is the chart here


def test_labels_sit_on_their_swing_candle_name_their_timeframe_and_stack(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    ctx = rt.run(rt.load(STRUCT, monkeypatch), bars, bar_secs=300, labels_kept=10_000)
    a = atr(bars, 14).to_numpy(float)
    t = (bars["open_time"].astype("int64") // 10**9).to_numpy(float)
    words = {2: "HH", 1: "LH", -1: "HL", -2: "LL"}
    for lab, s, code, px, pt in zip(ctx._labs.get(), ctx._slot.get(), ctx._code.get(),
                                    ctx._px.get(), ctx._pt.get()):
        assert lab.text == ["", "5m ", "15m ", "1h ", "4h "][s] + words[code]
        x = lab.position.time
        if s == 0:
            assert x == pt
        else:                                     # the chart candle that made that extreme
            m = [5, 15, 60, 240][s - 1] * 60
            inside = (t >= pt) & (t < pt + m)
            col = bars["high"] if code > 0 else bars["low"]
            ext = col[inside].max() if code > 0 else col[inside].min()
            assert x == t[inside][col[inside].to_numpy() == ext][-1]
        k = int(np.searchsorted(t, x))
        assert (lab.position.price > px) == (code > 0)              # above highs, below lows
    # labels on the same candle and side STACK: the first 0.8 ATR out, each next one at least
    # 1.4 ATR (of the earlier label's time) beyond the one before -- a later label in a quieter
    # market used to land on top of the first
    spots = {}
    for lab, code, px, off, unit in zip(ctx._labs.get(), ctx._code.get(), ctx._px.get(),
                                        ctx._loff.get(), ctx._lunit.get()):
        assert abs(lab.position.price - px) == pytest.approx(off)
        spots.setdefault((lab.position.time, code > 0), []).append((off, unit))
    assert [d for d in spots.values() if len(d) > 1]            # the case arises on this month
    for d in spots.values():
        assert d[0][0] == pytest.approx(0.8 * d[0][1])
        assert all(d[i + 1][0] - d[i][0] >= 1.4 * d[i][1] * 0.999 for i in range(len(d) - 1))
        assert lab.font_size == 9 + s and lab.text_color.name == ("green" if code in (2, -1)
                                                                       else "red")


@pytest.mark.parametrize("variant", [dict(live_merge="forming", realtime_last=60),
                                     dict(backfill_first=True), dict(lookahead=True),
                                     dict(realtime_last=400, seam="redo"),      # Exness's seam,
                                     dict(realtime_last=400, seam="again"),     # both endings
                                     dict(realtime_last=40, rerun=range(0, 6300, 7))])
def test_a_forming_candle_or_a_rerun_changes_no_label(monkeypatch, variant):
    # the terminal may show a higher timeframe's FORMING candle while a chart candle forms, or
    # the first one before it closes (the CSV docs say so); the labels must not notice. A candle
    # run again leaves exactly the straight run's state and chart
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(STRUCT, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300)
    ctx = rt.run(mod, bars, bar_secs=300, **variant)
    assert rt.state(ctx)[1] == rt.state(plain)[1]
    assert ctx._code.get() == plain._code.get() and ctx._pt.get() == plain._pt.get()
    if variant.get("seam"):                       # its own state too, after the seam
        assert rt.sec_state(ctx) == rt.sec_state(plain)
    assert ctx.chart.peak <= 90 and ctx.chart.forming == 0


def test_a_swing_of_exactly_the_threshold_is_confirmed(monkeypatch):
    # candles one point tall moving half a point a candle: ATR is exactly 1, so a 5-candle leg
    # is EXACTLY 3.5 ATR -- confirmed by the research rule (>=), missed by a strict one
    steps = [0.0] * 20
    for leg, n in enumerate([5, 8, 5, 6, 5, 9, 5, 5, 7, 5, 6, 5] * 3):
        steps += [0.5 if leg % 2 == 0 else -0.5] * n
    mid = 100 + np.cumsum(steps)
    t = pd.date_range("2026-10-01", periods=len(mid), freq="5min", tz="UTC")
    bars = pd.DataFrame(dict(open_time=t, open=mid, high=mid + 0.5, low=mid - 0.5, close=mid))
    ctx = rt.run(rt.load(STRUCT, monkeypatch), bars, bar_secs=300, labels_kept=10_000)
    want = _research(bars, 5, 5)
    assert len(want) >= 20 and _labels(ctx, 0) == [(r["code"], r["px"], r["pt"]) for r in want]


def test_settings_switch_timeframes_and_keep_labels_bounded(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(STRUCT, monkeypatch)
    full = rt.run(mod, bars, bar_secs=300, labels_kept=10_000)
    ctx = rt.run(mod, bars, bar_secs=300, labels_kept=5)
    have = [full._slot.get().count(s) for s in range(5)]
    assert [ctx._slot.get().count(s) for s in range(5)] == [min(5, h) for h in have]
    assert len(ctx.chart.items(rt.LabelAbs)) == sum(min(5, h) for h in have)
    assert ctx._pt.get()[-3:] == full._pt.get()[-3:]          # the newest are the ones kept
    ctx = rt.run(mod, bars, bar_secs=300, show_chart=False, show_1h=False)
    assert set(ctx._slot.get()) == {2, 4}
    one_h = rt.htf_bars(bars, 60)                      # on a 1h chart only 4h is above it
    ctx = rt.run(mod, one_h, bar_secs=3600, labels_kept=10_000)
    assert set(ctx._slot.get()) == {0, 4}
    ctx = rt.run(mod, bars, bar_secs=300, swing_size=6.0, labels_kept=10_000)
    want = _research(bars, 5, 5, swing=6.0)
    assert _labels(ctx, 0) == [(r["code"], r["px"], r["pt"]) for r in want]


@pytest.mark.parametrize("seam", ["redo", "again"])
def test_the_chart_running_before_its_timeframes_labels_the_same(monkeypatch, seam):
    # chart first at a shared close (see test_exness_trendlines_mtf): every label names the same
    # swing at the same candle; only when it appears can be a chart candle later
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(STRUCT, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300, labels_kept=10_000)
    ctx = rt.run(mod, bars, bar_secs=300, labels_kept=10_000, realtime_last=400, seam=seam,
                 main_first=True)

    def labels(c):
        return [(lab.text, lab.position.time, code, px, pt) for lab, code, px, pt in
                zip(c._labs.get(), c._code.get(), c._px.get(), c._pt.get())]
    assert labels(ctx) == labels(plain)
    assert ctx.chart.peak <= 90 and ctx.chart.forming == 0


def test_no_label_for_a_swing_from_before_the_chart(monkeypatch):
    # each timeframe arrives with its own history, ~1,000 candles on Exness, far older than the
    # chart's: swings from before the chart's first candle are counted but not drawn
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(STRUCT, monkeypatch)
    hist = {60: rt.htf_bars(bars, 60), 240: rt.htf_bars(bars, 240), 15: rt.htf_bars(bars, 15)}
    chart = bars.iloc[3000:].reset_index(drop=True)            # the chart starts 10 days in
    ctx = rt.run(mod, chart, bar_secs=300, labels_kept=10_000, frames=hist)
    t0 = chart["open_time"].iloc[0].timestamp()
    assert ctx._labs.get() and all(lab.position.time >= t0 for lab in ctx._labs.get())
    for s in (2, 3):                                           # yet every swing was counted
        want = _research(bars, [15, 60][s - 2], 5)
        assert ctx._seen.get()[s] == len(want)


@pytest.mark.parametrize("k0, what", [(589, "15m candle of 24 Sep 07:45"),
                                      (2200, "1h candle of 1 Oct 22:00")])
def test_a_candle_loaded_half_way_does_not_leave_a_swing_it_never_made(monkeypatch, k0, what):
    # 5 minutes into these candles the zigzag would confirm a swing; the finished candle makes a
    # new extreme instead, and confirms nothing. Loaded there, chart first at its close, the
    # version computed at load must not become a label that stays for good
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")[:k0 + 60]
    mod = rt.load(STRUCT, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300, labels_kept=10_000)
    ctx = rt.run(mod, bars, bar_secs=300, labels_kept=10_000, realtime_last=60, seam="redo",
                 main_first=True)
    assert [(c, p, t) for c, p, t in zip(ctx._code.get(), ctx._px.get(), ctx._pt.get())] == \
        [(c, p, t) for c, p, t in zip(plain._code.get(), plain._px.get(), plain._pt.get())]


# ── a slot at or below the chart is parked on daily candles (11 Nov 2026) ─────────────────────
# "Execution timed out" on a BTC 15m chart: the 5m and 15m slots, never drawn there, ran the
# chart's own lines a second and a third time (3x on 15m, 4x on 1h). Parked on the daily candles,
# each is one small context; from a daily chart up calc_on cannot go below, so they stay put.
PARKED = [(1, [1, 5, 15, 60, 240]), (5, [5, 1440, 15, 60, 240]), (15, [15, 1440, 1440, 60, 240]),
          (60, [60, 1440, 1440, 1440, 240]), (240, [240, 1440, 1440, 1440, 1440]),
          (1440, [1440, 1440, 1440, 1440, 1440])]


def _walk(minutes, n=400, seed=3):
    t = pd.date_range("2026-01-05", periods=n, freq=f"{minutes}min", tz="UTC")
    c = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.004, n)))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open_time": t, "open": o, "high": np.maximum(o, c) * 1.001,
                         "low": np.minimum(o, c) * 0.999, "close": c})


@pytest.mark.parametrize("chart_m, want", PARKED)
def test_a_slot_at_or_below_the_chart_is_parked_on_daily_candles(monkeypatch, chart_m, want):
    mod = rt.load(STRUCT, monkeypatch)
    ctx = rt.run(mod, _walk(chart_m), bar_secs=chart_m * 60)
    assert [r.minutes for r in ctx._requests] == want
    assert sum(r.minutes == chart_m for r in ctx._requests) == (5 if chart_m >= 1440 else 1)
