"""
The bias: quant.indicators.add_bias (the research rule) and exness/bias.py on the stand-in runtime
(tests/indie_standin.py), held to it on every timeframe -- what each timeframe's line says at every
chart candle is add_bias on that timeframe's own candles at its latest CLOSED one.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import indie_standin as rt
from quant.indicators import BIAS_LABELS, add_bias, ema

ROOT = Path(__file__).resolve().parents[1]
BIAS = ROOT / "exness" / "bias.py"
DATA = ROOT / "tests" / "data"
MINS = [1, 5, 15, 60, 240, 1440]


# ── the research rule ──
def _walk(steps, start=100.0):
    mid = start + np.cumsum(steps)
    t = pd.date_range("2026-10-01", periods=len(mid), freq="5min", tz="UTC")
    return pd.DataFrame(dict(open_time=t, open=mid, high=mid + 0.5, low=mid - 0.5, close=mid))


def test_a_clean_uptrend_is_strongly_bullish_and_its_mirror_strongly_bearish():
    # legs of 0.5 a candle with one-point candles: ATR is exactly 1, every 5-candle leg is a swing
    up = [0.0] * 20 + sum(([0.5] * 12 + [-0.5] * 6 for _ in range(30)), [])
    d = add_bias(_walk(up), fast=5, slow=20)
    assert d["bias_structure"].iloc[-1] == 1 and d["bias_ema"].iloc[-1] in (0, 1)
    assert d["bias"].max() == 2 and d["bias_label"].iloc[int(d["bias"].idxmax())] == \
        "Strongly bullish"
    dn = add_bias(_walk([-s for s in up], start=200.0), fast=5, slow=20)
    # the mirror, once both have their first swings (the zigzag starts out seeking a HIGH, so a
    # falling market's first swing is confirmed a leg later)
    np.testing.assert_array_equal(dn["bias_structure"].iloc[80:], -d["bias_structure"].iloc[80:])
    assert dn["bias"].min() == -2


def test_the_bias_is_swings_plus_emas_and_knows_nothing_later():
    b = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    d = add_bias(b)
    np.testing.assert_array_equal(d["bias"], d["bias_structure"] + d["bias_ema"])
    c, ef, es = b["close"], ema(b["close"], 50), ema(b["close"], 200)
    want = np.where((c > ef) & (ef > es), 1, np.where((c < ef) & (ef < es), -1, 0))
    np.testing.assert_array_equal(d["bias_ema"], want)
    assert set(d["bias"]) == {-2, -1, 0, 1, 2}
    assert (d["bias_label"] == d["bias"].map(BIAS_LABELS)).all()
    for k in (300, 2000, 5000):                       # causal: the first k candles alone agree
        part = add_bias(b.iloc[:k])
        np.testing.assert_array_equal(part["bias"], d["bias"].iloc[:k])
    with pytest.raises(ValueError, match="fast < slow"):
        add_bias(b, fast=200, slow=50)


# ── the Exness panel ──
def _frame(bars, minutes, chart_minutes):
    if minutes == chart_minutes:
        return bars
    f = rt.htf_bars(bars, minutes)
    last_close = bars["open_time"].iloc[-1].timestamp() + chart_minutes * 60
    return f[(f["open_time"].astype("int64") // 10**9 + minutes * 60) <= last_close]


@pytest.mark.parametrize("name, secs", [("usdjpy_5m_2026-10-22.csv", 300),
                                        ("gold_1m_2026-10-21.csv", 60)])
def test_every_timeframes_line_is_add_bias_at_its_last_closed_candle(monkeypatch, name, secs):
    bars = rt.read_bars(DATA / name)
    cm = secs // 60
    seen = []
    ctx = rt.run(rt.load(BIAS, monkeypatch), bars, bar_secs=secs,
                 spy=lambda c: seen.append(list(c._last.get())))
    seen = np.array(seen, dtype=float)
    close = (bars["open_time"].astype("int64") // 10**9).to_numpy(float) + secs
    for s, m in enumerate(MINS):
        if m < cm:
            assert np.isnan(seen[:, s * 8]).all()     # below the chart: never asked
            continue
        f = _frame(bars, m, cm).reset_index(drop=True)
        d = add_bias(f)
        ef, es = ema(f["close"], 50).to_numpy(), ema(f["close"], 200).to_numpy()
        fclose = (f["open_time"].astype("int64") // 10**9).to_numpy(float) + m * 60
        j = np.searchsorted(fclose, close, side="right") - 1        # its last closed candle
        has = j >= 0
        for q, col in enumerate([d["bias"], d["bias_structure"], d["bias_ema"], ef, es,
                                 f["close"]]):
            want = np.full(len(bars), np.nan)
            want[has] = np.asarray(col, float)[j[has]]
            np.testing.assert_array_equal(seen[:, s * 8 + q], want, err_msg=f"{m}m col {q}")
    rows = [r.text for r in ctx._rows.get()]
    assert len(rows) == 6 and rows[MINS.index(cm)].split()[0] == f"{cm}m"
    last = add_bias(bars)["bias_label"].iloc[-1]
    assert last in rows[MINS.index(cm)]


def test_the_panel_says_why_and_what_it_cannot_show(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    ctx = rt.run(rt.load(BIAS, monkeypatch), bars, bar_secs=300)
    rows = [r.text for r in ctx._rows.get()]
    assert rows[0].startswith("1m  -  below this chart")
    assert "EMAs warming up" in rows[5] or "no closed candle" in rows[5]   # 30 days of 1D
    d = add_bias(_frame(bars, 15, 5).reset_index(drop=True)).iloc[-1]
    why = {1: "HH + HL", -1: "LH + LL", 0: "swings mixed"}[int(d["bias_structure"])]
    assert rows[2].startswith("15m  " + d["bias_label"]) and why in rows[2]
    drawn = sorted(r.text for r in ctx.chart.items(rt.LabelRel))
    assert drawn == sorted(rows)
    colours = {2: ("green", 0.65), 1: ("green", 0.35), 0: ("rgb(128,128,128)", 0.3),
               -1: ("red", 0.35), -2: ("red", 0.65)}
    seen = set()
    for s, m in enumerate(MINS):
        if m < 5 or np.isnan(ctx._last.get()[s * 8]):
            continue
        score = int(ctx._last.get()[s * 8])
        lab = ctx._rows.get()[s]
        assert (lab.bg_color.name, lab.bg_color.alpha) == colours[score], m
        seen.add(score)
    one = rt.read_bars(DATA / "gold_1m_2026-10-21.csv")       # and every score's colour, on 1m
    for k in (100, 200, 300, 400, 500):
        c = rt.run(rt.load(BIAS, monkeypatch), one.iloc[:k], bar_secs=60)
        score = int(c._last.get()[0])
        lab = c._rows.get()[0]
        assert (lab.bg_color.name, lab.bg_color.alpha) == colours[score]
        seen.add(score)
    assert {2, -2} & seen and len(seen) >= 3


@pytest.mark.parametrize("variant", [dict(live_merge="forming", realtime_last=60),
                                     dict(backfill_first=True), dict(lookahead=True),
                                     dict(realtime_last=400, seam="redo"),      # Exness's seam,
                                     dict(realtime_last=400, seam="again"),     # both endings
                                     dict(realtime_last=40, rerun=range(0, 6300, 7))])
def test_a_forming_candle_or_a_rerun_changes_nothing(monkeypatch, variant):
    # compared on EVERY candle: the file ends at 00:00 UTC, where every timeframe closes at once,
    # so the final panel alone cannot see a state taken early
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(BIAS, monkeypatch)
    want, got = [], []
    plain = rt.run(mod, bars, bar_secs=300, spy=lambda c: want.append(rt.plain(c._last.get())))
    ctx = rt.run(mod, bars, bar_secs=300, spy=lambda c: got.append(rt.plain(c._last.get())),
                 **variant)
    if variant.get("lookahead"):
        # merged by OPENING time, a candle cut short by a market close (Friday's 20:00 4h candle
        # ends at 21:00, not 24:00) is never shown once closed: its state is taken a candle late.
        # The guarantee there is NEVER AHEAD -- every state used had closed by that candle's close
        close = (bars["open_time"].astype("int64") // 10**9).to_numpy(float) + 300
        for k, row in enumerate(got):
            for s, m in enumerate(MINS):
                if row[s * 8 + 7] != "nan":
                    assert row[s * 8 + 7] + m * 60 <= close[k]
        assert sum(g != w for g, w in zip(got, want)) < 0.05 * len(want)
    else:
        assert got == want
    assert [r.text for r in ctx._rows.get()] == [r.text for r in plain._rows.get()]
    assert rt.state(ctx)[1] == rt.state(plain)[1]
    if variant.get("seam"):                       # its own state too, after the seam
        assert rt.sec_state(ctx) == rt.sec_state(plain)
    assert ctx.chart.peak <= 90


def test_lower_timeframes_only_when_asked(monkeypatch):
    one = rt.read_bars(DATA / "gold_1m_2026-10-21.csv")
    five = rt.htf_bars(one, 5)
    mod = rt.load(BIAS, monkeypatch)
    with pytest.raises(rt.LowerTimeFrame):
        rt.run(mod, five, bar_secs=300, try_lower=True)
    ctx = rt.run(mod, five, bar_secs=300, try_lower=True, lower_ok=True, frames={1: one})
    row = ctx._rows.get()[0].text
    assert row.startswith("1m  ") and add_bias(one)["bias_label"].iloc[-1] in row
