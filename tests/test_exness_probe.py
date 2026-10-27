"""
exness/probe_timeframes.py on the stand-in runtime (tests/indie_standin.py): the report it writes
for a well-behaved terminal, and that each way a terminal could misbehave shows up in it -- a probe
that says ALL GOOD whatever happens would be worse than none.
"""
from pathlib import Path

import pytest

import indie_standin as rt

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "exness" / "probe_timeframes.py"
DATA = ROOT / "tests" / "data"


def _rows(ctx):
    return [r.text for r in ctx._rows.get() if r.text]


def test_a_well_behaved_terminal_reads_all_good(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    ctx = rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=300)
    rows = _rows(ctx)
    assert rows[-1].startswith("ALL GOOD"), rows
    n15 = len(rt.htf_bars(bars, 15))
    assert f"15m: {n15} candles" in rows[1]           # every 15m candle, the last one closed too
    assert "seen early 0" in rows[2] and "100% start on round UTC times" in rows[3]
    assert "not asked" in rows[4]
    drawn = [d.text for d in ctx.chart.items(rt.LabelRel)]
    assert sorted(drawn) == sorted(rows)               # what is on the chart is the report


def test_a_terminal_showing_a_candle_before_it_closes_is_reported(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    ctx = rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=300, backfill_first=True)
    rows = _rows(ctx)
    assert rows[-1].startswith("PROBLEM") and "seen early 0" not in rows[3]


def test_a_terminal_merging_by_opening_time_is_reported(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    ctx = rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=300, lookahead=True)
    rows = _rows(ctx)
    assert rows[-1].startswith("PROBLEM") and all("seen early 0" not in r for r in rows[1:4])


def test_values_coming_back_in_the_wrong_order_are_reported(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(PROBE, monkeypatch)
    calc = mod.Candles.calc
    monkeypatch.setattr(mod.Candles, "calc", lambda self: (lambda o: o[:8] + o[8:][::-1])(calc(self)))
    rows = _rows(rt.run(mod, bars, bar_secs=300))
    assert rows[-1].startswith("PROBLEM") and "wrong values 0" not in rows[1]


def test_live_forming_candles_are_counted_and_do_not_fail_the_probe(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    ctx = rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=300, realtime_last=6,
                 live_merge="forming")
    rows = _rows(ctx)
    closes = (bars["open_time"].iloc[-6:].astype("int64") // 10**9 + 300) % 900
    inside = int((closes != 0).sum())                  # a 15m candle opened and not yet closed
    assert f"live: forming {inside} of 6" in rows[1] and rows[-1].startswith("ALL GOOD")
    assert 0 < inside < 6
    assert ctx.chart.peak <= 90


def test_a_finer_timeframe_is_asked_only_when_switched_on(monkeypatch):
    one = rt.read_bars(DATA / "gold_1m_2026-10-21.csv")
    five = rt.htf_bars(one, 5)
    mod = rt.load(PROBE, monkeypatch)
    with pytest.raises(rt.LowerTimeFrame):            # an older Indie refuses: the error says so
        rt.run(mod, five, bar_secs=300, try_lower=True)
    ctx = rt.run(mod, five, bar_secs=300, try_lower=True, lower_ok=True, frames={1: one})
    assert _rows(ctx)[4].startswith(f"1m: {len(one)} candles")


def test_on_a_1m_chart_every_higher_timeframe_is_asked(monkeypatch):
    bars = rt.read_bars(DATA / "gold_1m_2026-10-21.csv")
    ctx = rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=60)
    rows = _rows(ctx)
    assert all("candles" in r for r in rows[1:4]) and rows[-1].startswith("ALL GOOD")
    assert "4h: " in rows[3]
