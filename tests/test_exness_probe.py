"""
exness/probe_timeframes.py (v2) on the stand-in runtime (tests/indie_standin.py): the report it
writes for a well-behaved terminal, and that each way a terminal could misbehave shows up in it --
a probe that says OK whatever happens would be worse than none.

The first probe on Exness (US Tech 100, 1m, 27 Oct 2026) read: every construct worked, ~1,000
candles per timeframe (15m 15 days, 1h 62, 4h 228), 100% on round UTC times, wrong values 0,
"seen early 1" on each timeframe and "forming 1 of 1": at the switch to live, each timeframe's
still-forming candle had been computed as if closed. v2 says WHERE an early view happens and
whether that candle is later computed twice; the stand-in's `seam` models both possibilities.
"""
from pathlib import Path

import pytest

import indie_standin as rt

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "exness" / "probe_timeframes.py"
DATA = ROOT / "tests" / "data"


def _rows(ctx):
    return [r.text for r in ctx._rows.get() if r.text]


def _run(monkeypatch, **kw):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    return bars, _rows(rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=300, **kw))


def test_a_well_behaved_terminal_reads_ok_and_asks_to_wait_for_a_close(monkeypatch):
    bars, rows = _run(monkeypatch)
    assert rows[0].startswith("Timeframe probe v2: chart 5m")
    n15 = len(rt.htf_bars(bars, 15))
    assert rows[1].startswith(f"15m: {n15} candles")
    assert "early: history 0, live 0" in rows[2] and "100% on round UTC times" in rows[3]
    assert "wrong 0" in rows[3] and "not asked" in rows[4]
    assert rows[5].startswith("OK: nothing early on history")
    assert rows[6].startswith("Leave it running")                  # nothing has closed live yet


def test_the_exness_seam_shows_as_live_early_views_and_is_not_a_problem(monkeypatch):
    # what Exness did: the forming candle, computed as if closed, handed over when going live.
    # Live from 03:50: every timeframe's candle is still open then (the file ends at 04:00)
    bars, rows = _run(monkeypatch, realtime_last=2, seam="redo")
    for r in rows[1:4]:
        assert "early: history 0, live 1" in r and "forming live 1/2" in r
    assert rows[5].startswith("OK")


def test_a_candle_computed_twice_with_new_values_is_reported(monkeypatch):
    # 40 live 5m candles: 15m and 1h candles close live after the seam
    bars, rows = _run(monkeypatch, realtime_last=40, seam="again")
    assert "twice 1 (changed 1)" in rows[1] and "twice 1 (changed 1)" in rows[2]
    assert rows[6].startswith("NOTE: a candle was computed twice with different values")


def test_a_candle_redone_from_scratch_reads_clean(monkeypatch):
    bars, rows = _run(monkeypatch, realtime_last=40, seam="redo")
    assert "twice 0 (changed 0)" in rows[1] and "twice 0 (changed 0)" in rows[2]
    assert rows[5].startswith("OK") and rows[6] == "Recomputed cleanly: no candle computed twice"


@pytest.mark.parametrize("variant", [dict(backfill_first=True), dict(lookahead=True)])
def test_early_views_on_history_are_a_problem(monkeypatch, variant):
    bars, rows = _run(monkeypatch, **variant)
    assert rows[5].startswith("PROBLEM") and any("early: history 0" not in r for r in rows[1:4])


def test_values_coming_back_in_the_wrong_order_are_reported(monkeypatch):
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(PROBE, monkeypatch)
    calc = mod.Candles.calc
    monkeypatch.setattr(mod.Candles, "calc",
                        lambda self: (lambda o: o[:10] + o[10:][::-1])(calc(self)))
    rows = _rows(rt.run(mod, bars, bar_secs=300))
    assert rows[5].startswith("PROBLEM") and "wrong 0" not in rows[1]


def test_live_forming_candles_are_counted(monkeypatch):
    bars, rows = _run(monkeypatch, realtime_last=6, live_merge="forming")
    closes = (bars["open_time"].iloc[-6:].astype("int64") // 10**9 + 300) % 900
    inside = int((closes != 0).sum())                  # a 15m candle opened and not yet closed
    assert f"forming live {inside}/6" in rows[1] and rows[5].startswith("OK")
    assert 0 < inside < 6


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
    rows = _rows(rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=60))
    assert all("candles" in r for r in rows[1:4]) and rows[5].startswith("OK")
    assert rows[3].startswith("4h: ")


@pytest.mark.parametrize("seam, verdict", [("redo", "Recomputed cleanly"),
                                           ("again", "NOTE: a candle was computed twice")])
def test_on_a_1h_chart_the_4h_close_alone_gives_the_verdict(monkeypatch, seam, verdict):
    # Exness, 27 Oct 2026: on a 1h chart only 4h is asked; its one live close answered the
    # question (twice 0) while v2 still said "leave it running" -- it waited for 15m and 1h
    bars = rt.htf_bars(rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv"), 60)
    rows = _rows(rt.run(rt.load(PROBE, monkeypatch), bars, bar_secs=3600, realtime_last=6,
                        seam=seam))
    assert "not asked" in rows[1] and "not asked" in rows[2] and "early: history 0" in rows[3]
    assert rows[5].startswith("OK") and rows[6].startswith(verdict)
