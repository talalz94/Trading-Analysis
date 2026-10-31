"""
Every Exness indicator on every kind of chart: 1m to weekly, including timeframes no list names
(3m, 30m), each with live candles and closed candles run again. Nothing may fail, nothing may move
while a candle forms, no update may come near Exness's 100 drawing changes, and a live run must
end exactly where a plain one does. The other test files hold each indicator to the research code;
this one is for the charts nobody thought to test.
"""
from pathlib import Path

import pytest

import indie_standin as rt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
FILES = ["probe_timeframes", "trendlines", "trendlines_mtf", "structure", "bias", "levels",
         "fair_value_gaps"]


def _chart(minutes):
    """A chart of `minutes` candles, from the gold 1m or USDJPY 5m fixtures."""
    if minutes < 5:
        one = rt.read_bars(DATA / "gold_1m_2026-10-21.csv")
        return rt.htf_bars(one, minutes) if minutes > 1 else one
    five = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    return (five if minutes == 5 else rt.htf_bars(five, minutes)).iloc[-1500:].reset_index(
        drop=True)


@pytest.mark.parametrize("minutes", [1, 3, 5, 15, 30, 60, 240, 1440, 10080])
@pytest.mark.parametrize("name", FILES)
def test_every_indicator_on_every_chart(monkeypatch, name, minutes):
    path = ROOT / "exness" / f"{name}.py"
    if name == "fair_value_gaps":
        pytest.skip("draws Rectangles: tests/test_exness_fvg.py runs it on its own stand-in")
    bars = _chart(minutes)
    mod = rt.load(path, monkeypatch)
    plain = rt.run(mod, bars, bar_secs=minutes * 60)
    live = min(40, len(bars) // 3)
    ctx = rt.run(mod, bars, bar_secs=minutes * 60, realtime_last=live,
                 rerun=range(0, len(bars), 11), seam="redo")
    if name != "probe_timeframes":                 # the probe reports live events by design
        assert rt.state(ctx)[1] == rt.state(plain)[1]
    assert ctx.chart.peak <= 90
    # a panel (the probe's report, the bias lines) is drawn as soon as the chart is live, from
    # closed candles only; everything else waits for a close
    assert ctx.chart.forming == 0 or name in ("probe_timeframes", "bias")


@pytest.mark.parametrize("name, cls", [("trendlines", "Main"), ("trendlines_mtf", "Lines")])
def test_a_line_checked_live_deeper_than_anything_while_loading(monkeypatch, name, cls):
    # Exness, Gold 1m, 30 Oct 2026: "Series offset=510 is out of size=506". A candle series
    # grows to the depth read while the chart LOADS; live, a line check reaching further back
    # than any did while loading failed. Go live just before the deepest check of the run --
    # deeper than every one before it -- and the fitter must read its own kept candles instead
    bars = rt.read_bars(DATA / "usdjpy_5m_2026-10-22.csv")
    mod = rt.load(ROOT / "exness" / f"{name}.py", monkeypatch)
    plain = rt.run(mod, bars, bar_secs=300)
    K = getattr(mod, cls)
    clear, deep = K._clear, {}

    def spy(self, slope, icpt, x0, kind, wick):
        if self.time_frame.to_minutes() == 5:                 # the chart's own timeframe
            deep[self.bar_index] = max(deep.get(self.bar_index, 0), self.bar_index - x0)
        return clear(self, slope, icpt, x0, kind, wick)
    monkeypatch.setattr(K, "_clear", spy)
    rt.run(mod, bars, bar_secs=300)
    monkeypatch.setattr(K, "_clear", clear)
    k = max(deep, key=lambda b: (deep[b], -b))                # the first of the deepest
    assert deep[k] > max(v for b, v in deep.items() if b < k) and deep[k] > 300
    ctx = rt.run(mod, bars, bar_secs=300, realtime_last=len(bars) - k + 3)
    assert rt.state(ctx)[1] == rt.state(plain)[1]
