"""
The news strip on pattern_figure. It is APPENDED, so the first promise is that a chart with news is
the chart without news plus some traces and one menu, with everything that was there untouched.
Then: markers sit on the bottom edge whatever the price axis does, each release lands on its own
candle in the chart's zone, the dropdown addresses only news traces, and rewind clips only the
actuals -- the schedule was known in advance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.news.calendar import finish
from quant.viz.news import SHOW
from quant.viz.studies import pattern_figure


def _ohlc(n=1500, tz="UTC"):
    rng = np.random.default_rng(2)
    close = 30000 + np.cumsum(rng.normal(0, 6, n))
    open_ = np.r_[close[0], close[:-1]]
    t = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open_time": t, "t": t.tz_convert(tz), "open": open_,
                         "high": np.maximum(open_, close) + 3, "low": np.minimum(open_, close) - 3,
                         "close": close, "volume": 1.0})


def _rel(t, title, impact="high", cur="USD", actual=None):
    return dict(t=pd.Timestamp(t, tz="UTC"), time_q="exact", currency=cur, title=title,
                impact=impact, actual=actual, forecast="0.3%", previous="0.2%", source="ff_feed")


NEWS = finish(pd.DataFrame([
    _rel("2026-01-06 13:30", "CPI m/m", actual="0.4%"),           # critical
    _rel("2026-01-06 13:30", "Core CPI m/m"),                      # critical, same minute
    _rel("2026-01-07 15:00", "ISM Services PMI", actual="52.1"),   # high
    _rel("2026-01-08 13:30", "Philly Fed", impact="medium"),
    _rel("2026-01-09 09:00", "Some Low Thing", impact="low"),
    _rel("2026-03-01 13:30", "Outside the chart"),
]))


def _fig(base, **kw):
    return pattern_figure(base, patterns=[], sr_zones=False, emas=(50,), chart_tf="5m",
                          line_picker=False, **kw)


def _json(x):
    import json
    from plotly.utils import PlotlyJSONEncoder
    return json.dumps(x.to_plotly_json(), cls=PlotlyJSONEncoder, sort_keys=True)


def test_news_only_appends_everything_that_was_there_is_untouched():
    base = _ohlc()
    a = _fig(base)
    b = _fig(base, news=NEWS)
    assert len(b.data) > len(a.data)
    for i in range(len(a.data)):
        assert _json(a.data[i]) == _json(b.data[i])
    assert [_json(m) for m in a.layout.updatemenus] == [_json(m) for m in b.layout.updatemenus[:-1]]
    # and an empty calendar, or none, changes nothing at all
    assert _json(_fig(base, news=NEWS.iloc[:0])) == _json(a)
    assert _json(_fig(base, news=None)) == _json(a)


def test_markers_sit_on_the_bottom_edge_on_their_own_fixed_axis():
    f = _fig(_ohlc(), news=NEWS)
    news = [tr for tr in f.data if tr.meta in ("news", "news_result")]
    axes = {tr.yaxis for tr in news}
    assert len(axes) == 1
    ax = f.layout["yaxis" + axes.pop()[1:]]
    assert list(ax.range) == [0, 1] and ax.fixedrange and ax.overlaying == "y" and not ax.visible
    assert all(max(y for y in tr.y if y is not None) <= 1 for tr in news)


def test_one_marker_per_release_minute_and_level_and_nothing_outside_the_chart():
    f = _fig(_ohlc(), news=NEWS)
    marks = {tr.legendgroup: tr for tr in f.data if tr.meta == "news" and tr.mode == "markers"}
    assert set(marks) == {"news:critical", "news:high", "news:medium", "news:low"}
    assert len(marks["news:critical"].x) == 1                       # CPI + Core CPI together
    assert "Core CPI m/m" in marks["news:critical"].customdata[0]
    assert "Outside" not in " ".join(c for tr in marks.values() for c in tr.customdata)


@pytest.mark.parametrize("tz", ["UTC", "Asia/Karachi"])
def test_a_release_lands_on_its_own_candle_in_the_charts_zone(tz):
    base = _ohlc(tz=tz)
    f = _fig(base, news=NEWS)
    cpi = next(tr for tr in f.data if tr.legendgroup == "news:critical" and tr.mode == "markers")
    want = pd.Timestamp("2026-01-06 13:30", tz="UTC").tz_convert(tz)
    assert pd.Timestamp(cpi.x[0]) == want
    assert str(pd.Timestamp(cpi.x[0]).tz) == str(want.tz)


def test_the_dropdown_addresses_only_news_traces_and_opens_where_asked():
    f = _fig(_ohlc(), news=NEWS, news_show="critical")
    menu = f.layout.updatemenus[-1]
    news_ix = {i for i, tr in enumerate(f.data) if tr.meta in ("news", "news_result")}
    labels = [b.label for b in menu.buttons]
    assert labels == ["News: critical", "high+", "medium+", "all", "off"] and menu.active == 0
    for b in menu.buttons:
        vis, ix = b.args
        assert set(ix) == news_ix
    off = menu.buttons[-1].args[0]["visible"]
    assert set(off) == {"legendonly"}
    # opening state: only critical on
    for i in news_ix:
        on = f.data[i].legendgroup == "news:critical"
        assert f.data[i].visible == (True if on else "legendonly")
    with pytest.raises(ValueError, match="news_show"):
        _fig(_ohlc(), news=NEWS, news_show="loud")


def test_schedule_hover_has_no_actuals_and_only_the_actuals_rewind():
    f = _fig(_ohlc(), news=NEWS, rewind=True)
    rw = set(f.layout.meta["rewind"])
    sched = [i for i, tr in enumerate(f.data) if tr.meta == "news"]
    res = [i for i, tr in enumerate(f.data) if tr.meta == "news_result"]
    assert res and not (set(sched) & rw) and set(res) <= rw
    assert all("A 0.4%" not in c for i in sched for c in (f.data[i].customdata or ()))
    assert any("A 0.4%" in c for i in res for c in f.data[i].customdata)
    assert any("better for USD" in c for i in res for c in f.data[i].customdata)


def test_release_lines_are_drawn_for_critical_and_high_only():
    f = _fig(_ohlc(), news=NEWS)
    lines = [tr.legendgroup for tr in f.data if tr.meta == "news" and tr.mode == "lines"]
    assert sorted(lines) == ["news:critical", "news:high"]
    assert set(SHOW["high"]) == {"critical", "high"}
