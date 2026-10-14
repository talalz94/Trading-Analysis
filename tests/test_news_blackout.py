"""
NEWS_BLACKOUT in a strategy file: no new trade around a covered release. Off must change nothing;
on, a trade whose entry bar CLOSES within the window is not taken, and one a minute outside it is.
The calendar is planted by hand around a real trade, so the boundary is known before the run.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHED = (ROOT / "data" / "binance_spot_BTCUSDT_5m.parquet").is_file()
pytestmark = pytest.mark.skipif(not CACHED, reason="needs the cached BTCUSDT data")
WINDOW = dict(VIEW_START="2026-06-01", END="2026-06-20", WARMUP_START="2026-05-15")


@pytest.fixture(scope="module")
def lab():
    from quant.research.fvg_lab import fvg_lab
    from quant.research.variant import variant
    from quant.research.workspace import lab_config
    return fvg_lab(variant(lab_config(), **WINDOW))


def calendar_at(*times):
    from quant.news.calendar import finish
    return finish(pd.DataFrame([dict(t=pd.Timestamp(t), time_q="exact", currency="USD",
                                     title="CPI m/m", impact="high", actual=None,
                                     forecast=None, previous=None, source="test")
                                for t in times]))


def plant(monkeypatch, *times):
    import quant.news
    ev = calendar_at(*times)
    monkeypatch.setattr(quant.news, "load_events", lambda *a, **k: ev)


def entry_close(lab, i=1):
    tr = lab.trades.iloc[i]
    return pd.Timestamp(tr["entry_time"]).tz_convert("UTC") + pd.Timedelta(minutes=5), tr


ON = dict(before=30, after=30, coverage={"USD": "critical"})


def same(a, b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_the_strategy_ships_with_it_off_and_off_changes_nothing(lab):
    from quant.research.workspace import load_strategy
    assert load_strategy("tl_break_confirm")["NEWS_BLACKOUT"] is None
    same(lab.retrade(NEWS_BLACKOUT=None).trades, lab.trades)


def test_a_calendar_with_nothing_in_the_window_warns_and_changes_nothing(lab, monkeypatch):
    plant(monkeypatch)
    with pytest.warns(UserWarning, match="nothing is blacked out"):
        out = lab.retrade(NEWS_BLACKOUT=ON).trades
    same(out, lab.trades)


@pytest.mark.parametrize("minutes, blocked", [(30, True), (35, False), (-30, True), (-35, False)])
def test_the_window_is_measured_from_the_entry_bars_close(lab, monkeypatch, minutes, blocked):
    close, tr = entry_close(lab)
    # +m: the release is m minutes AFTER the bar closes ("before" side); -m: m minutes before it
    plant(monkeypatch, close + pd.Timedelta(minutes=minutes))
    out = lab.retrade(NEWS_BLACKOUT=ON).trades
    taken = bool((out["entry_time"] == tr["entry_time"]).any())
    assert taken == (not blocked)
    if blocked:                         # and nothing else opened inside the window either
        lo, hi = close - pd.Timedelta(minutes=30), close + pd.Timedelta(minutes=30)
        closes = pd.to_datetime(out["entry_time"], utc=True) + pd.Timedelta(minutes=5)
        assert not closes.between(lo, hi).any()


def test_trades_before_the_first_window_are_untouched(lab, monkeypatch):
    close, tr = entry_close(lab, 3)
    plant(monkeypatch, close + pd.Timedelta(minutes=10))
    out = lab.retrade(NEWS_BLACKOUT=ON).trades
    before = lab.trades[lab.trades["entry_time"] < tr["entry_time"]]
    same(out[out["entry_time"] < tr["entry_time"]], before)


def test_a_mistyped_setting_is_refused_by_name(lab):
    with pytest.raises(ValueError, match="unknown keys.*befor"):
        lab.retrade(NEWS_BLACKOUT=dict(befor=30))


def test_retrade_equals_a_full_rebuild(lab, monkeypatch):
    from quant.research.fvg_lab import fvg_lab
    from quant.research.variant import variant
    from quant.research.workspace import lab_config
    close, _ = entry_close(lab)
    plant(monkeypatch, close + pd.Timedelta(minutes=20))
    fast = lab.retrade(NEWS_BLACKOUT=ON).trades
    slow = fvg_lab(variant(lab_config(), NEWS_BLACKOUT=ON, **WINDOW)).trades
    same(fast, slow)
    assert len(fast) < len(lab.trades) or not fast["entry_time"].isin(
        [lab.trades.iloc[1]["entry_time"]]).any()
