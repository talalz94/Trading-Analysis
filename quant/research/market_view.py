"""
THE MARKET VIEW: one instrument's latest days, every timeframe's structure, and the news. No
strategy, no trades -- the lab's chart without the lab's sim.

    view = market_view(cfg)     # loads to the latest CLOSED candle, every timeframe a layer needs
    view.summary()              # the window, how fresh it is, the releases in it and coming up
    view.figure()               # the chart

`cfg` is notebooks/market_view_config.py (workspace.market_view_config()); every value is read from
it and nothing here defaults a setting the config owns.

THE WINDOW IS COUNTED IN CANDLES, ending at the latest closed one. "The last two days" of a market
that shuts at the weekend is two days of TRADING: gold on a Monday morning reaches back to Thursday
instead of showing a weekend with a few hours of candles in it. For a 24/7 market (BTC) the candles
and the calendar agree.

A CANDLE STILL FORMING IS DROPPED from every timeframe. Binance and Dukascopy both return the bar in
progress, and a line, gap or zone fitted on it would move as that bar does -- a repaint, on exactly
the chart you look at live. Every frame therefore ends at its last CLOSED bar, which also makes the
view causal in the same sense as everything else in this project.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

from .fvg_lab import _get

# calendar days added around the candles the window needs, tried in turn, so a shut market (a
# weekend, Christmas) still yields enough history before the loader gives up
LOOKBACK_EXTRA_DAYS = (4, 10, 30)
FEATURES = (("SHOW_TRENDLINES", "LINE_TFS", "trendlines"), ("FVG_DRAW", "FVG_TFS", "gaps"),
            ("SHOW_SR_ZONES", "SR_TFS", "zones"))


def instrument(cfg) -> Tuple[str, str, str]:
    """(name, symbol, source) of the config's INSTRUMENT."""
    name, table = cfg.INSTRUMENT, cfg.INSTRUMENTS
    if name not in table:
        raise KeyError(f"INSTRUMENT = {name!r} is not one of INSTRUMENTS: {sorted(table)}. Add it "
                       f"there as dict(symbol=..., source='binance' | 'dukascopy')")
    spec = table[name]
    return name, spec["symbol"], spec.get("source", "binance")


def _check(cfg) -> None:
    """Each switched-on feature names the chart's own timeframe: its layer is the one always drawn
    (pattern_figure has no way to leave it out), so a list without it would not mean what it says."""
    from ..data.timeframe import tf_minutes
    for on, lst, what in FEATURES:
        tfs = tuple(_get(cfg, lst, ()) or ())
        for t in tfs:
            tf_minutes(t)                          # an unknown timeframe fails here, by name
        if _get(cfg, on, True) and cfg.TF not in tfs:
            raise ValueError(f"{lst} = {tfs} leaves out the chart's own timeframe {cfg.TF!r}, "
                             f"whose {what} are always drawn while {on} is on. Add it, or set "
                             f"{on} = False")


OPEN_KEYS = ("trendlines", "gaps", "zones")


def open_with(cfg) -> Tuple[str, ...]:
    """OPEN_WITH as the keys pattern_figure's `start_visible` reads. The chart's own timeframe and
    the others are named differently there ("fvg_bull" but "1h::FVG bullish"), which is exactly
    what a config should not have to know."""
    ow = _get(cfg, "OPEN_WITH", None) or {}
    bad = sorted(set(ow) - set(OPEN_KEYS))
    if bad:
        raise KeyError(f"OPEN_WITH has unknown keys {bad}; it takes {OPEN_KEYS}")
    own, keys = cfg.TF, []
    for tf in ow.get("trendlines", ()):
        keys += (["support_trendline", "resistance_trendline"] if tf == own else
                 [f"{tf}::support trendline", f"{tf}::resistance trendline"])
    for tf in ow.get("gaps", ()):
        for side in tuple(_get(cfg, "FVG_SHOW", ("bull", "bear"))):
            keys.append(f"fvg_{side}" if tf == own else f"{tf}::FVG {side}ish")
    for tf in ow.get("zones", ()):
        keys.append("sr_zones" if tf == own else f"{tf}::S/R zones")
    return tuple(keys)


# the data layer narrates every fetch in a dozen INFO lines: right for a script, noise in a
# notebook that prints its own summary. Raised to warnings while the view loads, then restored
_CHATTY = ("quant.data", "quant.data.dukascopy", "charting.binance_cache", "DUKASCRIPT")


class _WarningsOnly(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING


@contextmanager
def _quiet():
    # a FILTER, not setLevel: the Binance fetcher sets its own logger's level on every call
    f = _WarningsOnly()
    loggers = [logging.getLogger(n) for n in _CHATTY]
    try:
        for lg in loggers:
            lg.addFilter(f)
        yield
    finally:
        for lg in loggers:
            lg.removeFilter(f)


def _closed(df: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    """Only the bars that have CLOSED by `now`."""
    from ..data.timeframe import tf_minutes
    if not len(df):
        return df
    shut = pd.to_datetime(df["open_time"], utc=True) + pd.Timedelta(minutes=tf_minutes(tf))
    return df[(shut <= now).to_numpy()].reset_index(drop=True)


def _refresh_news(now) -> List[str]:
    """The calendar's sources, each in its own try. Returns what went wrong, if anything; a
    refusal from JBlanked for want of credits is expected and not reported."""
    from ..news import update
    try:
        lines = update.update_all(now=now, rebuild=False)
    except Exception as e:                       # the chart is worth having without fresh news
        return [f"news not refreshed: {type(e).__name__}: {e}"]
    return [f"news: {l}" for l in lines if "FAILED" in l and "credits" not in l]


@dataclass
class MarketView:
    """Everything the market view looks at, already loaded."""

    cfg: Any
    name: str
    symbol: str
    source: str
    df: pd.DataFrame                    # the chart's timeframe, warm-up included
    frames: Dict[str, pd.DataFrame]     # every other timeframe a layer needs, warm-up included
    view_start: pd.Timestamp            # the first candle on screen
    latest: pd.Timestamp                # the OPEN of the latest closed chart candle
    now: pd.Timestamp
    news: pd.DataFrame                  # covered releases from the warm-up to tomorrow
    notes: List[str] = field(default_factory=list)

    @property
    def window(self) -> pd.DataFrame:
        """The candles on screen."""
        t = pd.to_datetime(self.df["open_time"], utc=True)
        return self.df[(t >= self.view_start).to_numpy()]

    @property
    def fvg_kwargs(self) -> Dict[str, Any]:
        c = self.cfg
        return dict(mitigation=c.FVG_MITIGATION, min_size_atr=c.FVG_MIN_SIZE_ATR,
                    min_disp_atr=_get(c, "FVG_MIN_DISP_ATR", 0.0),
                    max_age_bars=_get(c, "FVG_MAX_AGE_BARS", 500),
                    atr_period=_get(c, "FVG_ATR_PERIOD", 14))

    def _layers(self, key: str, on: str) -> Dict[str, pd.DataFrame]:
        """The OTHER timeframes' frames for one feature (the chart's own layer is drawn anyway)."""
        if not _get(self.cfg, on, True):
            return {}
        return {k: self.frames[k] for k in (_get(self.cfg, key, ()) or ()) if k != self.cfg.TF}

    # ---------------------------------------------------------------- what is there
    def summary(self) -> None:
        """The window, how fresh its last candle is, and the news in it and just after it."""
        from ..data.timeframe import tf_minutes
        c, tz = self.cfg, _get(self.cfg, "NEWS_TZ", "Asia/Karachi")
        bar = pd.Timedelta(minutes=tf_minutes(c.TF))
        shut = self.latest + bar
        age = int((self.now - shut).total_seconds() // 60)
        print(f"{self.name}  ({self.symbol}, {self.source})   {c.TF} candles, the last "
              f"{c.DAYS} days of trading")
        print(f"window   {self.view_start:%a %d %b %H:%M} -> {shut:%a %d %b %H:%M} UTC   "
              f"({len(self.window):,} candles)")
        print(f"latest   closed {shut:%H:%M} UTC, {age:,} min ago"
              + ("   (the market is shut)" if age > 30 else ""))
        print("frames   " + "  |  ".join(f"{k} {len(f):,}" for k, f in
                                         [(c.TF, self.df), *self.frames.items()])
              + "   (bars, warm-up included)")
        if self.news is not None and len(self.news):
            ev = self.news
            t = pd.to_datetime(ev["t"], utc=True)
            show = ev["level"].isin(["high", "critical"])
            seen = ev[show & (t >= self.view_start) & (t <= shut)]
            ahead = ev[show & (t > self.now) & (t <= self.now + pd.Timedelta(hours=24))]
            for label, part in ((f"\nnews in the window ({tz}, high and critical)", seen),
                                ("\ncoming up in the next 24 h", ahead)):
                print(label + ("" if len(part) else ": none"))
                for e in part.to_dict("records"):
                    when = pd.Timestamp(e["t"]).tz_convert(tz).strftime("%a %d %b %H:%M")
                    nums = "  ".join(f"{k} {e[k]}" for k in ("actual", "forecast", "previous")
                                     if isinstance(e.get(k), str) and e[k])
                    mean = (f"  -> {e['sentiment']} for {e['currency']}"
                            if isinstance(e.get("sentiment"), str) and e["sentiment"] else "")
                    print(f"  {when}  {e['currency']} [{e['level']}] {e['title']}  {nums}{mean}")
        for n in self.notes:
            print("note: " + n)

    # ---------------------------------------------------------------- the chart
    def figure(self, **overrides):
        """The chart, as the config asks. `overrides` go to `pattern_figure`."""
        from ..data.timeframe import tf_minutes
        from ..viz import pattern_figure
        c = self.cfg
        lines = self._layers("LINE_TFS", "SHOW_TRENDLINES")
        gaps = self._layers("FVG_TFS", "FVG_DRAW")
        zones = self._layers("SR_TFS", "SHOW_SR_ZONES")
        shut = self.latest + pd.Timedelta(minutes=tf_minutes(c.TF))

        kw = dict(
            patterns=[], trendlines=_get(c, "SHOW_TRENDLINES", True),
            trendline_kwargs=c.TRENDLINE,
            # a line timeframe draws its own gaps only if FVG_TFS names it too
            tf_frames=lines, tf_fvgs={k: k in gaps for k in lines},
            fvg_frames={k: f for k, f in gaps.items() if k not in lines},
            fvgs=_get(c, "FVG_DRAW", True), fvg_kwargs=self.fvg_kwargs,
            fvg_extend=_get(c, "FVG_EXTEND", True),
            fvg_show=tuple(_get(c, "FVG_SHOW", ("bull", "bear"))),
            max_fvg_picker=_get(c, "MAX_FVG_PICKER", 6),
            sr_zones=_get(c, "SHOW_SR_ZONES", True), sr_frames=zones,
            sr_sensitivity=_get(c, "SR_SENSITIVITY", "normal"),
            zone_lookback_bars=_get(c, "ZONE_LOOKBACK", 2000),
            max_zones=_get(c, "SR_MAX_ZONES", 40), max_sr_picker=_get(c, "MAX_SR_PICKER", 6),
            sr_break_bars=_get(c, "SR_BREAK_BARS", None),
            view_start=self.view_start, emas=tuple(_get(c, "EMAS", ()) or ()), mtf_emas=None,
            chart_tf=c.TF, ema_frames=self.frames,
            max_picker_instances=_get(c, "MAX_PICKER", 24),
            max_shapes_per_tf=_get(c, "MAX_SHAPES_TF", 12),
            start_visible=open_with(c),
            height=_get(c, "HEIGHT", 900), rewind=_get(c, "REWIND", False),
            # short: the button rows sit over the left of the title, and the legend already
            # names every layer and timeframe
            title=(f"{self.name} {c.TF}  ·  {self.symbol}  ·  last {c.DAYS} days to "
                   f"{shut:%a %d %b %H:%M} UTC"))
        if self.news is not None and len(self.news):
            kw.update(news=self.news, news_show=_get(c, "NEWS_SHOW", "high"),
                      news_tz=_get(c, "NEWS_TZ", "Asia/Karachi"))
        # patterns, S/R style, period levels, sessions -- each only if the config asks for it
        from .chart_layers import chart_layers
        kw.update(chart_layers(c, symbol=self.symbol, source=self.source,
                               view_start=self.view_start, end=None))
        kw.update(overrides)
        return pattern_figure(self.df, **kw)


def market_view(cfg, now=None) -> MarketView:
    """Load the config's instrument up to its latest closed candle, plus every timeframe a layer
    needs, and the calendar. `now` is for tests; it defaults to the clock."""
    with _quiet():
        return _load(cfg, now)


def _load(cfg, now) -> MarketView:
    from ..data import get_ohlcv, tf_frames
    from ..data.timeframe import tf_minutes

    c = cfg
    _check(c)
    name, symbol, src = instrument(c)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    tf, notes = c.TF, []
    cov = _get(c, "NEWS", None)
    if cov and _get(c, "NEWS_UPDATE", True):
        notes += _refresh_news(now)

    # THE CHART'S CANDLES: the window plus its warm-up, counted in candles from the latest closed
    tfm = tf_minutes(tf)
    n_view = max(1, int(round(float(c.DAYS) * 1440 / tfm)))
    warm = int(_get(c, "TF_WARMUP_BARS", 800))
    need = n_view + warm
    df = pd.DataFrame()
    for extra in LOOKBACK_EXTRA_DAYS:
        start = now - pd.Timedelta(minutes=tfm * need) - pd.Timedelta(days=extra)
        df = _closed(get_ohlcv(symbol, tf, start=start.strftime("%Y-%m-%d %H:%M"), end=None,
                               source=src, tz="UTC"), tf, now)
        if len(df) >= need:
            break
    if not len(df):
        raise ValueError(f"no {tf} candles for {symbol} ({src}) in the last "
                         f"{LOOKBACK_EXTRA_DAYS[-1]} days -- is the symbol right?")
    if len(df) < n_view:
        notes.append(f"only {len(df):,} {tf} candles exist in the last {LOOKBACK_EXTRA_DAYS[-1]} "
                     f"days; the window is shorter than {c.DAYS} days")
    df = df.iloc[-need:].reset_index(drop=True)
    view_start = pd.Timestamp(df["open_time"].iloc[max(0, len(df) - n_view)])
    latest = pd.Timestamp(df["open_time"].iloc[-1])

    # EVERY OTHER TIMEFRAME a layer asks for, with the same warm-up IN ITS OWN BARS
    others = []
    for on, lst, _ in FEATURES:
        if _get(c, on, True):
            others += [k for k in (_get(c, lst, ()) or ()) if k != tf and k not in others]
    frames = {}
    if others:
        got = tf_frames(symbol, tuple(others), view_start=view_start, end=None,
                        warmup_bars=warm, source=src)
        frames = {k: _closed(f, k, now) for k, f in got.items()}

    news = pd.DataFrame()
    if cov:
        from ..news import load_events
        news = load_events(pd.Timestamp(df["open_time"].iloc[0]) - pd.Timedelta(days=1),
                           now + pd.Timedelta(days=2), coverage=cov)
    return MarketView(cfg=c, name=name, symbol=symbol, source=src, df=df, frames=frames,
                      view_start=view_start, latest=latest, now=now, news=news, notes=notes)
