"""
FROM AN EMAIL TO A CHART: `python -m quant.alerts show <ref>`.

Every signal row in an alert email carries a REF -- `btc.1m.202610191349.trendline-broken`: the
market, the timeframe, the candle's CLOSE in UTC and the alert's name. Given one, this fetches that
market's candles again, re-runs the alert exactly as the monitor does -- on the candles up to that
close, with the warm-up it judges with -- and draws the chart: the candles around it, the line or
shape that fired (highlighted), the candle it fired on, and what came after.

The alert is looked up by name in alerts/alerts_config.py, then in every alerts_config_*.py beside
it, so a ref from the server's emails resolves on the PC as long as that file is in the repo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from quant.data.base import TIME_COL

from .config import DEFAULT_CONFIG, ROOT, TF_MINUTES, ConfigError, base_of, load_config
from .detectors import REGISTRY

REF_RE = re.compile(r"^([a-z0-9-]+)\.(\d+[mh])\.(\d{12})\.([a-z0-9-]+)$")
OUT_DIR = ROOT / "reports" / "alerts"


def slug(s) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def make_ref(market: str, tf: str, close_time, alert: str) -> str:
    t = pd.Timestamp(close_time).tz_convert("UTC")
    return f"{slug(market)}.{tf}.{t:%Y%m%d%H%M}.{slug(alert)}"


def parse_ref(ref: str):
    """-> (market slug, tf, candle close UTC, alert slug)."""
    m = REF_RE.match(str(ref).strip().lower())
    if not m or m.group(2) not in TF_MINUTES:
        raise ValueError(f"not an alert ref: {ref!r} -- it looks like "
                         "btc.1m.202610191349.trendline-broken (from the email row)")
    close = pd.Timestamp(pd.to_datetime(m.group(3), format="%Y%m%d%H%M"), tz="UTC")
    return m.group(1), m.group(2), close, m.group(4)


def find_alert(ref: str, config: Optional[Path] = None):
    """-> (config, market, alert) the ref names. Searches the given config, else alerts_config.py
    and then every alerts_config_*.py beside it."""
    m_slug, tf, _, a_slug = parse_ref(ref)
    paths = ([Path(config)] if config else
             [DEFAULT_CONFIG] + sorted(DEFAULT_CONFIG.parent.glob("alerts_config_*.py")))
    for p in paths:
        try:
            cfg = load_config(p)
        except ConfigError:
            if config:
                raise
            continue
        markets = {slug(n): mk for n, mk in cfg.markets.items()}
        for a in cfg.alerts + cfg.off:
            if slug(a.name) == a_slug and tf in a.tfs and m_slug in markets:
                return cfg, markets[m_slug], a
    raise LookupError(f"no alert {a_slug!r} on {tf} for market {m_slug!r} in "
                      f"{', '.join(p.name for p in paths)}")


@dataclass
class AlertChart:
    ref: str
    market: str
    tf: str
    alert: str
    close: pd.Timestamp
    frame: pd.DataFrame                  # the timeframe's candles, the alert's bar included
    k: int                               # the alert's bar in `frame`
    hits: List[dict] = field(default_factory=list)   # what the alert says on that bar, re-run
    fig: object = None
    path: Optional[Path] = None

    @property
    def reproduced(self) -> bool:
        return bool(self.hits)


def alert_chart(ref: str, *, config=None, after: int = 60, before: int = 150, feed=None,
                now=None, write: bool = True, show: bool = True) -> AlertChart:
    """Re-run the alert `ref` names and draw it. `after` / `before` = candles drawn after / before
    the alert's own. Writes reports/alerts/<ref>.html and opens it (`write`, `show`)."""
    from .engine import tf_bars
    from .feeds import Feed
    ref = str(ref).strip().lower()
    _, tf, close, _ = parse_ref(ref)
    cfg, mk, alert = find_alert(ref, config)
    det = REGISTRY[alert.kind]
    step = pd.Timedelta(minutes=TF_MINUTES[tf])
    bar = close - step
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    end = min(now, close + after * step)
    warm = det.warmup_bars(alert.params)
    start = (bar - (warm + before) * step * 7 / 5 - pd.Timedelta(days=3)).floor("D")
    base = base_of(tf)
    got = _cached(mk, base, start, end) if feed is None else None
    if got is None:
        feed = feed or Feed()
        got = (feed.fetch([mk], {mk.name: start}, end) if base == "5m"
               else feed.fetch([mk], {mk.name: start}, end, base=base))[mk.name]
    if isinstance(got, Exception):
        raise got
    frame = tf_bars(got, tf, base)
    frame = frame[frame[TIME_COL] + step <= end].reset_index(drop=True)
    at = np.flatnonzero((frame[TIME_COL] == bar).to_numpy())
    if not len(at):
        raise LookupError(f"the feed has no {tf} candle closing at {close:%Y-%m-%d %H:%M} UTC "
                          f"for {mk.name}")
    k = int(at[0])
    # the window the monitor judged: the alert's warm-up before the FIRST new candle of that
    # look, ending on the alert's own. On a 15-minute clock a 1m or 5m candle is judged with the
    # rest of its quarter hour, so that look began at the quarter hour, not at this candle --
    # and a line fitted from another start can differ.
    every = cfg.delivery.every_minutes
    first_new = bar.floor(f"{every}min") if every and TF_MINUTES[tf] < every else bar
    j = int((frame[TIME_COL] < first_new).sum())
    lo = max(0, min(j, k) - warm)
    win = frame.iloc[lo:k + 1].reset_index(drop=True)
    win["t"] = win[TIME_COL]
    hits = [h for h in det.evaluate(win, alert.params, tf) if h["i"] == len(win) - 1]
    out = AlertChart(ref, mk.name, tf, alert.name, close, frame, k, hits)
    out.fig = _figure(out, alert, win, lo, before)
    if write:
        # written and opened HERE, not through quant.viz: importing that package pulls in
        # plotly-resampler and Dash, 11 of the 20 s a `show` took
        import webbrowser
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out.path = (OUT_DIR / f"{ref}.html").resolve()
        out.fig.write_html(str(out.path), include_plotlyjs=True, full_html=True)
        if show:
            webbrowser.open(out.path.as_uri())   # hands the file to the browser and returns
    return out


def _cached(mk, base: str, start, end) -> Optional[pd.DataFrame]:
    """The same candles through the PC's research cache (data/, incremental): the same provider,
    the same bid side, but months of 5m history are fetched once instead of on every `show` --
    a 4h alert took 2 min straight from the feed. None when the cache cannot serve it."""
    import logging
    from quant.data import get_ohlcv
    from .feeds import BASE_BAR, COLS
    quiet = logging.getLogger("quant.data")
    level = quiet.level
    quiet.setLevel(logging.WARNING)
    try:
        df = get_ohlcv(mk.symbol, base, start=start.strftime("%Y-%m-%d %H:%M"),
                       end=end.strftime("%Y-%m-%d %H:%M"), source=mk.source, tz="UTC")
    except Exception:
        return None
    finally:
        quiet.setLevel(level)
    if df is None or not len(df):
        return None
    df = df[COLS]
    return df[df[TIME_COL] + BASE_BAR[base] <= end].reset_index(drop=True)


# ── drawing ──────────────────────────────────────────────────────────────────────────────────
_GOLD, _UP, _DN = "#f2c94c", "#26a69a", "#ef5350"


def _times(frame, idx):
    """Frame index (may run past the end) -> times, stepping on by the timeframe."""
    t = frame[TIME_COL]
    step = t.iloc[-1] - t.iloc[-2] if len(t) > 1 else pd.Timedelta(minutes=1)
    return [t.iloc[i] if i < len(t) else t.iloc[-1] + (i - len(t) + 1) * step for i in idx]


def _figure(c: AlertChart, alert, win, lo, before):
    import plotly.graph_objects as go
    f, k = c.frame, c.k
    last = len(f) - 1
    art = (_draw_trendline(c, alert, win, lo, last) if alert.kind == "trendline" else
           _draw_shape(c, alert, win, lo, last) if alert.kind == "chart" else ([], k))
    traces, first = art
    v0 = max(0, min(k - before, first - 10))          # the whole line or shape, and some before
    view = f.iloc[v0:]
    fig = go.Figure(go.Candlestick(
        x=view[TIME_COL], open=view["open"], high=view["high"], low=view["low"],
        close=view["close"], name=f"{c.market} {c.tf}", increasing_line_color=_UP,
        decreasing_line_color=_DN, showlegend=False))
    for tr in traces:
        fig.add_trace(tr)
    bar = f.iloc[k]
    fig.add_trace(go.Scatter(
        x=[bar[TIME_COL]], y=[bar["close"]],
        mode="markers", name="the alert's candle (its close)",
        marker=dict(symbol="star", size=18, color=_GOLD, line=dict(color="#111", width=1)),
        hovertext=[" · ".join(h["what"] for h in c.hits) or alert.name], hoverinfo="text"))
    fig.add_shape(type="line", xref="x", yref="paper", x0=c.close, x1=c.close, y0=0, y1=1,
                  line=dict(color=_GOLD, width=1, dash="dot"))
    what = "; ".join(h["what"] + (f" ({h['note']})" if h.get("note") else "") for h in c.hits)
    status = what if c.hits else ("NOT reproduced on this data -- the line or shape is drawn as "
                                  "the detector sees it now")
    fig.update_layout(
        template="plotly_dark", height=760, xaxis_rangeslider_visible=False,
        title=dict(text=f"{alert.name} · {c.market} {c.tf} · candle closed "
                        f"{c.close:%a %d %b %Y %H:%M} UTC<br><sup>{status}</sup>", x=0.01),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=50, r=30, t=90, b=40))
    return fig


def _draw_trendline(c, alert, win, lo, last):
    """The line the alert is about: the one published on the alert's candle (formed), or the one
    that broke there (break). Solid while it was live, dotted after. -> (traces, first index)."""
    import plotly.graph_objects as go
    from quant.indicators import add_trendlines, trendline_table
    p = REGISTRY["trendline"].settings(alert.params)
    d = add_trendlines(win[["open", "high", "low", "close"]].copy(),
                       min_touches=int(p["min_touches"]), atr_mult=float(p["swing_size"]),
                       fit=p["fit"], line_tol_atr=float(p["touch_tol"]),
                       wick_tol_atr=float(p["wick_tol"]))
    tb = trendline_table(d)
    out, first = [], lo + len(win) - 1
    if not len(tb):
        return out, first
    k = len(win) - 1
    formed = tb[tb["born_i"] == k]
    broke = tb[tb["broke"] & (tb["end_i"] <= k)].sort_values("end_i")
    for h in c.hits:
        if "formed" in h["what"]:
            rows = formed[formed["kind"] == (1 if "resistance" in h["what"] else -1)]
        else:
            rows = broke[broke["kind"] == (1 if "resistance" in h["what"] else -1)].tail(1)
        for r in rows.itertuples():
            col = _DN if r.kind > 0 else _UP
            s, e = int(r.start_i), int(r.end_i)
            first = min(first, lo + s)
            xs = list(range(s, e + 1))
            out.append(go.Scatter(
                x=_times(c.frame, [lo + i for i in xs]), y=[r.slope * i + r.intercept for i in xs],
                mode="lines", line=dict(color=col, width=3),
                name=f"{'resistance' if r.kind > 0 else 'support'} line {r.id} "
                     f"({r.touches} touches)"))
            ext = list(range(e, last - lo + 1))
            if len(ext) > 1:                           # where it would have run on: dotted
                out.append(go.Scatter(
                    x=_times(c.frame, [lo + i for i in ext]),
                    y=[r.slope * i + r.intercept for i in ext], mode="lines",
                    line=dict(color=col, width=1.5, dash="dot"), name="the line, extended",
                    showlegend=False))
            out.append(go.Scatter(
                x=_times(c.frame, [lo + int(i) for i in r.touch_i]), y=list(r.touch_price),
                mode="markers", marker=dict(size=9, color=col, line=dict(color="#fff", width=1)),
                name="touches"))
    return out, first


def _draw_shape(c, alert, win, lo, last):
    """The chart pattern registered on the alert's candle: both boundaries, its pivots, the
    level it would break and the measured target. -> (traces, first index)."""
    import plotly.graph_objects as go
    from quant.indicators import add_chart_patterns
    p = REGISTRY["chart"].settings(alert.params)
    names = REGISTRY["chart"]._names(p)
    formed: list = []
    add_chart_patterns(win[["open", "high", "low", "close", "volume"]].copy(),
                       sensitivity=p["sensitivity"], patterns=names, _formed=formed)
    k = len(win) - 1
    out, first = [], lo + k
    for r in [r for r in formed if r["i"] == k]:
        piv = r.get("pivots") or []
        x0 = min((i for i, _ in piv), default=k)
        first = min(first, lo + x0)
        x1 = min(int(r.get("expires_i", k)), last - lo)
        xs = [x0, max(x1, k)]
        for key, nm in (("upper", "upper line"), ("lower", "lower line")):
            if r.get(key):
                sl, ic = r[key]
                out.append(go.Scatter(
                    x=_times(c.frame, [lo + i for i in xs]), y=[sl * i + ic for i in xs],
                    mode="lines", line=dict(color=_GOLD, width=2.5),
                    name=f"{r['pattern'].replace('_', ' ')}: {nm}"))
        if piv:
            out.append(go.Scatter(
                x=_times(c.frame, [lo + i for i, _ in piv]), y=[px for _, px in piv],
                mode="markers", marker=dict(size=8, color=_GOLD), name="pivots"))
        tgt = r["boundary"] + r["direction"] * r["height"]
        for y, nm, dash in ((r["boundary"], "breaks here", "dash"), (tgt, "target", "dot")):
            out.append(go.Scatter(
                x=_times(c.frame, [lo + k, max(last, lo + k + 10)]), y=[y, y], mode="lines",
                line=dict(color=_UP if r["direction"] > 0 else _DN, width=1.5, dash=dash),
                name=f"{nm} {y:,.2f}"))
    return out, first
