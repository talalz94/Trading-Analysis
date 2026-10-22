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
class Mark:
    """One thing drawn over the candles, in FRAME index space (an index may run past the last
    candle: a line extended into the future). The browser chart and the email picture draw the
    same marks, so they cannot disagree about which line fired."""
    kind: str                            # "line" | "points"
    xs: List[int]
    ys: List[float]
    color: str
    width: float = 2.0
    dash: str = "solid"                  # "solid" | "dot" | "dash"
    label: str = ""


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
    marks: List[Mark] = field(default_factory=list)  # the line / shape that fired
    first: int = 0                       # the earliest candle the marks reach back to
    fig: object = None
    path: Optional[Path] = None

    @property
    def reproduced(self) -> bool:
        return bool(self.hits)


def build_chart(cfg, alert, market: str, tf: str, close, bars: pd.DataFrame, *,
                end=None, ref: str = "", what: Optional[str] = None) -> AlertChart:
    """Candles + the alert re-run on the window the monitor judged + the marks it drew. `bars` are
    the market's stored bars of base_of(tf) -- the feed's for `show`, the state's for an email.
    `what`, when known (an email row), is the signal to find: if the monitor's window does not
    give it, the whole history up to the candle is tried -- the window `check` judges with."""
    from .engine import tf_bars
    det = REGISTRY[alert.kind]
    step = pd.Timedelta(minutes=TF_MINUTES[tf])
    close = pd.Timestamp(close).tz_convert("UTC")
    bar = close - step
    base = base_of(tf)
    frame = tf_bars(bars, tf, base)
    if end is not None:
        frame = frame[frame[TIME_COL] + step <= end]
    frame = frame.reset_index(drop=True)
    at = np.flatnonzero((frame[TIME_COL] == bar).to_numpy())
    if not len(at):
        raise LookupError(f"no {tf} candle closing at {close:%Y-%m-%d %H:%M} UTC for {market}")
    k = int(at[0])
    # the window the monitor judged: the alert's warm-up before the FIRST new candle of that
    # look, ending on the alert's own. On a 15-minute clock a 1m or 5m candle is judged with the
    # rest of its quarter hour, so that look began at the quarter hour, not at this candle --
    # and a line fitted from another start can differ.
    warm = det.warmup_for(alert.params, tf)
    every = cfg.delivery.every_minutes
    first_new = bar.floor(f"{every}min") if every and TF_MINUTES[tf] < every else bar
    j = int((frame[TIME_COL] < first_new).sum())
    # the fallback is bounded: the whole stored history of a 15m pattern took 5 s to re-run
    for lo in dict.fromkeys((max(0, min(j, k) - warm), max(0, k - 4 * warm))):
        win = frame.iloc[lo:k + 1].reset_index(drop=True)
        win["t"] = win[TIME_COL]
        hits = [h for h in det.evaluate(win, alert.params, tf) if h["i"] == len(win) - 1]
        if what is None or what in [h["what"] for h in hits]:
            break
    if what is not None:
        hits = [h for h in hits if h["what"] == what] or hits
    c = AlertChart(ref, market, tf, alert.name, close, frame, k, hits)
    last = len(frame) - 1
    marks, first = (_trendline_marks(alert, win, lo, last, hits) if alert.kind == "trendline" else
                    _shape_marks(alert, win, lo, last, hits) if alert.kind == "chart" else
                    _early_marks(lo, hits, k) if alert.kind == "early_break" else
                    ([], k))
    c.marks, c.first = marks, first
    return c


def alert_chart(ref: str, *, config=None, after: int = 60, before: int = 150, feed=None,
                now=None, write: bool = True, show: bool = True) -> AlertChart:
    """Re-run the alert `ref` names and draw it. `after` / `before` = candles drawn after / before
    the alert's own. Writes reports/alerts/<ref>.html and opens it (`write`, `show`)."""
    from .feeds import Feed
    ref = str(ref).strip().lower()
    _, tf, close, _ = parse_ref(ref)
    cfg, mk, alert = find_alert(ref, config)
    det = REGISTRY[alert.kind]
    step = pd.Timedelta(minutes=TF_MINUTES[tf])
    bar = close - step
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now).tz_convert("UTC")
    end = min(now, close + after * step)
    warm = det.warmup_for(alert.params, tf)
    start = (bar - (warm + before) * step * 7 / 5 - pd.Timedelta(days=3)).floor("D")
    base = base_of(tf)
    got = _cached(mk, base, start, end) if feed is None else None
    if got is None:
        feed = feed or Feed()
        got = (feed.fetch([mk], {mk.name: start}, end) if base == "5m"
               else feed.fetch([mk], {mk.name: start}, end, base=base))[mk.name]
    if isinstance(got, Exception):
        raise got
    try:
        out = build_chart(cfg, alert, mk.name, tf, close, got, end=end, ref=ref)
    except LookupError:
        raise LookupError(f"the feed has no {tf} candle closing at {close:%Y-%m-%d %H:%M} UTC "
                          f"for {mk.name}") from None
    out.fig = _figure(out, before)
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


# ── the marks: WHAT is drawn, shared by the browser chart and the email picture ─────────────
GOLD, UP, DN = "#f2c94c", "#26a69a", "#ef5350"


def _trendline_marks(alert, win, lo, last, hits):
    """The line the alert is about: the one published on the alert's candle (formed), or the one
    that broke there (break). Solid while it was live, dotted after. -> (marks, first index)."""
    from quant.indicators import add_trendlines, trendline_table
    p = REGISTRY["trendline"].settings(alert.params)
    d = add_trendlines(win[["open", "high", "low", "close"]].copy(),
                       min_touches=int(p["min_touches"]), atr_mult=float(p["swing_size"]),
                       fit=p["fit"], line_tol_atr=float(p["touch_tol"]),
                       wick_tol_atr=float(p["wick_tol"]),
                       max_distance_atr=float(p["max_distance"]),
                       prefer_newer=bool(p["prefer_newer"]),
                       break_sustain_bars=int(p["hold"]))
    tb = trendline_table(d)
    k = len(win) - 1
    out, first = [], lo + k
    if not len(tb):
        return out, first
    formed = tb[tb["born_i"] == k]
    broke = tb[tb["broke"] & (tb["end_i"] <= k)].sort_values("end_i")
    for h in hits:
        if "formed" in h["what"]:
            rows = formed[formed["kind"] == (1 if "resistance" in h["what"] else -1)]
        else:
            rows = broke[broke["kind"] == (1 if "resistance" in h["what"] else -1)].tail(1)
        for r in rows.itertuples():
            col = DN if r.kind > 0 else UP
            s, e = int(r.start_i), int(r.end_i)
            first = min(first, lo + s)
            xs = list(range(s, e + 1))
            out.append(Mark("line", [lo + i for i in xs], [r.slope * i + r.intercept for i in xs],
                            col, 3, label=f"{'resistance' if r.kind > 0 else 'support'} line "
                                          f"{r.id} ({r.touches} touches)"))
            ext = list(range(e, last - lo + 1))
            if len(ext) > 1:                           # where it would have run on: dotted
                out.append(Mark("line", [lo + i for i in ext],
                                [r.slope * i + r.intercept for i in ext], col, 1.5, "dot"))
            out.append(Mark("points", [lo + int(i) for i in r.touch_i], list(r.touch_price),
                            col, 9, label="touches"))
    return out, first


def _early_marks(lo, hits, k):
    """The coarser line an early break crossed, drawn on these finer candles from its first
    touch to the close through it, with its touches. -> (marks, first index)."""
    out, first = [], k
    for h in hits:
        curve = h.get("curve") or []
        if not curve:
            continue
        up = h.get("side") == "bull"
        col = DN if up else UP
        xs = [lo + j for j, _ in curve]
        first = min(first, xs[0])
        out.append(Mark("line", xs, [y for _, y in curve], col, 3,
                        label=h["what"].split(":")[0]))
        pts = [(lo + j, y) for j, y in h.get("touches") or [] if lo + j <= k]
        if pts:
            out.append(Mark("points", [x for x, _ in pts], [y for _, y in pts], col, 9,
                            label="touches"))
            first = min(first, min(x for x, _ in pts))
    return out, first


def _shape_marks(alert, win, lo, last, hits):
    """The chart pattern the alert is about: registered on the alert's candle (formed), or the
    live one of that pattern it broke out of (breakout) -- both boundaries, the pivots, the level
    it breaks and the measured target. -> (marks, first index)."""
    from quant.indicators import add_chart_patterns
    p = REGISTRY["chart"].settings(alert.params)
    names = REGISTRY["chart"]._names(p)
    formed: list = []
    add_chart_patterns(win[["open", "high", "low", "close", "volume"]].copy(),
                       sensitivity=p["sensitivity"], patterns=names, _formed=formed,
                       fit=p["fit"], wick_tol_atr=float(p["wick_tol"]),
                       line_tol_atr=float(p["touch_tol"]) or None)
    k = len(win) - 1
    out, first = [], lo + k
    shapes = []
    for h in hits:
        w = h["what"].lower()
        same = [r for r in formed if r["pattern"].replace("_", " ") in w]
        if "formed" in w:
            shapes += [r for r in same if r["i"] == k]
        else:                                          # the live shape it broke out of, on the
            way = 1 if h.get("side") == "bull" else -1 if h.get("side") == "bear" else 0
            live = [r for r in same if r["i"] < k and r.get("expires_i", k) >= k
                    and (not way or r["direction"] == way)]    # side it broke (a symmetrical
            shapes += live[-1:]                                # triangle registers both)
    for r in shapes:
        x1 = max(min(int(r.get("expires_i", k)), last - lo), k)
        level_shape = not r.get("upper") and not r.get("lower")
        for key, nm in (("upper", "upper line"), ("lower", "lower line")):
            if not r.get(key):
                continue
            sl, ic = r[key]
            # a boundary runs from ITS OWN first touch: drawn from the shape's earliest pivot, a
            # line touched only at the right end looked like it bounded candles it never met
            tch = r.get(f"{key}_touches") or r.get("pivots") or []
            x0 = min((i for i, _ in tch), default=k)
            first = min(first, lo + x0)
            xs = list(range(x0, x1 + 1))           # EVERY candle: two endpoints drawn on a time
            out.append(Mark("line", [lo + i for i in xs], [sl * i + ic for i in xs], GOLD, 2.5,
                            label=f"{r['pattern'].replace('_', ' ')}: {nm}"))   # axis bend the
            # line wherever the market shut (a weekend has no candles), 20 points off on Gold 15m
            if r.get(f"{key}_touches"):
                out.append(Mark("points", [lo + i for i, _ in tch], [px for _, px in tch], GOLD,
                                9, label=f"{nm} touches"))
        piv = r.get("pivots") or []
        if level_shape and piv:                        # a double top: its pivots ARE the shape
            first = min(first, lo + min(i for i, _ in piv))
            out.append(Mark("points", [lo + i for i, _ in piv], [px for _, px in piv], GOLD, 8,
                            label="pivots"))
        tgt = r["boundary"] + r["direction"] * r["height"]
        col = UP if r["direction"] > 0 else DN
        for y, nm, dash in ((r["boundary"], "breaks here", "dash"), (tgt, "target", "dot")):
            out.append(Mark("line", [lo + k, max(last, lo + k + 10)], [y, y], col, 1.5, dash,
                            label=f"{nm} {y:,.2f}"))
    return out, first


# ── the browser chart (plotly) ───────────────────────────────────────────────────────────────
def _times(frame, idx):
    """Frame index (may run past the end) -> times, stepping on by the timeframe."""
    t = frame[TIME_COL]
    step = t.iloc[-1] - t.iloc[-2] if len(t) > 1 else pd.Timedelta(minutes=1)
    return [t.iloc[i] if i < len(t) else t.iloc[-1] + (i - len(t) + 1) * step for i in idx]


def _figure(c: AlertChart, before):
    import plotly.graph_objects as go
    f, k = c.frame, c.k
    v0 = max(0, min(k - before, c.first - 10))       # the whole line or shape, and some before
    view = f.iloc[v0:]
    fig = go.Figure(go.Candlestick(
        x=view[TIME_COL], open=view["open"], high=view["high"], low=view["low"],
        close=view["close"], name=f"{c.market} {c.tf}", increasing_line_color=UP,
        decreasing_line_color=DN, showlegend=False))
    for m in c.marks:
        x = _times(f, m.xs)
        if m.kind == "points":
            fig.add_trace(go.Scatter(x=x, y=m.ys, mode="markers", name=m.label,
                                     marker=dict(size=m.width, color=m.color,
                                                 line=dict(color="#fff", width=1))))
        else:
            fig.add_trace(go.Scatter(x=x, y=m.ys, mode="lines", name=m.label or None,
                                     showlegend=bool(m.label),
                                     line=dict(color=m.color, width=m.width,
                                               dash={"solid": None}.get(m.dash, m.dash))))
    bar = f.iloc[k]
    fig.add_trace(go.Scatter(
        x=[bar[TIME_COL]], y=[bar["close"]],
        mode="markers", name="the alert's candle (its close)",
        marker=dict(symbol="star", size=18, color=GOLD, line=dict(color="#111", width=1)),
        hovertext=[" · ".join(h["what"] for h in c.hits) or c.alert], hoverinfo="text"))
    fig.add_shape(type="line", xref="x", yref="paper", x0=c.close, x1=c.close, y0=0, y1=1,
                  line=dict(color=GOLD, width=1, dash="dot"))
    what = "; ".join(h["what"] + (f" ({h['note']})" if h.get("note") else "") for h in c.hits)
    status = what if c.hits else ("NOT reproduced on this data -- the line or shape is drawn as "
                                  "the detector sees it now")
    fig.update_layout(
        template="plotly_dark", height=760, xaxis_rangeslider_visible=False,
        title=dict(text=f"{c.alert} · {c.market} {c.tf} · candle closed "
                        f"{c.close:%a %d %b %Y %H:%M} UTC<br><sup>{status}</sup>", x=0.01),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
        margin=dict(l=50, r=30, t=90, b=40))
    return fig
