"""
CHARTS INSIDE THE EMAIL -- a picture of the email's most important signals, so the line or shape
that fired is in front of you without a click, on a phone as much as on the PC.

DELIVERY charts=N draws the top N signal rows, ranked the way the subject ranks them (tier, then
the coarser timeframe): THE line or shape that fired, highlighted, on its candles, with a star on
the alert's candle. Each picture is the same model `python -m quant.alerts show` draws in the
browser (show.build_chart), so the two cannot disagree.

Built to be harmless on a small server:
  - nothing happens unless charts > 0, and matplotlib is imported only then (lazily, once);
  - no pyplot and no global state -- a Figure drawn straight onto an Agg canvas and dropped;
  - a time budget per email (chart_budget_seconds): pictures stop, the email still goes;
  - any failure -- matplotlib missing, a chart that cannot be built -- costs that picture only,
    is reported, and never holds up or changes the email around it.
Pictures travel as inline attachments (Content-ID), the one form Gmail shows inside the message;
it does not display images embedded in the HTML as data.
"""
from __future__ import annotations

import io
import json
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from quant.data.base import TIME_COL

from .config import TF_MINUTES, base_of

BG, GRID, TXT, MUTED = "#111418", "#2a2f36", "#e5e7eb", "#9aa0a6"
TIER_COLOUR = {1: "#dc2626", 2: "#ea580c", 3: "#6b7280"}
WIDTH_PX, HEIGHT_PX = 800, 380
MAX_CANDLES = 400                    # ~2 px a candle at 800 px; 250 cut a 15m line's earlier
                                     # touches off the left edge, leaving one touch in view


def render_png(c, *, tier: int = 3, caption: str = "", before: int = 120) -> bytes:
    """One chart as PNG bytes: the candles from just before the line or shape to now, its marks,
    the alert's candle starred and its close dotted."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from .show import GOLD, UP, DN
    f, k = c.frame, c.k
    v0 = max(0, min(k - before, c.first - 10), k - MAX_CANDLES)
    view = f.iloc[v0:]
    x = np.arange(v0, v0 + len(view))
    o, h, lo, cl = (view[n].to_numpy(float) for n in ("open", "high", "low", "close"))
    up = cl >= o
    colours = np.where(up, UP, DN)

    fig = Figure(figsize=(WIDTH_PX / 100, HEIGHT_PX / 100), dpi=100, facecolor=BG)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.035, 0.1, 0.88, 0.78))
    ax.set_facecolor(BG)
    ax.vlines(x, lo, h, colors=colours, linewidth=0.8, zorder=2)
    span = float(np.nanmax(h) - np.nanmin(lo)) or 1.0
    # bodies as ONE collection of thick segments: ax.bar made a Rectangle per candle and was most
    # of the drawing time (1.6 s for a 1m chart)
    body_px = 0.65 * (WIDTH_PX * 0.88) / max(len(view) + 10, 1)
    ax.vlines(x, np.minimum(o, cl), np.maximum(np.maximum(o, cl), np.minimum(o, cl) + span * 0.002),
              colors=colours, linewidth=max(1.0, body_px * 72 / 100), zorder=3)
    style = {"solid": "-", "dot": ":", "dash": "--"}
    for m in c.marks:
        if m.kind == "points":
            ax.scatter(m.xs, m.ys, s=(m.width * 0.9) ** 2, color=m.color, edgecolors="#ffffff",
                       linewidths=0.6, zorder=5)
        else:
            ax.plot(m.xs, m.ys, color=m.color, linewidth=max(1.0, m.width * 0.6),
                    linestyle=style.get(m.dash, "-"), zorder=4)
    ax.scatter([k], [f["close"].iloc[k]], marker="*", s=170, color=GOLD, edgecolors="#111111",
               linewidths=0.8, zorder=6)
    ax.axvline(k + 0.5, color=GOLD, linewidth=0.8, linestyle=":", zorder=1)

    # the candles decide the frame; a far target is named at the edge instead of squashing them
    pad = span * 0.08
    y0, y1 = float(np.nanmin(lo)) - pad, float(np.nanmax(h)) + pad
    levels = [m for m in c.marks if m.kind == "line" and len(set(m.ys)) == 1]  # breaks / target
    for m in levels:
        y = m.ys[0]
        if y0 - span * 0.6 <= y <= y1 + span * 0.6:
            y0, y1 = min(y0, y - pad), max(y1, y + pad)
    ax.set_ylim(y0, y1)
    for m in levels:                                   # one still off the frame: named at its edge
        y = m.ys[0]
        if not y0 <= y <= y1 and m.label:
            ax.text(0.99, 0.97 if y > y1 else 0.03, f"{m.label} {'▲' if y > y1 else '▼'}",
                    transform=ax.transAxes, ha="right", va="top" if y > y1 else "bottom",
                    color=m.color, fontsize=8)
    right = max(v0 + len(view) - 1, k + 8) + 2
    ax.set_xlim(v0 - 1, right)

    ax.yaxis.tick_right()
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.5)
    days = (f[TIME_COL].iloc[-1] - f[TIME_COL].iloc[v0]).total_seconds() / 86400
    fmt = ("%d %b" if TF_MINUTES[c.tf] >= 240 or days > 6 else
           "%d %b %H:%M" if days > 1 else "%H:%M")     # a time alone is ambiguous across days
    ticks = np.linspace(v0, v0 + len(view) - 1, 6).round().astype(int)
    ax.set_xticks(ticks)
    labels = ax.set_xticklabels([pd.Timestamp(f[TIME_COL].iloc[i]).strftime(fmt) for i in ticks])
    labels[0].set_horizontalalignment("left")          # the edge labels inside the picture
    labels[-1].set_horizontalalignment("right")

    fig.text(0.015, 0.93, f" P{tier} ", color="#ffffff", fontsize=10, fontweight="bold",
             va="center", bbox=dict(boxstyle="round,pad=0.25", facecolor=TIER_COLOUR[tier],
                                    edgecolor="none"))
    fig.text(0.075, 0.93, caption, color=TXT, fontsize=10, va="center")
    fig.text(0.985, 0.93, f"closed {c.close:%d %b %H:%M} UTC", color=MUTED, fontsize=8,
             va="center", ha="right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    fig.clear()
    return buf.getvalue()


def email_charts(events, cfg, bars_of: Callable[[str, str], pd.DataFrame], now
                 ) -> Tuple[List[dict], Optional[str]]:
    """The top `cfg.delivery.charts` signal rows of an email as pictures -> ([{cid, png, caption,
    tier}], what went wrong or None). `bars_of(market, base)` gives the stored bars."""
    d = cfg.delivery
    n = int(getattr(d, "charts", 0) or 0)
    if n <= 0:
        return [], None
    try:
        import matplotlib  # noqa: F401  -- only now: a server without charts never loads it
        from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: F401
    except ImportError:
        return [], ("matplotlib is not installed -- emails go without charts "
                    "(.venv/bin/pip install -r requirements-alerts.txt)")
    from .compose import TIER_TAG, _headline, rank_key, tier
    from .news import is_news
    from .show import build_chart
    alerts = {a.name: a for a in cfg.alerts + cfg.off}
    latest = {}
    for e in events:                                   # one picture per signal, its latest repeat
        if is_news(e) or e.get("late") or e["alert"] not in alerts:
            continue
        key = (e["alert"], e["market"], e["tf"], e["what"])
        if key not in latest or e["close_time"] > latest[key]["close_time"]:
            latest[key] = e
    out, errors = [], []
    t0 = time.monotonic()
    budget = float(getattr(d, "chart_budget_seconds", 20))
    for e in sorted(latest.values(), key=rank_key):
        if len(out) >= n:
            break
        if time.monotonic() - t0 > budget:
            errors.append(f"chart budget of {budget:g} s spent after {len(out)} chart(s)")
            break
        try:
            c = build_chart(cfg, alerts[e["alert"]], e["market"], e["tf"], e["close_time"],
                            bars_of(e["market"], base_of(e["tf"])), end=now, what=e["what"])
            t = tier(e)
            png = render_png(c, tier=t, caption=_headline(e))
            out.append(dict(cid=f"chart{len(out)}.{int(time.time() * 1000)}@quant-alerts",
                            png=png, caption=f"{TIER_TAG[t]} · {_headline(e)}", tier=t,
                            close=e["close_time"]))
        except Exception as ex:                       # this picture only
            errors.append(f"{e['market']} {e['tf']} {e['what']}: {type(ex).__name__}: {ex}")
    return out, ("; ".join(errors) or None)


def attach(html: str, charts: List[dict], *, inline: str = "cid") -> str:
    """Put the pictures at the top of the email, most important first. `inline="cid"` for an
    email (the images ride as attachments), "data" for a file viewed locally (the preview)."""
    if not charts:
        return html
    import base64
    from .compose import _OPEN_DIV
    parts = ['<h3 style="margin:18px 0 6px;font-size:15px;">Charts</h3>']
    for c in charts:
        src = (f"cid:{c['cid']}" if inline == "cid" else
               "data:image/png;base64," + base64.b64encode(c["png"]).decode())
        parts.append(
            f'<p style="margin:12px 0 4px;font-size:13px;color:#374151;">{c["caption"]}</p>'
            f'<img src="{src}" alt="{c["caption"]}" width="{WIDTH_PX}" '
            f'style="display:block;width:100%;max-width:{WIDTH_PX}px;height:auto;'
            f'border-radius:4px;">')
    return html.replace(_OPEN_DIV, _OPEN_DIV + "".join(parts), 1)


def pack(charts: List[dict]) -> Optional[str]:
    """For the outbox: the pictures as JSON, so a retried email still carries them."""
    import base64
    if not charts:
        return None
    return json.dumps([dict(cid=c["cid"], png=base64.b64encode(c["png"]).decode())
                       for c in charts])


def unpack(s: Optional[str]) -> List[Tuple[str, bytes]]:
    import base64
    if not s:
        return []
    return [(c["cid"], base64.b64decode(c["png"])) for c in json.loads(s)]
