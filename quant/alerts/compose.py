"""
One email from any number of events: a section per CATEGORY in the config's order, a row per
signal, urgent rows first and marked, and the events found late in their own section at the end.

Plain text and HTML are built from the same rows, and the HTML is inline-styled tables only --
what every mail client renders the same way.

With DELIVERY `subject="tiers"`, the subject leads with the email's most important row and its TIER,
so a mail filter can act on it (tier() has the rules):
  🔴 P1  act now   critical news, or high USD news, within the hour, and its result;
                   a trendline BREAK or a pattern BREAKOUT on 1h or above
  🟠 P2  look soon other high news within the hour; critical / USD news 2-6 h ahead; a moved
                   release; a break on 15m-30m; a line or pattern FORMED on 1h or above
  ⚪ P3  record    everything else: 1m / 5m signals, smaller formations, far reminders, digests
  e.g.  [Alerts] 🔴 P1 · USD CPI m/m in 15 min · BTC 1h broke support line · +4 more · 13:17 UTC

With DELIVERY `collapse_repeats`, the same signal repeating -- one alert, market, timeframe and
signal, five 1m closes through a line in one quarter hour -- is ONE row saying how many times and
from when to when. With "News" placed after the other CATEGORIES, the news comes last.
"""
from __future__ import annotations

import html as _html
import json
from collections import OrderedDict

import pandas as pd

from .config import TF_MINUTES

MAX_LATE_ROWS = 40


def _price(p: float) -> str:
    dp = 5 if p < 10 else 3 if p < 1000 else 2 if p < 10_000 else 1
    return f"{p:,.{dp}f}"


def _link(e) -> str:
    tv = json.loads(e["detail"] or "{}").get("tv")
    if not tv:
        return ""
    return (f"https://www.tradingview.com/chart/?symbol={tv}"
            f"&interval={TF_MINUTES.get(e['tf'], 60)}")


def _note(e) -> str:
    return json.loads(e["detail"] or "{}").get("note", "") or ""


def _ref(e) -> str:
    """The row's ref: `python -m quant.alerts show <ref>` draws that alert (quant.alerts.show)."""
    from .show import make_ref
    return make_ref(e["market"], e["tf"], e["close_time"], e["alert"]) if e.get("alert") else ""


def _when(s: str) -> str:
    return pd.Timestamp(s).tz_convert("UTC").strftime("%a %d %b %H:%M")


def _short(s: str) -> str:
    return pd.Timestamp(s).tz_convert("UTC").strftime("%d %b %H:%M")


def _hm(s: str) -> str:
    return pd.Timestamp(s).tz_convert("UTC").strftime("%H:%M")


def collapse(events, every_minutes: int = 0):
    """One row per (alert, market, timeframe, signal) -- and per `every_minutes` window of bar
    closes, so a preview spanning two days still reads window by window. The row kept is the
    LATEST; its note says how many times and from when."""
    groups = OrderedDict()
    for e in sorted(events, key=lambda e: e["close_time"]):
        slot = (pd.Timestamp(e["close_time"]).tz_convert("UTC").floor(f"{every_minutes}min")
                if every_minutes else None)
        key = (e["alert"], e["market"], e["tf"], e["what"], bool(e["late"]), slot)
        groups.setdefault(key, []).append(e)
    out = []
    for g in groups.values():
        last = dict(g[-1])
        if len(g) > 1:
            d = json.loads(last["detail"] or "{}")
            span = f"{len(g)} times, {_hm(g[0]['close_time'])}-{_hm(g[-1]['close_time'])} UTC"
            d["note"] = span + (f" · last: {d['note']}" if d.get("note") else "")
            last["detail"] = json.dumps(d)
            if any(x["priority"] == "urgent" for x in g):
                last["priority"] = "urgent"
        out.append(last)
    return out


def _rows(events):
    order = {"urgent": 0, "normal": 1}
    return sorted(events, key=lambda e: (order.get(e["priority"], 2), e["close_time"],
                                         e["market"], TF_MINUTES.get(e["tf"], 0)))


_OPEN_DIV = '<div style="font:14px -apple-system,Segoe UI,Arial,sans-serif;color:#111827;">'


TIER_TAG = {1: "🔴 P1", 2: "🟠 P2", 3: "⚪ P3"}


def tier(e) -> int:
    """1 = act now, 2 = look soon, 3 = for the record (see the module docstring)."""
    from quant.news.catalog import RANK
    from .news import is_news, notice_of
    if is_news(e):
        items = notice_of(e).get("items") or []
        lv = max((RANK.get(i.get("level"), 0) for i in items), default=0)
        usd = any(i.get("currency") == "USD" for i in items)
        top = lv >= RANK["critical"] or (usd and lv >= RANK["high"])
        if e["what"] == "reminder":
            tf = str(e["tf"])
            k = int(tf[2:]) if tf.startswith("T-") and tf[2:].isdigit() else 0
            if top:
                return 1 if k <= 60 else 2 if k <= 360 else 3
            return 2 if k <= 60 and lv >= RANK["high"] else 3
        if e["what"] == "result":
            return 1 if top else 2
        return 2 if e["tf"] in ("moved", "removed") else 3
    if e.get("late"):
        return 3
    m = TF_MINUTES.get(e["tf"], 0)
    w = e["what"].lower()
    line_tf = w.split(" ", 1)[0]
    if line_tf in TF_MINUTES and (" close below" in w or " close above" in w):
        # an EARLY break (early_break): a coarser line closed through on a finer candle. Ranked
        # by the LINE's timeframe, one step below that line's own confirmed break
        return 2 if TF_MINUTES[line_tf] >= 60 else 3
    if "broke" in w or "breakout" in w:
        return 1 if m >= 60 else 2 if m >= 15 else 3
    if "formed" in w:
        return 2 if m >= 60 else 3
    return 2 if m >= 240 else 3


def _headline(e) -> str:
    """A row in a few words, for the subject."""
    from .news import is_news, notice_of
    if is_news(e):
        return notice_of(e).get("subject") or e["what"]
    w = (e["what"].replace("broke a ", "broke ").replace(" trendline", " line"))
    if w.split(" ", 1)[0] in TF_MINUTES:              # "1h support line: 5m close below" names
        return f"{e['market']} {w}"                   # its own timeframes
    return f"{e['market']} {e['tf']} {w}"


def rank_key(e):
    """Most important first: the tier, news before signals, the coarser timeframe, the earlier."""
    return (tier(e), 0 if e.get("alert") == "news" else 1, -TF_MINUTES.get(e["tf"], 0),
            str(e.get("close_time", "")))


def tier_subject(events, cfg, now) -> str:
    """[Alerts] 🔴 P1 · <the most important row> · <the next> · +N more · HH:MM UTC"""
    ranked = sorted(events, key=rank_key)
    heads = list(dict.fromkeys(_headline(e) for e in ranked))
    more = len(heads) - 2
    now = pd.Timestamp(now).tz_convert("UTC")
    return (f"{cfg.delivery.subject_prefix} {TIER_TAG[tier(ranked[0])]} · "
            + " · ".join(heads[:2]) + (f" · +{more} more" if more > 0 else "")
            + f" · {now:%H:%M} UTC")


def compose(events, cfg, now, *, host: str = "", title: str = ""):
    """-> (subject, text, html); with DELIVERY subject="tiers" the subject is tier_subject()."""
    # an instant alert is an urgent one that does not wait for the clock: it reads as urgent
    events = [dict(e, priority="urgent") if e.get("priority") == "instant" else e for e in events]
    subject, text, html = _compose(events, cfg, now, host=host, title=title)
    if getattr(cfg.delivery, "subject", "classic") == "tiers" and events:
        subject = tier_subject(events, cfg, now)
    return subject, text, html


def _compose(events, cfg, now, *, host: str = "", title: str = ""):
    """-> (subject, text, html). News notices (quant.alerts.news) come first, in their own
    section; an email without them is exactly what it always was."""
    from .news import CATEGORY, is_news, news_html, news_subject, news_text
    news = [e for e in events if is_news(e)]
    if getattr(cfg.delivery, "collapse_repeats", False):
        events = news + collapse([e for e in events if not is_news(e)],
                                 getattr(cfg.delivery, "every_minutes", 0))
    if not news:
        return _compose_signals(events, cfg, now, host=host, title=title)
    rest = [e for e in events if not is_news(e)]
    now = pd.Timestamp(now).tz_convert("UTC")
    ntext = "\n".join(["NEWS", "----", *news_text(news)])
    nhtml = '<h3 style="margin:18px 0 6px;font-size:15px;">News</h3>' + news_html(news)
    if rest:
        subject, text, html = _compose_signals(rest, cfg, now, host=host, title=title)
        cats = list(cfg.categories)
        if CATEGORY in cats and cats.index(CATEGORY) > 0:
            # News placed after the other sections: it goes last, above the footer
            foot = text.rfind("\n-- quant alerts")
            text = text[:foot] + "\n\n" + ntext + text[foot:]
            cut = html.rfind('<p style="margin-top:22px;color:#9ca3af;font-size:12px;">')
            return f"{subject} · +{len(news)} news", text, html[:cut] + nhtml + html[cut:]
        return (f"{subject} · +{len(news)} news", ntext + "\n\n" + text,
                html.replace(_OPEN_DIV, _OPEN_DIV + nhtml, 1))
    bolt = "⚡ " if any(e["priority"] == "urgent" for e in news) else ""
    subject = f"{cfg.delivery.subject_prefix} {bolt}{news_subject(news)} · {now:%H:%M} UTC"
    foot = f"quant alerts{' on ' + host if host else ''}"
    text = "\n".join(([title, ""] if title else []) + [ntext, "", f"-- {foot} · {cfg.path.name}"])
    html = (_OPEN_DIV + (f'<p style="color:#6b7280;">{_html.escape(title)}</p>' if title else "")
            + nhtml + f'<p style="margin-top:22px;color:#9ca3af;font-size:12px;">'
            f'{_html.escape(foot)} &middot; {_html.escape(cfg.path.name)}</p></div>')
    return subject, text, html


def _compose_signals(events, cfg, now, *, host: str = "", title: str = ""):
    """The candle alerts' email -> (subject, text, html)."""
    live = [e for e in events if not e["late"]]
    late = [e for e in events if e["late"]]
    n_urgent = sum(e["priority"] == "urgent" for e in live)
    markets = list(dict.fromkeys(e["market"] for e in _rows(live) + _rows(late)))
    now = pd.Timestamp(now).tz_convert("UTC")

    bolt = "⚡ " if n_urgent else ""
    n = len(events)
    subject = (f"{cfg.delivery.subject_prefix} {bolt}{n} signal{'s' * (n != 1)}"
               f"{f' ({n_urgent} urgent)' if n_urgent else ''} · {', '.join(markets[:4])}"
               f"{' +' + str(len(markets) - 4) if len(markets) > 4 else ''}"
               f" · {now:%H:%M} UTC")

    sections = OrderedDict((c, []) for c in cfg.categories)
    for e in live:
        sections.setdefault(e["category"], []).append(e)
    sections = OrderedDict((c, _rows(v)) for c, v in sections.items() if v)
    late_rows = _rows(late)[:MAX_LATE_ROWS]

    # ── plain text ─────────────────────────────────────────────────────────────────────────
    t = [title] if title else []
    for cat, rows in sections.items():
        t += ["", cat.upper(), "-" * len(cat)]
        for e in rows:
            mark = "!" if e["priority"] == "urgent" else " "
            note = f"  ({_note(e)})" if _note(e) else ""
            t.append(f"{mark} {e['market']:<8} {e['tf']:>4}  {e['what']:<24} "
                     f"closed {_when(e['close_time'])} UTC  @ {_price(e['price'])}{note}  "
                     f"{_link(e)}" + (f"  ref {_ref(e)}" if _ref(e) else ""))
    if late:
        t += ["", "WHILE OFFLINE -- found late, for the record", "-" * 43]
        for e in late_rows:
            t.append(f"  {e['market']:<8} {e['tf']:>4}  {e['what']:<24} "
                     f"closed {_when(e['close_time'])} UTC  @ {_price(e['price'])}")
        if len(late) > len(late_rows):
            t.append(f"  ... and {len(late) - len(late_rows)} more")
    t += ["", f"-- quant alerts{' on ' + host if host else ''} · {cfg.path.name}"]
    text = "\n".join(t).lstrip("\n")

    # ── html ───────────────────────────────────────────────────────────────────────────────
    esc = _html.escape
    td = "padding:5px 8px;border-bottom:1px solid #e5e7eb;white-space:nowrap;"
    th = "padding:2px 8px;text-align:left;font-weight:400;font-size:12px;color:#9ca3af;"
    side_col = {"bull": "#15803d", "bear": "#b91c1c"}

    def table(rows, with_link=True):
        out = ['<table style="border-collapse:collapse;font:14px -apple-system,Segoe UI,'
               f'Arial,sans-serif;"><tr><th style="{th}"></th><th style="{th}">market</th>'
               f'<th style="{th}">signal</th><th style="{th}">close</th>'
               f'<th style="{th}">bar closed, UTC</th><th style="{th}"></th></tr>']
        for e in rows:
            urgent = e["priority"] == "urgent" and not e["late"]
            colour = side_col.get(e["side"], "#111827")
            link = _link(e)
            out.append(
                "<tr>"
                f'<td style="{td}padding-right:0;color:#b45309;">'
                f'{"&#9889;" if urgent else ""}</td>'
                f'<td style="{td}"><b>{esc(e["market"])}</b> '
                f'<span style="color:#6b7280;">{esc(e["tf"])}</span></td>'
                f'<td style="{td}color:{colour};font-weight:{600 if urgent else 400};">'
                f'{esc(e["what"])}'
                + (f'<br><span style="font-size:12px;color:#6b7280;font-weight:400;">'
                   f'{esc(_note(e))}</span>' if _note(e) else "")
                + (f'<br><span style="font:11px Consolas,monospace;color:#9ca3af;">'
                   f'ref {esc(_ref(e))}</span>' if with_link and _ref(e) else "")
                + '</td>'
                f'<td style="{td}font-variant-numeric:tabular-nums;">{_price(e["price"])}</td>'
                f'<td style="{td}color:#6b7280;">{esc(_short(e["close_time"]))}</td>'
                + (f'<td style="{td}"><a href="{esc(link)}">chart</a></td>'
                   if with_link and link else f'<td style="{td}"></td>')
                + "</tr>")
        out.append("</table>")
        return "".join(out)

    h = [_OPEN_DIV]
    if title:
        h.append(f'<p style="color:#6b7280;">{esc(title)}</p>')
    for cat, rows in sections.items():
        h.append(f'<h3 style="margin:18px 0 6px;font-size:15px;">{esc(cat)}</h3>')
        h.append(table(rows))
    if late:
        h.append('<h3 style="margin:22px 0 6px;font-size:15px;color:#6b7280;">While offline '
                 '<span style="font-weight:400;">&mdash; found late, for the record</span></h3>')
        h.append(table(late_rows, with_link=False))
        if len(late) > len(late_rows):
            h.append(f'<p style="color:#6b7280;">&hellip; and {len(late) - len(late_rows)} more'
                     "</p>")
    h.append(f'<p style="margin-top:22px;color:#9ca3af;font-size:12px;">quant alerts'
             f'{" on " + esc(host) if host else ""} &middot; {esc(cfg.path.name)}</p></div>')
    return subject, text, "".join(h)
