"""
Push notifications: every alert email also pops up on your phone and PC, through ntfy (ntfy.sh,
free, no account), sent by the server the moment the email is composed -- so it does not depend on
Gmail, its filters or its notification rules. DELIVERY push="ntfy" turns it on.

The settings live in the environment (.env, git-ignored), never in the repo:

    NTFY_TOPIC    the topic your phone and browser subscribe to. It works like a PASSWORD: anyone
                  who knows it can read your alerts and post to it, so make it long and random
                  (`echo NTFY_TOPIC=alerts-$(openssl rand -hex 12) >> .env`)
    NTFY_SERVER   default https://ntfy.sh (or your own ntfy server)
    NTFY_TOKEN    optional: an access token, for a server where the topic is reserved

The message is posted as JSON to the server's ROOT url, so the topic never appears in a URL -- nor
in an error message or a log line. One notification per email: its subject as the title, a line per
signal (repeats counted), the top row's tier as the priority (P1 urgent, P2 high, P3 default), and a
tap opens the top signal's chart on TradingView. ntfy.sh allows 250 messages a day per server; the
trendline config sends ~60 emails a day.
"""
from __future__ import annotations

import os
from collections import OrderedDict

from .notify import NotConfigured

PRIORITY = {1: 5, 2: 4, 3: 3}          # tier -> ntfy priority (5 urgent, 4 high, 3 default)
MAX_LINES = 8


class PushError(RuntimeError):
    pass


def message(events, subject: str, cfg) -> dict:
    """The notification for one email: what its subject says, a line per signal, most important
    first, and where a tap goes."""
    from .compose import _headline, _hm, _link, _price, rank_key, tier
    from .news import is_news
    prefix = cfg.delivery.subject_prefix
    title = subject[len(prefix):].strip() if subject.startswith(prefix) else subject
    ranked = sorted(events, key=rank_key)
    groups: OrderedDict = OrderedDict()
    for e in ranked:
        groups.setdefault(_headline(e), []).append(e)
    lines = []
    for head, g in groups.items():
        e = g[0]
        line = head if is_news(e) else f"{head} @ {_price(e['price'])} · {_hm(e['close_time'])} UTC"
        lines.append(line + (f" ({len(g)}x)" if len(g) > 1 else ""))
    if len(lines) > MAX_LINES:
        lines = lines[:MAX_LINES] + [f"+{len(lines) - MAX_LINES} more"]
    msg = dict(title=title, message="\n".join(lines), priority=PRIORITY[tier(ranked[0])])
    link = next((_link(e) for e in ranked if not is_news(e) and _link(e)), "")
    if link:
        msg["click"] = link
    return msg


class NtfyPusher:
    def __init__(self, topic: str, server: str = "https://ntfy.sh", token=None, timeout=10.0):
        self._topic, self.server, self._token, self.timeout = topic, server, token, timeout

    @classmethod
    def from_env(cls):
        import quant.config  # noqa: F401  -- loads .env into the environment
        topic = os.environ.get("NTFY_TOPIC", "").strip()
        if not topic:
            raise NotConfigured("set NTFY_TOPIC in .env (docs/ALERTS.md, 'Push notifications')")
        return cls(topic, os.environ.get("NTFY_SERVER") or "https://ntfy.sh",
                   os.environ.get("NTFY_TOKEN") or None)

    def send(self, msg: dict):
        from .feeds import session                     # kept connection, one certificate store
        headers = {"Authorization": f"Bearer {self._token}"} if self._token else {}
        r = session().post(self.server.rstrip("/") + "/", json=dict(msg, topic=self._topic),
                           headers=headers, timeout=self.timeout)
        if r.status_code >= 300:                       # the body names the problem, not the topic
            raise PushError(f"ntfy answered HTTP {r.status_code}: {r.text[:160].strip()}")


class ListPusher:
    """Keeps notifications in memory; `fail=n` raises on the next n sends. For tests."""

    def __init__(self, fail: int = 0):
        self.sent, self.fail = [], fail

    def send(self, msg: dict):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("fake ntfy outage")
        self.sent.append(msg)
