"""
Where an email goes. Every notifier has `send(subject, text, html, images=None)` and raises if it
did not go -- the outbox only marks an email sent when this returns. `images` = [(content id, PNG
bytes)]: the chart pictures, sent as inline attachments the HTML shows with <img src="cid:...">.

SmtpNotifier reads its settings from the environment (.env, git-ignored), never from a file in
the repo:

    ALERT_SMTP_USER       you@gmail.com
    ALERT_SMTP_PASSWORD   a Gmail APP PASSWORD (16 letters), not your account password
    ALERT_EMAIL_TO        where alerts go; comma-separate several (defaults to ALERT_SMTP_USER)
    ALERT_SMTP_HOST       default smtp.gmail.com
    ALERT_SMTP_PORT       default 587 (STARTTLS)
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path

import pandas as pd


class NotConfigured(RuntimeError):
    pass


class SmtpNotifier:
    def __init__(self, user, password, to, host="smtp.gmail.com", port=587, timeout=30):
        self.user, self._password = user, password
        self.to = [x.strip() for x in to.split(",") if x.strip()]
        self.host, self.port, self.timeout = host, int(port), timeout

    @classmethod
    def from_env(cls):
        import quant.config  # noqa: F401  -- loads .env into the environment
        env = os.environ
        missing = [k for k in ("ALERT_SMTP_USER", "ALERT_SMTP_PASSWORD") if not env.get(k)]
        if missing:
            raise NotConfigured(f"set {', '.join(missing)} in .env (see docs/ALERTS.md)")
        return cls(env["ALERT_SMTP_USER"], env["ALERT_SMTP_PASSWORD"],
                   env.get("ALERT_EMAIL_TO") or env["ALERT_SMTP_USER"],
                   env.get("ALERT_SMTP_HOST", "smtp.gmail.com"),
                   env.get("ALERT_SMTP_PORT", 587))

    def build(self, subject, text, html, images=None) -> EmailMessage:
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, self.user, ", ".join(self.to)
        msg["Date"], msg["Message-ID"] = formatdate(localtime=False), make_msgid()
        msg.set_content(text)
        msg.add_alternative(html, subtype="html")
        if images:                                     # text | (html + its pictures)
            part = msg.get_payload()[-1]
            for cid, png in images:
                part.add_related(png, maintype="image", subtype="png", cid=f"<{cid}>")
        return msg

    def send(self, subject, text, html, images=None):
        msg = self.build(subject, text, html, images)
        with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(self.user, self._password)
            s.send_message(msg)


class FileNotifier:
    """Writes each email to a folder instead of sending it (dry runs, and a record to read)."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    def send(self, subject, text, html, images=None):
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S_%f")
        for i, (cid, png) in enumerate(images or []):  # the pictures beside the page
            (self.folder / f"{stamp}_{i}.png").write_bytes(png)
            html = html.replace(f"cid:{cid}", f"{stamp}_{i}.png")
        (self.folder / f"{stamp}.html").write_text(
            f"<!doctype html><meta charset='utf-8'><title>{subject}</title>{html}",
            encoding="utf-8")


class ListNotifier:
    """Keeps emails in memory; `fail=n` raises on the next n sends. For tests."""

    def __init__(self, fail: int = 0):
        self.sent, self.fail, self.images = [], fail, []

    def send(self, subject, text, html, images=None):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("fake SMTP outage")
        self.sent.append((subject, text, html))
        self.images.append(list(images or []))
