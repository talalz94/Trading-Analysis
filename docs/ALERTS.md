# Alerts — an email when a pattern forms

`quant.alerts` watches markets on this PC and emails you when something you named forms on a closed
candle. You decide what to watch in **one file, `alerts/alerts_config.py`**: the markets, the
timeframes, the alerts and how often to be emailed. The monitor re-reads it every minute, so an edit
needs no restart.

> **Read this first.** An alert says *something formed*. It does not say *trade it*. The first
> alert kind is candlestick patterns, and over 62 patterns their hit rate is 50.1–50.5%: no
> directional edge (`docs/WHAT_WORKS.md`). Use alerts to bring a chart to your attention, not as
> signals.

## Contents

1. [Set up email (once)](#1-set-up-email-once)
2. [Try it without sending anything](#2-try-it-without-sending-anything)
3. [Start the monitor](#3-start-the-monitor)
4. [The config file](#4-the-config-file)
5. [When an email is sent](#5-when-an-email-is-sent)
6. [What survives an outage, and how](#6-what-survives-an-outage-and-how)
7. [Cost](#7-cost)
8. [Adding a new kind of alert](#8-adding-a-new-kind-of-alert)
9. [Files and commands](#9-files-and-commands)
10. [Known limits](#10-known-limits)

---

## 1. Set up email (once)

Alerts go out through Gmail's SMTP server: free, reliable, and nothing to host. Gmail will not
accept your normal password from a program, so it needs an **App Password** for this one use.

1. In your Google Account, go to **Security** and turn on **2-Step Verification** if it is off. App
   Passwords do not exist without it.
2. Go to **Security → App passwords** (or search the account settings for "App passwords"). Create
   one named `quant alerts`. Google shows 16 letters once; copy them.
3. Add these lines to **`.env`** in the repo root. `.env` is git-ignored and never committed.

   ```
   ALERT_SMTP_USER=you@gmail.com
   ALERT_SMTP_PASSWORD=the16lettersfromgoogle
   ALERT_EMAIL_TO=you@gmail.com
   ```

   `ALERT_EMAIL_TO` may list several addresses, comma-separated. If it is left out, mail goes to
   `ALERT_SMTP_USER`. Another provider works too: set `ALERT_SMTP_HOST` and `ALERT_SMTP_PORT`
   (STARTTLS; default `smtp.gmail.com`, `587`).
4. Prove it works:

   ```bash
   python -m quant.alerts test-email
   ```

To revoke access later, delete the App Password in your Google Account. Nothing else changes.

## 2. Try it without sending anything

```bash
python -m quant.alerts check
```

This fetches live data, runs every alert over the last 48 hours (`--hours 24` for a different
window), and writes the email it **would** have sent to `alerts/state/preview.html`. It sends
nothing and does not touch the monitor's state. Run it after every change to the config.

## 3. Start the monitor

```bash
powershell -ExecutionPolicy Bypass -File scripts/install_alerts_task.ps1
```

That registers a Windows scheduled task, `quant-alerts`, which:

- starts `pythonw -m quant.alerts run` (no console window) when you log on;
- re-triggers every 5 minutes as a **watchdog**. While the monitor is running the trigger is
  ignored. If it has died, it is back within 5 minutes, and its first tick catches up.

It runs as you, only while you are logged on, with no stored password. To remove it, run
`... install_alerts_task.ps1 -Uninstall`.

Day to day:

```bash
python -m quant.alerts status
```

`status` shows whether it is running, the last tick and the last email, what is queued, any feed
errors, and the last bar each alert judged. The log is `alerts/state/alerts.log`.

## 4. The config file

`alerts/alerts_config.py`, in three sections:

```python
MARKETS = {                                  # every market is fetched at 5m
    "Gold":   dict(symbol="XAUUSD",  source="dukascopy"),
    "USDJPY": dict(symbol="USDJPY",  source="dukascopy"),
    "BTC":    dict(symbol="BTCUSDT", source="binance"),
}
GROUPS = {"metals & FX": ("Gold", "USDJPY")} # names an alert can use; "all" always exists

CATEGORIES = ("Candle patterns",)            # the order sections appear in an email

ALERTS = [
    dict(name="engulfing 4h", category="Candle patterns", kind="candle",
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs=("4h",), priority="urgent"),
    ...
]

DELIVERY = dict(digest_minutes=15, catchup_max_age_minutes=30)
```

| key | meaning |
|---|---|
| `name` | Unique. It is also the alert's identity in the state: **renaming an alert starts it fresh**. |
| `category` | The email section it appears under. |
| `kind` | What can fire it. Today only `candle`; see §8 for adding more. |
| `markets` | Market names, group names, or `"all"`. |
| `tfs` | Any of `5m 15m 30m 1h 2h 4h`. All of them are built from the 5m feed. |
| `priority` | `urgent`: emailed on the tick that finds it. `normal`: waits for the digest. |
| `cooldown_bars` | After it fires on a market and timeframe, stay quiet there for this many of its bars. |
| `patterns` | (`candle`) Names from `quant/indicators/candle_patterns.py`, or `"all"`. |

A mistake (an unknown market, timeframe, pattern or kind) is refused with a message that names it.
The monitor then keeps running on the **last good config** and logs the error. A typo cannot
silently switch your alerts off.

**How often the shipped example fires** (Gold + USDJPY + BTC together, per day, Mar–Jul 2026):

| pattern | 15m | 1h | 4h |
|---|---|---|---|
| engulfing (bull + bear) | ~38 | ~10 | ~2.5 |
| hammer + shooting star | ~10 | ~2.4 | ~0.6 |

That is why engulfing is urgent only on 4h and not watched on 15m at all.

## 5. When an email is sent

- **A bar is judged once it is final**, never while it is still forming. A bar is final when it
  has been closed for 90 seconds (Dukascopy publishes ~1 minute late; waiting the same for every
  market puts one bar close's alerts in **one** email). Its 5m data must also reach its close, or
  the feed must still have nothing 20 minutes after the close. The second case is a shut market,
  for example Friday's last 4h bar.
- **Urgent** alerts are emailed on the tick that finds them. **Normal** ones wait for the digest:
  at most one email every `digest_minutes`. **Any email carries every pending alert**, so an urgent
  email also clears the waiting digest.
- **One email covers several markets.** The subject reads like
  `[Alerts] ⚡ 3 signals (1 urgent) · Gold, BTC · 14:01 UTC`. The body has a section per category,
  urgent rows first, with a TradingView link per row.
- **No history on switch-on.** The first time an alert sees a market and timeframe, it starts from
  that moment. Adding an alert does not mail you the past 19 days of matches.

## 6. What survives an outage, and how

| what fails | what happens |
|---|---|
| **The PC sleeps, reboots or is off** | On the next tick, every bar that closed meanwhile is judged in order. A signal found more than `catchup_max_age_minutes` after its bar closed is listed under **"While offline"** for the record. It is never sent as urgent, because it is no longer actionable. |
| **The monitor process dies** | The scheduled task restarts it within 5 minutes. |
| **A second copy starts** | It finds the lock (`alerts/state/alerts.lock`, an OS lock released when the process dies) and exits. |
| **A feed fails or hangs** | That market is skipped and retried after 1, 2, 4 … 15 minutes. The other markets carry on. A hung request has a timeout, and a market whose last request is still hanging is not sent another. |
| **Gmail is unreachable** | The email stays in the outbox and is retried after 1, 2, 4 … 30 minutes, for up to 20 attempts. |
| **A crash between finding a signal and recording it** | Every event has an id hashed from (alert, market, timeframe, bar, pattern). Recording it twice does nothing, so re-judging a bar can never send it again. |
| **A crash while writing a bar file** | Bar files are written to a temporary name and renamed into place, so a reader sees the old file or the new one, never half of each. |

All state lives in `alerts/state/` (git-ignored). To start completely fresh, stop the task and
delete that folder.

## 7. Cost

Measured on this PC:

- **Fetching.** A market is polled only when one of its bars is due to close, so with 15m as the
  finest timeframe that is 4 polls an hour, not 60. Each poll asks only for the new 5m bars:
  about 2 s from Binance and 4 s from Dukascopy, run in parallel. A shut market (a weekend) backs
  off to about 4 polls an hour.
- **Detecting.** Each alert is evaluated only on its new bars plus its warm-up (60 bars for
  candles), never on the whole history. That takes milliseconds, after a one-time ~2 s numba
  compile when the process starts.
- **Storage.** The last ~19 days of 5m bars per market (enough to warm up a 4h alert), a few
  hundred KB.
- **First run, or `check`.** A full 19-day backfill of three markets takes about 20 s.

## 8. Adding a new kind of alert

A kind is a `Detector` in `quant/alerts/detectors.py`. It gets one timeframe's **final** candles
(columns `open_time open high low close volume`) and returns the rows it fires on:

```python
class MyDetector(Detector):
    kind = "my_kind"
    def validate(self, params): ...           # raise KeyError/ValueError naming the problem
    def warmup_bars(self, params): return 200 # bars of history it needs before the bar it judges
    def evaluate(self, frame, params):
        return [dict(i=row, what="short phrase", side="bull"), ...]

REGISTRY = {d.kind: d for d in (CandleDetector(), MyDetector())}
```

The engine needs no change. The detector **must be causal**: running it on the first k bars must
give the same answers on those bars as running it on all of them. Every indicator in `quant/` is
built and tested this way, so wrapping one is safe. `tests/test_alerts.py` has the truncation test
to copy. The planned next kinds are notebook 21's setups and trendline breaks
(`add_trendlines` already gives `tl_break_up` / `tl_break_dn`).

## 9. Files and commands

```
alerts/alerts_config.py        WHAT to watch -- yours
alerts/state/                  git-ignored: alerts.db (SQLite), bars/*.parquet, alerts.log,
                               alerts.lock, preview.html
quant/alerts/config.py         reads and checks the config (fresh each tick)
            feeds.py           closed 5m bars from Binance / Dukascopy, with timeouts
            detectors.py       what can fire an alert (candle; add more here)
            engine.py          one resumable tick: fetch what is due, judge, queue, send
            store.py           SQLite state + atomic bar files
            compose.py         the email (text + HTML, categories in config order)
            notify.py          Gmail SMTP from .env; FileNotifier / ListNotifier
            runner.py          the loop, the lock, the log, `status`
scripts/install_alerts_task.ps1   the scheduled task (install / -Uninstall)
tests/test_alerts.py           fake feed + fake clock: every rule above is a test
```

```bash
python -m quant.alerts check        # dry run over the last 48h -> alerts/state/preview.html
python -m quant.alerts test-email   # prove the .env settings
python -m quant.alerts tick         # one pass, then exit (refuses while the monitor runs)
python -m quant.alerts run          # the monitor loop (what the scheduled task starts)
python -m quant.alerts status
```

## 10. Known limits

- **4h bars start at 00, 04, 08 … UTC**, the same as the research resampler. A broker chart that
  aligns 4h candles to the New York close draws different candles, so a 4h alert can differ from
  what that chart shows.
- **Dukascopy prices are the BID side**, as in the research data, so a candle can differ slightly
  from a mid-price chart.
- **It runs only while this PC is on and you are logged on.** It catches up afterwards, but cannot
  email you during the gap. Moving it to an always-on machine needs only the repo, `.env` and the
  scheduled task (or cron on Linux: the lock works there too).
