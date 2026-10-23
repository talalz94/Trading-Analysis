# Alerts — an email when a pattern forms

`quant.alerts` watches markets on this PC and emails you when something you named forms on a closed
candle. You decide what to watch in **one file, `alerts/alerts_config.py`**: the markets, the
timeframes, the alerts and how often to be emailed. The monitor re-reads it every minute, so an edit
needs no restart.

> **Read this first.** An alert says *something formed*. It does not say *trade it*. The patterns
> and indicators here were tested in this repo and none has a directional edge. Candlestick
> patterns, for example, hit 50.1–50.5% across all 62 (`docs/WHAT_WORKS.md`). Use alerts to bring
> a chart to your attention, not as signals.
>
> **The full menu** (every instrument, timeframe, pattern, alert kind and setup) is
> [ALERTS_CATALOGUE.md](ALERTS_CATALOGUE.md), or `python -m quant.alerts list`.

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
11. [News](#11-news)

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

**On a cloud server** (it keeps running while your PC is off): follow
[ALERTS_CLOUD.md](ALERTS_CLOUD.md). It covers Google Cloud's free tier step by step, and
`scripts/install_alerts_linux.sh` installs the monitor as a Linux service there. The rest of this
section is for running it on this PC instead. **Never both:** every alert would arrive twice.

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

`alerts/alerts_config.py` has three sections: **MARKETS**, **ALERTS** and **DELIVERY**. Every
alert has an **on/off switch**. The file ships with one alert of each kind written out and
switched off, with how often each fires, so turning one on means changing `on=False` to
`on=True`:

```python
MARKETS = {
    "Gold":     dict(symbol="XAUUSD",  source="dukascopy"),
    "BTC":      dict(symbol="BTCUSDT", source="binance"),
    "S&P 500":  dict(symbol="US500",   source="dukascopy"),     # any name you like
}

ALERTS = [
    dict(on=True,  name="engulfing 4h", kind="candle",                              # ~2.5/day
         patterns=("engulfing_bull", "engulfing_bear"),
         markets="all", tfs="4h", priority="urgent"),

    dict(on=False, name="trendline break", kind="trendline", min_touches=3,         # ~0.3/day
         markets="all", tfs=("1h", "4h"), priority="normal"),
]

DELIVERY = dict(digest_minutes=15, catchup_max_age_minutes=30)
```

| key | meaning |
|---|---|
| `on` | `True` = watching. `False` = kept in the file but not run (still checked for mistakes). |
| `name` | Unique. It is the alert's identity in the state, so **renaming an alert starts it afresh**. |
| `kind` | What to look for, followed by that kind's own settings (table below). |
| `markets` | Market names, group names, or `"all"`. |
| `tfs` | Any of `1m 5m 15m 30m 1h 2h 4h`, as one string or a tuple. All but 1m are built from the 5m feed; 1m alerts read 1m bars, fetched only for the markets a 1m alert watches and kept only for its few hours of warm-up. |
| `priority` | `urgent`: emailed on the tick that finds it. `normal`: waits for the digest. `instant`: like urgent, and on a clock (`every_minutes`) it does not wait for the boundary either -- see §5. |
| `cooldown_bars` | (optional) After it fires on a market and timeframe, stay quiet there for this many of its bars. |
| `category` | (optional) The email section. Defaults to the kind's own: Candle patterns, Chart patterns, Levels & lines, Indicators, Setups. |

**The kinds** (settings, defaults and an example of each are in [ALERTS_CATALOGUE.md](ALERTS_CATALOGUE.md)):

| category | kind | fires when |
|---|---|---|
| Candle patterns | `candle` | one of 62 candlestick patterns completes |
| Chart patterns | `chart` | one of 22 chart patterns **forms** and/or **breaks out** (`event=`), with its target and stop |
| Levels & lines | `trendline` | a fitted trendline **forms**, and/or price **breaks it by a rule** (below) |
| | `early_break` | a **coarser** timeframe's trendline closed through by a candle of **this** one: a 1h line on a 5m close, up to an hour before the 1h candle closes (below) |
| | `sr_zone` | a support/resistance zone is **broken by a rule**, or price bounces off it |
| | `fvg` | a fair value gap forms, or gets filled |
| Indicators | `ma_cross` | price crosses an EMA/SMA, or a golden/death cross |
| | `rsi` | RSI enters, or leaves, overbought/oversold |
| | `rsi_divergence` | bullish or bearish RSI divergence (arrives `right` bars late, by construction) |
| | `macd` | MACD crosses its signal line, or zero |
| | `stochastic` | %K crosses %D in overbought/oversold |
| | `supertrend` | Supertrend flips |
| | `heikin_ashi` | Heikin-Ashi changes colour after N candles |
| | `candle_run` | N candles in a row of one colour |
| | `big_candle` | a candle at least k × ATR long |
| Setups | `setup` | one of 28 research entry setups (break of structure, squeeze, WaveTrend cross, VWAP reversion, support break …) would open a trade |

**Formation and break, as two separate alerts.** `event="formed"` fires when the thing first
exists. A trendline exists from the candle its last touch is confirmed on. A chart pattern exists
once its swings are in place, and it then waits for a break, whether or not one ever comes.
`event="break"` (or `"breakout"` for chart patterns) fires when price breaks it, and
`event="both"` sends both. A formation note says where the break would be, for example
`breaks below 84,120 → target 82,000`. A break note says which line broke, for example
`2 closes below · line formed 10 Oct 04:05 · 3 touches`.

**Early breaks.** A 1h line only counts as broken when a 1h candle closes through it, which can
be up to an hour after price first went through. `early_break` watches the lines of the
timeframes in `line_tfs` (fitted exactly as the `trendline` kind fits them, with the same
settings) and fires on the first candle of the alert's own, finer timeframe that closes through
one -- once per line. On 22 Oct 2026 USDJPY's 1h support was first closed below by the 03:05
5m candle; the 1h break alert came at 04:02, 10 pips lower. The line is only used from the close
of the candle that published it, only complete coarse candles are fitted, and between their
closes the line moves on in time. When the line's own candle is the one that breaks it, that
candle's `trendline` alert is the signal and no early one is sent after it. Measured 12-22 Oct
(per market per day): 1h lines ~0.3 early alerts, every one confirmed by the 1h close; 15m lines
0.3-0.4, mostly NOT confirmed; 5m lines on 1m closes 1.2-1.7, 36-56% confirmed. The subject
ranks an early break by its LINE: 1h / 4h lines P2, finer ones P3 -- a step below the confirmed
break.

**Which line, and when it breaks (trendlines).** One support and one resistance line are live at
a time. `prefer_newer=True` lets a new line with as many touches take the place of the live one
when its last touch is more recent (default: the live one stays). `hold=N` makes a break wait
for N more closes in a row; a close back across meanwhile keeps the line alive and the count
restarts. That is different from `closes=2` below, which counts from the first close through
-- and at that first close the line is retired, so a break that wobbles back once is lost
altogether. On Gold 1m both of the user's lines (21 and 22 Oct) were lost with `closes=2` and
caught with `hold=1`. Measured on 1m (12-22 Oct, three markets): single-close breaks were closed
back across within 10 minutes ~70% of the time; with `hold=1`, 42% are still beyond the line 10
minutes later (31% before), at about the same number of alerts. The trendline config sets both
on its 1m alerts.

**Breaking a level by a rule.** Trendline and S/R-zone breaks take a rule. It is counted from the
first close through, over consecutive candles, against the level as it stood there: a trendline is
carried along its own slope, a zone is held flat. A close back across the level ends that break.

| setting | meaning |
|---|---|
| `closes=N` | N closes in a row beyond the level (`1`, the default, = the first close through) |
| `full=N` | N whole candles in a row beyond it, wicks included |
| `colour=N` | N candles in a row of the break's colour: red for a break down, green for up |
| `confirm=N` | (trendlines) then, within N candles, one of the break's colour must close beyond the whole candle that completed the rule. This is notebook 21's confirmation. |

Every setting you give must hold. The alert fires **once**, on the candle that completes the rule,
and the time in the email is that candle's close. So "alert me when 2 candles close below the 5m
trendline" is:

```python
dict(on=True, name="5m trendline: 2 closes beyond", kind="trendline",
     event="break", closes=2, markets="all", tfs="5m", priority="urgent"),
```

Use `full=2` instead when "below" should mean whole candles below, wicks included. A `closes` rule
means exactly what notebook 21's `tl_close_n` means, and a test holds the two equal. `full` and
`colour` deliberately differ from the research's `tl_full_m` / `tl_dir_n`, which count the break
candle itself. In the research version, "2 whole candles below" fired only when the very candle
that crossed the line was already wholly below it: 0.02 a day on BTC 5m, against 0.8 counted this
way.

**Which shapes.** Chart patterns take the same choice: `fit="strict"` fits each boundary through
two real swing extremes and keeps the shape only if no candle closed through either boundary
before it formed, and `touch_tol` sets how close a third swing must be to count (0 = each
pattern's own, 0.6-1.0 ATR). The default checks swing points only: a Gold 15m falling wedge (20 Oct
2026) had a top line price had closed above. Wedges and triangles on Gold + USDJPY + BTC, Mar-Aug
2026, per day: default 3.7, strict 1.9, strict with `touch_tol=0.5` 0.7.

**Which lines.** `fit="strict"` (with `touch_tol=0.3`) only accepts a line that runs exactly
through two real swing extremes with no candle closing through it before it is drawn; the default
fit checks the swings alone, so a wick can cross a line by two ATR (docs/CHART_GUIDE.md,
"Trendlines"). Strict roughly halves the lines and the alerts: on Gold + USDJPY + BTC, Mar–Aug
2026, "formed" on 1h/4h went from ~0.5 to ~0.3 a day. **`alerts/alerts_config_trendlines.py`** is
a ready-made config built on it: strict trendlines forming and breaking on 1m-4h (1m a little
looser, with `prefer_newer` and `hold=1`), early breaks of 15m-4h lines on 5m closes and of 5m
lines on 1m closes, wedges and triangles forming and breaking out on 15m-4h, high-impact news
reminders from a day ahead down to 15 minutes, and at most one email every 5 minutes with tiered
subjects and pictures, with the steps to preview it and switch to it at the top of the file.

**Rules that keep it honest:**
- **A setting a kind does not have is refused by name.** `perod=14` on an RSI alert is an error,
  not a silent default.
- **Every kind is causal.** It sees only closed candles, and the tests cut the data on the very
  candle each signal fires on, checking the signal is already there.
- **Switching an alert off and on again starts it afresh.** It does not mail you everything it
  missed while off. A PC that was switched off is different: that gap is caught up, under
  "While offline".
- **Mistakes keep the last good config.** Any mistake (an unknown market, timeframe, pattern,
  kind or setting) is refused with a message that names it. The monitor keeps running on the
  last good config and logs the error, so a typo cannot silently switch your alerts off.

**Markets.**
- `source="binance"`: any Binance spot pair.
- `source="dukascopy"`: 1,380 instruments, checked when the config loads. You can name them by:
  - a short name: `XAUUSD`, `WTI`, `US500`, `GER40`, `NAS100`, `DXY` …;
  - any 6-letter FX pair;
  - Dukascopy's own code, such as `COCOA.CMD/USD` or `AAPL.US/USD`.

**Heavier alerts need more history.** Each alert needs some history before its first candle, so
the monitor keeps enough 5m bars for the heaviest alert that is ON. The three alerts on by default
need 19 days. An EMA-200 on 4h needs about 100 days, and switching on everything in the file needs
147. That costs one larger download on the first run, then nothing extra.

## 5. When an email is sent

- **A bar is judged once it is final**, never while it is still forming. A bar is final when it
  has been closed for `settle_seconds` (90; waiting the same for every market puts one bar close's
  alerts in **one** email) -- or `instant_settle_seconds` (10) on a feed an instant alert reads.
  Its data must also reach its close, or the feed must still have nothing 20 minutes after the
  close. The second case is a shut market, for example Friday's last 4h bar. Measured 23 Oct 2026:
  both feeds hand over a 1m or 5m candle within seconds of its close, and every one of 36 asked
  for 0-8 s after the close was already final (Binance in 0.3 s; Dukascopy had it on calls started
  1 s after the close). The "Dukascopy is ~1 minute late" behind the 90 s was the fetch, not the
  data: each request loaded the certificate file again (§7).
- **Urgent** alerts are emailed on the tick that finds them. **Normal** ones wait for the digest:
  at most one email every `digest_minutes`. **Any email carries every pending alert**, so an urgent
  email also clears the waiting digest.
- **Or on a clock: `DELIVERY = dict(every_minutes=15)`.** Emails go only at the quarter hours
  (sent at :02 :17 :32 :47 UTC, once that boundary's bars are final), at most one each, carrying
  everything found since -- urgent alerts and news included; urgent no longer means "at once".
  A timeframe finer than the clock (1m, 5m) is fetched and judged only at the boundary, so a 1m
  signal waits for the quarter hour it falls in. A news reminder rides in the email of the quarter
  hour at or BEFORE its time (a 15-minute reminder for 08:30 arrives at 08:17), never after the
  release. A signal found after its boundary's email has gone (a late feed) waits for the next one.
- **`priority="instant"` skips the clock.** Its feed is fetched at every close of its timeframe
  (a 1m alert: every minute), judged `instant_settle_seconds` (10) after the close, and the signal
  is emailed on that tick -- with the monitor ticking at :20, ~20-30 s after the candle closed --
  carrying whatever else was waiting. Every alert on that feed is judged on the same fetch (one
  fetch serves them all), but the others still wait for their boundary. An instant email does not
  use up the boundary's, and it goes BEFORE a news result poll that may take its 25 s budget. A
  signal found late (after an outage) is not instant. Measured on Gold, USDJPY and BTC, 20-23 Oct
  2026 with every trendline break and early break instant: ~52 emails a day, the same as with
  everything on the 5-minute clock, because an instant email takes along what was waiting.
- **`collapse_repeats=True`** makes a signal that repeats inside one window ONE row: same alert,
  market, timeframe and signal, with "5 times, 10:01-10:13 UTC" in its note.
- **Section order** follows `CATEGORIES`; put `"News"` last there and the news comes after the
  signals instead of first.
- **`subject="tiers"`** leads the subject with the email's most important row and its tier, so
  nothing important hides behind "12 signals": 🔴 **P1** act now (critical news, or any high US
  news, within the hour and its result; a trendline break or pattern breakout on 1h/4h), 🟠 **P2**
  look soon (other high news within the hour, critical/US news 2-6 h out, a moved release, a 15m
  break, a 1h/4h formation), ⚪ **P3** for the record (1m/5m, smaller formations, far reminders).
  For example `[Alerts] 🔴 P1 · USD CPI m/m in 15 min · BTC 1h broke support line · +4 more ·
  13:17 UTC`. A Gmail filter on `subject:"P1"` can star it and mark it important; one on
  `subject:"P3"` can skip the inbox. Waits read as people say them: `in 1 day`, `in 12 h`,
  `in 2 h 30 min`, `in 15 min`.
- **One email covers several markets.** The subject reads like
  `[Alerts] ⚡ 3 signals (1 urgent) · Gold, BTC · 14:01 UTC`. The body has a section per category,
  urgent rows first, with a TradingView link per row.
- **No history on switch-on.** The first time an alert sees a market and timeframe, it starts from
  that moment. Adding an alert, or switching one back on, does not mail you past matches.
- **A signal may carry a note**, such as a chart pattern's target and stop, the trendline's level
  and touches, the gap's range or the RSI value. It is shown under the signal in the email.
- **`charts=N` puts pictures in the email**: the top N signal rows, ranked like the subject, drawn
  at the top of the email -- the candles, THE line or shape that fired with its touches, the level it
  breaks and its target, a star on the candle it fired on. Sent as inline attachments (Gmail does
  not show images embedded in the HTML), ~40 KB each. Measured for N: in a 48 h test 37 of 51
  emails had one signal row, 13 two and 1 three, so `charts=3` drew every row of every email; a
  picture costs ~0.1 s to rebuild the alert and ~0.3 s to draw on a desktop (x3-4 on an e2-micro),
  and the monitor's memory grew 8 MB with charts on and none after 45 more. It needs matplotlib
  (in requirements-alerts.txt); without it, or if a picture fails, or past `chart_budget_seconds`
  (20), the email goes without that picture and `status` / the log say why. `charts=0` (the
  default) is the email it always was.
- **Every row carries a ref**, like `btc.1m.202610191349.trendline-broken` (market, timeframe, the
  candle's close in UTC, the alert's name). On your PC, `python -m quant.alerts show <ref>` fetches
  that market's candles again, re-runs the alert on them exactly as the monitor judged them (on a
  clock, from the start of that boundary's window), says whether it fires there again, and
  opens a chart in `reports/alerts/<ref>.html`: the candles around it, THE line or shape that fired
  (with its touches or pivots, the level it would break and the measured target), the candle it
  fired on, and what came after (`--after 60` candles). The alert is found by name in
  `alerts/alerts_config.py` or any `alerts_config_*.py` beside it.

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
  finest timeframe that is 4 polls an hour, not 60 (an instant 1m alert: 60). Each poll asks only
  for the new bars: ~0.2-0.3 s from either provider, run in parallel. A shut market (a weekend)
  backs off to about 4 polls an hour. The feeds keep their connections open and verify them with
  ONE certificate store per process (`feeds.session()`): `requests` hands every new connection the
  CA file, and OpenSSL 3 (Python 3.13 here, Ubuntu 24.04 on the server) takes ~4 s of CPU to parse
  it -- twice per Dukascopy fetch, since the library opens a connection per page. That was most of
  a Dukascopy poll's 11-33 s, and a 1m poll every minute would have spent a quarter of the
  server's CPU on it. Now a poll costs ~0.02 s of CPU after a one-time ~4 s load; a wrong host, a
  self-signed, an expired and an untrusted certificate are still refused. The log line of every
  tick ends with how long it took.
- **Detecting.** Each alert is evaluated only on its new bars plus its warm-up, never on the whole
  history. Warm-ups run from 60 candles (candle patterns) to about 600 (an EMA-200). Most kinds
  take milliseconds; chart patterns take a few hundred. Each process pays a one-time compile of
  ~1–2 s.
- **Storage.** As many days of 5m bars per market as the heaviest alert that is on needs: 19 for
  the default three, up to ~150 with everything on. That is a few hundred KB to a few MB.
- **First run, or `check`.** A 19-day backfill of three markets takes about 20 s, and a 147-day one
  takes longer.

## 8. Adding a new kind of alert

A kind is a `Detector` in `quant/alerts/detectors.py`. It gets one timeframe's **final** candles
(columns `open_time t open high low close volume`) and returns the rows it fires on:

```python
class MyDetector(Detector):
    kind, category = "my_kind", "Indicators"
    summary = "what it fires on, for the catalogue"
    params = {"period": (14, "what it means")}        # every setting it takes, with its default
    example = dict(period=14)                         # shown in the catalogue
    def check(self, p): ...                           # optional: raise ValueError naming the problem
    def warmup_bars(self, params): return 200         # history needed before the bar it judges
    def evaluate(self, frame, params, tf):
        return [dict(i=row, what="short phrase", side="bull", note="optional"), ...]

REGISTRY = {d.kind: d for d in (..., MyDetector())}
```

The engine needs no change. Unknown settings are refused for you, and the catalogue lists the new
kind once `python -m quant.alerts list --write-docs` regenerates it (a test fails until it does).
The detector **must be causal**: on the candles up to bar i it must say exactly what it says about
them on all the candles. `test_no_alert_kind_ever_repaints` checks that for every registered kind,
cutting the data at each candle that fires. Every indicator in `quant/` is built this way, so
wrapping one is safe.

Not built yet: **notebook 21's own setup** (a 5m entry on a confirmed break of a 15m/1h/4h
trendline). It needs several timeframes' lines at once, which a single-timeframe detector does not
see. That is also why `mtf_trendline` is left out of the setups.

## 9. Files and commands

```
alerts/alerts_config.py        WHAT to watch -- yours
alerts/state/                  git-ignored: alerts.db (SQLite), bars/*.parquet, alerts.log,
                               alerts.lock, preview.html
quant/alerts/config.py         reads and checks the config (fresh each tick)
            feeds.py           closed 5m bars from Binance / Dukascopy, with timeouts
            detectors.py       the 16 alert kinds (REGISTRY); add more here
            catalogue.py       the menu, built from the registries (`list`)
            engine.py          one resumable tick: fetch what is due, judge, queue, send
            store.py           SQLite state + atomic bar files
            compose.py         the email (text + HTML, categories in config order)
            notify.py          Gmail SMTP from .env; FileNotifier / ListNotifier
            runner.py          the loop, the lock, the log, `status`
            news.py            the NEWS block: reminders, previews, results (section 11)
scripts/install_alerts_task.ps1   the scheduled task on this PC (install / -Uninstall)
scripts/install_alerts_linux.sh   the systemd service on a Linux server (install, update, --uninstall)
requirements-alerts.txt        what a server needs: numpy pandas numba pyarrow scipy requests
                               dukascopy-python python-dotenv -- no plotting, no notebooks
tests/test_alerts.py           fake feed + fake clock: every rule above is a test
```

```bash
python -m quant.alerts list         # every instrument, timeframe, pattern, kind and setup
python -m quant.alerts list --write-docs   # regenerate docs/ALERTS_CATALOGUE.md
python -m quant.alerts check        # dry run over the last 48h -> alerts/state/preview.html
python -m quant.alerts test-email   # prove the .env settings
python -m quant.alerts tick         # one pass, then exit (refuses while the monitor runs)
python -m quant.alerts run          # the monitor loop (what the scheduled task starts)
python -m quant.alerts status
python -m quant.alerts news         # the covered calendar and when each reminder will go
```

## 10. Known limits

- **4h bars start at 00, 04, 08 … UTC**, the same as the research resampler. A broker chart that
  aligns 4h candles to the New York close draws different candles, so a 4h alert can differ from
  what that chart shows.
- **Dukascopy prices are the BID side**, as in the research data, so a candle can differ slightly
  from a mid-price chart.
- **A setup alert fires once per setup episode**, through the same one-entry rule the backtest
  uses, but it ignores the backtest's open-trade limit. It can fire while a trade from the
  previous signal would still have been open.
- **Indicators are computed on each alert's own window** (its warm-up plus the new candles). An
  EMA there matches one computed over all history to a fraction of a percent, not to the last
  digit.
- **Chart links:** Binance markets and Dukascopy FX/metals get a TradingView link. For other
  Dukascopy instruments, add `tv="EXCHANGE:SYMBOL"` to the market to get one.
- **On this PC it runs only while the PC is on and you are logged on.** It catches up afterwards,
  but cannot email you during the gap. On a server it runs all the time: see
  [ALERTS_CLOUD.md](ALERTS_CLOUD.md).
- **Binance refuses data to US servers and to many datacenter IP ranges** (HTTP 451). The feed
  then switches by itself to Binance's market-data host, `data-api.binance.vision`, which gives the
  same candles. Whether that host answers a given server is tested in step 7 of ALERTS_CLOUD.md.
  If both refuse, use Dukascopy's `BTC/USD`, which trades 24/7 and sits within ~0.03% of Binance's
  price. Its candle-pattern alerts do not always match Binance's chart: 60 of 108 hourly alerts
  matched over 20 days.

## 11. News

The `NEWS` block (§4 of `alerts/alerts_config.py`) adds notices driven by the economic calendar
rather than by candles:
- the week ahead, on Sunday evening;
- tomorrow's releases, every evening;
- reminders 60 and 15 minutes before critical releases, and 15 before high ones;
- the actual against the forecast once it is out;
- a release that moved.

Each notice goes once. A reminder is never sent after its release, and a preview found too late is
dropped. An outage or a broken calendar never stops a candle alert. A candle alert that closes
within 30 minutes of high news says so in its note.

```python
NEWS = dict(
    on=True,
    currencies={"USD": "medium", "EUR": "high", "GBP": "high", "JPY": "high"},
    timezone="Asia/Karachi",
    week_ahead="Sun 20:00", day_ahead="20:00",
    remind_before={"critical": (60, 15), "high": (15,)},
    results="high", warn_alerts=30,
)
```

- **Volume:** about 24 emails a week as shipped, measured on 2024's calendar.
- **Reminders can go hours ahead.** Each level takes any minutes from 1 to 2880 (two days), so
  `remind_before={"critical": (1440, 60, 15)}` adds a reminder 24 hours before every critical
  release. Same-minute releases still share one email.
- **Results** for US releases need `FRED_API_KEY` in `.env`. CPI, PPI, payrolls, unemployment and
  hourly earnings come from the BLS about a minute after 8:30. Retail sales, PCE, GDP, claims,
  durable goods, JOLTS and the regional surveys come when FRED posts them, usually within the
  hour, and the Fed's rate from its statement at 14:00. Results for other currencies need
  JBlanked's paid credits. Without either key the rest still works.
- **A late result goes only if it was awaited.** A number is asked for for 3 hours. One that lands
  more than an hour after its release is mailed only if the monitor was already waiting for it,
  never after the PC was off.
- **Where the calendar lives:** the monitor keeps its own small copy in `alerts/state/news/`, and
  refreshes Forex Factory's weekly feed every 30 minutes.
- **Dry run:** `python -m quant.alerts news` shows what is covered and when each reminder will go.

The sources, criticality levels and sentiment rules are explained in [NEWS.md](NEWS.md).

