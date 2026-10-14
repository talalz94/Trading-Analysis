# News: the economic calendar

`quant.news` knows **what is released when, how much it matters, what was expected and what came
out**. It feeds three things:

- **research:** news-day filters and features on any bar grid, plus the news-day experiment;
- **alerts:** reminders, previews and results by email;
- **charts:** a TradingView-style strip of releases along the bottom.

## Getting it running

```bash
python -m quant.news import-archive
```
This is needed once, and only for research. It downloads 2007 – April 2025 (68 MB), keeps a
repaired 1.4 MB copy in `data/news/`, and deletes the download.

```bash
python -m quant.news update
```
Run this whenever you like. It fetches this week's calendar, plus FRED, the US actuals and
JBlanked when their keys are set.

```bash
python -m quant.news official --validate
```
Once, with the FRED key set: the **US actuals since 2022** from the agencies that publish them
(about a minute, ~1,350 releases, 23 KB), compared title by title with the archive's numbers.
After that it keeps itself current (`update` and the alerts monitor fetch only what is new).

```bash
python -m quant.news show --days 7
```
This prints the next 7 days in Pakistan time, filtered by the default coverage.

The two optional keys go in `.env` (git-ignored), and you create them yourself:

| key | where | what it adds |
|---|---|---|
| `FRED_API_KEY` (pick API **version 1**) | fredaccount.stlouisfed.org → API Keys | official US headline dates, past and scheduled, **and the US actuals** (below) |
| `BLS_API_KEY` (optional) | data.bls.gov/registrationEngine | lifts the Bureau of Labor Statistics' limit from 25 to 500 requests a day; not needed |
| `JBLANKED_API_KEY` | jblanked.com → profile → API key | actuals for **other currencies**, but only with **paid credits** |

Everything works without them. Without FRED, US dates after April 2025 come only from this week's
feed, and there are no US actuals after April 2025.

**JBlanked needs paid credits.** Checked 2026-10-13 with a free key: its `range`, `week` and
`today` calendar endpoints all answer HTTP 401, "This endpoint requires credits". The US actuals
no longer depend on it (they come from the agencies); without credits there are simply no actuals
for EUR, GBP, JPY and the rest after April 2025. The monitor notices the refusal and leaves the key
alone for a day rather than asking every 5 minutes, and `python -m quant.alerts news` says why.

## Where the news comes from, and how far each source is trusted

| source | gives | reliability |
|---|---|---|
| Forex Factory weekly feed | this week, every currency: exact time, impact, forecast, previous | **exact**: the time carries its own offset. No actuals. |
| Forex Factory archive (scraped by a third party, research only) | 2007-01 → 2025-04: impact, actual, forecast, previous | **dates right, clock repaired** (below) |
| JBlanked API (Forex Factory data) | actuals, by date range | its timestamps have no zone (stored raw). See *JBlanked's clock* below. |
| FRED (St. Louis Fed) | official US **headline** dates for CPI, PPI, payrolls, retail sales, PCE, GDP | official; dates only, so the time is the official one. See *FRED's calendar* below |
| The Fed's own calendar (federalreserve.gov) | every FOMC meeting 2021 → 2027, no key | official: decision 14:00, press conference 14:30, projections at the marked meetings |
| **The US agencies** (BLS, FRED's archive of past vintages, the Fed's statements) | **US actuals** for 21 titles, 2022 → now, and live at the release | official: they ARE the number every calendar reprints. Identical to the archive on every release checked (below) |

**The Forex Factory website is not scraped.** Its calendar and news pages sit behind a Cloudflare
bot check: a plain request gets "403 Just a moment…". The weekly feed is Forex Factory's own
export, and it is open.

**The archive's clock, repaired.** Two faults were measured, and both are fixed in
`quant/news/sources.py`:

1. **It was scraped in Tehran time, with Iran's abolished daylight saving still applied.**
   - From 2023 every stamp was an hour early: claims showed 7:30, ISM 9:00, the Fed 13:00.
   - Re-read in the true zone, they land exactly on the official times. Claims are 8:30 every
     year, ISM 10:00, the Fed 14:15 until 2012 then 14:00, and the ECB moved from 13:45 to 14:15
     in 2022.
2. **30% of rows have no time.**
   - Forex Factory prints a time only on the first row of a group of releases, and the scrape was
     re-sorted afterwards.
   - Their **dates** are right: all 24 CPI dates of 2023–24 were checked.
   - Their time comes from the official US schedule (CPI, payrolls, PPI, retail sales, GDP and
     PCE at 8:30 New York), or else from the same event's nearest timed release in its home zone.
   - The `time_q` column says which: `exact` / `official` / `typical` / `date` / `allday`.

**JBlanked's clock.** Its client's offsets (GMT = 3, EST = 7) point to MetaTrader broker time,
i.e. New York + 7 hours, which follows US daylight saving. That is a guess from constants, so
nothing assumes it:
- **Stored raw.** Rows are kept exactly as sent.
- **Zone decided at every build.** The evidence is the feed's exact times, the archive's exact
  times, and the **official US release times** on JBlanked's own dates, which are ground truth in
  any season.
- **No answer while a tie remains.** For half the year, broker time, Athens, Moscow and fixed
  GMT+3 all read the same. Broker time and Athens differ only in the weeks when US and EU clocks
  change on different Sundays. So until the data contains such a week, no zone is chosen. The
  rows then keep only their date: they lend their **numbers** to the matching release, and its
  time comes from the feed or the official schedule.
- **Units.** JBlanked's bare numbers are put into Forex Factory's printed units by the ratio of
  their forecasts. "254" or "254000" beside "140K" both become "254K".
- **Status.** `python -m quant.news status` shows the zone chosen and why (`jb_zone_evidence`).

**FRED's calendar.** FRED lists every day its data for a release *changed*, not only the headline
publication. That adds about one false date a year per release: CPI's February seasonal-factor
revisions, retail sales' April annual revision, payrolls' August benchmark preview. Its "FOMC Press
Release" changes almost every business day (3,894 dates since 2014).
- **The fix:** each headline day is taken from the series' **first print**
  (`series/observations`, `output_type=4`): CPIAUCSL, PPIFIS/PPIFGS, PAYEMS, RSAFS, PCEPI and GDP.
  GDP's second and third estimates come from its next two vintages. Future days come from FRED's
  schedule, one per month.
- **Checked against the archive, 2022 – April 2025:** identical dates for every release, no extras
  and none missing. The 2025 shutdown comes out right: September's CPI on 24 Oct, no October CPI,
  November's on 18 Dec, payrolls on 20 Nov and 16 Dec.
- **Fed decisions:** these come from the Fed's own page instead. Its 26 decisions in 2022–Apr 2025
  match the archive exactly.

**The US actuals, from the agencies** (`quant/news/official.py`). These are permitted, free,
official sources; nothing is scraped from behind a bot check.

| titles | history | live, at the release |
|---|---|---|
| CPI m/m and y/y, core CPI m/m and y/y, PPI and core PPI m/m, payrolls, unemployment rate, average hourly earnings | FRED | the BLS API, seconds after 8:30 (no key needed) |
| retail sales and core retail, core PCE, personal spending, Advance / Prelim / Final GDP, claims, durable goods and core, JOLTS, Philly Fed, Empire State | FRED | FRED, when it posts (usually within the hour) |
| Federal Funds Rate | FRED's target range, the day after each decision | the Fed's statement page at 14:00 |

- **Read as published that day, never from today's revised data.** Payrolls in July 2024 printed
  114K: 158,723 minus June *as revised in that same report* (158,609). From today's history the
  same subtraction gives 85K. FRED keeps every past version of a series (ALFRED), and one request
  per series returns all the versions needed.
- **Printed the way Forex Factory prints them:** rounded half-up from the published levels, in its
  units ("0.4%", "206K", "7.60M", "4.50%"), never "-0.0%".
- **Checked against the archive, 2022 – April 2025: 900 of 901 releases identical**, every title
  (claims 170 of 170, payrolls 40 of 40, CPI 39 of 39, all three GDP stages 13 of 13 …). The one
  difference in the whole check is the Fed of 26 January 2022, which Forex Factory printed as
  "<0.25%" and the Fed's range gives as 0.25%. `python -m quant.news official --validate` prints
  the table.
- **And after the archive ends:** the week of 12 Oct 2026, Forex Factory's own feed listed as
  "previous" exactly the agencies' last number for all 11 covered US releases (CPI ×4, PPI ×2,
  retail ×2, claims, Philly, Empire). Note it shows retail sales at 1.2%, the number AS RELEASED;
  today's revised data says 1.1%, which is why history is read from the vintages.
- **Edge cases, each from a real release:**
  - two months published on one day (the 2025 shutdown catch-up: JOLTS for September and October
    on 9 December) print the later month, as every calendar did;
  - GDP's second estimate sometimes repeats the first (27 Nov 2024, 27 Feb 2025), and FRED then
    records nothing that day, so the three stages are read on the release days from FRED's
    calendar instead of counted from FRED's revisions;
  - FRED sometimes posts days late (the Philly Fed of 17 Nov 2022 is stamped Sunday the 27th): a
    weekend posting keeps only its date and is tied back to the release up to 14 days earlier;
  - December's y/y comes out in January, so the live BLS request reaches two years back.
- **A live number counts only when it is new**, from either source: its month must be LATER than
  the last one published, and from FRED the series must also list a new version dated the release
  day (`series/vintagedates`). FRED's values themselves cannot say when they were published: by
  default every one carries TODAY's date.
- **Tested live on the CPI of 14 Oct 2026**, and the first version failed. The BLS API answered
  HTTP 503 at 8:30, the collector fell back to FRED, and the old check (that "today" date) passed
  AUGUST's numbers off as September's: 0.4% for 0.6%, the meaning reversed on three of four lines.
  The rule above replaced it, a test now replays that morning, and the same release through the
  real servers now gives 0.6% / 0.3% / 3.7% / 2.6% from the BLS and from FRED alike. The
  reminders (11:30, 12:15 UTC) and the day-ahead (15:00) went to the second.
- **Cost:** the whole history since 2022 takes about a minute (four series fetched at once; one
  shared pacer keeps every request inside FRED's 120 a minute). The alerts monitor keeps only 150
  days: about 35 s on its first run, which it does AFTER that tick's candle emails are out. After
  that, every 6 hours only the series whose release the calendar shows since the last stored one
  (usually none), and every series once a day. At a release, one poll every 45 s for up to 3 h, at
  most 25 s of requests per poll (the titles asked at once), run AFTER the tick's candle emails
  are out, BLS capped at 20 a day without its key (it allows 25). A source
  that fails is not asked again for 30 minutes (the feed: 5). Plain HTTP, no browser, nothing new
  to install: the cloud server runs it as it is.
- **Limits:** US only. ISM, consumer confidence, the Michigan surveys and similar private releases
  have no free official feed, so they get no actual. Other currencies need JBlanked's credits.
- **Forecasts are not an agency number.** They come only from Forex Factory: the archive to April
  2025 and the weekly feed from now on (every fetch is logged). Releases between May 2025 and the
  start of the log have their actual but no forecast, so no surprise or sentiment.

**The merge** keys a release on currency + title + its day in the home zone. Each field is then
taken from the most trusted source that has it:

| field | taken from |
|---|---|
| time | the best `time_q` first |
| forecast | the feed's last observation **before** the release (a forecast revised afterwards is not what the market expected) |
| actual | the agencies, then JBlanked, then the archive |
| impact | the feed, then the archive. A release no source rates on its own day (claims after April 2025, known only from the agencies) takes the rating Forex Factory **last** gave the same title, so it is not left at level "none", which no chart or alert shows |

An event that drops out of its week's latest feed was moved or cancelled, and leaves the calendar.

## What each release carries

| column | meaning |
|---|---|
| `level` | criticality: `low` · `medium` · `high` (Forex Factory's impact) · **`critical`**. Critical is an upgrade of high only, for the releases that move everything: US CPI, payrolls, the Fed, PCE, PPI, retail sales, advance GDP; and the big central-bank decisions elsewhere. The full list is in `python -m quant.alerts list` §7. |
| `actual` / `forecast` / `previous` | as printed ("0.3%", "245K"); `_v` columns hold the numbers |
| `surprise` | actual − forecast |
| `surprise_z` | that surprise in units of the same release's **past** surprises (root mean square of the earlier ones, at least 8). It is causal: a z-score never knows a later release. |
| `direction` | +1 if a higher number is good for the currency, −1 if lower is (unemployment, claims, inventories …). These are Forex Factory's own "usual effect" rules, and they agree on 376 of 376 events. |
| `sentiment` | `better` / `worse` / `as expected` **for the currency**. "Hot CPI" is good for the dollar and often bad for Nasdaq and gold, so nothing here claims what it means for a market. |

## In research

```python
from quant.news import load_events, news_features, news_blackout
ev = load_events("2024-01-01", "2025-01-01", coverage={"USD": "high"})
f = news_features(df, ev)        # news_next_min, news_prev_min, news_day, news_last_surprise_z
mask = news_blackout(df, ev, before=30, after=30)    # True = within 30 min of covered news
```

**In a strategy file** (`notebooks/strategies/<name>.py`):
```python
NEWS_BLACKOUT = dict(before=30, after=30, coverage={"USD": "critical"})   # None = off
```
- **What it does:** no new trade from `before` minutes ahead of a covered release to `after` minutes
  past it, counted from each bar's close. Trades already open are left alone.
- **How it runs:** it is a re-trade setting, so a COMPARE arm costs one simulation.
- **When there is nothing to go on:** if the calendar covers nothing in the window, the lab warns
  rather than treat the window as news-free.
- **Why "critical":** those releases are covered the same way in every quarter. Lesser red releases
  are known only to April 2025.
- **Result:** measured on the trendline strategy, it touched 11 of 1,194 trades. Verdict in
  WHAT_WORKS.md.

**What is and is not lookahead:**
- **Not lookahead:** "CPI in 40 minutes". The schedule is published in advance.
- **Lookahead if used too early:** a release's actual. So `news_last_surprise_z` takes a release
  only from the first bar that **closes after** it.
- **One soft spot:** unscheduled items, such as a speech added that morning. The calendar does not
  say when they were added. Keep to `high` and up for scheduled data.

**The news-day experiment** (`experiments/nq_news_reversion.py`, verdict in WHAT_WORKS.md) tests
JJ Simon's "A+ setup": revert to the pre-news price on 8:30 red-folder days. It loses, and is
negative even before costs.

## In the alerts

The `NEWS` block in `alerts/alerts_config.py` controls everything. See `python -m quant.alerts
news` for what it currently covers and when each reminder will go.

| notice | when | priority |
|---|---|---|
| week ahead | once a week (Sunday 20:00 your time); waits up to 11 h for Forex Factory to publish the new week | normal |
| day ahead | every evening (20:00); tomorrow's covered releases; no email when there are none | normal |
| reminders | 60 and 15 min before critical, 15 before high; same-minute releases share one | urgent |
| results | the actual vs the forecast, the surprise and its meaning: about a minute after 8:30 for BLS releases, when FRED posts for the rest, 14:00-ish for the Fed (needs the FRED key) | urgent |
| moved / removed | a covered high+ release that changed time or left the calendar | urgent if critical |
| calendar stale | the feed has not updated for 6 h; once a day | normal |
| on candle alerts | an alert closing within 30 min of high news says "news: USD CPI m/m in 12 min" | — |

How it behaves:
- **Each notice goes once**, even across a restart. Its id is a hash of what it is about.
- **A reminder is never sent after its release.** After an outage, only the latest one still
  ahead goes out ("in 3 min"). The others are recorded as skipped.
- **Stale digests are dropped.** A preview found more than 6 h late (12 h for the week) is not
  sent at all.
- **A result is fresh or it is not sent.** A number is asked for for 3 hours, because FRED can post
  an hour late. One that arrives more than 60 minutes after its release still goes, but only if
  the monitor was already waiting for it; the same number found after the PC was off is old news
  and is not mailed.
- **A release no source will ever answer** (ISM) does not hold up the rest of its group: CPI goes
  as soon as CPI is in.
- **News problems never stop candle alerts.** A failing or broken calendar only produces an
  error in the log.
- **Volume:** about 24 emails a week with the shipped settings, measured on 2024's calendar.

**Default coverage:**
- USD from medium up;
- EUR, GBP and JPY from high up;
- also available: AUD, CAD, CHF, NZD, CNY, or `"*"` for every currency not named.

## On the charts

```python
pattern_figure(df, ..., news=load_events(start, end, coverage={"USD": "medium"}),
               news_show="high", news_tz="Asia/Karachi")
```

In the lab notebook, set `NEWS` and `NEWS_SHOW` in `notebooks/lab_config.py`. The strip shows:
- **Markers** pinned to the bottom edge, coloured and sized by level.
- **A dotted line** at each critical and high release.
- **Hover:** the time in your zone and in New York, with forecast, previous and, once out, the
  actual and its meaning. US actuals are there from 2022 to the latest release.
- **A dropdown** under the chart, bottom right: `News: critical | high+ | medium+ | all | off`.
- **Legend:** each level is also its own legend entry.

**Rewind** keeps the schedule visible (it was known in advance), but hides each actual until the
moment it came out.
