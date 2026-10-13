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
Run this whenever you like. It fetches this week's calendar, plus FRED and JBlanked when their
keys are set.

```bash
python -m quant.news show --days 7
```
This prints the next 7 days in Pakistan time, filtered by the default coverage.

The two optional keys go in `.env` (git-ignored), and you create them yourself:

| key | where | what it adds |
|---|---|---|
| `FRED_API_KEY` | fredaccount.stlouisfed.org → API Keys | official US release dates, past and scheduled |
| `JBLANKED_API_KEY` | jblanked.com → profile → API key | **actual** values as they come out, plus Apr 2025 → now history (`python -m quant.news backfill-jb`) |

Everything works without them. Without JBlanked there are no result emails and no history after
April 2025. Without FRED, US dates come from Forex Factory only.

## Where the news comes from, and how far each source is trusted

| source | gives | reliability |
|---|---|---|
| Forex Factory weekly feed | this week, every currency: exact time, impact, forecast, previous | **exact**: the time carries its own offset. No actuals. |
| Forex Factory archive (scraped by a third party, research only) | 2007-01 → 2025-04: impact, actual, forecast, previous | **dates right, clock repaired** (below) |
| JBlanked API (Forex Factory data) | actuals, by date range | its timestamps have no zone (stored raw). See *JBlanked's clock* below. |
| FRED (St. Louis Fed) | official US release dates | official; dates only, so the time is the official one |

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

**The merge** keys a release on currency + title + its day in the home zone. Each field is then
taken from the most trusted source that has it:

| field | taken from |
|---|---|
| time | the best `time_q` first |
| forecast | the feed's last observation **before** the release (a forecast revised afterwards is not what the market expected) |
| actual | JBlanked, then the archive |

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
| results | the actual vs the forecast, the surprise and its meaning, within ~5 min (JBlanked) | urgent |
| moved / removed | a covered high+ release that changed time or left the calendar | urgent if critical |
| calendar stale | the feed has not updated for 6 h; once a day | normal |
| on candle alerts | an alert closing within 30 min of high news says "news: USD CPI m/m in 12 min" | — |

How it behaves:
- **Each notice goes once**, even across a restart. Its id is a hash of what it is about.
- **A reminder is never sent after its release.** After an outage, only the latest one still
  ahead goes out ("in 3 min"). The others are recorded as skipped.
- **Stale digests are dropped.** A preview found more than 6 h late (12 h for the week) is not
  sent at all.
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
  actual and its meaning.
- **A dropdown** under the chart, bottom right: `News: critical | high+ | medium+ | all | off`.
- **Legend:** each level is also its own legend entry.

**Rewind** keeps the schedule visible (it was known in advance), but hides each actual until the
moment it came out.
