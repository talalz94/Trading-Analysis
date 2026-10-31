# Indicators for the Exness web terminal

The research code's chart tools, ported to **Indie**, the language of the Exness web terminal's
indicator editor. Each file in `exness/` is one indicator. Every one is tested candle by candle
against the research code it copies, on a stand-in for the Exness runtime (`tests/indie_standin.py`).
Indie can't run on this PC, so the tests prove the **logic**. Pasting a file into the terminal is
the test of the **syntax**.

## Installing one

1. Exness web terminal → **Indicators** → the code editor (**Create**).
2. Paste the whole file and press **Save**.
3. Press **Add to chart**. Its settings open from the indicator's name on the chart.

If Save or Add to chart shows an error, copy the message back. It names the line and the cause.

## Which file does what

| File | Shows | Needs other timeframes |
|---|---|---|
| `fair_value_gaps.py` | fair value gaps as boxes, filled and expired ones removed | no |
| `trendlines.py` | the email alerts' strict trendlines, chart timeframe only | no |
| `probe_timeframes.py` | **paste first**: a report on what Exness lets an indicator read | tests it |
| `trendlines_mtf.py` | trendlines of the chart's timeframe **and** 15m, 1h and 4h, with early-break marks | yes |
| `structure.py` | HH / HL / LH / LL swing labels for the chart's timeframe and 15m, 1h and 4h | yes |
| `bias.py` | a panel with each timeframe's bias (1m 5m 15m 1h 4h 1D) | yes |
| `levels.py` | previous day / week / month high and low; Sydney, Tokyo, London and New York session highs and lows | no |

## Paste the probe first

`probe_timeframes.py` (v2) writes a short report in the top-left corner. Put it on a 1m or 5m chart,
and for the last line, leave it running past the next 1h close. Per timeframe it gives:

- **candles and days** of that timeframe the indicator was given;
- **% on round UTC times**: whether 4h candles open at 00, 04, 08 ... UTC;
- **early: history N, live M**:
  - **history** counts chart candles on history that were handed a candle that had not closed.
    It must be 0.
  - **live** counts the same thing on live candles. It is expected (see below), and the
    indicators ignore it.
- **forming live**: live candles shown a forming candle;
- **twice ... (changed ...)**: candles computed a second time, and with different values;
- **wrong**: values that came back wrong or out of order (must be 0).

The verdict lines:
- **OK** (nothing early on history) or **PROBLEM**.
- After a live close, either **Recomputed cleanly** or **NOTE: a candle was computed twice**.

**What the first probe showed on Exness** (US Tech 100, 1m, 27 Oct 2026):

- **Everything works:** the secondary contexts, `param_ref`, 16 values with none wrong, and the
  text labels.
- **About 1,000 candles of each timeframe,** whatever the chart holds: 15m 15 days, 1h 62 days,
  4h 228 days. The chart itself had 1.2 days of 1m. The alerts fit 4h lines on 505 4h candles, so
  the 1h and 4h lines have the history they need.
- **100% on round UTC times.**
- **Exactly one early view per timeframe, at the switch to live.** When the chart loads, Exness
  computes each timeframe's still-forming candle as if it had closed, and hands it over as the
  freshest data, as Indie's docs describe. The indicators refuse it, because its candle has not
  closed by the chart candle's close.

**The second probe** (US Tech 100, 1h chart, past a live 4h close, 27 Oct 2026) answered the last
question:
- **early: history 0** over 279 days of 4h, so the merge on history is clean.
- **twice 0:** the candle that was forming at load closed live and was **redone from scratch**,
  which is Indie's documented rollback.
- **wrong 0.**

Exness therefore behaves exactly as the indicators assume. Whether a candle is computed a second
time is the `twice` column. The indicators are
exact either way: each timeframe saves its state before a candle and, if the same candle comes
again, starts it from that state. Both endings are tested (`seam="redo"` and `seam="again"` in the
stand-in).

`Also try 1m below the chart` asks for a timeframe finer than the chart. That needs Indie 5.14, and
on an older Indie it fails with an error, which is itself the answer.

## How "safe" is guaranteed

Every multi-timeframe indicator follows the same two rules, so none of them can look ahead:

1. **Each timeframe's calculation runs on closed candles only.** A 1h line, swing or bias changes
   when a 1h candle closes, never while one forms.
2. **The chart takes a higher timeframe's state only once its candle has closed by the chart
   candle's close.** If the terminal ever hands over a candle early, it is ignored.
3. **At the very instant a candle closes, its state counts only if its close is the chart's
   close.** A 1h candle and the 5m candle ending with it close at the same price, because they are
   the same market at the same moment. Exness does not say whether, at that instant, the chart or
   the 1h timeframe updates first. Chart first, just after a load, it could be handed the version
   of that candle computed at load. That version fails this test, so the final one is taken a chart
   candle later. Tested on real candles: a 1h support broken by a candle that had not broken it 5
   minutes in, and a 15m candle whose first 5 minutes alone would confirm a swing it never makes.
4. **A candle computed twice counts once.** Each timeframe keeps a copy of its state from before
   the latest candle and starts again from it if the same candle comes back. Exness redoes it
   anyway, so this is insurance.

The tests run every indicator against two badly-behaved terminals as well as a correct one: one
that shows the forming higher-timeframe candle live, and one that shows the first candle before it
closes. Neither changes a single line, label or mark. Live candles, candles run twice, and candles
corrected after they closed also leave exactly the state of a clean run.

A third bad terminal hands every chart candle the final values of the higher-timeframe candle it
is in: merging by opening time, as Indie's `lookahead=True` does. Against it the promise is
**never ahead**, not "identical". A candle cut short by a market close (Friday's 20:00–24:00 4h
candle ends at 21:00) is never shown after it has closed, so its state is taken one candle late.
Indie's default merges by close, where this cannot happen, and the probe would show such a
terminal as `seen early` above 0.

Live, the terminal decides whether a higher timeframe's closed candle reaches the chart on the same
chart candle or the next one. A line can therefore appear one chart candle later live than after a
reload. No early mark is judged on a candle whose higher-timeframe state is late.

## trendlines_mtf.py: trendlines from several timeframes

- **The lines** are the email alerts' (`quant.indicators.add_trendlines`, strict), fitted on each
  timeframe's own candles with the alerts' own settings for it (`Settings: auto`). The fitting is
  `trendlines.py`'s own code, copied unchanged.
- **Colour** is the direction: red for resistance, green for support. **Thickness** is the
  timeframe: the chart's own lines are thinnest, 4h thickest. A higher timeframe's live line has a
  tag at its tip (`1h R`, `4h S`).
- **Broken** lines stay, faded. **Dropped** lines stay, grey and dotted (`Old lines kept` per
  timeframe).
- **Early marks** (`1h early down`) go on the first chart candle that **closes** through a live
  higher-timeframe line: a 5m close below a 1h support, up to an hour before the 1h candle can
  close there. This is the email alerts' early-break rule, once per line. On the USDJPY fixture
  from the alerts it marks the 03:05 5m close below the 1h support, as the alert did.
- **Where the line is drawn:** at every chart candle, the line passes through the price that
  candle's close is judged against. A candle closing beyond the drawn line is exactly what makes a
  mark. Between its own closes the line moves on in time, and across a shut market (a weekend) it
  holds, as in the alerts.
- Timeframes at or below the chart's are skipped (on a 1h chart, only 4h is added).
- **Higher-timeframe lines need history.** On 12 days of 1m Gold, the strict fitter finds 3 lines
  in 920 15m candles, none in 230 1h candles and none in 62 4h candles (BTC: 2, 0, 0). The alerts fit
  a 4h line on about 505 4h candles. So the 1h and 4h lines appear only if Exness gives the
  indicator months of those candles; the probe's report says how many it gets.
- **Cost:** on 14,000–18,000 1m candles, 5–6 s in plain Python on the stand-in (about the single
  timeframe port's cost), at most 4 drawing changes in any update.

## structure.py: HH / HL / LH / LL

- **The swings** are the research zigzag's (`add_zigzag_structure`), the same swings the
  trendlines use. A swing high is confirmed once price falls `Swing size` × ATR below the highest
  high since the last swing low, and the mirror for lows. A swing high above the previous one is HH,
  otherwise LH. A swing low above the previous one is HL, otherwise LL.
- **Only swings the chart holds** are labelled. Each timeframe arrives with ~1,000 candles of its
  own, mostly from before the chart's first candle; those swings are counted but not drawn.
- **When a label appears:** on the candle that **confirms** the swing (for a 1h swing, at the close
  of the 1h candle that confirms it). It is **drawn** at the swing itself (for a 1h swing, at the
  chart candle that made that 1h candle's high or low). So it sits where the swing was but appears
  later, as it does in the alerts. It never moves or disappears afterwards.
- **Telling them apart:** green for HH and HL, red for LH and LL. A higher timeframe's label names
  its timeframe (`1h HH`), is larger, and sits further from the candle, so labels at the same high
  stack instead of overlapping.

## bias.py: one line per timeframe

The rule is `quant.indicators.add_bias`, applied to each timeframe's own candles:

- **swings:** +1 after a higher high **and** a higher low, -1 after a lower high **and** a lower
  low, 0 otherwise;
- **EMAs:** +1 when close > EMA 50 > EMA 200, -1 when close < EMA 50 < EMA 200, 0 otherwise (and
  while the EMA 200 is still warming up);
- **bias = swings + EMAs:** +2 strongly bullish, +1 bullish, 0 neutral, -1 bearish, -2 strongly
  bearish. Each line says why, e.g. `1h  Bullish  (HH + HL, EMAs mixed)`.

**What it is not:** a signal. It describes what each timeframe looks like. Tested here as a trade
filter, an EMA bias and HH/HL structure added nothing (`docs/WHAT_WORKS.md`).

Timeframes below the chart's (1m on a 5m chart) need Indie 5.14 (`Also lower timeframes`). The
probe says whether you have it.

## levels.py: day, week, month and sessions

- **Previous day / week / month high and low**, drawn across the whole current period, as
  `quant.indicators.period_levels` gives them. The previous period is the last one **with
  candles**, so Monday's previous day is Friday on a market shut at weekends. A period the chart
  holds only part of is never used. `Day starts`:
  - **UTC midnight**: as Exness's own daily candles.
  - **New York midnight**.
  - **17:00 New York**: the FX day, so Sunday evening belongs to Monday.
- **Each period shows only on charts finer than the period**: on a daily chart the previous day
  is just the candle before, so it shows the week and month only.
- **Sessions** show on charts of 1h and less, whose candles fit inside the hours; a 4h candle
  straddles a session's open and close. Sydney 07–16, Tokyo 09–18, London 08–17 and New York
  08–17, each in **its own
  clock** with its own daylight saving, Monday to Friday local. Each session's high and low are
  drawn over the session (solid), then carried on (dotted) until that session opens again. While a
  session is open, its lines show the high and low **so far**.
- Indie has no time-zone library, so the daylight-saving rules are written out (US, UK/EU, Sydney).
  A test holds them to Python's zoneinfo hour by hour for 2024–2028, and holds every session window
  to `session_windows`.

## Exness's limits

Learnt on the FVG indicator, and the reason each file looks the way it does:

- **The Indie version is between 5.13 and 5.18.** It accepts `Rectangle` and `line_style` (5.13),
  and still has the "cannot erase" error that 5.19 removed. So tables (5.17) and reading
  timeframes below the chart (5.14) may be missing. No file uses tables, and lower timeframes are
  opt-in.
- **`TimeFrame.to_seconds` is missing.** Only `to_minutes` is used.
- **Named colours:** only those proven to exist are used (GREEN, RED and BLUE, plus NAVY and
  YELLOW from the docs). Every other colour is `rgba(...)`.
- **All state is kept in `Var`.** Exness rolls a candle back and runs it again whenever its value
  is updated, even after it closed. A plain list doesn't roll back.
- **At most 100 drawing changes per update.** Every file makes at most 90, erasures first. What
  doesn't fit waits for the next candle, and only the drawing waits, never the logic.
- **How far back a script may read its candles, learnt the hard way** (Gold 1m, 30 Oct 2026:
  `Series offset=510 is out of size=506`): a candle series grows to whatever depth is read while
  the chart loads, and once it is live, a deeper read is an error. A trendline checked live can
  reach further back than any did while loading, so both trendline indicators now keep their own
  copy of the last 2,000 candles and read only that. The deepest any line needed on six long
  histories (1m to 4h, 6,000 to 60,000 candles each) was 769 candles, and drawing one 979; a line
  older than 2,000 candles would not be drawn rather than fail. The stand-in enforces the rule, and
  a test goes live just before the deepest check of a run: the old code fails it with the same
  error.
- **Run time is not published.**

## Tests

`tests/test_exness_*.py`, on the stand-in in `tests/indie_standin.py`:

- **fvg:** add_fvgs' exact gaps.
- **trendlines:** add_trendlines' exact lines.
- **probe:** a good terminal reads ALL GOOD, and each bad behaviour is reported.
- **levels:** zoneinfo, session_windows, period_levels, session extremes.
- **structure:** add_zigzag_structure's labels on every timeframe, each on its confirming candle.
- **bias:** add_bias at every candle on every timeframe.
- **trendlines_mtf:** each timeframe's lines and breaks, the drawn lines, and early marks equal to
  the alerts' rule.

Each file's rules are mutation-checked: broken on purpose one at a time, and the tests must fail.
