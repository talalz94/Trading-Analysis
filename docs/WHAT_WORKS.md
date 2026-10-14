# What works and what doesn't — the short version

> The full evidence, phase by phase, is in [`STRATEGY_SEARCH_LOG.md`](STRATEGY_SEARCH_LOG.md).
> The plan that was executed is [`SEARCH_PLAN.md`](SEARCH_PLAN.md).
> Everything below is measured on BTCUSDT and XAUUSD, 2025-01 → 2026-07, three regime windows
> (+26%, −24%, −28%), always after costs, always compared against a **null entry** that trades on a
> fixed schedule with no signal at all.

---

> ## CORRECTION 2026-08-24 — the chart-pattern re-measurement is COMPLETE
>
> `_dedupe` in `quant/indicators/chart_patterns.py` ranked all detections globally by how well-formed
> they were, so a pattern confirming at bar 5,000 could delete one that had already fired at bar
> 2,997 — **a past signal depending on future bars**. Fixed in `cd70932`. Phases 12-15 were re-run on
> the causal detector and the rows below now carry the corrected numbers.
>
> **Retracted:** `chart_short_rr8` (8 of 9 windows -> **5 of 9**, the same as the metronome) and the
> monotonic target ladder (+0.012 rising to +0.152 -> **no trend at all**). Both were the lookahead.
>
> **Strengthened:** the expansion finding, from 9-32% to **18-42%** more forward range. Removing a
> hindsight filter made it bigger, which fits — the old path kept shapes selected after the fact.
>
> **Unaffected:** candlesticks, levels, entry families, exits, regimes, the cost floor, every
> metronome comparison. `add_candle_patterns` has no de-duplication step and no other phase used
> chart patterns.

---

## FAIR VALUE GAPS (2026-09-15) — no edge as a veto, at any setting tried

An unmitigated fair value gap behind a trendline break does **not** tell you to skip the trade.
`experiments/fvg_veto_search.py`, 24 gate settings x 6 windows over five months, BTCUSDT 5m, scored
by **R per trade** with the top winner trimmed:

| | observed | expected by chance |
|---|---|---|
| settings clearing all 6 windows | **0** | 0.38  (24 x 0.5^6) |
| settings winning >= 4 of 6 | 4 | **8.2**  (24 x 34.4%) |

Fewer survivors than chance, which is the same shape every other entry family in this document
produced. R per trade is the right score because a veto can only REMOVE trades: removing them at
random lowers total R while leaving the per-trade number alone, so only a per-trade improvement
means the removed trades were worse than the kept ones.

**Both directions were tested**, since they are exact mirrors and guessing would have been a coin
flip: `retrace` (an unfilled bullish gap below blocks a long — price returns to sweep it) and
`barrier` (an unfilled bearish gap above blocks a long — you are buying into a seller's zone).
`barrier` sat at the top of the table at 4 of 6 windows and `retrace` at 3, but 4 of 6 is a 34%
event and 24 settings should produce about eight of them.

**What NOT to conclude from notebook 19.** On its default nine-day window the whole sim takes eight
trades, 4 winners and 4 losers. A three-trade block that happens to be all losers arises by chance
7.1% of the time (C(4,3)/C(8,3)), so the 96-cell grid first run there produced about seven such
cells for free — and those 96 cells are not 96 observations, they all score the SAME eight trades.

**What the gap census DOES say,** and it is structural rather than predictive: on 5m BTC over 40
days there are 1,974 gaps at `min_size_atr=0.25` with ~96 unmitigated at any moment, and the median
gap is filled within **4-6 bars**. A gap of either direction is therefore live on almost every bar,
which is why including the chart's own timeframe turns any gap veto into a constant rather than a
filter — it blocked all 8 trades. Any rule built on FVGs has to start by getting the population
down: `min_size_atr >= 0.5` (above the median) and coarser timeframes only.

## S/R ZONES (2026-09-27) — they mark areas, not levels; tuned for clarity, not edge

`experiments/sr_zone_search.py`: 288 settings (pivot width, cluster %, min touches, break
retirement, anchoring) x 5m / 15m / 1h / 4h x BTCUSDT's 9 standard windows, 2022-2026. Selected
on 2022H1-2024H1, **reported on 2024H2-2026H1**, which the selection never saw.

A **test** is price returning to a live zone after moving at least 1 ATR away; it **holds** if it
closes 1 ATR back on the side it came from before closing 1 ATR through. The hold rate means
nothing alone — any level in a driftless market holds about half the time — so it is read
against two nulls:

| | 5m | 15m | 1h | 4h |
|---|---|---|---|---|
| train -> forward rank correlation | **+0.81** | **+0.66** | +0.02 | +0.16 |
| settings beating **shuffled bars** in every window | **67** of 282 | **22** of 247 | 0 of 138 | 0 of 23 |
| ...expected by chance | 0.55 | 0.48 | 0.27 | 0.04 |
| mean forward edge vs the **same zones moved 1-3 ATR** | **-0.015** | **-0.022** | **-0.012** | -0.009 |

**Against shuffled bars the detector finds something real on 5m and 15m** — far more survivors
than chance and a ranking that holds up out of sample. **Against the shift null it finds nothing:**
a price 1-3 ATR from the zone holds as often as, or slightly more often than, the zone itself. The
likely reading is that real BTC has short-term mean reversion that shuffled bars lack, and any
nearby level captures it. **A zone marks an area where price turns; the exact line carries no
information.** Do not trade the line as a precise level.

**What that did to the tuning.** Ranking every setting on edge picked degenerate ones — 15m's
"best" drew 0.5 lines and went negative forward. So `SR_SENSITIVITY` is a **clarity dial**: each
level is a band of lines on screen (tight 2-4, normal 4-8, loose 8-16 within 20 ATR of price) and
edge only breaks ties inside it. On 1h and 4h the tie-break is noise (rank corr +0.02 / +0.16):
any setting in a band is as good as any other there.

| lines drawn near price | 5m | 15m | 1h | 4h |
|---|---|---|---|---|
| old defaults | 18.3 | 21.1 | 17.2 | 8.3 |
| `normal` | **5.5** | **4.8** | **4.3** | **4.2** |

On notebook 21's chart that is **~52 S/R lines on screen at any moment down to ~7**, with no
measurable change in how often price reacts at them (every forward edge within +-0.03, a hold-rate
standard error being ~0.02).

**Three things measured along the way:**
- **Anchoring** (a zone may not be wider than `cluster_pct`) moves edge by +-0.003 on every
  timeframe — noise. It is a correctness fix at no cost, and every preset uses it.
- **Retiring broken zones costs reaction on 5m and 15m** (forward edge +0.017 with no retirement
  against -0.008 to -0.020 with it on 5m; +0.013 against -0.019 to -0.036 on 15m). Broken levels
  keep acting as levels — role reversal. So the table does not force retirement there; `normal`
  gets its clarity from tighter clusters instead. `SR_BREAK_BARS = 4` forces it, at that measured
  cost. On 1h and 4h it runs the other way (+0.029 and +0.073 at `break_bars = 8`), which is why
  their presets do retire — but those two rankings are noise, so read that as "no harm" at most.
- **The legacy eviction rule was wrong**: it dropped the least-touched zone, so on a trending
  instrument old strong levels held every slot. BTC 4h, 2022-26, ended with 40 live zones a median
  67% from price and 1 within 5%. The corrected engine drops the zone farthest from price.

### Does an OLD level matter more? (2026-09-27) — no, and no less either

`experiments/level_age_study.py` tags every test with the level's age in days and reads each age
bucket against the same shift null. Same yardstick, same train/forward split. Pooled across
timeframes:

| forward edge vs the level moved 1-3 ATR | young (<30 days) | old (>30 days) |
|---|---|---|
| S/R, `normal` presets | +0.002 (n 13,970) | -0.017 (n 1,152, se 0.020) |
| S/R, retirement OFF | +0.002 (n 14,943) | -0.003 (n 3,654, se 0.011) |
| FVG, first return | -0.002 (n 23,655) | +0.037 (n 188, se 0.041) |

Every old-level cell flips sign between train and forward (S/R +0.042 -> -0.017; FVG -0.103 ->
+0.037). **An old level holds exactly as often as a young one, and neither beats a price 1-3 ATR
away** — on 4h that is $2-5k at BTC 86k. Price does come back to old areas; the old LINE is not
what it reacts to. The age limits are therefore a CLARITY choice, never an edge one: keeping old
levels does not hurt reaction and does not help it.

How often price returns to an old gap at all: **94% of first returns into an FVG happen within a
day of its birth**, 1.8% after a week. The S/R zones never reach a year old, even with retirement
off — the 40-zone cap evicts the farthest first.

The limits these replace were never measured: `FVG_MAX_AGE_BARS = 500` was a speed bound on the
mitigation scan, and `ZONE_LOOKBACK = 2000` was a workaround for the legacy eviction bug above.
Both are counted in each layer's OWN bars, so the same number means very different ages:

| | 5m | 15m | 1h | 4h |
|---|---|---|---|---|
| 500 bars | 1.7 d | 5.2 d | 21 d | 83 d |
| 2000 bars | 6.9 d | 21 d | 83 d | 333 d |

The S/R tuning study ran on UNBOUNDED history, so its line counts assume no lookback cut.

---

## EXIT RULES (2026-09-27) — neither beats noise; the strategy they were tried on loses to costs

Two rules, both proposed from ONE trade (notebook 21's #04: 0.1R short of a 3R target, then
stopped out):

- **move the target** (`USE_TP_MOVE`) — after a near miss (the best price got `near` of the way,
  then a close gave `back` of it back) or a long wait (`after_bars`), the target moves to `to` of
  its distance;
- **FVG exit** (`USE_FVG_EXIT`) — a live gap sitting within `near_r` R of the target when the trade
  opens; price filling it closes the trade.

`experiments/exit_rules_search.py`: notebook 21's exact settings over 19 quarters, 2022Q1-2026Q3
(1,194 trades), 26 arms fixed before the run, selected on 2022Q1-2024Q2 and judged on
2024Q3-2026Q3, scored as total R minus the baseline's, per quarter.

| | |
|---|---|
| arms beating the baseline in every quarter | 0 of 26 |
| train -> forward rank correlation of the arms | **-0.50** — the best on train were the worst forward |
| largest quarterly t-statistic, any arm | 1.31, with 26 tried |
| best train arm (near 0.6, back 0.4, to 0.4) | +39.2R train -> **-29.7R forward** |
| arms positive in BOTH halves | 6 of 26 — chance predicts 6.5 |

The arms that do no harm are the gentlest (near 0.9; a gap exited at its far edge), and they are
the ones closest to doing nothing. **Why the idea fails:** the near miss it is built for is real,
but a pullback that goes on to the full target is MORE common. The gentlest version fired 59 times
in 1,194 trades — 23 of those would have stopped out, 35 still reached the full 3R. Aggressive
versions cut far more winners than they rescue.

**The larger finding is the strategy itself.** Notebook 21's settings lose over four and a half
years: 1,194 trades, **-0.157R per trade**, 26.8% win, 17 of 19 quarters negative. The same trades at
ZERO cost make **+0.08R per trade**, so costs (8 bps fee + 1.5 bps slippage per side, against a
median stop 0.94% from entry) take ~0.24R from every trade — three times the gross result. That
+0.08R has not been tested against the `time_entry` metronome, so it is not evidence of an edge
either. The window the notebook shows (+4R on 8 trades) is a good stretch of a losing rule. No exit
rule can repair that; the lever is the entry, or the cost per R.

Both rules stay in the engine, off by default, because they are correct tools and cheap to re-test
on a different entry. Neither should be switched on on this evidence.

### When does it work? Market conditions at entry (2026-09-27) — nothing predicts; stop width sets the cost

`experiments/regime_study.py`: every one of the 1,194 trades, 16 conditions read at its entry bar
(volatility level and rank, a recent vol shock, Kaufman efficiency, 1h choppiness, RSI 5m/1h,
momentum into the break, 1h/4h trend bias and slope, stretch from the entry EMA, range position,
stop width, relative volume, side, trigger timeframe, session), judged train 2022Q1-2024Q2 ->
forward 2024Q3-2026Q3, counting a relationship only if p < 0.05 in BOTH halves with the same sign.

| question | features that hold in both halves | all features together (forward AUC) |
|---|---|---|
| what separates winners from losers | **0 of 15** (chance ~0.02) | **0.48** [0.43, 0.53] |
| when does moving the target help | 0 of 15 | 0.47 [0.40, 0.55] |
| when does the FVG exit help | 0 of 15 | 0.51 [0.45, 0.58] |

The model fits the train half at AUC 0.93-0.99 and is a coin flip forward: it memorised. Session,
trigger timeframe and side move the averages in one half and not the other (Asia vs the rest:
p 0.21 train, 0.33 forward). **Nothing in the market at entry says whether this strategy wins, or
which exit to use** — the same answer the hybrid-strategy search gave on the entry families.

**The trap the first pass fell into.** Scored on R, 11 of 15 features "held" with Spearman up to
0.54. It was costs: 73% of trades lose -1R MINUS costs, and costs are a fixed % of price, so a tight
stop makes the same fee a bigger slice of R; ranking by R ranked the losers by stop width. Scored on
win/loss and on zero-cost R, every one of them vanished. **Never rank trades by R to find
predictors — rank by outcome, and read costs separately.**

**The one lever: stop width in price terms.** Costs take 0.43R from a trade in the quietest third
of markets and 0.10R in the most volatile (Spearman -0.92). Skipping entries whose stop is tighter
than 0.76% of price (the train tercile) leaves the gross result unchanged (+0.006 -> -0.029 train,
+0.180 -> +0.214 forward) and cuts costs from ~0.24R to ~0.14R a trade: net -0.235 -> -0.168R train,
-0.069 -> +0.062R forward, better in **15 of 19 quarters (7 of 9 forward)**. It is cost management,
not an edge — train is still negative — but it is consistent, and it is the only thing here that is.

Also measured: 34% of losers were +1R before they turned, but breakeven at 1R LOSES 24.8R forward
(1.5R: +27.6 train, -12.2 forward). The winners pull back through +1R too.

**Which line timeframes to trade** (`TRADE_TFS`, 2026-09-29, the user's settings: 5m confirmation
on, FVG exit on). Judged per trade, since each arm removes trades:

| trading | trades | R/trade train | R/trade forward | vs all three, per trade |
|---|---|---|---|---|
| 5m + 15m + 1h | 955 | -0.240 | -0.053 | -- |
| 15m + 1h | 515 | -0.255 | +0.009 | -0.015 / +0.061, t 0.17 |
| 1h only | 160 | -0.236 | -0.066 | +0.004 / -0.013, t -0.25 |
| 5m only | 717 | -0.324 | -0.108 | **-0.084 / -0.055, t -3.35** |

5m-line trades on their own are the worst, consistently. Dropping them halves the trade count and
leaves the rest no better or worse per trade. Nothing here turns the strategy positive.

**Four time and stop rules** (2026-09-29, the user's settings then: 15m+1h+4h lines trading, FVG exit
on; 546 trades, -0.111R per trade). The config's example values, one arm each:

| rule | train dR | forward dR | forward quarters beaten | t |
|---|---|---|---|---|
| DECAY: target -2% of its distance every 24h | -3.7 | -0.5 | 2/9 | -0.69 |
| STEP: at 50% of the way stop +50% of risk, at 80% to breakeven | +15.5 | -11.7 | 2/9 | 0.40 |
| TSTOP: after 12h, on a close in profit, stop to breakeven | +5.4 | -31.2 | 3/9 | -1.26 |
| HARD: close 15m-line trades after 48h | +16.6 | -18.8 | 2/9 | -0.15 |

Train -> forward rank correlation -0.40: the ones that helped 2022-24 are the ones that hurt
2024-26. **None beats noise.**

### Other markets (2026-10-01) — one candidate, and it did not replicate

`experiments/instrument_scan.py` / notebook 22: notebook 21's settings as they stood (15m+1h+4h lines
trading, FVG exit on, target decay 5% per 24h), each market with its own costs, 2022Q1-2026Q3.
**Promising** = R per trade > 0 after costs in BOTH halves AND the trendline entries beat a
metronome (an entry every hour on the same side of the 15m EMA, same exits) in both.

| market | trades | R/trade | train | forward | costs/trade | entry vs metronome (train / fwd) |
|---|---|---|---|---|---|---|
| **USDJPY** | 387 | **+0.147** | +0.196 | +0.089 | 0.018 | **+0.157 / +0.247, t 2.82** |
| Gold | 358 | +0.030 | -0.176 | +0.279 | 0.028 | -0.201 / +0.248 |
| EURJPY | 388 | +0.051 | +0.250 | -0.132 | 0.032 | +0.100 / -0.074 |
| GBPUSD | 402 | -0.010 | -0.047 | +0.033 | 0.029 | +0.060 / +0.072 |
| EURUSD | 401 | -0.014 | -0.148 | +0.145 | 0.026 | -0.162 / +0.150 |
| GBPJPY | 428 | -0.058 | -0.001 | -0.120 | 0.033 | -0.060 / -0.029 |
| WTI oil | 414 | -0.102 | -0.141 | -0.051 | 0.029 | -0.036 / +0.105 |
| BTC | 567 | -0.148 | -0.253 | -0.038 | 0.201 | +0.065 / +0.270 |

USDJPY passed, survived double the spread and a 4-hourly metronome, made money on both sides, and its
entries beat the null in 15 of 19 quarters. **But it did not replicate:** EURJPY and GBPJPY -- added
afterwards to test exactly that -- fail, EURJPY flipping from +0.250 to -0.132. With four coin flips
per market, ~0.5 of 8 markets pass by luck alone; one did. 2022 also supplied 32R of USDJPY's 57R.
Read USDJPY as a lead to watch forward, not a result.

Two things that generalise: **costs are a BTC problem** -- 0.20R a trade there against 0.02-0.03R on
gold and FX, where the stops are wide against the spread; and **a trending market flatters any entry**
-- on gold the hourly metronome made +0.046R a trade, MORE than the trendline entries (+0.030R).

### New York open reversion ("fair pricing theory") on Nasdaq 100 (2026-10-13) — no edge, and no bias

From a 2026 podcast with a prop-firm trader, JJ Simon (summary in docs/SESSION_NOTES.md). He trades
NQ futures on the 1-minute chart and says the edge per trade is "slight".
`experiments/nq_open_reversion.py` tests the rule as he states it:
- fair price = the 9:30 ET open; entries only 9:30–11:00, back TOWARD fair price;
- on a candle close: a displacement (bigger body than the opposite-coloured candle before it,
  closing beyond its wick) or a break of structure (a close beyond the latest swing);
- at least 80% of the target left to fair price;
- static 25 / 38 points (his evaluation setting, 2026 points, scaled to each day's price);
- one position at a time; three losses in a row ends the session; NQ futures costs 0.725 points.

Data: Dukascopy NAS100 1m, 1,243 days, 2022Q1–2026Q3.

| arm | trades | win % | R gross | R net | train | forward | quarters + |
|---|---|---|---|---|---|---|---|
| his rules (either entry) | 5,115 | 38.5 | -0.029 | **-0.080** (t -4.7) | -0.060 | -0.104 | 2/19 |
| break of structure only | 3,406 | 39.1 | -0.015 | -0.066 | -0.064 | -0.069 | 2/19 |
| displacement only | 4,004 | 38.7 | -0.025 | -0.075 | -0.060 | -0.092 | 3/19 |
| target = fair price itself | 3,586 | 21.3 | -0.050 | -0.100 | -0.023 | -0.189 | 6/19 |
| without the 3-loss stop | 6,438 | 38.8 | -0.022 | -0.073 | -0.060 | -0.089 | 1/19 |
| metronome TOWARD the open (every 5 min) | 7,510 | 38.5 | -0.030 | -0.080 | -0.073 | -0.088 | 3/19 |
| metronome AWAY from the open | 8,340 | 40.6 | +0.023 | -0.027 | -0.060 | +0.009 | 8/19 |

- **There is no reversion bias to capture.** Trading toward the open every 5 minutes does WORSE
  than trading away from it, in both halves. If anything the first 90 minutes drift away from the
  open.
- **His entries add nothing.** They score the same as the toward-metronome (+0.013 train,
  -0.015 forward). The win rate is 38.5% where 40.8% is needed after costs.
- **Negative every year 2022–2026**, both price-scaled and at his literal fixed 25/38 points.
- **Not tested:**
  - the opening continuation trade;
  - his discretion (moving fair price, "wick rejection" entries).
  - His A+ setup is tested below.
- The away-metronome's +0.009 forward is not a finding. It was a null, t 0.45, and negative in
  train.
- **What this does NOT say:** whether the method pays on prop firms. With many accounts and losses
  capped at evaluation fees, a firm's payout rules can turn a small negative edge per trade into
  payouts. That is a bet on the firm's rules, not on the market. These trades carry no edge of
  their own.

### A news blackout on the trendline strategy (2026-10-14) — too rare to matter

`NEWS_BLACKOUT` in `notebooks/strategies/tl_break_confirm.py` means no new trade from 30 minutes
before to 30 minutes after a USD critical release: CPI, payrolls, the Fed, PCE, PPI, retail sales,
GDP. Those are the releases the calendar covers in every quarter. One arm, fixed before the run,
judged by `experiments/evaluate.py` over 19 quarters on BTC 5m.

| | trades | R / trade | train dR/trade | fwd dR/trade | t/trade |
|---|---|---|---|---|---|
| strategy as set | 1,194 | -0.157 | | | |
| news blackout 30/30 | 1,183 (-11) | | +0.023 | -0.007 | 0.93 |

- **It touches 11 trades in four and a half years.** Critical US news falls inside this strategy's
  entries too rarely to change anything.
- The train gain (+15.1R over 9 quarters) does not carry forward (-3.6R). Per trade it is t 0.93.
- It stays OFF. It is not harmful, just unmeasurable. A wider window or more currencies would be a
  new arm with its own chance baseline, not a re-run of this one.

### His "A+" setup: the expected-news reversion (2026-10-13) — negative even before costs

His favourite trade, the one he risks most on. On 8:30 a.m. New York "red folder" days (CPI,
PPI …) the news is "priced in", so the huge candle off the release is "unfair" and price returns
to where it stood before the news. `experiments/nq_news_reversion.py`:
- **News days** come from the new economic calendar (`quant.news`, docs/NEWS.md): days with a US
  high/critical release at 8:30, about 100 a year.
- **Fair price** is the 8:29 close.
- **Entries** use his displacement / break-of-structure rules toward it, on candles from 8:31 to
  10:59.
- **Everything else** is the open test's: price-scaled 25/38, 3 losses end the day, NQ costs.

**With the complete calendar** (every red-folder release, Nov 2021 – Apr 2025: 302 news days):

| arm | trades | R gross | R net | train | forward |
|---|---|---|---|---|---|
| his rules, every 8:30 red day | 1,723 | -0.035 | **-0.091** (t -3.1) | -0.103 | -0.054 |
| CPI / PPI / payrolls days only | 738 | -0.027 | -0.083 | -0.096 | -0.046 |
| entering only from 9:30 | 1,490 | -0.009 | -0.065 | -0.070 | -0.050 |
| small surprises only (\|z\| < 1.5, "priced in") | 1,554 | -0.027 | -0.082 | -0.094 | -0.051 |
| big surprises only ("outsiders") | 161 | -0.155 | -0.216 | -0.217 | -0.204 |
| metronome TOWARD the pre-news price | 2,245 | -0.018 | -0.073 | -0.082 | -0.048 |
| metronome AWAY from it | 2,370 | +0.024 | -0.033 | -0.024 | -0.066 |
| his rules on QUIET days (no US high news) | 1,518 | -0.013 | -0.072 | -0.066 | -0.103 |

**Over the whole span** (2022Q1 – 2026Q3, 309 news days). This uses the releases the official sources
cover throughout: FRED's CPI, PPI, payrolls, retail sales, PCE and GDP, plus the Fed's own
decisions. Both halves use one definition.

| arm | trades | R gross | R net | train | forward | quarters + |
|---|---|---|---|---|---|---|
| his rules, official 8:30 days | 1,814 | -0.018 | **-0.069** (t -2.4) | -0.099 | -0.029 | 7/19 |
| CPI / PPI / payrolls days only | 1,015 | -0.017 | -0.067 | -0.096 | -0.033 | 6/19 |
| metronome TOWARD the pre-news price | 2,389 | +0.013 | -0.038 | -0.082 | +0.014 | 7/19 |
| metronome AWAY from it | 2,345 | -0.010 | -0.062 | -0.042 | -0.091 | 7/19 |
| big surprises only (to Apr 2025) | 136 | -0.129 | -0.191 | -0.190 | -0.204 | 2/9 |

- **It loses, and before costs too.** -0.035R gross means the reversion is not there to be paid
  for.
- **News days are no better than quiet days** with the complete calendar (-0.091 vs -0.072). The
  8:29 price is not special on news days. (The whole-span quiet days, -0.100, include lesser red
  releases the later period cannot see, so they are not a clean comparison.)
- **His entries add nothing**: against the metronome they are -0.020 train and -0.006 forward on
  the complete calendar, and -0.017 / -0.043 over the whole span.
- **Toward vs away flips between halves**: away is better in train, toward in forward. So there is
  no consistent bias.
- **His one checkable claim holds in direction**: "outsiders" (a surprise of 1.5 times the
  release's usual size or more, judged only from earlier releases) are much worse. But that is 28
  days, and avoiding them still leaves -0.082R.
 Stop moves in particular convert more of the trades that would have
reached 3R into breakeven exits than they rescue from the stop -- the breakeven-at-1R result again.

---

## The one-paragraph answer

**No entry rule tested in this project has a validated edge.** Nine entry families, both directions,
eight level types, four interaction types, six timeframes, two instruments, 150 management
combinations and ten regime conditioners — the search kept producing *fewer* survivors than chance
would, and the two configurations that looked genuinely good both failed the gates that discriminate:
one is carried by under 1% of its trades; the other turned out to be short beta, losing in four of five bull windows once the sample was balanced. The
binding constraint is not the signal, it is the **cost floor**.

---

## The number that governs everything

Spread as a percentage of one R (1.5 × ATR stop):

| | 1m | 5m | 15m | 1h |
|---|---|---|---|---|
| BTCUSDT (spread 6.0 ≈ 0.7 bp) | **10.8%** | 3.8% | 2.0% | 0.9% |
| XAUUSD (spread 0.25/oz) | **7.0%** | 2.9% | 1.6% | 0.8% |

Which converts into a **break-even win rate**, `p* = (1 + cost) / (1 + RR)`:

| market | at 1R | at 1.5R | at 2R | at 3R |
|---|---|---|---|---|
| BTC 1m | 55.4% | 44.3% | **36.9%** | 27.7% |
| BTC 15m | 51.0% | 40.8% | 34.0% | 25.5% |

**Every entry family measured 31–35% wins at 2R on BTC 1m. The requirement is 36.9%.** That gap —
two to six percentage points — is the whole result. Nothing in the search closed it.

---

## ✗ What does NOT work

| Idea | Verdict | The evidence that killed it |
|---|---|---|
| Prev-day / week / session high-lows, opening range, round numbers, S/R zones, VWAP bands | ✗ no edge | Null-differenced screen: symmetry 10/50 pairs vs ~12 by chance; 2 surviving cells of 320 vs ~16 by chance |
| Break / **fade** / **sweep** / **retest** of any level | ✗ no edge | Same screen. "Fade the break" — the previously untested inverse — is not supported |
| BOS / CHoCH structure breaks | ✗ | Phase 2: worse than the metronome |
| **Volume** confirmation (first use in this project) | ✗ worse than random | Phase 2: −0.052 vs null −0.045 |
| Squeeze release, EMA-gradient extremes | ✗ worse than random | Phase 2 |
| WaveTrend cross, RSI divergence (confirmed *and* zero-lag), VWAP reversion | ✗ | Beat the null only by clawing back cost drag, still negative |
| **Reversion to the New York open** (Nasdaq 100 1m, JJ Simon's "fair pricing theory") | ✗ no bias | Toward-the-open metronome -0.080R vs away +0.009R forward; his displacement/structure entries = the metronome; -0.08R a trade after NQ costs, negative every year 2022-26 |
| **News blackout** (no new trade 30 min either side of USD critical news), trendline strategy | ✗ no effect | 11 of 1,194 trades touched; +0.023R/trade train, -0.007 forward, t 0.93 |
| **Reversion to the pre-news price on 8:30 news days** (his "A+" setup) | ✗ | -0.091R a trade, -0.035R even before costs; news days no better than quiet days; his entries = the metronome |
| Hour-of-day / weekday filters | ✗ | No quarter-to-quarter persistence; trained pick ranked 110/127 out-of-sample |
| Higher timeframe as a cure | ✗ | `real − null` is negative at every symbol × timeframe and gets **worse** higher up |
| Support/resistance-break family (the original setup) | ✗ | Parameter spike with negative neighbours; 1 of 36 survivors where chance predicts ~4 |
| Bail-out rules as the sole exit | ✗ dies on costs | +0.025R at zero cost → −0.007R at 0.7 bp |
| **Hybrid / conditional strategies** — volatility terciles, trend vs chop, range position, session, HTF bias (with *and* against) | ✗ | 3 survivors of 304, chance ~38. The metronome filtered to the US session scores +0.070 and clears the same bar |
| **Volatility compression then breakout** (quiet range, big move off the 200-EMA) | x the filter HURTS | Paired control over 288 matched pairs: compression helped in **10.1%**, median delta **-0.045R**. Quiet periods are followed by *smaller* absolute moves (57 vs 103 bps); longer coils do NOT spring harder (flat 78-86 bps); release direction is a coin flip (47.8% up). Best configs *fade* the break and die on trim-top-K (5 trades of 351) |
| **Improving the best strategy with S/R zones / EMA bias / HH-HL structure** | x no improvement | Phase 8: every filter keeping >=294 trades scored BELOW the unfiltered base; the ones that look better run on 11-71 trades. Only 1 of 153 configs beat the base at >=300 trades, by +0.006R |
| **Faster timeframes for the best strategy** | x worse | Unconditioned `wt_cross_short`: 15m **+0.065**, 5m -0.063, 1m -0.056. On 1m *every* conditioner is negative |
| **Long side vs short side** | ~ mirror images, both beta | Phase 9, nine windows over a full cycle: longs +0.014 mean in bulls / -0.063 in bears; shorts -0.044 in bulls / +0.048 in bears. Neither is inherently better; the earlier apparent superiority of shorts was a biased sample |
| **Trailing stops as an exit family** | x worst tested | Capture only **37%** of the favourable move vs **87-93%** for fixed targets; bottom of all 29 exit designs in Phase 11 |
| **"The move keeps going after I exit"** | x illusion | The metronome shows MORE continuation than the strategy (cont30 1.07 vs 1.00) - it is the max-of-a-random-walk effect |
| **Candlestick patterns** (62, incl. every classic reversal) | x no directional edge | Hit rate **50.1-50.5%**, NEGATIVE excess over a null that preserves their exact candle shapes and destroys only the order. Several INVERT (`three_inside_down` -0.57, `hanging_man` -0.42). Only `marubozu_bull` is consistent, and it is a momentum bar |
| **Chart patterns as a directional trade** (19: double/triple tops, H&S, wedges, triangles, flags) | x **settled: no directional edge** | Re-measured on the causal detector against a shuffled null that the metric handles correctly (null scores +0.011, 51.0% of cells positive). Median hit rate **0.480 real vs 0.500 null** — real data is WORSE than shuffled — and the real hit rate exceeds 50% in **1 of 9** windows. Paired per window, **0 of 13** patterns beat their own null in >=8 of 9 (chance 0.3). `chart_short_rr8` fell from 8 of 9 windows to **5 of 9**, matching the metronome; trim-top-K takes every configuration to about -1.0 after 5 trades; no bootstrap lower bound is positive in any window. The monotonic target ladder is gone |
| **Trendline breaks** (support/resistance, 144 settings x 3 TFs x 2 instruments x 9 windows) | x no directional edge | Hit rate above 0.50 in only **36% of 102 cells** — *below* the 50% chance rate — with a median of exactly **0.5000**. 56 of 102 cells beat their shuffled null (55%, chance 50%); positive AND consistent in >=8 of 9 windows: **2, against a chance expectation of 2.0**. Worst on 5m (22.5% of cells above a coin flip). The multi-timeframe test is **INCONCLUSIVE, not negative** — a higher-timeframe line is live on only 3-22% of bars, so classifying lower-timeframe breaks by it produced 5 classified events against 1,705 unclassified |
| **S/R zones as precise levels** (288 settings x 4 TFs x 9 windows) | ✗ no level-specific edge | Beat shuffled bars on 5m (67 survivors vs 0.55) but NOT the same zones moved 1-3 ATR (mean forward edge -0.015 5m, -0.022 15m): a nearby price holds as well. Zones mark areas. Retiring broken zones measurably LOWERS reaction on 5m/15m (role reversal) |
| **Consecutive-loss lockouts / daily loss caps** | ✗ ruled out analytically | Trade outcomes are **independent** (all autocorrelations inside ±0.027, `P(loss\|prev loss) − P(loss)` ≤ 0.011 and sign-flipping). Caps made expectancy *worse*: −0.023 → −0.057 |

## ~ What genuinely helps, but is not an edge

| Idea | What it actually does |
|---|---|
| **Time stop** (new engine feature, `max_bars_in_trade`) | Median expR −0.045 → −0.027 at 15–30 bars, median drawdown 62.9% → 55.8%. It helps the **null entry** by about as much, so it is cutting cost-weighted dead time, not finding better exits. Keep it; don't count it as alpha. |
| **Trailing stop** | Turns almost anything positive in-sample — including a random entry — by harvesting BTC's fat-tailed trend structure. Not an entry edge. |
| **Higher timeframe for drawdown** | Median max drawdown 64% → 32% → 14% from 1m → 5m → 15m. Real, and unrelated to expectancy. |
| **Hit the target, then exit on the first opposite candle** | Best of 29 exit designs (+0.0333 mean, +0.0375 over the metronome) and better than any trail. Real, but worth ~+0.01R - it improves a losing system, it does not make one win. |
| **Breakeven at 1-1.5R with a 3R target** | +0.0095R over a plain 3R target, the best simple exit change. Needed an engine fix: a TakeProfit closing 0% was silently inert. |
| **Smaller position size** | The ONLY risk control that works. 1% → 0.25% risk leaves `expR` identical and divides drawdown by the same factor — it buys nothing you couldn't get by trading smaller. |

## ✓ The one positive, actionable rule found

**Chart patterns predict EXPANSION, not direction — and this is the one pattern finding that survived.** Forward 20-bar range after a confirmed shape, against a shuffled null sitting at **0.97** (i.e. no expansion, exactly as it must): `triple_bottom` **1.42**, `triple_top` 1.25, `symmetrical_triangle` 1.22, `head_shoulders` 1.21, `inverse_head_shoulders` 1.18. Previously reported as 9-32% on the superseded detector; removing the hindsight de-duplication made it **larger**, which fits the mechanism — the old path kept shapes selected after the fact and diluted the effect. It survives in PRICE terms as well as ATR units, so it is not the denominator trap. The effect is 0.17-0.36% over 20 bars against a 0.007% spread, **10-25x the cost floor** — unusually for this project, cost is not what stops it. The **sub-50% hit rate** is: exploiting expansion without direction needs a wide target, and phase 14 shows every such configuration collapsing to about -1.0 once the best 5 trades of ~45 are removed. Useful as a **risk signal** — size down, widen stops, expect a move — not as a trade.

**Compression releases cluster at the US open.** 35% of 3,622 compression episodes release in
12:00-14:00 UTC (cash open 13:30) and travel ~40% further there (111-125 bps vs 60-90). Real, but
it is a session-volume effect and it did NOT convert into a tradable rule. The weekend version is
backwards: 99.8% of weekend compressions resolve on the weekend, and they are *smaller*.

**Do not trade into a volatility shock, and do not chase the hour after it.** Conditioning entries on
the shock bar itself costs **−0.2596R** — the single largest effect measured anywhere in this search —
and the following 60 minutes costs −0.0888R. It holds across every entry family and both directions.
Since `vol_shock` is the announcement proxy (no economic calendar is available to this platform),
this is the empirical form of "stay out of the release." It is a rule about when *not* to trade
rather than an edge, but it is large, consistent, and the only Phase-4 result worth acting on.

## ? The only two things that are not obviously nothing

**0. A chart pattern breaking AT an S/R support zone** (measured on the superseded detector; not re-tested) — the only filter in the whole project that improved a rule on a like-for-like basis instead of by starving it: +0.0197 -> +0.0495 while giving up just 13% of its trades. Unvalidated, but the shape of the result is right.

**1. `pdlow_fade_long` + trailing stop (BTC 1m)** — long on a fade of the previous day's low.
Worst-window +0.136R against the null's +0.040R, survives 2× spread where the null does not, and is
positive across every stop and trail value tested. **But:** the top 1% of trades supply 40–77% of all
profit, deleting the best 10 trades of 2,081 turns it negative, and its bootstrap lower bound is
below zero in two of three windows.

**2. `wt53_short|atr2.5/rr1.5` (BTC 15m) — RETRACTED.** It looked like the cleanest result in the
search (+0.074 / +0.148 / +0.077, 509 trades, 17.3% drawdown). Phase 9 re-ran it on nine windows
spanning a full cycle and it **loses in four of the five bull windows**:

```
2022H1  2022H2  2023H1  2023H2  2024H1  2024H2  2025H1  2025H2  2026H1
 bear    bear    bull    bull    bull    bull    bull    bear    bear
+0.196  -0.043  -0.202  -0.092  -0.131  -0.079  +0.074  +0.148  +0.048
```

It scored well only because the original window set was 2/3 bear. **It is short beta, not an edge.**

Neither is fundable. Both are what a 648-config search returns when the true edge is zero.

---

## The methodology that produced these verdicts

Three separate times, a plausible-looking measurement turned out to be measuring itself. These are
now standing rules:

1. **Difference every forward-return screen against a within-day-shuffled null.** Two independent
   control schemes (whole-window drift adjustment, day-matched) both manufactured edges of exactly
   the size being hunted. Shuffled data with no structure produced *larger* "edges" than the real
   series.
2. **Every entry competes against a metronome.** `time_entry` trades on a fixed schedule with no
   signal, through the same stops, targets, costs and windows. Most "real" entries lose to it.
3. **Count survivors against chance.** `n_configs × 0.5^n_windows`. Phase 1: 2 vs ~16. Phase 2: 1 vs
   ~22. Phase 4: 3 vs ~38. Phase 5: 18 vs ~81. Cross-instrument: **0 vs ~81**. Six counts, five of
   them below chance. A search that keeps returning fewer survivors than random is describing an
   empty space.
4. **Trim the top K trades, not just the top one.** Fat-tailed payoffs pass every conventional gate.
   Deleting the best 10 of 600–2,300 trades turned *every* Phase-3 survivor negative.
5. **Bootstrap per window, never pooled.** Pooling let candidates show `p_neg ≈ 0.0001` when no
   individual window's lower bound cleared zero.
6. **A gate the metronome also passes is not a gate.** Rotating out-of-sample was passed by a random
   entry with a trailing stop, which retires that test for this payoff shape.

7. **Judge a filter at constant trade count.** Apparent edge and sample size moved in opposite
   directions almost monotonically in Phase 8 — cut any sample hard enough and some subset looks
   excellent. A 7-trade configuration scored +1.49R.
8. **Balance the window set before trusting a directional result.** Three windows drawn from a
   single falling market cannot separate "short edge" from "short beta". Every gate in Phases 6
   and 8 was applied to the eventual winner and none caught it, because they all took the window
   set as given. Phase 9's nine-window cycle retracted it in one table.
9. **Measure capture before optimising exits.** "Am I leaving money in winners?" and "am I giving winners back?" have opposite fixes. Fixed targets already capture 87-93%; the real leak was that **30% of trades reaching +1R end up losing**.
---

## What would have to change

The search space of *signals on 1-minute BTC* is exhausted; more indicators will not help, because
the constraint is arithmetic. What could move the needle, in order of expected value:

1. **A lower cost structure.** At 0.7 bp the 1m game needs a 36.9% win rate at 2R. Maker-only
   execution, or a venue with materially tighter effective spread, changes the required edge more
   than any indicator could.
2. **Longer holding periods.** Everything here targeted minutes to hours. The cost floor falls ~12×
   from 1m to 1h and ~27× to 4h. That contradicts "quick trades", which is the honest tension.
3. **A different edge class entirely** — funding-rate / basis, cross-exchange dislocation,
   order-book microstructure. None is reachable with OHLCV bars, which is all this platform ingests.
4. **Test the assumption before building on it.** The clustering test settled the entire risk layer
   in one measurement: outcomes are independent, so no lockout can help. That is cheaper and more
   honest than sweeping lockout lengths and keeping the best-looking one.
5. **Accepting a trend-following payoff for what it is** — 15–30% win rate, median trade a full
   stop-out, profits from a handful of runners, 25–80% drawdowns. That is a real and well-documented
   way to make money. It is also the **worst possible profile to leverage**, which is the direct
   conflict with the stated goal of "quick easy trades with high leverage".
