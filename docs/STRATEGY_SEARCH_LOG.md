# Strategy Search Log

> Running record of the automated search: **what works, what doesn't, and why**. Negative results are
> recorded as carefully as positive ones — most of the value here is in knowing which doors are shut.
> Companion to [`SESSION_NOTES.md`](SESSION_NOTES.md) (project gotchas) and
> [`../experiments/search_harness.py`](../experiments/search_harness.py) (the evaluator).
>
> Backup before this search began: branch `research/scenario-framework-2026-07-30`, tag
> **`backup-pre-search`** (commit `7f03e7f`).

---

## Ground rules the harness enforces

Each of these came from a result that looked good and wasn't.

| Rule | Why |
|---|---|
| Costs always on (`spread=6.0` ≈ 0.7 bp on BTC) | anything averaging <0.5R per win dies to sub-bp friction |
| Score across **3 regimes**, judge the **worst** | short-only made +294% in a −46% year and lost 43% in a +26% one |
| `allow_leverage=True` **and** `max_notional_pct=5000` | otherwise risk sizing silently caps at ~1× equity (realised 0.25%, not 1%) |
| Expectancy in **R**, not % | strips sizing out so configs compare |
| **Outlier guard**: `expR_ex_top`, `med_R` | one +202R trade can carry a whole window (see §2) |
| Selection needs a **hold-out** | the weekday filter ranked top-decile in-sample, bottom-15% OOS |

Regime windows (BTC 1m): `bull25` 2025-01→07 **+26.0%** · `bear25` 2025-07→2026-01 **−24.4%** ·
`bear26` 2026-01→07 **−28.3%**.

---

## 1. The LONG mirror — built, and it does NOT rescue the setup

`resistance_break_long@v1.0 / v1.2` mirrors the support-break short (above a rising EMA + break of a
resistance zone), with stops/TPs/secondaries all made side-aware (`_side` injected by `compose`, so a
long's stop sits below entry on a *low*). Verified: every stop lands on the correct side for all 6
stop methods × both sides.

**Result: the long side is worse than the short side almost everywhere.** Of 36 configs (4 scenarios
× 3 stops × 3 TPs), the 18 long variants occupy nearly the whole bottom of the table; the best long
(`resiv1.2|atr|trailing`) is positive in 2 of 3 windows but **−0.162 expR** in the third.

**Reading:** the setup *shape* is not a source of edge in either direction. Earlier short-side profits
were the −46% market, not the pattern. Mirroring removes the directional excuse and the edge does not
appear — which is consistent with the earlier finding that the entry has no measurable predictive
power.

⚠️ Caveat: the window set is 1 bull / 2 bear, which structurally disadvantages longs. Needs a
bull-heavy window set before calling the long side definitively dead.

---

## 2. ⚠️ THE BIGGEST TRAP FOUND SO FAR — one trade carrying a whole result

Two configs looked like the first genuine cross-regime winners:

```
                        expR_min  bull25   bear25   bear26
suppv1.2|atr|trailing     0.1416  0.1416   0.1965   0.1577   <- positive in ALL THREE
suppv1.0|high_of_n|trail  0.0469  0.0511   0.1442   0.0469
```

Then the outlier guard:

```
config                  window  expR   expR_ex_top   med_R   top_R   totR   totR_ex_top
v1.2|atr|trailing       bull25  0.1416     -0.0075  -1.0397  202.2  192.0        -10.1
v1.2|atr|trailing       bear25  0.1965      0.0308  -1.0439  195.9  232.3         36.3
v1.0|high_of_n|trail    bull25  0.0511      0.0047  -1.0054   35.7   39.3          3.6
```

**`v1.2|atr|trailing` in the bull window is ONE trade.** Best trade +202.2R, total 192.0R — remove it
and everything else sums to **−10.1R**. Not an edge; a lottery ticket that happened to print.

**`med_R ≈ −1.0` for both.** The *median* trade is a full stop-out: more than half of all trades lose
exactly 1R, and the entire P&L is a thin tail of runners. That is the trailing-stop profile.

**This profile is the opposite of the stated goal.** "Quick easy trades with high leverage" implies a
high hit-rate; this is <50% win rate, 80–85% drawdown, median leverage 7× (max 48× against the 50×
cap). At high leverage an 80% drawdown is ruin.

→ Harness now reports `expR_ex_top`, `expR_ex_top_min`, `med_R`, `top_R` on every config, and nothing
is called a winner on `expR` alone.

---

## 3. The central tension (working hypothesis)

Everything tested so far lands on one of two horns:

| Profile | Example | Fails because |
|---|---|---|
| **High win rate, small wins** | `opposite_candle`, `profit_candle_bail` | avg win ~0.25R over thousands of trades → **destroyed by costs** (+0.025R at zero cost → −0.007R at 0.7 bp) |
| **Low win rate, tail wins** | `atr`/`high_of_n` + `trailing` | med_R = −1.0, DD 80%+, result carried by 1–2 trades → **not tradeable at leverage** |

The goal ("quick easy trades, high leverage") needs the first profile with a bigger average win — i.e.
the same hit-rate but a larger move per trade. **That is the argument for testing higher execution
timeframes**: on 5m/15m the same setup shape yields moves several times larger in price terms while
the spread stays fixed, which directly attacks the binding constraint.

---

## Status / next

- [x] Backup + tag `backup-pre-search`
- [x] Long mirror scenario, side-aware stops/TPs/secondaries (196 tests green)
- [x] Harness with cost, multi-regime, sizing and outlier guards
- [x] Sweep 1: sides × stops × TPs → `experiments/results_sweep1_sides.csv`
- [ ] Sweep 2: higher execution timeframes (5m / 15m) — attacks the cost constraint
- [ ] Daily / session loss cap (engine feature) — attacks the drawdown constraint
- [ ] MTF bias filters, session windows, announcement effects
- [ ] Train/test selection with the all-candidates null

---

## 4. Execution timeframe — hypothesis WRONG, but a useful side effect

Hypothesis: spread is fixed in price units, so a bigger bar earns more per unit of friction → 5m/15m
should beat 1m. **Not supported.** Median `expR_ex_top_min` across 36 configs per timeframe:

| TF | median expR_ex_top_min | median trades | median max DD |
|---|---|---|---|
| 1m | −0.0794 | 3,816 | **64.5%** |
| 5m | −0.0910 | 820 | 32.2% |
| 15m | **−0.1373** | 246 | **14.3%** |

15m is the *worst* on expectancy. Bigger bars did not buy an edge.

**But drawdown collapses with timeframe — 64% → 32% → 14%.** For a leveraged account that matters more
than a small expectancy difference: the 1m variants are unusable at leverage regardless of expR.

**Survivors of the strict test** (positive in all 3 regimes *after removing each window's best trade*),
out of 108 configs:

```
tag                               n     expR_min  expR_ex_top_min  med_R    top_R  ret_min  dd_max
15min|suppv1.0|high_of_n|rr      156      0.0823           0.0484  -1.0024    2.0     3.50   11.57
1min|suppv1.0|high_of_n|trailing 2056     0.0469           0.0047  -1.0072   35.7    15.16   49.31
```

2 survivors from 108 candidates is *at or below* chance — no celebration warranted. But the 15m one is
structurally different: `tp_rr=2` **caps max R at 2.0**, so no single trade can carry it, and DD is
11.6%. Note it is `v1.0` (the S/R-zone version) — the same version that showed the only consistent
sign in the earlier entry-edge analysis.

---

## 5. Deep dive: `15m | support_break_short@v1.0 | high_of_n | rr(2R)`

### Encouraging

**Cost-insensitive** — the friction that killed everything on 1m is irrelevant here:

```
spread   0.0 -> mean expR +0.1986      spread  12.0 -> +0.1906
spread   6.0 -> +0.1946                spread  24.0 -> +0.1699
```

**Hold-out passes.** Train 2025 → test 2026 untouched:

```
TRAIN 2025: n=93 win=40.9% expR=+0.2105 ret=+20.40% dd= 7.64% pf=1.34
TEST  2026: n=68 win=42.6% expR=+0.2489 ret=+17.56% dd=10.79% pf=1.41
```

That is the first config all session where train and test both hold up.

### Why I do NOT trust it yet

1. **Parameter sensitivity is spikes, not plateaus** — the overfit signature. A real edge degrades
   smoothly; this flips sign between adjacent settings:
   ```
   stop_n     10:-0.029   15:-0.027   20:+0.195   30:+0.121   40:+0.112   <- cliff between 15 and 20
   grad_max None:+0.035  0.0:+0.195  -0.02:+0.047  -0.05:+0.068           <- spike exactly at 0.0
   ```
   The chosen config sits on a spike in *every* dimension.
2. **The quarterly mean is carried by a 5-trade quarter.** 2025Q1 shows expR +1.136 on **n=5**. Drop
   it and the mean falls from **+0.195 to +0.038**; positive quarters go 4/7 → 3/6, a coin flip.
3. **Still directional.** By BTC direction: down-quarters 3/4 positive, up-quarters 1/3. It loses in
   rallies (2025Q2 +29.9% → −0.054; 2026Q3 +9.1% → −0.256), i.e. much of the "edge" is still the
   short side meeting a falling market.
4. **161 trades total.** Far too few to separate +0.2 expR from noise.

**Verdict: promising shape, not a validated edge.** The bounded payoff, low DD and cost-insensitivity
are genuinely attractive properties; the parameter spikes and sample size say do not size it up.

---

## 6. What this means for the stated goal

Goal: *quick easy trades, high leverage.* Against that:

- **1m execution is out.** 64% median drawdown; leverage turns that into ruin.
- **Trailing exits are out** for this goal — `med_R = −1.0` means you lose most trades and wait for a
  tail. That is neither "quick" nor "easy", and the equity path is brutal.
- **Bounded-R targets on 15m** are the only profile matching the goal so far: capped downside per
  trade, ~11% DD, cost-insensitive. But hit rate is only ~41%, so it is not "easy" either — it is a
  1-in-2.4 winner paying 2R.
- **Nothing found so far justifies high leverage.** Every candidate either has ruinous DD or a sample
  too small to size on.

---

## 7. Daily / session loss cap — built (engine feature), and honestly assessed

New `BacktestConfig.daily_loss_limit` (% of the equity a session opened with) + `session_col`
(default: calendar day). Once a session's **realised** loss reaches the cap, no NEW trade opens for
the rest of that session. Open trades are never force-closed — their stops still run — because
"broker stops you out" and "stop taking new risk" are different things and only the latter is modelled.
Default `0` = off and is byte-identical (test asserts it).

**Important limitation:** the cap can only act on *realised* loss, so it cannot prevent the trade that
breaches it. On 15m, where the strategy averages ~1.1 trades per session, that means **the cap is
essentially inert** — at 2% it removed 0 of 144 trades.

**Does it help? Only on drawdown, and only where trade frequency is high.**

```
1m  suppv1.0|high_of_n|trailing        expR                         maxDD
window     cap:  1%      2%      4%     off  |  1%     2%     4%     off
bear25         0.2352  0.1345  0.1590  0.1442| 22.8   31.1   34.3   33.8
bear26        -0.0607  0.0393  0.0296  0.0469| 33.4   39.1   45.3   46.5
bull25        -0.0220  0.0873  0.0582  0.0508| 40.2   36.1   46.1   49.3
```

- **Expectancy: no consistent gain.** A 1% cap helps `bear25` (+0.144→+0.235) and *hurts* the other
  two (bull25 +0.051→−0.022). It cuts losers and winners alike.
- **Drawdown: a real, consistent gain.** 1% cap lowers max DD in every window (49.3→40.2, 46.5→33.4,
  33.8→22.8). For a leveraged account that is the point — it buys survivability, not edge.

**Verdict: use it as a risk control, never as an alpha source.** It is worth having on any 1m variant
(which otherwise runs 40–65% drawdowns); it does nothing on 15m.

---

## 8. Running scoreboard

| Idea | Verdict | Evidence |
|---|---|---|
| Support-break short, 1m | ✗ beta only | +294% in a −46% year, −43% in a +26% one |
| Long mirror (resistance-break) | ✗ worse than short | 18 long variants occupy the bottom of sweep 1 |
| Bail-out rules as sole exit | ✗ dies on costs | +0.025R at 0 cost → −0.007R at 0.7 bp |
| Hour-of-day filters | ✗ no persistence | quarter-to-quarter rank rho −0.03 |
| Weekday filters | ✗ fails OOS | trained pick ranked 110/127 on test data |
| Higher TF for cost relief | ✗ hypothesis wrong | median expR *falls* 1m→15m |
| Higher TF for drawdown | ✓ real | median max DD 64% → 32% → 14% |
| Trailing exits | ~ tail-driven | med_R = −1.0; one +202R trade carried a window |
| Bounded-R target on 15m (`v1.0/high_of_n/rr2`) | ? promising, unproven | train +0.21 → test +0.25 expR, DD 11%, cost-insensitive — but parameter *spikes* and only 161 trades |
| Daily loss cap | ✓ for drawdown only | −DD in every window; expectancy inconsistent |

**Nothing here yet justifies high leverage.** The one promising candidate needs far more trades and a
parameter plateau before it could be sized.

---

## 9. ⛔ The 15m candidate is OVERFIT — verdict downgraded

§5 ran parameter sensitivity on the *mean* across quarters, which was too forgiving. Re-run on
**`expR_min`** (the worst of the three regimes — the metric that actually decides), the picture is
unambiguous:

```
tp_rr            1.0: -0.0274   1.5: -0.0199   2.0: +0.0823   2.5: +0.0087   3.0: -0.2681
stop_n            10: -0.0496    15: -0.1443    20: +0.0823    30: -0.0355    40: +0.0087
grad_max        None: -0.1887   0.0: +0.0823  -0.02: -0.1879  -0.05: -0.2204
sr_min_touches     1: +0.0823      2: -0.0356      3: +0.0449
```

**The chosen config is the only positive cell on essentially every axis, and every neighbour is
negative.** `stop_n` 15 → 20 swings −0.144 → +0.082; `grad_max` −0.02 → 0.0 swings −0.188 → +0.082.
That is a single lucky cell in a 4-D grid, not an edge with a basin around it.

Also: **1 of 36 configs survives the strict test, where chance alone predicts ~4** (`36 × 0.5³`). The
search is producing *fewer* survivors than random, which is what you would expect if there is no edge
anywhere in this family.

The clean train→test result (+0.21 → +0.25) does not rescue it: with 161 trades and a spike-shaped
parameter surface, one hold-out passing is entirely consistent with luck.

**Verdict: no validated edge found in the support/resistance-break family, on any timeframe, either
direction.** This is consistent with the earlier direct measurement that the entry has no predictive
power once drift is removed.

### What I would try next (not yet tested)

The family itself looks exhausted. Directions that are genuinely different rather than more parameter
tuning:

1. **Different signal class entirely** — the breaks are momentum-continuation bets. Mean-reversion at
   S/R (fade the break) is the untested inverse and costs nothing to run through the same harness.
2. **Volatility-regime conditioning** — the efficiency-ratio work in `analytics/timing.py` showed BTC
   1m is choppy at *all* hours; a filter on *realised volatility* rather than clock time is untried.
3. **Squeeze / WaveTrend as the primary trigger** — both are ported and charted but have never been
   the entry, only decoration.
4. **A bull-heavy window set** — the current 1-bull / 2-bear split structurally penalises longs, so
   the long mirror has not had a fair test.

---

# PHASE 1 — Reference levels: do they predict anything at all?

New primitives built (`quant/indicators/levels.py`, all causal, all tested): previous-day
high/low/close/mid, previous-week high/low, session structure (Asia/EU/US blocks, running extremes,
previous-session extremes, opening range), session/day VWAP with σ bands, **volume features** (never
used anywhere in this project before), and round numbers. Plus `level_events()`, which classifies each
bar's interaction with any level as **break / fade / sweep / retest**.

Screen: at every interaction, forward return over {5,15,30,60,120} bars in ATR units, in the direction
a trader would actually take, across 3 regime windows × 6 symbol/timeframe combinations
(BTC 1m/5m/15m, XAUUSD 1m/5m, PAXGUSDT 5m).

## 1a. First attempt — whole-window drift adjustment → BIASED

Pooling by interaction type gave a suspiciously clean picture: **every short cell positive, every long
cell negative**, across all four interaction types (`sweep short` 42.4% consistent, med_h60 +0.118 vs
`sweep long` 9.3%, med_h60 −0.115).

That is not a level effect. A genuine mean-reversion effect must be **symmetric** — if sweeps of highs
revert, so must sweeps of lows. A clean long/short split is residual short bias: level events cluster
in trending stretches, so the short-side event set inherits the drift of the days it lands on, which a
single whole-window mean does not remove.

## 1b. Second attempt — day-matched controls → ALSO BIASED

Compare each event with the mean forward return of **all bars on the same calendar day**. This flipped
the result and looked excellent: breaks and retests strongly negative on both sides, fade and sweep
positive on both sides, 20/50 mirrored pairs positive against a chance rate of ~12.

## 1c. ⚠️ The null control that killed both — `experiments/null_control.py`

The day-mean control contains bars from *after* the event, so a signal firing near a day's extremes
shows reversion **by construction**: within a finite path, points at the top of the day's range are
followed by lower prices whether or not the market actually reverts.

Test: rebuild the series by **shuffling bars within each calendar day**. Each day keeps its exact
drift, volatility, candle shapes and volume — only the *order* is destroyed, which is where any level
effect must live. Levels on that series cannot predict anything. Run the identical screen:

```
      event   bet  real_h60  null_h60  excess_h60
       fade  long    0.0595    0.2685     -0.2090   <- the null "edge" is 4.5x the real one
      sweep short    0.2194    0.1051      0.1144
       fade short    0.0674    0.1232     -0.0558
      sweep  long    0.1345    0.1975     -0.0630
```

**The shuffled data — which has no structure at all — produces the same "edges", and on `fade long` a
much bigger one.** Both screens were measuring their own methodology.

## 1d. The screen done correctly — real minus its own null

`experiments/phase1c_null_differenced.py`: identical pipeline on the real series and on 3 within-day
shuffles; the reported edge is `real − mean(null)`.

```
      event   bet  exc_h15  exc_h30  exc_h60  exc_h120  pct_pos_h30  pct_pos_h60
      sweep  long  -0.0151   0.0034   0.0307    0.0897         44.4         53.7
      sweep short  -0.0183  -0.0002  -0.0111   -0.0187         45.5         43.9
first_break short   0.0114  -0.0006  -0.0490   -0.1156         33.3         24.1
       fade short  -0.0074  -0.0178   0.0068   -0.0352         47.0         51.5
       fade  long  -0.0152  -0.0211   0.0201    0.0774         40.7         55.6
     retest  long  -0.0014  -0.0242  -0.0138   -0.0276         31.8         33.3
```

Everything collapses to ±0.02–0.05 ATR with hit rates near 50% — noise.

- **Symmetry test: 10/50 mirrored pairs positive on both sides, against a chance rate of ~12.**
  *Below* chance.
- **Individual cells: 2 survivors out of 320** (positive excess in ≥5 of 6 runs with |z|>2), where
  chance predicts ~16. Again below chance.

### Verdict — Phase 1

**No reference level carries an exploitable directional edge on BTC or gold, at any of the tested
timeframes, for any of break / fade / sweep / retest.** This covers everything requested: prev-day
high/low, session high/low, weekly, opening range, S/R zones, VWAP bands and round numbers. The
"fade the break" hypothesis from §9 above is included and is **not** supported.

### Methodological rule now in force

> Any screen based on forward returns must be differenced against its own within-day-shuffled null.
> Two independent, plausible-looking control schemes both manufactured edges of exactly the size we
> were hunting for. A positive number means nothing until the same pipeline has been shown to produce
> zero on data with no structure.

---

# PHASE 2 — Every entry family as the PRIMARY trigger

Built `quant/scenarios/entries.py`: nine entry families, each registered **long and short** (a
one-sided result is a directional bet, not an edge), all runnable through the same engine, stops,
targets, costs and regime windows.

| Family | Trigger |
|---|---|
| `level_event` | break / fade / sweep / retest of any reference level (prev-day, session, opening range, VWAP bands, S/R zones) — covers "level fade", "liquidity sweep" and "opening-range break" in one parameterised trigger |
| `bos` / `choch` | break of the last **adaptive zigzag** swing, with the trend or against it |
| `vol_thrust` | volume z-score spike with a directional candle — **volume used for the first time** |
| `squeeze` | squeeze fires off, traded with the momentum histogram |
| `wt_cross` | WaveTrend cross inside the OB/OS band |
| `rsi_div` | RSI divergence, confirmed or the zero-lag provisional variant |
| `vwap_rev` | price beyond N σ of session VWAP, turning back |
| `grad_extreme` | EMA gradient beyond its own trailing quantile |
| **`time_entry`** | **NULL BENCHMARK — enter every N bars, no signal at all** |

## The control that makes this readable

`time_entry` runs identical stops, targets, costs and windows with **no information whatsoever**. Any
entry that cannot beat a metronome contributed nothing — whatever it scores belongs to the exit rules.
This is precisely the mistake that made ~150 earlier configs look interesting: they varied exits only.

## Result — 176 configs × 3 regime windows

```
NULL BENCHMARK (no signal, fixed schedule): best expR_min = -0.0445, best expR_mean = -0.0351

family  configs  best_min  best_mean  med_min  n_pos_all_windows  beats_null_min
 choch       12    0.0023     0.0310  -0.0568                  1            True
 level       80   -0.0019     0.0716  -0.0670                  0            True
 vwapz       12   -0.0156     0.0264  -0.0408                  0            True
   div        8   -0.0217     0.0027  -0.0528                  0            True
    wt       12   -0.0356    -0.0034  -0.0652                  0            True
  NULL       12   -0.0445    -0.0351  -0.0843                  0           False
   vol       12   -0.0521    -0.0420  -0.1209                  0           False
   sqz        8   -0.0671    -0.0498  -0.0987                  0           False
   bos       12   -0.0795    -0.0415  -0.1165                  0           False
  grad        8   -0.0855    -0.0623  -0.1312                  0           False
```

**Positive in all 3 regimes with ≥300 trades: 1 of 176. Chance alone predicts ~22.**

The single survivor (`choch/atr4.0/long`, expR_min **+0.0023**) is zero to three decimal places, and
its `expR_ex_top_min` is **−0.0000** — remove each window's best trade and the edge is gone. Max
drawdown 38.5%, worst-window return −6.2%.

### What the numbers say

1. **No entry family beats random timing by a meaningful margin.** Four families clear the null, but
   only by clawing back cost drag — `level` reaches −0.0019 versus the null's −0.0445, i.e. it
   recovers the spread and adds nothing.
2. **`medR_min ≈ −1.03` for every single family.** The median trade is a full stop-out everywhere.
   That is the signature of no directional information plus costs.
3. **Volume, squeeze, gradient and plain BOS are *worse* than the metronome** — they select *more*
   costly moments, not better ones.
4. Survivor count below chance, for the third time in this project. A search that keeps producing
   fewer survivors than random is telling you the space is empty, not that you need a finer grid.

### Verdict — Phase 2

**None of the nine entry families, in either direction, is tradable on BTC 1m after costs.** That
covers every "new method" on the plan: level fades, liquidity sweeps, opening-range breaks, structure
breaks (BOS *and* CHoCH), volume confirmation, squeeze release, WaveTrend, RSI divergence (confirmed
*and* zero-lag) and VWAP reversion.

---

# PHASE 3 — Management (incl. the TIME STOP), and the trap it walked into

**New engine feature:** `BacktestConfig.max_bars_in_trade` — close any trade still open N bars after
entry, at that bar's close (`time_stop`). Checked after SL/TP and exit signals, so a trade closing for
a real reason keeps that reason; at `0` the block is skipped and the path is byte-identical (golden
tests pass). Also new: `experiments/mgmt_sweep.py::EntryBundle`, which prepares an entry's indicators
once and re-applies only the stop/target per combination — byte-identical to `run_pieces`, ~9× faster.

10 entries × 150 management combos (5 stops × 6 targets × 5 time stops) × 3 regimes.

## What management actually did

```
by target (median expR_min)          by stop (median expR_min)     time stop (median expR_min)
  trailing0.5   -0.0198               pct0.5   -0.0179              0    -0.0448
  rr3           -0.0216               pct0.2   -0.0286              15   -0.0268
  rr_partials   -0.0314               atr2.5   -0.0328              30   -0.0272
  rr2           -0.0321               atr1.5   -0.0480              60   -0.0344
  rr1.5         -0.0402               atr1     -0.0652              120  -0.0356
  rr1           -0.0548
```

**The time stop helps, mildly and universally** (−0.045 → −0.027 at 15–30 bars) — and it helps the
null entry by about as much, so it is cutting cost-weighted dead time, not finding better exits. It
also cuts median drawdown 62.9% → 55.8%. Worth keeping; not an edge.

Then: **60 of 1200 configs are positive in all three regimes**, and *every one of them uses a trailing
stop or rr3*. Best: `choch_long|atr1/trailing0.5/t60` at +0.171R, `pdlow_fade_long|atr1/trailing0.5/t30`
at +0.136R with a **+115% worst-window return and 32% max drawdown**.

That looked like the first real finding in this project.

## ⛔ PHASE 6 — the gates say it is a lottery

A 1×ATR initial stop with a 0.5% trail on BTC 1m means risking ~1R (≈37 price units) to chase a
~13R runner. The payoff is deliberately fat-tailed, so the usual gates stop discriminating.

### Where the profit comes from

```
cfg                    win     n   expR  win%  med_R  max_R  top1_share  top1pct_share
pdlow_fade_long/t30  bear25  598  0.172  23.2  -1.03   17.7        17.3           70.9
choch_long/t60       bear25 1069  0.245  15.3  -1.06   66.4        25.4          138.2
rtdiv_short/t120     bear26 2331  0.079  15.1  -1.07   48.7        26.5          312.2
NULL_short/t60       bear25 1318  0.046  15.6  -1.06   32.2        53.2          476.2
```

`top1pct_share` above 100% means everything outside the top 1% of trades sums to a **net loss**.

### Gate B — TRIM-TOP-K (worst window)

```
cfg                   full   drop1   drop5  drop10  drop25  drop1%  drop5%
pdlow_fade_long/t30  0.136   0.117   0.050  -0.038  -0.262   0.032  -0.313
pdlow_fade_long/t15  0.134   0.118   0.046  -0.027  -0.193   0.030  -0.256
choch_long/t60       0.171   0.127   0.032  -0.094  -0.309  -0.094  -0.578
vwapz2_short/t60     0.098   0.084   0.032  -0.014  -0.123  -0.091  -0.485
rtdiv_short/t120     0.079   0.058  -0.009  -0.066  -0.182  -0.169  -0.634
NULL_short/t60       0.040   0.019  -0.059  -0.133  -0.305  -0.194  -0.640
```

**Deleting the ten best trades out of 600–2300 turns every single candidate negative.**

### Gate C — spread sensitivity

```
cfg                      x1      x2      x4
pdlow_fade_long/t30  0.1363  0.0336 -0.1390
choch_long/t60       0.1710 -0.0125 -0.2847
vwapz2_short/t60     0.0984 -0.0394 -0.2730
NULL_short/t60       0.0397 -0.1198 -0.3646
```

At 1.4 bp (still generous for retail crypto) only `pdlow_fade_long` is still alive, barely.

### Gate E — rotating out-of-sample: **the gate itself fails**

Every candidate passes (all held-out windows positive) — **and so does the metronome**
(`NULL_short/t60`: +0.040 / +0.046 / +0.044, all positive). A random-entry rule with a trailing stop
passes rotating OOS. On this payoff shape, that test has no discriminating power at all.

### What the trailing stop is really doing

The trailing stop is not finding good entries; it is harvesting BTC's fat-tailed trend structure with
whatever entries you feed it. That is why the **random** entry also turns positive (+0.040R) the
moment it is given a trailing stop, and why the ranking of families barely changes.

**Verdict:** the Phase-3 "survivors" are trend-capture on a fat-tailed asset, carried by <1% of
trades, indistinguishable from a random-entry version at any useful resolution, and dead at 2× spread.
`pdlow_fade_long` is the only one that clears the null by a real margin (+0.136 vs +0.040) and
survives 2× spread — it is the single thing in this whole search that is not obviously nothing, and it
is still uninvestable at leverage: **its worst-window drawdowns run 26–32%, its median trade is a full
stop-out, and six trades make its year.**

---

# PHASE 5 — Timeframe and a SECOND INSTRUMENT (the structural test)

## The cost floor — why 1-minute BTC was never going to work

Spread as a percentage of one R (1.5 × ATR stop, median ATR over 2026 H1):

| | 1m | 5m | 15m | 1h | 4h |
|---|---|---|---|---|---|
| **BTCUSDT** (spread 6.0) | **10.8%** | 3.8% | 2.0% | 0.9% | 0.4% |
| **XAUUSD** (spread 0.25) | **7.0%** | 2.9% | 1.6% | 0.8% | 0.4% |

On BTC 1m an entry must generate **more than 0.108R of raw edge** before it pays for the spread
alone. That single number explains the null benchmark's −0.045R and why every family in Phase 2 sat
just below zero. It is also why "quick trades on the 1-minute chart" is the hardest possible version
of this problem, not the easiest.

## Does the cost relief convert into edge? No.

9 entries + 2 nulls × 12 management combos × 3 regimes × {5m, 15m, 1h} × {BTC, gold}.

```
median expR_min          real     NULL   real - NULL
BTCUSDT  5min          -0.0625  -0.0596      -0.0028
BTCUSDT  15min         -0.1267  -0.0828      -0.0439
BTCUSDT  1h            -0.1884  -0.0901      -0.0984
XAUUSD   5min          -0.1848  -0.1327      -0.0521
XAUUSD   15min         -0.2695  -0.1354      -0.1342
XAUUSD   1h            -0.3993  -0.2262      -0.1730
```

**`real − NULL` is negative in every cell and gets WORSE at higher timeframes.** Cheaper trading did
not reveal a hidden edge; it just made the metronome cheaper too. Gold is uniformly negative.

## ⛔ The cross-instrument gate — the single most damning number in this project

> **0 of 324** entry × management × timeframe combinations are positive on **both** BTC and gold.
> Chance alone predicts ~81.

Not "few". **Zero.** Whatever the BTC numbers are measuring does not exist in another instrument.

The best-looking config anywhere, `wt53_short|atr2.5/rr1.5` on **BTC 15m**, is instructive: +0.074 /
+0.148 / +0.077 across all three regimes, 509 trades, 17.3% drawdown, bounded rr target (not a
trailing stop), and it survives removing the best trade. It looks like the real thing. Then:

* the **same config on gold 15m**: −0.129 / −0.167 / +0.140;
* ~170 trades per window with sd(R) ≈ 1.2 gives a standard error of ≈ 0.09 per window — **+0.074 is
  less than one standard error from zero**;
* "positive in all 3 windows" is a p = 1/8 event, and 648 configs were tested.

Survivors overall: **18 of 648, where chance predicts ~81.** Below chance again.

---

# PHASE 6F — Bootstrap: is any of it distinguishable from zero?

20,000 resamples, IID and moving-block (block = 25 trades, so clustered winners can't be broken up).

```
per-window 5th-percentile lower bound on expR (worst window shown)
cfg                  min_lo5  min_lo1  max_p_neg  survives
choch_long/t60       -0.0097  -0.0974     0.0595     False
pdlow_fade_long/t30  -0.0115  -0.0837     0.0616     False
vwapz2_short/t60     -0.0146  -0.0553     0.0743     False
pdlow_fade_long/t15  -0.0125  -0.0693     0.0654     False
rtdiv_short/t120     -0.0516  -0.1011     0.1648     False
NULL_short/t60       -0.1163  -0.1770     0.3411     False   <- control
NULL_long/t30        -0.2118  -0.2573     0.9120     False   <- control
```

**Not one candidate holds a lower bound above zero in every window.** Pooling all three windows makes
them look decisive (`p_neg` ≈ 0.0001) — which is exactly the mistake this harness exists to prevent,
since pooling lets one good regime carry the others.

The candidates *are* clearly better than the metronome (`p_neg` 0.06–0.16 versus 0.21–0.67). "Better
than random" is real. It is just not the same claim as "positive after costs with any confidence".

---

# PHASE 4 — Hybrid / conditional strategies: does any regime rescue a dead entry?

10 entries (incl. both nulls) × 19 conditioners × 2 management combos × 3 regimes = 380 configs.
Conditioners: volatility terciles, trend/chop by Kaufman efficiency ratio, volatility-shock windows
(the announcement proxy — no calendar is available), position in the recent range, the three sessions,
and 5-minute EMA bias with *and* against.

The trap this has to avoid: almost any filter improves a losing rule, because cutting trades cuts cost
drag. So each conditioner is applied to the real entries **and to the metronome**, and what counts is
the real lift MINUS the null's lift under the same filter.

## Raw lift (median over entries and management)

```
cond                med_lift  med_expR  med_trades      cond            med_lift  med_expR
vol_high             +0.0151   -0.0418        1616      vol_mid          -0.0380   -0.0741
no_shock60           +0.0033   -0.0345        3874      sess_eu          -0.0628   -0.1200
htf_against          -0.0039   -0.0434        2124      range_upper      -0.0867   -0.1435
vol_high+trending    -0.0072   -0.0545         736      post_shock60     -0.0888   -0.1245
trending             -0.0105   -0.0456        1841      sess_us          -0.1173   -0.1543
choppy               -0.0186   -0.0760         761      vol_low+choppy   -0.1196   -0.1752
sess_asia            -0.0244   -0.0786         260      shock_now        -0.2280   -0.2584
htf_with             -0.0317   -0.0824        1738
```

**Only `vol_high` (+0.015) and `no_shock60` (+0.003) improve anything at all.** Every session filter,
both HTF-bias directions, both range positions and both trend/chop states make things worse.

## Null-adjusted lift — the honest version

```
cond                real_lift  null_lift      net
htf_against           +0.0037    -0.0543  +0.0580
no_shock+trending     -0.0006    -0.0528  +0.0522
trending              -0.0045    -0.0402  +0.0356
...
shock_now             -0.2596    -0.0977  -0.1618
```

Note what the top rows actually say: `real_lift ≈ 0`. These conditioners score a positive *net* only
because they damage the metronome more than they damage the signal — they preserve what little the
entry has, they do not add to it.

**Survivors: 3 of 304, where chance predicts ~38.** Eight times below chance, for the fifth time.

And the decisive detail: **`NULL_short@sess_us|atr1.5/rr2` scores +0.0701, positive in all three
regimes.** The metronome, filtered to the US session, clears the same bar the "hybrid strategies" do.
That is why the null ceiling is 0.0701, and why only three configs are left above it — all of them
with 100–150 trades per window, where the standard error on expR is ≈0.1 and +0.09 is under one sigma.

## ✓ The one genuinely actionable finding in Phase 4

**Trading INTO a volatility shock is reliably catastrophic: −0.2596R real lift, the worst number in
the entire search.** `post_shock60` (the hour after) is also strongly negative at −0.0888. Since
`vol_shock` is the announcement proxy, this is the empirical version of "don't trade the release, and
don't chase the hour after it." It is a negative rule rather than an edge, but it is robust, large,
and consistent across every entry family — the only conditioner result big enough to act on.

---

# PHASE 6R — The risk layer: can a loss cap or lockout add anything?

Daily loss caps, max-trades-per-day and consecutive-loss lockouts all rest on one assumption: **that
losses cluster**. If trade outcomes are independent, a lockout cannot improve expectancy — it only
removes exposure, scaling wins and losses down together. That is testable, and testing it is more
honest than sweeping lockout lengths and keeping whichever looked best.

## Do losses cluster? No.

```
entry              trades  p_loss  p_loss_after_loss  excess     acf1     acf2     acf3     acf5
orlow_break_short    6371  0.6481             0.6574 +0.0093  +0.0287  -0.0111  +0.0162  -0.0086
choch_long           2768  0.6354             0.6384 +0.0030  +0.0046  +0.0089  +0.0157  -0.0142
NULL_long            4125  0.6707             0.6722 +0.0014  +0.0054  +0.0433  -0.0133  -0.0069
rtdiv_short          7177  0.6465             0.6443 -0.0022  -0.0057  -0.0121  -0.0061  -0.0053
wt53_long           10174  0.6587             0.6554 -0.0033  -0.0131  +0.0003  -0.0099  -0.0102
pdlow_fade_long      1821  0.6371             0.6260 -0.0111  -0.0292  +0.0030  -0.0231  -0.0103
```

Standard error on each autocorrelation ≈ **0.027**. Every `acf` sits inside that band and `excess`
(the extra chance of a loss right after a loss) is at most ±0.011 — and it changes sign across
entries, which is what pure noise looks like.

**Trade outcomes are independent. Consecutive-loss lockouts are therefore ruled out analytically, not
by sweeping them.** Losing streaks in this data are the ordinary consequence of a 65% loss rate, not
a regime you can detect and sit out.

## What the risk controls actually do

```
                dd_max  expR_min  ret_min          drawdown by entry:  none -> risk_0.25%
daily_cap_1%    45.55    -0.0571   -26.45          choch_long          38.5 -> 11.2
daily_cap_2%    50.02    -0.0291   -33.49          orlow_break_short   63.5 -> 20.9
daily_cap_4%    53.27    -0.0159   -30.87          vwapz2_long         79.9 -> 31.3
max_1_open      61.83    -0.0230   -43.05          pdlow_fade_long     25.1 ->  7.1
none            61.83    -0.0230   -43.05
risk_0.25%      20.29    -0.0230   -10.50   <- identical expectancy, one third the drawdown
risk_0.5%       37.30    -0.0230   -21.63
```

* **Daily loss caps make expectancy WORSE** (−0.023 → −0.057 at a 1% cap) and do not reliably cut
  drawdown — `orlow_break_short` got *worse*, 63.5% → 75.1%, because blocking new entries changes
  which trades exist rather than removing bad ones.
* **`max_open_trades=1` changes nothing** — the setups rarely overlap anyway.
* **Position size is the only control that works, and it works linearly**: 1% → 0.25% risk leaves
  `expR` byte-identical and divides drawdown and return by the same factor. Exactly what theory
  predicts for independent outcomes, and it buys nothing you couldn't get by trading smaller.

**Verdict: there is no risk-management trick here. With independent trade outcomes the only lever is
size, and size trades return for drawdown one-for-one.**

---

# FINAL SCOREBOARD

| Phase | What was tested | Survivors | Chance predicts |
|---|---|---|---|
| 1 | 8 level types × 4 interactions, null-differenced | **2 / 320 cells** | ~16 |
| 2 | 9 entry families, long and short | **1 / 176** | ~22 |
| 3 | 150 management combos | 60 / 1200 — **all fail trim-top-K** | ~150 |
| 4 | 19 regime conditioners (hybrids) | **3 / 304** | ~38 |
| 5 | 3 timeframes × 2 instruments | **18 / 648** | ~81 |
| 5 | **positive on BTC *and* gold** | **0 / 324** | ~81 |
| 6 | bootstrap lower bound > 0 per window | **0 / 7 candidates** | — |

Six independent counts, five of them **below** chance. A search that keeps returning fewer survivors
than random selection would is not under-tuned; it is describing an empty space.

---

# PHASE 7 — The volatility-compression theory ("quiet range, then a big move off the 200-EMA")

Proposed from a chart of **Saturday 2026-08-01**: BTC ranged all day in a very tight band, broke below
the 1-minute 200-EMA around 15:00–16:00 UTC and then dropped hard. Hypothesis: compression predicts a
big move, the EMA break gives the direction, and the trigger may be the US open on weekdays.

New primitives: `add_compression` (`quant/indicators/volatility.py`) — range ratio, ATR ratio,
generalised NR7 rank, Bollinger width, and a trailing PERCENTILE RANK of the ATR ratio (`cmp_pct`),
which is the one to condition on because an absolute volatility cut silently becomes a date filter.
New scenario family `compression_break_{long,short}@v1` with `cb_quiet_q = 1.0` disabling the filter —
that switch is the control the whole phase turns on.

## The instance itself

Quantified, the chart is real: ATR(14) fell to **1.9** by hour 13 and reached **39.0** by hour 18, a
20x expansion, with `cmp_pct` in the bottom 1–13% of its own history through hours 0–9.

But ranked against history it is not a big move at all:

```
                       2026-08-01    median day
daily range              139 bps       297 bps     -> 9th PERCENTILE of 578 days
max 120-bar travel       121 bps        85 bps     -> 69th percentile
```

**It was one of the quietest days in the sample.** The visual drama comes from the price axis
auto-scaling to a ~175 bps window, in which a 120 bps move fills the screen.

## The theory has four claims. Three are false.

**1. "Quiet then big move" — FALSE in absolute terms.** Forward 120-bar travel by compression decile:

```
decile 0 (quietest)  57.1 bps          decile 9 (loudest)  103.1 bps
```

Volatility is PERSISTENT, not mean-reverting. In ATR units the same data reads 19.14 vs 10.71 — a
"1.79x expansion after quiet" — but that is the **denominator shrinking**, the identical trap that
killed the Phase-1 screens. Net of the shuffled-null control the range-relative effect is **+4 to
+14%**, small but not zero.

**2. "The longer it coils, the bigger the spring" — FALSE.** Over 3,622 compression episodes:

```
wait to release   <30   30-60  60-120  120-240  240-480  480+
travel (bps)     78.5    84.3    85.5     84.9     81.7   85.8
```

Flat. A ten-hour compression releases no harder than a thirty-minute one.

**3. The 200-EMA cross carries NO direction.** Releases resolve upward **47.8%** of the time; hit
rates 46–53%, split along long/short lines — the residual short-bias artifact, not signal.

**4. "The move comes at the US open" — TRUE, with a caveat.**

```
release hour (UTC)   12:00   13:00   14:00   typical other hour
share of releases    13.3%   16.1%    5.4%          ~4%
median travel        111     125     121 bps       60-90 bps
```

**35% of releases land in 12:00–14:00 UTC** (US cash open 13:30) and are ~40% bigger. But the weekend
half is backwards: of 826 weekend compressions, **99.8% released while it was still the weekend** (2
of 826 waited for Monday), and weekend releases are *smaller* — Sat 47 bps, Sun 64 bps, versus
85–120 bps on weekdays.

## The strategy: the compression filter SUBTRACTS value

100 entry variants (2 triggers x breakout/fade x 4 quiet thresholds incl. OFF x 3 range lengths x
both sides) x 12 management combos x 3 regimes = 1,200 configs.

**The paired control — identical breakout, filter ON (q=0.2) vs OFF, same management, 288 pairs:**

```
compression HELPED in 29 of 288 pairs (10.1%)      median delta  -0.0452
```

It makes things worse nine times out of ten, by 4.5 percentage points of R. Median `expR_min` by
threshold confirms it — every filtered variant is below the unfiltered one:

```
q            ALL      0.35     0.2      0.1
breakout   -0.0670  -0.0857  -0.1116  -0.1334
fade       -0.0291  -0.0556  -0.0842  -0.0677
```

The best configs are all `rangeF` — **FADING the range break**, i.e. the opposite of the proposed
trade — and they are the familiar trailing-stop shape (`medR_min` -1.06, drawdowns 50–94%).
Trim-top-K finishes them:

```
cfg                                 n_min   full   drop5  drop10  drop5%  top1pct_share
rangeF_s120_q0.1_short                351  0.1768 -0.1049 -0.2775 -0.6772          99.2
rangeF_s120_q0.35_short               954  0.1738  0.0422 -0.0608 -0.6739         101.4
rangeF_s120_ALL_short (NO filter)    2490  0.1116  0.0529  0.0073 -0.6445         152.7
NULL_short (metronome)               1041 -0.0039 -0.0713 -0.1270 -0.3672        5445.8
```

Deleting **five** trades out of 351 kills the best filtered config, whose top 1% of trades supply
**99.2%** of its profit. Note also that the UNFILTERED version is the more robust of the two — more
evidence the compression condition subtracts rather than adds.

**Survivors: 49 of 1,152, where chance predicts ~144.** Below chance, for the seventh time.

## Verdict — Phase 7

**The compression breakout is not tradable, and the compression filter is actively harmful.** The one
true part of the theory — releases cluster at the US open and are larger there — is a session-volume
effect that applies to weekday compressions, not to the weekend setup that prompted it. And the chart
that started it was a 9th-percentile day whose apparent violence was an artifact of axis scaling.

### Engine / harness notes from this phase

* `Scenario.prepare_keys` declares which params change `prepare()`'s output, so a sweep shares one
  prepared frame across variants that differ only in trigger or threshold (100 prepares to 10 here).
* **A bug worth recording:** caching the whole `EntryBundle` under that key made every variant run
  with the FIRST variant's trigger and threshold. The sweep returned 288 pairs with deltas of
  *exactly* 0.0000 — which reads as a clean negative result, not as a failure. Per-variant entry
  params now pass through `run()`, prepare-affecting overrides raise, and
  `test_sweep_shares_frames_without_leaking_params` locks it. Tests 234 to 239.

---

# PHASE 8 — Can the one survivor be improved? S/R zones, EMA bias, HH/HL structure, faster TFs

`wt_cross_short` on BTC 15m was the best result of the search. Everything that might improve it had
only ever been tested as a STANDALONE ENTRY, never as a gate on top of it. This phase closes that gap:
base entry x 17 conditioners x 3 timeframes x 3 management combos, with the metronome carrying every
conditioner so a filter that merely removes costly bars can be told apart from one that finds good ones.

## 1. Faster timeframes make it WORSE, not better

Unconditioned base, `atr2.5/rr1.5`:

```
tf     expR_min   trades
15min   +0.0647      513
 5min   -0.0631     1534
 1min   -0.0555     8693
```

This is the cost floor doing exactly what it must: spread is 2.0% of R at 15m and 10.8% at 1m. The
strategy is not "better if you trade it faster" — 15m is the slowest of the three and the only
positive one.

**On 1m every single conditioner is negative** (best five: −0.027 to −0.051). There is nothing to
improve there.

## 2. On 15m, every filter that keeps enough trades makes it worse

Base = 513 trades, +0.0647. Sorted by how many trades survive the filter:

```
cond                    trades   expR_min   ex_top
none                       513    +0.0647   +0.0549   <- the unfiltered base
no_shock                   487    +0.0391   +0.0310
reject_resistance          460    +0.0439   +0.0333
above_ema200               444    +0.0382   +0.0299
at_resistance_0.25         418    +0.0365   +0.0249
trending                   378    +0.0269   +0.0165
struct_up                  294    -0.0492   -0.0658
at_resistance_0.1          218    +0.1088   +0.0871
vol_high                   182    +0.0426   +0.0079
far_from_support           142    -0.0718   -0.0989
struct_down                 80    -0.1263   -0.1715
below_ema200                71    +0.1169   +0.0444
resist+below_ema            62    +0.2882   +0.2411
resist+struct_down          60    -0.3053   -0.3804
struct_down+below_ema       18    -0.1710   -0.3824
resist+struct+ema           14    -0.5047   -1.0052
below_ema50                 11    -1.0078   -1.0090
```

**Every filter retaining 294+ trades scores BELOW the unfiltered base.** Every filter that looks
better does so on 62–218 trades. The apparent edge and the sample size move in opposite directions
almost monotonically — which is the signature of noise, not of a filter that works: cut any sample
hard enough and some subset looks excellent by chance.

The `below_ema50` row is the reductio: **7 trades on 5m, expR +1.49**. It is not a strategy, it is
seven coin flips.

## 3. The decisive count

Configs beating the unfiltered 15m base **with >=300 trades, positive in all three regimes, and still
positive after removing each window's best trade: 1 of 153.**

That one is `at_resistance_0.25 | atr2.5/rr2` — 402 trades, +0.0711 against the base's +0.0647. An
improvement of **+0.006R**, against a per-window standard error of roughly 0.09. It is indistinguishable
from the base, which is itself indistinguishable from zero.

## 4. What the ideas DO show

Fairly stated, they are not worthless — they are just far too weak to survive the sample cost.
Net-of-null lift is positive and *consistent across timeframes* for two of them:

```
                  1min      5min     15min
at_resistance   +0.058    +0.072    +0.206
below_ema200    -0.049    +0.223    +0.033
```

`at_resistance` points the right way on all three timeframes, and `resist+below_ema` is positive on
both 5m and 15m. So "short into resistance, below the 200-EMA" is directionally the right instinct.
It simply does not produce enough effect per trade to pay for the trades it discards.

## Verdict — Phase 8

**No. S/R zones, EMA bias and HH/HL structure do not improve the strategy, and shorter timeframes make
it worse.** The single best filtered configuration adds +0.006R on 402 trades. Everything that looks
like a real improvement is running on 11–71 trades.

The general lesson, which now applies to every future "add a filter" idea here: **a filter must be
judged at constant trade count, or the sample-size effect will always manufacture a winner.** The 15m
base only produces ~170 trades per regime window; there is not enough data at that timeframe to
support any conditioning at all, and at the timeframes with enough data (1m, 5m) the cost floor has
already eaten the edge.
