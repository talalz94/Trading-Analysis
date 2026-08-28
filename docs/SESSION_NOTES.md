# Session Notes / Handoff

> Point a fresh session or model here for the **non-obvious context** — research findings, gotchas,
> recently-fixed bugs, and open threads — that isn't derivable from the code alone. Pair with
> [`../CLAUDE.md`](../CLAUDE.md) (architecture) and [`../README.md`](../README.md) (usage).
> Keep this current; it is the running memory of the project.

## Current state (2026-07)
- Repo: https://github.com/talalz94/Trading-Analysis · branch `main`. **121 tests pass** (`pytest tests/`).

## Parked ideas — DO NOT ACTION YET (user will return to these)
> Captured 2026-07-27 at the user's request ("store these, don't work on them yet"). Three threads.

**1 — Manually backtrack & refine the support-break scenario/configs against a specific BTC window.**
Review `support_break_short@*` + the stop/TP configs (see the 2026-07-26 entry below) by eye, walk the
trades bar-by-bar, and refine them for the reference chart the user sent:
- **Reference chart:** BTCUSD **Coinbase, 1-minute**, window ≈ **13:30 → 20:15** (tz unstated on the
  screenshot), price band ≈ **64,480–65,520**, last price ~65,151. Shape = a long **range ~64,600–64,720
  until ≈17:15**, then a **strong impulsive up-swing 17:15 → 19:15** to ~65,500, then a **rollover/pullback
  into 20:15**. (This is a LONG-swing chart — note our scenario is short-only, so refining here may mean
  building the long mirror or judging where a short would/wouldn't have fired.)
- Chart overlays present (for context / possible replication): **Colored EMA/SMA (9, 200)**,
  **MA Shift [ChartPrime] (EMA, 40, hl2, 15, 0.5)** as the bottom histogram, plus manual arrows /
  trendlines / horizontal rays / S-R text.
- Goal: does the setup fire where a human would draw it? Tighten entry/stop/TP params to match.

**2 — Adaptive "when to get out of a caught swing" exit (trade management).** Once a good swing is
caught, choose the exit *adaptively*:
- pick the **trail EMA by the steepness / ROC of price (or of momentum)** — steeper move → faster EMA,
  shallower → slower EMA (ride longer);
- and/or **exit after N opposite-colour candles**, allowing a **larger % give-back the bigger the
  favourable move was** (asymmetric pullback tolerance scaled to the move captured).
- Belongs as new versioned `TPMethod`s in `quant/scenarios/takeprofits.py` (e.g. `roc_trail_ema@v1`,
  `asym_pullback@v1`); may need an ROC/steepness helper indicator. Ties to the `trailing`/`n_green_exit`
  family already built.

**3 — Test the 2 screenshot indicators, alone and in combos.** The two TradingView indicators shown:
- **Colored EMA/SMA (9, 200)** — a colour-by-slope MA overlay (the green/red/orange fast+slow lines).
- **MA Shift [ChartPrime] (EMA, 40, hl2, 15, 0.5)** — the bottom-pane teal/orange histogram (price-vs-
  shifted-MA deviation / momentum). Needs a faithful port (like SQZMOM/WaveTrend in `indicators/momentum.py`).
- Test **each alone**, then **in combo with WaveTrend (WT) + Squeeze (SQZMOM)** and/or **price structure**
  (HH/HL, S/R zones, 200-EMA). Likely a new notebook + faithful indicator port(s), reusing `studies_figure`.

### 2026-07-30 — Divergence lag quantified; ADAPTIVE (ZigZag) structure beats the fractal

Measured on BTC 1m, 2026-05-01 -> 2026-07-29 (128,161 bars).

**Divergence lag == `right` exactly. `left` is FREE** (those bars are already history), so an
asymmetric pivot keeps significance at minimal delay. Smaller `right` also produces MORE signals —
lag and noise trade off directly:

| left | right | lag | signals |
|---|---|---|---|
| 10 | 1 | 1 | 1727 |
| 10 | 2 | **2** | 1523 |
| 10 | 5 | 5 | 1379 |
| 10 | 10 | 10 | 1034 |

BUG FIXED: `scenario_figure` derived the divergence pivot from `struct_lr // 2`, so `STRUCT_LR=10`
silently forced a 5-bar lag the user never chose. Now explicit via `div_left` / `div_right`.

**Zero-lag IS possible but costs ~47% false positives.** `add_rsi_divergence_realtime` fires on the
running extreme vs the last confirmed swing — no lookahead, appears on the bar it becomes true.
Provisional **4300** signals vs **1523** confirmed; only **53%** are ratified by a confirmed
divergence within 15 bars. Frame it as an early warning, never as a better detector. There is no
lag-free AND accurate option — confirming a pivot requires future bars by definition.

**Structure: ATR-scaled ZigZag (directional-change events) is the right adaptive method.** Surveyed
fixed fractals, ATR/%-threshold ZigZag, efficiency-ratio adaptive lookbacks, and Perceptually
Important Points (offline only — not causal, unusable here).

| method | pivots | mean lag | max lag | up/down/side % |
|---|---|---|---|---|
| fractal lr=5 | 19,563 | 5.0 | 5 | 21 / 40 / 40 |
| fractal lr=10 | 9,994 | 10.0 | 10 | 21 / 40 / 39 |
| fractal lr=20 | 5,023 | 20.0 | 20 | 22 / 38 / 40 |
| **zigzag 1.5xATR** | 24,988 | **1.8** | 14 | 31 / 33 / 36 |
| **zigzag 2.5xATR** | 12,838 | **3.6** | 27 | 31 / 33 / 36 |

`zigzag 2.5xATR` gives MORE pivots than `fractal lr=10` at ~1/3 the lag. Lag at 1.5xATR: median **1**,
p75 2, p95 4 bars — short on sharp reversals, longer in chop (correct: chop shouldn't manufacture
structure). Regime split is also more balanced than the fractal's.
-> `add_zigzag_structure(df, atr_mult=...)` in `indicators/structure.py`; it is now the chart default
(`struct_method='zigzag'`). The fractal path stays for comparison.

### 2026-07-29 — THE ENTRY ITSELF HAS NO EDGE (the root cause of everything above)

Prompted by the right question from the user: *"the main driver of the profits should be the main
logic — or are you just looking at the secondary scenario results?"* Correct challenge; every test
until now measured entry+stop+exit **bundles**. Isolated the entry two ways, no stop and no target.

**Test 1 — drift-adjusted mean forward return.** For every raw entry signal (from `compose()`, so all
signals, not just those that became trades), forward move signed for a short, in ATR units, minus the
same statistic over all bars in that window (= the drift). `support_break_short@v1.2`:
- BULL +26%: edge **+0.08 … +0.36** ATR across horizons
- BEAR1 −24%: edge **−0.02 … −0.66** ATR  ← *negative*
- BEAR2 −28%: edge **−0.68 … +0.10** ATR (mixed)
Sign flips by regime; |t| never clears noise (max 2.15 across 54 cells tested, and those t-stats are
optimistic because the forward windows overlap).

**Test 2 — post-signal PATH (the fair test for a trailing exit).** A trailing stop harvests trending
behaviour, not mean forward return, so also measured the 60-min forward path: Kaufman efficiency of
the path after a signal vs a random-bar baseline, and MFE/MAE excursion asymmetry.
- **Efficiency delta ≈ 0.000 … 0.004 in every window for both v1.0 and v1.2.** The path after a
  signal is *exactly as choppy* as a random moment. **A trailing stop has nothing extra to harvest.**

**Conclusion: the entry contributes no measurable predictive power.** That is *why* exit choice
dominated every earlier result — when the exit rule explains the P&L, it means the entry does not.
Leverage multiplies zero edge to zero.

**The one consistent signal in the whole analysis — and it says we optimised the WRONG version.**
`support_break_short@v1.0` (the S/R-zone version) is the only piece with a consistent sign across all
three independent regimes:
- drift-adjusted 5-min edge **+0.068 / +0.038 / +0.037** ATR (BULL / BEAR1 / BEAR2) — all positive
- MFE/MAE excursion delta **+0.127 / +0.031 / +0.074** — all positive (ratio 1.04–1.09 vs base 0.95–1.02)
All of notebooks 10–12 tuned **v1.2** (no-structure). v1.2's own numbers flip sign by regime.
Effect size is small (~0.05 ATR ≈ $2/unit on BTC vs ~$6/unit round-trip at 0.7bp), so it likely does
not clear costs as-is — but if anything here is worth pursuing, it is v1.0, and the direction is
**fewer, higher-conviction entries** (it currently fires ~8–10/day), not more exit tuning.

### 2026-07-29 — Time-of-day / day-of-week: HOURS ARE NOISE, WEEKDAY PERSISTS

BTC 1m, 2025-01-01 → 2026-07-29 (826,561 bars). New module `quant/analytics/timing.py` +
`notebooks/12_time_of_day.ipynb` (`ROLL_FREQ` = the rolling cadence: 'M'/'Q'/'2W').

**Hour-of-day is NOT tradeable — do not build an hour blocker.** Four independent checks agree:
- Efficiency ratio (chop) by hour spans only **0.1127–0.1375** (22% relative); *every* hour is choppy.
- Best/worst hours **disagree across timeframes**: 1m best `[21,11,1,9]` vs 15m best `[23,6,8,19]`
  (no overlap); hour 3 is 1m-worst but 5m-best. Only hour **18** is bad on all three.
- **Quarter-to-quarter rank correlation of the hourly means: mean rho = −0.030** (5 of 6 negative,
  all p > 0.30). The ranking reshuffles every quarter.
- **Walk-forward blocking the prior quarter's worst 6 hours helped in 2 of 6 quarters** — worse than
  a coin flip. Strategy level: bear-vs-bull hourly rank correlation +0.217 (p = 0.308); only **4 of
  24** hours positive in both regimes, *below* the ~6 expected by chance.

**Day-of-week DOES persist (weak but real).**
- Weekday rank correlation quarter-to-quarter: **mean rho = +0.315**, all six ≥ 0 (0.75, 0.61, 0.21,
  0.21, 0.11, 0.00) — a clear contrast with the hourly −0.030.
- **Sunday is positive in 7 of 7 quarters** (mean follow-through +0.162 ATR vs ~0.02–0.05 for other
  days). Sign test p ≈ 0.008; t-test p = 0.047 (NOT significant after Bonferroni for 7 tests, 0.0071).
  Suggestive, not proven. Saturday is the weakest (3/7 positive, mean −0.022).
- Weekday filters on the strategy (spread 6.0, bear/bull): `all days` +0.107/−0.059 →
  `Mon+Thu+Sun` **+0.194/+0.177**, `Sun only` +0.167/+0.613, `exclude Sat` +0.135/+0.002.
  **CAVEAT: those day-sets were chosen after looking at the full sample, so the "bull" column is not
  a clean OOS test.** Only Sunday has independent support (picked on 7/7 sign consistency in the
  direction-agnostic measure, not on strategy PnL).

**Gotcha:** by-weekday stats of an *unfiltered* run ≠ running *with* a weekday filter — with
`max_open_trades=1`, blocking other days frees the slot so a different trade set is taken (Sunday was
−0.063R unfiltered but +0.167R as `Sun only`). Always re-run with the filter applied.

#### ...and the clean OOS test REFUTED it. Weekday filtering does NOT work.
Pre-registered protocol — select on **TRAIN 2025** only, evaluate on **TEST 2026-01→07** untouched;
control = score **all 127** non-empty weekday subsets on TEST (so regime cancels out).
- **Both selection rules made things worse.** ALL DAYS `expR +0.0721, +36.8%` →
  Rule A (`Mon+Thu+Fri+Sat+Sun`, picked on follow-through) `+0.0345, −0.6%`;
  Rule B (`Mon+Tue+Thu+Sun`, picked on train expectancy) `+0.0376, +2.5%`.
- **Both landed in the BOTTOM ~15% of all 127 subsets** (rank 110/127 and 107/127, where 1 = best;
  random would be ~64). Median subset `+0.0960`. So the trained picks were *worse than choosing days
  at random*.
- **Per-day sign flips: 5 of 7.** Train→test rank correlation **rho = −0.143 (p = 0.76)**.
  **Sunday: train rank 1 (+0.542) → test rank 6 (−0.013).** **Wednesday: train rank 6 (−0.224) →
  test rank 2 (+0.306).** Only Mon (and weakly Tue) kept its sign.
- Subsets containing Sunday averaged `+0.0721` vs `+0.1396` without it — including Sunday *hurt*.

**Correction to the entry above:** the +0.315 quarter-to-quarter persistence was real *in the
direction-agnostic follow-through metric*, but it does **not** translate into a tradeable weekday
filter. Sunday's 7/7 quarterly consistency did not survive contact with an unseen period. Treat both
hour-of-day and day-of-week filtering as CLOSED — negative results, do not revisit without a
materially longer history and this same OOS + all-subset control protocol.

**Method to reuse:** measure direction-agnostically (efficiency ratio, EMA follow-through in ATR
units) so market drift can't fake a result → rank-correlate consecutive periods → walk-forward the
blocking rule → only then check the strategy in a regime the buckets weren't chosen from.

### 2026-07-29 — 1-year validation of the support-break shorts: NO ROBUST EDGE FOUND

Tested the notebook-10/11 exit variants on **BTC 1m, 2025-07-29 → 2026-07-29** (525,601 bars).
Result: **the apparent winners are regime beta and/or die on costs.** Record this before re-running
the same experiments.

- **The window was a bear market: BTC −45.8%** (117,989 → 63,983). Everything here is **short-only**
  (`Scenario.side = 'short'` for all `support_break_short` versions), so a passive 1× short would have
  made ~+45.8%. Any short-only result in this window must be compared against that, not against zero.
- **SIZING BUG in my own test harness (watch for it):** `allow_leverage=True` is *not* enough —
  `max_notional_pct` defaults to **100**, which re-caps notional at ~1× equity. With a 0.25%-wide stop
  the realised risk was **0.25%**, not the configured 1%, and `ret_pct`/`sharpe`/`dd` were meaningless
  while `R` stayed valid. Set `max_notional_pct` high (e.g. 5000) for risk-based sizing to bind. Same
  75.5R went from **+0.1% to +56.7%** once uncapped.
- **Full year (correct sizing), ranked by expectancy:** trailing 2% `0.188R`, trailing 1% `0.136R`,
  rr3R `0.054`, rr2R `0.054`, bail-shaky `0.031`, bail-profit-candle `0.025`, rr1R `0.003`.
- **Split-half:** trailing is stable across both (bear) halves; **`bail: shaky` is NEGATIVE in H1**
  (−0.009R, −21.7%) and positive in H2 (+0.074R) — unstable.
- **OUT-OF-SAMPLE REGIME (2025-01-01 → 2025-07-29, BTC +26.0%): everything collapses.**
  trailing 2% `−0.079R` (−53.7%), bail-shaky `−0.041R`, trailing 1% `−0.036R` (−42.9%), rr2R
  `−0.020R`. Only `bail: profit candle` stayed positive (`+0.008R`, +10.6%). **So the trailing
  variants are directional beta, not edge.**
- **COSTS KILL THE BAIL RULES.** All the above ran at `spread=0`. At `spread=6.0` price units
  (**~0.7 bp**, deliberately optimistic) `bail: profit candle` goes **−0.0067R** in the bear year and
  −0.0225R in the bull. Its average win is only **0.25R** over ~2,600 trades, so it cannot absorb any
  friction. `trailing 1%` survives costs (+0.107R at 6.0) — but only in the bear window.
- **Refinement attempt (stricter down-gradient gate `grad_max` × `trail_pct`, spread 6.0):** exactly
  **1 of 10** cells was positive in both regimes (`grad_max=-0.05, trail_pct=2.0`: bear +0.055, bull
  +0.110) — but its neighbours flip sign (`-0.05/1.0` → bear −0.002; `-0.10/2.0` → bull −0.030).
  An isolated cell in a noisy grid is an overfit, not a discovery. **Do not trust it without a
  proper walk-forward.**

**Conclusion / next direction:** the binding constraint is that the family is **short-only**, so it
inherits market direction. Worth trying before more parameter tuning: (a) build the **long mirror**
(resistance-break-long) so both sides can trade; (b) a **higher-timeframe regime filter** (MTF EMA
bias) so shorts only fire in a confirmed downtrend — `grad_max` is a weak, same-timeframe proxy for it.

### 2026-07-29 — Scenario round 2: stop lag, level B, secondary rules

**GOTCHA worth remembering — pivot stops lag by construction.** User ran
`support_break_short@v1.2 + swing_high@v1 + rr@v1` on BTC 1m 2026-07-28 and asked why the stop sat so
far above the entry. Reproduced: entry bar 237 (`03:57`, 63284.60), `swing_last_high` = **63375.39**
(the `03:44` high), `× 1.0005` buffer → stop **63407.08**. Cause: `stop_swing_lr=12` means a pivot
needs **12 bars of confirmation to its RIGHT**, so the newest usable swing is always ≥12 bars old —
the nearer high (`03:55`, 63336.70) only confirmed at bar 238, *one bar after* the entry. **Lowering
`lr` does NOT fix this** (verified: `prev_high@v1` with `lr=3` returns the identical 63407.08). Any
pivot level has this property; it is anti-lookahead working correctly.
→ Fix = a lag-free definition. New **`leg_high@v1`**: running max of `high` since price last closed
below the EMA line (episode-restarted `cummax`), optional `stop_above_ema_only`. On that trade it
gives **63373.05** (~34 pts tighter), cutting the loss from -19.35 to -13.98. `high_of_n@v1` is the
other lag-free option.

**Level B is now a first-class column.** Each version writes `break_level_raw` (the live candidate)
and `break_level` (**latched at the first break of the episode**). The latch matters: an S/R support
steps DOWN to the next zone the instant price closes through it, so the live value stops naming the
level that actually broke — which is also why `break_mode='full_below'` returned 0 trades before
latching. `scenario_figure` draws it per trade (amber dashed, own legend group).
`break_mode = close_below` (default, the original rule) `| full_below`. Optional, default-off
`break_min_ema_dist_pct` / `break_max_ema_dist_pct` gate level B on its distance from the EMA line —
the notebook's level-B audit table prints that distance so thresholds can be chosen from real trades.

**Secondary rules = the 4th composable piece** (`quant/scenarios/secondaries.py`). Needed a real
kernel feature: "armed since THIS trade opened" is per-trade path-dependent state a global signal
array can't express. `Signals.exit_arm_*` / `exit_alt_*` + per-trade `o_armed`; arming is evaluated
before entries open, so **a trade arms only from the bar AFTER its entry** and can't inherit arming
from before it existed. Once armed, `exit_alt_*` closes it **only while in profit** (close reason
`alt_exit`). `sec_replace_tp` additionally drops the TP ladder for armed trades (stop/trailing never
affected). All flag-guarded → runs without a secondary are byte-identical (golden tests green).
`shaky_trend_bail@v1` turned the 63407 stop-out above into an `alt_exit` at 63269.30, in profit.

**Also:** `opposite_candle@v1` TP ("ride it while candles stay red, bail on the first green");
`candle_src='real'|'ha'` threaded through every candle test in the scenarios; `scenario_figure`
title/legend no longer overlap (title 14–34px, legend 79–99px, plot starts at 110px);
`zone_top@v1` now uses `break_level`, and `next_zone@v1` adds S/R zones itself — so every
stop×TP×scenario combination is runnable.

**Env gotcha:** the registered Jupyter kernels are **Python 3.9 / 3.10** while the project runs on
**3.13** — `nbconvert --execute` fails on old pandas. Notebook 10 was verified by executing its code
cells under the 3.13 interpreter instead. Registering a 3.13 kernel would fix this properly.

### 2026-07-26 — Scenario framework + EMA-gradient (earlier session)
- **`quant/scenarios/`** — new package: a run = **`Scenario` × `StopMethod` × `TPMethod` × params**,
  each `name@version`, registered, bundled into a **`Preset`** (JSON in `scenarios/presets/`,
  reloadable on any symbol/TF). `run_pieces(df, scen, stop, tp, params=…, cfg=…)` / `run(preset)` →
  `ScenarioRun`. Built on the existing engine + `RuleStrategy` (one-per-episode short rule, re-arm
  when price closes back above the EMA line) — **no kernel change, golden parity untouched.**
  - Scenarios: `support_break_short@v1.0` (core: close below smoothed 200-EMA + support-zone break),
    `v1.1` (+HH/HL structure into resistance), `v1.2` (no S/R: N red below EMA + prior-N-low break),
    `v1.3` (nearest-trough support). `entry_mode = break_close|wait_n|n_red|pullback` (the user's Q2).
  - Stops (Q1): `pct / swing_high / high_of_n / zone_top / atr`. TPs (Q3): `rr / fixed_pct / trailing /
    n_green_exit / back_above_ema / next_zone / rr_partials`. Signal-TPs write `Signals.exit_short`;
    cfg-TPs set `BacktestConfig`. **`zone_partials` (true absolute multi-zone partials) NOT built** —
    would need a per-trade TP-ref kernel branch mirroring `sl_ref_short_col` (deferred).
- **EMA-gradient indicator** `add_ema_gradient` (in `indicators/momentum.py`): `grad_line` =
  EMA(smooth) of EMA(period) — the red-dotted trend line; `grad` = ATR-normalised slope (≈±0.15 on
  BTC 1m; sign = trend direction, small threshold ≈±0.1 = flat). Scale is smaller than the
  screenshot's ±1 TradingView oscillator — cosmetic only; the threshold is the knob.
- **New conditions** (`strategies/conditions.py`): `broke_below_support` (close < prior-bar support),
  `broke_above_resistance`, `broke_below_prev_low(n)`, `n_red_below(n, level)`.
- **Viz** `quant/viz/studies.py`: `scenario_figure(run)` — one annotated chart (candles + EMA line +
  causal S/R zones + entries/stops/exits + gradient pane + crosshair); `zoom_to_trade(fig, run, i)`.
  **Notebook `10_scenario_lab`** drives it all (config → chart → zoom → variant sweep → save preset).
- **Sweep read (recent BTC 1m, in-sample, one window):** shorts get stopped often — BTC was trending
  UP, so this is the honest result and the chart shows why. Tight stops (atr/zone_top) → more trades,
  higher Sharpe than wide swing-high stops; `atr@v1 + trailing@v1` was the best risk-adjusted combo
  (~Sharpe 1.5). Not validated OOS; the point of this session was the **framework**, not an edge.

### Earlier 2026-07
- Package `quant/` + `experiments/` + `scripts/` + `notebooks/`. Legacy code removed.
- Strategies (REGISTRY): `ema_ribbon, ema_cross, rsi_reversal, macd_trend, heikin_ashi, supertrend,
  mtf_trend, key_level`. **`EmaCross` is the workhorse** for EMA-variant research (every variant is a param).
- Data sources: `binance` (PAXGUSDT/XAUTUSDT/BTCUSDT) and `dukascopy` (true spot XAUUSD/FX, working).
- Cached now: PAXGUSDT 1m/5m/15m/1h to ~2026-07-22; **Dukascopy XAUUSD 1m/5m/15m/1h/4h from 2026-01**
  (1m back to 2024-06). Re-run `python scripts/fetch_gold_dukascopy.py` to extend (resumable).

## Notebooks (what each is for)
1. `01_research_cycle` — full manual cycle (data → indicators → strategy/rules → SL·TP → sim → stats → chart → save).
2. `02_inference_experiments` — the inference layer: define an idea → sweep → interpret → pick → validate OOS.
3. `03_ema_cross_study` — answers the EMA-cross research questions Q1–Q9 via staged sweeps.
4. `04_trend_runner_leverage` — full-candle trend-runner + Exness-style leverage scaling ($1000 / 1:2000).
5. `05_simple_1m_ema_breakout` — long-only 1m: enter full candle > EMA200, exit full candle < EMA50,
   stop = previous swing low; candle chart with toggleable entry/exit/stop + trade table.

## Research findings so far (gold; validate before trusting)
Study run on PAXGUSDT, ~2025-06→2026-07, spread 0.12–0.24. **In-sample unless noted; one asset.**
- **Timeframe:** shorter wins for this idea — 1m/5m positive, 15m/1h weak/negative. Best risk-adjusted ~5m.
- **Entry:** the **"full candle beyond the EMA"** filter is what lifts it above a bare cross; 1–2
  confirmation candles good, 3+ too late, requiring candle *colour* hurt.
- **EMA period:** shorter (20) ≫ longer (200) at 5m.
- **HTF bias:** helped at 15m, **hurt at 5m** — timeframe-dependent, not a free win.
- **Heikin-Ashi:** did not help vs regular candles.
- **Session:** Tokyo/London positive, **New York worst (avoid)**, Sydney weak.
- **Exits:** a **wide fixed target (≈3R) beat trailing stops and HA/shorter-EMA "ride" exits**, which
  overtrade and bleed to costs at 1m/5m.
- **Best OOS-robust config found:** `EmaCross(ema_period=20, entry_mode='full_candle', confirm_n=2)`
  on **5m**, SL 0.6% / TP 2–3R, risk 1%/trade → **train +14% (Sharpe 1.24) / test +22% (Sharpe 1.31)**.
  Modest (PF ~1.07–1.11) → **very cost-sensitive**.
- **Long vs short (evaluate separately via `allow_long`/`allow_short`):** longs carry the edge, shorts
  lose standalone, but the **two-sided version has the best Sharpe/drawdown** (sides hedge each other).
- **Naive 1m long-only** (nb05) loses ~−90% — spread eats ~10k trades. Filters are mandatory.

**Honest caveat:** these are a proxy (PAXG), mostly in-sample, and I selected some params after
seeing results. Re-run on **Dukascopy XAUUSD** + a proper **walk-forward** before acting.

## Gotchas / footguns (read before debugging)
- **Restart the Jupyter kernel** after editing `quant/` code — a running kernel holds the old code;
  the notebook cell text is unchanged but behaviour won't update until re-import.
- **Costs:** `spread` is the bid/ask WIDTH in **price units** (gold ≈ 0.12–0.30); cost = spread × qty.
  Standard Exness account → `spread≈0.12, commission_per_lot=0`; Raw account → `spread≈0.02 + commission_per_lot`.
- **Sides:** the STRATEGY controls long/short (`allow_long`/`allow_short`); `BacktestConfig.allow_short`
  defaults True and is just an engine gate (set False only to force long-only).
- **Leverage:** return is set by **risk-per-trade** (`sizing_value`), not leverage; leverage only raises
  the size ceiling + stop-out risk. Watch `max_drawdown_pct` and count of `margin_call` closes (aim 0).
- **Single data cache** at repo-root `data/` (absolute); providers no longer write to `cwd/data`.
- **`get_ohlcv` extends** the cache when you request beyond it (don't assume a warm cache is complete).
- **Sweep + indicator params:** strategies must name indicator columns **per-parameter** (e.g.
  `EmaCross` uses `ema_entry_px_20`) — `run_grid` caches indicator columns by name, so a fixed name
  makes a param sweep silently reuse one value. (The `experiments.Experiment` path re-prepares per
  combo and is immune, but keep the convention.)
- **Charts:** overlays (trade markers, stop-loss) are added un-aggregated and AFTER the resampled hf
  traces; don't interleave `_add_static` with `_add_hf` or plotly-resampler desyncs / asserts on x.

## Bugs fixed this session (don't reintroduce)
- Notebook setup cell now re-imports `quant` after the sys.path shim (was NameError).
- `get_ohlcv` fetches the gap when the request exceeds the cache (was silent truncation).
- `.gitignore` `data/` → `/data/` (was excluding the whole `quant/data/` package from git).
- Split cache (`notebooks/data/` from cwd default) consolidated; `BinanceSource` uses absolute dir.
- `EmaCross` EMA columns are period-encoded (fixes `run_grid` ema_period sweeps).
- Spread model simplified (fixed width; dropped the wrong `spread_per_lot` widening term).
- Chart: native-candle initial window, rich trade hover, stop-loss lines, resampler monotonic crash.

## Open threads / best next steps (highest value first)
1. **Re-run the EMA-cross study on Dukascopy XAUUSD** (true spot) to confirm the 5m edge isn't a PAXG artifact.
2. **Walk-forward validation** (rolling train/test) — turn the manual OOS split into a systematic cell/util.
3. **Volatility "avoid-chop" filter** (trade only when ATR% is in a healthy band) — `EmaCross` already
   computes `atr_14`; add `vol_lo/vol_hi` fields + a `vol_regime` signal primitive.
4. **`SessionBreakout` strategy** (London/NY opening-range breakout) — a complementary, low-correlation edge.
5. Engine extensions if needed: swap/overnight financing, partial (not all-at-once) stop-out, native ATR stop mode.

## Verify / run
```bash
pip install -e .           # add "quant[data]" for Dukascopy
python -m pytest tests/    # 40 pass
python scripts/fetch_gold_dukascopy.py   # (re)fetch true spot XAU/USD, resumable
jupyter lab notebooks/     # restart kernel after any quant/ code change
```

---

## Phase A findings — gold market character (Discovery 2024-06 → 2025-08, Dukascopy XAUUSD 1m)

Cost model: spread 0.25 $/oz, slippage 1bp/side → **~2.9 bp round-trip** at price ~2740.

1. **No cheap statistical edge.** Variance ratio ~0.94 (mild mean-reversion) across 5m–120m; lag-1
   autocorr ≈ 0. Conditional-fade test: **no move-size bucket beats cost** — even top-1% extreme
   moves revert only ~2.5 bp vs 2.9 bp cost, and signs are unstable across windows. High-frequency
   momentum AND mean-reversion are both < cost. → naive RSI-fade / MA-scalp are out.
2. **Fat tails exist and are harvestable.** After a new N-bar high, MFE ≈ MAE (no directional edge)
   BUT **MFE > 2×cost in 68–86%** of cases (more at longer horizons); p90 MFE 31–64 bp. 60m-return
   kurtosis 8.5. → the raw material for **breakout/trend-CAPTURE with trailing / RR-ladder exits**
   is there; profitability must come from **exit design + selectivity**, not the entry's mean edge.
3. **Bull-market regime bias.** 58.8% of time above EMA200(15m) — gold rallied hard 2024–25.
   Long-biased backtests are flattered by this; MUST validate OOS (2026 H1) + walk-forward, and
   test SHORT side too. Median run on one side of EMA200 = ~60 min (whippy); p90 ~19h.
4. **Negative skew (−0.47).** Sharper down-moves than up → short setups may have positive-skew
   payoff; long trend-following faces occasional sharp reversals.
5. **Session:** overlap 12–16 UTC = the big moves (avg 2.7 bp/bar) but near-random (VR 0.97);
   afternoon/late (16–24 UTC) reverts more (VR 0.90 / 0.75) but too small to trade after cost.
   Hourly directional drift negligible; no day-of-week edge.

**Strategic pivot for Phase B:** de-prioritise high-frequency fade/scalp; prioritise **selective,
low-frequency breakout/trend-capture on 5m/15m with trailing / RR-ladder exits** (harvest the fat
right tail), S/R zones as entry/stop/target structure, HTF+session filters for selectivity, and
**test both directions** while staying sceptical of the 2024–25 bull bias.

## Phase B findings — trend/breakout capture family (EmaCross entry + exit matrix)

- **3-month subset (2025 Q1) looked promising**: long trailing-trend-capture (EMA50, trail ~0.3%)
  PF ~1.15-1.2, broad plateau. **Full 14-month Discovery killed it**: PF collapses to ~1.03-1.05.
- **Per-quarter: best config positive in only 2/6 quarters** (wins in gold up-quarters, loses in
  chop/down). Short side is a disaster (PF 0.7-0.8). It's inconsistent long-beta, not an edge.
- Refinements that *did* matter: **5m > 1m**, **EMA100 > EMA50** → 5m/EMA100/trail0.4 reaches 5/6
  positive quarters, PF 1.12, +10.5%, DD 5.9%. **HTF trend filters did NOT help** (removed trades
  without improving quality).
- **DECISIVE BENCHMARK:** vs buy-and-hold gold over Discovery (+48.2%, Sharpe 2.14), the best
  strategy makes **+10.5%, Sharpe 1.11**, 76% time in market. **It underperforms holding on BOTH
  raw and risk-adjusted terms.** → the trend-capture family is expensive diluted beta, not alpha.

**Deeper problem:** Discovery (2024-06 → 2025-08) is a near-monotonic gold BULL (+48%). On such a
sample, any long strategy just tracks the trend, any short loses, and **nothing intraday beats
buy-and-hold** — the sample can't distinguish timing-alpha from beta. "Profitable" must be
redefined as **beat buy-and-hold risk-adjusted** (or protect capital in the OOS/other regimes),
which is a much harder and different target than a standalone positive backtest. Decision point
before continuing.

## Goal redefined (user decision)

Beating buy-and-hold is OFF the table (impossible on a monotonic bull; can't distinguish alpha
from beta). New target: a **controlled-risk absolute-return system** — market-neutral or
capital-protection — judged on **Sharpe / Sortino / max-DD / consistency across quarters & OOS**,
with positive expectancy NET of the 0.25 cost. Trade count can be low; selectivity + defined
risk/reward is the path. First test: symmetric S/R (long at support, short at resistance, stop
beyond the zone, RR target) — structural, defined-risk, not pure beta.

## Phase B breakthrough — the edge lives at HIGHER timeframe (1h/4h)

Testing genuinely different families first confirmed the cost wall intraday:
- Symmetric S/R mean-reversion: PF 0.63-0.71, clear LOSER (bounce doesn't beat cost).
- Opening-range breakout (NY): PF ~1.00, break-even net of cost (short side faintly +).
→ At 0.25 cost (~2.9bp), NO intraday family beats cost.

**But cost is a FIXED per-trade tax — negligible when a trade rides a multi-day move.** At 1h/4h,
long trend-capture (EMA breakout + % trailing) works:
- **4h EMA50 sl/trail 2%: Sharpe 3.22, PF 5.6, DD 3.4%, +19.3%, 6/6 quarters +, 22 trades.**
- 1h EMA50 2%: Sharpe 2.79, PF 2.67, 6/6q. | 1h EMA20 1%: Sharpe 2.26, +29%, 119 trades, 6/6q.
- vs buy-hold gold (Sharpe 2.14): these **beat it RISK-ADJUSTED** (higher Sharpe, DD 3-7% << gold's)
  — satisfying the controlled-risk goal even though raw return (+19-29%) < buy-hold (+48%).

**CAVEATS (must validate):** (1) still LONG-only / bull-dependent — short side loses badly; needs
OOS (2026) + ideally a bear regime. (2) best configs have few trades (22-37) → overfit risk; prefer
higher-trade 1h configs (~120 trades) for robustness. (3) Sharpe 3.2 on 22 trades likely optimistic.

**Next:** focus Phase C on 1h/4h trend-capture — sweep the full exit matrix (trailing/RR/BE/
time-stop) + a trend-regime filter, rank by consistency×Sharpe×trade-count, then Selection then
lockbox OOS.

## Phase C/F — FINAL VERDICT (the lockbox caught it)

Phase C exit-matrix sweep on 1h/4h: huge robust plateau (179/215 configs Sharpe>1.5 & PF>1.3);
winners = tight stop (1-2%) + RR-3 take-profit. Held on Selection (OOS#1, 2025-09→2026-01):
PF 2.5-3.1, Sharpe ~5, DD ~7%. Looked like a validated controlled-risk edge.

**Then the LOCKBOX (2026-02→06) — which is the ONE period gold FELL (-16.9%) — failed both frozen
finalists:**
- A (EMA50 tight+trail+RR3): -5.1%, PF 0.77, DD 13.5%  (risk mgmt limited the bleed)
- B (EMA20 stop+RR3):        -19.3%, PF 0.50, DD 23.4%
Cost stress (0.30) barely changes it — cost was never the issue at 1h.
Walk-forward whole span: 8/10 quarters positive, but the 2 losers = the 2026 downturn quarters.

**Conclusion:** what we found is well-risk-managed **long-only gold BETA**, not alpha. It makes money
when gold rises (Discovery + Selection, both bull) and loses when gold falls (lockbox). The rigorous
lockbox (a period the strategy was never tuned on, which happened to contain the regime shift)
PREVENTED deploying a strategy that would have lost real money in 2026 — which every in-sample and
Selection test missed. Config A's tight stop/trailing did contain the drawdown (-5% vs gold -17%),
so risk management is real, but the entry is beta.

**No validated regime-independent controlled-risk edge exists in this 2-year (mostly-bull) sample.**
To build/validate one that survives bear+range REQUIRES the deeper Dukascopy history (2003-2024) —
the option previously declined. Otherwise: treat any long-gold trend system as beta, size it as
such, and do NOT trust bull-only backtests.

## Deep-history verdict (2005-2026, 1h, all regimes) — DEFINITIVE

Fetched Dukascopy XAUUSD 1h+4h back to 2005 (410 -> 4000, bull/bear/range).
- **Trend-following (both directions), 22yr: net -6258, positive 8/22 yrs.** BULL +2276, RANGE -6504,
  BEAR -2029. Shorts total -6900 — LOSE EVEN IN THE 2013 -28% CRASH. No trend edge exists at 1h.
- **User's ride-the-EMA idea** (enter above EMA200, ride till cross under smaller EMA): PF 0.98-1.06.
  The "low win-rate, win big" mechanic is real (30% win, big winners) BUT nets positive only via bull
  (+5475 bull vs -2848 bear+range). Bull-beta.
- **EMA-cross continuation base rate = 52% (coin flip).** A full EMA cross carries ~no directional
  info net of cost. The "price always continues" intuition is false over 22yr.
- **Adaptive macro-direction filter** (long above / short below daily EMA200): net +14875, PF 1.14,
  but ALL from bull longs (+23634 bull; RANGE -7348, BEAR -1410; shorts still -8357). Concentrates
  beta, doesn't create a regime-independent edge.

**FINAL:** No profitable, regime-independent, controlled-risk intraday/short-swing gold strategy
exists in 22 years at 0.25 cost. Every 'winner' is long-gold BETA that pays only in bull markets;
shorts never work (gold spikes up, drifts down). The only honest artifact is a LONG-ONLY,
macro-filtered (daily EMA200) trend overlay — a low-drawdown way to hold gold in uptrends, sized as
beta, NOT alpha. Mean-reversion/scalping lose to the fixed cost. Adaptation cannot manufacture an
edge that isn't in the data.

## BREAKTHROUGH — conditional entry + let-winners-run (user was right to push)

Aggregate tests said "no edge"; that was too COARSE. Slicing the EMA-cross continuation by context,
with IS(2005-18)/OOS(2019-26) persistence, revealed durable pockets: US-pm(16-21) & Asia(0-7)
sessions, macro-uptrend (EMA200 rising), higher-low structure — each small but OOS-persistent.
Open-space/S-R proximity did NOT persist (data-mined).

**The exit was the other half.** Same conditional entry, capping winners (RR2) → Sharpe 0.27,
IS-negative. LETTING WINNERS RUN (trailing 2-2.5%) → the edge appears. Final strategy:

  ENTRY (1h): 2 closes above EMA50  +  session in {US-pm 16-21, Asia 0-7}  +  EMA200 rising
              +  higher-low structure (swing_last_low rising)   [one per episode, long only]
  EXIT: 0.5% initial stop, 2.0-2.5% TRAILING stop (ride winners), no fixed TP.

Full 2005-2026: **+503% (trail2%) / +720% (trail2.5%), PF 1.6-1.8, Sharpe 1.07-1.18, maxDD 25-29%,
15/22 years positive.** Robust: trail 2/2.5/3% all work (plateau); positive IS AND OOS; positive in
BULL(+44k) AND RANGE(+10k); only mildly negative BEAR(-4k, filter minimises exposure); survives
spread 0.30; 68% of profit survives excluding 2025.

HONEST CAVEATS: still LONG-only (a conditioned trend-capture, not market-neutral); Sharpe ~1.1 &
DD ~26% = a real trend system, not a holy grail; 36% win rate (must stomach long losing streaks);
session condition was selected on IS (mild bias, but IS/OOS/regime/plateau evidence is strong).
Both of the user's instincts were correct: (1) condition by time/structure, (2) small win-rate/win-big.

NEXT: formalise as a RuleStrategy + notebook; final walk-forward + lockbox; adaptive re-optimisation.

## EMA-crossover sweep (Stage 1) — findings

Swept fast×slow pairs (10-300) × TF(1m/5m/15m/1h/4h) × exit(opposite-cross vs trailing) × direction,
cost 0.25, gold.
- **Opposite-cross exit = dead everywhere** (whipsaw). Only TRAILING exits produce positives — the
  exit does the work, not the crossover entry (consistent with all prior findings).
- **Plain crossovers marginal**: best raw 1h 50x200+trail = PF 1.33, Sharpe 0.66. Short side loses
  (gold). Long-only lifts it to PF 1.61.
- **Session filter (US-pm 16-21 / Asia 0-7) is a robust improvement**: 1h 50x200 long+session+trail
  -> PF 1.90, Sharpe 0.90, **DD 11.4%**, positive IS+OOS AND bull/range/bear. A PLATEAU of good 1h
  pairs (50x100 PF1.86 Sh0.95 DD17; 50x200; 100x300 DD10.4; 20x100; 20x200) — not a spike.
- **Candle-confirmation (2-green) OVER-filters** the crossover (it already implies momentum) -> kills it.
- **Small TFs (1m/5m) show Sharpe 2.3+ but ONLY on the 2024-26 BULL sample** (no deep small-TF data)
  -> treat as bull-beta, UNVALIDATED. 15m weak.

**Best crossover strategy:** 1h EMA 50x100 or 50x200, LONG-only, US-pm|Asia session, 0.5-1% stop +
2% trailing. Comparable to the earlier '2 closes>EMA50' strategy but LOWER drawdown (11-17% vs 25%).
Session/time-of-day genuinely helps; short side and candle-confirm do not.

NOT yet done: band-width / cut-through-rate, day-of-week, MTF-crossover (fast on 1TF vs slow on
higher TF), adaptive walk-forward re-optimisation. Small-TF needs deep data to validate.

---
---

# SESSION 2026-08-24 — systematic strategy search, then the pattern libraries

Everything below supersedes the exploratory notes above. Read
[`WHAT_WORKS.md`](WHAT_WORKS.md) first for the verdicts and
[`STRATEGY_SEARCH_LOG.md`](STRATEGY_SEARCH_LOG.md) for the evidence; this is the working context.

## The headline

**No entry rule tested in this project has a validated edge**, and the binding constraint is
arithmetic rather than signal quality. Spread is **10.8% of one R** on BTC 1m at a 1.5xATR stop,
which requires a **36.9% win rate at 2R** to break even; every entry family measured **31-35%**.

## Methodology rules now in force (each earned by a wrong result)

1. **Difference every forward-return screen against a within-day-shuffled null**
   (`experiments/null_control.py`). Two independent, plausible control schemes both MANUFACTURED
   edges of exactly the size being hunted; shuffled data with no structure produced larger ones.
2. **Every entry competes against a metronome** — `time_entry_{long,short}@v1` enters on a fixed
   schedule with no signal, through the same stops, targets, costs and windows. Most "real" entries
   lose to it.
3. **Count survivors against chance** (`n_configs x 0.5^n_windows`, and use the right binomial tail
   for "k of n windows" filters — getting this wrong once made 6 survivors look like 15x chance when
   it was 1.35x).
4. **Trim the top K trades, not just the top one.** Fat-tailed payoffs pass every conventional gate;
   deleting the best 10 of 600-2,300 turned every Phase-3 survivor negative.
5. **Bootstrap per window, never pooled.** Pooling showed `p_neg ~ 0.0001` when no individual
   window's lower bound cleared zero.
6. **A gate the metronome also passes is not a gate.** Rotating out-of-sample was passed by a random
   entry with a trailing stop, which retires that test for fat-tailed payoffs.
7. **Judge a filter at constant trade count.** Apparent edge and sample size moved in opposite
   directions almost monotonically; a 7-trade configuration scored +1.49R.
8. **Balance the window set before trusting a directional result.** Three windows from one falling
   market cannot separate "short edge" from "short beta" — this retracted the eventual winner.
9. **Measure capture before optimising exits.** Fixed targets already capture 87-93%; the real leak
   was that 30% of trades reaching +1R end up losing.
10. **Check ATR-normalised results in PRICE terms.** A "1.79x expansion after quiet" was entirely the
    denominator shrinking. The pattern-expansion result was re-checked this way and survived.

## Window sets

`experiments/phase9_long_fair_test.WINDOWS` is the **balanced nine-window cycle** (BTC 15m,
2022-2026): five bull (+26% to +84%) and four bear (-16% to -57%). Use this, not the old three-window
set from `search_harness.make_windows`, which was 2/3 bear and produced a retracted result.

## What was built

* `quant/indicators/levels.py` — prev-day/week/session levels, opening range, session VWAP + bands,
  volume features (first use in this project), round numbers, `level_events()`.
* `quant/indicators/candle_patterns.py` — 62 candlestick patterns.
* `quant/indicators/chart_patterns.py` — 19 chart patterns over fitted trendlines.
* `quant/strategies/regimes.py` — regime conditioners with rolling-quantile thresholds.
* `quant/scenarios/entries.py` — 12 entry families, each long AND short, incl. the metronome.
* Engine: `max_bars_in_trade` time stop; a TakeProfit closing 0% now still moves the stop (that was
  silently inert, so "breakeven at 1R" did nothing). Golden parity preserved throughout.
* `experiments/mgmt_sweep.py` — prepares an entry once and reuses it; identical trades to
  `run_pieces`, ~9x faster.

## Bugs found the hard way — worth not repeating

* **`EntryBundle` cached under a prepare-only key** made every variant run with the FIRST variant's
  trigger and threshold. 288 configs returned deltas of *exactly* 0.0000, which reads as a clean
  negative rather than a failure. Locked by `test_sweep_shares_frames_without_leaking_params`.
* **`add_candle_patterns` called twice** left duplicate columns; `df[col]` then returns a DataFrame
  and every boolean test silently misbehaves — named-subset entries produced ZERO trades, no error.
* **Chart patterns called any two pivots a trendline.** Two points define a line exactly, so every
  pair "fit". Now RANSAC + envelope with a touch requirement.
* **An open Jupyter autosaved a stale notebook over a fix, and `git add -A` committed the revert.**
  Check the diff when committing alongside an open notebook.

## Current state of the pattern work  (rewritten 2026-08-24 evening)

### The de-duplication lookahead — read this before trusting any older pattern number

`_dedupe` ranked every detection globally by how well-formed it was, so a pattern confirming at bar
5,000 could delete one that had already fired at bar 2,997. **A past signal depended on future bars.**
Fixed in `cd70932`; overlaps now resolve in confirmation order, which is also what a trader
experiences (the earlier signal is already taken and cannot be un-taken).

The truncation test missed it for a year because it checked ONE boundary at default sparse settings,
where no overlap clash ever straddled that point. Now parametrized over six boundaries at settings
dense enough to generate overlapping wedges, plus an assertion that the fixture really produces them.

What it was worth, BTC 15m 2022H1, `rising_wedge` at its own best settings: detections 67 -> 83,
drift-adjusted move **1.34 -> 0.53 ATR**, hit rate **56.7% -> 51.8%**. At phase 12-14 settings across
all nine windows, **21% of firings changed** (rising_wedge 45%, symmetrical_triangle 55%).

### The verdict, from the clean re-run

The metric itself is sound: on within-day-shuffled bars the same 108-setting grid scores **+0.011**
with 51.0% of cells positive, so a positive result on real data is signal, not measurement artifact.

**Direction is dead, three independent ways.** Paired against its own shuffled twin per window,
**0 of 13** patterns beat the null in >=8 of 9 windows (chance 0.3; best three reach 7 of 9, p=0.09).
Median hit rate **0.480 real vs 0.500 null** — real data is WORSE than shuffled. The real hit rate
exceeds 50% in **1 of 9** windows.

**Expansion survived and grew.** Forward 20-bar range against a null of ~0.97: triple_bottom **1.42**,
triple_top 1.25, symmetrical_triangle 1.22, head_shoulders 1.21, inverse_head_shoulders 1.18. The old
figure was 9-32%; it is now 18-42%. Removing the hindsight de-dup STRENGTHENED it, consistent with the
old path diluting the effect by keeping shapes selected after the fact. **A chart pattern says a big
move is coming, not which way.**

Effect size is 0.17-0.36% over 20 bars against a 0.007% spread — **10-25x the cost floor**. Unusually
for this project, cost is not the binding constraint here; the sub-50% hit rate is.

### "Best settings per pattern" has no stable answer — this is the methodology lesson

Re-running the identical grid on the fixed detector **moved the winning cell for 13 of 13 patterns**,
including patterns whose rank correlation across all 108 settings is 0.98-0.999. The surface is flat
and the argmax is decided by noise. **Never hand over a per-pattern argmax table.**

`experiments/phase15b_pattern_controls.py` therefore selects settings on SAMPLE SIZE (outcome-blind:
strictest setting still yielding >=10 detections/window in all 9 windows), with the quality knobs taken
from the pooled scan over 13 patterns x 9 windows. That scan independently landed on
**min_touches=3**, and the planted **min_touches=2 control now scores WORST** — so the touch
requirement is doing real work, which it was not able to show before the fix.

### Methodology rule 11 (new)

**Re-run the whole search after fixing a detector bug, and compare the argmax, not just the mean.** A
21% perturbation of the signal set left rank correlations at 0.98+ while relocating every single
winner. Stability of the ranking says nothing about stability of the optimum.

## Detector geometry work (2026-08-25) — two bugs from a rendered chart, then two new families

Both bugs were found by LOOKING at a chart, not by a test. Worth remembering: the test suite was
green throughout, because neither defect violates an invariant anyone had thought to assert.

### Bug A — the dots were not touches

`_best_line` computes the RANSAC inlier mask that decides which pivots the boundary is fitted
through, then **discarded it** and returned only the count. The row published `pivot_i` (the whole
lookback window) and `pattern_figure` drew that with the hover label "<pattern> touch".

Measured on BTC 1m over five days: **210 dots for 106 real touches** — half the dots marked a swing
touching nothing. Across all patterns on 15m the trendline shapes were drawing **42-63%** spurious
dots. `_Fit` now keeps its inliers; the shapes table publishes `touch_i` / `touch_price` / `touch_t`;
the figure draws those. Level shapes (H&S, double/triple tops) have no fitted lines, so their
defining extremes ARE the touches and they legitimately draw 100% of their pivots.

### Bug B — symmetrical triangles were not symmetrical

The test was only "upper falls, lower rises, converging". Nothing tied the two slopes together, so a
boundary falling at **-12.9** against one rising at **0.76** qualified. Ratios reached **17:1**,
median 3.3. Added `slope_sym_max` (default 2.5) and applied it to broadening formations too, where
ratios reached 10:1 — a megaphone is symmetric by definition. Same chart: 16 triangles -> 10, worst
ratio 2.22.

A shape that converges but fails the bound is **dropped, not reclassified**: it is not flat-topped or
flat-bottomed either, so it is not a classical named pattern.

### New: CHANNELS (`channel_up`, `channel_down`, `channel_flat`)

Two PARALLEL boundaries holding a roughly constant width. `width_tol_frac` (0.25) IS the parallelism
test — if the gap is the same at both ends the lines are parallel, and "fixed size with some margin
of error" is exactly that number.

Adding them exposed a THIRD latent bug: `converging` meant only `w1 < w0`, any hair of narrowing, so
**a dead-parallel channel was being emitted as a wedge**. `converge_min_frac` (0.25) now requires a
shape to actually narrow, which makes channels and wedges mutually exclusive by construction. Wedge
counts fell 32->29 and 27->21, broadening 10->6 and 12->6 — those were the degenerate ones.

Channels also fixed the FLAG model. A bull flag is textbook a counter-sloping channel after a pole,
but the only flaggable shapes were rectangles, so only perfectly flat flags could ever be found.
`_FLAGGABLE` now includes channels, with an up-channel after an up-pole correctly NOT a flag.

### New: TRENDLINES (`quant/indicators/trendlines.py`)

One fitted line, not a closed shape. `add_trendlines` maintains the live support and resistance line
bar by bar and flags `tl_break_up` / `tl_break_dn` on the close through it; a broken line is retired.
Same RANSAC + envelope fit and the same causality rule (published only from the bar its last pivot
CONFIRMED). Verified causal by truncation at three points.

One bug caught while building it: a line SUPERSEDED by a better fit never had its `end_i` set, ran to
the end of the chart, and extrapolated to endpoints like **-125,741** on a 60,000 instrument. Same
extrapolation trap the pattern targets already guard against.

### Calibration for the new shapes

Channels were added after the Phase-15 sweep, so their settings were picked by the SAME outcome-blind
rule applied afterwards (strictest setting still giving >=10 detections per window across the nine
windows): `atr_mult=2.5, min_span_bars=100`, giving minima of 20 and 18 per window. **`channel_flat`
clears no setting** — BTC almost never holds a strict fixed-width sideways band for 100+ bars. That is
a property of the instrument, not a detector failure, and it is documented as such.

`TUNED_SPECS` is now 16 entries. Tests: 302 passing.

## Shape-quality audit (2026-08-25) — `experiments/pattern_quality_audit.py`

Built because three detector bugs in a row were found by LOOKING at a chart while the suite stayed
green: every existing test asserts an INVARIANT (causality, subset relations, monotonicity) and none
asserted that a shape RESEMBLES ITS OWN DEFINITION. Six checks, each with a pass criterion derived
from the pattern's definition. Run over the nine windows on **BTC 15m, BTC 1h and XAUUSD 15m** —
4,026 shapes.

### What passed, and it is worth knowing these are now measured rather than assumed

* **Causality: 0 mismatches** over 12 truncation points x 9 windows x 3 instruments, on REAL data.
  The suite only ever truncated a synthetic random walk.
* **Definitional contract: 100%** on all 16 patterns, all three runs.
* **Scale invariance CONFIRMED.** Median containment differs by at most **0.006** between BTC near
  $60,000 and gold near $2,000, per pattern. The detector really is ratio-based; nothing absolute
  leaked in. Same for 15m vs 1h.
* **Touch residuals 0.13 median** against the 0.6 ATR tolerance that admitted them — the fits are not
  being dragged to the limit, which is what you would see if the tolerance were doing the work.
* Trendlines: zero off-chart endpoints, zero touch-count mismatches, zero breaks on the wrong side.

### Three findings that came out of it

1. **LEVEL SHAPES ARE LOCKED AT EXACTLY 1:1.** For double/triple tops and bottoms and both
   head-and-shoulders, `pattern_target` is the neckline plus the height and `pattern_stop` is the
   extreme that height was measured from — so reward IS risk, by construction. Measured
   reward-to-risk 1.000 with **zero variance across all 1,936 level-shape detections**. With a 0.48
   hit rate that pairing is a guaranteed loser, and it is 48% of everything the library detects. Not
   a bug (it is the textbook measured move) but it was undocumented; `pattern_target@v1` now says so.
   The fitted-boundary shapes are fine — their stops come from real swings and R:R runs 1.4-3.7.

2. **EXTRAPOLATION IS THE TRIANGLES' PROBLEM.** `max_wait=150` lets a shape wait that long for its
   break, and the boundaries are extrapolated the whole way. Split by draw distance:

   | drawn past last pivot | n | containment | below 0.80 | width drift |
   |---|---|---|---|---|
   | short | 239 | 0.990 | **0.0%** | 0.085 |
   | long  | 226 | 0.913 | **32.7%** | 0.452 |

   Same detector, same patterns — only the extrapolation differs. Channels are immune (drift 0.006:
   parallel lines stay parallel); converging shapes are not. **Proposed fix, NOT yet applied because
   it changes what confirms and would invalidate the calibration and every phase result: expire a
   converging pattern once price passes a fraction of the way to its APEX**, which is also the
   textbook rule.

3. **The audit had a hole in its own verdict** — `rr_ok` was computed by check 4 and never consulted,
   so extreme reward-to-risk could not fail the run. 18 of 4,026 shapes (0.45%) sit outside the
   0.2-20 band, worst 41.6, all wedges and channels. Now wired in, along with a triangle-containment
   gate.

## In flight

Nothing. Phases 12, 13, 14 and 15 have all re-run on the causal detector and their results are
committed, raw CSVs included, so nothing needs a two-hour re-run to re-derive.

## Not done / next

* The 3 channels and the trendlines have NOT been through the research pipeline — no phase-12 screen,
  no null differencing, no metronome comparison. They are drawing tools until they have been. Given
  every other family measured, expect no directional edge; a trendline break in particular is a close
  cousin of the level-break family that was already measured and found to carry none.
* **DONE this session:** plotting the tuned settings (notebook 15). The outcome-blind table lives in
  `experiments/phase15b_pattern_controls.FINALISTS` and draws with `notebooks/14_pattern_viewer.ipynb`.
  It is unaffected by the retractions, because those settings were selected on sample size rather
  than on score.
* Rewrite the pattern sections of `STRATEGY_SEARCH_LOG.md` — `WHAT_WORKS.md` is already corrected,
  the phase-by-phase log is not.
* Cross-instrument for chart patterns still needs more gold data to be answerable (1 of 9 windows
  clears the 40-trade floor).
* The parked ideas in `memory/parked-strategy-ideas.md` remain parked.
