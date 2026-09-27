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

## Correctness pass on the detector (2026-08-25/28)

Driven by the shape-quality audit and the contact sheet. **Five defects**, each surfaced by a
MEASUREMENT rather than by reading code, with the test suite green throughout — every existing test
asserted an invariant and none asserted that a shape resembles its own definition.

1. **Converging shapes drawn and confirmed past their APEX.** Beyond it the upper boundary sits below
   the lower one and the shape is inverted. `max_apex_frac` (0.75) constrains exactly the right family
   for free: channels are parallel (no apex) and broadening shapes met theirs in the past.
2. **Shapes that did not contain their own price action.** The apex limit alone does not bite when
   convergence is slow. `min_containment` (0.80) is the direct statement instead of a proxy, checked
   at the break where every bar is already past, so no lookahead. Worst containment **0.309 -> 0.810**,
   shapes below 0.80 **18.3% -> ZERO**, and detections went slightly UP because rejecting a malformed
   shape lets a better overlapping one survive de-duplication.
3. **Requesting a subset of patterns silently lost shapes.** The flag rename ran unconditionally, so a
   channel relabelled `bull_flag` was dropped by the `not in hits` filter instead of staying a
   channel. Channels-only returned 21 where channels+flags returned 21 channels plus 36 flags, and the
   count moved with `pole_mult` — a parameter unrelated to channels. Now stable at 55.
4. **TUNED_SPECS covered 16 of 22 patterns**, so `chart_pattern_shapes_tuned` never requested
   rectangles, flags or pennants and they showed zero detections — not because none exist but because
   nothing asked for them.
5. **Degenerate stops.** On a long, steeply rising wedge the rising lower boundary catches up with the
   highest high, so the stop landed a hair from the break: reward-to-risk **585:1** on gold. Being on
   the correct side of entry was tested; being a meaningful DISTANCE from it was not. `min_risk_frac`
   (0.05), which is equivalent to capping R:R at 1/min_risk_frac since the target is one height away.

Trendlines also gained identity: a refit used to spawn a whole new line record with a reset `born_i`,
which is why 16% of lines lived under 10 bars. Median life **55 -> 92 bars**.

### Audit verdict after the fixes

PASS on **BTC 15m, BTC 1h and XAUUSD 15m** — 4,077 shapes. Zero causality mismatches over 12
truncation points x 9 windows x 3 instruments, contract 100%, zero trendline invariant violations.
**Scale invariance confirmed**: per-pattern median containment differs by at most 0.006 between BTC
near $60,000 and gold near $2,000.

### Recalibration

The new gates reject shapes the old settings were calibrated to produce, so only 7 of 22 patterns
still cleared the >=10-per-window floor. `experiments/recalibrate_tuned_specs.py` re-derives by the
identical outcome-blind rule (count only, never forward returns). **11 of 22 clear it**; the other 11
are marked `# rare in BTC 15m` and set to the most permissive combination — reported as rare rather
than loosened until they appear, because manufacturing detections by relaxing geometry is the failure
this whole line of work exists to prevent.

`TUNED_COMMON` shrank to the knobs the pooled scan genuinely fixed for everything
(`min_touches`, `line_tol_atr`); `min_height_atr` moved into each entry, since 11 of 22 override it
and a "common" constant half the entries contradict is a lie about the data.

**The phase 12-15 research numbers are now STALE** — they were measured before these gates. Given the
verdict there was comprehensively negative and the goal here was correctness, they were not re-run.

## Drawing defects found from a rendered chart (2026-08-29/30)

The user opened a chart and found three things wrong that the audit had passed. Six defects behind
them, and — worth recording — **two of the three "failures" the new audit checks first reported were
flaws in the CHECKS, not the code.**

### The six defects

1. **Trendlines drawn from `born_i`**, which is after every pivot they are fitted through, so ALL 36
   lines in the sample had their touch dots outside the drawn segment. Now drawn from the first touch.
2. **Zero-life lines.** Several pivots can confirm on one bar, so a line could be born and superseded
   before ever publishing. Those were still drawn across 343 bars; the worst sat **71 ATR** from price.
   A line that never published is now dropped.
3. **Broken lines resurrecting.** A retired line was refit from the same pivots a bar later, which is
   why consecutive lines shared a `start_i` and differed in slope by under 5%. `retired` now
   remembers touch-sets: "a broken line is retired" has to mean it stays retired.
4. **Converging shapes did not close.** Boundaries stopped at the break, so a triangle read as two
   lines heading toward each other. They now extend to the APEX.
5. **Level-shape dots on the wrong side.** A triple top's pivot list carries its three highs AND the
   two troughs that set the neckline; all were drawn as touches, putting **33-40% of dots on the low
   of a candle**. `_touch_pivots` now keeps only the side the pattern is named for.
6. **"Equal" tops that were not.** Equality was checked only between ADJACENT pairs and scaled by the
   whole pivot window's range, letting three tops sit **14.8 ATR apart**, 65% of the shape's height.
   Now the FULL SPREAD must fit inside `level_tol_atr`, whose default is DERIVED not tuned: a level
   is a horizontal trendline, so 2 x `line_tol_atr`.

Plus one real anomaly the new checks caught: **support and resistance could CROSS** — one bar in
~156,000 had support above resistance, letting a single bar break both ways. The weaker line now
gives way.

### METHODOLOGY RULE 12 — audit a rule in the rule's own frame of reference

The new checks reported 8 dots off their line and 38 lines far from price. Measured correctly: **0
and 0**. The dot check used the ATR at CONFIRMATION when the fit used the ATR at FORMATION; the
distance check used the window-median ATR when the retirement rule uses the local one. This is the
same mistake the audit already made once with the contract frame, three times in total. Getting the
frame wrong makes a check STRICTER than reality and manufactures defects — the safe direction to be
wrong in, but it cost two rounds of investigation.

### Audit state

PASS on BTC 15m, BTC 1h and XAUUSD 15m: 3,508 shapes, **21,763 drawn dots with zero off their line
and zero on the wrong side**, zero causality mismatches, all trendline invariants zero. Two new
checks now gate the verdict — every drawn dot must sit on what it marks, and every drawn line must
stay near the price it describes. 317 tests.

## In flight — trendline settings search

`experiments/trendline_settings_search.py`, results read by `notebooks/16_trendline_research.ipynb`.
144 settings x 3 timeframes (1h/15m/5m) x 2 instruments x 9 windows, ~3 hours.

Four stages kept apart: QUALITY (outcome-blind, and the only stable answer), SCREEN (null-differenced
with a matched control), MULTI-TIMEFRAME, CONFIRMATION on the second instrument.

Three sizing decisions came from measurement rather than assumption, and are worth keeping:

* **The obvious MTF test is unanswerable at this event rate.** Requiring a break on BOTH timeframes
  gives 0-1 events: 1h yields ~6 up-breaks in six months. Reporting that as a null result would have
  been reporting arithmetic as a finding. The higher timeframe now supplies a STATE (its line sloping
  with or against the break), splitting hundreds of events.
* **1m is excluded.** Its cache covers only part of the nine-window span, so it turned a compute job
  into 1,052 network pages per window — four hours of fetching before any measurement — against a
  10.8%-of-R cost floor no entry family here clears. `prewarm()` now touches every cache up front.
* **The grid was cut 324 -> 144** after measuring 0.22s per detection per 4,400 bars: at 324, stage 1
  alone ran 5 hours. Cheap to cut, because the argmax of the score surface is known to be noise.

## Trendline settings search — RESULTS (2026-08-30)

`experiments/trendline_settings_search.py`, 144 settings x 3 timeframes x 2 instruments x 9 windows.
Ran in **72 minutes**, against a 3-hour estimate — the cost model built at 324 combinations
over-predicted, so trust the measurement over the extrapolation next time.

### Settings (stage 1, outcome-blind) — stable, but read the caveat

Every selection converged on `min_touches=4, line_tol_atr=1.0, min_span_bars=80`, with `atr_mult`
varying by timeframe: **1h -> 2.5, 15m -> 3.5-5.0, 5m -> 5.0**. Slower timeframes want a finer zigzag
because they have fewer bars to find swings in.

**CAVEAT, and it matters: `min_touches=4` was chosen by the tie-break, not by being better.** The
selection rule takes the STRICTEST setting that still clears the sample floor, and 4 is stricter than
3. On the gate itself 3 passes **80.1%** of the time against **53.5%** for 4. So the honest label is
"the strictest that still works", not "the best". `min_span_bars=80` and `min_touches=4` are also the
maximum values in the grid, so stricter settings were never tested.

Pass rates by knob: `atr_mult` 2.5=92% / 3.5=70% / 5.0=39%; `line_tol_atr` 0.4=54% / 1.0=80%;
`min_span_bars` and `break_buffer_atr` are flat (66-67%), so neither is doing much.

### Direction (stage 2) — dead, consistent with everything else here

* hit rate above 0.50 in **36% of 102 cells** — BELOW the 50% chance rate — median exactly **0.5000**
* 56 of 102 cells beat their shuffled null: **55%, chance 50%**
* positive AND consistent in >=8 of 9 windows: **2, chance expectation 2.0**
* worst on 5m: 22.5% of cells above a coin flip, median hit 0.4896

### Multi-timeframe (stage 3) — INCONCLUSIVE, and must not be reported as negative

The redesign (HTF slope as a state rather than requiring two breaks) still lacks power: a
higher-timeframe line is live on only **3-22% of bars**, so classifying lower-timeframe breaks by it
produced **5 classified events against 1,705 unclassified**. The question is unanswered, not answered
in the negative. Answering it needs either a state that exists on most bars (an EMA slope, a
structural bias) or far more data.

## Chart viewer (2026-08-31) — notebook 15, and three lessons worth keeping

`quant.viz.pattern_figure` gained the controls needed to actually READ a dense chart. What matters
here is less the features than what building them revealed.

### WARM-UP HELPS MOST INDICATORS AND HURTS S/R ZONES

Loading history so trendlines, zones and a 200-EMA have something to stand on is right for almost
everything — and wrong for S/R zones. The zone tracker keeps the `max_zones` most-touched levels,
and in a trending market those are all OLD. Measured on BTC 5m running 66k to 82k with a two-month
warm-up: **1 of 40 zones was anywhere near the price on screen**. With a 2,000-bar lookback, 38 are.
`zone_lookback_bars` therefore builds zones from a BOUNDED history ending at the view, deliberately
different from every other indicator on the chart.

Two related traps in the same code: `max_zones` caps zones held SIMULTANEOUSLY, not rows — each zone
emits a row per price revision, so 40 zones produced 1,366 segments and the legend read "S/R zones
(767)". Drawing every row also stacks a zone's whole history of slightly-different levels; only the
LAST revision is the level in force.

### WARM-UP SHOULD BE COMPUTED, NOT DRAWN

The browser was handed all 20,161 candles to display 2,593. Everything is now calculated on the full
frame and drawn from `view_start` — 87% fewer points, plus patterns that confirmed entirely inside
the warm-up and trendlines that died before the window opened are dropped.

### PLOTLY CAN CASCADE, THROUGH THE LAYOUT PATCH

A dropdown of 130 entries is no more readable than the overlapping lines it replaces. Plotly has no
native cascading menu, but a button's `update` method carries a LAYOUT patch alongside the trace
patch, and a menu's buttons are a layout property — so the family menu rewrites
`updatemenus[1].buttons`. Picking "support trendline" leaves the instance menu showing only support
trendlines.

Also added: EMA overlays (on by default — they were previously added as `legendonly`, so they
appeared in the legend and nowhere else), and a STAR at each pattern's confirmation bar with the bar
count since its last pivot in the hover. That lag is large and was invisible: median 15 bars on BTC
15m, about 42 for a double top.

### A PROCESS NOTE THAT COST TWO ROUNDS

The generated notebook was overwritten twice by the user's own edits (open in Jupyter, dates changed,
saved). On top of that, a builder patch anchored on a line ending `fig` silently did nothing because
the cell ends `# fig` — `str.replace` does not complain when it matches nothing. **Patch notebooks IN
PLACE and assert every anchor.** Regenerating destroys the user's settings; unasserted replaces
destroy the edit without telling you.

## Multi-timeframe patterns on one chart (`tf_frames=`)

`pattern_figure(df_15m, tf_frames={"4h": df_4h, "1h": df_1h})` draws other timeframes' shapes and
trendlines over the chart you are watching. Load the frames with `quant.data.tf_frames`.
**`notebooks/17_multi_timeframe_patterns.ipynb`** is the ready configuration. It is a NEW notebook
rather than more edits to 15, because a running Jupyter kernel saved its in-memory copy of 15 over
the committed one three separate times this session.

**Why it is additive rather than a rewrite.** Everything `pattern_figure` emits is already
`(timestamp, price)`, and a fitted slope is only ever evaluated at an index *of the frame it was
fitted on*. So each layer keeps its own `t` array, its own view index and its own bar count, and the
timestamp is the only coordinate the frames share. No index remapping exists anywhere in the feature.
The chart layer's traces come out byte-identical with extra layers loaded — asserted in
`test_extra_timeframes_leave_the_chart_timeframes_own_drawing_untouched`.

**Causality.** Bars carry their OPEN time everywhere in this project, so a 4h pattern confirming on
the bar stamped 12:00 is not knowable until 16:00. Events (the confirmation star, the trendline break
X, the "line confirmed" diamond, the right end of an extended line) are stamped at `open + interval`;
geometry (left ends, touch pivots) stays at the open, where the candle sits. The interval is
`median(diff(t))` of the frame itself — there is no timeframe-string parser in the drawing path, so a
relabelled or gappy frame still reports the truth. Same discipline as `align_timeframes(shift_bars=1)`.

**The engineering problem was payload size, not drawing.** Dropdown buttons carry a `visible` vector
as long as the whole trace list, so the menu costs *buttons x traces*. Naively adding four timeframes
to the nine-day BTC 15m window gave **1,027 traces x 479 buttons = 9.8 MB of dropdown JSON and an
11.4 s build** — a chart that does not open, which is worse than one that is cluttered. Three changes
brought it to **447 traces / 67 buttons / 1.23 MB**:

* **three cascading menus** — timeframe, then pattern, then occurrence. The first attempt folded the
  timeframe into the family label to avoid the nesting cost, but that left one 51-row list, and
  **plotly menus do not scroll**: they render every button and clip the overflow, so at 969px in a
  900px figure the last families could not be reached. Cascading keeps each menu short by
  construction. It is affordable because the wide `▸ all timeframes` state lists no families at all
  (it prompts for one) and each occurrence appears under its own family only;
* **multi-select is the LEGEND, not the menus.** A plotly dropdown is single-choice, so choosing a
  family promotes its occurrences to individual legend entries — that is what lets you keep two of
  five channel-ups with the EMAs untouched;
* **`legend_instances=` is the only way to select ACROSS families.** Menu buttons are STATELESS:
  each press writes a whole state, so choosing the 5m family destroys the 15m pick and *one 15m
  resistance line plus two 5m ones plus a few 1m shapes* is inexpressible in the menus at any depth
  of cascade. Pinning those families instead puts every one of their occurrences in the legend
  permanently, each starting off. Legend entries toggle independently, accumulate, and the menus are
  written never to drop a pinned one — so any combination can be clicked together. Keys are
  `"name"` (chart timeframe, matching `patterns=`), `"5m::name"`, `"*::name"`, `"5m"`, or `"all"`,
  with spaces and underscores equivalent because patterns use one and trendlines the other. Pin only
  what you are working on; `"all"` is a legend hundreds of rows long. Notebook 17 prints the keys;
* **each instance is listed once.** It used to appear in the initial menu, under its timeframe and
  under its family; three copies of the dominant term was 1.1 MB of the 1.6 MB. Wide selections now
  list families, and the family itself lists its instances;
* **the caps bound what is DRAWN and what is LISTED together.** They used to trim only the menu,
  which left a legend reading `(11)` beside a dropdown offering nine and two lines on the chart that
  could not be isolated at all. The cap is now derived from the family count BEFORE anything is
  drawn, so the two numbers cannot disagree. `max_picker_instances` (60) is shared across a layer's
  families with a floor of one each, so it steers the total rather than capping it exactly;
  `max_shapes_per_tf` (40) is the per-family ceiling. The chart's own layer is never trimmed.

With the cascade and 1h + 5m + 1m loaded: **461 traces, menus of 5 / 3 / 1 rows, 1.39 MB**, and the
tallest menu you can ever open is 21 rows.

**Measured cost** (BTCUSDT, view 2026-06-01 -> 2026-06-10, 800 bars warm-up per timeframe):

| tf | bars | detect | shapes | confirming in view | build with it added |
|---|---|---|---|---|---|
| 4h | 859 | 0.15 s | 7 | **0** | — |
| 1h | 1,033 | 0.18 s | 8 | 3 | 5.5 s (with 4h) |
| 15m (chart) | 1,729 | 0.46 s | 24 | 11 | — |
| 5m | 3,457 | 0.90 s | 29 | 23 | 6.9 s |
| 1m | 14,401 | 4.88 s | 174 | **155** | 16.7 s |

**Coarse timeframes need a WIDE view; fine ones need a NARROW one.** A 4h shape spans 100-200 4h
bars = 17-33 days, so over a nine-day window 4h confirmed **nothing** — that is arithmetic, not a
bug. At the other end 1m confirmed 155 shapes over the same window, which is the clutter rather than
a feature. `PATTERN_TFS = ("4h", "1h")` is the useful default; reach for 1m/5m on a one- or two-day
window.

**Higher-timeframe EMAs (`mtf_emas=`).** Same spec as `scenario_figure` — `("1h", 200)`,
`("1h", [50, 200])`, or a list of pairs. Drawn as a **step** (`line_shape="hv"`) that changes only at
its own bar's close, for the same reason the event markers do: that is the only moment the value is
final, and an interpolated 1h EMA shows numbers nobody could have read. A timeframe already in
`tf_frames` is reused rather than resampled, which is not cosmetic — an EMA needs `period` bars of
its OWN timeframe, and a three-week 15m chart resamples to only 126 4h bars, so a 4h EMA200 came out
**entirely NaN**. That now raises a warning naming the shortfall instead of silently drawing nothing,
because a missing line looks exactly like one you forgot to switch on.

**`y_from_price` (new, on by default).** A 1h falling wedge that began before the window ran from
86,000 down to a 51,800 apex, and plotly's autorange stretched a 60k-76k chart out to 50k-85k,
leaving the price action a third of the pane. The line was correct — it really is that tall. The
y-axis is now scaled to the candles on screen. Pass `y_from_price=False` for the old behaviour.

## Multi-timeframe trendline trading simulation (notebook 18)

Trades fitted support/resistance lines from 1m, 5m, 15m and 1h on a 1m chart. Pieces:
`quant.indicators.mtf_trendlines` (fit per timeframe, merge with `shift_bars=1`),
`mtf_trendline_long/short@v1` in `quant/scenarios/entries.py`, `trendline_trigger()` for
attribution, and `pattern_figure(trades=..., trade_tp_rr=...)` for the overlay.

**The bug worth remembering: a line is RETIRED the moment it breaks.** So "two closes beyond the
line" asked the second bar to compare itself against a level that had just become NaN, and NaN
comparisons are False. Measured: the base timeframe fired **0** times at `tl_close_n=2` while the
merged higher timeframes, whose values persist across their own bar, fired hundreds — the rule
silently meant something different on the frame it was fitted on. The entry now holds the last
published value for exactly the length of the confirmation window (a past value, so still causal),
after which 1m went 0 -> 48 long / 56 short over the same window.

**Result on BTCUSDT 1m, 2026-06-01 -> 2026-06-10**, `n=2` closes, stop at the 20-bar extreme,
breakeven at 1R, target 3R, `fee_bps=8` + `slippage_bps=1.5`, 1% risk per trade:

| | trades | win | PF | total R | return |
|---|---|---|---|---|---|
| long | 18 | **0.0%** | 0.00 | -19.2 | -10.67% |
| short | 26 | 23.1% | 1.17 | -2.7 | +1.42% |
| both | 44 | 13.6% | 0.48 | -21.8 | **-9.25%** |

**The cost floor is the story, and it is measurable on these very trades.** Median risk was 0.568%
of price, so 1% equity risked is a ~1.8x notional and a round trip costs **0.28R**. Break-even win
rate `p* = (1 + 0.28) / (1 + 3) = 32.1%`; realised **13.6%**. The price move on a stop was a clean
-1.02R, so the engine is doing exactly what it should — the losses are the rule and the costs, not
the mechanics. Tightening the stop makes this WORSE, not better: at `STOP_N=20` on a shorter warm-up
the risk was 0.366% and the cost went to 0.44R.

This has NOT been through the phase-12 screen — no shuffled null, no metronome, no bootstrap, no
trim-top-K — and `WHAT_WORKS.md` already records trendline breaks as carrying no directional edge
(hit rate above 0.50 in 36% of 102 cells, below chance). Treat the numbers above as a description of
one window.

### Naming every object on the chart

Reading the chart was guesswork: four concatenated traces meant the stop you hovered could belong to
any of forty-four trades, and "which line caused this" was not answerable at all. Everything now
carries an id, and all of them come from ONE place so they cannot drift:

* a **line** is `5m·S7` — `trendline_table` gained an `id` column (`R0`, `S0`, … per kind, in
  publication order) and the figure prefixes the timeframe;
* a **pattern** is `1h·rising_wedge#2`;
* a **trade** is `#12`, and its entry, entry level, stop, target and exit all hover with it.

`add_trendline_ids` adds `tl_res_n` / `tl_sup_n` — which line is live on each bar — merged across
timeframes like the levels. `trendline_trigger` reads it at the entry bar and returns a `cause`
column naming the exact line. It needed the same hold as the level: the line retires on the bar it
breaks, so at an `n=2` entry its id was already gone and every base-timeframe trade reported `?`.

Two more dropdowns (longs / shorts / winners / losers, then a trade) isolate **one trade plus the
line that caused it** — nothing else on the chart. Rows read `#07 S 1m·S59  -2.88R sl`. A line named
as a cause is exempt from `max_picker_instances` and `max_shapes_per_tf`: the caps exist to keep the
figure openable, but not at the price of hiding the answer the picker exists to give. Cost: 693
traces and 3.6 MB with 40 trades and four timeframes loaded.

### How long to wait before trading a break (`experiments/trendline_wait_search.py`)

`TL_CLOSE_N` was one number for every timeframe, which cannot be right: two 15m closes is half an
hour and two 1m closes is two minutes. It now takes a dict keyed by timeframe, resolved by
`tl_per_tf()`, and the same applies to `tl_full_m` and `tl_touch_k`.

Swept on BTCUSDT 5m + 15m + 1h lines, **5.5 months in 6 windows**, each timeframe alone so a good
15m number cannot be an artifact of 5m trades that happened to be open. Scored by **windows
positive**, not total R.

| tf | side | trades | windows + | total R | |
|---|---|---|---|---|---|
| 5m | long | 53 | 4/6 | -2.0 | wait=4; R rises monotonically from -33.8 at wait 1 |
| 5m | short | 54 | 4/6 | +1.3 | wait=4; -8.0 at wait 1 |
| 15m | short | 19 | 4/6 | +9.4 | wait=5, thin |
| 15m | long | 35 | 3/6 | -9.2 | negative at every wait |
| 1h | either | 9-11 | 1-3/6 | - | too few trades to infer anything |

**Nothing cleared 6 of 6**, and six settings over six windows produce a 6/6 winner by luck about 9%
of the time, so no wait is validated. The one thing worth keeping is the **monotone trend on 5m**:
longer waits improve total R *and* per-trade R (-0.32 -> -0.08 for longs), so it is not merely fewer
trades paying fewer fees. Hence the notebook default `{"5m": 4, "15m": 5, "1h": 4, "*": 2}`, with 1h
matched to the others rather than set to its own 11-trade argmax.

The sweep says something louder than any wait: **longs lose at every wait on every timeframe** and
shorts do not. That is a downtrending sample, not an edge.

### Two red lines

The isolated trade view drew entry, stop and target as ONE trace, so a losing trade rendered all
three in red and the reader reasonably concluded the stop was below a short's entry. Each level is
its own trace now, in the grouped view's colours: grey entry, red stop, green target.

The same screenshot showed the second half of the confusion. Trade `#05` was a short with entry
63,203 and a stop at 64,010, and it exited at 63,213 — 1.2% of the way to that stop. Nothing was
wrong: `BE_AT_RR=1.0` had walked the stop to breakeven after the trade reached +1R, and the chart
draws the stop **as set at entry**. Such trades are now tagged `sl@be` in the picker and say so on
hover.

Notebook 18 was cut from 26 cells to 10 — config, run, chart, results, one appendix.

### Candle colour is not the same condition as closing beyond the line

`tl_close_n` counts closes past the trendline. That is not "n red candles": a candle can close
below a support and still be green, and on BTC 5m over this window **30 of 72 short entry bars were
not red at all** — the entry bar is the *n*th close past the line, several bars after the break, and
by then the candle is often a green pullback bar.

`tl_dir_n` counts candles of the trade's own colour (red for a short, green for a long), per
timeframe like the rest, joined by the same `tl_combine`. On BTC 5m + 15m + 1h lines, short entries
went 72 -> 42 -> 18 -> 8 as `tl_dir_n` rose 0 -> 1 -> 2 -> 3.

**Whose candles?** `tl_dir_src="chart"` counts the execution frame's; `"line"` counts the
timeframe the line was fitted on, so a 15m line can demand three red *15m* candles. That needed
`add_candle_runs()` in `quant/indicators/candles.py`, which returns the RUN LENGTH rather than a
fixed-n predicate: `consecutive_red(df, 3)` on a 5m grid counts 5m candles no matter whose line you
are trading, whereas a run counted on the 15m frame and merged through `align_timeframes` is
measured in 15m bars and stays invisible until that bar closed. Both directions of the source are
looser or tighter for a real reason — the line's own clock is slower, so `"line"` fires more often
than `"chart"` at the same n.

### Extending a broken line, and asking for colour AND clearance together

`tl_dir_full_n` wants n candles that are BOTH the trade's colour AND wholly clear of the line, wicks
included -- a red candle whose HIGH is still under the support. On BTC 5m short entries: 74 -> 39 ->
15 -> 5 at n = 0/1/2/3. Under `tl_combine="all"` it is exactly `tl_full_m` and `tl_dir_n` set to the
same number; it exists as one condition so it survives `"any"` and can be OR-ed against something
else (82 entries vs 1,431 for the split spelling under `"any"`).

**`tl_extend` (new, default True) is the part that was quietly wrong.** A line is retired the moment
price closes through it, so any multi-bar confirmation has to be told where the line WOULD have been
-- previously a flat `ffill`. A sloping line does not stay flat: a rising support keeps rising, so
over the three or four bars a rule spans, a flat hold makes the test lenient on one side and strict
on the other. The held value is now extrapolated along the line's own `{prefix}_{sup,res}_slope`.
It reads only already-published values, so it is still causal (truncation test).

This DOES move existing numbers slightly -- short entry bars went 72 -> 74 on the notebook's window,
because the level a rule is measured against now moves during the hold.

### "The move stopped convincing me" — a trade that arms itself

`entry_retest_bail@v1`. A trade that goes your way and then keeps coming back to where you got in is
oscillating around your entry, not trending, and the 3R target it was given is a bet on a move that
is not happening. Arm on either count and hand it to "exit on the first opposite candle while in
profit":

* `sec_touch_n` — bars since entry that traded back THROUGH the entry price;
* `sec_wrong_m` — candles formed WHOLLY on the losing side of it.

**Both had to go in the kernel.** Every other arming condition here is a global signal array, but
the level these measure against is *each trade's own entry price*, which no such array can express —
it is why `shaky_trend_bail` settles for watching an EMA instead. `o_touch` / `o_wrong` sit beside
`o_armed`, are copied on compaction, reset at entry, and the whole block is skipped when both counts
are 0 (golden parity holds).

**"In profit" was not enough.** The engine's gate is "the fill is on the favourable side of entry",
and on BTC 5m that let three of ten bail-outs exit for LESS THAN THE ROUND TRIP: one moved 0.92 in
price against 15.39 of fees, closing at -14.47. `alt_min_pct` is the move the exit must clear, as a
percent of entry. A round trip at `fee_bps=8` + `slippage_bps=1.5` is 19 bps, and `sec_min_pct=0.2`
took the net-negative bail-outs from 3 to 0.

On the notebook's window: 9 trades / +3.4R without it, 15 trades / -3.6R with it at
`sec_touch_n=2`. More trades because closing early frees the single slot sooner. Nine days and two
configurations is not evidence either way — it is off by default.

### EMAs stay on while one trade is isolated

Already true: `_isolate_trade` never touched traces whose legendgroup starts `ema`/`mtfema`, and a
test now pins it across every trade row. The reason none showed was that notebook 18 passed
`emas=()`. It now sets `EMAS` and `MTF_EMAS`.

### Settings moved OUT of notebook 18, because the generator kept overwriting them

Notebook 18 is regenerated from a build script, and that script hardcoded every setting. So every
time a new one was added, the rebuild re-imposed all the others. Commit **8a8b8de** did exactly that
while "making the notebook clean": `TF` went `1m` -> `5m`, `LINE_TFS` `("5m","15m","1h")` ->
`("15m","1h")`, `WARMUP_START` `2026-05-25` -> `2026-05-01`. None of it was asked for, and none of
it was mentioned.

Settings now live in `notebooks/nb18_config.py`, which nothing regenerates. The notebook imports it
with `importlib.reload`, so editing the file lands without a kernel restart — and as a side effect
Jupyter can no longer clobber the settings when it saves, which is the other way they have gone
missing in this project.

`tests/test_notebook_config.py` pins the rule that makes this hold: **the notebook may READ a
setting and must never STATE one.** It parses every code cell and fails on any assignment to a name
the config owns, so the same mistake cannot be reintroduced quietly. It also asserts each of the
three changed values has exactly one home, parsed with `ast` rather than matched by regex so prose
about them cannot count.

The three values were left where they ended up rather than silently reverted — changing them back
without asking would have been the same mistake twice. The config's docstring names them and the
originals.

### A one-line rule language for the notebook config

`quant/strategies/rulespec.py`. `ENTRY_RULES` in notebook 18 is now a list of sentences:

```
"long:  2 candles EMA_5m_200 close above"
"short: 3 candles EMA_15m_50 formed fully below red"
"1 candle SUP_15m touched"
```

The `Cond` algebra could already express every one of these — `(full_above(ema(200), n=3) & red(3))`
— but only as code, and a config is not the place for code: that expression is three ideas spelled
as one, and changing a timeframe means editing it rather than a token.

Grammar: `<n> <reference> <close|formed fully|touched> [above|below] [red|green|NA]`, order-free
with filler words ignored, so `"red 3 the EMA_5m_200 is formed fully below"` parses identically. A
`long:` / `short:` prefix scopes a rule to one side. References are `EMA_<tf>_<period>`,
`RES_/SUP_<tf>`, or any column already on the frame; `resolve_ref` maps them, with the chart's own
timeframe unprefixed (`EMA_5m_200` is `ema_200` on a 5m chart and `5m__ema_200` on a 1m one).

Three decisions worth keeping:

* **`touch` refuses a side.** A touch is the candle straddling the level; "touched above" would be a
  different test, and silently picking one reading is worse than refusing.
* **`build_rules` returns None when no rule applies**, rather than an empty `all` (vacuously true) or
  an empty `any` (never true). Guessing either would turn "I wrote no short rules" into "no short may
  ever trade".
* **A bad sentence raises, naming the problem.** A rule that quietly matches nothing filters every
  entry away and looks like a strategy that found no setups.

`required_emas()` says which EMA columns a rule set needs, and `add_mtf_emas()` (new, in
`indicators/overlays.py`) adds them from the LOADED frames rather than by resampling like
`build_mtf` — an EMA200 wants 200 bars of its own timeframe, and a short base frame silently yields
NaN. Causal by the same `shift_bars=1` rule as everything else, with a truncation test.

Watch one thing when reading results: with `max_open_trades=1`, filtering ONE side can **increase**
the total trade count, because a slot the filtered trade would have taken is now free for the other.

### Where the breakeven stop came from

Nobody asked for it explicitly. "Use the best known setting" was read against `WHAT_WORKS.md`'s
"breakeven at 1-1.5R with a 3R target, +0.0095R over a plain 3R target", so `be_ladder@v1` came in
at its default `be_at_rr=1.0`. The notebook said "breakeven at 1R" without saying that this MOVES
THE STOP, which is why a trade closing as `stop_loss` with its exit at the entry level read as a
broken chart. Now stated in the config.

Measured on the notebook's own nine-day window it cost **-3.55R over 8 trades**, but -3.00R of that
is a single trade (#04, which reached 1.52R before reversing). Over the same 6 windows and 5.5
months the wait search used, 129 trades: **+6.2R total, +0.048R per trade, helping in 4 of 6
windows** — same sign as the documented number. Nine days says nothing either way; the expected
value of the rung over 8 trades is about +0.08R, far smaller than one trade's swing.

## Not done / next

* The multi-timeframe question is still OPEN — see the power problem above. If it is worth pursuing,
  use a higher-timeframe state that exists on most bars rather than one that exists on 3-22% of them.
* Trendline settings could be swept STRICTER: `min_touches=4` and `min_span_bars=80` are the grid
  maxima and were selected, so the useful range may lie beyond what was tested.

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
* Multi-timeframe patterns are a DRAWING feature only. Nothing about whether a 4h shape confirmed
  under a 1h trend is worth trading has been measured, and the phase-12 screen has not been run on
  any cross-timeframe combination. Expect nothing until it has been.

---

## An EMA of a timeframe finer than the chart was the chart's own EMA

`MTF_EMAS = [("1m", ...)]` on a 5m chart drew a "1m EMA 200" that was the **5m** EMA 200. Measured
on BTC, 2026-05-01 → 2026-06-10:

```
the 5m chart frame has 11,521 bars
resample_ohlcv(5m frame, '1min') -> 11,521 bars     # upsampling cannot invent bars never loaded
'1m EMA200' vs the 5m EMA200: identical on 100.0% of bars, max difference 0.000000
a REAL 1m EMA200 is computed on 57,601 bars
  at the last 5m bar:  fake 62,096.11   real 61,786.14   5m 62,096.11
```

310 points out, with nothing on the chart to say so. `pattern_figure` fell back to
`resample_ohlcv(chart_frame, ...)` whenever `tf_frames` had no frame for a timeframe — sound going
*up* (5m → 15m), meaningless going *down*. It now **refuses** the downward case by name and takes
`ema_frames={"1m": df_1m}` for it; `mtf_ema_timeframes(spec, chart_tf=)` says which frames a spec
needs, and notebook 18 loads exactly those. Those frames are kept **out of `FRAMES`** on purpose:
`FRAMES` is what trendlines are fitted on, so adding 1m there would have started trading 1m lines.

**The general shape of this bug:** a fallback that is correct in one direction and silent in the
other. The fix is not a better fallback, it is refusing the direction that cannot work.

## Two plotly defaults that fight the trade picker

- `legend.itemdoubleclick` defaults to `"toggleothers"` — a double-click **isolates** that entry.
  With one trade isolated, double-clicking an EMA hid the trade and the trendline that caused it.
  Set to `"toggle"`, so a double-click does what a single click does.
- That default was also the only way to turn every EMA off at once, so replacing it needed a
  control: an `EMAs on / off` button pair (`method="restyle"` over just the EMA trace indices, so it
  costs nothing and cannot disturb anything else) plus `emas_on=` / `EMAS_ON` for the starting
  state. `emas_on=False` draws every EMA `legendonly` — still listed, still individually clickable,
  which was the point. Every picker button inherits it, because their `visible` vectors are built
  from each trace's initial `visible`.

The buttons menu is appended **last** so the existing absolute `updatemenus[4]` references in the
trade cascade do not shift. Four tests were locating that menu as `updatemenus[-2]`; they now find
it by content, and a button's layout patch is read through a helper because a `restyle` button puts
trace indices in `args[1]` where an `update` button puts the layout dict.

## Appending a setting is safe against a rebuild, not against an open editor tab

`EMAS_ON` was appended to `nb18_config.py` and committed. It was gone from the working file an hour
later, and the notebook died on `NameError: name 'EMAS_ON' is not defined` — while the user's own
edit to `ENTRY_RULES`, made in the same file, was present. An editor tab holding the pre-append
version had saved its buffer back over the file, taking the tail with it.

Appending is still right: it cannot touch a line the user has edited, which was the 8a8b8de failure.
But it is not enough on its own, because the file the user is looking at may predate the append. So
cell 2 now carries `SETTING_DEFAULTS`: any setting `nb18_config` does not define falls back and the
notebook prints the line to paste in. `tests/test_notebook_config.py` checks each default names a
setting the notebook actually reads AND that the config file has a line for it — a default with no
home in the config would be a second place a value is stated, which is the original bug again.

---

## Fair value gaps: three bugs the fixtures caught, and two that cost 4x

**`top` and `bot` written in candle order inverted every bearish gap.** A bullish band is
`(high[i-2], low[i])` and a bearish one `(high[i], low[i-2])` — transcribing both in that order puts
the LOWER number in `top` for bearish gaps. An inverted band inverts its own mitigation test, its
midpoint and the box drawn for it, and none of it looks wrong on a chart. Caught by a hand-built
three-candle fixture asserting `bot < top`, which is now a test.

**Performance: two "optimisations" that made it 4x slower.** Vectorising the gap ids with
`np.where(...) + cumcount().to_numpy().astype(str)` and the time columns with
`tv[...].astype("datetime64[ns]")` took `add_fvgs` on 230k bars from 1.08 s to **2.70 s**. Cause: a
tz-aware column comes back from `.to_numpy()` as an **object array of Timestamps**, so `.astype` was
converting 58,776 Python objects one at a time — 2.6 s of a 3.4 s call, more than detection,
mitigation and the per-bar pass combined. `pd.DatetimeIndex(...).take(idx)` stays in the C path.
Final: **270 ms**. The lesson is narrow and worth keeping: on a tz-aware column, `.to_numpy()` is a
Python-object trapdoor.

**numba allocates on every `np.full` inside a loop.** `_live_state` used a two-element array per bar
for "nearest gap per direction" — 460k heap allocations on a 1m year, and the single biggest cost in
the function. Scalars instead.

## Drawing hundreds of boxes without hundreds of traces

`fill="toself"` on a Scattergl takes any number of `None`-separated polygons, so a whole layer of
gaps costs **one trace** (plus one invisible centroid-marker trace for hover — a filled polygon only
reports its outline, so hovering the middle of a box said nothing). That matters more here than
anywhere else on this chart: every picker button carries a `visible` vector as long as the trace
list, so cost is buttons x traces. Measured: 241 -> 353 traces and 0.72 -> 1.29 MB of JSON with four
gap layers on, and no measurable change in build time — `_best_line` in `chart_patterns.py` is 4.3 s
of the 4.9 s and dominates everything.

Two ordering facts, both learned the hard way: gap boxes must be added **before** every other layer
or plotly paints them on top of the trendlines they are background for; and the picker's family key
must be the **prefixed** label, because "FVG bullish" is the name of four different families at once
and keying by the bare name made every timeframe's menu show the 4h layer's counts.

## The veto: measured, and it is nothing

See `docs/WHAT_WORKS.md`. The part worth repeating here is the methodology trap it walked into
first. On notebook 19's nine-day window the sim takes 8 trades, 4 winners and 4 losers; a 96-cell
gate sweep showed `barrier` blocking losers and `retrace` blocking winners with striking
consistency. That consistency is an artifact: **all 96 cells score the same eight trades**, so they
are one observation, not 96 — and a three-trade block that is all losers arises by chance 7.1% of
the time, meaning ~7 such cells come free. Run across six windows and scored by windows-won against
the chance baseline, 0 of 24 settings clear all six and 4 clear >=4 where chance predicts 8.2.

R per trade, not total R, is the score for anything that only REMOVES trades: random removal from a
positive-expectancy strategy lowers total R while leaving the per-trade number alone.

---

## Notebook 19, second pass: settings → one call → a chart

The complaint was "functions are out there in the open... it only makes me confused about what to
do and where". The Run cell defined `_filter` and `_run` and assembled `PARAMS` inline — about
forty lines of machinery in front of the person using it. All of it moved to
**`quant/research/fvg_lab.py`**, a new package for notebook-facing orchestration. The notebook is
now **7 code cells, 36 lines**, and two tests keep it there: no cell may define a function or
class, and the notebook is capped at 9 cells / 60 lines.

`quant/research/` is the only package boundary here that is easy to get wrong, so its `__init__`
docstring states it: it CALLS indicators, scenarios and viz; it holds no strategy logic, no
indicator maths, no drawing, no sweeps — and **no default for any setting a config owns**, because
a value stated in two places is precisely the 8a8b8de bug. Settings arrive as the config MODULE and
are read through `_get(cfg, name, default)`, so a config predating a setting degrades rather than
raising. `docs/NOTEBOOK_GUIDE.md` is the full version, for whoever touches a notebook next.

One naming wrinkle worth knowing: `quant/research/__init__.py` re-exports the function `fvg_lab`,
which **shadows the submodule of the same name** — `import quant.research.fvg_lab as fl` gives you
the function, not the module. Import the names, not the path.

### Re-arranging a config without changing it

Both config files were reorganised into numbered sections. The check that makes that safe, and
which now lives in the notebook guide:

```python
exec(compile(old_text, "old", "exec"), a); exec(compile(new_text, "new", "exec"), b)
assert not [k for k in a if k.isupper() and k in b and repr(a[k]) != repr(b[k])]
```

nb18: 42 settings before and after, zero values changed. nb19: 59 → 60, zero changed, the one
addition being `START_VISIBLE` which had been hardcoded in a cell (so rule 1 now covers it).

## Isolating one timeframe's gaps

The legend already toggled each layer, but as `4h · FVG bullish` / `4h · FVG bearish` — two clicks
per timeframe and six rows to hunt through. A button row (`FVGs: all | 4h | 1h | 15m | 5m | 1m |
none`) does it in one press. The mechanism worth remembering: **`restyle` accepts a LIST of
visibilities alongside a list of trace indices**, so a single button can show one layer and hide
the rest — which a single style value cannot express.

Gap layers may be FINER than the chart (1m gaps on a 5m chart draw fine, 2,930 of them). No
resampling is involved: each layer is detected on its own real frame, so the trap that made a "1m
EMA" come back as the 5m EMA cannot recur here.

---

## S/R zones, one layer per timeframe

Same treatment the gaps got, because the underlying fact is the same: **a level clustered from 1h
pivots is not the level clustered from 5m pivots**. Each frame is clustered in its own bar space and
only becomes a chart coordinate when a bar index turns into a timestamp — no index remapping, the
same reasoning the pattern and trendline layers already used. `zone_lookback_bars` is therefore in
each layer's OWN bars, so 2000 is about a week on 5m and about eleven months on 4h.

Zones are named (`4h·Z8`) like everything else here, and hover lives on a **midpoint marker** as
well as the line, because a horizontal line is a one-pixel target for a cursor.

**Two bugs this turned up, both worth remembering:**

**A bare `except Exception` returning a plausible number.** I added a small helper to recompute the
chart's bar length in minutes, wrapped in `try/except: return 1.0`. `bar_interval` is imported
*inside* `pattern_figure`, so the module-level helper hit a `NameError` every time — and the fallback
turned that into a silent 1.0, which made every layer's weight saturate at the cap and look merely
"a bit wrong" rather than broken. The block now runs after `base_min` is computed, so the helper is
gone entirely. A fallback that produces a believable value is worse than a crash.

**Shade indexed by position, not by rank.** `_SR_SHADES[position_in_the_ordered_list]` meant the
chart's own layer was the brightest grey when drawn alone and a different grey the moment a 4h frame
was added — the same levels changing appearance because of an unrelated setting. Anchored on the
chart's own layer instead (`3 - (anchor - n)`), which also restores its original `#8e9aa6` exactly.
`test_adding_sr_layers_leaves_the_chart_own_layer_alone` locks it.

**The isolate button, again.** `restyle` takes a LIST of visibilities beside a list of trace
indices, so one button shows one layer and hides the rest. Three rows now sit on the chart —
`FVGs:`, `S/R:`, `EMAs on/off` — stacked at y=1.105/1.165/1.045.

---

## A setting that only one place knew how to satisfy

`MTF_EMAS = [("1m", ...)]` in nb19 raised *"1m is FINER than the chart and no 1m frame was given"*.
The refusal was right — resampling a 5m frame to 1m returns the 5m EMA under another name — but the
only way out looked like adding `"1m"` to `FVG_TFS` or `LINE_TFS`, settings that have nothing to do
with EMAs and which would have started drawing 1m gaps or **trading 1m trendlines**.

The lab now loads frames for EMA-only timeframes itself, which is what notebook 18 already did. Four
frame sets, one per job: LINE (fitted and traded) · GAP (drawn, may veto) · S/R (clustered, drawn) ·
EMA (feeds `mtf_emas`). The general shape worth remembering: **when a setting can only be satisfied
by editing an unrelated setting, the wiring is wrong, not the user.**

---

## Five chart complaints, and the two bigger things underneath

**`patterns=[]` meant "draw all 22 chart patterns".** `nm = list(want) if want else list(_PD)` —
an empty list is falsy, so every notebook asking for trendlines only (`patterns=[]`, which is all of
16, 18, 19 and the lab) was detecting and drawing the entire pattern library. That is most of the
clutter, most of the dropdown, and the slowest call in the figure. `want=None` is now "everything"
and `want=[]` is "none", with detection skipped rather than filtered. Notebook 19: **446 → 292
traces, 5.0 s → 2.3 s.**

**The Jupyter kernel is not the repo interpreter.** The registered `python3` kernel is **Python 3.9
with plotly 5.14.1**; the repo runs 3.13 / plotly 6.3.1. Every notebook execution verified in this
project has been on 5.14. `xaxis.autorangeoptions` (plotly 5.19+) imported fine in the repo and
raised in the kernel, and because nbconvert then never writes the file, an "are there errors?" check
read the *previous* successful outputs and said no. Two lessons: wrap version-dependent plotly
properties, and check that nbconvert actually **wrote** before trusting a clean notebook.

### The four smaller ones

- **Autoscale escaped the window.** `range=` only sets the opening view; the button rescales to the
  DATA, and a 4h gap born 131 days earlier or a trendline fitted through pivots ten days back
  dragged the chart to all of it. Three layers of fix: gap boxes clamp their left edge to the
  window; trendline segments clip with **y interpolated along the fitted line**, so the slope stays
  honest where truncating x alone would have tilted it; and `autorangeoptions.minallowed` bounds the
  axis where plotly supports it. Worst reach: 131 days → **1 hour**.
- **Picking a trade re-showed layers you had switched off.** Every picker button carried a
  visibility vector as long as the whole figure. They now pass an explicit trace-index list as the
  third `update` argument, so the bulk EMA / gap / level layers are never addressed — a button that
  does not address a trace leaves it exactly as you set it. The isolatable `i#` instances stay
  picker-owned, or selecting one gap from the dropdown would stop working.
- **Two dropdowns overlapped.** A plotly dropdown is as wide as its widest row and the menus sit at
  fixed x. `_short()` caps every menu label at 34 characters, applied where the rows are built so it
  holds for patterns, trendlines, gaps and levels alike.
- **Two trades on the same trendline are not a duplicate.** A steeply falling 15m resistance dropped
  ~940 points between the two: a 5m close cleared it at 61,960 and again nine hours later at 61,019,
  while the 15m line was still live because its own timeframe never closed through it. Both are
  labelled `15m·R8`, which is what reads as a duplicate. `tl_once_per_line` (default **off**) spends
  a line on its first entry.

---

## Kernel upgraded to the interpreter the tests actually use

`~/AppData/Roaming/jupyter/kernels/python3` now points at `C:\Python313\python.exe`, overriding the
system kernelspec that came with Python 3.9. Before: notebooks ran **3.9 / pandas 1.5.3 / numpy
1.23.5 / numba 0.53.1 / plotly 5.14.1**; tests ran **3.13 / 2.3.3 / 2.3.3 / 0.62.1 / 6.3.1**.

Checked before touching anything, by running the same computation under both and comparing:

```
                  kernel 3.9                repo 3.13
close_sum         864754757.890000    ==    864754757.890000
gaps 1128         top/bot/dead sums   ==    identical
zones 133         8344662.529556      ==    8344662.529556
trades 8          R_sum 6.994632      ==    6.994632, same causes in order
```

**No number moved.** What changed is speed — figure build **5,220 ms → 2,093 ms** (~2.5×), the full
lab 1,275 → 998 ms; data load and detection were already noise. The old kernelspec is preserved in
the backup under `_jupyter_kernels_backup/python3_py39_ORIGINAL`.

The real argument was never speed: 514 tests all ran on 3.13, so **nothing tested the configuration
the notebooks were actually using**. That blind spot is what produced the `autorangeoptions` break.

### A failed nbconvert leaves the old outputs in place

When that plotly call raised, nbconvert never wrote the file — so a script that opened the notebook
and looked for error outputs read the *previous, successful* run and reported clean. Check that
nbconvert **wrote** before trusting a green notebook.

## Rewind: one clip does every shape

`quant/viz/asof.py`. Arm the button, click a candle, and everything derived snaps back to what was
knowable there. It works because the data was already causal — every artefact carries its own birth,
so rewinding is a **filter over what is already true**, not a reconstruction.

The whole tool is one operation: *drop segments starting after k, clip the rest at k with y
interpolated*. Every trace here is `None`-separated segments, so that single rule handles all of it
with nothing special-cased — a gap box ends at k (so one **mitigated later still reads open**, which
is the entire point), a trendline keeps its slope, a later trade disappears. Candles and EMAs are
left running deliberately: you study a bar to see what happened next.

Three things worth keeping:

- **A deny-list, not an allow-list.** Naming the rewindable kinds missed the per-trade
  entry/stop/target traces, which carry neither a name nor a legendgroup — and the next layer added
  would have been missed too. "Everything except the candles and the EMAs" cannot rot.
- **plotly.py 6 base64-encodes numeric columns**, so `gd.data[i].y` is not always an array —
  three trade-marker traces arrived as `{dtype, bdata}` and `.slice()` threw. `gd._fullData[i]` is
  plotly's decoded copy, where the same column is a `Float64Array`.
- **The S/R lookahead this exposed.** `sr_zone_table` emits a row per price revision and the normal
  chart draws only the LAST — today's level painted across the whole window. Harmless live, a quiet
  lookahead when studying history, so `rewind=True` keeps every revision and the clip picks the one
  that was live.

Verified by driving a real chart in a browser: 345 rewindable traces, **0 left past the cut**,
candles and EMAs still running, the marker drawn, and `live` restoring exact point counts.

`AS_OF` is the other half and cost almost nothing: since every detector is causal, **an earlier END
is a point-in-time run**. Slower and not clickable, but nothing is filtered and it works inline.

### The rewind bug only a real click could find

Driving the finished chart in a browser turned up something no Python test would have: the cut was
landing **five hours early**.

`plotly_click` hands back `ev.points[0].x` as a *formatted* string — `"2026-06-06 06:10"`, with no
offset — and `new Date()` reads an offset-less string as **local time**. On a UTC+5 machine that is
01:10Z, so everything between 01:10 and the clicked candle was dropped instead of clipped. The
symptom was subtle in the right way: the chart still looked plausibly rewound, just to the wrong bar.

Measured:

```
clicked candle        2026-06-06 06:10 (chart/UTC)
cut actually applied  2026-06-06 01:10Z
local offset          -300 minutes
```

The fix is to take the point's **stored** value — `gd._fullData[p.curveNumber].x[p.pointIndex]` —
which still carries `+00:00` and parses correctly. The as-of marker had the mirror of the same
problem: `x0: new Date(cut)` renders in the browser's zone, so it is drawn at the raw x instead. The
readout now says `UTC` explicitly.

After the fix, verified on the same click: as-of line at exactly `06:10:00Z`, **0 of 345 rewindable
traces past the cut**, the 15m bullish gap layer *clipped* to end at 06:10 rather than dropped, and
the only thing still drawn to the right is the candlestick trace — which is the intended behaviour.

**The general lesson:** a browser feature is not verified until it has been clicked in a browser.
Both bugs in this tool (the base64 columns and this one) were invisible to the Python side.

---

## Autoscale jumped to the year 2000: `minallowed` must be epoch milliseconds

`xaxis.autorangeoptions.minallowed` was being written as a date STRING
(`"2026-06-01T00:00:00+00:00"`, straight from the pandas Timestamp). plotly.js cannot compare a
string there — autorange produced **no valid range at all** and the axis fell back to its empty-date
default, which is `2000-01-01 .. 2001-01-01`. Pressing Autoscale threw the chart 26 years into the
past and the only way back was a page reload.

Every string form fails; only the number works. Measured in the browser:

```
"2026-06-01T00:00:00+00:00"  ->  ["2000-01-01", "2001-01-01"]
"2026-06-01T00:00:00"        ->  ["2000-01-01", "2001-01-01"]
"2026-06-01"                 ->  ["2000-01-01", "2001-01-01"]
Date.UTC(2026, 5, 1)         ->  ["2026-06-01", "2026-06-10 05:27"]   correct
```

Now `int(pd.Timestamp(_vs).value // 1_000_000)`, with a test asserting the type.

## A rewound gap box collapsed into a line: a polygon is not a polyline

The generic clip walked each `None`-separated segment to the cut and stopped. That is right for a
line, and wrong for a **closed polygon**: a gap is `[x0,x1,x1,x0,x0] x [bot,bot,top,top,bot]`, so
truncating at the cut kept only the first edge — the bottom — and every rewound gap drew as a flat
line at its lower boundary.

Two clips now, dispatched on `fill === 'toself'`:

- **line — TRUNCATE**, interpolating y onto the cut. Moving x without recomputing y would tilt a
  trendline, which is worse than not clipping.
- **box — CLAMP** each vertex back to the cut. The polygon stays closed, so the box keeps the gap's
  real height and only gets shorter. Verified on a box spanning the cut: 5 vertices, 2 distinct y
  levels, height 356.03 preserved, drawn 05:45 → 06:10.

## Button rows that lied about the chart

Both timeframe rows opened highlighting **all** while `START_VISIBLE = ("trades",)` meant every gap
and level layer started `legendonly`. `_row_active` now reads the traces' initial visibility: all
visible → `all`, none visible → `none`, mixed → `-1` (highlight nothing) rather than name a button
that is not true.

## "That big candle should have left an FVG" — why it did not

Trade #04's entry candle (2026-06-05 18:50, a 578-point red bar) looks like it must have gapped.
It did not, and the reason is worth keeping: **a fair value gap is the space between the candle's
NEIGHBOURS, not the size of the candle itself.**

```
18:45   low  59,669.60      <- the first candle's low
18:50   the displacement candle, close 59,207.42
18:55   high 59,649.99      <- the third candle wicked almost all the way back up
```

The gap is `high[18:55] < low[18:45]` → a band of **19.61 points, 0.059 ATR** — the 18:55 candle
rallied back and ate all but a sliver of the space the drop had opened. At `FVG_MIN_SIZE_ATR = 0.50`
it is filtered at detection, correctly; it was also mitigated six bars later. Set the gate to 0 to
see gaps like it, at the cost of ~2,000 gaps on 5m.

---

## Two rewind bugs, both in the same blind spot

**Isolatable gaps rewound into a sliver.** The box/line dispatch read `gd._fullData[i].fill`, and
plotly only computes full defaults for traces it has actually drawn — every isolatable gap is built
`visible:false`, so `fill` was **undefined for all 41 of them** and they were clipped as polylines.
Pick one gap out of the dropdown, rewind, and it collapsed. `gd.data[i]` (the spec) always has it.
After: 26 polygons across the cut, **0 collapsed, 0 heights changed**.

**The trade picker looked broken while rewound.** It was not — the trades had not happened yet, so
every row drew nothing and nothing said so. Stress-tested across three rounds of rewind/pick/live:
zero JS errors, picker recovering every time. The fix is legibility. The readout now reads
`as of 2026-06-02 00:00 UTC · 2 of 8 trades so far (the rest are hidden — press live)`, counting
ENTRY markers only — counting everything in the `trades` legendgroup counts exits too and reported
"16 trades" for eight.

No FVG box ever extended, on any layer or any cut: 0 of 153 grew, 73 shrank, 80 unchanged. An
earlier scan that appeared to show 36 extended S/R segments was my own matching bug — several
revisions share a start bar, so keying by start alone matched the wrong one.

## How long after a break can a trade still fire?

`hold = max(tl_close_n, tl_full_m, tl_dir_n, tl_dir_full_n)` bars **of the line's own timeframe**.
At the current config (all 1) that is one bar: 5 minutes for a 5m line, 15 for a 15m, 60 for a 1h.
Measured on the eight live trades, **every one fired while its line was still live** — the hold is
never reached, because `TL_CLOSE_N = 1` puts the entry on the break bar itself.

The entry rule (`EMA_15m_100`) does not widen this: it is ANDed on the same bar, so if it fails
there the trade is simply not taken — it cannot wait. Raising `TL_CLOSE_N` does not give a waiting
window either; it asks for **n consecutive closes** beyond the line, a different condition.
`TL_CONFIRM_N` is the setting that actually means "within n bars".

## TL_CONFIRM_N — wait for the move to repeat itself

A break candle that is huge next to its neighbours and closes a whole new body beyond the line looks
decisive, and is exactly the shape that gets swept back, so its close is the worst price of the
move. `TL_CONFIRM_N = n` asks the market to do it again: within n chart candles one candle must be
the **trade's own colour** AND close beyond the **whole** trigger candle (below its low for a
short). The entry is on that close; no confirmation in the window means no trade.

Beyond the trigger's low rather than its close, deliberately — the point is a fresh body clear of
the entire bar, and closing back inside its range IS the sweep.

One thing it breaks if you are not careful: the line is retired at the break, so a delayed entry
lands on a bar where the line's id is already gone and every confirmed trade is labelled `15m·?`.
The id is now held for the trigger's own window **plus** the wait. The LEVEL hold is deliberately
not extended — that would let the break itself fire later, which is a different change.

On this window (8 trades, so read nothing into it): `n=2` → 6 trades, `n=3` and `n=5` → 7.

## An unfilled FVG expired in the counts but ran forever on the chart

A thick bearish box ran the full width of the chart through price that had plainly traded
through it. It had not been filled — it had EXPIRED, and nothing recorded that. `_live_state`
retires a gap once it outlives `max_age_bars`, but `dead_i` records only mitigation, so an aged-out
gap kept `dead_i = -1` and every consumer read "still open": the chart drew it to the right edge,
the census counted it open, while the counts the veto reads had dropped it days earlier. On a
20-day window 75 of the 80 "open" 5m gaps had expired. `end_i` / `end_reason`
(filled | expired | open) are now the one source of truth; the 12 per-bar columns the veto reads
were proved byte-identical against the old code. The empty-table branch also lacked every derived
column, so a quiet frame would have raised on `end_reason`.

The test is an INVARIANT, not a case: for every gap on every bar, the table's [born, end) must
equal the live count. Its first fixture put the gap at bar 2, where ATR is still NaN, so the size
gate dropped it and the test passed on an empty table — the recurring trap in this project.

## S/R zones: three corrections, a study, and a clarity knob

**The corrections** (one engine, `_sr_core`, behind both `add_sr_zones` and `sr_zone_table`):
1. **Anchored clustering.** Pivots joined a zone if within `cluster_pct` of its RUNNING MEAN,
   which moves as pivots join — zones crept to 0.257% against a 0.15% setting. Now the whole band
   (min-max of the pivots) must fit.
2. **Break retirement.** Crossing a zone did nothing; 48 of 54 zones on one chart had been broken
   and held beyond. `break_bars` consecutive closes on the far side retire it; a close back RESETS
   the count, so a wick-and-reclaim does not kill a level.
3. **Farthest-from-price eviction.** Found by the study, not the chart: the legacy rule dropped the
   least-touched zone, so old strong levels held every slot and each new zone near price was
   evicted as it formed. BTC 4h ended 2022-26 with 40 zones a median 67% from price, 1 within 5%.

Also found while reading the legacy code, and fixed on the new path: the two "mirror" functions
evicted with DIFFERENT tie-breaks, and `sr_zone_table` used the SLOT index as the zone id, so an
evicted zone and its replacement shared an id and the chart's one-line-per-zone step dropped one.

**Legacy is preserved on its original code path** and reached by `SR_SENSITIVITY = None`. That was
proved byte-identical against the actual pre-change code, run in a temporary git worktree —
asserting it would not have been enough. `add_sr_zones` defaults to legacy too, because the
scenarios that read it (`level_event`, `zone_top`) have results recorded in docs/.

**The study** (`experiments/sr_zone_search.py`) and its verdict are in docs/WHAT_WORKS.md. The short
version: zones beat shuffled bars but not the same zones moved 1-3 ATR, so they mark AREAS. That
turned the knob from a quality dial into a CLARITY dial, and the first selection rule — argmax of
train edge — had to go: it chose a 15m setting that drew 0.5 lines and lost forward.

Two measurement lessons worth keeping:
- **Clutter must be measured as what is DRAWN.** The first metric counted every live zone,
  including ones 25% from price that no chart shows; it now counts zones within 20 ATR.
- **Distinct zones are not lines on screen.** On notebook 21 the 4h layer went from 9 zones to 18
  and looked like a regression — but with retirement each is short-lived, and lines visible at any
  moment went 5.1 -> 1.2. Across all layers: ~52 -> ~7.

**Precedence, and what that means per notebook.** An explicit setting always wins over the table
(`sr_params(tf, level, **overrides)`, None = not stated). Notebook 21 states `SR_SENSITIVITY` in
place of its four S/R values and gets the full preset. Notebooks 18-20 still state
`ZONES_LEFT` etc., so they keep those and take only `break_bars` and anchoring from the table —
their files were not edited. Backups: `notebooks/backups/2026-09-27_pre_sr_tuning/`.

### Level age (2026-09-27) — the two age limits had never been measured

Asked whether `FVG_MAX_AGE_BARS` and `ZONE_LOOKBACK` were tested: they were not.
`max_age_bars=500` was chosen when the FVG detector was built, to bound the per-gap mitigation scan
(0 = no limit is exact but O(bars) per unfilled gap); `fvg_veto_search` varied only the VETO's age
(20/60), never the detector's. `zone_lookback_bars=2000` (7543d20) fixed "1 of 40 zones near price" —
a symptom of the legacy least-touched eviction, which the S/R study later fixed at the root. Its
reason is gone; the setting stayed because it is the user's.

`experiments/level_age_study.py` measured age directly (verdict in docs/WHAT_WORKS.md): no age
effect either way. The user's example — BTC back at 86k months later — is real price behaviour, but
the evidence says price reacts to the AREA, and a line placed $2-5k away does as well. Keeping old
levels is harmless; it just adds lines.

Two traps in the measurement worth keeping:
- **Age counts from the zone's FIRST pivot**, not from when it earned enough touches to be drawn.
  The segment table's first KEPT row is the latter, so birth is taken from the unfiltered segments.
- **The shift null must keep the level's birth and size.** A shifted FVG band that already surrounds
  price at birth has no approach side and is not a test; it is skipped, as a real one would be.

### Exit rules: moved target and FVG exit (2026-09-27)

Asked for two rules from one screenshot (nb21 #04, 0.1R short of 3R, then stopped). Built into the
ENGINE, because both are per-trade: a target that moves is that trade's own state, and "a gap near
THIS trade's target" depends on its own stop. Neither can be a global signal array.

- `tp_move_*` (BacktestConfig): once per trade, on the bar's CLOSE, after every exit, applied from
  the NEXT bar. A bar that reached the target zone and gave it back has unknown intrabar order, so
  it may not also fill the new target -- the test's bar 2 would have, at 97.4 against 98.2.
  Rungs that close nothing (breakeven moves) are not moved: they are stop rules. Already past the
  new target -> closed at that close.
- `tp_ref_*_col` (BacktestConfig): a second target per trade, read ONCE at the entry bar. When one
  bar reaches it and a ladder level, the NEARER one fills first -- price cannot reach the far one
  without passing the near one. The stop still wins a bar that hits both under stop_first.
- The FVG level: `fvg_on_grid` places every timeframe's gaps on the base grid (live from the third
  candle's CLOSE, mitigation re-checked on the base bars, expiry in the gap's own bars), and
  `strategies.fvg_target` picks, per bar, the gap a trade opened there would aim at. It uses the
  lab's own `fvg_kwargs`, so the gap a trade exits on is one the chart draws.
- Off is byte-identical (golden tests) and adds no trade columns. With either on, trades gain
  `target_moved_*` / `ref_target`, and the chart draws a moved target as a STEP and the FVG level as
  a second segment -- inside the existing target trace, because every picker addresses a trade's
  traces by the fixed TRADE_TRACES count.

`FvgLab.retrade(**exit_settings)` re-runs only the sim (EXIT_KEYS), proved equal to a full rebuild;
`run_arms(CFG)` uses it when every COMPARE arm touches exit settings only -- 4 arms in 6 s, not 24.

**Found on the way: `run_grid` had been broken since e61d7ab.** It built the kernel's arguments by
hand and fell behind when the secondary-exit arguments were added: every sweep died on "expected 70
arguments, got 59", and no test ran it. It now goes through `invoke_kernel` like `run_backtest`,
and `test_the_sweep_runs_again` runs it.

**The test harness** (`experiments/exit_rules_search.py`) builds 4h from 1h (the 4h cache starts
2025-08) and switches drawing-only settings off; both were proved to change no trade on notebook
21's window. The 2022Q1 warm-up needed 1h bars from 2021-08, which the loader FETCHED from Binance's
public API into the cache (~2,400 bars) -- normal incremental behaviour, noted because it happened.

Verdict in docs/WHAT_WORKS.md: no arm beats noise, and the strategy loses 0.157R per trade after
costs over 1,194 trades. The lesson worth keeping: **a rule drawn around one trade always fixes that
trade.** Notebook 21's window made both look like +3R to +5R improvements; four and a half years
said the aggressive versions cost 30R forward.

### Strategies as files, notebooks as views (2026-09-27)

Asked for a structure that scales: every idea had become a new numbered notebook with a copied
config, and the copy tests kept breaking on the owner's own edits (FVG_MAX_AGE_BARS in 21 was the
latest). Now: `notebooks/strategies/<name>.py` (what is traded), `lab_config.py` + `lab.ipynb` (one
strategy on one window) and `evaluate_config.py` + `evaluate.ipynb` (the verdict over 19 quarters).
Details in docs/NOTEBOOK_GUIDE.md "Start here".

Decisions worth knowing:
- The split is by WHO READS A SETTING: the strategy file holds everything the sim or the loader
  needs to produce trades (including gap detection, since the veto and FVG exit read gaps); the
  view holds the window and everything only the chart reads. Proved exact on the first file:
  `tl_break_confirm.py` + `lab_config.py` reproduce `nb21_config.py` value for value.
- `COMPARE` lives in the strategy, not the view: the same arms are the lab's table on a window and
  the evaluation's table over the quarters. The first file adds `no confirmation`, the question
  notebook 21 was built around -- over 19 quarters it is noise too (t 1.12; +45R train, -10R
  forward).
- `RETRADE_KEYS` widened from the four exit toggles to every setting the loader does not read
  (confirmation, stops, targets, costs, veto). The guard is a test that PARSES the loader's source
  for `c.X` / `_get(c, "X")` and fails on any overlap -- a list kept correct by memory would not
  stay correct.
- `experiments/evaluate.py` owns the quarter/window/4h helpers and `exit_rules_search` imports them;
  after the refactor it reproduced its recorded CSV exactly.
- Drawing-only settings default OFF in the evaluation window (`_WINDOW_ONLY`), and a test checks the
  evaluation's window reproduces the lab's trades on the lab's window.

