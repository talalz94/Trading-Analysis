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
