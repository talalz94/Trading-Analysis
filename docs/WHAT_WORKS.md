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
