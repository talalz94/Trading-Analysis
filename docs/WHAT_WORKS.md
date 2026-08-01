# What works and what doesn't — the short version

> The full evidence, phase by phase, is in [`STRATEGY_SEARCH_LOG.md`](STRATEGY_SEARCH_LOG.md).
> The plan that was executed is [`SEARCH_PLAN.md`](SEARCH_PLAN.md).
> Everything below is measured on BTCUSDT and XAUUSD, 2025-01 → 2026-07, three regime windows
> (+26%, −24%, −28%), always after costs, always compared against a **null entry** that trades on a
> fixed schedule with no signal at all.

---

## The one-paragraph answer

**No entry rule tested in this project has a validated edge.** Nine entry families, both directions,
eight level types, four interaction types, six timeframes, two instruments, 150 management
combinations and ten regime conditioners — the search kept producing *fewer* survivors than chance
would, and the two configurations that looked genuinely good both failed the gates that discriminate:
one is carried by under 1% of its trades, the other does not exist on a second instrument. The
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
| **Consecutive-loss lockouts / daily loss caps** | ✗ ruled out analytically | Trade outcomes are **independent** (all autocorrelations inside ±0.027, `P(loss\|prev loss) − P(loss)` ≤ 0.011 and sign-flipping). Caps made expectancy *worse*: −0.023 → −0.057 |

## ~ What genuinely helps, but is not an edge

| Idea | What it actually does |
|---|---|
| **Time stop** (new engine feature, `max_bars_in_trade`) | Median expR −0.045 → −0.027 at 15–30 bars, median drawdown 62.9% → 55.8%. It helps the **null entry** by about as much, so it is cutting cost-weighted dead time, not finding better exits. Keep it; don't count it as alpha. |
| **Trailing stop** | Turns almost anything positive in-sample — including a random entry — by harvesting BTC's fat-tailed trend structure. Not an entry edge. |
| **Higher timeframe for drawdown** | Median max drawdown 64% → 32% → 14% from 1m → 5m → 15m. Real, and unrelated to expectancy. |
| **Smaller position size** | The ONLY risk control that works. 1% → 0.25% risk leaves `expR` identical and divides drawdown by the same factor — it buys nothing you couldn't get by trading smaller. |

## ✓ The one positive, actionable rule found

**Do not trade into a volatility shock, and do not chase the hour after it.** Conditioning entries on
the shock bar itself costs **−0.2596R** — the single largest effect measured anywhere in this search —
and the following 60 minutes costs −0.0888R. It holds across every entry family and both directions.
Since `vol_shock` is the announcement proxy (no economic calendar is available to this platform),
this is the empirical form of "stay out of the release." It is a rule about when *not* to trade
rather than an edge, but it is large, consistent, and the only Phase-4 result worth acting on.

## ? The only two things that are not obviously nothing

**1. `pdlow_fade_long` + trailing stop (BTC 1m)** — long on a fade of the previous day's low.
Worst-window +0.136R against the null's +0.040R, survives 2× spread where the null does not, and is
positive across every stop and trail value tested. **But:** the top 1% of trades supply 40–77% of all
profit, deleting the best 10 trades of 2,081 turns it negative, and its bootstrap lower bound is
below zero in two of three windows.

**2. `wt53_short|atr2.5/rr1.5` (BTC 15m)** — the cleanest-shaped result in the search: bounded 1.5R
target, no trailing stop, +0.074 / +0.148 / +0.077 across all three regimes, 509 trades, only 17.3%
drawdown. **But:** the identical config on gold 15m scores −0.129 / −0.167 / +0.140, and with ~170
trades per window the standard error is ≈0.09 — **+0.074 is less than one standard error from zero**.

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
