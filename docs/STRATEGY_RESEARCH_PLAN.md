# Gold (Dukascopy XAUUSD) — Strategy Research Plan

Goal: find a **robustly profitable, out-of-sample-verified** strategy on true-spot XAU/USD, using
the platform's full toolkit (MA/EMA + smoothing, MTF, RSI, S/R zones, Heikin-Ashi, big-move,
swings/structure stops, per-rule exits/stops, rule sequencing, session/time filters). Dukascopy
gold only. **Rigor over raw return** — a strategy that only works in-sample is worth nothing.

**Locked decisions (from planning):** cost = **spread 0.25 $/oz, no commission**; entry timeframes
**1m / 5m / 15m** (swept), higher TFs (**1h / 4h**) used only as a **bias filter**; **full**
anti-overfit protocol; and **heavy emphasis on exit/stop variety** — fixed % target, fixed cash
target, trailing stop, break-even, RR ladders/partials, time-stop — treated as a first-class search
axis, not an afterthought.

---

## 0. Guardrails (read first)

The #1 failure mode of "try many combinations until something is profitable" is **overfitting** —
with enough parameter combos, some will look great by luck. Every phase below is designed against
that:

- **Lockbox out-of-sample (OOS):** a final data slice touched exactly **once**, at the very end.
  Nothing is tuned on it. If the strategy dies there, it dies.
- **Prefer plateaus, not spikes:** a good parameter set has *profitable neighbours*. A lone spike
  surrounded by losers is curve-fit noise. Sweep heatmaps must show a region, not a pixel.
- **Minimum sample size:** ≥100 trades (ideally ≥300) per result, or it's not significant.
- **Net of realistic costs, always.** Gross-profitable / net-losing is the norm on 1m gold.
- **Consistency across sub-periods and walk-forward**, not one lucky quarter.
- **Kill criteria stated up front** (see §9) so we don't rationalise a bad result.

---

## 1. Data & cost model

**Data (cached):** XAUUSD 1m, `source='dukascopy', market='cfd'`, 2024-06-03 → 2026-07-24 (~25
months). Higher timeframes derived by resampling 1m (`build_mtf`), so MTF features span the full 2
years. Weekend gaps are normal (metals/FX week: Sun 22:00 → Fri 22:00 UTC-ish).

**Costs (LOCKED):** `spread=0.25` $/oz, **no commission**, no separate fee. Slippage kept at
`slippage_bps=1.0`. On 1m the median bar range is ~0.6, so cost ≈ 40% of a typical bar — **every
strategy is stress-tested at spread 0.30** to confirm it survives cost drift.

**Contract:** gold = 100 oz/lot (`contract_size=100`) if we test leverage/lots; otherwise cash
sizing with `risk_pct_equity`.

---

## 2. Validation protocol (the backbone)

Three fixed, non-overlapping splits (approx; finalise on go):

| Split | Range | Months | Use |
|---|---|---|---|
| **Discovery** (in-sample) | 2024-06 → 2025-08 | ~14 | explore, sweep, form hypotheses |
| **Selection** (validation) | 2025-09 → 2026-01 | ~5 | pick final params/strategies; anything that fails here is dropped |
| **Lockbox** (OOS) | 2026-02 → 2026-07 | ~5.5 | touched **once** at the end; the verdict |

Plus an **independent walk-forward**: rolling train 6 mo → test 2 mo, stepped monthly across the
whole span. A strategy must be profitable in the **majority of walk-forward test windows**, not just
on average. This catches regime-dependence that a single split hides.

Robustness gates a candidate must pass before touching the lockbox:
1. Positive net expectancy in Discovery **and** Selection.
2. Parameter plateau (neighbours within ±1 step also profitable).
3. ≥100 trades per split; profit factor ≥ 1.15 net; max DD tolerable.
4. Survives spread +0.05 and slippage ×1.5.
5. Walk-forward: profitable in ≥60% of test windows.

---

## 3. Phase A — Market character (before any strategy)

Understand gold's behaviour so strategy choice is informed, not random. Uses `analytics.attribution`
+ ad-hoc notebooks.

- **Session/time-of-day:** confirm the 13–16 UTC overlap edge; map range & directional bias by hour,
  session (`london/newyork/tokyo/sydney`), weekday. → drives universal time filters.
- **Volatility regime:** ATR percentile over rolling window; tag bars high/low-vol. Gold trends in
  high-vol, chops in low-vol — different families suit each.
- **Trend vs range regime:** EMA(50) slope sign + separation of fast/slow EMAs (or ADX if we add it).
  → decides where trend-following vs mean-reversion is even viable.
- **Auto-correlation / follow-through:** do big moves continue (momentum) or revert (fade)? Measures
  the base edge each family is betting on. Also by session.
- **S/R behaviour:** at swing-cluster zones, do prices reverse or break? Split by touch count **and**
  reaction strength (the metric we flagged) — this decides how we score zones later.

**Deliverable:** a "gold character" notebook (11) + a one-page findings note in SESSION_NOTES.

---

## 4. Phase B — Strategy families (each is a hypothesis to test)

Each family gets: a clear hypothesis, a parameter grid, a Discovery sweep (`run_grid`), robustness
screening, then Selection confirmation. Map of tool → role: MA/EMA=trend/bias, RSI=momentum/
reversion, S/R=entry/target/stop, HA=smoothed trend & exhaustion exits, big_candle=breakout/
exhaustion, swings=structure stops, sequencing=pullback/continuation, time filters=conditioning.

**B1. Trend / MA breakout** (our built strategy). *Hypothesis: gold trends intraday during the
active session.*
Grid: MA `type∈{sma,ema}`, trend length `{50,100,200}`, `smooth∈{0,9}`, entry confirm `N∈{1,2,3}`,
candle `{real,ha}`, exit `{opposite, full_below fast MA, ha_red N}`, `one_entry_per_cross∈{T,F}`,
stop `{swing_low, atr×{1.5,2,3}, pct×{0.4,0.6}}`, session `{all, 12–17 UTC}`.

Global axis: **entry timeframe `{1m,5m,15m}`** (resampled from 1m) is swept across every family; the
hypothesis is that the edge strengthens on 5m/15m as cost drag falls. Higher TFs `{1h,4h}` appear
only as bias filters (B2), never as the entry grain.

**B2. MTF-biased trend.** *Hypothesis: aligning entries with a higher-TF trend filters chop.*
Grid: bias TF `{1h,4h}` × EMA `{20,50,200}`; entry TF `{1m,5m,15m}`; only long above / short below
the HTF EMA. Cross with B1 entries.

**B3. RSI mean-reversion (range regime only).** *Hypothesis: in low-vol/range, gold fades extremes.*
Grid: RSI period `{7,14,21}`, OS/OB `{20/80,25/75,30/70}`, exit `{RSI midline, RR{1,1.5}, opposite
band}`, stop `{atr×2, pct}`, regime gate `{range-only via EMA-slope/ATR}`, session filter.

**B4. S/R zone bounce / break.** *Hypothesis: strong zones produce tradeable rejections.*
Grid: swing `left/right∈{5,8,12}`, `cluster_pct∈{0.10,0.15,0.25}`, `min_touches∈{1,2,3}`,
`within_pct∈{0.05,0.1}`, entry `{bounce_off_support, break+retest}`, target `{opposite zone, RR}`,
stop `{beyond zone, atr}`. Later: swap touch-count for **reaction-strength** scoring and re-test.

**B5. Volatility / range breakout.** *Hypothesis: session-open range breaks continue.*
Grid: `big_candle mult∈{1.5,2,3}` continuation, or breakout of prior N-bar high/low
`N∈{15,30,60}`, session-open range (e.g. first 30–60m of NY), stop = opposite side of range, target
= RR or measured move.

**B6. Heikin-Ashi trend.** *Hypothesis: HA smooths noise for cleaner trend holds.*
Grid: HA candle conditions on B1/B2, `n_consec∈{2,3,4}`, HA-flip exits, HA-based EMA source.

Each family is run first on a **3-month Discovery subset** (fast, ~30 ms/combo) to prune, then the
survivors on the full Discovery span.

---

## 5. Phase C — Exit / stop / sizing design (orthogonal axis)

The entry edge and the exit design are separate problems; a mediocre entry with a great exit can
win. **You explicitly want this axis explored hard** — so for the top 2–3 entries from Phase B we
sweep the FULL exit/stop matrix (reuse `experiments/exit_design`). The engine supports all of these
natively (`tp_mode`, laddered `TakeProfit`, `trail_mode`, `move_stop_mode`):

- **Profit-taking / closes:**
  - **Fixed % target** (`tp_mode='entry_pct'`, e.g. 0.2 / 0.3 / 0.5 %) → close at target.
  - **Fixed cash/price target** (`tp_mode='price_abs'`, e.g. +$2 / +$4 per oz) → close.
  - **RR targets** (`tp_mode='rr'`, R multiples 1 / 1.5 / 2) off the stop distance.
  - **RR ladders / partials** (`TakeProfit` list: 50% at 1R then runner to 2R).
  - **Time-stop:** force-exit after N bars if neither target nor stop hit.
  - **Signal exits:** opposite EMA, S/R target zone, ha_red / opposite-candle exhaustion.
- **Stops:**
  - **Fixed %** (`sl_mode='entry_pct'`), **fixed cash** (`price_abs`), **ATR×k** (k∈{1.5,2,3}).
  - **Structure** (`ref_col` swing-low / S/R zone).
  - **Trailing** (`trail_mode='pct'|'price_abs'`, ratchets on high-water mark) — pct and ATR-based.
  - **Break-even move** after 1R (`move_stop_mode='breakeven'`), then optionally trail.
- **Sizing:** `risk_pct_equity` (compounding) vs fixed; test 0.5–1.5% risk.

The sweep crosses {target type} × {stop type} × {trail on/off} × {break-even on/off} for each top
entry. Objective: maximise **expectancy × √trades** and net profit factor, not win rate.

---

## 6. Phase D — Conditioning (when does it work?)

For each surviving strategy, use `analytics.attribution` to find **where the edge lives**:

- by **hour / session** — restrict to the profitable window (likely the 13–16 UTC overlap).
- by **weekday** — drop dead days.
- by **volatility regime** — many edges only exist in high (or only low) vol.
- by **month/quarter** — check the edge isn't one lucky run.

Conditioning is applied as `session`/`hours`/`weekdays` filters + regime gates — but **only if the
conditioned version also holds in Selection** (conditioning is a prime overfit vector, so it must
survive OOS too).

---

## 7. Phase E — Hybrid / ensemble assembly

Combine the survivors using the rule-builder's real strengths:

- **Regime switch:** trend family in high-vol/trending, mean-reversion (RSI/S/R fade) in
  low-vol/range — selected by a regime gate condition.
- **Filter stack:** HTF-EMA bias (B2) **AND** session filter (D) **AND** entry trigger (B1/B4).
- **Sequenced legs:** primary breakout → secondary S/R-pullback continuation (`after=`), each with
  its own exit/stop (per-rule exits/stops we built).
- **Multi-entry, one book:** several `Rule`s in one `RuleStrategy`, each tagged, sharing a risk book.

Ensemble is only worthwhile if it beats each component on **risk-adjusted** terms in Selection.

---

## 8. Phase F — Final validation (the verdict)

Run the single chosen strategy (params frozen from Selection) on the **lockbox OOS** — once.
Then robustness stress:
- Walk-forward across the full span (profitable in ≥60% of windows).
- Cost sensitivity (spread 0.20 → 0.35).
- Parameter perturbation (±1 step on every knob — equity shouldn't collapse).
- Monte-Carlo trade-order shuffle → distribution of DD/terminal equity (is the result fragile?).
- Start-date sensitivity (does skipping the first month change the verdict?).

A pass here = a strategy worth paper-trading. A fail = we learned what doesn't work (also valuable),
and iterate — but **never re-tune on the lockbox**; if we exhaust it, we need fresh OOS data.

---

## 9. Objective function & selection criteria

Rank candidates by a **composite robust score**, not raw return:
`score = f(net profit factor, Sharpe/Sortino, expectancy) × penalty(max DD) × penalty(low trades)`,
subject to hard gates: **≥100 trades, PF ≥ 1.15 net, positive in every sub-period, plateau present.**

**Kill criteria (stop pursuing a family):** negative net expectancy in Discovery after full sweep;
edge exists only in <20% of walk-forward windows; profitability vanishes at spread +0.05; requires
>5 conditioning filters to be profitable (that's fitting).

---

## 10. Deliverables & tooling map

- **Sweeps:** `quant.optimize.run_grid` (array-native, bounded-parallel) — subset-then-refine.
- **Inference:** `experiments/` layer (strategy_space × cfg_space, objective, ranked results.csv +
  report.md per study). New studies: `experiments/gold_trend.py`, `gold_sr.py`, `gold_regime.py`.
- **Attribution:** `analytics.attribution` for the "when it works" phase.
- **Visualisation:** sweep heatmaps (`viz.heatmaps.sweep_heatmap`) to eyeball plateaus; ResearchChart
  for trade inspection; S/R zone overlay (nb 08).
- **Notebooks:** 11 = gold character (Phase A); 12 = family sweeps (Phase B/C); 13 = conditioning
  + hybrid (D/E); 14 = final OOS + robustness (F). Kept lean via the tag/helper system.
- **Running notes:** findings appended to `docs/SESSION_NOTES.md` each phase.

---

## 11. Honest expectations

On 1m gold with realistic costs, a *durable* edge is genuinely hard — most "profitable" backtests
are overfit or cost-blind. Likely outcomes, in order of probability: (a) a modest edge that only
survives on a **higher timeframe** (5m/15m) and/or **only in the active session** with tight
conditioning; (b) a regime-switching hybrid that's positive but low-Sharpe; (c) nothing that
survives OOS — in which case the deliverable is a rigorous map of what *doesn't* work, which is
itself worth having. The process is the product; the rigor is what makes any "yes" trustworthy.

---

## Execution order (once approved)

1. **Phase 0:** confirm cost model + split boundaries; set up `experiments/` studies scaffolding.
2. **Phase A:** gold-character notebook → findings.
3. **Phase B:** family sweeps on 3-mo subset → prune → full Discovery.
4. **Phase C/D:** exit design + conditioning on survivors.
5. **Phase E:** hybrid assembly; confirm on Selection.
6. **Phase F:** single lockbox OOS run + robustness. Verdict.
