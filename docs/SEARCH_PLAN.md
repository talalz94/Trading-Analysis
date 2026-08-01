# Search Plan — everything to test, and every variable

> Written before running anything. Scope: **entry logic first** (the axis never properly varied),
> then management, then risk. Results go to [`STRATEGY_SEARCH_LOG.md`](STRATEGY_SEARCH_LOG.md).

## What was actually covered before, and what wasn't

| Tested | Never tested |
|---|---|
| stop methods (7), TP methods (9), secondaries (3) | **prev-day / session / weekly levels** |
| execution TF (1m/5m/15m) | **fade / rejection / failed-break** at a level |
| long vs short mirror | **volume** — in the data, never used once |
| hour-of-day, weekday filters | **VWAP** — not built |
| daily loss cap | **structure (HH/HL) as entry or filter** |
| one entry family: break-of-S/R-below-EMA | **squeeze / wavetrend / RSI-div as primary trigger** |
| BTC only | **a second instrument** (gold XAUUSD + PAXG are cached) |
| | **time stop** (exit after N bars) |
| | **opening-range** breakout/fade |

The single biggest gap: **every test varied the exit, never the entry.** One entry family, ~150
configs of exits around it.

---

## Phase 1 — Do these levels predict anything at all?

Fast screen *before* any backtest: at each level interaction, measure forward return over
{5, 15, 30, 60, 120} bars, **minus the same-window baseline drift**, in ATR units, per regime. If
drift-adjusted edge is ~0 and flips sign across regimes, the level is dead and no exit rule saves it.
This is the method that correctly killed the break entry.

### Levels to build (all causal)
| Level | Definition |
|---|---|
| `pd_high/low/close/mid` | previous **calendar day** completed extremes |
| `sess_prev_high/low` | previous **session** extremes (Asia 0-8, EU 8-16, US 16-24 UTC) |
| `sess_run_high/low` | **running** extreme of the current session (expanding, no lookahead) |
| `wk_high/low` | previous week |
| `or_high/low` | **opening range** — first N minutes of a session (N ∈ {15, 30, 60}) |
| `rnd_above/below` | round numbers (BTC step ∈ {500, 1000}) |
| `sr_support/resistance` | existing S/R zones (already built) |
| `vwap`, `vwap±kσ` | **session VWAP + bands — to build** |

### Interaction types (the axis that was barely explored)
| Event | Meaning |
|---|---|
| `break` | close through the level (what was tested — mostly dead) |
| **`fade`** | reached the level but closed back — rejection |
| **`sweep`** | wick pokes through, close returns — liquidity grab / failed break |
| **`retest`** | break, then return to the level and hold |
| `approach` | price within X% of the level, no interaction yet |

**Variables:** `tol_pct` ∈ {0.02, 0.05, 0.1, 0.25} · `min_touches` ∈ {1, 2, 3} ·
level age · distance from EMA in ATR · direction (above/below).

---

## Phase 2 — Entry families to test as PRIMARY triggers

Each scored on the Phase-1 screen first; only survivors get a full backtest.

1. **Level fade** — the inverse of what was tested. Breaks failing is what the data keeps saying.
2. **Liquidity sweep + reverse** — sweep prev-day/session extreme, close back inside, enter the
   other way. Variables: wick depth ∈ {0.1, 0.25, 0.5} ATR, close-back-within ∈ {1, 2, 3} bars.
3. **Opening-range break / fade** — session OR of N min; enter break or fade. Variables: N,
   session, break vs fade, min OR width in ATR.
4. **Structure (BOS / CHoCH)** — enter on break of the last swing high/low from
   `add_zigzag_structure`; CHoCH = first opposing BOS. Variables: `atr_mult` ∈ {1.5, 2.5, 4},
   require HTF structure agreement.
5. **Volume-confirmed moves** — volume has never been touched. Variables: volume z-score ∈
   {1, 2, 3} over lookback ∈ {20, 50, 100}, relative-volume vs same-time-of-day.
6. **Squeeze release** — SQZMOM on→off with momentum sign as the trigger.
7. **WaveTrend cross** in OB/OS zones.
8. **RSI divergence** (confirmed, `right` ∈ {1,2,3,5}; and the zero-lag provisional variant).
9. **VWAP reversion / trend** — distance from session VWAP in σ.
10. **EMA-gradient extreme** — enter when |grad| is in its top/bottom decile.

---

## Phase 3 — Management (only on entries that survived Phase 2)

| Variable | Values |
|---|---|
| stop | pct {0.2,0.4,0.6,1.0} · atr ×{1,1.5,2,3} · leg/prev-pivot/high_of_n {10,20,40} · structure (last HL/LH) · level-based (other side of the level) |
| target | rr {1,1.5,2,3,4} · fixed % · trailing {0.5,1,2}% · next level · opposite candle {1,2,3} · partials |
| **time stop** | exit after N bars ∈ {5,15,30,60,120} — never tested, directly serves "quick trades" |
| breakeven | move stop to BE at {0.5, 1}R |
| secondaries | shaky bail · run-past-TP · profit-candle · none |

---

## Phase 4 — Regime & timing conditioners (applied to survivors, not searched blindly)

- **Volatility regime**: ATR percentile {low/mid/high}, efficiency ratio terciles (already built).
- **Session**: Asia / EU / US / overlaps — as a *conditioner* on a working entry, not a standalone
  filter (standalone hour-of-day is already proven dead).
- **Announcement proxy**: no calendar available, so proxy with volatility-spike windows — bars where
  realised vol jumps >Nσ; test trading *into*, *during*, and *after*.
- **HTF bias**: price vs 5m/15m/1h EMA {50,100,200}; structure agreement.
- **Weekend vs weekday** (crypto 24/7).

---

## Phase 5 — Risk (survivability, not alpha)

daily loss cap {1,2,4}% · max trades/day {1,3,5,10} · **consecutive-loss lockout** {2,3,4} (untested)
· risk/trade {0.25,0.5,1}% · volatility-scaled sizing.

---

## Phase 6 — Robustness gates (a config must pass ALL of these)

1. positive in **every** regime **after removing each window's best trade**;
2. `med_R` not pinned at −1.0;
3. a **plateau** in parameter space, not a spike;
4. trained pick lands near the top of the **test** ranking (all-candidates null);
5. ≥ ~300 trades;
6. survives spread ×2;
7. **works on a second instrument** — gold XAUUSD / PAXG are cached and have never been used;
8. drawdown survivable at intended leverage.

Nothing so far has passed more than 4 of these.

---

## Kill criteria (stated up front, to avoid rationalising later)

- Phase-1 drift-adjusted edge ≈ 0 and sign-flipping → drop the level, no backtest.
- Survivor count ≤ chance (`n_configs × 0.5^n_windows`) → treat the whole family as noise.
- Any result carried by one trade → discard, do not "investigate further".
- Parameter spike with negative neighbours → overfit, discard.

## Order of execution

Phase 1 (screen, fast) → Phase 2 on survivors → Phase 3 management → Phase 4 conditioners →
Phase 5 risk → Phase 6 gates. Document after **each** phase, negatives included.
