# Session Notes / Handoff

> Point a fresh session or model here for the **non-obvious context** — research findings, gotchas,
> recently-fixed bugs, and open threads — that isn't derivable from the code alone. Pair with
> [`../CLAUDE.md`](../CLAUDE.md) (architecture) and [`../README.md`](../README.md) (usage).
> Keep this current; it is the running memory of the project.

## Current state (2026-07)
- Repo: https://github.com/talalz94/Trading-Analysis · branch `main`. **40 tests pass** (`pytest tests/`).
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
