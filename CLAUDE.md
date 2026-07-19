# CLAUDE.md — Project Orientation

> Fast orientation for any new AI session or developer. Keep this current as the code evolves.
> **Companion docs:** [`README.md`](README.md) (full user guide), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> (design + roadmap), [`docs/INDICATOR_GUIDE.md`](docs/INDICATOR_GUIDE.md) (indicator reference).

## 1. What this is

`quant` — a Python quantitative research & backtesting platform. Purpose: **fast strategy
research and backtesting**, testing thousands→millions of parameter combinations over years of
1-minute data. Design: **vectorized signal generation + a numba-JIT position/PnL engine**.
Initial focus asset: **gold** (via Binance PAXGUSDT / XAUTUSDT; more providers planned).

Not for live trading. This was refactored from an organically-grown notebook codebase; the legacy
code has been removed (the user keeps an external backup). Everything lives in `quant/` now.

## 2. Architecture & data flow

```
quant.data.get_ohlcv(symbol, tf, start, end, source, tz)      # cached, incremental, pushdown load
   -> pandas OHLCV (open_time UTC + 't' display tz)
strategy.prepare(df)          # add vectorized indicator columns
strategy.signals(df)          # -> Signals (entry/exit bool arrays) + universal time filter
run_backtest(df, signals, cfg)  # numba kernel: SL/TP/trailing/partials/sizing -> SimResult
   -> SimResult(trades, equity_curve, stats)                  # stats incl Sharpe/Sortino/Calmar
quant.viz.ResearchChart / quant.optimize.run_grid             # responsive charts / param sweeps
```

Package layout:
```
quant/
├── config.py            # Settings + .env loading (secrets from env only)
├── logging_utils.py     # logger + tqdm helpers
├── data/                # get_ohlcv, sources (binance.py+_binance_fetch.py, dukascopy.py=true spot
│                        #   XAU/USD), store.py (pushdown), cache.py (generic incremental), timeframe.py
├── indicators/          # vectorized compute (overlays, oscillators, candles/HA, volatility, structure)
├── signals/             # primitives.py (numpy boolean predicates) + time_filters.py
├── engine/              # kernel.py (@njit), config.py (BacktestConfig/TakeProfit/Signals), run.py
├── strategies/          # base.Strategy + ema_ribbon, rsi, macd, heikin_ashi, supertrend, mtf, key_level
├── analytics/           # metrics.py (full stats) + fast.py (array-native sweep stats)
├── viz/                 # responsive.py (plotly-resampler, millions of pts) + charts.py (static)
└── optimize/            # grid.py (bounded-parallel sweeps) + search.py (optional optuna)
examples/gold_ema_demo.py · notebooks/ · experiments/ · tests/ · docs/ · data/ (parquet cache, gitignored)
```

`experiments/` is a SEPARATE inference layer that composes core APIs to find best settings for an
idea (best MTF EMA combo, best session, best stop-loss). **Never put experiment-specific logic in
`quant/`** — experiments import and orchestrate the core; the core stays generic.

Leverage/margin: `BacktestConfig(margin_enabled=True, leverage=…, contract_size=…,
sizing_mode="lots", stop_out_level=…)` — Exness-style used/free margin + stop-out liquidation
(reason `margin_call`). Opt-in; the non-margin path is unchanged (golden tests).

Costs (Exness-style): `spread` (fixed bid/ask WIDTH in price units; buy at ask, sell at bid — cost
scales with qty automatically), `commission_per_lot` (per side), plus `fee_bps`/`slippage_bps`.
Zero-cost defaults keep golden parity. Every `BacktestConfig` field is documented in its docstring.

Notebooks: `notebooks/01_research_cycle.ipynb` (full cycle) + `02_inference_experiments.ipynb`
(inference). Experiment method: `docs/EXPERIMENT_GUIDE.md`.

## 3. Key conventions & invariants (do not break)

- **Time:** `open_time` tz-aware UTC (cache); `t` = display-tz copy used by sim/plots/filters.
- **OHLCV** columns lowercase: `open/high/low/close/volume`.
- **Indicator columns:** vectorized helpers add named columns (e.g. `ema_50`, `rsi_14`, `macd`,
  `st_dir`, `swing_last_low`). MTF columns are prefixed `{tf}__` (e.g. `5min__ema_50`).
- **Signals are the single representation** for manual runs AND sweeps — vectorized numpy bool
  arrays (`quant.signals`). No more "express every strategy twice".
- **Anti-lookahead:** HTF features are shifted +1 HTF bar before as-of merge (`data/timeframe.py`);
  swing/pivot levels are only exposed after confirmation. Preserve this.
- **Engine (numba) is validated at exact parity** vs the original simulator — locked by golden
  regression tests (`tests/test_engine_golden.py`). Keep those green.
- **TP levels are fixed at entry** (rr uses the original stop distance) — intended, documented in
  `engine/config.py`.
- **Performance:** prefer vectorized/numpy or numba over Python loops; never `df.iloc[i]` in hot
  paths; keep the sweep path array-native (no per-combo DataFrames).

## 4. Environment

- Python 3.13, Windows (PowerShell primary; Bash available). Run from repo root with
  `PYTHONPATH=.` or `pip install -e .`.
- Installed & used: pandas, numpy, pyarrow, polars, **numba**, plotly, **plotly-resampler**,
  tsdownsample, joblib, scipy, tqdm, requests. (optuna optional, for search.)
- Data cache: `data/binance_{market}_{SYMBOL}_{tf}.parquet` (+ `.partials/` checkpoints).
  BTC/PAXG 1m ≈ 742k rows (2025-01 → 2026-05). Fits in RAM; the challenge is compute, solved by
  the JIT engine (~16-25 ms per full-year backtest, warm).

## 5. How to run

```python
from quant.data import get_ohlcv
from quant.strategies import EmaRibbon
from quant.engine import BacktestConfig
df  = get_ohlcv("PAXGUSDT", "1m", start="2025-06-01", end="2026-05-31", tz="UTC")
cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, exit_enabled=True,
                     sl_mode="entry_pct", sl_value=0.6, tp_mode="rr", tp_value=2.0,
                     sizing_mode="risk_pct_equity", sizing_value=1.0)
res = EmaRibbon(fast=50, slow=200, confirm_n=5).backtest(df, cfg)
print(res.stats)
```
- Demo/benchmark: `python examples/gold_ema_demo.py`
- Tests: `python -m pytest tests/` (parity/golden + strategy + viz smoke)
- New strategy = one dataclass in `quant/strategies/` (`prepare` + `build_signals`); sweep with
  `quant.optimize.run_grid`.

## 6. Working agreements for AI sessions

- Never commit/print secrets; secrets come only from `.env` (git-ignored). If keys were exposed,
  advise rotation (the user's action).
- Keep the golden engine tests green; add parity/golden tests when extending the kernel.
- Preserve anti-lookahead and the incremental cache.
- Update this file + README + docs when architecture changes.
