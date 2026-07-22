# CLAUDE.md — Project Orientation (full context for a fresh session)

> Read this first. It is the single source of truth for picking up this project cold — no prior
> conversation needed. Keep it current when the architecture changes.
> **Companion docs:** [`README.md`](README.md) (user guide) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
> (design + roadmap) · [`docs/INDICATOR_GUIDE.md`](docs/INDICATOR_GUIDE.md) · [`docs/EXPERIMENT_GUIDE.md`](docs/EXPERIMENT_GUIDE.md)
> · [`experiments/README.md`](experiments/README.md) · **[`docs/SESSION_NOTES.md`](docs/SESSION_NOTES.md)
> (research findings, gotchas, next steps — read this for running context).**

---

## 1. What this is

`quant` — a Python **quantitative research & backtesting platform**. Purpose: **fast strategy
research and backtesting**, testing thousands→millions of parameter combinations over years of
1-minute data. **Not for live trading.** Focus asset: **gold** (Binance PAXGUSDT/XAUTUSDT proxies,
and **true spot XAU/USD via Dukascopy**); the data layer is source-agnostic for other assets.

Design in one line: **vectorized signal generation + a numba-JIT position/PnL engine**. Signals are
whole-series boolean numpy arrays computed once; execution runs in a compiled kernel. Result: a
1-year 1-minute backtest runs in ~0.05–0.25 s (kernel ~6–48 ms); sweeps are ~200× faster than the
original per-bar simulator (which has been removed).

History: refactored from an organically-grown notebook codebase (now deleted; the user keeps an
external backup). Everything lives in `quant/` + `experiments/`.

## 2. Repo / project status

- **GitHub:** https://github.com/talalz94/Trading-Analysis · default branch **`main`**.
- Git user: Talal. Commit trailer used: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Secrets:** only from `.env` (git-ignored). `config.py`/`quant/config.py` load it. History is
  clean of keys. The user was advised to **rotate the old Binance keys** (read-only market keys;
  not needed for data — the public API is used).
- **Not committed (git-ignored):** `.env`, `data/` (parquet cache), `**/reports/`,
  `**/experiments/results/`, `__pycache__`, notebooks' checkpoints.
- **Tests:** `python -m pytest tests/` → **40 passing**. Keep them green.

## 3. Architecture & data flow

```
quant.data.get_ohlcv(symbol, tf, start, end, source, market, tz)      # cached, incremental, pushdown
   -> pandas OHLCV (open_time tz-aware UTC + 't' display-tz column)
strategy.prepare(df)          # add vectorized indicator columns
strategy.signals(df)          # -> Signals(entry/exit bool arrays); base applies time filters
run_backtest(df, signals, cfg)  # @njit kernel: SL/TP/trailing/partials/sizing/margin/costs -> SimResult
   -> SimResult(trades, equity_curve, stats)   # stats incl Sharpe/Sortino/Calmar/…
quant.viz.ResearchChart / quant.optimize.run_grid / experiments.Experiment
```

## 4. Package layout (complete, current)

```
quant/
├── config.py            # Settings + .env loading (secrets from env only)
├── logging_utils.py     # logger + tqdm helpers
├── cli.py               # `quant fetch|backtest|report` (entry point: quant = quant.cli:main)
├── data/
│   ├── base.py          #   DataSource protocol + OHLCV contract + validate_ohlcv
│   ├── binance.py       #   BinanceSource + get_source registry (binance, dukascopy)
│   ├── _binance_fetch.py#   proven incremental Binance klines fetcher (retries, partial checkpoints)
│   ├── dukascopy.py     #   DukascopySource — true spot XAU/USD + FX (needs dukascopy-python)
│   ├── cache.py         #   generic incremental Parquet cache for range-based providers
│   ├── store.py         #   pushdown load (column projection + date range) via polars/pyarrow
│   ├── loader.py        #   get_ohlcv(): fast cache path OR incremental fetch
│   └── timeframe.py     #   resample_ohlcv + align_timeframes + build_mtf (anti-lookahead)
├── indicators/          # vectorized compute ONLY (no plotting); recursive parts in numba
│   ├── overlays.py      #   ema, sma, add_emas, add_smas
│   ├── oscillators.py   #   add_rsi (Wilder), add_macd, add_stochastic
│   ├── candles.py       #   add_heikin_ashi (recursive ha_open in numba)
│   ├── volatility.py    #   atr, add_atr, add_supertrend (recursive part in numba)
│   └── structure.py     #   add_swings (swing_last_low/high = ref_col stop sources), add_pivot_points
├── signals/
│   ├── primitives.py    #   numpy-native bool predicates (cross_up, last_all_above, consecutive_*,
│   │                    #     refs_ordered, rising/falling, all_of/any_of/none_of, …)
│   └── time_filters.py  #   in_session, hour_between, between_times, weekday_in, not_weekend
├── engine/
│   ├── kernel.py        #   @njit run_kernel (the hot core) + equity_stats reducer + _fill_px/_costs
│   ├── config.py        #   BacktestConfig (fully documented), TakeProfit, Signals
│   └── run.py           #   run_backtest(df, signals, cfg) -> SimResult ; invoke_kernel helper
├── strategies/
│   ├── base.py          #   Strategy ABC (prepare + build_signals; signals() adds time filters)
│   ├── ema_cross.py     #   configurable price-vs-EMA (cross/close/full_candle, confirm_n, HA, HTF
│   │                    #     bias, exit_mode opposite/none/ha_flip/below_ema) — the variant explorer
│   └── ema_ribbon, rsi, macd, heikin_ashi, supertrend, mtf, key_level  (+ REGISTRY)
├── analytics/
│   ├── metrics.py       #   compute_stats (return, win/loss, PF, expectancy, Sharpe/Sortino/Calmar,
│   │                    #     drawdown, recovery, side splits)
│   ├── attribution.py   #   by_hour/weekday/month/session/regime + monthly_returns
│   └── fast.py          #   array-native stats for sweeps (no DataFrame build) + equity_stats use
├── viz/
│   ├── responsive.py    #   ResearchChart / price_chart / equity_chart (plotly-resampler; millions of pts)
│   ├── charts.py        #   static price+trades, equity+drawdown (HTML-export safe)
│   └── heatmaps.py      #   monthly-returns, hour×weekday, parameter-sweep heatmaps
├── optimize/
│   ├── grid.py          #   run_grid (bounded-parallel sweep; fast array path; backend threading/loky)
│   └── search.py        #   optuna_search (optional; Bayesian)
└── reporting/report.py  #   summary / print_summary / to_html (standalone offline report)

experiments/             # INFERENCE layer — find best settings for an idea (SEPARATE from core)
├── base.py              #   Experiment (strategy_space + cfg_space, objective, run() -> results/)
├── ema_mtf.py · session_timing.py · exit_design.py   # worked examples (each with a description)
└── README.md
notebooks/               # 01_research_cycle · 02_inference_experiments · 03_ema_cross_study (Q1-Q9)
                         #   · 04_trend_runner_leverage · 05_simple_1m_ema_breakout
scripts/fetch_gold_dukascopy.py   # resumable true-spot XAUUSD fetch (1m/5m/15m/1h/4h)
examples/gold_ema_demo.py · tests/ (9 files) · docs/ · data/ (cache, git-ignored)
```

## 5. The engine (`quant/engine`) — capabilities

`BacktestConfig` (every field documented in its docstring — do `help(BacktestConfig)`):

- **Stop loss** (`sl_mode`): `none` | `entry_pct` | `price_abs` | `ref_col` (structure level from a
  column, e.g. `sl_ref_long_col="swing_last_low"`, with `sl_buffer_pct`, `sl_max_ref_risk_pct`,
  fallback). **Trailing** (`trail_mode` `pct`/`price_abs`, ratchets on high-water mark).
- **Take profit**: `take_profits=(TakeProfit(mode, value, close_pct, move_stop_mode, move_stop_value), …)`
  laddered/partial; or convenience `tp_mode`/`tp_value`. Modes: `entry_pct`|`price_abs`|`rr`.
  **TP prices are FIXED at entry** (rr uses the original stop distance) — intentional, documented.
- **Sizing** (`sizing_mode`): `cash` | `risk_pct_equity` (compounds) | `risk_amount` | `lots`.
- **Costs (Exness-style):** `spread` = fixed bid/ask WIDTH in price units (buy at ask, sell at bid;
  cost = spread × qty = spread × contract_size × lots — scales with volume automatically),
  `commission_per_lot` (per side), `fee_bps` (% per side), `slippage_bps`.
- **Leverage/margin (opt-in `margin_enabled`):** `leverage`, `contract_size` (gold=100 oz/lot),
  used/free margin, free-margin gate on entry, **stop-out liquidation** at `stop_out_level`% margin
  level (close reason `margin_call`). Non-margin path uses `allow_leverage`/`max_notional_pct` cap.
- **Fills:** `max_open_trades` (multi-position), `allow_short`, `intrabar_priority`
  (`stop_first`/`take_profit_first`), force-close at end.
- Close-reason codes: `signal`, `stop_loss`, `take_profit`, `forced_close_end`, `margin_call`.

`SimResult` = `trades` (DataFrame; incl. per-trade `stop_price`), `equity_curve`
(t/equity/open_trades/drawdown), `stats` (dict),
`elapsed_s` (kernel time).

## 6. Key conventions & invariants (DO NOT BREAK)

- **Time:** `open_time` tz-aware UTC (cache); `t` = display-tz copy used by sim/plots/filters.
- **OHLCV** columns lowercase `open/high/low/close/volume`.
- **Indicator columns** named (e.g. `ema_50`, `rsi_14`, `macd`, `st_dir`, `swing_last_low`); MTF
  columns prefixed `{tf}__` (e.g. `5min__ema_50`).
- **Signals are the single representation** for manual runs AND sweeps (numpy bool arrays). Never
  express a strategy twice.
- **Anti-lookahead:** HTF features shifted +1 HTF bar before as-of merge (`data/timeframe.py`);
  swing/pivot levels exposed only after confirmation (`indicators/structure.py`). Preserve this.
- **Golden parity:** the numba kernel was validated at exact per-trade parity vs the original
  simulator; locked by `tests/test_engine_golden.py`. **Zero-cost / non-margin defaults must stay
  byte-identical** — when extending the kernel, keep those paths unchanged and re-run golden tests.
- **Performance:** vectorized numpy / numba over Python loops; never `df.iloc[i]` in hot paths; keep
  the sweep path array-native (no per-combo DataFrames); hoist config-derived kernel args out of loops.

## 7. Data layer specifics

- `get_ohlcv(symbol, tf, start, end, source="binance", market="spot", tz="UTC", refresh=False)`.
- Sources: **`binance`** (crypto + PAXG/XAUT gold proxies) and **`dukascopy`** (true spot XAUUSD/FX
  from 2003; `pip install "quant[data]"`; live fetch verified working). Add a provider by
  implementing `DataSource.fetch` and routing ranges through `quant.data.cache.incremental_fetch`.
- Cache file: `data/{source}_{market}_{SYMBOL}_{tf}.parquet` (+ `data/.partials/` for Binance
  checkpoints). Incremental (only missing ranges fetched); reads use column+date pushdown.
- MTF: `build_mtf(df, {"5min": lambda d: add_emas(d,[50,200])})` → `5min__ema_50` on the base grid.

## 8. Optimization & experiments

- **Sweeps:** `quant.optimize.run_grid(df, StrategyCls, grid, cfg, valid_fn=…, keep_stats=…)` —
  precomputes indicators once, array-native stats, bounded parallel (`n_jobs=CPU−2`, threading).
  `backend="loky"` exists but is usually **slower** here (pickles the frame). **Optuna** via
  `optimize.search.optuna_search` for large continuous spaces.
- **Speed levers for millions of combos:** subset-then-refine (shorter window / coarser TF first),
  then confirm on full 1m; use Optuna over brute grid. Rough: full-year 1m ~130 ms/combo threaded;
  3-month window ~30 ms/combo.
- **Experiments/inference** (`experiments/`): declare `strategy_space` (strategy fields incl
  `session`/`hours`) + `cfg_space` (BacktestConfig fields incl SL/TP/leverage), an objective, and a
  `description`. `Experiment.run()` sweeps (grouping by strategy so signals compute once, cfg
  variants reuse them), ranks, and writes `experiments/results/<name>/` (results.csv, best.json,
  report.md). **Never put experiment logic in `quant/` core.** See `docs/EXPERIMENT_GUIDE.md`.

## 9. Environment

- Python **3.13**, Windows (PowerShell primary; Bash available). Run from repo root with
  `PYTHONPATH=.` or `pip install -e .`. Jupyter kernel must be Python ≥3.11 / pandas ≥2.2.
- Installed & used: pandas, numpy, pyarrow, polars, **numba**, plotly, **plotly-resampler**,
  tsdownsample, joblib, scipy, tqdm, requests, dukascopy-python. Optional: optuna.
- Data cached: BTC/PAXG 1m ≈ 742k rows (2025-01 → 2026-05), 5m/15m/1h derived. Fits in RAM; the
  challenge is compute, solved by the JIT engine.

## 10. How to run

```python
from quant.data import get_ohlcv
from quant.strategies import EmaRibbon
from quant.engine import BacktestConfig
df  = get_ohlcv("PAXGUSDT", "1m", start="2025-06-01", end="2026-05-31", tz="UTC")   # or source="dukascopy", "XAUUSD"
cfg = BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5, spread=0.20, exit_enabled=True,
                     sl_mode="entry_pct", sl_value=0.6, tp_mode="rr", tp_value=2.0,
                     sizing_mode="risk_pct_equity", sizing_value=1.0)
res = EmaRibbon(fast=50, slow=200, confirm_n=5).backtest(df, cfg)
print(res.stats)
```
- Notebooks: `notebooks/01_research_cycle.ipynb` (full cycle), `02_inference_experiments.ipynb`.
- CLI: `quant backtest --symbol PAXGUSDT --tf 1m --start 2025-06-01 --strategy ema_ribbon --params '{"fast":50,"slow":200,"confirm_n":5}'`
- Demo: `python examples/gold_ema_demo.py` · Tests: `python -m pytest tests/`
- New strategy = one dataclass in `quant/strategies/` (`prepare` + `build_signals`); sweep with `run_grid`.

## 11. Tests (what's covered)

`tests/`: `test_engine_golden.py` (numeric regression vs legacy-validated engine),
`test_leverage.py` (margin/stop-out/lots), `test_costs.py` (spread/commission — incl the EUR/USD
$12 example), `test_signals.py` (primitive correctness), `test_strategies.py` (all 7 run),
`test_analytics.py` (attribution + reporting), `test_viz.py` (responsive chart resamples),
`test_dukascopy.py` (provider mapping + incremental cache, mocked fetch). 40 total (incl. costs, leverage, dukascopy, EmaCross).

## 12. Known limitations / honest caveats

- Data providers: only **binance** + **dukascopy** implemented (OANDA/CSV/metals-API planned).
- Leverage: models used/free margin + stop-out, but **no swap/overnight financing**; stop-out
  liquidates **all** positions (not partial); spread is a fixed width (no dynamic/time-varying spread).
- No **walk-forward / out-of-sample** automation yet (the experiment guide describes doing it manually).
- Responsive chart interactivity needs a **live Jupyter kernel** (static HTML export keeps only the
  initial view); static charts are the fallback.
- Very large sweeps ("millions") realistically need Optuna + subset-then-refine, not brute force on
  full-year 1m (see §8).

## 13. Roadmap (next candidates)

OANDA practice provider; CSV/TradingView import; walk-forward/OOS tooling; swap/rollover + partial
stop-out; (if needed) per-strategy fused-numba signal generation for extreme sweeps; add a `LICENSE`.

## 14. Working agreements for AI sessions

- Never commit/print secrets; secrets come only from `.env`. If exposed, advise rotation (user's action).
- Keep `tests/` green; keep golden parity (zero-cost/non-margin paths byte-identical); add
  parity/golden tests when extending the kernel.
- Preserve anti-lookahead + the incremental cache.
- Put experiment-specific logic in `experiments/`, never in `quant/` core.
- Update this file + README + docs when architecture changes. Commit in logical units; push only
  when asked (remote `origin` → GitHub `main`).
