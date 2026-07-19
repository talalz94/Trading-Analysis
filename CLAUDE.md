# CLAUDE.md — Project Orientation

> Purpose: let any new AI session (or developer) understand this project quickly and
> continue development safely. Keep this file up to date as the architecture evolves.
>
> **Companion docs:**
> - [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — the proposed target architecture + phased refactor roadmap.
> - [`docs/INDICATOR_GUIDE.md`](docs/INDICATOR_GUIDE.md) — indicator/feature library reference + usage.

---

## 1. What this project is

A **Python quantitative research & backtesting platform** for commodity/asset trading.
Primary purpose: **fast strategy research and backtesting on historical data** — *not* live
trading. The intended workload is testing **thousands–millions of strategy/parameter
combinations** across **years of 1-minute data** on multiple assets.

Initial focus asset: **Gold**. Today gold is proxied via Binance **PAXGUSDT** (PAX Gold) and
**XAUTUSDT** (Tether Gold). BTCUSDT is also cached. True spot gold / OANDA / metals-API
sources are a planned addition (see ARCHITECTURE.md → Data layer).

## 2. Current status (2026-07)

The code **works** but has grown organically. Key realities to know before touching anything:

- **Not a git repository.** Versioning is done by hand-copying files (`data-Copy1.py`,
  `simulator - Copy (2).py`, `Main_Simulation-Copy3.ipynb`, …). There are **24** such
  backup/`Copy` files. **First recommended action of any refactor: `git init`** so we can
  delete the copies safely. Until then, do not delete `*Copy*` files without asking.
- **Strategies live in ~17 near-duplicate Jupyter notebooks** (`Main_Simulation-*.ipynb`).
  ~60–75% of each notebook is copy-pasted boilerplate; only the "rules" cell differs.
- **Secrets are committed in plaintext:** `config.py` holds live Binance `api_key`/`api_secret`.
  These should be rotated and moved to environment variables / `.env` (git-ignored). The keys
  are not actually needed for public historical klines (`data.py` uses the public data API).
- The high-performance stack is **already installed** (see §8): `polars`, `numba`, `vectorbt`,
  `pyarrow`, `joblib`, `scipy`. The current code uses almost none of it — everything is
  hand-rolled pandas + a Python per-bar loop.

## 3. Repository map

```
Binance/
├── config.py                 # ⚠ Binance API key/secret in plaintext (rotate + move to env)
├── data.py                   # Data layer: cached Binance klines fetch (MATURE, keep/evolve)
├── data-Copy1.py, -Copy2.py  # stale backups (ignore)
├── pipeline.py               # build_feature_df(): raw OHLCV -> +MAs -> +indicator columns
├── indicators_load.py        # INDICATOR_REGISTRY (dict name->Indicator class) + col() helper
├── indicators/               # Indicator library (compute + plotly add_traces per indicator)
│   ├── moving_average.py, macd.py, stochastic.py, bollinger_bands.py, momentum.py,
│   │   volume_ma.py          # ✅ vectorized
│   ├── market_structure.py, supertrend.py, rsi_divergence.py,
│   │   trend_channels.py, support_resistance.py  # ⚠ contain per-bar Python loops (slow)
│   ├── precomputed.py        # "plot-only" pseudo-indicator: validates existing columns
│   └── *-Copy1/2.py          # stale backups (ignore)
├── features/                 # Vectorized derived-signal columns (the "good" direction)
│   ├── ema_spreads.py, ema_compression.py, cross_through.py,
│   │   ema_diagnostics.py, stochastic_signals.py
├── precomputed_factory.py    # Rewrites a compute-spec into a plot-only "precomputed" spec
├── simulation/
│   ├── rules.py              # Rule / RuleGroup / ALL / ANY  (lambda-based strategy DSL)
│   ├── context_mixins.py     # RuleContextMixin: ~70 per-bar boolean predicates (the DSL vocab)
│   ├── rule_features.py      # ⭐ VECTORIZED versions of some predicates (add_cross_compare, …)
│   ├── simulator.py          # ⭐ run_simulation(): the per-bar event-loop engine (HOT PATH)
│   ├── timeframe_utils.py    # HTF feature shifting (anti-lookahead) + feature_columns()
│   ├── context_mixins-Copy1.py, simulator*-Copy*.py  # stale backups (ignore)
├── analytics.py              # trades_to_frame() + a couple of plotly trade charts
├── plotter.py                # plot_interactive(): candles + indicator overlays (plotly)
├── plot_simulation.py        # plot_simulation(): adds trade entries/exits/SL/TP markers
├── mtf_plot.py               # multi-timeframe plot helpers
├── plot_toggles.py           # toggle which indicators render on the chart
├── ema_diagnostic_plots.py   # EMA spread / cut-through diagnostic panels
├── research/ema_move_analyzer.py  # feature-gen + forward-return threshold grid search
├── Main_Simulation-*.ipynb   # ~17 strategy notebooks (EMA, supertrend, stochastic, MACD, …)
├── optimization_results_*.csv# saved sweep outputs (loose CSVs, some hand-copied)
└── data/
    ├── binance_spot_{SYMBOL}_{TF}.parquet   # main cache (one file per symbol+timeframe)
    └── .partials/<stem>/part_<startms>_<endms>.parquet  # incremental download checkpoints
```

## 4. Data flow (end-to-end, as it works today)

```
fetch_binance_klines(symbol, interval, start, end)          [data.py]
    -> reads data/binance_{market}_{SYMBOL}_{TF}.parquet + .partials/*
    -> fetches ONLY missing ranges from Binance (incremental), checkpoints each page
    -> returns tidy OHLCV DataFrame (open_time tz-aware UTC, open/high/low/close/volume, …)
        │
        ▼
build_feature_df(raw_df, tz, ma_windows, indicators=[IndicatorSpec(...)])   [pipeline.py]
    -> adds 't' = open_time.tz_convert(tz)   (the plotting/sim time axis)
    -> adds MA{w} columns
    -> for each IndicatorSpec: INDICATOR_REGISTRY[name].compute(df, cfg, tag)
       creates columns named  "{tag}__{BASE}"  (e.g. "ema__EMA_50", "rsi14__RSI")
        │  (run once per timeframe: df_1m, df_5m, df_15m, df_1h)
        ▼
align_timeframes(base_df=df_1m, other_dfs={"5m":df_5m, "15m":df_15m, "1h":df_1h})  [simulator.py]
    -> HTF feature columns shifted +1 candle (anti-lookahead)  [timeframe_utils.py]
    -> merge_asof onto 1m 't'; columns become  "{tf}__{tag}__{COL}"  (e.g. "5m__ema__EMA_50")
    -> optional features: add_ema_compression_features(), add_cross_through_features(), …
        │  => one wide "merged" DataFrame on the 1m grid
        ▼
Strategy(open_rules_long=ALL(Rule("...", lambda c: c.cross_up_pair("close","MA50")), ...),
         close_rules_long=ALL(...))                                    [simulation/rules.py]
        │
        ▼
run_simulation(df=merged, strategy, cfg=SimConfig(exit=TradeExitConfig(...)))  [simulator.py]
    -> for i in range(n_bars):            # ⚠ PURE PYTHON PER-BAR LOOP (the bottleneck)
         mark-to-market open trades; manage SL/TP/partials/trailing;
         eval close rules (lambdas); eval open rules (lambdas); size & open trades
    -> returns SimResult(trades, events, equity_curve, stats)
        │
        ▼
analytics.trades_to_frame(...) / _build_stats(...)  -> metrics
plot_simulation(...) / plot_interactive(...)        -> plotly charts
```

## 5. Key conventions & data contract

- **Time:** `open_time` is tz-aware UTC (from Binance ms). `t` = `open_time` converted to a
  display tz (e.g. `Asia/Karachi`). Simulation & plotting use `t`; the cache stores `open_time`.
- **OHLCV columns** are lowercase: `open, high, low, close, volume` (+ `quote_volume`,
  `num_trades`, `taker_buy_base`, `taker_buy_quote`).
- **Indicator output columns:** `f"{tag}__{BASE}"`. `tag` is a unique per-instance id you choose
  in the `IndicatorSpec`. Cross-timeframe (after `align_timeframes`) become `f"{tf}__{tag}__{COL}"`.
  Feature columns use `f"{name}__{suffix}"`.
- **Anti-lookahead:** higher-timeframe features are shifted +1 HTF candle before merge_asof so a
  5m candle stamped 10:00 (which closes at 10:05) is only visible to 1m bars from 10:05 onward.
  Preserve this invariant in any redesign.
- **Strategy DSL:** `Rule(name, fn)` where `fn` is `lambda c: <bool>` using the `RuleContextMixin`
  vocabulary (`c.v(col, shift)`, `c.gt/lt`, `c.cross_up_pair`, `c.consecutive_green`, `c.flag`,
  `c.close_above_all`, `c.crossed_through_refs`, …). Combine with `ALL(...)` / `ANY(...)`.

## 6. The exit / risk engine (simulator.py) — capabilities that must be preserved

`TradeExitConfig` already supports a rich, realistic execution model. Any new engine must keep
semantic parity:
- **Stop loss** (`StopLossConfig`): `entry_pct`, `price_abs`, or `ref_col` (structure-based, e.g.
  swing low/high column) with buffer, `max_ref_risk_pct` cap, and fixed fallback.
- **Take profit** (`TakeProfitConfig`, multiple, laddered): `entry_pct`, `price_abs`, or `rr`
  (R-multiple); `close_pct` for **partial exits**; post-TP **stop movement** (`breakeven`,
  `entry_pct`, `price_abs`, `ref_col`).
- **Position sizing** (`PositionSizingConfig`): `cash`, `risk_pct_equity`, `risk_amount`;
  notional cap; optional leverage.
- **Fills:** slippage (bps), fees (bps), `intrabar_priority` (stop_first / take_profit_first),
  `max_open_trades` (multi-position), force-close at end, compounding via mark-to-market equity.

## 7. Known bottlenecks & tech debt (see ARCHITECTURE.md for the fix)

1. **Per-bar Python event loop** in `run_simulation` (`simulator.py:1237`) is the dominant cost.
   Every strategy rule is a Python lambda evaluated one bar at a time (~742k bars/asset/year).
   `df.iloc[i]` Series creation per bar (`:1173`, `:1295`) compounds it. Infeasible for millions
   of combos.
2. **Two parallel signal systems.** Manual notebooks use the ergonomic per-bar lambda DSL
   (`context_mixins`); optimizer notebooks bypass it by hand-writing NumPy boolean masks into
   `__open_signal`/`__close_signal` flag columns. **Every strategy is expressed twice.** Unify
   into one vectorized signal layer.
3. **Slow indicators** with per-bar loops: `support_resistance.py:425`, `trend_channels.py:287`,
   `supertrend.py:32/49`, `rsi_divergence.py:40`, `market_structure.py:78`. Candidates for numba.
4. **Storage:** one monolithic Parquet per symbol+TF; appending rewrites the whole file. No
   `source` dimension (Binance-only). No column projection / date-range pushdown on read.
5. **Layer coupling:** indicators mix `compute()` (data) with `add_traces()` (plotly viz).
   `INDICATOR_REGISTRY` is duplicated in `indicators_load.py` (12) and `indicators/__init__.py` (10, stale).
6. **No config files, no CLI, no tests, no packaging** (`requirements.txt`/`pyproject.toml` absent).
   Experiments are not reproducible; results are loose CSVs.
7. **Massive notebook duplication** (~17 strategy notebooks + 24 `Copy` files).

## 8. Environment

- **Python 3.13.1** (Windows). Shell: PowerShell (primary), Bash available.
- Installed & relevant: `pandas 2.3`, `numpy 2.3`, `pyarrow 22`, **`polars 1.35`**,
  **`numba 0.62`**, **`vectorbt 0.28`**, `plotly 6.3`, `tqdm`, `scipy 1.16`, `joblib 1.5`,
  `requests`, `matplotlib`.
- Not installed (candidates to add): `duckdb`, `optuna`, `bottleneck`/`numexpr`, `ta-lib`.
- Data cached: BTC/PAXG 1m ≈ **742k rows** each (2025-01 → 2026-05); 5m/15m/1h derived. XAUT from
  2026-03. ~109 MB total. So **data fits comfortably in RAM** — the challenge is compute, not I/O.

## 9. How to run something today (quick reference)

There is no CLI yet; runs happen in notebooks. Minimal programmatic flow:

```python
from data import fetch_binance_klines
from pipeline import build_feature_df, IndicatorSpec
from simulation.simulator import Strategy, SimConfig, run_simulation, align_timeframes
from simulation.rules import Rule, ALL

raw = fetch_binance_klines("PAXGUSDT", "1m", start="2025-01-01", market="spot")
df, specs, ma_cols = build_feature_df(raw, tz="UTC", ma_windows=[50, 200])
strat = Strategy(open_rules_long=ALL(Rule("cross", lambda c: c.cross_up_pair("close", "MA50"))),
                 close_rules_long=ALL(Rule("cross dn", lambda c: c.cross_down_pair("close", "MA50"))))
res = run_simulation(df, strat, SimConfig(initial_cash=10_000, fee_bps=10))
print(res.stats)
```

## 10. Working agreements for AI sessions

- **Do not commit or print secrets.** If you touch `config.py`, move secrets to env and flag it.
- **Preserve the incremental-download cache** (`data.py` + `data/.partials/`) — it's proven.
- **Preserve anti-lookahead** HTF shifting and the exit-engine semantics (§6).
- Prefer **vectorized / numba** implementations over Python loops. Avoid `df.iloc[i]` in hot paths.
- Update this file and the companion docs when you change architecture.
