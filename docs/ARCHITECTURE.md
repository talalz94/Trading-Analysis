# Proposed Architecture & Refactor Roadmap

> Status: **APPROVED — implementation underway.** Decisions in §9 are settled. This document
> describes the *target* design; the legacy system is documented in [`../CLAUDE.md`](../CLAUDE.md).
>
> **Progress:**
> - ✅ Phase 0 — foundations: git init, `.gitignore`/`.gitattributes`, secrets → `.env`, `pyproject.toml` + pinned `requirements.txt`.
> - ✅ Phase 1 — end-to-end vertical slice: the `quant/` package (data → signals → numba engine → analytics → viz → optimize) with an `EmaRibbon` gold strategy, **validated at exact numerical parity** vs the legacy engine and ~1000× faster. See `examples/gold_ema_demo.py` and `tests/`.
> - ⏭ Next — engine feature completion (partial/laddered TPs, trailing/stop-movement, `ref_col` structure stops, multi-timeframe alignment), then port remaining strategies and retire notebooks.

---

## 1. Design goals (in priority order)

1. **Speed** — test thousands→millions of parameter combos over years of 1m data.
2. **Maintainability & modularity** — clean layer boundaries, no duplicated logic.
3. **Scalability** — multiple assets, multiple sources, larger histories.
4. **Extensibility** — a new strategy or indicator is a small, plug-in unit of code.
5. **Research productivity** — reproducible experiments, good logging/progress, clean charts.
6. **Considerate compute** — never saturate the laptop; worker count is bounded and configurable.

## 2. The core performance decision (why everything else follows)

The current engine is a **Python per-bar event loop** that evaluates each strategy rule as a Python
lambda, bar by bar (`simulator.py`). At ~742k bars/asset/year, one backtest is already slow; a
million-combo sweep is infeasible. The optimizer notebooks already worked around this by
precomputing entry/exit **NumPy boolean masks** and feeding a trivial rule that just reads a flag
column — proving the path forward.

**Target model = separate signal generation from position simulation:**

```
        VECTORIZED                         COMPILED (numba) or VECTORBT
┌────────────────────────────┐        ┌──────────────────────────────────┐
│ indicators + features      │        │ position/PnL engine:             │
│ → entry_long[], exit_long[]│  ───▶  │ walk bars once in machine code,  │
│ → entry_short[], exit_short│        │ apply SL/TP/partials/trailing/   │
│ (bool arrays, whole series)│        │ sizing/fees/slippage → trades    │
└────────────────────────────┘        └──────────────────────────────────┘
   done once per param set,               ~milliseconds per backtest;
   fully numpy/polars                     sweeps are embarrassingly parallel
```

Signals are pure vectorized array math (fast, and identical for manual runs *and* sweeps — one
representation, not two). Only the **path-dependent** part (a stop that depends on the fill of an
earlier TP, trailing stops, multi-position cash) needs a sequential pass — and that pass is JIT
compiled, so 742k bars run in milliseconds.

**Recommended engine:** a **purpose-built `@njit` numba engine** that reproduces the existing
`TradeExitConfig` semantics exactly (structure `ref_col` stops, laddered partial TPs, post-TP stop
moves, intrabar priority, risk sizing, multi-position). Rationale: those semantics are specific and
already well-specified; a custom kernel preserves them precisely and keeps full control.
`vectorbt` (already installed) is kept available as a fast path for simple signal-only sweeps and
for cross-checking the custom engine's numbers. See §9 for the decision.

## 3. Target package layout

Convert the loose scripts into an installable package `quant/` (name TBD), with notebooks/CLIs as
thin entry points:

```
quant/
├── config/                 # pydantic-settings; YAML/TOML experiment configs; secrets from env
│   └── settings.py
├── data/                   # DATA LAYER (source-agnostic)
│   ├── sources/            # pluggable providers behind one interface
│   │   ├── base.py         #   DataSource protocol: fetch(symbol, tf, start, end) -> frame
│   │   ├── binance.py      #   ← today's data.py logic (incremental cache, retries) refactored
│   │   ├── oanda.py        #   spot gold / FX (planned)
│   │   ├── metals_api.py   #   spot XAU (planned)
│   │   └── csv_file.py     #   TradingView / arbitrary CSV import
│   ├── store.py            # Parquet/Arrow store: partitioned, incremental, column+range pushdown
│   └── catalog.py          # what's cached, ranges, gaps (from cache_info)
├── indicators/             # pure compute only (NO plotting) — vectorized/numba kernels
│   ├── base.py, registry.py
│   └── *.py                # ema, macd, rsi, supertrend(njit), market_structure(njit), sr(njit)…
├── features/               # derived vectorized signal columns (as today, cleaned up)
├── signals/                # UNIFIED signal layer (replaces dual DSL/mask systems)
│   ├── primitives.py       # vectorized cross_up, prev_all_below, consecutive_green, refs_ordered…
│   └── expr.py             # compose primitives → entry/exit boolean arrays
├── engine/                 # SIMULATION ENGINE
│   ├── kernel.py           # @njit core: bars → trades/equity (SL/TP/partial/trail/sizing)
│   ├── run.py              # Python wrapper: config → arrays → kernel → SimResult
│   └── vbt_adapter.py      # optional vectorbt path
├── strategies/             # PLUG-AND-PLAY strategies
│   ├── base.py             # Strategy protocol: params schema + build_signals(df, params)
│   └── ema_ribbon.py, supertrend.py, stochastic.py, …  # one file each, param-driven
├── optimize/               # parameter sweeps
│   ├── grid.py             # grid/random search; resumable; checkpointed to parquet
│   ├── search.py           # optuna (Bayesian) wrapper (optional)
│   └── runner.py           # joblib parallelism, bounded workers, progress
├── analytics/              # metrics + attribution (Sharpe, Sortino, by hour/day/session/regime)
├── viz/                    # plotly charts (price+trades, equity, drawdown, heatmaps, returns)
├── reporting/              # HTML/notebook report assembly, comparison tables
└── cli.py                  # `quant fetch|backtest|optimize|report`

experiments/                # reproducible experiment configs (YAML) + outputs
tests/                      # unit tests (indicators, signals, engine parity, metrics)
```

Notebooks become **thin**: load config → call `quant` functions → show charts. No algorithm code.

## 4. Layer-by-layer plan

### 4.1 Data layer
- **Keep** the proven incremental-download + partial-checkpoint mechanism from `data.py`; refactor
  it behind a `DataSource` interface so Binance is one provider among several.
- Add a **`source`** dimension to cache keys: `{source}_{market}_{symbol}_{tf}` (today it's
  Binance-only, so `source` is implied). Lets Binance-XAUT, OANDA-XAUUSD, metals-API-XAU coexist.
- **User selects source** per run via config. Cache stays local; only missing ranges are fetched.

### 4.2 Storage
- Move from one monolith Parquet/symbol/TF to a **date-partitioned dataset** (e.g. by month) via
  pyarrow/polars, so incremental appends write a new partition instead of rewriting the whole file.
- Read path uses **column projection + timestamp predicate pushdown** (polars `scan_parquet` /
  pyarrow dataset) — load only the columns and date range a run needs → lower memory.
- Keep Parquet (columnar, compressed, ubiquitous). Feather/Arrow considered for hot intermediate
  caches. Optional dictionary/`zstd` compression.
- Cache computed **indicator/feature frames** keyed by (symbol, tf, indicator-set hash) so sweeps
  don't recompute EMAs every run.

### 4.3 Indicators
- **Split compute from plotting.** `indicators/` becomes pure functions returning columns; plotly
  rendering moves to `viz/`. Removes the plotly dependency from the compute path and makes outputs
  cacheable.
- **Vectorize/JIT the slow ones** (`support_resistance`, `trend_channels`, `supertrend`,
  `rsi_divergence`, `market_structure`) with `@njit` kernels over numpy arrays.
- Single source-of-truth registry; delete the stale `indicators/__init__.py` copy.

### 4.4 Unified signal layer
- Re-implement the ~70 `RuleContextMixin` predicates as **vectorized column ops** (`signals/
  primitives.py`) — each returns a full boolean array. `cross_up_pair(a,b)` →
  `(a.shift()<=b.shift()) & (a>b)`; `consecutive_green(n)` → rolling sum of green flags; etc.
- A strategy composes primitives into `entry_long/exit_long/entry_short/exit_short` arrays. This is
  the **single** representation used by both interactive runs and sweeps (kills the double-expression
  problem).
- Anti-lookahead HTF shifting is preserved at the alignment step.

### 4.5 Simulation engine
- `engine/kernel.py`: one `@njit` function consuming numpy arrays (`open/high/low/close`, signal
  bools, SL/TP/sizing params) → arrays of trade records + equity curve.
- Full parity with today's `TradeExitConfig` (§6 of CLAUDE.md). Validated against the current engine
  on identical inputs before cutover.
- `engine/run.py` keeps a friendly Python API (`run_backtest(df, strategy, cfg) -> SimResult`).

### 4.6 Optimization
- `optimize/grid.py`: enumerate param space (grid/random), **resumable** (checkpoint to Parquet, not
  loose CSV), dedup by param hash.
- `optimize/runner.py`: **joblib** parallelism with **bounded workers** — default
  `n_jobs = max(1, cpu_count - 2)` and a config cap, so the laptop stays usable (directly addresses
  the "don't render my laptop unusable" requirement). Per-combo work is tiny (signals + JIT sim), so
  throughput scales with cores.
- `optimize/search.py`: optional **optuna** Bayesian search for large spaces where grid is wasteful.

### 4.7 Analytics
- Extend metrics beyond today's set to include **Sharpe, Sortino, Calmar/recovery, expectancy,
  profit factor, max drawdown, avg trade, exposure** (some already exist in `_build_stats`).
- **Attribution:** best hour-of-day, weekday, month, market session, volatility regime, trend
  regime — grouped/vectorized over the trades+bars frames.
- Robustness helpers: parameter-stability surfaces, walk-forward splits (foundation for ML later).

### 4.8 Visualization & reporting
- `viz/`: price+trades (entries/exits/SL/TP), equity curve, drawdown, monthly/yearly returns
  heatmaps, parameter heatmaps/comparison charts. Plotly, theme-consistent, downsampled for 1m spans
  so charts stay responsive.
- `reporting/`: assemble a per-experiment HTML/notebook report + comparison tables across strategies/
  assets.

## 5. Cross-cutting: UX, reproducibility, tooling
- `git init` + `.gitignore` (data/, caches, secrets) as step 0; delete the 24 `Copy` backups.
- `pyproject.toml` + pinned `requirements`; installable `quant` package.
- **Config files** (YAML/TOML) drive experiments; a config hash + code version stamps every result
  → reproducible.
- Structured **logging** + **tqdm** progress on all long tasks (already a strength — keep it).
- **Secrets** to env/`.env`; rotate the currently-committed Binance keys.
- **Tests**: indicator correctness, signal primitives, engine parity vs legacy, metric math.

## 6. Migration strategy (no big-bang)
1. Stand up `quant/` alongside the existing code; **do not** break current notebooks.
2. Port the data layer first (behavior-preserving), then indicators (with parity tests), then the
   engine (validated numerically against `run_simulation`), then signals, then optimize/analytics/viz.
3. Recreate 1–2 representative strategies (EMA ribbon, Supertrend) in the new framework and confirm
   identical stats on the same data before deprecating notebooks.
4. Once parity holds, migrate remaining strategies and retire the `Copy`/duplicate files.

## 7. Performance — target vs measured

Measured on the cached full year of PAXGUSDT 1m (**524,161 bars**), warm (post JIT-compile),
on this dev laptop:

| Operation | Legacy | New (`quant`) | Speedup |
|---|---|---|---|
| Position kernel only | — | **~16–25 ms** | — |
| Full single backtest (kernel + trades + stats + equity curve) | ~30 s (extrapolated) | **~65–150 ms** | ~200–400× |
| Per-backtest in a sweep (fast array-native stats) | ~2.5 s (43k-bar month) | **~65 ms serial / ~93 ms per combo across 6 workers** | — |
| Data load (pushdown, warm cache) | — | **~1.4 s** for the full year | — |

Numerical parity vs the legacy engine is **exact** (per-trade PnL diff `0.0`; see `tests/`).

Projected sweep throughput (full-year 1m): ~100k combos in ~2.5 h across 6 workers; shorter windows
or coarser timeframes scale down proportionally. **Path to faster** (for the "millions" goal):
(a) cache shared signal sub-components across combos, (b) optionally JIT-compile a strategy's whole
signal→sim→stats pipeline into one numba function, (c) loky/process parallelism over memmapped arrays
to escape the GIL that currently caps threaded signal-building. These are deferred, not blockers.

Memory: bounded by column-projected, date-sliced loads; sweeps reuse one shared feature frame.

## 8. What we explicitly keep (already good)
- Incremental download + durable partial checkpoints (`data.py`).
- Anti-lookahead HTF feature shifting (`timeframe_utils.py`).
- The rich exit/risk model (`TradeExitConfig`) — ported, not redesigned.
- Structured logging + tqdm progress.
- Parquet on-disk format (evolved to partitioned).

## 9. Decisions (settled)
1. **Engine:** ✅ custom numba kernel (exact parity, full control). vectorbt kept available as an
   optional cross-check path.
2. **Rebuild vs evolve:** ✅ greenfield `quant/` package with staged migration; legacy code untouched.
3. **First milestone:** ✅ end-to-end vertical slice for one EMA strategy on gold — done.
4. **Package name:** ✅ `quant/`.
