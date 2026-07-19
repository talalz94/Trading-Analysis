# quant — Quantitative Research & Backtesting Platform

A fast, modular Python platform for **researching and backtesting rule-based trading
strategies** on historical data. Built for scale: test **thousands→millions of parameter
combinations over years of 1-minute data** without freezing your machine.

Initial focus asset is **gold** (via Binance PAX Gold / Tether Gold), with a source-agnostic
data layer designed to grow to other providers and asset classes (crypto, FX, equities, indices).

> This is a **research** tool, not a live-trading system.

---

## Table of contents
- [Overview](#overview) · [How it stands out](#how-it-stands-out) · [Folder structure](#folder-structure)
- [Installation](#installation) · [Quick start](#quick-start)
- [Running the project](#running-the-project) · [Data configuration](#data-configuration)
- [Indicators](#indicator-system) · [Strategies](#strategy-framework)
- [Stop loss](#stop-loss-configuration) · [Take profit](#take-profit-configuration)
- [Backtesting model](#backtesting-model) · [Optimization](#optimization) · [Visualization](#visualization)
- [Troubleshooting](#troubleshooting)

---

## Overview

The core idea: **separate signal generation from execution**.

```
data ─▶ indicators ─▶ signals (vectorized bool arrays) ─▶ numba engine ─▶ results ─▶ analytics / charts / reports
```

1. **Signals are computed with fully vectorized NumPy** (whole-series boolean arrays), once per
   parameter set.
2. **Execution runs in a numba-JIT compiled kernel** — one pass over the bars applying stop-loss,
   take-profit, trailing, partial exits, sizing, fees and slippage.

A single 1-year 1-minute backtest runs in **tens of milliseconds** (the compiled kernel alone is
~16–25 ms for ~500k bars), and a parameter sweep is embarrassingly parallel. The engine is
**validated at exact numerical parity** against the original reference simulator (per-trade PnL
difference `0.0`), locked in by golden regression tests.

### How it stands out
- **One representation for strategies.** The same vectorized signals drive both a single backtest
  and a million-combo sweep — no expressing each strategy twice.
- **Numba execution engine** with a full, realistic exit model (SL / laddered partial TPs /
  trailing / stop-movement / structure stops / risk-based sizing) — not a toy vectorized PnL.
- **Charts that stay responsive at millions of points.** Viewport-based resampling
  (peak-preserving) keeps 1-minute-over-years charts smooth in Jupyter — no waiting, no freezing.
- **Incremental, cached data.** Only missing candles are downloaded; interrupted downloads resume.
- **Plug-and-play strategies.** A new strategy is one dataclass.
- **Bounded parallelism** (`n_jobs = CPU−2` by default) so sweeps don't lock up your laptop.

### Folder structure
```
quant/
├── config.py          # settings + .env loading (secrets from env only)
├── logging_utils.py   # logger + progress-bar helpers
├── data/              # get_ohlcv, sources (binance), Parquet store (pushdown), MTF resample/align
├── indicators/        # vectorized compute: overlays, oscillators, candles(HA), volatility, structure
├── signals/           # numpy boolean primitives + time/session filters
├── engine/            # numba kernel + BacktestConfig/TakeProfit/Signals + run_backtest wrapper
├── strategies/        # Strategy base + EmaRibbon/RSI/MACD/HeikinAshi/Supertrend/MTF/KeyLevel
├── analytics/         # metrics (Sharpe/Sortino/Calmar/…) + attribution + fast sweep stats
├── viz/               # responsive charts (plotly-resampler) + static charts + heatmaps
├── optimize/          # grid sweep (bounded parallel) + optional Optuna search
├── reporting/         # performance summary + standalone HTML report
└── cli.py             # `quant fetch | backtest | report`
examples/  notebooks/  tests/  docs/  data/ (parquet cache, git-ignored)
```

**Key components:** `quant.data.get_ohlcv` (data), `quant.strategies.*` (strategies),
`quant.engine.BacktestConfig` + `run_backtest` (execution), `quant.optimize.run_grid` (sweeps),
`quant.viz.ResearchChart` (charts), `quant.reporting` (reports).

---

## Installation

**Requirements:** Python **3.11+** (developed on 3.13). Works on Windows, macOS, Linux.

```bash
# 1. clone
git clone <your-repo-url> quant && cd quant

# 2. create & activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. install
pip install -e .            # installs the package + dependencies from pyproject.toml
# or: pip install -r requirements.txt

# 4. (optional) extras
pip install optuna          # Bayesian parameter search
```

**Secrets (optional).** Historical data uses Binance's *public* API — no keys needed. If you later
add authenticated endpoints, copy `.env.example` to `.env` and set `BINANCE_API_KEY` /
`BINANCE_API_SECRET`. `.env` is git-ignored; never commit keys.

**Jupyter.** To use the notebook with the right interpreter, register your venv as a kernel:
```bash
pip install jupyterlab ipykernel
python -m ipykernel install --user --name quant --display-name "Python (quant)"
jupyter lab      # then open notebooks/01_research_template.ipynb and pick the "Python (quant)" kernel
```
> Charts need a live kernel + `ipywidgets` (installed automatically). Requires pandas ≥ 2.2.

### Verify the install
```bash
python -m pytest tests/     # ~26 tests: engine parity/golden, strategies, viz, analytics
python examples/gold_ema_demo.py
```

---

## Quick start

```python
from quant.data import get_ohlcv
from quant.strategies import EmaRibbon
from quant.engine import BacktestConfig
from quant.reporting import print_summary

# 1. data (downloads missing candles once, then cached)
df = get_ohlcv("PAXGUSDT", "1m", start="2025-06-01", end="2026-05-31", tz="UTC")

# 2. execution model
cfg = BacktestConfig(
    initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
    exit_enabled=True, sl_mode="entry_pct", sl_value=0.6,
    tp_mode="rr", tp_value=2.0, sizing_mode="risk_pct_equity", sizing_value=1.0,
)

# 3. strategy + backtest
res = EmaRibbon(fast=50, slow=200, confirm_n=5).backtest(df, cfg)

# 4. results
print_summary(res, df=df)      # headline stats + best session/weekday
res.stats                      # full dict; res.trades / res.equity_curve are DataFrames
```

---

## Running the project

### Command line
```bash
# download / update cached data
quant fetch PAXGUSDT 1m --start 2025-06-01 --end 2026-05-31

# run a backtest and print a summary
quant backtest --symbol PAXGUSDT --tf 1m --start 2025-06-01 --strategy ema_ribbon \
      --params '{"fast":50,"slow":200,"confirm_n":5}' --sl-value 0.6 --tp rr:2.0

# run and write a standalone HTML report
quant report   --symbol PAXGUSDT --tf 1m --start 2025-06-01 --strategy supertrend \
      --params '{"period":10,"multiplier":3.0}' --out reports/supertrend.html
```

### Notebook
Open `notebooks/01_research_template.ipynb` — a guided workflow: load → backtest → responsive
chart → attribution → parameter sweep → HTML report → build-your-own-strategy.

### Selecting a market / asset
Set the `symbol` (and `market`) in `get_ohlcv` / the CLI. Cached gold proxies:
`PAXGUSDT` (PAX Gold), `XAUTUSDT` (Tether Gold); `BTCUSDT` also cached. Any Binance spot symbol
works. Crypto/FX/stocks/indices become available as their data providers are added (see below).

### Generating reports & viewing charts
- **Reports:** `quant.reporting.to_html(res, "reports/run.html", df=df, price_df=df)` — offline,
  shareable (stats + price/trades + equity/drawdown + monthly + hour×weekday heatmaps).
- **Charts (Jupyter):** `quant.viz.ResearchChart(df).add_ema(50).add_trades(res.trades).show()`.

---

## Data configuration

`get_ohlcv(symbol, tf, start, end, *, source="binance", market="spot", tz="UTC", refresh=False)`

| What | How |
|---|---|
| **Symbol / ticker** | `symbol="PAXGUSDT"` (any Binance symbol) |
| **Timeframe** | `tf="1m"` (also `5m`, `15m`, `1h`, …) |
| **Date range** | `start`/`end` (ISO strings). Only candles in range are returned; only *missing* ones are downloaded. |
| **Provider** | `source="binance"` (implemented). The `DataSource` interface (`quant/data/base.py`) is built for more — see roadmap. |
| **Display timezone** | `tz="UTC"` → adds a tz-aware `t` column used by charts, the sim, and time filters. |
| **Force refresh** | `refresh=True` re-checks the source for new candles. |

**Where data is cached:** `data/{source}_{market}_{SYMBOL}_{tf}.parquet` (e.g.
`data/binance_spot_PAXGUSDT_1m.parquet`), git-ignored. In-progress downloads checkpoint to
`data/.partials/…` so an interrupted or rate-limited download **resumes from the last candle**.

**Incremental downloads.** The fetcher loads the cache, computes which ranges are missing (before,
after, and internal gaps), downloads only those, saves each API page immediately, then
consolidates. Reads use **column projection + date-range pushdown** (polars) so a run only loads
what it needs — low memory even with millions of rows cached.

**Adding a data provider (roadmap).** Implement the `DataSource` protocol
(`fetch(symbol, interval, start, end) -> OHLCV DataFrame`) in `quant/data/`, register it in
`quant/data/binance.py`'s source registry, and select it via `source="..."`. Planned providers:
real spot gold, OANDA (FX/metals), metals APIs, CSV/TradingView import.

---

## Indicator system

Indicators are **vectorized compute functions** (recursive ones JIT-compiled) that add named
columns. Rendering is separate (in `quant.viz`).

```python
from quant.indicators import add_emas, add_rsi, add_macd, add_heikin_ashi

df = add_emas(df, [20, 50, 200])   # -> ema_20, ema_50, ema_200
df = add_rsi(df, 14)               # -> rsi_14
df = add_macd(df, 12, 26, 9)       # -> macd, macd_signal, macd_hist
df = add_heikin_ashi(df)           # -> ha_open/high/low/close, ha_green
```

| Function | Columns added |
|---|---|
| `add_emas(df, [..])` / `add_smas` | `ema_{p}` / `sma_{p}` |
| `add_rsi(df, 14)` | `rsi_14` |
| `add_macd(df, 12, 26, 9)` | `macd`, `macd_signal`, `macd_hist` |
| `add_stochastic(df, 14, 3, 3)` | `stoch_k`, `stoch_d` |
| `add_atr(df, 14)` / `add_supertrend(df, 10, 3.0)` | `atr_14` / `st`, `st_dir`, `st_up` |
| `add_heikin_ashi(df)` | `ha_*`, `ha_green` |
| `add_swings(df, 10, 10)` | `swing_high/low`, `swing_last_high/low` (lookahead-safe stop refs) |
| `add_pivot_points(df, "1D")` | `piv_pp/r1/s1/r2/s2` |

**Modify parameters:** change the call arguments (periods, multipliers, lookbacks).

**Create + register a new indicator:**
```python
# quant/indicators/oscillators.py  (or a new module)
def add_willr(df, period=14, prefix="willr"):
    hh = df["high"].rolling(period).max(); ll = df["low"].rolling(period).min()
    out = df.copy()
    out[f"{prefix}_{period}"] = -100 * (hh - df["close"]) / (hh - ll)
    return out
```
Then export it in `quant/indicators/__init__.py`. Keep it vectorized; if a recursive loop is
unavoidable, write the kernel with `@njit` over numpy arrays (see `volatility.py`, `candles.py`).

**Multi-timeframe indicators** (lookahead-safe): `quant.data.build_mtf`
```python
from quant.data import build_mtf
from quant.indicators import add_emas
mtf = build_mtf(df, {"5min":  lambda d: add_emas(d, [50, 200]),
                     "15min": lambda d: add_emas(d, [50])})
# -> columns like 5min__ema_50, 15min__ema_50 aligned onto the 1m grid (HTF shifted +1 bar)
```

---

## Strategy framework

A strategy is **one dataclass** with `prepare()` (add indicator columns) and `build_signals()`
(return entry/exit boolean arrays). Every field is a tunable parameter — sweep them with no code
changes. The base class adds **free time-filtering** (`session` / `hours` / `weekdays` /
`avoid_weekends`) applied to entries.

```python
from dataclasses import dataclass
from quant.strategies.base import Strategy
from quant.indicators import add_rsi
from quant.engine import Signals
from quant import signals as S

@dataclass
class MyRsi(Strategy):
    name: str = "my_rsi"
    period: int = 14
    def prepare(self, df):
        return add_rsi(df, self.period)
    def build_signals(self, df):
        r = f"rsi_{self.period}"
        return Signals(entry_long=S.cross_up(df, r, 30), exit_long=S.cross_down(df, r, 70))

MyRsi(period=14).backtest(df, cfg).stats
```

Built-in strategies (in `quant.strategies`): `EmaRibbon`, `RsiReversal`, `MacdTrend`,
`HeikinAshiTrend`, `SupertrendFlip`, `MtfTrend`, `KeyLevelBounce`.

### Signal primitives (`quant.signals`)
`cross_up/cross_down`, `above/below`, `above_all/below_all`, `last_all_above/below`,
`prev_all_above/below`, `consecutive_green/red`, `rising/falling`, `refs_ordered`,
`all_of/any_of/none_of`, and time filters `in_session/hour_between/between_times/weekday_in/not_weekend`.

### Examples

**EMA strategy** — *buy when price crosses EMA(20) and stays above EMA(50) for 3 candles; exit on
the opposite EMA cross:*
```python
from quant.strategies import EmaRibbon
EmaRibbon(fast=20, slow=50, confirm_n=3)      # exit = price crosses back below EMA(20)
```

**RSI strategy** — *long after RSI < 30 for 3 bars, short after RSI > 70 for 5 bars:*
```python
from quant.strategies import RsiReversal
RsiReversal(period=14, oversold=30, long_consec=3, overbought=70, short_consec=5,
            allow_short_signals=True)
```
Multi-timeframe RSI confirmation: precompute a higher-TF RSI with `build_mtf` and add it to the
entry with `S.all_of(...)` in a custom `build_signals`.

**Heikin Ashi strategy** — *enter after N consecutive bullish HA candles; exit on colour reversal:*
```python
from quant.strategies import HeikinAshiTrend
HeikinAshiTrend(n_consec=3)
```

**Multi-timeframe strategy** — *1-minute entry trigger, 5-minute trend filter, 15-minute momentum:*
```python
from quant.strategies import MtfTrend
MtfTrend(fast_1m=50, trend_5m_fast=50, trend_5m_slow=200, mom_15m=50)
```

**Time-based strategy** — *any strategy, restricted by session / hours / weekday:*
```python
EmaRibbon(fast=50, slow=200, session="london")          # London session only
EmaRibbon(fast=50, slow=200, hours=(13, 20))            # 13:00–20:00 (tz of the `t` column)
EmaRibbon(fast=50, slow=200, weekdays=(0,1,2,3,4), avoid_weekends=True)
```
*(News-blackout windows are a planned extension.)*

**Key-level strategy** — *buy bounces off swing-low support, take profit into resistance, stop
below the swing:*
```python
from quant.strategies import KeyLevelBounce
strat = KeyLevelBounce(left=10, right=10, near_pct=0.15)
cfg = BacktestConfig(exit_enabled=True, sl_mode="ref_col", sl_ref_long_col="swing_last_low",
                     sl_buffer_pct=0.05, tp_mode="rr", tp_value=2.0,
                     sizing_mode="risk_pct_equity", sizing_value=1.0)
```
Uses swing highs/lows (`add_swings`) and pivot points (`add_pivot_points`) as levels.

---

## Stop-loss configuration

Set on `BacktestConfig` (with `exit_enabled=True`):

| Type | Config |
|---|---|
| **Fixed percentage** | `sl_mode="entry_pct", sl_value=0.6` (0.6% from entry) |
| **Fixed points** | `sl_mode="price_abs", sl_value=500` ($500 from entry) |
| **Swing low / high (structure)** | `sl_mode="ref_col", sl_ref_long_col="swing_last_low", sl_ref_short_col="swing_last_high", sl_buffer_pct=0.05` — with `sl_max_ref_risk_pct` cap and `sl_fallback_mode/value` when the level is unusable |
| **Local peak / trough** | same `ref_col` mechanism against any level column you compute (pivots, prior-day low, …) |
| **Trailing stop** | `trail_mode="pct", trail_value=0.5` (or `"price_abs"`) — ratchets on the high-water mark |
| **Move to breakeven / new level after a TP** | per take-profit: `move_stop_mode="breakeven"` (or `entry_pct`/`price_abs`) |

**Rule-based exits** ("exit after N red candles", "exit on EMA cross") are expressed as the
strategy's **exit signals** (`exit_long`/`exit_short`), not price stops — e.g.
`exit_long = S.consecutive_red(df, 3)` or `S.cross_down(df, "close", "ema_50")`.

---

## Take-profit configuration

| Type | Config |
|---|---|
| **Risk/Reward** | `tp_mode="rr", tp_value=2.0` (2R from the original stop distance) |
| **Fixed percentage / points** | `tp_mode="entry_pct", tp_value=1.5` / `tp_mode="price_abs", tp_value=800` |
| **Partial / laddered exits** | `take_profits=(TakeProfit("rr",1.0,close_pct=50,move_stop_mode="breakeven"), TakeProfit("rr",2.5,close_pct=100))` — close 50% at 1R (move stop to breakeven), the rest at 2.5R |
| **Trailing stop** | `trail_mode="pct", trail_value=0.5` (rides winners) |
| **EMA exit / opposite signal / candle-count** | strategy exit signals (`exit_long`), e.g. `S.cross_down(df,"close","ema_50")`, `S.consecutive_red(df,3)` |

```python
from quant.engine import TakeProfit
cfg = BacktestConfig(exit_enabled=True, sl_mode="entry_pct", sl_value=0.6,
                     take_profits=(TakeProfit("rr", 1.0, close_pct=50, move_stop_mode="breakeven"),
                                   TakeProfit("rr", 2.5, close_pct=100)),
                     sizing_mode="risk_pct_equity", sizing_value=1.0)
```
> TP levels are fixed at entry (rr uses the original stop distance) — predictable and intended.

---

## Backtesting model

`BacktestConfig` fields:

| Concept | Field(s) | Notes |
|---|---|---|
| **Account balance** | `initial_cash` | starting equity |
| **Risk per trade** | `sizing_mode="risk_pct_equity", sizing_value=1.0` | risk 1% of equity per trade (needs a stop). Also `"risk_amount"` (fixed $) or `"cash"` (fixed notional) |
| **Compounding** | (automatic) | risk-based sizing uses *current* equity, so wins compound |
| **Commission** | `fee_bps` | per-side, in basis points (8 = 0.08%) |
| **Slippage / spread** | `slippage_bps` | applied to every fill; also your spread proxy |
| **Leverage** | `allow_leverage=True`, `max_notional_pct` | notional cap as % of equity (cap model, not full margin/liquidation) |
| **Multiple open positions** | `max_open_trades` | >1 allows concurrent trades |
| **Long / short** | `allow_short=True` + strategy short signals | |
| **Intrabar priority** | `intrabar_priority="stop_first"` | when a bar hits both SL and TP |
| **Lot size** | via sizing | crypto uses fractional quantity; position size comes from the sizing mode |

The result (`SimResult`) has `.trades` (per-trade DataFrame), `.equity_curve`
(per-bar equity + drawdown), and `.stats` — total return, win/loss, profit factor, expectancy,
**Sharpe / Sortino / Calmar**, max drawdown, recovery factor, average trade, exposure, fees, and
per-side breakdowns.

**Attribution** (`quant.analytics`): `by_hour`, `by_weekday`, `by_month`, `by_session`,
`by_regime` (trend × volatility), `monthly_returns` — find *when* and *in what conditions* a
strategy works.

---

## Optimization

Sweep any strategy's parameters. Indicator columns are computed once and shared; each backtest is
milliseconds; runs in bounded parallel (`n_jobs = CPU−2` by default).

```python
from quant.optimize import run_grid
grid = {"fast": [20, 30, 50, 75, 100], "slow": [150, 200, 300], "confirm_n": [1, 3, 5, 10]}
results = run_grid(df, EmaRibbon, grid, cfg, valid_fn=lambda p: p["fast"] < p["slow"],
                   keep_stats=["num_trades", "total_return_pct", "sharpe", "max_drawdown_pct"])
results.sort_values("sharpe", ascending=False).head()
```
Visualize with `quant.viz.sweep_heatmap(results, x="fast", y="slow", z="sharpe")`.

**Bayesian search** (optional, needs `optuna`):
```python
from quant.optimize import optuna_search
study = optuna_search(df, EmaRibbon,
                      {"fast": (10, 100), "slow": (120, 400), "confirm_n": (1, 15)},
                      cfg, metric="sharpe", n_trials=200, valid_fn=lambda p: p["fast"] < p["slow"])
study.best_params
```

---

## Experiments / inference

Beyond raw sweeps, the **`experiments/` layer** answers *"what are the best settings for this
idea?"* — it composes the core APIs and **never modifies `quant/`**. Each experiment declares a
search space + objective + a plain-English description, ranks the trials, and writes
`experiments/results/<name>/` (`results.csv`, `best.json`, and a `report.md` documenting what it
tested, the winning settings, and a reproduce snippet).

```bash
python -m experiments.ema_mtf          # best multi-timeframe EMA combination
python -m experiments.session_timing   # best market session / time of day
python -m experiments.exit_design      # best stop-loss % and take-profit R
```
```python
from experiments.exit_design import build
ranked = build().run()          # ranked DataFrame + writes the results folder
```
`strategy_space` varies strategy fields (EMA periods, session, hours…); `cfg_space` varies
execution-config fields (stop-loss, take-profit, trailing, leverage…). Write your own in a few
lines — see [`experiments/README.md`](experiments/README.md).

## Visualization

**Responsive research chart (Jupyter)** — smooth at millions of candles via viewport resampling
(peak-preserving); pan/zoom reloads only the visible window; toggle layers via the legend:
```python
from quant.viz import ResearchChart, equity_chart
ch = ResearchChart(df, candles=True)
ch.add_ema(50); ch.add_ema(200)
ch.add_trades(res.trades)          # entry/exit markers (thousands stay responsive)
ch.show()
equity_chart(res.equity_curve)     # responsive equity curve
```

**Analytical charts** (static; also used in HTML reports):
- `equity_and_drawdown(res.equity_curve)` — equity + drawdown
- `price_and_trades(df, res.trades)` — price with entry/exit markers
- `monthly_returns_heatmap(analytics.monthly_returns(res.equity_curve))`
- `hour_weekday_heatmap(res.trades, metric="total_pnl")`
- `sweep_heatmap(results, "fast", "slow", "sharpe")`

**Performance summary / report:**
```python
from quant.reporting import print_summary, to_html
print_summary(res, df=df)
to_html(res, "reports/run.html", df=df, price_df=df, title="PAXG EMA")   # offline HTML
```

---

## Troubleshooting

- **`ModuleNotFoundError: quant`** — run from the repo root, `pip install -e .`, or add the repo to
  `sys.path` (the notebook does this automatically).
- **Charts don't render / no interactivity in Jupyter** — the responsive chart needs a *live
  kernel* + `ipywidgets`; static HTML export keeps only the initial view. In VS Code, ensure the
  Jupyter extension is installed. As a fallback, use the static `price_and_trades` /
  `equity_and_drawdown`.
- **`Invalid frequency: ME` / odd pandas errors** — your Jupyter kernel is an old Python/pandas.
  Use Python ≥ 3.11 with pandas ≥ 2.2 (register your venv as a kernel — see [Installation](#installation)).
- **First backtest is slow (~1–2 s)** — that's numba compiling the kernel once; subsequent runs are
  ~milliseconds. Compiled code is cached on disk between sessions.
- **Slow / heavy sweeps** — lower `n_jobs`, narrow the grid, or backtest a shorter window /
  coarser timeframe first. `n_jobs` defaults to `CPU−2` to keep the machine usable.
- **Binance rate limits / network drops** — downloads checkpoint and resume automatically; just
  re-run `get_ohlcv`.

---

## Documentation
- [`CLAUDE.md`](CLAUDE.md) — architecture orientation for contributors / AI sessions
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — design + roadmap
- [`docs/INDICATOR_GUIDE.md`](docs/INDICATOR_GUIDE.md) — indicator & signal reference
- [`notebooks/01_research_template.ipynb`](notebooks/01_research_template.ipynb) — guided workflow

## License
Add your license of choice (e.g. MIT) as `LICENSE`.
