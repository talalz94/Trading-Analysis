# Experiments / inference layer

Search for the **best settings given an idea** — without touching the core engine.

> *"Find the best EMA combination across timeframes."* · *"What time of day does this work best?"*
> *"What's the best stop-loss type and value?"*

Each experiment declares a **search space** (over strategy params and/or execution-config fields),
an **objective metric**, and a plain-English **description**. Running it sweeps the space (reusing
the core `quant` engine — nothing in `quant/` is modified), ranks the results, and writes a
self-contained folder under `experiments/results/<name>/`:

- `results.csv` — every trial, ranked
- `best.json` — the winning parameters + metric
- `report.md` — what the experiment was about, the best settings, a reproduce snippet, and the top table

## Run the included experiments
```bash
python -m experiments.ema_mtf          # best multi-timeframe EMA combination
python -m experiments.session_timing   # best market session / time of day
python -m experiments.exit_design      # best stop-loss % and take-profit R
python -m experiments.sr_zone_search   # S/R zone settings per timeframe (~8 min); the
                                       #   presets it picks ARE quant.indicators.SR_TUNED --
                                       #   a test fails if the two drift apart
python -m experiments.sr_zone_search --picks-only   # re-select from the saved results
python -m experiments.level_age_study   # do OLD S/R zones / FVGs hold better? (~10 s) -- no
python -m experiments.exit_rules_search # move-the-target / FVG-exit on nb21, 19 quarters (~3 min) -- no
python -m experiments.regime_study      # what differs when nb21's strategy wins vs loses (~2 min) -- nothing but costs
python -m experiments.instrument_scan   # nb21's strategy on gold/oil/FX vs a metronome (~8 min) -- USDJPY only, unreplicated
python -m experiments.nq_open_reversion # JJ Simon's NY-open reversion on Nasdaq 100 1m (~15 min download) -- no bias, -0.08R
python -m experiments.nq_news_reversion # his A+ news-day reversion to the 8:29 price (needs quant.news) -- -0.09R, negative before costs
# any strategy file and its COMPARE arms, quarter by quarter: notebooks/evaluate.ipynb, or
#   from experiments.evaluate import evaluate; evaluate(evaluate_config()).arms()
```
Or from Python / a notebook:
```python
from experiments.exit_design import build
ranked = build().run()          # returns a ranked DataFrame; also writes results/exit_design/
ranked.head()
```

## Write your own experiment
Create `experiments/my_idea.py`:
```python
from quant.engine import BacktestConfig
from quant.strategies import EmaRibbon
from experiments.base import Experiment

def build():
    return Experiment(
        name="my_idea",
        description="One sentence on what you're testing and why.",
        strategy_cls=EmaRibbon,
        base_cfg=BacktestConfig(initial_cash=10_000, fee_bps=8, slippage_bps=1.5,
                                exit_enabled=True, sl_mode="entry_pct",
                                sizing_mode="risk_pct_equity", sizing_value=1.0),
        symbol="PAXGUSDT", tf="1m", start="2025-06-01", end="2026-05-31",
        strategy_space={"fast": [20, 50, 100], "slow": [150, 200, 300], "confirm_n": [1, 3, 5]},
        cfg_space={"sl_value": [0.4, 0.6, 1.0], "tp_value": [1.5, 2.0, 3.0], "tp_mode": ["rr"]},
        metric="sharpe", direction="max", min_trades=20,
        valid_fn=lambda p: p["fast"] < p["slow"],
    )

if __name__ == "__main__":
    build().run()
```

- **`strategy_space`** varies fields on the strategy dataclass (e.g. `fast`, `slow`, `confirm_n`,
  `session`, `hours`).
- **`cfg_space`** varies `BacktestConfig` fields (e.g. `sl_mode`, `sl_value`, `tp_mode`, `tp_value`,
  `trail_mode`, `leverage`).
- Trials are grouped by strategy params so indicators/signals compute once per strategy variant;
  config variations reuse the signals and only re-run the numba kernel (fast). Runs in bounded
  parallel (`n_jobs = CPU−2`).

## Design principle
The core (`quant/`) stays generic and stable. Experiments are throwaway/iterative research
artifacts that *compose* the core APIs. Keep experiment-specific logic here, never in `quant/`.
