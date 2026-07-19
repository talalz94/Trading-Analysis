# Experiment Design & Execution Guide

How to go from a trading **idea** to the **best settings** — and to trust the result. The
`experiments/` layer runs the search; this guide is the *method*.

> Golden rule: experiments compose the `quant` core, they never modify it. Keep experiment code
> in `experiments/`, keep the engine generic.

---

## 1. The workflow

```
idea → search space → objective → run → interpret → pick best → validate (out-of-sample) → doc
```

1. **State the idea in one sentence** (this becomes the experiment's `description`).
   *"Which EMA periods across 1m/5m/15m maximize risk-adjusted return?"*
2. **Turn it into a search space:**
   - `strategy_space` — fields on the strategy dataclass (EMA periods, `confirm_n`, `session`, `hours`).
   - `cfg_space` — `BacktestConfig` fields (`sl_value`, `tp_value`, `trail_mode`, `leverage`, …).
3. **Choose an objective** (`metric`, `direction`) and a **sanity filter** (`min_trades`) so a
   2-trade fluke can't "win".
4. **Run** (`Experiment.run()`), which sweeps the space and writes `results.csv` + `best.json` +
   `report.md`.
5. **Interpret** the ranked table (below).
6. **Validate** the winner out-of-sample before believing it.

## 2. Template (copy, rename, edit)

```python
# experiments/my_idea.py
from quant.engine import BacktestConfig
from quant.strategies import EmaRibbon
from experiments.base import Experiment

DESCRIPTION = "One sentence: what am I testing and why?"

def build() -> Experiment:
    base_cfg = BacktestConfig(
        initial_cash=10_000, fee_bps=8, slippage_bps=1.5, spread=0.20,
        exit_enabled=True, sl_mode="entry_pct",
        sizing_mode="risk_pct_equity", sizing_value=1.0,
    )
    return Experiment(
        name="my_idea",
        description=DESCRIPTION,
        strategy_cls=EmaRibbon,
        base_cfg=base_cfg,
        symbol="XAUUSD", tf="1m", start="2022-01-01", end="2024-12-31",  # source below
        # source="dukascopy",    # (Experiment loads via get_ohlcv; set on the call if not binance)
        strategy_space={"fast": [20, 50, 100], "slow": [150, 200, 300], "confirm_n": [1, 3, 5]},
        cfg_space={"sl_value": [0.4, 0.6, 1.0], "tp_value": [1.5, 2.0, 3.0], "tp_mode": ["rr"]},
        metric="sharpe", direction="max", min_trades=30,
        valid_fn=lambda p: p["fast"] < p["slow"],
    )

if __name__ == "__main__":
    build().run()          # writes experiments/results/my_idea/
```
Run it: `python -m experiments.my_idea`  ·  or in a notebook: `from experiments.my_idea import build; ranked = build().run()`.

## 3. Reading the results

`report.md` shows the top rows; `results.csv` has everything. For each combo you get
`num_trades, total_return_pct, win_rate_pct, profit_factor, sharpe, max_drawdown_pct`.

- **Rank by the objective**, but **look at the neighbourhood, not just row 1.** A good setting sits
  on a *plateau* of similar-scoring neighbours (robust); a lone spike surrounded by poor scores is
  probably **overfit** (curve-fit to noise).
- **Check `num_trades`.** High metric + few trades = not significant. That's what `min_trades` guards.
- **Cross-check metrics.** A high Sharpe with a tiny profit factor or huge drawdown is fragile.
- Use `quant.viz.sweep_heatmap(results, x="fast", y="slow", z="sharpe")` to *see* the plateau.

## 4. Picking — and trusting — the winner

1. Take the best **robust** combo (on a plateau), not necessarily the top row.
2. **Validate out-of-sample:** re-run that single combo on a *different* period than you searched
   (e.g. search 2022–2023, validate 2024). If it collapses, it was overfit.
3. Prefer **fewer knobs**. Every parameter you tune is a degree of freedom to overfit; a simple
   setting that survives out-of-sample beats a 6-parameter optimum that doesn't.
4. Sanity-check costs: re-run the winner with realistic `spread`/`commission_per_lot` — an edge
   that only exists at zero cost isn't an edge.

## 5. Scaling to many combinations

- Trials are **grouped by strategy params** so indicators/signals compute once per strategy variant;
  `cfg_space` variants reuse the signals and only re-run the numba kernel (very fast). Put things
  that don't change indicators (stop-loss, take-profit, leverage) in `cfg_space`.
- Runs in **bounded parallel** (`n_jobs = CPU−2`). For very large CPU-bound sweeps, the underlying
  `run_grid` also accepts `backend="loky"` (separate processes).
- For a first pass, **search a shorter window or a coarser timeframe**, find the plateau, then
  confirm on full 1-minute history. This is far faster than brute-forcing everything at 1m.
- For big continuous spaces, prefer **Optuna** (`quant.optimize.optuna_search`) over a full grid.

## 6. Anti-overfitting checklist
- [ ] Winner sits on a plateau (neighbours score similarly).
- [ ] Enough trades to be meaningful (`min_trades`).
- [ ] Survives an out-of-sample period.
- [ ] Survives realistic spread/commission.
- [ ] Not reliant on many finely-tuned knobs.

See [`../experiments/README.md`](../experiments/README.md) for the API and the three shipped
examples (`ema_mtf`, `session_timing`, `exit_design`).
