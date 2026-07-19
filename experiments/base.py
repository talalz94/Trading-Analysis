"""
Experiment / inference framework — find the BEST settings given an idea.

This layer sits ON TOP of the `quant` core and never modifies it. An `Experiment` declares a
search space over strategy parameters and/or execution-config fields, an objective metric, and a
plain-English `description`. Running it sweeps the space (reusing the core engine), ranks the
results, and writes a self-contained folder: `results.csv`, `best.json`, and a `report.md` that
records what the experiment was about, the winning settings, and a reproduce snippet.

Efficiency: trials are grouped by unique strategy parameters so indicators + signals are computed
once per strategy variant; execution-config variations (SL/TP/…) then reuse those signals and only
re-run the fast numba kernel.

Split of the search space:
  - `strategy_space`: fields on the Strategy dataclass (e.g. fast, slow, confirm_n, session, hours)
  - `cfg_space`:      fields on BacktestConfig (e.g. sl_mode, sl_value, tp_mode, tp_value, trail_*)
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Type

import numpy as np
import pandas as pd

from quant.analytics.fast import fast_stats, periods_per_year
from quant.engine import BacktestConfig
from quant.engine.run import invoke_kernel
from quant.logging_utils import get_logger, maybe_tqdm
from quant.optimize.grid import default_n_jobs, expand_grid
from quant.strategies.base import Strategy

_log = get_logger("experiments")


@dataclass
class Experiment:
    name: str
    description: str
    strategy_cls: Type[Strategy]
    base_cfg: BacktestConfig
    # data
    symbol: str = "PAXGUSDT"
    tf: str = "1m"
    start: str = "2025-06-01"
    end: Optional[str] = "2026-05-31"
    source: str = "binance"
    market: str = "spot"
    tz: str = "UTC"
    # search
    strategy_space: Dict[str, Sequence] = field(default_factory=dict)
    cfg_space: Dict[str, Sequence] = field(default_factory=dict)
    metric: str = "sharpe"
    direction: str = "max"                 # "max" | "min"
    min_trades: int = 10                   # drop degenerate results before ranking
    valid_fn: Optional[Callable[[dict], bool]] = None   # over merged {**strat, **cfg}

    # ---- run ----
    def run(self, df: Optional[pd.DataFrame] = None, *, n_jobs: Optional[int] = None,
            out_dir: str = "experiments/results", write: bool = True) -> pd.DataFrame:
        if df is None:
            from quant.data import get_ohlcv
            df = get_ohlcv(self.symbol, self.tf, start=self.start, end=self.end,
                           source=self.source, market=self.market, tz=self.tz, progress=False)

        strat_combos = expand_grid(self.strategy_space) if self.strategy_space else [{}]
        cfg_combos = expand_grid(self.cfg_space) if self.cfg_space else [{}]
        n_jobs = default_n_jobs() if n_jobs is None else max(1, n_jobs)
        _log.info("experiment '%s': %d strategy x %d cfg = %d trials | bars=%d",
                  self.name, len(strat_combos), len(cfg_combos),
                  len(strat_combos) * len(cfg_combos), len(df))

        def _eval_group(sp: dict) -> List[dict]:
            strat = self.strategy_cls(**sp)
            prepared = strat.prepare(df)
            sig = strat.signals(prepared, time_col="t")
            n = len(prepared)
            el, xl, es, xs = sig.as_u8(n)
            close = prepared["close"].to_numpy(np.float64)
            high = prepared["high"].to_numpy(np.float64)
            low = prepared["low"].to_numpy(np.float64)
            open_ = prepared["open"].to_numpy(np.float64)
            ppy = periods_per_year(prepared["t"])
            rows = []
            for cp in cfg_combos:
                merged = {**sp, **cp}
                if self.valid_fn is not None and not self.valid_fn(merged):
                    continue
                cfg = dataclasses.replace(self.base_cfg, **cp) if cp else self.base_cfg
                out = invoke_kernel(open_, high, low, close, el, xl, es, xs, cfg, df=prepared)
                stats = fast_stats(out[0], out[9], out[11], initial_cash=float(cfg.initial_cash),
                                  final_cash=out[13], ppy=ppy)
                rows.append({**merged, **stats})
            return rows

        if n_jobs == 1 or len(strat_combos) == 1:
            bar = maybe_tqdm(True, total=len(strat_combos), desc=self.name, unit="grp")
            all_rows: List[dict] = []
            for sp in strat_combos:
                all_rows.extend(_eval_group(sp))
                if bar:
                    bar.update(1)
            if bar:
                bar.close()
        else:
            from joblib import Parallel, delayed
            groups = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(_eval_group)(sp) for sp in strat_combos)
            all_rows = [r for g in groups for r in g]

        results = pd.DataFrame(all_rows)
        ranked = self._rank(results)
        if write:
            self._write(ranked, out_dir)
        return ranked

    # ---- ranking / output ----
    def _rank(self, results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return results
        r = results[results.get("num_trades", 0) >= self.min_trades].copy()
        if r.empty:
            r = results.copy()
        r = r.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.metric])
        r = r.sort_values(self.metric, ascending=(self.direction == "min")).reset_index(drop=True)
        return r

    def _param_keys(self) -> List[str]:
        return list(self.strategy_space) + list(self.cfg_space)

    def _write(self, ranked: pd.DataFrame, out_dir: str) -> str:
        d = Path(out_dir) / self.name
        d.mkdir(parents=True, exist_ok=True)
        ranked.to_csv(d / "results.csv", index=False)

        best = ranked.iloc[0].to_dict() if not ranked.empty else {}
        keys = self._param_keys()
        best_params = {k: _py(best.get(k)) for k in keys}
        (d / "best.json").write_text(json.dumps(
            {"metric": self.metric, "direction": self.direction, "best_value": _py(best.get(self.metric)),
             "best_params": best_params}, indent=2), encoding="utf-8")

        cols = keys + [c for c in ["num_trades", "total_return_pct", "win_rate_pct",
                                   "profit_factor", "sharpe", "max_drawdown_pct"] if c in ranked.columns]
        top = ranked[cols].head(15)
        strat_best = {k: _py(best.get(k)) for k in self.strategy_space}
        cfg_best = {k: _py(best.get(k)) for k in self.cfg_space}
        try:
            top_table = top.to_markdown(index=False)   # needs `tabulate`
        except Exception:
            top_table = "```\n" + top.to_string(index=False) + "\n```"
        report = f"""# Experiment: {self.name}

{self.description}

- **Asset / timeframe:** {self.symbol} {self.tf}  ({self.start} → {self.end})
- **Objective:** {'maximize' if self.direction == 'max' else 'minimize'} **{self.metric}**  (min {self.min_trades} trades)
- **Search space:** strategy={self.strategy_space or '—'} ; cfg={self.cfg_space or '—'}
- **Trials ranked:** {len(ranked)}

## Best settings
```
best {self.metric} = {_py(best.get(self.metric))}
strategy params: {strat_best}
config params:   {cfg_best}
```

### Reproduce
```python
from quant.strategies import {self.strategy_cls.__name__}
from quant.engine import BacktestConfig
strat = {self.strategy_cls.__name__}(**{strat_best})
cfg = <base_cfg with {cfg_best}>
res = strat.backtest(df, cfg)
```

## Top {len(top)} results
{top_table}

*Full ranked results in `results.csv`; best in `best.json`.*
"""
        (d / "report.md").write_text(report, encoding="utf-8")
        _log.info("experiment '%s' -> %s (best %s=%.4f)", self.name, d, self.metric,
                  best.get(self.metric, float("nan")))
        return str(d)


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
