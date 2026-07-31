"""
Versioned, composable trading *scenarios*.

A scenario is a hand-described setup ("support-break short below a down-sloping 200-EMA") captured as
**versioned Python logic** plus a **named, versioned Stop method and Take-Profit method**, all bundled
into a **Preset** (a small JSON file) that reproduces the exact backtest on any symbol / timeframe.

    run = Scenario(entry logic) × StopMethod × TPMethod × params

Each of the three pieces is independently named + versioned so we can tweak one, keep the others, and
tie new variants to a strategy without rewriting anything. Everything reuses the existing engine:
`Scenario` builds an entry/exit/reset `Cond` (from `quant.strategies.conditions`), the stop/TP methods
contribute `BacktestConfig` fields (and any per-bar reference columns), and we hand the composed
`(df, Signals, BacktestConfig)` to `run_backtest`.

Storage model (per the design):
  * LOGIC  — Python templates here + in `support_break.py`, `stops.py`, `takeprofits.py` (versioned).
  * CONFIG — tuned `Preset`s saved as JSON under `scenarios/presets/*.json` (git-tracked, reloadable).
"""
from __future__ import annotations

import dataclasses
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..engine import BacktestConfig, SimResult, run_backtest
from ..strategies.builder import Rule, RuleStrategy
from ..strategies.conditions import Cond


# ---------------------------------------------------------------------------
# The three composable, versioned pieces
# ---------------------------------------------------------------------------
class Scenario(ABC):
    """The entry logic of a setup, as versioned Python. Subclass per version and register it.

    `entry(p)` returns the entry `Cond` (short setups here); `reset(p)` returns the episode-boundary
    `Cond` that re-arms the one-entry-per-setup logic (e.g. price closing back above the EMA line);
    `prepare(df, p)` adds the indicator columns the conditions read.
    """
    name: str = "scenario"
    version: str = "v1.0"
    description: str = ""
    side: str = "short"
    default_params: Dict = {}

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def prepare(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        return df

    @abstractmethod
    def entry(self, p: dict) -> Cond:
        ...

    def reset(self, p: dict) -> Optional[Cond]:
        return None

    def __repr__(self) -> str:
        return f"Scenario({self.key})"


class StopMethod(ABC):
    """A versioned stop-loss recipe: adds any reference column (`prepare`) and sets the
    `BacktestConfig` stop fields (`apply`)."""
    name: str = "stop"
    version: str = "v1"
    description: str = ""
    default_params: Dict = {}

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def prepare(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        return df

    @abstractmethod
    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        ...

    def __repr__(self) -> str:
        return f"StopMethod({self.key})"


class TPMethod(ABC):
    """A versioned take-profit / exit recipe. It may realise its target as `BacktestConfig` fields
    (`apply` — e.g. rr / trailing / laddered partials) and/or as an exit **signal** (`exit` returns a
    `Cond` — e.g. N green candles, price back above the EMA, price into the next support zone)."""
    name: str = "tp"
    version: str = "v1"
    description: str = ""
    default_params: Dict = {}

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def prepare(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        return df

    def apply(self, cfg: BacktestConfig, p: dict) -> BacktestConfig:
        return cfg

    def exit(self, p: dict) -> Optional[Cond]:
        return None

    def __repr__(self) -> str:
        return f"TPMethod({self.key})"


class SecondaryMethod(ABC):
    """A versioned SECONDARY rule that only plays out *after the main logic has opened a trade*.

    Two conditions, both evaluated per bar while a trade is open:
      `arm(p)`  — once true on a bar AFTER this trade's entry bar, the trade is ARMED, permanently;
      `exit(p)` — once armed, this closes the trade — but ONLY while it is in profit.

    The arming is per-trade and path-dependent, so the engine tracks it inside the kernel (`o_armed`,
    `alt_exit_enabled`); a plain signal array cannot express "since THIS trade opened". A trade can
    neither arm on its own entry bar nor inherit arming from before it existed. Trades closed this
    way carry the `alt_exit` close reason. Set `sec_replace_tp` to also drop the take-profit once
    armed. Leaving the secondary unset keeps the original, byte-identical engine path.
    """
    name: str = "secondary"
    version: str = "v1"
    description: str = ""
    default_params: Dict = {}

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    def prepare(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        return df

    @abstractmethod
    def arm(self, p: dict) -> Cond:
        ...

    @abstractmethod
    def exit(self, p: dict) -> Cond:
        ...

    def __repr__(self) -> str:
        return f"SecondaryMethod({self.key})"


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------
SCENARIOS: Dict[str, Scenario] = {}
STOP_METHODS: Dict[str, StopMethod] = {}
TP_METHODS: Dict[str, TPMethod] = {}
SECONDARIES: Dict[str, SecondaryMethod] = {}


def register_scenario(obj: Scenario) -> Scenario:
    SCENARIOS[obj.key] = obj
    return obj


def register_stop(obj: StopMethod) -> StopMethod:
    STOP_METHODS[obj.key] = obj
    return obj


def register_tp(obj: TPMethod) -> TPMethod:
    TP_METHODS[obj.key] = obj
    return obj


def register_secondary(obj: SecondaryMethod) -> SecondaryMethod:
    SECONDARIES[obj.key] = obj
    return obj


def _resolve(reg: dict, key: str, kind: str):
    if key not in reg:
        raise KeyError(f"unknown {kind} {key!r}. Available: {sorted(reg)}")
    return reg[key]


def resolve_scenario(key: str) -> Scenario:
    return _resolve(SCENARIOS, key, "scenario")


def resolve_stop(key: str) -> StopMethod:
    return _resolve(STOP_METHODS, key, "stop method")


def resolve_tp(key: str) -> TPMethod:
    return _resolve(TP_METHODS, key, "tp method")


def resolve_secondary(key: str) -> SecondaryMethod:
    return _resolve(SECONDARIES, key, "secondary method")


def resolve_secondaries(spec) -> list:
    """Accept None, one key, or a list of keys. Secondary rules do NOT conflict — each one only adds
    another way for an open trade to become ARMED, and they share the same in-profit-gated exit — so
    several can run together (e.g. bail when the trend goes shaky AND ride a winner past its TP)."""
    if spec is None:
        return []
    if isinstance(spec, str):
        spec = [spec]
    return [resolve_secondary(k) for k in spec]


# ---------------------------------------------------------------------------
# Preset: the reloadable, versioned config
# ---------------------------------------------------------------------------
@dataclass
class Preset:
    """A named, versioned scenario configuration. `to_json`/`from_json` round-trip to
    `scenarios/presets/*.json`, so a tuned setup reproduces on any symbol / timeframe."""
    name: str
    scenario: str                          # "support_break_short@v1.0"
    stop: str                              # "swing_high@v1"
    tp: str                                # "rr@v1"
    secondary: Optional[object] = None     # "shaky_trend_bail@v1", or a LIST of keys (they combine)
    params: Dict = field(default_factory=dict)
    cfg: Dict = field(default_factory=dict)     # BacktestConfig overrides (spread, sizing, ...)
    data: Dict = field(default_factory=dict)    # get_ohlcv kwargs (symbol, tf, start, end, source, ...)
    description: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Preset":
        return cls(**{k: d[k] for k in d if k in {f.name for f in dataclasses.fields(cls)}})

    @classmethod
    def from_json(cls, path) -> "Preset":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Composition + run
# ---------------------------------------------------------------------------
@dataclass
class ScenarioRun:
    """Everything a run produced — the result plus the prepared frame + pieces, so the notebook can
    both inspect the stats and draw the fully-annotated scenario chart."""
    result: SimResult
    df: pd.DataFrame
    cfg: BacktestConfig
    params: dict
    scenario: Scenario
    stop: StopMethod
    tp: TPMethod
    secondary: Optional[object] = None      # list of SecondaryMethod (or None)
    preset: Optional[Preset] = None

    @property
    def trades(self):
        return self.result.trades

    @property
    def stats(self):
        return self.result.stats


def merged_params(scenario: Scenario, stop: StopMethod, tp: TPMethod, overrides: dict,
                  secondary=None) -> dict:
    """Piece defaults, then the preset/user overrides on top (flat dict; the pieces use distinct
    field names)."""
    p: dict = {}
    p.update(scenario.default_params)
    p.update(stop.default_params)
    p.update(tp.default_params)
    for sec in (secondary if isinstance(secondary, (list, tuple)) else
                ([secondary] if secondary is not None else [])):
        p.update(sec.default_params)
    p.update(overrides or {})
    return p


def _prepare_mtf(df: pd.DataFrame, spec) -> pd.DataFrame:
    """Add higher-timeframe MA columns (`{tf}__ema_{period}`) so `below_mtf`/`above_mtf` can be used.

    Routed through `build_mtf`, which shifts each HTF feature one HTF bar before the as-of merge, so
    a value on a base-timeframe bar was already CLOSED at that moment. Accepts the same shapes as the
    chart: ('5min', 200) | ('5min', [100, 200]) | [('5min',100), ('15min',200)].
    """
    if not spec:
        return df
    from ..data import build_mtf
    from ..indicators.overlays import add_emas
    pairs = []
    if isinstance(spec, (tuple, list)) and len(spec) == 2 and isinstance(spec[0], str):
        tf, per = spec
        pairs = [(tf, int(x)) for x in (per if isinstance(per, (list, tuple)) else [per])]
    else:
        for item in spec:
            tf, per = item
            pairs += [(tf, int(x)) for x in (per if isinstance(per, (list, tuple)) else [per])]
    by_tf = {}
    for tf, per in pairs:
        by_tf.setdefault(tf, []).append(per)
    return build_mtf(df, {tf: (lambda ps: (lambda z: add_emas(z, ps)))(sorted(set(ps)))
                          for tf, ps in by_tf.items()})


def compose(df: pd.DataFrame, scenario: Scenario, stop: StopMethod, tp: TPMethod,
            params: dict, base_cfg: BacktestConfig, secondary=None, entry_filter=None):
    """Compose the pieces into (prepared_df, Signals, BacktestConfig).

    prepare order matters: scenario indicators first, then stop/TP/secondary reference columns, then
    the entry/exit conditions are compiled on the fully-prepared frame.
    """
    p = merged_params(scenario, stop, tp, params, secondary)
    p["_side"] = scenario.side          # stops / TPs / secondaries mirror themselves from this
    prepared = _prepare_mtf(df, p.get("mtf_emas"))
    prepared = scenario.prepare(prepared, p)
    prepared = stop.prepare(prepared, p)
    prepared = tp.prepare(prepared, p)
    secs = list(secondary) if isinstance(secondary, (list, tuple)) else         ([secondary] if secondary is not None else [])
    for sec in secs:
        prepared = sec.prepare(prepared, p)

    entry = scenario.entry(p)
    if entry_filter is not None:
        # an extra user rule ANDed onto the scenario's own trigger. Build it with the Cond algebra:
        #   (below_mtf('5min',200,n=3) & red(2)) | above_mtf('15min',50)
        entry = entry & entry_filter
    exit_cond = tp.exit(p)
    reset = scenario.reset(p)
    rule = Rule("break", entry=entry, exit=exit_cond, side=scenario.side, once=True)
    # Time filters: allow-lists (keep) AND block-lists (exclude). All optional, all read from params.
    strat = RuleStrategy(indicators=(), rules=(rule,), reset=reset,
                         session=p.get("session"), hours=p.get("hours"),
                         weekdays=p.get("weekdays"), avoid_weekends=bool(p.get("avoid_weekends", False)),
                         block_hours=p.get("block_hours"), block_weekdays=p.get("block_weekdays"),
                         block_times=p.get("block_times"), filter_tz=p.get("filter_tz"))
    signals = strat.signals(prepared)

    if secs:
        # per-trade arm + in-profit-gated exit (engine-side; see Signals docstring).
        # Several secondaries simply OR together: any of them can arm the trade, and any of their
        # exit conditions can then close it (still only while in profit).
        arm = np.logical_or.reduce([np.asarray(s.arm(p)(prepared), bool) for s in secs])
        alt = np.logical_or.reduce([np.asarray(s.exit(p)(prepared), bool) for s in secs])
        rep = bool(p.get("sec_replace_tp", False))
        on_tp = bool(p.get("sec_arm_on_tp", False))
        if scenario.side == "short":
            signals = dataclasses.replace(signals, exit_arm_short=arm, exit_alt_short=alt,
                                          alt_replace_tp=rep, alt_arm_on_tp=on_tp)
        else:
            signals = dataclasses.replace(signals, exit_arm_long=arm, exit_alt_long=alt,
                                          alt_replace_tp=rep, alt_arm_on_tp=on_tp)

    cfg = dataclasses.replace(base_cfg, exit_enabled=True)
    if scenario.side == "short":
        cfg = dataclasses.replace(cfg, allow_short=True)
    cfg = stop.apply(cfg, p)
    cfg = tp.apply(cfg, p)
    return prepared, signals, cfg, p


def run(preset: Preset, df: Optional[pd.DataFrame] = None, *, entry_filter=None,
        time_col: str = "t", price_col: str = "close") -> ScenarioRun:
    """Load (if needed) → prepare → compose stop+TP → backtest. Returns a `ScenarioRun`."""
    scenario = resolve_scenario(preset.scenario)
    stop = resolve_stop(preset.stop)
    tp = resolve_tp(preset.tp)
    secondary = resolve_secondaries(preset.secondary)
    if df is None:
        from ..data import get_ohlcv
        df = get_ohlcv(**preset.data)
    base_cfg = BacktestConfig(**preset.cfg)
    prepared, signals, cfg, p = compose(df, scenario, stop, tp, preset.params, base_cfg, secondary,
                                        entry_filter=entry_filter)
    res = run_backtest(prepared, signals, cfg, time_col=time_col, price_col=price_col)
    return ScenarioRun(result=res, df=prepared, cfg=cfg, params=p, scenario=scenario, stop=stop,
                       tp=tp, secondary=(secondary or None), preset=preset)


def run_pieces(df: pd.DataFrame, scenario: str, stop: str, tp: str, *,
               secondary=None, entry_filter=None,
               params: Optional[dict] = None, cfg: Optional[dict] = None,
               time_col: str = "t", price_col: str = "close") -> ScenarioRun:
    """Convenience for notebooks/sweeps: run directly from piece keys on an already-loaded frame,
    without constructing a Preset first.

    `entry_filter` is an extra `Cond` ANDed onto the scenario's entry — build it with the normal
    boolean algebra, e.g. `(below_mtf('5min',200,n=3) & red(2)) | above_mtf('15min',50)`.
    NOTE: a Cond is code, not data, so it is NOT stored in a Preset; pass it at run time.
    """
    preset = Preset(name="adhoc", scenario=scenario, stop=stop, tp=tp, secondary=secondary,
                    params=params or {}, cfg=cfg or {})
    return run(preset, df=df, entry_filter=entry_filter, time_col=time_col, price_col=price_col)
