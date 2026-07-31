"""
Declarative rule-builder: compose a strategy from named entry->exit rules.

A `RuleStrategy` is a list of `Rule`s over a shared set of declared indicators. Each rule pairs an
entry condition with its own exit condition, so complex multi-setup strategies read as a clean list
instead of a pile of flags:

    from quant.strategies import RuleStrategy, Rule
    from quant.strategies.conditions import ema, heikin_ashi, swings, above, full_above, full_below, ha_red

    strat = RuleStrategy(
        indicators=[ema(200), ema(50), heikin_ashi(), swings(5, 5)],
        reset=full_below("ema_200"),                    # a full candle below EMA200 ends an "episode"
        rules=[
            Rule("breakout",                            # PRIMARY: EMA200 breakout, one per episode
                 entry=full_above("ema_200", n=2),
                 exit=full_below("ema_50") | ha_red(3),
                 once=True),
            Rule("pullback",                            # SECONDARY: only after breakout traded+closed
                 entry=above("close", "ema_200") & full_above("ema_50", n=2),
                 exit=full_below("ema_50") | ha_red(3),
                 after="breakout", once=True),
        ],
    )
    res = strat.backtest(df, cfg)                       # stop-loss comes from cfg (e.g. ref_col swing_last_low)

Semantics
---------
- `once=True` (or a custom `rearm` condition) makes a rule fire only ONCE per episode, re-arming
  when `rearm` (default: the strategy `reset`) becomes true. This is the "one entry per cross" idea,
  generalized — the re-arm condition is independent of the entry condition.
- `after="name"` gates a rule so it can only fire once the named earlier rule has fired this episode
  (latched until `reset`). With `max_open_trades=1` the engine won't actually open the second trade
  until the first has closed, giving "secondary entry only after the primary trade closed".
- Exits: rules can share an exit or use different ones. When every rule on a side shares the same
  exit, the strategy compiles to a plain `Signals` and runs on the standard kernel at full speed
  (byte-identical to hand-written signals). Distinct per-rule exits engage the leg-aware kernel
  path (see engine); the zero-cost/non-margin numbers are unchanged either way.

Stops are configured on `BacktestConfig` (shared across rules), e.g.
`sl_mode="ref_col", sl_ref_long_col="swing_last_low"`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import signals as S
from ..engine import Signals
from .base import Strategy
from .conditions import Cond, Indicator, apply_indicators
from .indicators import Ind


@dataclass
class Rule:
    """One entry->exit rule. `side` is 'long' or 'short'."""
    name: str
    entry: Cond
    exit: Optional[Cond] = None
    side: str = "long"
    once: bool = False                 # fire only once per episode (re-arm on `rearm`/`reset`)
    rearm: Optional[Cond] = None       # custom re-arm condition (defaults to the strategy `reset`)
    after: Optional[str] = None        # only eligible after this earlier rule fired this episode
    stop: Optional[object] = None      # per-rule StopSpec; None -> use the shared BacktestConfig stop

    def __post_init__(self):
        if self.side not in ("long", "short"):
            raise ValueError(f"Rule {self.name}: side must be 'long'/'short', got {self.side!r}")


@dataclass
class CompiledRules:
    """Per-rule entry/exit boolean arrays + which side, in priority (declared) order.
    Each leg is (name, entry, exit, stop-or-None)."""
    entry_long: np.ndarray
    exit_long: np.ndarray
    entry_short: np.ndarray
    exit_short: np.ndarray
    long_legs: List[Tuple[str, np.ndarray, np.ndarray, object]]
    short_legs: List[Tuple[str, np.ndarray, np.ndarray, object]]
    shared_long_exit: bool
    shared_short_exit: bool


@dataclass
class RuleStrategy(Strategy):
    name: str = "rule_strategy"
    indicators: Tuple[Indicator, ...] = ()
    rules: Tuple[Rule, ...] = ()
    reset: Optional[Cond] = None       # episode boundary: re-arms `once` rules and `after` latches

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute every declared indicator ONCE, in declared order.

        Accepts tagged handles (`MA('trend', 200, …)`) and the legacy specs (`ema(200)`) side by
        side. Identical re-declarations are skipped; a tag reused with DIFFERENT settings is an
        error (that silent overwrite is exactly what tags exist to prevent).
        """
        out = df.copy()
        seen_tags: dict = {}
        seen_specs: set = set()
        for ind in self.indicators:
            if isinstance(ind, Ind):
                sig = ind.signature()
                prev = seen_tags.get(ind.tag)
                if prev is not None:
                    if prev != sig:
                        raise ValueError(
                            f"Indicator tag {ind.tag!r} declared twice with different settings. "
                            f"Give them distinct tags.")
                    continue
                seen_tags[ind.tag] = sig
                out = ind.apply(out)
                # Guard the classic trap: a long-period (or higher-timeframe) indicator that never
                # warms up on a short window is all-NaN, which silently yields zero signals.
                empty = [c for c in ind.columns
                         if c in out.columns and not out[c].notna().any()]
                if empty:
                    import warnings
                    hint = (f" — {ind.tf} bars in this window may be fewer than the period"
                            if ind.tf else " — the window may be shorter than the period")
                    warnings.warn(
                        f"Indicator {ind.tag!r} produced no values ({', '.join(empty)} all NaN)"
                        f"{hint}. Rules using it will never fire.", stacklevel=2)
            else:                                   # legacy Indicator spec
                if ind.key in seen_specs:
                    continue
                seen_specs.add(ind.key)
                out = ind.apply(out)
        return out

    # ---- compilation ----
    def _compile(self, df: pd.DataFrame) -> CompiledRules:
        n = len(df)
        zeros = np.zeros(n, dtype=bool)
        reset_mask = self.reset(df) if self.reset is not None else zeros
        fires: dict = {}                 # rule name -> post-one-shot fired array
        long_legs: List[Tuple[str, np.ndarray, np.ndarray]] = []
        short_legs: List[Tuple[str, np.ndarray, np.ndarray]] = []

        for rule in self.rules:
            fire_ready = rule.entry(df)
            if rule.after is not None:
                if rule.after not in fires:
                    raise ValueError(
                        f"Rule {rule.name!r}: after={rule.after!r} must reference an EARLIER rule.")
                # Eligible only from the bar AFTER the prerequisite fired (a secondary entry can't
                # trigger on the primary's own entry bar — the primary trade hasn't closed yet).
                prereq = np.zeros(n, bool)
                prereq[1:] = fires[rule.after][:-1]
                gate = S.latch(prereq, reset_mask)
                fire_ready = fire_ready & gate
            if rule.once or rule.rearm is not None:
                rearm = rule.rearm(df) if rule.rearm is not None else reset_mask
                fired = S.arm_reset_trigger(fire_ready, rearm)
            else:
                fired = np.asarray(fire_ready, bool)
            fires[rule.name] = fired
            ex = rule.exit(df) if rule.exit is not None else zeros
            (long_legs if rule.side == "long" else short_legs).append(
                (rule.name, fired, ex, rule.stop))

        def _combine(legs):
            if not legs:
                return zeros, zeros, True
            entry = np.zeros(n, bool)
            for _, e, _x, _s in legs:
                entry |= e
            exits = [x for _, _e, x, _s in legs]
            shared = all(np.array_equal(x, exits[0]) for x in exits[1:])
            union_exit = np.zeros(n, bool)
            for x in exits:
                union_exit |= x
            return entry, union_exit, shared

        el, xl, shared_l = _combine(long_legs)
        es, xs, shared_s = _combine(short_legs)
        return CompiledRules(el, xl, es, xs, long_legs, short_legs, shared_l, shared_s)

    @staticmethod
    def _entry_leg(side_legs, base_idx, n):
        """Per-bar global leg index for one side: the first (highest-priority) firing rule wins;
        `base_idx` is where this side's legs start in the shared leg table."""
        entry_leg = np.full(n, base_idx, np.int64)   # default is moot when no entry fires the bar
        assigned = np.zeros(n, bool)
        for local, leg in enumerate(side_legs):
            f = np.asarray(leg[1], bool)
            take = f & ~assigned
            entry_leg[take] = base_idx + local
            assigned |= f
        return entry_leg

    def build_signals(self, df: pd.DataFrame) -> Signals:
        c = self._compile(df)
        n = len(df)
        has_short = bool(c.entry_short.any())
        legs = list(c.long_legs) + list(c.short_legs)
        distinct_exits = (not c.shared_long_exit) or (not c.shared_short_exit)
        any_stop = any(leg[3] is not None for leg in legs)
        if not distinct_exits and not any_stop:
            # Shared exit and no per-rule stops -> plain Signals on the standard (fastest) path.
            return Signals(entry_long=c.entry_long, exit_long=c.exit_long,
                           entry_short=c.entry_short if has_short else None,
                           exit_short=c.exit_short if has_short else None)
        # Leg mode: one shared leg table (long legs first, then short). Each trade closes on the
        # exit of the rule that opened it, and (if set) uses that rule's own stop (o_leg in kernel).
        exit_legs = np.zeros((len(legs), n), np.uint8)
        for idx, leg in enumerate(legs):
            exit_legs[idx] = np.asarray(leg[2], bool).astype(np.uint8)
        entry_leg_long = self._entry_leg(c.long_legs, 0, n)
        entry_leg_short = self._entry_leg(c.short_legs, len(c.long_legs), n)
        stop_specs = [leg[3] for leg in legs] if any_stop else None
        return Signals(
            entry_long=c.entry_long, exit_long=c.exit_long,
            entry_short=c.entry_short if has_short else None,
            exit_short=c.exit_short if has_short else None,
            entry_leg_long=entry_leg_long, entry_leg_short=entry_leg_short,
            exit_legs_long=exit_legs, exit_legs_short=exit_legs, stop_specs=stop_specs)

    def params(self) -> dict:
        # RuleStrategy carries Cond/Rule objects, not sweepable scalars; keep params light.
        return {"name": self.name, "n_rules": len(self.rules)}

    def describe(self) -> str:
        lines = [f"RuleStrategy '{self.name}' ({len(self.rules)} rules):"]
        for r in self.rules:
            seq = f" after={r.after}" if r.after else ""
            once = " once" if (r.once or r.rearm is not None) else ""
            ex = r.exit.name if r.exit is not None else "(cfg SL/TP only)"
            lines.append(f"  [{r.side}] {r.name}{seq}{once}: entry {r.entry.name}  ->  exit {ex}")
        if self.reset is not None:
            lines.append(f"  reset (episode boundary): {self.reset.name}")
        return "\n".join(lines)
