"""
Management sweeps that don't recompute the entry every time.

`run_pieces` prepares the whole scenario from scratch per call, which is correct but wasteful for a
management sweep: the entry's indicators (zigzag, squeeze, wavetrend, VWAP, session levels...) are
identical across every stop x target x time-stop combination, and on 300k 1-minute bars they cost far
more than the kernel does. `EntryBundle` prepares the entry ONCE, then re-applies only the stop and
target for each combination. Same code path as `compose()` otherwise — same registries, same Rule,
same RuleStrategy, same engine call — so results are identical to `run_pieces`, just faster.
"""
from __future__ import annotations

import dataclasses
import itertools
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from quant.engine import BacktestConfig, run_backtest
from quant.scenarios.base import (_prepare_mtf, merged_params, resolve_scenario,
                                  resolve_secondaries, resolve_stop, resolve_tp)
from quant.strategies.builder import Rule, RuleStrategy

from .search_harness import CFG_LEVERED, SPREAD_BTC, r_stats


class EntryBundle:
    """One entry family + its parameters, prepared once and reusable across management combos."""

    def __init__(self, df: pd.DataFrame, scenario: str, entry_params: Optional[dict] = None,
                 *, entry_filter=None):
        self.key = scenario
        self.scen = resolve_scenario(scenario)
        self.entry_params = dict(entry_params or {})
        self.entry_filter = entry_filter          # default; `run` may override per call
        p = {**self.scen.default_params, **self.entry_params, "_side": self.scen.side}
        self._df = self.scen.prepare(_prepare_mtf(df, p.get("mtf_emas")), p)

    def run(self, stop: str, tp: str, params: Optional[dict] = None,
            cfg: Optional[dict] = None, secondary=None, entry_filter=None):
        st, tpm = resolve_stop(stop), resolve_tp(tp)
        secs = resolve_secondaries(secondary)
        p = merged_params(self.scen, st, tpm, {**self.entry_params, **(params or {})}, secs)
        p["_side"] = self.scen.side
        d = tpm.prepare(st.prepare(self._df, p), p)
        for s in secs:
            d = s.prepare(d, p)

        entry = self.scen.entry(p)
        # A regime conditioner only gates an existing trigger, so it must NOT force the entry's
        # indicators to be recomputed - the filter is applied here, not at construction.
        filt = self.entry_filter if entry_filter is None else entry_filter
        if filt is not None:
            entry = entry & filt
        rule = Rule("entry", entry=entry, exit=tpm.exit(p), side=self.scen.side, once=True)
        strat = RuleStrategy(indicators=(), rules=(rule,), reset=self.scen.reset(p),
                             session=p.get("session"), hours=p.get("hours"),
                             weekdays=p.get("weekdays"),
                             block_hours=p.get("block_hours"), block_times=p.get("block_times"))
        sig = strat.signals(d)
        if secs:
            arm = np.logical_or.reduce([np.asarray(s.arm(p)(d), bool) for s in secs])
            alt = np.logical_or.reduce([np.asarray(s.exit(p)(d), bool) for s in secs])
            fld = dict(exit_arm_short=arm, exit_alt_short=alt) if self.scen.side == "short" \
                else dict(exit_arm_long=arm, exit_alt_long=alt)
            sig = dataclasses.replace(sig, alt_replace_tp=bool(p.get("sec_replace_tp", False)),
                                      alt_arm_on_tp=bool(p.get("sec_arm_on_tp", False)), **fld)
        bc = BacktestConfig(**{**(cfg or {}), "exit_enabled": True,
                              "allow_short": self.scen.side == "short"})
        bc = tpm.apply(st.apply(bc, p), p)
        return run_backtest(d, sig, bc)


def sweep(windows: Dict[str, pd.DataFrame], entries: List[tuple], mgmt: List[dict], *,
          spread: float = SPREAD_BTC, cfg_extra: Optional[dict] = None,
          verbose: bool = True) -> pd.DataFrame:
    """`entries` = [(tag, scenario_key, entry_params, entry_filter), ...];
    `mgmt` = [{'stop':..., 'tp':..., 'params':{...}, 'cfg':{...}, 'tag':...}, ...].

    Scored the same way as `search_harness.score`: the WORST window is the headline, because a
    config that only works in one regime is directional beta.
    """
    base_cfg = {**CFG_LEVERED, "spread": spread, **(cfg_extra or {})}
    rows, t0 = [], time.time()
    cache: Dict[tuple, Dict[str, EntryBundle]] = {}
    for ei, (etag, skey, eparams, efilter) in enumerate(entries, 1):
        # Entries that differ ONLY by their regime filter share one prepared frame.
        ck = (skey, tuple(sorted((k, repr(v)) for k, v in (eparams or {}).items())))
        if ck not in cache:
            cache[ck] = {w: EntryBundle(d, skey, eparams) for w, d in windows.items()}
        bundles = cache[ck]
        for m in mgmt:
            per = {}
            for w, b in bundles.items():
                try:
                    per[w] = r_stats(b.run(m["stop"], m["tp"], m.get("params"),
                                           {**base_cfg, **m.get("cfg", {})},
                                           secondary=m.get("secondary"),
                                           entry_filter=efilter))
                except Exception as exc:
                    per[w] = dict(n=0, error=f"{type(exc).__name__}: {exc}"[:90])

            def _v(k):
                return [float(v[k]) for v in per.values()
                        if isinstance(v.get(k), (int, float)) and v[k] == v[k]]

            exps, ex_top, meds, rets = _v("expR"), _v("expR_ex_top"), _v("med_R"), _v("ret")
            errs = sorted({v["error"] for v in per.values() if v.get("error")})
            row = dict(tag=f"{etag}|{m.get('tag', m['stop'] + '/' + m['tp'])}",
                       entry=etag, mgmt=m.get("tag", f"{m['stop']}/{m['tp']}"),
                       n_total=int(sum(v.get("n", 0) for v in per.values())),
                       expR_min=round(min(exps), 4) if exps else np.nan,
                       expR_mean=round(float(np.mean(exps)), 4) if exps else np.nan,
                       expR_ex_top_min=round(min(ex_top), 4) if ex_top else np.nan,
                       medR_min=round(min(meds), 4) if meds else np.nan,
                       ret_min=round(min(rets), 2) if rets else np.nan,
                       dd_max=round(max((v.get("dd") or 0) for v in per.values()), 2),
                       n_windows_pos=int(sum(1 for e in exps if e > 0)),
                       error="; ".join(errs)[:110])
            for w, v in per.items():
                row[f"expR_{w}"] = v.get("expR")
                row[f"n_{w}"] = v.get("n")
            rows.append(row)
        if verbose:
            print(f"  entry {ei}/{len(entries)} {etag}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows).sort_values("expR_min", ascending=False).reset_index(drop=True)


def mgmt_grid(stops: Iterable[tuple], tps: Iterable[tuple],
              time_stops: Iterable[int] = (0,)) -> List[dict]:
    """Cartesian product of (stop, its params) x (tp, its params) x time stop."""
    out = []
    for (sk, sp), (tk, tp), ts in itertools.product(stops, tps, time_stops):
        out.append(dict(stop=sk, tp=tk, params={**sp, **tp},
                        cfg=({"max_bars_in_trade": int(ts)} if ts else {}),
                        tag=f"{sk.split('@')[0]}{_short(sp)}/{tk.split('@')[0]}{_short(tp)}"
                            + (f"/t{ts}" if ts else "")))
    return out


def _short(d: dict) -> str:
    return "".join(f"{v:g}" for v in d.values()) if d else ""
