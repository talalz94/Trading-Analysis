"""
NOTEBOOK 21'S STRATEGY ON OTHER MARKETS -- which, if any, shows promise?

Runs notebook 21's EXACT trading settings (read fresh from its config, so this always tests what 21
trades) on every market in notebook 22's config, quarter by quarter over 2022Q1-2026Q3, with each
market's own costs. Four runs per market per quarter:

  as set     notebook 21 as it stands, exits and all                   -- the headline
  zero cost  the same trades, costs removed                            -- the rule vs the costs
  plain      every optional exit rule off: stop + 3R target only       -- the entry, on its own
  metronome  an entry every NULL_EVERY bars, ON THE SAME SIDE OF THE 15m EMA the entry rule asks for,
             with the plain exits                                      -- the null

A MARKET SHOWS PROMISE ONLY IF the strategy makes money after costs in BOTH halves AND its entries
beat the metronome's in both (plain vs metronome, per trade). The second test is the one that
matters: a market that trended for two years makes nearly any entry on the trend's side look good,
and the metronome takes the same side of the same EMA without a trendline. The project's standing
rule (CLAUDE.md, "make every entry compete against the time_entry metronome").

DATA. Only 5m is downloaded (Dukascopy for gold / oil / FX, Binance for BTC); 15m, 1h and 4h are
resampled from it, which is exact going down and makes every market's frames come from one source.
Finer frames (a 1m EMA on the chart) still come from the loader.

COSTS. A market's settings REPLACE notebook 21's fee_bps / slippage_bps / spread /
commission_per_lot -- anything unstated is 0 -- so Binance's 8 bps fee is never charged on a gold
CFD. The spreads are typical retail all-in figures, stated in notebook 22's config.

    python -m experiments.instrument_scan                 (~20 min; the downloads come first)
    python -m experiments.instrument_scan --report-only   re-read the saved trades
Out: experiments/results_instrument_scan.csv   one row per (market, quarter, run, trade)
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
OUT = ROOT / "experiments" / "results_instrument_scan.csv"

RUNS = ("as set", "zero cost", "plain", "metronome")
_COST_KEYS = ("fee_bps", "slippage_bps", "spread", "commission_per_lot")
_EXIT_OFF = dict(USE_TP_MOVE=False, USE_FVG_EXIT=False, USE_TP_DECAY=False, USE_STOP_STEPS=False,
                 USE_STOP_TIME=False, USE_HARD_CLOSE=False, USE_RETEST_BAIL=False)


# ---------------------------------------------------------------- settings
def base_settings(cfg) -> Dict[str, object]:
    """Notebook 21's settings, read fresh from the file notebook 22 names."""
    from quant.research.workspace import load_settings
    return load_settings(NOTEBOOKS / cfg.BASE)


def market_cfg(base: Dict[str, object], market: dict) -> SimpleNamespace:
    """The base settings on one market: its symbol, its data source, ITS costs."""
    costs = dict(base["COSTS"])
    costs.update({k: market.get(k, 0) for k in _COST_KEYS})
    return SimpleNamespace(**{**base, "SYMBOL": market["symbol"], "SOURCE": market["source"],
                              "COSTS": costs})


# ---------------------------------------------------------------- frames: everything from 5m
@contextmanager
def frames_from_5m():
    """While active, every timeframe of 5m or coarser is RESAMPLED from the 5m cache."""
    import quant.data as qdata
    from quant.data import get_ohlcv, resample_ohlcv
    from quant.data.timeframe import tf_frames as _real, tf_minutes

    def frames(symbol, tfs, *, view_start, end, warmup_bars=800, source="binance",
               market="spot", tz="UTC", **kw):
        tfs = tuple(dict.fromkeys(tfs))
        fine = tuple(t for t in tfs if tf_minutes(t) < 5)
        out = _real(symbol, fine, view_start=view_start, end=end, warmup_bars=warmup_bars,
                    source=source, market=market, tz=tz, **kw) if fine else {}
        vs = pd.Timestamp(view_start)
        vs = vs.tz_localize("UTC") if vs.tzinfo is None else vs
        for tf in sorted((t for t in tfs if t not in fine), key=tf_minutes, reverse=True):
            start = (vs - pd.Timedelta(minutes=tf_minutes(tf) * int(warmup_bars))).strftime("%Y-%m-%d")
            d5 = get_ohlcv(symbol, "5m", start=start, end=end, source=source, market=market, tz=tz)
            out[tf] = (d5 if tf_minutes(tf) == 5 else
                       resample_ohlcv(d5, f"{tf_minutes(tf)}min")).reset_index(drop=True)
        return out

    qdata.tf_frames = frames
    try:
        yield
    finally:
        qdata.tf_frames = _real


# ---------------------------------------------------------------- the null
def _metronome(lab, cfg, every: int) -> pd.DataFrame:
    """An entry every `every` bars on the EMA side the entry rule asks for, plain exits."""
    from quant.scenarios import run_pieces
    from quant.strategies.conditions import raw
    from quant.strategies.rulespec import build_rules
    vs = pd.Timestamp(cfg.VIEW_START, tz="UTC")
    in_window = raw(lambda d: (pd.to_datetime(d["t"]) >= vs).to_numpy(), "in_window")
    p = dict(stop_n=cfg.STOP_N, stop_buffer_pct=cfg.STOP_BUFFER_PCT, be_at_rr=cfg.BE_AT_RR,
             tp_rr=cfg.TP_RR, te_every=int(every))
    parts = []
    for side, on in (("long", cfg.TRADE_LONG), ("short", cfg.TRADE_SHORT)):
        if not on:
            continue
        rule = build_rules(cfg.ENTRY_RULES, side, chart_tf=cfg.TF,
                           combine=getattr(cfg, "RULES_COMBINE", "all"))
        flt = in_window if rule is None else (in_window & rule)
        r = run_pieces(lab.D, f"time_entry_{side}@v1", "high_of_n@v1", "be_ladder@v1",
                       params=p, cfg=dict(cfg.COSTS), entry_filter=flt)
        parts.append(r.trades)
    t = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(t):
        risk = (t["entry_price"] - t["stop_price"]).abs()
        t["R"] = t["pnl"] / (risk * t["qty"])
    return t


# ---------------------------------------------------------------- the scan
@dataclass
class Scan:
    trades: pd.DataFrame          # market, quarter, phase, run, entry, side, R, why
    train: list
    forward: list

    def summary(self) -> pd.DataFrame:
        """One row per market, best-looking first. Read `verdict` before any number."""
        rows = []
        t = self.trades[self.trades.R.notna()]
        for mk, g in t.groupby("market", sort=False):
            a, z = g[g.run == "as set"], g[g.run == "zero cost"]
            pl, mt = g[g.run == "plain"], g[g.run == "metronome"]

            def rpt(x, ph=None):
                x = x if ph is None else x[x.phase == ph]
                return float(x.R.mean()) if len(x) else np.nan

            def trim(x, k=5):
                x = x.R.sort_values()
                return float(x.iloc[:-k].mean()) if len(x) > k else np.nan

            # the entry against the null, quarter by quarter, per trade
            pq = pl.groupby("quarter").R.mean() - mt.groupby("quarter").R.mean()
            pq = pq.dropna()
            sd = pq.std(ddof=1)
            per_q = a.groupby("quarter").R.sum().reindex(self.train + self.forward).fillna(0)
            edge_tr = rpt(pl, "train") - rpt(mt, "train")
            edge_fw = rpt(pl, "forward") - rpt(mt, "forward")
            ok = (rpt(a, "train") > 0 and rpt(a, "forward") > 0 and edge_tr > 0 and edge_fw > 0)
            rows.append({
                "market": mk, "trades": len(a),
                "R/trade": round(rpt(a), 3), "train": round(rpt(a, "train"), 3),
                "forward": round(rpt(a, "forward"), 3),
                "zero cost": round(rpt(z), 3), "cost/trade": round(rpt(z) - rpt(a), 3),
                "trim top 5": round(trim(a), 3),
                "quarters > 0": f"{int((per_q > 0).sum())}/{len(per_q)}",
                "entry vs null, train": round(edge_tr, 3),
                "entry vs null, fwd": round(edge_fw, 3),
                "t": round(pq.mean() / (sd / np.sqrt(len(pq))), 2) if sd > 0 else np.nan,
                "verdict": "promising" if ok else "no",
            })
        out = pd.DataFrame(rows)
        return out.sort_values("forward", ascending=False).reset_index(drop=True)

    def by_quarter(self) -> pd.DataFrame:
        """Total R per quarter, per market, as set."""
        a = self.trades[(self.trades.run == "as set") & self.trades.R.notna()]
        return (a.groupby(["market", "quarter"]).R.sum().unstack("quarter")
                .reindex(columns=self.train + self.forward).fillna(0).round(1))

    def verdict(self) -> str:
        s = self.summary()
        n = len(s)
        good = list(s.market[s.verdict == "promising"])
        lines = [f"{n} markets, {len(self.train) + len(self.forward)} quarters each.",
                 "A market counts only if it makes money after costs in BOTH halves AND its "
                 "trendline entries beat the metronome's in both."]
        lines.append(f"promising: {', '.join(good)}" if good else
                     "promising: none -- no market passes both tests")
        # four coin flips (two halves x two tests) for a market with no edge: about 1 in 16 passes
        lines.append(f"markets that pass by luck alone: ~{n / 16:.1f} of {n} -- a lone pass needs "
                     f"replicating on a market it was not picked on")
        best_t = s["t"].abs().max()
        lines.append(f"largest |t| of the entry against the null: {best_t:.2f}"
                     + (" -- below 2.5, noise" if best_t < 2.5 else " -- worth a closer look"))
        return "\n".join(lines)


def scan(cfg, markets: Optional[dict] = None, progress: bool = True, save: bool = True) -> Scan:
    """Run the four runs for every market and quarter. `cfg` is notebook 22's config module."""
    from experiments.evaluate import quarters, window_cfg
    from quant.research import fvg_lab
    train, forward = quarters(*cfg.TRAIN), quarters(*cfg.FORWARD)
    base = base_settings(cfg)
    rows = []
    t0 = time.time()
    with frames_from_5m():
        for name, mk in (markets or cfg.MARKETS).items():
            mc = market_cfg(base, mk)
            for q in train + forward:
                w = window_cfg(mc, q, cfg.DATA_END)
                lab = fvg_lab(w)
                free = {**w.COSTS, **{k: 0 for k in _COST_KEYS}}
                runs = {"as set": lab.trades,
                        "zero cost": lab.retrade(COSTS=free).trades,
                        "plain": lab.retrade(**_EXIT_OFF).trades,
                        "metronome": _metronome(lab, w, cfg.NULL_EVERY)}
                for run, t in runs.items():
                    for r in t.itertuples():
                        rows.append(dict(market=name, quarter=q,
                                         phase="train" if q in train else "forward", run=run,
                                         entry=str(r.entry_time)[:16], side=r.side, R=float(r.R),
                                         why=r.close_reason))
                if progress:
                    print(f"{name:8s} {q}: {len(lab.trades):3d} trades  "
                          f"{lab.trades.R.sum() if len(lab.trades) else 0:+6.1f}R  "
                          f"({time.time() - t0:.0f}s)", flush=True)
    out = Scan(trades=pd.DataFrame(rows), train=train, forward=forward)
    if save:
        out.trades.to_csv(OUT, index=False)
    return out


def load_scan(cfg) -> Scan:
    from experiments.evaluate import quarters
    return Scan(trades=pd.read_csv(OUT), train=quarters(*cfg.TRAIN),
                forward=quarters(*cfg.FORWARD))


def lab_for(cfg, market: str, start: str, end: str):
    """Notebook 21's lab on one market and window, frames built exactly as the scan builds them --
    for drawing the chart of a market the scan picked out."""
    from quant.research import fvg_lab
    mc = market_cfg(base_settings(cfg), cfg.MARKETS[market])
    w = SimpleNamespace(**{**vars(mc), "VIEW_START": start, "END": end, "AS_OF": None,
                           "WARMUP_START": (pd.Timestamp(start) - pd.DateOffset(months=1))
                           .strftime("%Y-%m-%d")})
    with frames_from_5m():
        return fvg_lab(w)


if __name__ == "__main__":
    sys.path.insert(0, str(NOTEBOOKS))
    import nb22_config as CFG
    s = load_scan(CFG) if "--report-only" in sys.argv else scan(CFG)
    pd.set_option("display.width", 220)
    print(s.summary().to_string(index=False))
    print(s.verdict())
