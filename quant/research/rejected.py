"""
THE TRADES THAT DID NOT HAPPEN — every trendline break, and why it was turned down.

A filter is only judgeable against what it removed. The chart shows the trades that were taken, so
a rule that quietly deletes a good setup looks identical to one that deletes nothing: both just
leave a shorter list. This puts the rejections back on the chart, each labelled with the reason,
so "my confirmation cost me a winner" is a thing you can SEE rather than infer from a total.

It re-runs the conditions rather than instrumenting the engine, because the engine is a compiled
kernel and the question is about the SIGNAL, not the fill. Four reasons, in the order they apply:

    entry rule      the ENTRY_RULES sentence for that side was false on the break bar
    cancelled       price closed back over rs inside the confirmation window
    not confirmed   the window expired with no qualifying candle
    already open    the signal was good but a position was already running (max_open_trades)

Anything not listed here became a trade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["rejected_setups"]

_REASONS = ("entry rule", "cancelled", "not confirmed", "already open")


def rejected_setups(lab) -> pd.DataFrame:
    """One row per break that did not become a trade. Columns: t, side, tf, reason, price."""
    from ..scenarios.entries import _confirm_beyond, _tl_entry, tl_per_tf
    from ..strategies.rulespec import build_rules

    D, p = lab.D, lab.params
    c = lab.cfg
    o, h, lo, cl = (D[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    t = D["t"]

    busy = np.zeros(len(D), bool)                 # bars a position was already running over
    tr = lab.trades
    if len(tr) and {"entry_i", "exit_i"} <= set(tr.columns):
        for a, b in zip(tr["entry_i"].to_numpy(int), tr["exit_i"].to_numpy(int)):
            busy[a:max(a + 1, b)] = True

    rows = []
    for pref in tuple(p.get("tl_prefixes") or ("",)):
        tf = (p.get("tl_tf_names") or {}).get(pref, pref or "chart")
        for side in ("long", "short"):
            if not getattr(c, "TRADE_LONG" if side == "long" else "TRADE_SHORT", True):
                continue
            raw = _tl_entry(dict(p, tl_prefixes=(pref,), tl_confirm_n=0), side)(D)
            if not raw.any():
                continue
            if str(p.get("tl_confirm_from", "every")).lower() == "first":
                raw = raw & ~np.concatenate(([False], raw[:-1]))

            rule = build_rules(getattr(c, "ENTRY_RULES", []), side, chart_tf=c.TF,
                               combine=getattr(c, "RULES_COMBINE", "all"))
            rule_ok = rule(D) if rule is not None else np.ones(len(D), bool)

            n = tl_per_tf(p, "tl_confirm_n", pref)
            kw = dict(need_all=bool(p.get("tl_confirm_all", False)),
                      mode=str(p.get("tl_confirm_mode", "close")),
                      colour=bool(p.get("tl_confirm_colour", True)))
            cancel = bool(p.get("tl_confirm_cancel", False))

            for i in np.flatnonzero(raw):
                if not rule_ok[i]:
                    reason, at = "entry rule", i
                elif n > 0:
                    one = np.zeros(len(D), bool); one[i] = True
                    got = _confirm_beyond(one, o, h, lo, cl, side, n, cancel=cancel, **kw)
                    if got.any():
                        at = int(np.flatnonzero(got)[0])
                        if not busy[at]:
                            continue                      # this one traded
                        reason = "already open"
                    else:
                        # separate the two failures: a cancel is a different story from a timeout
                        no_cancel = _confirm_beyond(one, o, h, lo, cl, side, n, cancel=False, **kw)
                        reason = "cancelled" if (cancel and no_cancel.any()) else "not confirmed"
                        at = i
                elif busy[i]:
                    reason, at = "already open", i
                else:
                    continue
                rows.append({"t": t.iloc[at], "side": side, "tf": tf, "reason": reason,
                             "price": float(cl[at]), "break_t": t.iloc[i]})

    out = pd.DataFrame(rows, columns=["t", "side", "tf", "reason", "price", "break_t"])
    return out.sort_values("t").reset_index(drop=True) if len(out) else out
