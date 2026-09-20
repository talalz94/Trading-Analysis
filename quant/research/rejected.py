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

    # the LINE each rejection belongs to, so the chart can draw the setup that never traded and
    # not just a dot. Geometry comes from the line's own frame -- the same source the drawn
    # trendlines use, so a missed setup and a traded one cannot disagree about where the line was.
    geom = {}
    for pref, frame in (lab.line_frames or {}).items():
        from ..indicators.trendlines import trendline_table
        try:
            tt = trendline_table(frame)
        except Exception:
            continue
        tv = frame["t"].to_numpy()
        for r in tt.itertuples():
            geom[(pref, r.id)] = (tv[int(r.start_i)], float(r.y_start),
                                  tv[int(r.end_i)], float(r.y_end))

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
                # WHICH line, and where it ran -- `{pref}tl_{res|sup}_n` is the id live on that
                # bar, the same column a taken trade uses to name the line it broke.
                idc = f"{pref}tl_{'res' if side == 'long' else 'sup'}_n"
                num = int(D[idc].to_numpy()[i]) if idc in D.columns else -1
                lid = f"{'R' if side == 'long' else 'S'}{num}" if num >= 0 else ""
                g = geom.get((tf, lid), (None, None, None, None))
                rows.append({"t": t.iloc[at], "side": side, "tf": tf, "reason": reason,
                             "price": float(cl[at]), "break_t": t.iloc[i],
                             "line": f"{tf}·{lid}" if lid else "",
                             "x0": g[0], "y0": g[1], "x1": g[2], "y1": g[3]})

    out = pd.DataFrame(rows, columns=["t", "side", "tf", "reason", "price", "break_t",
                                      "line", "x0", "y0", "x1", "y1"])
    if not len(out):
        return out
    # WARM-UP REJECTIONS ARE NOT MISSED TRADES. The frame starts at WARMUP_START so the lines and
    # gaps have history, but the chart opens at VIEW_START -- a rejection from three weeks before
    # the window is off-screen, un-clickable, and would pad the picker with rows nobody can look at.
    vs = getattr(c, "VIEW_START", None)
    if vs is not None:
        out = out[out["t"] >= pd.Timestamp(vs, tz="UTC")]
    return out.sort_values("t").reset_index(drop=True)
