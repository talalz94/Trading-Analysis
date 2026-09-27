"""
WHEN DOES NOTEBOOK 21'S STRATEGY WORK, AND DOES THAT SAY WHICH EXIT TO USE?

For every trade notebook 21's settings took over 2022Q1-2026Q3, record what the market looked like
AT THE ENTRY BAR -- using only what was knowable at that bar's close -- and ask three questions:

  1. does anything separate the winners from the losers?        outcomes: win, R0 (zero-cost R)
  2. does anything say when MOVING THE TARGET helps rather than hurts?           outcome: dR_move
  3. does anything say when the FVG EXIT helps rather than hurts?                outcome: dR_fvg

WHY NOT PLAIN R FOR QUESTION 1 -- the first run did that and it was an artifact. 73% of trades are
losers, each worth -1R MINUS COSTS, and costs are a fixed % of price: a tight stop makes the same fee
a bigger slice of R. Ranked by R, the losers are simply ranked by stop width, so anything tied to
volatility "predicted" R (Spearman up to 0.54, 11 of 15 features "holding") while a model of all of
them scored AUC 0.48 on the forward half. So question 1 is asked of WIN/LOSS and of R at ZERO cost,
and the cost effect is reported on its own, as what it is.

dR is the arm's R minus the strategy's own R on the SAME entry (an exit rule can also free the slot
for a different later trade; only matched entries are compared, so dR is the rule's effect on that
trade and nothing else).

THE FEATURES, all side-ALIGNED where direction matters (positive = in the trade's favour, so a long
and a short with the same number mean the same thing):
  volatility   atr_pct (ATR as % of price) . atr_rank (ATR's percentile in the last 30 days)
               . shock (a 99th-percentile bar in the last hour -- the one large effect ever found
               here, WHAT_WORKS "volatility shock")
  trend/chop   er_5m (Kaufman efficiency, 4h of 5m bars) . chop_1h (choppiness index, 1h)
  momentum     rsi_5m . rsi_1h . mom_4h (the move INTO the break, in ATR)
  HTF bias     bias_1h (distance from the 1h EMA200, in 1h ATR) . bias_4h (from the 4h EMA50)
               . slope_4h (the 4h EMA50's slope)
  stretch      ema_dist (distance from the entry rule's own 15m EMA100, in ATR) . range_pos
               (where in the last day's range)
  the trade    risk_atr (stop distance in ATR) . vol_rel (volume vs its 4h mean) . side . trigger
               (5m/15m/1h line) . session

HOW IT IS JUDGED -- the project's rules, because 16 features x 3 outcomes is 48 chances to fool
yourself:
  * every relationship is measured on TRAIN (2022Q1-2024Q2) and again on FORWARD (2024Q3-2026Q3);
    a feature counts only if it is significant (Spearman p < 0.05) in BOTH, with the same sign.
    By chance that happens ~0.06 times in 48 tests;
  * tercile cut points come from TRAIN and are applied unchanged to FORWARD;
  * a model of all features at once (gradient boosting, shallow) is fitted on TRAIN and scored on
    FORWARD by AUC with a bootstrap interval -- the "everything together" answer -- and turned into a
    policy (trade / use the arm only where the model says) whose FORWARD R is the only number that
    would justify using it.

Prior art, so this is not read in isolation: WHAT_WORKS "hybrid / conditional strategies" tested
volatility terciles, trend vs chop, range position, session and HTF bias on the entry families and
found 3 survivors of 304 where chance predicts ~38.

Run:  python -m experiments.regime_study            (~4 min)
Out:  experiments/results_regime_study.csv           one row per trade: features and outcomes
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import quant.data as qdata                                                  # noqa: E402
from quant.data.timeframe import align_timeframes, tf_frames as _tf_frames  # noqa: E402
from quant.indicators import add_rsi, ema                                   # noqa: E402
from quant.indicators.volatility import atr                                 # noqa: E402
from quant.research import fvg_lab                                          # noqa: E402
from experiments.evaluate import frames_4h_from_1h, quarters, window_cfg   # noqa: E402

OUT = ROOT / "experiments" / "results_regime_study.csv"
TRAIN = quarters("2022Q1", "2024Q2")
FORWARD = quarters("2024Q3", "2026Q3")
DATA_END = "2026-08-31"
NUMERIC = ["atr_pct", "atr_rank", "er_5m", "chop_1h", "rsi_5m", "rsi_1h", "mom_4h", "bias_1h",
           "bias_4h", "slope_4h", "ema_dist", "range_pos", "risk_atr", "vol_rel", "shock"]
CATEGORICAL = ["side", "trigger", "session"]
OUTCOMES = ["win", "R0", "dR_move", "dR_fvg"]


# ---------------------------------------------------------------- features
def _chop(f: pd.DataFrame, n: int = 14) -> pd.Series:
    tr = pd.concat([f.high - f.low, (f.high - f.close.shift()).abs(),
                    (f.low - f.close.shift()).abs()], axis=1).max(axis=1)
    rng = f.high.rolling(n).max() - f.low.rolling(n).min()
    return 100 * np.log10(tr.rolling(n).sum() / rng) / np.log10(n)


def _htf(frame: pd.DataFrame, ema_n: int, tag: str) -> pd.DataFrame:
    f = frame[["t", "open", "high", "low", "close"]].copy()
    a = atr(f, 14)
    e = ema(f.close, ema_n)
    out = pd.DataFrame({"t": f.t, "open": f.open, "high": f.high, "low": f.low, "close": f.close,
                        f"ema_{tag}": e, f"atr_{tag}": a, f"slope_{tag}": (e - e.shift(6)) / a})
    if tag == "1h":
        out["rsi_1h"] = add_rsi(f, 14)["rsi_14"]
        out["chop_1h"] = _chop(f)
    return out


def features(lab, trades: pd.DataFrame) -> pd.DataFrame:
    """One row per trade, every value read at its ENTRY bar -- the close the trade was taken on."""
    D = lab.D.reset_index(drop=True)
    c, h, lo, v = D.close, D.high, D.low, D.volume
    a5 = atr(D, 14)
    rsi5 = add_rsi(D[["open", "high", "low", "close"]], 14)["rsi_14"]
    er = (c - c.shift(48)).abs() / c.diff().abs().rolling(48).sum()
    ret = c.pct_change().abs()
    q99 = ret.rolling(8640, min_periods=2000).quantile(0.99).shift(1)
    shock = (ret > q99).astype(float).rolling(12).max()
    atr_rank = a5.rolling(8640, min_periods=2000).rank(pct=True)
    hi_d, lo_d = h.rolling(288).max(), lo.rolling(288).min()
    rpos = (c - lo_d) / (hi_d - lo_d)
    htf = align_timeframes(D[["t"]].copy(),
                           {"1h": _htf(lab.line_frames["1h"], 200, "1h"),
                            "4h": _htf(lab.gap_frames["4h"], 50, "4h")}, shift_bars=1)
    ema15 = D["15m__ema_100"] if "15m__ema_100" in D else pd.Series(np.nan, index=D.index)

    i = trades["entry_i"].to_numpy()
    s = np.where(trades["side"].to_numpy() == "long", 1.0, -1.0)
    ci = c.to_numpy()[i]
    at = a5.to_numpy()[i]
    risk = (trades["entry_price"] - trades["stop_price"]).abs().to_numpy()
    hr = pd.to_datetime(trades["entry_time"]).dt.hour.to_numpy()
    f = pd.DataFrame({
        "atr_pct": 100 * at / ci,
        "atr_rank": atr_rank.to_numpy()[i],
        "er_5m": er.to_numpy()[i],
        "chop_1h": htf["1h__chop_1h"].to_numpy()[i],
        "rsi_5m": np.where(s > 0, rsi5.to_numpy()[i], 100 - rsi5.to_numpy()[i]),
        "rsi_1h": np.where(s > 0, htf["1h__rsi_1h"].to_numpy()[i],
                           100 - htf["1h__rsi_1h"].to_numpy()[i]),
        "mom_4h": s * (ci - c.shift(48).to_numpy()[i]) / at,
        "bias_1h": s * (ci - htf["1h__ema_1h"].to_numpy()[i]) / htf["1h__atr_1h"].to_numpy()[i],
        "bias_4h": s * (ci - htf["4h__ema_4h"].to_numpy()[i]) / htf["4h__atr_4h"].to_numpy()[i],
        "slope_4h": s * htf["4h__slope_4h"].to_numpy()[i],
        "ema_dist": s * (ci - ema15.to_numpy()[i]) / at,
        "range_pos": np.where(s > 0, rpos.to_numpy()[i], 1 - rpos.to_numpy()[i]),
        "risk_atr": risk / at,
        "vol_rel": v.to_numpy()[i] / v.rolling(48).mean().to_numpy()[i],
        "shock": shock.to_numpy()[i],
        "side": trades["side"].to_numpy(),
        "trigger": trades["trigger"].to_numpy() if "trigger" in trades else "",
        "session": pd.cut(hr, [-1, 6, 12, 20, 24], labels=["asia", "europe", "us", "late"]).astype(str),
    })
    # how far the trade got, in R, before it ended -- descriptive, for "why did the losers lose"
    hi_a, lo_a = h.to_numpy(), lo.to_numpy()
    mfe = []
    for r, side, rk in zip(trades.itertuples(), s, risk):
        seg = slice(int(r.entry_i) + 1, int(r.exit_i) + 1)
        best = (hi_a[seg].max() - r.entry_price) if side > 0 else (r.entry_price - lo_a[seg].min())
        mfe.append(best / rk if rk > 0 and seg.stop > seg.start else 0.0)
    f["mfe_R"] = mfe
    return f


# ---------------------------------------------------------------- collect
def collect() -> pd.DataFrame:
    import nb21_config as CFG
    rows = []
    qdata.tf_frames = frames_4h_from_1h
    t0 = time.time()
    try:
        for q in TRAIN + FORWARD:
            lab = fvg_lab(window_cfg(CFG, q, DATA_END))
            base = lab.trades
            if not len(base):
                continue
            mv = lab.retrade(USE_TP_MOVE=True).trades.set_index("entry_time")
            fx = lab.retrade(USE_FVG_EXIT=True).trades.set_index("entry_time")
            free = {**CFG.COSTS, "fee_bps": 0, "slippage_bps": 0}
            z = lab.retrade(COSTS=free).trades.set_index("entry_time")
            f = features(lab, base)
            f.insert(0, "quarter", q)
            f.insert(1, "phase", "train" if q in TRAIN else "forward")
            f.insert(2, "entry_time", base["entry_time"].astype(str).to_numpy())
            f["R"] = base["R"].to_numpy()
            f["close_reason"] = base["close_reason"].to_numpy()
            et = base["entry_time"]
            f["R0"] = z["R"].reindex(et).to_numpy()          # the same trade at zero cost
            f["win"] = (base["R"].to_numpy() > 0).astype(float)
            f["cost_R"] = f["R0"] - f["R"]                    # what costs took, in R
            f["dR_move"] = (mv["R"].reindex(et).to_numpy() - base["R"].to_numpy())
            f["dR_fvg"] = (fx["R"].reindex(et).to_numpy() - base["R"].to_numpy())
            rows.append(f)
            print(f"{q}: {len(base)} trades  ({time.time() - t0:.0f}s)", flush=True)
    finally:
        qdata.tf_frames = _tf_frames
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(OUT, index=False)
    return out


# ---------------------------------------------------------------- judge
def univariate(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tr, fw = t[t.phase == "train"], t[t.phase == "forward"]
    for y in OUTCOMES:
        for x in NUMERIC:
            a = tr[[x, y]].dropna()
            b = fw[[x, y]].dropna()
            if y.startswith("dR"):   # an exit rule's effect only exists where it changed something
                a, b = a[a[y] != 0], b[b[y] != 0]
            if len(a) < 30 or len(b) < 30:
                continue
            ra, pa = spearmanr(a[x], a[y])
            rb, pb = spearmanr(b[x], b[y])
            cuts = a[x].quantile([1 / 3, 2 / 3]).to_numpy()
            lo_f, hi_f = b[b[x] <= cuts[0]][y].mean(), b[b[x] > cuts[1]][y].mean()
            rows.append(dict(outcome=y, feature=x, n_train=len(a), n_fwd=len(b),
                             rho_train=ra, p_train=pa, rho_fwd=rb, p_fwd=pb,
                             fwd_low_third=lo_f, fwd_high_third=hi_f,
                             holds=bool(pa < 0.05 and pb < 0.05 and np.sign(ra) == np.sign(rb))))
    return pd.DataFrame(rows)


def categorical(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for x in CATEGORICAL:
        for lvl, g in t.groupby(x):
            r = {"feature": x, "level": lvl}
            for ph in ("train", "forward"):
                gg = g[g.phase == ph]
                r[f"n_{ph}"] = len(gg)
                r[f"win%_{ph}"] = 100 * gg.win.mean()
                r[f"R0_{ph}"] = gg.R0.mean()
                r[f"dR_move_{ph}"] = gg.dR_move[gg.dR_move != 0].mean()
                r[f"dR_fvg_{ph}"] = gg.dR_fvg[gg.dR_fvg != 0].mean()
            rows.append(r)
    return pd.DataFrame(rows)


def model(t: pd.DataFrame, y: str, seed: int = 0) -> dict:
    """All features at once: fit on TRAIN, score on FORWARD. The honest 'everything together'."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = pd.get_dummies(t[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    X = X.fillna(X[t.phase == "train"].median())
    ok = t[y].notna() & ((t[y] != 0) if y.startswith("dR") else True)
    target = (t[y] > 0).astype(int)
    tr = (t.phase == "train") & ok
    fw = (t.phase == "forward") & ok
    m = GradientBoostingClassifier(max_depth=2, n_estimators=150, learning_rate=0.05,
                                   subsample=0.8, random_state=seed)
    m.fit(X[tr], target[tr])
    p_tr, p_fw = m.predict_proba(X[tr])[:, 1], m.predict_proba(X[fw])[:, 1]
    rng = np.random.default_rng(seed)
    yf = target[fw].to_numpy()
    boots = []
    for _ in range(1000):
        k = rng.integers(0, len(yf), len(yf))
        if yf[k].min() != yf[k].max():
            boots.append(roc_auc_score(yf[k], p_fw[k]))
    # THE POLICY: act only where the model is more confident than its own TRAIN median
    cut = np.median(p_tr)
    val = t.loc[fw, y].to_numpy()
    take = p_fw > cut
    imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(outcome=y, n_train=int(tr.sum()), n_fwd=int(fw.sum()),
                auc_train=roc_auc_score(target[tr], p_tr), auc_fwd=roc_auc_score(yf, p_fw),
                auc_fwd_lo=np.percentile(boots, 2.5), auc_fwd_hi=np.percentile(boots, 97.5),
                fwd_mean_all=val.mean(), fwd_mean_chosen=val[take].mean() if take.any() else np.nan,
                fwd_mean_rejected=val[~take].mean() if (~take).any() else np.nan,
                top_features=", ".join(imp.index[:4]))


def report(t: pd.DataFrame):
    pd.set_option("display.width", 200)
    tr, fw = t[t.phase == "train"], t[t.phase == "forward"]
    print(f"\n{len(t)} trades: {len(tr)} train, {len(fw)} forward.  "
          f"R/trade train {tr.R.mean():+.3f}, forward {fw.R.mean():+.3f}")

    print("\n--- WHY THE LOSERS LOST: how far a trade got before it ended (max favourable, in R)")
    lose = t[t.R < 0]
    for k in (0.5, 1.0, 2.0, 2.5):
        print(f"  losers that reached {k:>3}R first: {100 * (lose.mfe_R >= k).mean():5.1f}%")
    print(f"  winners: {len(t[t.R > 0])}, losers: {len(lose)}")

    u = univariate(t)
    print("\n--- ONE FEATURE AT A TIME (Spearman; a relationship COUNTS only if p<0.05 in both "
          "halves, same sign)")
    for y in OUTCOMES:
        g = u[u.outcome == y].sort_values("p_train")
        same = int((np.sign(g.rho_train) == np.sign(g.rho_fwd)).sum())
        print(f"\n  {y}: {int(g.holds.sum())} of {len(g)} hold (chance ~{len(g) * 0.05 * 0.05 / 2:.2f});"
              f"  same sign in both halves {same} of {len(g)} (chance {len(g) / 2:.0f})")
        print(g[["feature", "rho_train", "p_train", "rho_fwd", "p_fwd", "fwd_low_third",
                 "fwd_high_third", "holds"]].round(3).to_string(index=False))
    print("\n--- CATEGORIES (win %, zero-cost R; dR only where the rule changed the trade)")
    print(categorical(t).round(3).to_string(index=False))
    print("\n--- THE COST EFFECT, on its own: what costs took from each trade (R), by stop width")
    for x in ("risk_atr", "atr_pct"):
        r, p_ = spearmanr(t[x], t.cost_R, nan_policy="omit")
        cuts = t[x].quantile([1 / 3, 2 / 3]).to_numpy()
        print(f"  {x:9s} rho {r:+.2f}   costs per trade: tightest third "
              f"{t[t[x] <= cuts[0]].cost_R.mean():.3f}R   widest third "
              f"{t[t[x] > cuts[1]].cost_R.mean():.3f}R")
    print("\n--- THE ONE LEVER: skip trades whose stop is tight in PRICE terms (cut on train)")
    t = t.assign(risk_pct=t.risk_atr * t.atr_pct)
    cut = t[t.phase == "train"].risk_pct.quantile(1 / 3)
    for ph in ("train", "forward"):
        d = t[t.phase == ph]
        k = d[d.risk_pct > cut]
        print(f"  {ph:8s} all {len(d):4d} trades {d.R.mean():+.3f}R/trade   stop > {cut:.2f}%: "
              f"{len(k):4d} trades {k.R.mean():+.3f}R/trade   (gross {d.R0.mean():+.3f} -> "
              f"{k.R0.mean():+.3f}, costs {d.cost_R.mean():.3f} -> {k.cost_R.mean():.3f})")
    per = t.groupby("quarter")[["R", "risk_pct", "phase"]].apply(
        lambda d: pd.Series({"better": d[d.risk_pct > cut].R.sum() > d.R.sum(),
                             "phase": d.phase.iloc[0]}))
    print(f"  total R better in {int(per.better.sum())} of {len(per)} quarters "
          f"({int(per[per.phase == 'forward'].better.sum())} of "
          f"{int((per.phase == 'forward').sum())} forward)")
    print("\n--- ALL FEATURES TOGETHER (fit on train, scored on forward)")
    mm = pd.DataFrame([model(t, y) for y in ("win", "dR_move", "dR_fvg")])
    print(mm.round(3).to_string(index=False))
    return u, mm


if __name__ == "__main__":
    t = pd.read_csv(OUT) if "--report-only" in sys.argv else collect()
    report(t)
