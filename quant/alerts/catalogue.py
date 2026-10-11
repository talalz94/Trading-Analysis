"""
The menu: every instrument, timeframe, pattern, alert kind and setup an alert can use -- built from
the registries themselves, so it is always exactly what the code accepts.

    python -m quant.alerts list                 print it
    python -m quant.alerts list --write-docs    rewrite docs/ALERTS_CATALOGUE.md from it
"""
from __future__ import annotations

from collections import Counter

from .config import ROOT, TF_MINUTES
from .detectors import CATEGORY_ORDER, REGISTRY, setup_keys

DOC = ROOT / "docs" / "ALERTS_CATALOGUE.md"

BINANCE_EXAMPLES = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
                    "PAXGUSDT (gold-backed token)", "XAUTUSDT (gold-backed token)")
_GROUP_NAMES = {"FX": "FX pairs and spot metals", "VCCY": "Crypto (CFDs)",
                "IDX": "Stock indices", "CMD": "Commodities", "BND": "Bonds", "ETF": "ETFs"}


def _by_direction(names, direction_of):
    out = {"Bullish": [], "Bearish": [], "Either way": []}
    for n in names:
        out[{1: "Bullish", -1: "Bearish"}.get(direction_of[n], "Either way")].append(n)
    return out


def catalogue(markdown: bool = False) -> str:
    from quant.data.dukascopy import _SYMBOL_MAP, dukascopy_instruments
    from quant.indicators import (CANDLE_PATTERNS, CANDLE_PATTERN_DIRECTION, CHART_PATTERNS,
                                  CHART_PATTERN_DIRECTION)
    from quant.scenarios.base import SCENARIOS

    out = []
    h1 = (lambda s: out.extend([f"## {s}", ""])) if markdown else \
         (lambda s: out.extend(["", s.upper(), "=" * len(s)]))
    h2 = (lambda s: out.extend([f"### {s}", ""])) if markdown else \
         (lambda s: out.extend(["", s, "-" * len(s)]))
    code = (lambda s: f"`{s}`") if markdown else (lambda s: s)

    def para(s):
        out.extend([s, ""] if markdown else [s])

    def words(items, per_line=8):
        items = [code(i) for i in items]
        if markdown:
            out.extend([", ".join(items), ""])
        else:
            for k in range(0, len(items), per_line):
                out.append("  " + ", ".join(items[k:k + per_line]))

    if markdown:
        out += ["# Alerts catalogue: everything an alert can be set on", "",
                "Generated from the code by `python -m quant.alerts list --write-docs`, so it "
                "lists exactly what `alerts/alerts_config.py` accepts. Do not edit it by hand. "
                "How to use it is in [ALERTS.md](ALERTS.md).", "",
                "> Nothing here has a measured trading edge. The candle and chart patterns were "
                "tested and are no better than a coin flip (docs/WHAT_WORKS.md). An alert says "
                "*look at this chart*, not *trade it*.", ""]
    else:
        out += ["Everything an alert can be set on (docs/ALERTS_CATALOGUE.md has the same).",
                "None of it has a measured edge: an alert says 'look', not 'trade'."]

    # ── 1. instruments ──────────────────────────────────────────────────────────────────────
    h1("1. Instruments (MARKETS)")
    para('source="binance": any Binance SPOT symbol. A wrong one shows as a feed error in '
         '`python -m quant.alerts status`. For example:')
    words(BINANCE_EXAMPLES, 5)
    duk = dukascopy_instruments()
    para("")
    para(f'source="dukascopy": {len(duk):,} instruments, priced on the BID side. Use a short '
         "name below, any 6-letter FX pair, or Dukascopy's own code exactly as listed.")
    h2("Short names")
    rows = [f"{k} = {v}" for k, v in _SYMBOL_MAP.items()]
    words(rows, 4)
    groups = Counter(duk.values())
    for g in ("FX", "VCCY", "IDX", "CMD", "BND"):
        codes = sorted(c for c, grp in duk.items() if grp == g)
        h2(f"{_GROUP_NAMES[g]} ({len(codes)})")
        words([c.replace("/", "") if g == "FX" else c for c in codes], 10)
    stocks = {g: n for g, n in groups.items() if g not in _GROUP_NAMES}
    h2(f"ETFs ({groups.get('ETF', 0)}) and single stocks ({sum(stocks.values()):,})")
    nice = {"US": "US", "UK": "UK", "HONG": "Hong Kong"}
    by_market = ", ".join(f"{nice.get(g, g.title())} {n}" for g, n in
                          sorted(stocks.items(), key=lambda x: (-x[1], x[0])))
    para("Too many to list. Use Dukascopy's own code, e.g. "
         + ", ".join(code(c) for c in ("AAPL.US/USD", "TSLA.US/USD", "BMW.DE/EUR"))
         + f". Stocks by market: {by_market}.")

    # ── 2. timeframes ───────────────────────────────────────────────────────────────────────
    h1("2. Timeframes (tfs)")
    words(list(TF_MINUTES), 10)
    para("")
    para("Every market is fetched at 5m and the rest are built from it. Candles start on UTC "
         "boundaries: 4h candles open at 00, 04, 08 ... UTC.")

    # ── 3/4. patterns ───────────────────────────────────────────────────────────────────────
    for title, kind, names, direction in (
            ("3. Candlestick patterns", "candle", CANDLE_PATTERNS, CANDLE_PATTERN_DIRECTION),
            ("4. Chart patterns", "chart", CHART_PATTERNS, CHART_PATTERN_DIRECTION)):
        h1(f"{title} ({len(names)}), for kind=\"{kind}\"")
        para('Use the names in patterns=(...), or patterns="all" / "bullish" / "bearish".')
        for label, group in _by_direction(names, direction).items():
            if group:
                h2(f"{label} ({len(group)})")
                words(group, 6)

    # ── 5. kinds ────────────────────────────────────────────────────────────────────────────
    h1(f"5. Alert kinds ({len(REGISTRY)})")
    para("kind= picks what to look for. Each kind takes only its own settings, and any other "
         "setting is refused by name. direction= is always both | up | down.")
    for cat in CATEGORY_ORDER:
        dets = [d for d in REGISTRY.values() if d.category == cat]
        if not dets:
            continue
        h2(cat)
        for d in dets:
            ex = ", ".join(f"{k}={v!r}" for k, v in d.example.items())
            line = (f'dict(on=True, name="...", kind="{d.kind}"{", " + ex if ex else ""}, '
                    f'markets="all", tfs="1h")')
            if markdown:
                out += [f"**`{d.kind}`**: {d.summary}", ""]
                out += ["| setting | default | meaning |", "|---|---|---|"]
                out += [f"| `{k}` | `{dv!r}` | {txt.replace('|', chr(92) + '|')} |"   # a | in a
                        for k, (dv, txt) in d.params.items()]                         # cell splits it
                out += ["", f"```python\n{line}\n```", ""]
            else:
                out += ["", f"  {d.kind}: {d.summary}"]
                for k, (dv, txt) in d.params.items():
                    out.append(f"      {k:<13} = {dv!r:<10} {txt}")
                out.append(f"      e.g. {line}")

    # ── 6. setups ───────────────────────────────────────────────────────────────────────────
    keys = setup_keys()
    h1(f"6. Setups ({len(keys)}), for kind=\"setup\"")
    para("The research entry setups in quant/scenarios. An alert fires on the bar the "
         "backtest would have entered, once per setup episode. Change a setup's own "
         "settings with options={...}, e.g. "
         + code('dict(kind="setup", setup="rsi_div_long@v1", options=dict(div_right=3))')
         + ". Left out: time_entry (the metronome benchmark) and mtf_trendline, which needs "
         "several timeframes' lines at once.")
    if markdown:
        out += ["| setup | side | what it is |", "|---|---|---|"]
        out += [f"| `{k}` | {SCENARIOS[k].side} | {SCENARIOS[k].description.split('. ')[0]} |"
                for k in keys]
        out.append("")
    else:
        for k in keys:
            out.append(f"  {k:<28} {SCENARIOS[k].side:<5}  "
                       f"{SCENARIOS[k].description.split('. ')[0][:90]}")
    return "\n".join(out).strip() + "\n"


def write_docs() -> str:
    DOC.write_text(catalogue(markdown=True), encoding="utf-8")
    return str(DOC)
