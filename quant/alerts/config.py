"""
The alerts config: read fresh from `alerts/alerts_config.py` on every tick, checked before use.

Read FRESH (runpy, not import), so an edit takes effect on the next tick without restarting the
monitor -- and a stale module can never answer instead of the file on disk. Every mistake a config
can hold (an unknown market, timeframe or pattern, a group that names nothing) is refused here with
a message that names it, rather than surfacing as an alert that silently never fires.
"""
from __future__ import annotations

import runpy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "alerts" / "alerts_config.py"
DEFAULT_STATE = ROOT / "alerts" / "state"

BASE_TF = "5m"                                   # every market is stored at 5m; the rest is resampled
TF_MINUTES = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240}
SOURCES = ("binance", "dukascopy")
PRIORITIES = ("urgent", "normal")


@dataclass(frozen=True)
class Market:
    name: str
    symbol: str
    source: str
    tv: str                                      # TradingView symbol, for the link in the email


@dataclass(frozen=True)
class Alert:
    name: str
    category: str
    kind: str
    markets: Tuple[str, ...]
    tfs: Tuple[str, ...]
    priority: str = "normal"
    cooldown_bars: int = 0                       # no repeat for this market+tf within N of its bars
    params: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Delivery:
    digest_minutes: int = 15                     # normal alerts: at most one email this often
    catchup_max_age_minutes: int = 30            # older than this when found -> "while offline"
    close_grace_minutes: int = 20                # a bar the feed still has not delivered this long
                                                 # after it closed is taken as never coming (market
                                                 # shut). Longer is safer: a feed that is merely LATE
                                                 # would otherwise have a half-built candle judged
    settle_seconds: int = 90                     # judge a bar this long after it closes: Dukascopy
                                                 # lags ~1 min, and waiting the same for every market
                                                 # puts one bar close's alerts in ONE email
    subject_prefix: str = "[Alerts]"


@dataclass(frozen=True)
class AlertsConfig:
    markets: Dict[str, Market]
    alerts: Tuple[Alert, ...]
    delivery: Delivery
    categories: Tuple[str, ...]                  # the order categories appear in an email
    path: Path
    off: Tuple[Alert, ...] = ()                  # alerts written but switched off (on=False)

    def pairs(self):
        """Every (market, tf) some alert watches, and the alerts on each."""
        out: Dict[Tuple[str, str], list] = {}
        for a in self.alerts:
            for m in a.markets:
                for tf in a.tfs:
                    out.setdefault((m, tf), []).append(a)
        return out


class ConfigError(ValueError):
    pass


def _tuple(v):
    if v is None:
        return ()
    return (v,) if isinstance(v, str) else tuple(v)


def _market(name, m) -> Market:
    src = m.get("source")
    if src not in SOURCES:
        raise ConfigError(f"market {name!r}: source must be one of {SOURCES}, got {src!r}")
    sym = str(m.get("symbol", "")).strip()
    if not sym:
        raise ConfigError(f"market {name!r} has no symbol")
    tv = m.get("tv")
    if src == "binance":
        sym = sym.upper()
        if not sym.isalnum():
            raise ConfigError(f"market {name!r}: {sym!r} is not a Binance symbol like BTCUSDT")
        tv = tv or f"BINANCE:{sym}"
    else:
        from quant.data.dukascopy import _instrument, dukascopy_instruments
        try:
            code = _instrument(sym)
        except ValueError:
            code = None
        group = dukascopy_instruments().get(code)
        if group is None:
            raise ConfigError(f"market {name!r}: Dukascopy has no instrument {sym!r}; "
                              "see `python -m quant.alerts list`")
        if not tv and group == "FX":                 # FX and metals chart on OANDA; others
            tv = "OANDA:" + code.replace("/", "")      # need tv= to get a chart link
    return Market(name, sym, src, tv or "")


def load_config(path: Path = DEFAULT_CONFIG) -> AlertsConfig:
    from .detectors import CATEGORY_ORDER, REGISTRY
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no alerts config at {path}")
    try:
        ns = runpy.run_path(str(path))
    except Exception as e:                           # a typo saved mid-edit
        raise ConfigError(f"{path.name} does not run: {type(e).__name__}: {e}") from None

    markets = {name: _market(name, m) for name, m in (ns.get("MARKETS") or {}).items()}
    if not markets:
        raise ConfigError("MARKETS is empty -- nothing to watch")

    groups = {"all": tuple(markets)}
    for g, members in (ns.get("GROUPS") or {}).items():
        bad = [x for x in _tuple(members) if x not in markets]
        if bad:
            raise ConfigError(f"group {g!r} names unknown markets {bad}")
        groups[g] = _tuple(members)

    alerts, off, names = [], [], set()
    for i, a in enumerate(ns.get("ALERTS") or []):
        a = dict(a)
        name = a.pop("name", f"alert {i + 1}")
        if name in names:                            # the name is the alert's identity in the
            raise ConfigError(f"two alerts are named {name!r}")      # state: it must be unique
        names.add(name)
        on = a.pop("on", True)
        kind = a.pop("kind", None)
        if kind not in REGISTRY:
            raise ConfigError(f"alert {name!r}: kind must be one of {sorted(REGISTRY)}, got {kind!r}")
        ms = []
        for x in _tuple(a.pop("markets", "all")):
            if x in groups:
                ms += list(groups[x])
            elif x in markets:
                ms.append(x)
            else:
                raise ConfigError(f"alert {name!r}: {x!r} is neither a market nor a group")
        tfs = _tuple(a.pop("tfs", ()))
        bad = [t for t in tfs if t not in TF_MINUTES]
        if not tfs or bad:
            raise ConfigError(f"alert {name!r}: tfs must be some of {list(TF_MINUTES)}, got {tfs}")
        prio = a.pop("priority", "normal")
        if prio not in PRIORITIES:
            raise ConfigError(f"alert {name!r}: priority must be one of {PRIORITIES}")
        alert = Alert(name=name, category=a.pop("category", None) or REGISTRY[kind].category,
                      kind=kind, markets=tuple(dict.fromkeys(ms)), tfs=tfs, priority=prio,
                      cooldown_bars=int(a.pop("cooldown_bars", 0)), params=a)
        try:                                         # checked even when OFF, so switching one
            REGISTRY[kind].validate(alert.params)    # on can never be the moment it breaks
        except (KeyError, ValueError) as e:
            raise ConfigError(f"alert {name!r}: {e}") from None
        (alerts if on else off).append(alert)

    delivery = Delivery(**(ns.get("DELIVERY") or {}))
    cats = list(_tuple(ns.get("CATEGORIES")))
    used = {a.category for a in alerts}
    cats += [c for c in CATEGORY_ORDER if c in used and c not in cats]
    cats += sorted(used - set(cats))
    return AlertsConfig(markets=markets, alerts=tuple(alerts), delivery=delivery,
                        categories=tuple(cats), path=path, off=tuple(off))
