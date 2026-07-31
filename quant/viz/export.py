"""
Chart export / display helpers.

These live here (not in notebooks) so a research notebook stays a sequence of declarations:
data -> indicators -> rules -> run -> view, with no helper definitions cluttering the narrative.
"""
from __future__ import annotations

import webbrowser
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from ..indicators import add_heikin_ashi


def heikin_ashi_candles(df: pd.DataFrame, *, prefix: str = "ha") -> pd.DataFrame:
    """Return a frame whose OHLC columns ARE the Heikin Ashi values (real OHLC kept as `real_*`).

    Feed this to `ResearchChart.add_candle_alt(...)` to get the on-plot Real/Heikin-Ashi toggle.
    """
    out = add_heikin_ashi(df, prefix=prefix)
    return out.rename(columns={
        "open": "real_open", "high": "real_high", "low": "real_low", "close": "real_close",
        f"{prefix}_open": "open", f"{prefix}_high": "high",
        f"{prefix}_low": "low", f"{prefix}_close": "close",
    })


def open_in_browser(fig, filename: Union[str, Path] = "chart.html", *, show: bool = True,
                    extra_js: Optional[str] = None) -> Path:
    """Write `fig` to a self-contained HTML file and open it in a new browser tab.

    Reliable across notebook front-ends whose inline renderer won't draw candlestick traces.
    `extra_js` (e.g. `quant.viz.CROSSHAIR_JS`) is injected as a <script> after the plot — used to
    add a full-height cursor crosshair spanning all subplot panes. Returns the path.
    """
    out = Path(filename).resolve()
    fig.write_html(str(out), include_plotlyjs=True, full_html=True)
    if extra_js:
        html = out.read_text(encoding="utf-8")
        html = html.replace("</body>", f"<script>{extra_js}</script></body>", 1)
        out.write_text(html, encoding="utf-8")
    if show:
        webbrowser.open(out.as_uri())
    return out
