"""
Copy exness/trendlines.py's trendline FITTER into exness/trendlines_mtf.py (its `Lines` secondary
context), verbatim but for the three drawing calls a secondary context cannot make.

Indie has no imports between indicator files, so the multi-timeframe indicator carries its own
copy of the fitter. Run this after changing the fitter in exness/trendlines.py:

    python scripts/sync_exness_fitter.py

tests/test_exness_trendlines_mtf.py fails while the two copies differ.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SINGLE = ROOT / "exness" / "trendlines.py"
MTF = ROOT / "exness" / "trendlines_mtf.py"
FIT = ["_atr", "_zigzag", "_add_pivot", "_price", "_clear", "_rowsum", "_fit_side", "_better",
       "_take", "_retire", "_publish"]
CUT = [("_take", "                if self._seg_on.get()[s]:\n"
                 "                    self._place(s, self._seg_tip.get()[s])    # its whole span"
                 " moves with it\n"),
       ("_publish", "                    self._draw_live(s, i)\n")]
START = "    def _atr(self) -> float:\n"          # the copy runs from here ...
END = "\n\n@indicator("                            # ... to the chart's indicator


def _method(src: str, name: str) -> str:
    m = re.search(rf"\n(    def {name}\(.*?)(?=\n    def |\n    # ── |\Z)", src, re.S)
    if not m:
        raise ValueError(f"{name} not found in {SINGLE.name}")
    return m.group(1).rstrip() + "\n"


def fitter_block(single_src: str) -> str:
    """The fitter as trendlines_mtf.py must carry it."""
    src = single_src.replace("\r\n", "\n")
    body = {n: _method(src, n) for n in FIT}
    for name, line in CUT:
        if body[name].count(line) != 1:
            raise ValueError(f"the drawing call in {name} changed: update CUT in {__file__}")
        body[name] = body[name].replace(line, "")
    return "\n".join(body[n] for n in FIT)


def current_block(mtf_src: str) -> str:
    src = mtf_src.replace("\r\n", "\n")
    a = src.index(START)
    return src[a:src.index(END, a)].rstrip() + "\n"


def main() -> int:
    want = fitter_block(SINGLE.read_text(encoding="utf-8"))
    src = MTF.read_text(encoding="utf-8").replace("\r\n", "\n")
    a = src.index(START)
    b = src.index(END, a)
    if src[a:b].rstrip() + "\n" == want:
        print("already in sync")
        return 0
    MTF.write_text(src[:a] + want.rstrip() + src[b:], encoding="utf-8", newline="\n")
    print(f"copied the fitter into {MTF.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
