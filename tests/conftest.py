"""Put the repo root on sys.path so both `quant` and legacy modules import in tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
