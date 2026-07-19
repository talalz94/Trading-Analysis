"""
Backward-compatible config shim.

Secrets now live in `.env` (git-ignored), NOT in this file. This module loads them
from the environment so existing `import config; config.api_key` usage keeps working.

Setup:
  1. Copy .env.example -> .env and fill in BINANCE_API_KEY / BINANCE_API_SECRET.
  2. (Recommended) Rotate the old keys in your Binance account, since they were
     previously committed in plaintext.

Note: public historical klines (data.py) do NOT require these keys; they are only
used for authenticated endpoints (e.g. the live-trading notebook).
"""
from __future__ import annotations

import os

# Load .env if python-dotenv is available (optional dependency).
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # Minimal fallback: parse a local .env by hand so this works without dotenv.
    try:
        from pathlib import Path
        _env = Path(__file__).with_name(".env")
        if _env.exists():
            for _line in _env.read_text(encoding="utf-8").splitlines():
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

api_key = os.environ.get("BINANCE_API_KEY", "")
api_secret = os.environ.get("BINANCE_API_SECRET", "")
