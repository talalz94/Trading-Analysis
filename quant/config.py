"""
Global settings and path resolution.

Uses stdlib dataclasses (no hard pydantic dependency for the core). Secrets come from
the environment / .env only — never hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv_if_present() -> None:
    """Load .env into os.environ if python-dotenv is available, else parse manually."""
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return
    except Exception:
        pass
    env = _repo_root() / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _repo_root() -> Path:
    # quant/ lives directly under the repo root.
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    repo_root: Path = field(default_factory=_repo_root)
    data_dir: Path = field(default_factory=lambda: _repo_root() / "data")
    cache_dir: Path = field(default_factory=lambda: _repo_root() / "quant_cache")

    display_tz: str = "UTC"
    log_level: str = "INFO"

    @property
    def binance_api_key(self) -> str:
        return os.environ.get("BINANCE_API_KEY", "")

    @property
    def binance_api_secret(self) -> str:
        return os.environ.get("BINANCE_API_SECRET", "")


_load_dotenv_if_present()
SETTINGS = Settings()
