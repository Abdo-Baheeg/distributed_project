"""Load `.env` early in process (no secrets committed; `.env` is gitignored)."""

from __future__ import annotations

from pathlib import Path


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env", override=False)
