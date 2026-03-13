from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_universe_generator_cache_key(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def universe_generator_cache_dir(cache_root: str | Path) -> Path:
    path = Path(cache_root) / "universe_generator"
    path.mkdir(parents=True, exist_ok=True)
    return path


def universe_generator_cache_path(cache_root: str | Path, cache_key: str) -> Path:
    return universe_generator_cache_dir(cache_root) / f"{cache_key}.json"


def load_universe_generator_cache(cache_root: str | Path, cache_key: str) -> dict[str, Any] | None:
    path = universe_generator_cache_path(cache_root, cache_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def store_universe_generator_cache(cache_root: str | Path, cache_key: str, payload: dict[str, Any]) -> Path:
    path = universe_generator_cache_path(cache_root, cache_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
