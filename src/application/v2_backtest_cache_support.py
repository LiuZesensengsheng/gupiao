from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Mapping


def _file_mtime_token(path_like: object) -> int:
    try:
        path = Path(str(path_like))
    except Exception:
        return 0
    if not path.exists():
        return 0
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return 0


def build_prepared_backtest_cache_key(settings: Mapping[str, object]) -> str:
    payload = {
        "version": "v2-prepared-cache-2",
        "config_path": str(Path(str(settings.get("config_path", ""))).resolve()),
        "source": str(settings.get("source", "")),
        "data_dir": str(Path(str(settings.get("data_dir", ""))).resolve()),
        "watchlist": str(Path(str(settings.get("watchlist", ""))).resolve()),
        "watchlist_mtime": _file_mtime_token(settings.get("watchlist", "")),
        "universe_file": str(Path(str(settings.get("universe_file", ""))).resolve()),
        "universe_mtime": _file_mtime_token(settings.get("universe_file", "")),
        "source_universe_manifest_path": str(settings.get("source_universe_manifest_path", "")),
        "source_universe_manifest_mtime": _file_mtime_token(settings.get("source_universe_manifest_path", "")),
        "universe_limit": int(settings.get("universe_limit", 0)),
        "universe_tier": str(settings.get("universe_tier", "")),
        "start": str(settings.get("start", "")),
        "end": str(settings.get("end", "")),
        "min_train_days": int(settings.get("min_train_days", 0)),
        "use_margin_features": bool(settings.get("use_margin_features", False)),
        "margin_market_file": str(Path(str(settings.get("margin_market_file", ""))).resolve()),
        "margin_market_mtime": _file_mtime_token(settings.get("margin_market_file", "")),
        "margin_stock_file": str(Path(str(settings.get("margin_stock_file", ""))).resolve()),
        "margin_stock_mtime": _file_mtime_token(settings.get("margin_stock_file", "")),
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
        "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
        "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def prepared_backtest_cache_path(*, cache_root: str, cache_key: str) -> Path:
    root = Path(str(cache_root)) / "prepared"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{cache_key}.pkl"


def load_pickle_cache(path: str | Path) -> object | None:
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None


def store_pickle_cache(path: str | Path, payload: object) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
