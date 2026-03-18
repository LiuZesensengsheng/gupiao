from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.application.v2_contracts import DailyRunResult
from src.infrastructure.strategy_memory import remember_daily_run


@dataclass(frozen=True)
class DailySnapshotContext:
    settings: dict[str, object]
    manifest: dict[str, object]
    manifest_path: Path | None
    snapshot: object
    resolved_run_id: str


@dataclass(frozen=True)
class DailyUniverseContext:
    market_security: object
    current_holdings: list[object]
    stocks: list[object]
    sector_map: dict[str, str]


@dataclass(frozen=True)
class DailyCacheKeyDependencies:
    resolve_manifest_path: Callable[..., Path]
    resolve_info_file_from_settings: Callable[[dict[str, object]], str]


def file_mtime_token(path_like: object) -> int:
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


def daily_result_cache_key(
    *,
    strategy_id: str,
    settings: dict[str, object],
    artifact_root: str,
    run_id: str = "",
    snapshot_path: str = "",
    allow_retrain: bool = False,
    disable_learned_policy: bool = False,
    deps: DailyCacheKeyDependencies,
) -> str:
    policy_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    manifest_path = deps.resolve_manifest_path(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        run_id=run_id,
        snapshot_path=snapshot_path,
    )
    info_file = deps.resolve_info_file_from_settings(settings)
    payload = {
        "version": "v2-daily-cache-5",
        "strategy_id": str(strategy_id),
        "config_path": str(Path(str(settings.get("config_path", ""))).resolve()),
        "source": str(settings.get("source", "")),
        "watchlist": str(Path(str(settings.get("watchlist", ""))).resolve()),
        "watchlist_mtime": file_mtime_token(settings.get("watchlist", "")),
        "universe_file": str(Path(str(settings.get("universe_file", ""))).resolve()),
        "universe_mtime": file_mtime_token(settings.get("universe_file", "")),
        "universe_limit": int(settings.get("universe_limit", 0)),
        "universe_tier": str(settings.get("universe_tier", "")),
        "source_universe_manifest_path": str(settings.get("source_universe_manifest_path", "")),
        "source_universe_manifest_mtime": file_mtime_token(settings.get("source_universe_manifest_path", "")),
        "start": str(settings.get("start", "")),
        "end": str(settings.get("end", "")),
        "min_train_days": int(settings.get("min_train_days", 0)),
        "step_days": int(settings.get("step_days", 0)),
        "l2": float(settings.get("l2", 0.0)),
        "max_positions": int(settings.get("max_positions", 0)),
        "use_margin_features": bool(settings.get("use_margin_features", False)),
        "margin_market_file": str(settings.get("margin_market_file", "")),
        "margin_market_mtime": file_mtime_token(settings.get("margin_market_file", "")),
        "margin_stock_file": str(settings.get("margin_stock_file", "")),
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
        "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
        "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
        "margin_stock_mtime": file_mtime_token(settings.get("margin_stock_file", "")),
        "info_file": str(Path(str(info_file or "")).resolve()) if info_file else "",
        "info_file_mtime": file_mtime_token(info_file),
        "info_hash": str(settings.get("info_hash", "")),
        "use_info_fusion": bool(settings.get("use_info_fusion", False)),
        "info_shadow_only": bool(settings.get("info_shadow_only", True)),
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "info_types": [str(item) for item in settings.get("info_types", [])],
        "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
        "info_cutoff_time": str(settings.get("info_cutoff_time", "23:59:59")),
        "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
        "published_policy_path": str(policy_path.resolve()),
        "published_policy_mtime": file_mtime_token(policy_path),
        "run_id": str(run_id),
        "snapshot_path": str(snapshot_path),
        "allow_retrain": bool(allow_retrain),
        "disable_learned_policy": bool(disable_learned_policy),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_mtime": file_mtime_token(manifest_path),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def daily_result_cache_path(
    *,
    cache_root: str,
    cache_key: str,
) -> Path:
    root = Path(str(cache_root))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"daily_{cache_key}.pkl"


def load_daily_cached_result(
    *,
    cache_path: Path,
    refresh_cache: bool,
    memory_root: Path,
    emit_progress: Callable[[str, str], None],
) -> DailyRunResult | None:
    if not refresh_cache and cache_path.exists():
        emit_progress("daily", "命中日运行缓存，直接复用结果")
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, DailyRunResult):
                return remember_daily_run(
                    memory_root=memory_root,
                    result=cached,
                )
        except Exception:
            return None
        return None
    emit_progress("daily", "日运行缓存未命中，开始重建")
    return None


def build_daily_symbol_names(
    *,
    current_holdings: list[object],
    stocks: list[object],
    stock_rows: list[object],
    composite_state: object,
) -> dict[str, str]:
    symbol_names: dict[str, str] = {}
    for item in current_holdings:
        if getattr(item, "name", None):
            symbol_names[str(item.symbol)] = str(item.name)
    for item in stocks:
        if getattr(item, "name", None):
            symbol_names[str(item.symbol)] = str(item.name)
    for row in stock_rows:
        name = getattr(row, "name", "")
        if name:
            symbol_names[str(row.symbol)] = str(name)
    for state in getattr(composite_state, "stocks", []):
        symbol_names.setdefault(str(state.symbol), str(state.symbol))
    return symbol_names
