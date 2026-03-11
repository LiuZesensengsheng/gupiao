from __future__ import annotations

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
