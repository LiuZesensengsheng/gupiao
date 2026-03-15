from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Iterable

import pandas as pd

from src.application.v2_contracts import InfoItem


@dataclass(frozen=True)
class InfoManifestDependencies:
    sha256_file: Callable[[object], str]
    stable_json_hash: Callable[[object], str]


def info_source_breakdown(items: Iterable[InfoItem]) -> dict[str, int]:
    counts = {
        "market_news": 0,
        "announcements": 0,
        "research": 0,
    }
    for item in items:
        subset = str(getattr(item, "source_subset", "")).strip()
        if subset in counts:
            counts[subset] = int(counts[subset] + 1)
    return counts


def build_info_manifest_payload(
    *,
    settings: dict[str, object],
    info_file: str,
    info_items: list[InfoItem],
    as_of_date: pd.Timestamp,
    config_hash: str,
    shadow_enabled: bool,
    shadow_report: dict[str, object] | None = None,
    deps: InfoManifestDependencies,
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for item in info_items:
        counts[item.info_type] = int(counts.get(item.info_type, 0) + 1)
    source_breakdown = info_source_breakdown(info_items)
    date_window = {
        "start": "",
        "end": "",
    }
    if info_items:
        date_window = {
            "start": str(min(item.date for item in info_items)),
            "end": str(max(item.date for item in info_items)),
        }
    info_hash = deps.sha256_file(info_file) if info_file else ""
    if not info_hash:
        info_hash = deps.stable_json_hash([asdict(item) for item in info_items])
    return {
        "info_file": str(info_file),
        "info_hash": str(info_hash),
        "info_item_count": int(len(info_items)),
        "info_type_counts": counts,
        "info_source_breakdown": source_breakdown,
        "market_news_count": int(source_breakdown.get("market_news", 0)),
        "announcement_count": int(source_breakdown.get("announcements", 0)),
        "research_count": int(source_breakdown.get("research", 0)),
        "date_window": date_window,
        "coverage_summary": {} if shadow_report is None else dict(shadow_report.get("coverage_summary", {})),
        "market_coverage_ratio": float(
            (shadow_report or {}).get("coverage_summary", {}).get("market_coverage_ratio", 0.0)
        ),
        "stock_coverage_ratio": float(
            (shadow_report or {}).get("coverage_summary", {}).get("stock_coverage_ratio", 0.0)
        ),
        "config_hash": str(config_hash),
        "info_shadow_enabled": bool(shadow_enabled),
        "info_shadow_only": bool(settings.get("info_shadow_only", True)),
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "info_types": [str(item) for item in settings.get("info_types", [])],
        "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
        "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
        "as_of_date": str(as_of_date.date()),
    }
