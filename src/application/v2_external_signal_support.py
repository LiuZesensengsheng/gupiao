from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from src.application.v2_contracts import CompositeState
from src.infrastructure.external_signal_features import (
    build_capital_flow_state,
    build_external_signal_manifest,
    build_macro_context_state,
)


def attach_external_signals_to_state(
    *,
    state: CompositeState,
    capital_flow_state: object,
    macro_context_state: object,
) -> CompositeState:
    return replace(
        state,
        capital_flow_state=capital_flow_state,
        macro_context_state=macro_context_state,
    )


def build_external_signal_package(
    *,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: list[object],
) -> dict[str, object]:
    capital_flow_state = build_capital_flow_state(
        as_of_date=as_of_date,
        capital_flow_file=settings.get("capital_flow_file"),
        lookback_days=int(settings.get("capital_flow_lookback_days", 20)),
        info_items=info_items,
    )
    macro_context_state = build_macro_context_state(
        as_of_date=as_of_date,
        macro_file=settings.get("macro_file"),
        lookback_days=int(settings.get("macro_lookback_days", 60)),
        info_items=info_items,
    )
    manifest = build_external_signal_manifest(
        settings=settings,
        as_of_date=as_of_date,
        info_items=info_items,
        capital_flow_state=capital_flow_state,
        macro_context_state=macro_context_state,
    )
    return {
        "capital_flow_state": capital_flow_state,
        "macro_context_state": macro_context_state,
        "capital_flow_snapshot": asdict(capital_flow_state),
        "macro_context_snapshot": asdict(macro_context_state),
        "manifest": manifest,
    }


def merge_external_signal_manifest_summary(
    *,
    info_manifest: dict[str, object],
    external_signal_manifest: dict[str, object],
) -> dict[str, object]:
    merged = dict(info_manifest)
    coverage = external_signal_manifest.get("coverage", {})
    if isinstance(coverage, dict):
        merged["external_event_item_count"] = int(coverage.get("event_item_count", 0))
        merged["external_publisher_count"] = int(coverage.get("publisher_count", 0))
        merged["external_source_subset_count"] = int(coverage.get("source_subset_count", 0))
        merged["external_event_tag_count"] = int(coverage.get("event_tag_count", 0))
    merged["capital_flow_snapshot"] = dict(external_signal_manifest.get("capital_flow_snapshot", {}))
    merged["macro_context_snapshot"] = dict(external_signal_manifest.get("macro_context_snapshot", {}))
    merged["external_signal_version"] = str(external_signal_manifest.get("external_signal_version", ""))
    merged["external_signal_enabled"] = bool(external_signal_manifest.get("external_signal_enabled", False))
    return merged


def ensure_external_signal_manifest_path(run_dir: Path) -> Path:
    return run_dir / "external_signal_manifest.json"
