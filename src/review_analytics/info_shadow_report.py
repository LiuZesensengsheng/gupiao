from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from src.application.v2_contracts import InfoItem


@dataclass(frozen=True)
class InfoShadowReportDependencies:
    build_info_shadow_variant: Callable[..., dict[str, object]]
    filter_info_items_by_source_subset: Callable[[Iterable[InfoItem], str], list[InfoItem]]
    event_tag_counts: Callable[[Iterable[InfoItem]], dict[str, int]]
    info_source_breakdown: Callable[[Iterable[InfoItem]], dict[str, int]]


def build_info_shadow_report(
    *,
    validation_trajectory: Any,
    holdout_trajectory: Any,
    settings: dict[str, object],
    info_items: list[InfoItem],
    deps: InfoShadowReportDependencies,
) -> dict[str, object]:
    all_variant = deps.build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=info_items,
    )
    market_news_variant = deps.build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=deps.filter_info_items_by_source_subset(info_items, "market_news"),
    )
    announcement_variant = deps.build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=deps.filter_info_items_by_source_subset(info_items, "announcements"),
    )
    research_variant = deps.build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=deps.filter_info_items_by_source_subset(info_items, "research"),
    )
    holdout_steps = list(getattr(holdout_trajectory, "steps", []) or [])
    return {
        "info_shadow_enabled": bool(settings.get("use_info_fusion", False)),
        "shadow_only": bool(settings.get("info_shadow_only", True)),
        "market_shadow_modes": dict(all_variant.get("market_shadow_modes", {})),
        "stock_shadow_modes": dict(all_variant.get("stock_shadow_modes", {})),
        "model_samples": dict(all_variant.get("model_samples", {})),
        "quant_only": {
            "avg_20d_rank_ic": float(np.mean([float(step.horizon_metrics["20d"]["rank_ic"]) for step in holdout_steps])) if holdout_steps else 0.0,
            "avg_20d_top_bottom_spread": float(np.mean([float(step.horizon_metrics["20d"]["top_bottom_spread"]) for step in holdout_steps])) if holdout_steps else 0.0,
            "event_day_hit_rate": float(all_variant.get("quant_event_day_hit_rate", 0.0)),
        },
        "quant_plus_info_shadow": {
            key: value
            for key, value in all_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "market_news_only": {
            key: value
            for key, value in market_news_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "announcements_only": {
            key: value
            for key, value in announcement_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "research_only": {
            key: value
            for key, value in research_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "all_info_combined": {
            key: value
            for key, value in all_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "coverage_summary": dict(all_variant.get("coverage_summary", {})),
        "top_positive_stock_deltas": list(all_variant.get("top_positive_stock_deltas", [])),
        "top_negative_stock_deltas": list(all_variant.get("top_negative_stock_deltas", [])),
        "event_tag_distribution": deps.event_tag_counts(info_items),
        "info_source_breakdown": deps.info_source_breakdown(info_items),
        "last_market_info_state": dict(all_variant.get("last_market_info_state", {})),
        "last_date": str(all_variant.get("last_date", "")),
    }
