from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from src.application.v2_contracts import CompositeState, InfoItem


_REALIZED_COL_BY_HORIZON = {
    "1d": "realized_ret_1d",
    "2d": "realized_ret_2d",
    "3d": "realized_ret_3d",
    "5d": "realized_ret_5d",
    "10d": "realized_ret_10d",
    "20d": "realized_ret_20d",
}


@dataclass(frozen=True)
class InfoResearchDependencies:
    emit_progress_fn: Callable[[str, str], None]
    load_or_build_v2_backtest_trajectory_fn: Callable[..., Any]
    split_research_trajectory_fn: Callable[..., tuple[Any, Any, Any]]
    trajectory_step_count_fn: Callable[[Any], int]
    load_v2_runtime_settings_fn: Callable[..., dict[str, object]]
    resolve_v2_universe_settings_fn: Callable[..., dict[str, object]]
    load_v2_info_items_for_date_fn: Callable[..., tuple[str, list[InfoItem]]]
    fit_v2_info_shadow_models_fn: Callable[..., tuple[dict[str, object], dict[str, object]]]
    enrich_state_with_info_fn: Callable[..., CompositeState]
    build_shadow_scored_rows_for_step_fn: Callable[..., tuple[pd.DataFrame, bool]]
    stock_policy_score_fn: Callable[[object], float]
    panel_slice_metrics_fn: Callable[..., tuple[float, float, float, float]]
    filter_info_items_by_source_subset_fn: Callable[[Iterable[InfoItem], str], list[InfoItem]]
    event_tag_counts_fn: Callable[[Iterable[InfoItem]], dict[str, int]]
    info_source_breakdown_fn: Callable[[Iterable[InfoItem]], dict[str, int]]


@dataclass(frozen=True)
class V2InfoResearchResult:
    strategy_id: str
    forecast_backend: str
    retrain_days: int
    training_window_days: int | None
    split_mode: str
    embargo_days: int
    horizons: tuple[str, ...]
    min_tag_count: int
    max_tag_count: int
    trajectory_steps: int
    fit_steps: int
    evaluation_steps: int
    fit_scope: str
    evaluation_scope: str
    evaluation_window_start: str
    evaluation_window_end: str
    latest_state_date: str
    info_file: str
    info_manifest: dict[str, object]
    source_variants: dict[str, dict[str, object]]
    timestamp_variants: dict[str, dict[str, object]]
    tag_variants: list[dict[str, object]]

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["horizons"] = list(self.horizons)
        return payload

    def summary(self) -> dict[str, object]:
        all_info = dict(self.source_variants.get("all_info", {}))
        return {
            "strategy_id": self.strategy_id,
            "forecast_backend": self.forecast_backend,
            "trajectory_steps": int(self.trajectory_steps),
            "fit_steps": int(self.fit_steps),
            "evaluation_steps": int(self.evaluation_steps),
            "fit_scope": self.fit_scope,
            "evaluation_scope": self.evaluation_scope,
            "evaluation_window": {
                "start": self.evaluation_window_start,
                "end": self.evaluation_window_end,
            },
            "latest_state_date": self.latest_state_date,
            "info_item_count": int(self.info_manifest.get("info_item_count", 0)),
            "publish_timestamp_coverage_ratio": float(
                self.info_manifest.get("publish_timestamp_coverage_ratio", 0.0)
            ),
            "all_info_best_horizon": str(all_info.get("best_horizon", "")),
            "all_info_best_rank_ic": float(all_info.get("best_horizon_rank_ic", 0.0)),
            "all_info_best_spread": float(all_info.get("best_horizon_top_bottom_spread", 0.0)),
            "source_item_counts": {
                key: int(dict(value).get("item_count", 0))
                for key, value in self.source_variants.items()
            },
            "top_tag_summaries": [
                {
                    "event_tag": str(item.get("event_tag", "")),
                    "item_count": int(item.get("item_count", 0)),
                    "best_horizon": str(item.get("best_horizon", "")),
                    "best_rank_ic": float(item.get("best_horizon_rank_ic", 0.0)),
                }
                for item in self.tag_variants[:5]
            ],
        }


def _trajectory_steps(trajectory: object | None) -> list[object]:
    return list(getattr(trajectory, "steps", []) or [])


def _clone_trajectory(trajectory: object | None, steps: list[object]) -> object | None:
    if trajectory is None or not steps:
        return None
    prepared = getattr(trajectory, "prepared", None)
    try:
        return type(trajectory)(prepared=prepared, steps=list(steps))
    except TypeError:
        return SimpleNamespace(prepared=prepared, steps=list(steps))


def _trajectory_window(trajectory: object | None) -> tuple[str, str]:
    steps = _trajectory_steps(trajectory)
    if not steps:
        return "", ""
    return str(getattr(steps[0], "date", ""))[:10], str(getattr(steps[-1], "date", ""))[:10]


def _last_state_date(trajectory: object | None) -> str:
    steps = _trajectory_steps(trajectory)
    if not steps:
        return ""
    last_step = steps[-1]
    composite_state = getattr(last_step, "composite_state", None)
    market_state = getattr(composite_state, "market", None)
    market_date = str(getattr(market_state, "as_of_date", "") or "").strip()
    return market_date or str(getattr(last_step, "date", ""))[:10]


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _item_has_timestamp(item: InfoItem) -> bool:
    return bool(str(getattr(item, "publish_datetime", "")).strip())


def _filter_timestamp_items(items: Iterable[InfoItem], mode: str) -> list[InfoItem]:
    if mode == "timestamped_only":
        return [item for item in items if _item_has_timestamp(item)]
    if mode == "untimestamped_only":
        return [item for item in items if not _item_has_timestamp(item)]
    return list(items)


def _build_info_manifest(
    *,
    info_file: str,
    info_items: list[InfoItem],
    as_of_date: pd.Timestamp,
    deps: InfoResearchDependencies,
) -> dict[str, object]:
    publish_count = sum(1 for item in info_items if _item_has_timestamp(item))
    date_window = {"start": "", "end": ""}
    if info_items:
        date_window = {
            "start": str(min(item.date for item in info_items)),
            "end": str(max(item.date for item in info_items)),
        }
    return {
        "info_file": str(info_file),
        "as_of_date": str(as_of_date.date()),
        "info_item_count": int(len(info_items)),
        "date_window": date_window,
        "publish_timestamp_count": int(publish_count),
        "publish_timestamp_coverage_ratio": float(publish_count / max(1, len(info_items))) if info_items else 0.0,
        "event_tag_distribution": deps.event_tag_counts_fn(info_items),
        "info_source_breakdown": deps.info_source_breakdown_fn(info_items),
    }


def _build_quant_scored_rows_for_step(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    stock_policy_score_fn: Callable[[object], float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stock in state.stocks:
        frame = stock_frames.get(stock.symbol)
        if frame is None:
            continue
        row = frame[frame["date"] == date]
        if row.empty:
            continue
        payload = row.iloc[0]
        rows.append(
            {
                "symbol": str(stock.symbol),
                "score": float(stock_policy_score_fn(stock)),
                "realized_ret_1d": float(payload.get("excess_ret_1_vs_mkt", np.nan)),
                "realized_ret_2d": float(payload.get("excess_ret_2_vs_mkt", np.nan)),
                "realized_ret_3d": float(payload.get("excess_ret_3_vs_mkt", np.nan)),
                "realized_ret_5d": float(payload.get("excess_ret_5_vs_mkt", np.nan)),
                "realized_ret_10d": float(payload.get("excess_ret_10_vs_mkt", np.nan)),
                "realized_ret_20d": float(payload.get("excess_ret_20_vs_sector", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _variant_cache_key(items: Iterable[InfoItem]) -> str:
    rows = [
        (
            str(getattr(item, "date", "")),
            str(getattr(item, "target_type", "")),
            str(getattr(item, "target", "")),
            str(getattr(item, "horizon", "")),
            str(getattr(item, "direction", "")),
            str(getattr(item, "info_type", "")),
            str(getattr(item, "source_subset", "")),
            str(getattr(item, "event_tag", "")),
            str(getattr(item, "publish_datetime", "")),
            str(getattr(item, "title", "")),
        )
        for item in items
    ]
    raw = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _build_evaluation_step_cache(
    *,
    evaluation_trajectory: object | None,
    horizons: tuple[str, ...],
    deps: InfoResearchDependencies,
) -> list[dict[str, object]]:
    steps = _trajectory_steps(evaluation_trajectory)
    stock_frames = getattr(getattr(evaluation_trajectory, "prepared", None), "stock_frames", {}) or {}
    cached_steps: list[dict[str, object]] = []
    for step in steps:
        as_of_date = pd.Timestamp(getattr(step, "date"))
        quant_rows = _build_quant_scored_rows_for_step(
            state=step.composite_state,
            stock_frames=stock_frames,
            date=as_of_date,
            stock_policy_score_fn=deps.stock_policy_score_fn,
        )
        quant_metrics = {
            horizon: deps.panel_slice_metrics_fn(quant_rows, realized_col=_REALIZED_COL_BY_HORIZON[horizon])
            for horizon in horizons
            if horizon in _REALIZED_COL_BY_HORIZON
        }
        cached_steps.append(
            {
                "date": as_of_date,
                "state": step.composite_state,
                "quant_rows": quant_rows,
                "quant_metrics": quant_metrics,
                "stock_frames": stock_frames,
            }
        )
    return cached_steps


def _evaluate_info_variant(
    *,
    name: str,
    info_items: list[InfoItem],
    fit_trajectory: object | None,
    evaluation_step_cache: list[dict[str, object]],
    settings: dict[str, object],
    horizons: tuple[str, ...],
    deps: InfoResearchDependencies,
) -> dict[str, object]:
    stock_models: dict[str, object] | None = None
    market_models: dict[str, object] | None = None
    if info_items and _trajectory_steps(fit_trajectory):
        stock_models, market_models = deps.fit_v2_info_shadow_models_fn(
            trajectory=fit_trajectory,
            settings=settings,
            info_items=info_items,
        )

    metric_acc: dict[str, dict[str, list[float]]] = {
        horizon: {
            "available_rows": [],
            "shadow_rank_ic": [],
            "shadow_top_decile_return": [],
            "shadow_top_bottom_spread": [],
            "shadow_top_k_hit_rate": [],
            "quant_rank_ic": [],
            "quant_top_decile_return": [],
            "quant_top_bottom_spread": [],
            "quant_top_k_hit_rate": [],
        }
        for horizon in horizons
        if horizon in _REALIZED_COL_BY_HORIZON
    }
    event_hits_shadow: list[float] = []
    event_hits_quant: list[float] = []
    primary_horizon = "20d" if "20d" in metric_acc else (next(iter(metric_acc.keys()), ""))
    for step_cache in evaluation_step_cache:
        as_of_date = pd.Timestamp(step_cache["date"])
        enriched_state = deps.enrich_state_with_info_fn(
            state=step_cache["state"],
            as_of_date=as_of_date,
            info_items=info_items,
            settings=settings,
            stock_models=stock_models,
            market_models=market_models,
        )
        shadow_rows, event_day = deps.build_shadow_scored_rows_for_step_fn(
            state=enriched_state,
            stock_frames=step_cache.get("stock_frames", {}) or {},
            date=as_of_date,
        )
        quant_metrics = dict(step_cache.get("quant_metrics", {}))
        for horizon, bucket in metric_acc.items():
            realized_col = _REALIZED_COL_BY_HORIZON[horizon]
            valid_rows = shadow_rows.dropna(subset=["score", realized_col]) if not shadow_rows.empty else shadow_rows
            bucket["available_rows"].append(float(len(valid_rows)))
            s_rank_ic, s_top, s_spread, s_hit = deps.panel_slice_metrics_fn(shadow_rows, realized_col=realized_col)
            q_rank_ic, q_top, q_spread, q_hit = quant_metrics.get(horizon, (0.0, 0.0, 0.0, 0.0))
            bucket["shadow_rank_ic"].append(float(s_rank_ic))
            bucket["shadow_top_decile_return"].append(float(s_top))
            bucket["shadow_top_bottom_spread"].append(float(s_spread))
            bucket["shadow_top_k_hit_rate"].append(float(s_hit))
            bucket["quant_rank_ic"].append(float(q_rank_ic))
            bucket["quant_top_decile_return"].append(float(q_top))
            bucket["quant_top_bottom_spread"].append(float(q_spread))
            bucket["quant_top_k_hit_rate"].append(float(q_hit))
            if event_day and horizon == primary_horizon:
                event_hits_shadow.append(float(s_hit))
                event_hits_quant.append(float(q_hit))

    horizon_metrics: dict[str, dict[str, float]] = {}
    for horizon, bucket in metric_acc.items():
        shadow_rank_ic = _mean(bucket["shadow_rank_ic"])
        shadow_spread = _mean(bucket["shadow_top_bottom_spread"])
        shadow_hit = _mean(bucket["shadow_top_k_hit_rate"])
        quant_rank_ic = _mean(bucket["quant_rank_ic"])
        quant_spread = _mean(bucket["quant_top_bottom_spread"])
        quant_hit = _mean(bucket["quant_top_k_hit_rate"])
        horizon_metrics[horizon] = {
            "available_rows": int(round(sum(bucket["available_rows"]))),
            "shadow_rank_ic": shadow_rank_ic,
            "shadow_top_decile_return": _mean(bucket["shadow_top_decile_return"]),
            "shadow_top_bottom_spread": shadow_spread,
            "shadow_top_k_hit_rate": shadow_hit,
            "quant_rank_ic": quant_rank_ic,
            "quant_top_decile_return": _mean(bucket["quant_top_decile_return"]),
            "quant_top_bottom_spread": quant_spread,
            "quant_top_k_hit_rate": quant_hit,
            "uplift_rank_ic": float(shadow_rank_ic - quant_rank_ic),
            "uplift_top_bottom_spread": float(shadow_spread - quant_spread),
            "uplift_top_k_hit_rate": float(shadow_hit - quant_hit),
        }

    best_horizon = ""
    best_rank_ic = -1e18
    best_spread = 0.0
    for horizon, payload in horizon_metrics.items():
        rank_ic = float(payload.get("shadow_rank_ic", 0.0))
        if rank_ic > best_rank_ic:
            best_horizon = horizon
            best_rank_ic = rank_ic
            best_spread = float(payload.get("shadow_top_bottom_spread", 0.0))
    if best_rank_ic <= -1e17:
        best_rank_ic = 0.0

    return {
        "name": name,
        "item_count": int(len(info_items)),
        "source_breakdown": deps.info_source_breakdown_fn(info_items),
        "event_tag_distribution": deps.event_tag_counts_fn(info_items),
        "stock_shadow_modes": {
            horizon: str(getattr(model, "mode", ""))
            for horizon, model in (stock_models or {}).items()
        },
        "market_shadow_modes": {
            horizon: str(getattr(model, "mode", ""))
            for horizon, model in (market_models or {}).items()
        },
        "model_samples": {
            "stock": {
                horizon: int(getattr(model, "samples", 0))
                for horizon, model in (stock_models or {}).items()
            },
            "market": {
                horizon: int(getattr(model, "samples", 0))
                for horizon, model in (market_models or {}).items()
            },
        },
        "horizon_metrics": horizon_metrics,
        "best_horizon": best_horizon,
        "best_horizon_rank_ic": float(best_rank_ic),
        "best_horizon_top_bottom_spread": float(best_spread),
        "event_day_hit_rate_shadow": _mean(event_hits_shadow),
        "event_day_hit_rate_quant": _mean(event_hits_quant),
    }


def run_v2_info_research(
    *,
    dependencies: InfoResearchDependencies,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    info_file: str | None = None,
    info_lookback_days: int | None = None,
    info_half_life_days: float | None = None,
    use_info_fusion: bool | None = None,
    use_learned_info_fusion: bool | None = None,
    info_shadow_only: bool | None = None,
    info_types: str | None = None,
    info_source_mode: str | None = None,
    info_subsets: str | None = None,
    info_cutoff_time: str | None = None,
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    artifact_root: str = "artifacts/v2",
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
    training_window_days: int | None = 480,
    split_mode: str = "purged_wf",
    embargo_days: int = 20,
    horizons: Iterable[str] = ("1d", "2d", "3d", "5d", "10d", "20d"),
    min_tag_count: int = 5,
    max_tag_count: int = 12,
) -> V2InfoResearchResult:
    del artifact_root
    normalized_horizons = tuple(str(item).strip() for item in horizons if str(item).strip() in _REALIZED_COL_BY_HORIZON)
    dependencies.emit_progress_fn("info-research", f"载入信息研究轨迹: backend={forecast_backend}")
    trajectory = dependencies.load_or_build_v2_backtest_trajectory_fn(
        config_path=config_path,
        source=source,
        start_date=start_date,
        end_date=end_date,
        lookback_years=lookback_years,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        info_file=info_file,
        info_lookback_days=info_lookback_days,
        info_half_life_days=info_half_life_days,
        use_info_fusion=use_info_fusion,
        use_learned_info_fusion=use_learned_info_fusion,
        info_shadow_only=info_shadow_only,
        info_types=info_types,
        info_source_mode=info_source_mode,
        info_subsets=info_subsets,
        info_cutoff_time=info_cutoff_time,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        training_window_days=training_window_days,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    trajectory_steps = dependencies.trajectory_step_count_fn(trajectory)
    settings = dependencies.load_v2_runtime_settings_fn(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        info_file=info_file,
        info_lookback_days=info_lookback_days,
        info_half_life_days=info_half_life_days,
        info_cutoff_time=info_cutoff_time,
        use_info_fusion=use_info_fusion,
        use_learned_info_fusion=use_learned_info_fusion,
        info_shadow_only=info_shadow_only,
        info_types=info_types,
        info_source_mode=info_source_mode,
        info_subsets=info_subsets,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        training_window_days=training_window_days,
        start_date=start_date,
        end_date=end_date,
        lookback_years=lookback_years,
    )
    settings = dependencies.resolve_v2_universe_settings_fn(settings=settings, cache_root=cache_root)

    if trajectory is None:
        empty_manifest = _build_info_manifest(
            info_file=str(info_file or ""),
            info_items=[],
            as_of_date=pd.Timestamp(datetime.now().date()),
            deps=dependencies,
        )
        return V2InfoResearchResult(
            strategy_id=strategy_id,
            forecast_backend=forecast_backend,
            retrain_days=int(retrain_days),
            training_window_days=training_window_days,
            split_mode=split_mode,
            embargo_days=int(embargo_days),
            horizons=normalized_horizons,
            min_tag_count=int(min_tag_count),
            max_tag_count=int(max_tag_count),
            trajectory_steps=0,
            fit_steps=0,
            evaluation_steps=0,
            fit_scope="empty",
            evaluation_scope="empty",
            evaluation_window_start="",
            evaluation_window_end="",
            latest_state_date="",
            info_file=str(info_file or ""),
            info_manifest=empty_manifest,
            source_variants={},
            timestamp_variants={},
            tag_variants=[],
        )

    train_trajectory, validation_trajectory, holdout_trajectory = dependencies.split_research_trajectory_fn(
        trajectory,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )
    fit_steps = _trajectory_steps(validation_trajectory) or (_trajectory_steps(train_trajectory) + _trajectory_steps(validation_trajectory))
    fit_trajectory = _clone_trajectory(trajectory, fit_steps) if fit_steps else trajectory
    fit_scope = "validation_only" if _trajectory_steps(validation_trajectory) else "full_fallback"
    if fit_scope != "validation_only" and fit_trajectory is not trajectory:
        fit_scope = "train_plus_validation"
    evaluation_trajectory = holdout_trajectory if _trajectory_steps(holdout_trajectory) else validation_trajectory
    evaluation_scope = "holdout" if _trajectory_steps(holdout_trajectory) else "validation_fallback"
    if not _trajectory_steps(evaluation_trajectory):
        evaluation_trajectory = trajectory
        evaluation_scope = "full_fallback"

    latest_state_date = _last_state_date(evaluation_trajectory or trajectory)
    info_as_of_date = pd.Timestamp(latest_state_date or datetime.now().date())
    info_file_path, info_items = dependencies.load_v2_info_items_for_date_fn(
        settings=settings,
        as_of_date=info_as_of_date,
        learned_window=True,
    )
    info_manifest = _build_info_manifest(
        info_file=info_file_path,
        info_items=info_items,
        as_of_date=info_as_of_date,
        deps=dependencies,
    )
    evaluation_step_cache = _build_evaluation_step_cache(
        evaluation_trajectory=evaluation_trajectory,
        horizons=normalized_horizons,
        deps=dependencies,
    )

    dependencies.emit_progress_fn(
        "info-research",
        (
            "开始评估信息分层: "
            f"items={len(info_items)}, fit_steps={len(_trajectory_steps(fit_trajectory))}, "
            f"evaluation_steps={len(_trajectory_steps(evaluation_trajectory))}, "
            f"cached_quant_steps={len(evaluation_step_cache)}"
        ),
    )
    variant_result_cache: dict[str, dict[str, object]] = {}

    def _evaluate_cached_variant(*, name: str, variant_items: list[InfoItem]) -> dict[str, object]:
        cache_key = _variant_cache_key(variant_items)
        cached = variant_result_cache.get(cache_key)
        if cached is not None:
            dependencies.emit_progress_fn("cache", f"命中 info research variant 缓存: name={name}, key={cache_key}")
            payload = copy.deepcopy(cached)
            payload["name"] = name
            return payload
        dependencies.emit_progress_fn("info-research", f"评估 variant: {name} | items={len(variant_items)}")
        payload = _evaluate_info_variant(
            name=name,
            info_items=variant_items,
            fit_trajectory=fit_trajectory,
            evaluation_step_cache=evaluation_step_cache,
            settings=settings,
            horizons=normalized_horizons,
            deps=dependencies,
        )
        variant_result_cache[cache_key] = copy.deepcopy(payload)
        return payload

    source_variants = {
        "all_info": _evaluate_cached_variant(
            name="all_info",
            variant_items=list(info_items),
        )
    }
    for subset in ("market_news", "announcements", "research"):
        source_variants[subset] = _evaluate_cached_variant(
            name=subset,
            variant_items=dependencies.filter_info_items_by_source_subset_fn(info_items, subset),
        )

    timestamp_variants = {}
    for key in ("all_items", "timestamped_only", "untimestamped_only"):
        timestamp_variants[key] = _evaluate_cached_variant(
            name=key,
            variant_items=_filter_timestamp_items(info_items, key),
        )

    tag_counts = dependencies.event_tag_counts_fn(info_items)
    ranked_tags = sorted(
        [(str(tag), int(count)) for tag, count in tag_counts.items() if str(tag).strip() and int(count) >= int(min_tag_count)],
        key=lambda item: (-item[1], item[0]),
    )[: max(0, int(max_tag_count))]
    tag_variants: list[dict[str, object]] = []
    for event_tag, count in ranked_tags:
        payload = _evaluate_cached_variant(
            name=event_tag,
            variant_items=[item for item in info_items if str(getattr(item, "event_tag", "")) == event_tag],
        )
        payload["event_tag"] = event_tag
        payload["item_count"] = count
        tag_variants.append(payload)

    evaluation_window_start, evaluation_window_end = _trajectory_window(evaluation_trajectory)
    dependencies.emit_progress_fn("info-research", "信息层独立研究完成")
    return V2InfoResearchResult(
        strategy_id=strategy_id,
        forecast_backend=forecast_backend,
        retrain_days=int(retrain_days),
        training_window_days=training_window_days,
        split_mode=split_mode,
        embargo_days=int(embargo_days),
        horizons=normalized_horizons,
        min_tag_count=int(min_tag_count),
        max_tag_count=int(max_tag_count),
        trajectory_steps=int(trajectory_steps),
        fit_steps=int(dependencies.trajectory_step_count_fn(fit_trajectory)),
        evaluation_steps=int(dependencies.trajectory_step_count_fn(evaluation_trajectory)),
        fit_scope=fit_scope,
        evaluation_scope=evaluation_scope,
        evaluation_window_start=evaluation_window_start,
        evaluation_window_end=evaluation_window_end,
        latest_state_date=latest_state_date,
        info_file=info_file_path,
        info_manifest=info_manifest,
        source_variants=source_variants,
        timestamp_variants=timestamp_variants,
        tag_variants=tag_variants,
    )


def persist_v2_info_research_artifacts(
    result: V2InfoResearchResult,
    *,
    artifact_root: str = "artifacts/v2",
) -> dict[str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(str(artifact_root)) / result.strategy_id / "info_research" / run_id
    manifest_path = run_dir / "info_research_manifest.json"
    sources_path = run_dir / "source_variants.json"
    timestamps_path = run_dir / "timestamp_variants.json"
    tags_path = run_dir / "tag_variants.json"

    manifest_payload = result.summary()
    manifest_payload["result"] = result.to_payload()
    _json_dump(manifest_path, manifest_payload)
    _json_dump(sources_path, result.source_variants)
    _json_dump(timestamps_path, result.timestamp_variants)
    _json_dump(tags_path, result.tag_variants)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "source_variants_path": str(sources_path),
        "timestamp_variants_path": str(timestamps_path),
        "tag_variants_path": str(tags_path),
    }
