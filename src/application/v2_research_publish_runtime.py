from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.application.v2_contracts import InfoItem, V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult
from src.application.v2_external_signal_support import (
    ensure_external_signal_manifest_path,
    merge_external_signal_manifest_summary,
)
from src.application.v2_info_shadow_runtime import (
    info_payload_enabled as info_payload_enabled_for_settings,
    info_shadow_enabled as info_shadow_enabled_for_settings,
)
from src.application.v2_leader_runtime import build_leader_artifact_payloads, build_research_label_artifact_payloads
from src.application.v2_signal_training_runtime import build_signal_training_artifacts
from src.contracts.artifacts import add_artifact_metadata


@dataclass(frozen=True)
class ResearchPublishDependencies:
    load_v2_runtime_settings_fn: Callable[..., dict[str, object]]
    resolve_v2_universe_settings_fn: Callable[..., dict[str, object]]
    stable_json_hash_fn: Callable[[object], str]
    sha256_file_fn: Callable[[object], str]
    sha256_text_fn: Callable[[str], str]
    compose_run_snapshot_hash_fn: Callable[..., str]
    load_or_build_v2_backtest_trajectory_fn: Callable[..., Any]
    split_research_trajectory_fn: Callable[..., tuple[Any, Any, Any]]
    build_frozen_daily_state_payload_fn: Callable[..., dict[str, object]]
    build_frozen_linear_forecast_bundle_fn: Callable[[Any], dict[str, object]]
    resolve_info_file_from_settings_fn: Callable[[dict[str, object]], str]
    load_v2_info_items_for_date_fn: Callable[..., tuple[str, list[InfoItem]]]
    build_info_shadow_report_fn: Callable[..., dict[str, object]]
    build_info_manifest_payload_fn: Callable[..., dict[str, object]]
    build_external_signal_package_for_date_fn: Callable[..., dict[str, object]]
    parse_boolish_fn: Callable[[object, bool], bool]
    decode_composite_state_fn: Callable[[object], object | None]
    enrich_state_with_info_fn: Callable[..., object]
    attach_external_signals_to_composite_state_fn: Callable[..., tuple[object, dict[str, object]]]
    attach_insight_memory_to_state_fn: Callable[..., object]
    build_insight_artifact_payloads_fn: Callable[..., dict[str, object]]
    serialize_composite_state_fn: Callable[[object], dict[str, object]]
    load_json_dict_fn: Callable[[object], dict[str, object]]
    tier_latest_manifest_path_fn: Callable[..., Path]
    tier_latest_policy_path_fn: Callable[..., Path]
    load_backtest_payload_from_manifest_fn: Callable[[dict[str, object], Path], dict[str, object]]
    load_backtest_payload_for_run_fn: Callable[..., dict[str, object]]
    summary_from_payload_fn: Callable[[V2BacktestSummary, dict[str, object]], V2BacktestSummary]
    pass_release_gate_fn: Callable[..., tuple[bool, list[str]]]
    pass_default_switch_gate_fn: Callable[..., tuple[bool, list[str], dict[str, float]]]
    emit_progress_fn: Callable[[str, str], None]
    remember_research_run_fn: Callable[..., Path]


def with_backtest_metadata(
    summary: V2BacktestSummary,
    *,
    run_id: str,
    snapshot_hash: str,
    config_hash: str,
) -> V2BacktestSummary:
    payload = asdict(summary)
    payload["run_id"] = str(run_id)
    payload["snapshot_hash"] = str(snapshot_hash)
    payload["config_hash"] = str(config_hash)
    return V2BacktestSummary(**payload)


def publish_research_artifacts(
    *,
    dependencies: ResearchPublishDependencies,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
    config_path: str = "config/api.json",
    source: str | None = None,
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
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    settings: dict[str, object] | None = None,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    cache_root: str = "artifacts/v2/cache",
    retrain_days: int = 20,
    forecast_backend: str = "linear",
    training_window_days: int | None = 480,
    trajectory: Any | None = None,
    publish_forecast_models: bool = True,
    split_mode: str = "purged_wf",
    embargo_days: int = 20,
    update_latest: bool = True,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
) -> dict[str, str]:
    settings = settings or dependencies.load_v2_runtime_settings_fn(
        config_path=config_path,
        source=source,
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
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        training_window_days=training_window_days,
    )
    settings = dependencies.resolve_v2_universe_settings_fn(settings=dict(settings), cache_root=cache_root)
    context = _build_publish_runtime_context(
        dependencies=dependencies,
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        settings=settings,
        baseline=baseline,
        calibration=calibration,
        learning=learning,
    )
    forecast_artifacts = _build_forecast_publish_artifacts(
        dependencies=dependencies,
        settings=settings,
        context=context,
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        cache_root=cache_root,
        retrain_days=retrain_days,
        forecast_backend=forecast_backend,
        training_window_days=training_window_days,
        trajectory=trajectory,
        publish_forecast_models=publish_forecast_models,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )
    paths = _build_publish_paths(
        dependencies=dependencies,
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        base_dir=context.base_dir,
        universe_tier_value=context.universe_tier_value,
    )

    info_artifacts = _build_info_publish_artifacts(
        dependencies=dependencies,
        settings=settings,
        context=context,
        forecast_artifacts=forecast_artifacts,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        cache_root=cache_root,
        retrain_days=retrain_days,
        forecast_backend=forecast_backend,
        training_window_days=training_window_days,
        publish_forecast_models=publish_forecast_models,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )
    insight_artifacts = _build_insight_publish_artifacts(
        dependencies=dependencies,
        settings=settings,
        info_artifacts=info_artifacts,
    )
    leader_artifacts = _build_leader_publish_artifacts(
        dependencies=dependencies,
        settings=settings,
        info_artifacts=info_artifacts,
        forecast_artifacts=forecast_artifacts,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )

    gate_artifacts = _evaluate_publish_gates(
        dependencies=dependencies,
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        context=context,
        paths=paths,
    )
    manifests = _build_publish_manifests(
        dependencies=dependencies,
        strategy_id=strategy_id,
        settings=settings,
        calibration=calibration,
        context=context,
        paths=paths,
        forecast_artifacts=forecast_artifacts,
        info_artifacts=info_artifacts,
        insight_artifacts=insight_artifacts,
        leader_artifacts=leader_artifacts,
        split_mode=split_mode,
        embargo_days=embargo_days,
        publish_forecast_models=publish_forecast_models,
        release_gate=gate_artifacts.release_gate,
        default_switch_gate=gate_artifacts.default_switch_gate,
    )

    _write_and_publish_artifacts(
        dependencies=dependencies,
        context=context,
        paths=paths,
        forecast_artifacts=forecast_artifacts,
        info_artifacts=info_artifacts,
        insight_artifacts=insight_artifacts,
        leader_artifacts=leader_artifacts,
        manifests=manifests,
        gate_artifacts=gate_artifacts,
        publish_forecast_models=publish_forecast_models,
        update_latest=update_latest,
    )

    memory_path = dependencies.remember_research_run_fn(
        memory_root=Path(str(artifact_root)) / "memory",
        strategy_id=strategy_id,
        run_id=context.run_id,
        baseline=context.baseline_meta,
        calibration=calibration,
        learning=learning,
        release_gate_passed=gate_artifacts.release_gate_passed,
        universe_id=context.universe_id,
        universe_tier=context.universe_tier_value,
        universe_size=int(context.universe_size),
        external_signal_version=str(settings.get("external_signal_version", "v1")),
        external_signal_enabled=bool(settings.get("external_signals", True)),
    )

    return {
        "run_dir": str(context.base_dir),
        "run_id": context.run_id,
        "baseline_reference_run_id": context.baseline_reference_run_id,
        "universe_tier": context.universe_tier_value,
        "universe_id": context.universe_id,
        "universe_size": str(context.universe_size),
        "source_universe_manifest_path": context.source_universe_manifest_path,
        "info_manifest": str(paths.info_manifest_path),
        "info_shadow_report": str(paths.info_shadow_report_path),
        "info_hash": info_artifacts.info_hash,
        "info_item_count": str(info_artifacts.info_manifest.get("info_item_count", 0)),
        "info_shadow_enabled": "true" if info_artifacts.info_shadow_enabled else "false",
        "external_signal_manifest": str(paths.external_signal_manifest_path),
        "insight_manifest": str(paths.insight_manifest_path),
        "viewpoints": str(paths.viewpoints_path),
        "theme_episodes": str(paths.theme_episodes_path),
        "stock_roles": str(paths.stock_roles_path),
        "execution_plan": str(paths.execution_plan_path),
        "leader_manifest": str(paths.leader_manifest_path),
        "leader_candidates": str(paths.leader_candidates_path),
        "leader_training_labels": str(paths.leader_training_labels_path),
        "exit_training_labels": str(paths.exit_training_labels_path),
        "leader_rank_model": str(paths.leader_rank_model_path),
        "exit_behavior_model": str(paths.exit_behavior_model_path),
        "external_signal_version": str(settings.get("external_signal_version", "v1")),
        "external_signal_enabled": "true" if bool(settings.get("external_signals", True)) else "false",
        "generator_manifest": str(settings.get("generator_manifest_path", "")),
        "generator_version": str(settings.get("generator_version", "")),
        "generator_hash": str(settings.get("generator_hash", "")),
        "coarse_pool_size": str(settings.get("coarse_pool_size", 0)),
        "refined_pool_size": str(settings.get("refined_pool_size", 0)),
        "selected_pool_size": str(settings.get("selected_pool_size", 0)),
        "use_us_index_context": "true" if bool(settings.get("use_us_index_context", False)) else "false",
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "dataset_manifest": str(paths.dataset_path),
        "policy_calibration": str(paths.calibration_path),
        "learned_policy_model": str(paths.learning_path),
        "forecast_models_manifest": str(paths.forecast_models_path) if publish_forecast_models else "",
        "frozen_forecast_bundle": str(paths.frozen_forecast_bundle_path) if publish_forecast_models else "",
        "frozen_daily_state": str(paths.frozen_state_path) if publish_forecast_models else "",
        "backtest_summary": str(paths.backtest_path),
        "consistency_checklist": str(paths.consistency_path),
        "rolling_oos_report": str(paths.rolling_oos_path),
        "research_manifest": str(paths.manifest_path),
        "published_policy_model": str(paths.latest_policy_path),
        "strategy_memory": str(memory_path),
        "capital_flow_snapshot": json.dumps(
            info_artifacts.external_signal_package.get("capital_flow_snapshot", {}),
            ensure_ascii=False,
        ),
        "macro_context_snapshot": json.dumps(
            info_artifacts.external_signal_package.get("macro_context_snapshot", {}),
            ensure_ascii=False,
        ),
        "release_gate_passed": "true" if gate_artifacts.release_gate_passed else "false",
        "default_switch_gate_passed": "true" if gate_artifacts.default_switch_gate_passed else "false",
        "snapshot_hash": context.snapshot_hash,
        "config_hash": context.config_hash,
    }


@dataclass(frozen=True)
class InfoPublishArtifacts:
    info_shadow_enabled: bool
    info_shadow_report: dict[str, object]
    info_file_path: str
    info_as_of_date: pd.Timestamp
    info_items: list[InfoItem]
    info_manifest: dict[str, object]
    info_hash: str
    external_signal_package: dict[str, object]
    external_signal_manifest: dict[str, object]
    frozen_daily_state: dict[str, object]
    frozen_forecast_bundle: dict[str, object]


@dataclass(frozen=True)
class PublishGateArtifacts:
    gate_ok: bool
    release_gate_passed: bool
    release_gate: dict[str, object]
    default_switch_gate_passed: bool
    default_switch_gate: dict[str, object]


@dataclass(frozen=True)
class PublishManifestArtifacts:
    dataset_manifest: dict[str, object]
    calibration_manifest: dict[str, object]
    backtest_manifest: dict[str, object]
    consistency_manifest: dict[str, object]
    rolling_oos_manifest: dict[str, object]
    research_manifest: dict[str, object]


@dataclass(frozen=True)
class PublishRuntimeContext:
    run_id: str
    created_at: str
    base_dir: Path
    symbols: list[str]
    universe_tier_value: str
    universe_id: str
    universe_size: int
    universe_generation_rule: str
    source_universe_manifest_path: str
    active_default_universe_tier: str
    candidate_default_universe_tier: str
    baseline_reference_run_id: str
    config_hash: str
    learning_manifest: dict[str, object]
    policy_hash: str
    universe_hash: str
    model_hashes: dict[str, str]
    snapshot_hash: str
    baseline_meta: V2BacktestSummary
    calibrated_meta: V2BacktestSummary
    learned_meta: V2BacktestSummary


@dataclass(frozen=True)
class ForecastPublishArtifacts:
    trajectory: Any | None
    frozen_daily_state: dict[str, object]
    forecast_models_manifest: dict[str, object]
    frozen_forecast_bundle: dict[str, object]
    train_window: dict[str, object]
    validation_window: dict[str, object]
    holdout_window: dict[str, object]
    regime_counts: dict[str, int]


@dataclass(frozen=True)
class InsightPublishArtifacts:
    manifest: dict[str, object]
    viewpoints: list[dict[str, object]]
    theme_episodes: list[dict[str, object]]
    stock_roles: list[dict[str, object]]
    execution_plan: list[dict[str, object]]


@dataclass(frozen=True)
class LeaderPublishArtifacts:
    manifest: dict[str, object]
    candidates: list[dict[str, object]]
    training_label_manifest: dict[str, object]
    leader_training_labels: list[dict[str, object]]
    exit_training_labels: list[dict[str, object]]
    signal_training_manifest: dict[str, object]
    leader_rank_model: dict[str, object]
    exit_behavior_model: dict[str, object]


@dataclass(frozen=True)
class PublishPaths:
    dataset_path: Path
    calibration_path: Path
    learning_path: Path
    forecast_models_path: Path
    frozen_forecast_bundle_path: Path
    frozen_state_path: Path
    backtest_path: Path
    consistency_path: Path
    rolling_oos_path: Path
    info_manifest_path: Path
    info_shadow_report_path: Path
    external_signal_manifest_path: Path
    insight_manifest_path: Path
    viewpoints_path: Path
    theme_episodes_path: Path
    stock_roles_path: Path
    execution_plan_path: Path
    leader_manifest_path: Path
    leader_candidates_path: Path
    leader_training_labels_path: Path
    exit_training_labels_path: Path
    leader_rank_model_path: Path
    exit_behavior_model_path: Path
    manifest_path: Path
    latest_policy_path: Path
    latest_manifest_path: Path
    tier_latest_policy_path: Path
    tier_latest_manifest_path: Path


def _window_payload(trajectory: Any | None) -> dict[str, object]:
    if trajectory is None or not getattr(trajectory, "steps", None):
        return {"start": "", "end": "", "n_steps": 0}
    return {
        "start": str(trajectory.steps[0].date.date()),
        "end": str(trajectory.steps[-1].next_date.date()),
        "n_steps": int(len(trajectory.steps)),
    }


def _build_publish_runtime_context(
    *,
    dependencies: ResearchPublishDependencies,
    strategy_id: str,
    artifact_root: str,
    settings: dict[str, object],
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
) -> PublishRuntimeContext:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_at = datetime.now().isoformat(timespec="seconds")
    base_dir = Path(str(artifact_root)) / str(strategy_id) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    universe_path = Path(str(settings.get("universe_file", "")))
    symbols = [str(item) for item in settings.get("symbols", [])]
    universe_tier_value = str(settings.get("universe_tier", "")).strip()
    universe_id = str(settings.get("universe_id", "")).strip() or universe_tier_value or universe_path.stem or "v2_universe"
    universe_size = int(settings.get("universe_size", len(symbols)) or len(symbols))
    universe_generation_rule = str(settings.get("universe_generation_rule", "")).strip() or "external_universe_file"
    source_universe_manifest_path = str(
        settings.get("source_universe_manifest_path", settings.get("universe_file", ""))
    )
    active_default_universe_tier = str(settings.get("active_default_universe_tier", "favorites_16")).strip()
    candidate_default_universe_tier = str(settings.get("candidate_default_universe_tier", "generated_80")).strip()
    baseline_reference_run_id = str(settings.get("baseline_reference_run_id", "")).strip()
    config_hash = dependencies.stable_json_hash_fn(settings)
    learning_manifest = add_artifact_metadata(
        asdict(learning.model),
        artifact_type="learned_policy_model",
    )
    policy_hash = dependencies.stable_json_hash_fn(learning_manifest)
    universe_hash = (
        str(settings.get("universe_hash", ""))
        or dependencies.sha256_file_fn(universe_path)
        or dependencies.stable_json_hash_fn(symbols)
    )
    model_hashes = {
        "market_model": dependencies.sha256_text_fn("mkt_lr_v2"),
        "sector_model": dependencies.sha256_text_fn("sector_lr_v2"),
        "stock_model": dependencies.sha256_text_fn("stock_lr_v2"),
        "cross_section_model": dependencies.sha256_text_fn("cross_section_v2"),
        "learned_policy_model": policy_hash,
    }
    snapshot_hash = dependencies.compose_run_snapshot_hash_fn(
        run_id=run_id,
        strategy_id=strategy_id,
        config_hash=config_hash,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        model_hashes=model_hashes,
    )
    baseline_meta = with_backtest_metadata(
        baseline,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
    )
    calibrated_meta = with_backtest_metadata(
        calibration.calibrated,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
    )
    learned_meta = with_backtest_metadata(
        learning.learned,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
    )
    return PublishRuntimeContext(
        run_id=run_id,
        created_at=created_at,
        base_dir=base_dir,
        symbols=symbols,
        universe_tier_value=universe_tier_value,
        universe_id=universe_id,
        universe_size=universe_size,
        universe_generation_rule=universe_generation_rule,
        source_universe_manifest_path=source_universe_manifest_path,
        active_default_universe_tier=active_default_universe_tier,
        candidate_default_universe_tier=candidate_default_universe_tier,
        baseline_reference_run_id=baseline_reference_run_id,
        config_hash=config_hash,
        learning_manifest=learning_manifest,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        model_hashes=model_hashes,
        snapshot_hash=snapshot_hash,
        baseline_meta=baseline_meta,
        calibrated_meta=calibrated_meta,
        learned_meta=learned_meta,
    )


def _build_forecast_publish_artifacts(
    *,
    dependencies: ResearchPublishDependencies,
    settings: dict[str, object],
    context: PublishRuntimeContext,
    strategy_id: str,
    config_path: str,
    source: str | None,
    universe_file: str | None,
    universe_limit: int | None,
    universe_tier: str | None,
    cache_root: str,
    retrain_days: int,
    forecast_backend: str,
    training_window_days: int | None,
    trajectory: Any | None,
    publish_forecast_models: bool,
    split_mode: str,
    embargo_days: int,
) -> ForecastPublishArtifacts:
    empty_window = {"start": "", "end": "", "n_steps": 0}
    frozen_daily_state: dict[str, object] = {}
    forecast_models_manifest: dict[str, object] = {}
    frozen_forecast_bundle: dict[str, object] = {}
    train_window = empty_window
    validation_window = empty_window
    holdout_window = empty_window
    regime_counts: dict[str, int] = {}
    if publish_forecast_models:
        if trajectory is None:
            trajectory = dependencies.load_or_build_v2_backtest_trajectory_fn(
                config_path=str(settings.get("config_path", config_path)),
                source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
                universe_file=str(settings.get("universe_file", universe_file))
                if settings.get("universe_file", universe_file) is not None
                else None,
                universe_limit=(
                    int(settings.get("universe_limit"))
                    if settings.get("universe_limit") is not None
                    else (int(universe_limit) if universe_limit is not None else None)
                ),
                universe_tier=str(settings.get("universe_tier", universe_tier)),
                retrain_days=int(retrain_days),
                cache_root=cache_root,
                refresh_cache=False,
                forecast_backend=forecast_backend,
                training_window_days=(
                    int(settings.get("training_window_days"))
                    if settings.get("training_window_days") is not None
                    else training_window_days
                ),
                use_us_index_context=bool(settings.get("use_us_index_context", False)),
                us_index_source=str(settings.get("us_index_source", "akshare")),
            )
        if trajectory is not None:
            train_traj, validation_traj, holdout_traj = dependencies.split_research_trajectory_fn(
                trajectory,
                split_mode=split_mode,
                embargo_days=embargo_days,
            )
            train_window = _window_payload(train_traj)
            validation_window = _window_payload(validation_traj)
            holdout_window = _window_payload(holdout_traj)
            frozen_daily_state = dependencies.build_frozen_daily_state_payload_fn(
                trajectory=trajectory,
                split_mode=split_mode,
                embargo_days=embargo_days,
            )
            for step in holdout_traj.steps:
                regime = str(step.composite_state.risk_regime or "unknown")
                regime_counts[regime] = int(regime_counts.get(regime, 0)) + 1
            if (
                str(forecast_backend).strip().lower() == "linear"
                and getattr(trajectory, "prepared", None) is not None
                and hasattr(trajectory.prepared, "market_valid")
                and hasattr(trajectory.prepared, "panel")
                and hasattr(trajectory.prepared, "market_feature_cols")
                and hasattr(trajectory.prepared, "feature_cols")
                and hasattr(trajectory.prepared, "dates")
            ):
                frozen_forecast_bundle = dependencies.build_frozen_linear_forecast_bundle_fn(trajectory.prepared)
        forecast_models_manifest = add_artifact_metadata(
            {
                "run_id": context.run_id,
                "strategy_id": str(strategy_id),
                "forecast_backend": str(forecast_backend),
                "split_mode": str(split_mode),
                "embargo_days": int(embargo_days),
                "retrain_days": int(retrain_days),
                "training_window_days": (
                    int(settings.get("training_window_days"))
                    if settings.get("training_window_days") is not None
                    else None
                ),
                "use_us_index_context": bool(settings.get("use_us_index_context", False)),
                "us_index_source": str(settings.get("us_index_source", "akshare")),
                "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
                "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
                "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
                "model_hashes": context.model_hashes,
                "data_window": {
                    "start": str(settings.get("start", "")),
                    "end": str(settings.get("end", "")),
                },
                "regime_counts": regime_counts,
                "frozen_bundle_ready": bool(frozen_forecast_bundle),
            },
            artifact_type="forecast_models_manifest",
        )
    return ForecastPublishArtifacts(
        trajectory=trajectory,
        frozen_daily_state=frozen_daily_state,
        forecast_models_manifest=forecast_models_manifest,
        frozen_forecast_bundle=frozen_forecast_bundle,
        train_window=train_window,
        validation_window=validation_window,
        holdout_window=holdout_window,
        regime_counts=regime_counts,
    )


def _build_publish_paths(
    *,
    dependencies: ResearchPublishDependencies,
    artifact_root: str,
    strategy_id: str,
    base_dir: Path,
    universe_tier_value: str,
) -> PublishPaths:
    return PublishPaths(
        dataset_path=base_dir / "dataset_manifest.json",
        calibration_path=base_dir / "policy_calibration.json",
        learning_path=base_dir / "learned_policy_model.json",
        forecast_models_path=base_dir / "forecast_models_manifest.json",
        frozen_forecast_bundle_path=base_dir / "frozen_forecast_bundle.json",
        frozen_state_path=base_dir / "frozen_daily_state.json",
        backtest_path=base_dir / "backtest_summary.json",
        consistency_path=base_dir / "consistency_checklist.json",
        rolling_oos_path=base_dir / "rolling_oos_report.json",
        info_manifest_path=base_dir / "info_manifest.json",
        info_shadow_report_path=base_dir / "info_shadow_report.json",
        external_signal_manifest_path=ensure_external_signal_manifest_path(base_dir),
        insight_manifest_path=base_dir / "insight_manifest.json",
        viewpoints_path=base_dir / "viewpoints.json",
        theme_episodes_path=base_dir / "theme_episodes.json",
        stock_roles_path=base_dir / "stock_roles.json",
        execution_plan_path=base_dir / "execution_plan.json",
        leader_manifest_path=base_dir / "leader_manifest.json",
        leader_candidates_path=base_dir / "leader_candidates.json",
        leader_training_labels_path=base_dir / "leader_training_labels.json",
        exit_training_labels_path=base_dir / "exit_training_labels.json",
        leader_rank_model_path=base_dir / "leader_rank_model.json",
        exit_behavior_model_path=base_dir / "exit_behavior_model.json",
        manifest_path=base_dir / "research_manifest.json",
        latest_policy_path=Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json",
        latest_manifest_path=Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json",
        tier_latest_policy_path=dependencies.tier_latest_policy_path_fn(
            artifact_root=artifact_root,
            strategy_id=strategy_id,
            universe_tier=universe_tier_value,
        ),
        tier_latest_manifest_path=dependencies.tier_latest_manifest_path_fn(
            artifact_root=artifact_root,
            strategy_id=strategy_id,
            universe_tier=universe_tier_value,
        ),
    )


def _default_info_shadow_report(settings: dict[str, object]) -> dict[str, object]:
    return {
        "info_shadow_enabled": False,
        "shadow_only": bool(settings.get("info_shadow_only", True)),
        "quant_only": {},
        "quant_plus_info_shadow": {},
        "market_news_only": {},
        "announcements_only": {},
        "research_only": {},
        "all_info_combined": {},
        "coverage_summary": {},
        "top_positive_stock_deltas": [],
        "top_negative_stock_deltas": [],
        "event_tag_distribution": {},
        "info_source_breakdown": {},
        "last_market_info_state": {},
        "last_date": "",
        "market_shadow_modes": {},
        "stock_shadow_modes": {},
        "model_samples": {"market": {}, "stock": {}},
    }


def _build_info_publish_artifacts(
    *,
    dependencies: ResearchPublishDependencies,
    settings: dict[str, object],
    context: PublishRuntimeContext,
    forecast_artifacts: ForecastPublishArtifacts,
    config_path: str,
    source: str | None,
    universe_file: str | None,
    universe_limit: int | None,
    universe_tier: str | None,
    cache_root: str,
    retrain_days: int,
    forecast_backend: str,
    training_window_days: int | None,
    publish_forecast_models: bool,
    split_mode: str,
    embargo_days: int,
) -> InfoPublishArtifacts:
    info_shadow_enabled = False
    info_shadow_report = _default_info_shadow_report(settings)
    info_file_path = dependencies.resolve_info_file_from_settings_fn(settings)
    info_as_of_date = pd.Timestamp(
        context.learned_meta.end_date or context.baseline_meta.end_date or settings.get("end", "today")
    ).normalize()
    info_items: list[InfoItem] = []
    trajectory = forecast_artifacts.trajectory
    info_payload_requested = info_payload_enabled_for_settings(settings)
    info_shadow_requested = info_shadow_enabled_for_settings(settings)
    if info_payload_requested:
        if trajectory is None:
            trajectory = dependencies.load_or_build_v2_backtest_trajectory_fn(
                config_path=str(settings.get("config_path", config_path)),
                source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
                universe_file=(
                    str(settings.get("universe_file", universe_file))
                    if settings.get("universe_file", universe_file) is not None
                    else None
                ),
                universe_limit=(
                    int(settings.get("universe_limit"))
                    if settings.get("universe_limit") is not None
                    else (int(universe_limit) if universe_limit is not None else None)
                ),
                universe_tier=str(settings.get("universe_tier", universe_tier)),
                dynamic_universe=dependencies.parse_boolish_fn(settings.get("dynamic_universe_enabled", False), False),
                generator_target_size=int(settings.get("generator_target_size", settings.get("universe_limit", 0)) or 0),
                generator_coarse_size=int(settings.get("generator_coarse_size", 0) or 0),
                generator_theme_aware=dependencies.parse_boolish_fn(settings.get("generator_theme_aware", True), True),
                generator_use_concepts=dependencies.parse_boolish_fn(settings.get("generator_use_concepts", True), True),
                retrain_days=int(retrain_days),
                cache_root=cache_root,
                refresh_cache=False,
                forecast_backend=forecast_backend,
                training_window_days=(
                    int(settings.get("training_window_days"))
                    if settings.get("training_window_days") is not None
                    else training_window_days
                ),
                use_us_index_context=bool(settings.get("use_us_index_context", False)),
                us_index_source=str(settings.get("us_index_source", "akshare")),
            )
        info_file_path, info_items = dependencies.load_v2_info_items_for_date_fn(
            settings=settings,
            as_of_date=info_as_of_date,
            learned_window=True,
        )
        if trajectory is not None and info_items and info_shadow_requested:
            _, validation_traj, holdout_traj = dependencies.split_research_trajectory_fn(
                trajectory,
                split_mode=split_mode,
                embargo_days=embargo_days,
            )
            info_shadow_report = dependencies.build_info_shadow_report_fn(
                validation_trajectory=validation_traj,
                holdout_trajectory=holdout_traj,
                settings=settings,
                info_items=info_items,
            )
            info_shadow_enabled = True
    info_manifest = dependencies.build_info_manifest_payload_fn(
        settings=settings,
        info_file=info_file_path,
        info_items=info_items,
        as_of_date=info_as_of_date,
        config_hash=context.config_hash,
        shadow_enabled=info_shadow_enabled,
        shadow_report=info_shadow_report,
    )
    info_hash = str(info_manifest.get("info_hash", ""))
    external_signal_package = dependencies.build_external_signal_package_for_date_fn(
        settings=settings,
        as_of_date=info_as_of_date,
        info_items=info_items,
    )
    external_signal_manifest = dict(external_signal_package.get("manifest", {}))
    info_manifest = merge_external_signal_manifest_summary(
        info_manifest=info_manifest,
        external_signal_manifest=external_signal_manifest,
    )

    decorated_daily_state = dict(forecast_artifacts.frozen_daily_state)
    decorated_forecast_bundle = dict(forecast_artifacts.frozen_forecast_bundle)
    if publish_forecast_models and decorated_daily_state:
        frozen_composite = dependencies.decode_composite_state_fn(decorated_daily_state.get("composite_state"))
        if frozen_composite is not None:
            if bool(settings.get("use_info_fusion", False)) and info_items:
                frozen_composite = dependencies.enrich_state_with_info_fn(
                    state=frozen_composite,
                    as_of_date=info_as_of_date,
                    info_items=info_items,
                    settings=settings,
                )
            frozen_composite, _ = dependencies.attach_external_signals_to_composite_state_fn(
                state=frozen_composite,
                settings=settings,
                as_of_date=info_as_of_date,
                info_items=info_items,
            )
            frozen_composite = dependencies.attach_insight_memory_to_state_fn(
                state=frozen_composite,
                settings=settings,
                as_of_date=info_as_of_date,
                info_items=info_items,
            )
            decorated_daily_state["composite_state"] = dependencies.serialize_composite_state_fn(frozen_composite)
        decorated_daily_state = add_artifact_metadata(
            decorated_daily_state,
            artifact_type="frozen_daily_state",
        )
    if publish_forecast_models and decorated_forecast_bundle:
        decorated_forecast_bundle = add_artifact_metadata(
            decorated_forecast_bundle,
            artifact_type="forecast_bundle",
        )

    return InfoPublishArtifacts(
        info_shadow_enabled=info_shadow_enabled,
        info_shadow_report=info_shadow_report,
        info_file_path=str(info_file_path),
        info_as_of_date=info_as_of_date,
        info_items=info_items,
        info_manifest=info_manifest,
        info_hash=info_hash,
        external_signal_package=external_signal_package,
        external_signal_manifest=external_signal_manifest,
        frozen_daily_state=decorated_daily_state,
        frozen_forecast_bundle=decorated_forecast_bundle,
    )


def _build_insight_publish_artifacts(
    *,
    dependencies: ResearchPublishDependencies,
    settings: dict[str, object],
    info_artifacts: InfoPublishArtifacts,
) -> InsightPublishArtifacts:
    frozen_composite = None
    if isinstance(info_artifacts.frozen_daily_state, dict):
        frozen_composite = dependencies.decode_composite_state_fn(
            info_artifacts.frozen_daily_state.get("composite_state")
        )
    if frozen_composite is None:
        return InsightPublishArtifacts(
            manifest={
                "as_of_date": "",
                "enable_insight_memory": bool(settings.get("enable_insight_memory", True)),
                "insight_notes_dir": str(settings.get("insight_notes_dir", "input/insight_notes")),
                "viewpoint_count": 0,
                "theme_episode_count": 0,
                "stock_role_count": 0,
                "execution_plan_count": 0,
                "source_breakdown": {},
                "phase_counts": {},
                "role_counts": {},
                "role_downgrade_count": 0,
            },
            viewpoints=[],
            theme_episodes=[],
            stock_roles=[],
            execution_plan=[],
        )
    if getattr(frozen_composite, "viewpoints", None) or getattr(frozen_composite, "theme_episodes", None):
        insight_state = frozen_composite
    else:
        insight_state = dependencies.attach_insight_memory_to_state_fn(
            state=frozen_composite,
            settings=settings,
            as_of_date=info_artifacts.info_as_of_date,
            info_items=info_artifacts.info_items,
        )
    payloads = dependencies.build_insight_artifact_payloads_fn(
        state=insight_state,
        settings=settings,
        execution_plans=[],
    )
    return InsightPublishArtifacts(
        manifest=dict(payloads.get("insight_manifest", {})),
        viewpoints=[dict(item) for item in payloads.get("viewpoints", []) if isinstance(item, dict)],
        theme_episodes=[dict(item) for item in payloads.get("theme_episodes", []) if isinstance(item, dict)],
        stock_roles=[dict(item) for item in payloads.get("stock_roles", []) if isinstance(item, dict)],
        execution_plan=[dict(item) for item in payloads.get("execution_plan", []) if isinstance(item, dict)],
    )


def _build_leader_publish_artifacts(
    *,
    dependencies: ResearchPublishDependencies,
    settings: dict[str, object],
    info_artifacts: InfoPublishArtifacts,
    forecast_artifacts: ForecastPublishArtifacts,
    split_mode: str,
    embargo_days: int,
) -> LeaderPublishArtifacts:
    frozen_composite = None
    if isinstance(info_artifacts.frozen_daily_state, dict):
        frozen_composite = dependencies.decode_composite_state_fn(
            info_artifacts.frozen_daily_state.get("composite_state")
        )
    fit_trajectory = None
    evaluation_trajectory = None
    if forecast_artifacts.trajectory is not None:
        train_trajectory, validation_trajectory, evaluation_trajectory = dependencies.split_research_trajectory_fn(
            forecast_artifacts.trajectory,
            split_mode=split_mode,
            embargo_days=embargo_days,
        )
        fit_steps = list(getattr(train_trajectory, "steps", []) or []) + list(getattr(validation_trajectory, "steps", []) or [])
        if fit_steps:
            fit_trajectory = type(forecast_artifacts.trajectory)(
                prepared=getattr(forecast_artifacts.trajectory, "prepared", None),
                steps=fit_steps,
            )
    payloads = build_leader_artifact_payloads(
        state=frozen_composite,
        trajectory=evaluation_trajectory,
        top_k=3,
        limit=16,
    )
    label_payloads = build_research_label_artifact_payloads(
        trajectory=forecast_artifacts.trajectory,
        min_leader_theme_size=3,
        min_exit_theme_size=2,
        exit_candidate_limit=8,
    )
    fit_label_payloads = build_research_label_artifact_payloads(
        trajectory=fit_trajectory,
        min_leader_theme_size=3,
        min_exit_theme_size=2,
        exit_candidate_limit=8,
    )
    evaluation_label_payloads = build_research_label_artifact_payloads(
        trajectory=evaluation_trajectory,
        min_leader_theme_size=3,
        min_exit_theme_size=2,
        exit_candidate_limit=8,
    )
    signal_payloads = build_signal_training_artifacts(
        leader_fit_rows=[dict(item) for item in fit_label_payloads.get("leader_training_labels", []) if isinstance(item, dict)],
        leader_evaluation_rows=[
            dict(item) for item in evaluation_label_payloads.get("leader_training_labels", []) if isinstance(item, dict)
        ],
        exit_fit_rows=[dict(item) for item in fit_label_payloads.get("exit_training_labels", []) if isinstance(item, dict)],
        exit_evaluation_rows=[
            dict(item) for item in evaluation_label_payloads.get("exit_training_labels", []) if isinstance(item, dict)
        ],
        l2=1.0,
    )
    manifest = dict(payloads.get("leader_manifest", {}))
    manifest["training_labels"] = dict(label_payloads.get("training_label_manifest", {}))
    manifest["signal_models"] = dict(signal_payloads.get("signal_training_manifest", {}))
    leader_rank_model = add_artifact_metadata(
        dict(signal_payloads.get("leader_rank_model", {})),
        artifact_type="leader_rank_model",
    )
    exit_behavior_model = add_artifact_metadata(
        dict(signal_payloads.get("exit_behavior_model", {})),
        artifact_type="exit_behavior_model",
    )
    return LeaderPublishArtifacts(
        manifest=manifest,
        candidates=[dict(item) for item in payloads.get("leader_candidates", []) if isinstance(item, dict)],
        training_label_manifest=dict(label_payloads.get("training_label_manifest", {})),
        leader_training_labels=[
            dict(item) for item in label_payloads.get("leader_training_labels", []) if isinstance(item, dict)
        ],
        exit_training_labels=[
            dict(item) for item in label_payloads.get("exit_training_labels", []) if isinstance(item, dict)
        ],
        signal_training_manifest=dict(signal_payloads.get("signal_training_manifest", {})),
        leader_rank_model=leader_rank_model,
        exit_behavior_model=exit_behavior_model,
    )


def _evaluate_publish_gates(
    *,
    dependencies: ResearchPublishDependencies,
    artifact_root: str,
    strategy_id: str,
    context: PublishRuntimeContext,
    paths: PublishPaths,
) -> PublishGateArtifacts:
    gate_ok, gate_reasons = dependencies.pass_release_gate_fn(
        baseline=context.baseline_meta,
        candidate=context.learned_meta,
    )
    previous_manifest = dependencies.load_json_dict_fn(paths.tier_latest_manifest_path)
    previous_gate_ok = False
    previous_reason = "missing previous same-tier latest manifest"
    if previous_manifest:
        previous_backtest = dependencies.load_backtest_payload_from_manifest_fn(
            previous_manifest,
            paths.tier_latest_manifest_path,
        )
        prev_baseline_payload = previous_backtest.get("baseline", {}) if isinstance(previous_backtest, dict) else {}
        prev_learned_payload = previous_backtest.get("learned", {}) if isinstance(previous_backtest, dict) else {}
        if isinstance(prev_baseline_payload, dict) and isinstance(prev_learned_payload, dict):
            prev_baseline = dependencies.summary_from_payload_fn(context.baseline_meta, prev_baseline_payload)
            prev_learned = dependencies.summary_from_payload_fn(context.learned_meta, prev_learned_payload)
            previous_gate_ok, previous_reasons = dependencies.pass_release_gate_fn(
                baseline=prev_baseline,
                candidate=prev_learned,
            )
            previous_reason = "" if previous_gate_ok else "; ".join(previous_reasons)
    release_gate_passed = bool(gate_ok and previous_gate_ok)
    release_gate = {
        "current_passed": bool(gate_ok),
        "current_reasons": gate_reasons,
        "previous_passed": bool(previous_gate_ok),
        "previous_reason": previous_reason,
        "require_two_consecutive": True,
        "passed": bool(release_gate_passed),
    }

    baseline_reference_payload = dependencies.load_backtest_payload_for_run_fn(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        run_id=context.baseline_reference_run_id,
    )
    baseline_reference_learned_payload = (
        baseline_reference_payload.get("learned", {})
        if isinstance(baseline_reference_payload, dict)
        else {}
    )
    baseline_reference_summary = (
        dependencies.summary_from_payload_fn(context.learned_meta, baseline_reference_learned_payload)
        if isinstance(baseline_reference_learned_payload, dict) and baseline_reference_learned_payload
        else context.learned_meta
    )
    switch_current_ok = False
    switch_current_reasons = ["switch gate skipped: not candidate default universe tier"]
    switch_previous_ok = False
    switch_previous_reason = "missing previous same-tier switch gate"
    switch_deltas = {
        "excess_annual_return_delta": 0.0,
        "information_ratio_delta": 0.0,
        "max_drawdown_diff": 0.0,
    }
    if context.universe_tier_value == context.candidate_default_universe_tier and context.baseline_reference_run_id:
        switch_current_ok, switch_current_reasons, switch_deltas = dependencies.pass_default_switch_gate_fn(
            baseline_reference=baseline_reference_summary,
            candidate=context.learned_meta,
        )
        if previous_manifest:
            previous_switch_gate = previous_manifest.get("default_switch_gate", {})
            if isinstance(previous_switch_gate, dict):
                switch_previous_ok = bool(previous_switch_gate.get("current_passed", False))
                switch_previous_reason = "" if switch_previous_ok else str(previous_switch_gate.get("current_reasons", ""))
    default_switch_gate_passed = bool(release_gate_passed and switch_current_ok and switch_previous_ok)
    default_switch_gate = {
        "baseline_reference_run_id": context.baseline_reference_run_id,
        "current_passed": bool(switch_current_ok),
        "current_reasons": switch_current_reasons,
        "previous_passed": bool(switch_previous_ok),
        "previous_reason": switch_previous_reason,
        "require_two_consecutive": True,
        "deltas": switch_deltas,
        "passed": bool(default_switch_gate_passed),
    }
    return PublishGateArtifacts(
        gate_ok=bool(gate_ok),
        release_gate_passed=release_gate_passed,
        release_gate=release_gate,
        default_switch_gate_passed=default_switch_gate_passed,
        default_switch_gate=default_switch_gate,
    )


def _write_publish_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_and_publish_artifacts(
    *,
    dependencies: ResearchPublishDependencies,
    context: PublishRuntimeContext,
    paths: PublishPaths,
    forecast_artifacts: ForecastPublishArtifacts,
    info_artifacts: InfoPublishArtifacts,
    insight_artifacts: InsightPublishArtifacts,
    leader_artifacts: LeaderPublishArtifacts,
    manifests: PublishManifestArtifacts,
    gate_artifacts: PublishGateArtifacts,
    publish_forecast_models: bool,
    update_latest: bool,
) -> None:
    paths.latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    _write_publish_json(paths.dataset_path, manifests.dataset_manifest)
    _write_publish_json(paths.calibration_path, manifests.calibration_manifest)
    _write_publish_json(paths.learning_path, context.learning_manifest)
    if publish_forecast_models:
        _write_publish_json(paths.forecast_models_path, forecast_artifacts.forecast_models_manifest)
        _write_publish_json(paths.frozen_forecast_bundle_path, info_artifacts.frozen_forecast_bundle)
        _write_publish_json(paths.frozen_state_path, info_artifacts.frozen_daily_state)
    _write_publish_json(paths.backtest_path, manifests.backtest_manifest)
    _write_publish_json(paths.consistency_path, manifests.consistency_manifest)
    _write_publish_json(paths.rolling_oos_path, manifests.rolling_oos_manifest)
    _write_publish_json(paths.info_manifest_path, info_artifacts.info_manifest)
    _write_publish_json(paths.info_shadow_report_path, info_artifacts.info_shadow_report)
    _write_publish_json(paths.external_signal_manifest_path, info_artifacts.external_signal_manifest)
    _write_publish_json(paths.insight_manifest_path, insight_artifacts.manifest)
    _write_publish_json(paths.viewpoints_path, {"items": insight_artifacts.viewpoints})
    _write_publish_json(paths.theme_episodes_path, {"items": insight_artifacts.theme_episodes})
    _write_publish_json(paths.stock_roles_path, {"items": insight_artifacts.stock_roles})
    _write_publish_json(paths.execution_plan_path, {"items": insight_artifacts.execution_plan})
    _write_publish_json(paths.leader_manifest_path, leader_artifacts.manifest)
    _write_publish_json(paths.leader_candidates_path, {"items": leader_artifacts.candidates})
    _write_publish_json(paths.leader_training_labels_path, {"items": leader_artifacts.leader_training_labels})
    _write_publish_json(paths.exit_training_labels_path, {"items": leader_artifacts.exit_training_labels})
    _write_publish_json(paths.leader_rank_model_path, leader_artifacts.leader_rank_model)
    _write_publish_json(paths.exit_behavior_model_path, leader_artifacts.exit_behavior_model)
    _write_publish_json(paths.manifest_path, manifests.research_manifest)
    _write_publish_json(paths.tier_latest_manifest_path, manifests.research_manifest)
    if gate_artifacts.gate_ok:
        _write_publish_json(paths.tier_latest_policy_path, context.learning_manifest)
    allow_default_latest_update = bool(
        not context.universe_tier_value or context.universe_tier_value == context.active_default_universe_tier
    )
    if update_latest and gate_artifacts.release_gate_passed and allow_default_latest_update:
        _write_publish_json(paths.latest_policy_path, context.learning_manifest)
        _write_publish_json(paths.latest_manifest_path, manifests.research_manifest)
    elif update_latest and gate_artifacts.release_gate_passed and not allow_default_latest_update:
        note = "current universe tier is not the active default tier; updated tier latest only"
        if gate_artifacts.default_switch_gate_passed:
            note += "; default switch gate passed and can be promoted after active tier changes"
        dependencies.emit_progress_fn("publish", note)
    elif not gate_artifacts.release_gate_passed:
        dependencies.emit_progress_fn("publish", "release gate not passed; latest policy/manifest not updated")


def _build_publish_manifests(
    *,
    dependencies: ResearchPublishDependencies,
    strategy_id: str,
    settings: dict[str, object],
    calibration: V2CalibrationResult,
    context: PublishRuntimeContext,
    paths: PublishPaths,
    forecast_artifacts: ForecastPublishArtifacts,
    info_artifacts: InfoPublishArtifacts,
    insight_artifacts: InsightPublishArtifacts,
    leader_artifacts: LeaderPublishArtifacts,
    split_mode: str,
    embargo_days: int,
    publish_forecast_models: bool,
    release_gate: dict[str, object],
    default_switch_gate: dict[str, object],
) -> PublishManifestArtifacts:
    dataset_manifest = add_artifact_metadata(
        {
            "strategy_id": str(strategy_id),
            "config_path": str(settings.get("config_path", "")),
            "source": str(settings.get("source", "")),
            "watchlist": str(settings.get("watchlist", "")),
            "universe_tier": context.universe_tier_value,
            "universe_id": context.universe_id,
            "universe_size": int(context.universe_size),
            "universe_generation_rule": context.universe_generation_rule,
            "source_universe_manifest_path": context.source_universe_manifest_path,
            "universe_file": str(settings.get("universe_file", "")),
            "universe_limit": int(settings.get("universe_limit", 0)),
            "dynamic_universe_enabled": bool(settings.get("dynamic_universe_enabled", False)),
            "generator_manifest": str(settings.get("generator_manifest_path", "")),
            "generator_version": str(settings.get("generator_version", "")),
            "generator_hash": str(settings.get("generator_hash", "")),
            "coarse_pool_size": int(settings.get("coarse_pool_size", 0)),
            "refined_pool_size": int(settings.get("refined_pool_size", 0)),
            "selected_pool_size": int(settings.get("selected_pool_size", 0)),
            "theme_allocations": [dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
            "use_us_index_context": bool(settings.get("use_us_index_context", False)),
            "us_index_source": str(settings.get("us_index_source", "akshare")),
            "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
            "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
            "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
            "start": str(settings.get("start", "")),
            "end": str(settings.get("end", "")),
            "symbols": context.symbols,
            "symbol_count": int(settings.get("symbol_count", len(context.symbols))),
            "universe_hash": context.universe_hash,
            "config_hash": context.config_hash,
            "info_file": str(info_artifacts.info_file_path),
            "event_file": str(settings.get("event_file", info_artifacts.info_file_path)),
            "info_hash": info_artifacts.info_hash,
            "info_shadow_enabled": bool(info_artifacts.info_shadow_enabled),
            "use_info_fusion": bool(settings.get("use_info_fusion", False)),
            "use_learned_info_fusion": bool(
                settings.get("use_learned_info_fusion", settings.get("use_learned_news_fusion", False))
            ),
            "info_shadow_only": bool(settings.get("info_shadow_only", True)),
            "info_item_count": int(info_artifacts.info_manifest.get("info_item_count", 0)),
            "info_source_mode": str(settings.get("info_source_mode", "layered")),
            "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
            "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
            "capital_flow_file": str(settings.get("capital_flow_file", "")),
            "macro_file": str(settings.get("macro_file", "")),
            "external_signal_manifest": str(paths.external_signal_manifest_path),
            "external_signal_version": str(settings.get("external_signal_version", "v1")),
            "external_signal_enabled": bool(settings.get("external_signals", True)),
            "enable_insight_memory": bool(settings.get("enable_insight_memory", True)),
            "insight_notes_dir": str(settings.get("insight_notes_dir", "input/insight_notes")),
            "execution_overlay_enabled": bool(settings.get("execution_overlay_enabled", True)),
            "event_lookback_days": int(settings.get("event_lookback_days", settings.get("info_lookback_days", 45))),
            "capital_flow_lookback_days": int(settings.get("capital_flow_lookback_days", 20)),
            "macro_lookback_days": int(settings.get("macro_lookback_days", 60)),
            "event_risk_cutoff": float(settings.get("event_risk_cutoff", 0.55)),
            "catalyst_boost_cap": float(settings.get("catalyst_boost_cap", 0.12)),
            "flow_exposure_cap": float(settings.get("flow_exposure_cap", 0.08)),
            "capital_flow_snapshot": dict(info_artifacts.external_signal_package.get("capital_flow_snapshot", {})),
            "macro_context_snapshot": dict(info_artifacts.external_signal_package.get("macro_context_snapshot", {})),
            "active_default_universe_tier": context.active_default_universe_tier,
            "candidate_default_universe_tier": context.candidate_default_universe_tier,
        },
        artifact_type="dataset_manifest",
    )
    calibration_manifest = add_artifact_metadata(
        {
            "best_score": float(calibration.best_score),
            "best_policy": asdict(calibration.best_policy),
            "trials": calibration.trials,
            "policy_hash": dependencies.stable_json_hash_fn(asdict(calibration.best_policy)),
        },
        artifact_type="policy_calibration",
    )
    backtest_manifest = add_artifact_metadata(
        {
            "baseline": asdict(context.baseline_meta),
            "calibrated": asdict(context.calibrated_meta),
            "learned": asdict(context.learned_meta),
        },
        artifact_type="backtest_summary",
    )
    consistency_manifest = add_artifact_metadata(
        {
            "run_id": context.run_id,
            "universe_tier": context.universe_tier_value,
            "universe_id": context.universe_id,
            "universe_size": int(context.universe_size),
            "split_mode": str(split_mode),
            "embargo_days": int(embargo_days),
            "train_window": forecast_artifacts.train_window,
            "validation_window": forecast_artifacts.validation_window,
            "holdout_window": forecast_artifacts.holdout_window,
            "snapshot_hash": context.snapshot_hash,
            "config_hash": context.config_hash,
            "policy_hash": context.policy_hash,
            "universe_hash": context.universe_hash,
            "model_hashes": context.model_hashes,
            "info_hash": info_artifacts.info_hash,
            "info_source_mode": str(settings.get("info_source_mode", "layered")),
            "external_signal_enabled": bool(settings.get("external_signals", True)),
            "external_signal_version": str(settings.get("external_signal_version", "v1")),
            "enable_insight_memory": bool(settings.get("enable_insight_memory", True)),
            "execution_overlay_enabled": bool(settings.get("execution_overlay_enabled", True)),
            "use_us_index_context": bool(settings.get("use_us_index_context", False)),
            "us_index_source": str(settings.get("us_index_source", "akshare")),
        },
        artifact_type="consistency_checklist",
    )
    rolling_oos_manifest = add_artifact_metadata(
        {
            "run_id": context.run_id,
            "universe_tier": context.universe_tier_value,
            "windows": [
                {
                    "name": "window_1",
                    "start": context.learned_meta.start_date,
                    "end": context.learned_meta.end_date,
                    "excess_annual_return": float(context.learned_meta.excess_annual_return),
                    "information_ratio": float(context.learned_meta.information_ratio),
                    "max_drawdown": float(context.learned_meta.max_drawdown),
                },
                {
                    "name": "window_2",
                    "start": context.calibrated_meta.start_date,
                    "end": context.calibrated_meta.end_date,
                    "excess_annual_return": float(context.calibrated_meta.excess_annual_return),
                    "information_ratio": float(context.calibrated_meta.information_ratio),
                    "max_drawdown": float(context.calibrated_meta.max_drawdown),
                },
            ],
            "regime_breakdown": forecast_artifacts.regime_counts,
        },
        artifact_type="rolling_oos_report",
    )
    research_manifest = add_artifact_metadata(
        {
            "run_id": context.run_id,
            "strategy_id": str(strategy_id),
            "created_at": context.created_at,
            "baseline_reference_run_id": context.baseline_reference_run_id,
            "universe_tier": context.universe_tier_value,
            "universe_id": context.universe_id,
            "universe_size": int(context.universe_size),
            "universe_generation_rule": context.universe_generation_rule,
            "source_universe_manifest_path": context.source_universe_manifest_path,
            "info_hash": info_artifacts.info_hash,
            "info_shadow_enabled": bool(info_artifacts.info_shadow_enabled),
            "use_info_fusion": bool(settings.get("use_info_fusion", False)),
            "use_learned_info_fusion": bool(
                settings.get("use_learned_info_fusion", settings.get("use_learned_news_fusion", False))
            ),
            "info_source_mode": str(settings.get("info_source_mode", "layered")),
            "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
            "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
            "external_signal_version": str(settings.get("external_signal_version", "v1")),
            "external_signal_enabled": bool(settings.get("external_signals", True)),
            "use_us_index_context": bool(settings.get("use_us_index_context", False)),
            "us_index_source": str(settings.get("us_index_source", "akshare")),
            "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
            "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
            "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
            "dynamic_universe_enabled": bool(settings.get("dynamic_universe_enabled", False)),
            "generator_manifest": str(settings.get("generator_manifest_path", "")),
            "generator_version": str(settings.get("generator_version", "")),
            "generator_hash": str(settings.get("generator_hash", "")),
            "coarse_pool_size": int(settings.get("coarse_pool_size", 0)),
            "refined_pool_size": int(settings.get("refined_pool_size", 0)),
            "selected_pool_size": int(settings.get("selected_pool_size", 0)),
            "theme_allocations": [dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
            "capital_flow_snapshot": dict(info_artifacts.external_signal_package.get("capital_flow_snapshot", {})),
            "macro_context_snapshot": dict(info_artifacts.external_signal_package.get("macro_context_snapshot", {})),
            "enable_insight_memory": bool(settings.get("enable_insight_memory", True)),
            "insight_notes_dir": str(settings.get("insight_notes_dir", "input/insight_notes")),
            "execution_overlay_enabled": bool(settings.get("execution_overlay_enabled", True)),
            "use_us_index_context": bool(settings.get("use_us_index_context", False)),
            "us_index_source": str(settings.get("us_index_source", "akshare")),
            "data_window": {
                "start": str(settings.get("start", "")),
                "end": str(settings.get("end", "")),
            },
            "config_hash": context.config_hash,
            "snapshot_hash": context.snapshot_hash,
            "policy_hash": context.policy_hash,
            "universe_hash": context.universe_hash,
            "model_hashes": context.model_hashes,
            "split_mode": str(split_mode),
            "embargo_days": int(embargo_days),
            "dataset_manifest": str(paths.dataset_path),
            "policy_calibration": str(paths.calibration_path),
            "learned_policy_model": str(paths.learning_path),
            "forecast_models_manifest": str(paths.forecast_models_path) if publish_forecast_models else "",
            "frozen_forecast_bundle": str(paths.frozen_forecast_bundle_path) if publish_forecast_models else "",
            "frozen_daily_state": str(paths.frozen_state_path) if publish_forecast_models else "",
            "backtest_summary": str(paths.backtest_path),
            "consistency_checklist": str(paths.consistency_path),
            "rolling_oos_report": str(paths.rolling_oos_path),
            "info_manifest": str(paths.info_manifest_path),
            "info_shadow_report": str(paths.info_shadow_report_path),
            "external_signal_manifest": str(paths.external_signal_manifest_path),
            "insight_manifest": str(paths.insight_manifest_path),
            "viewpoints": str(paths.viewpoints_path),
            "theme_episodes": str(paths.theme_episodes_path),
            "stock_roles": str(paths.stock_roles_path),
            "execution_plan": str(paths.execution_plan_path),
            "insight_summary": insight_artifacts.manifest,
            "leader_manifest": str(paths.leader_manifest_path),
            "leader_candidates": str(paths.leader_candidates_path),
            "leader_training_labels": str(paths.leader_training_labels_path),
            "exit_training_labels": str(paths.exit_training_labels_path),
            "leader_rank_model": str(paths.leader_rank_model_path),
            "exit_behavior_model": str(paths.exit_behavior_model_path),
            "leader_summary": leader_artifacts.manifest,
            "training_label_summary": leader_artifacts.training_label_manifest,
            "signal_model_summary": leader_artifacts.signal_training_manifest,
            "published_policy_model": str(paths.latest_policy_path),
            "latest_research_manifest": str(paths.latest_manifest_path),
            "tier_published_policy_model": str(paths.tier_latest_policy_path),
            "tier_latest_research_manifest": str(paths.tier_latest_manifest_path),
            "release_gate": release_gate,
            "default_switch_gate": default_switch_gate,
        },
        artifact_type="research_manifest",
    )
    return PublishManifestArtifacts(
        dataset_manifest=dataset_manifest,
        calibration_manifest=calibration_manifest,
        backtest_manifest=backtest_manifest,
        consistency_manifest=consistency_manifest,
        rolling_oos_manifest=rolling_oos_manifest,
        research_manifest=research_manifest,
    )
