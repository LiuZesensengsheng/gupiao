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
    forecast_backend: str = "linear",
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
    )
    settings = dependencies.resolve_v2_universe_settings_fn(settings=dict(settings), cache_root=cache_root)
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
    universe_hash = str(settings.get("universe_hash", "")) or dependencies.sha256_file_fn(universe_path) or dependencies.stable_json_hash_fn(symbols)
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

    def _window_payload(trajectory: Any | None) -> dict[str, object]:
        if trajectory is None or not getattr(trajectory, "steps", None):
            return {"start": "", "end": "", "n_steps": 0}
        return {
            "start": str(trajectory.steps[0].date.date()),
            "end": str(trajectory.steps[-1].next_date.date()),
            "n_steps": int(len(trajectory.steps)),
        }

    trajectory = None
    frozen_daily_state: dict[str, object] = {}
    forecast_models_manifest: dict[str, object] = {}
    frozen_forecast_bundle: dict[str, object] = {}
    train_window = {"start": "", "end": "", "n_steps": 0}
    validation_window = {"start": "", "end": "", "n_steps": 0}
    holdout_window = {"start": "", "end": "", "n_steps": 0}
    regime_counts: dict[str, int] = {}
    if publish_forecast_models:
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
            cache_root=cache_root,
            refresh_cache=False,
            forecast_backend=forecast_backend,
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
                "run_id": run_id,
                "strategy_id": str(strategy_id),
                "forecast_backend": str(forecast_backend),
                "split_mode": str(split_mode),
                "embargo_days": int(embargo_days),
                "use_us_index_context": bool(settings.get("use_us_index_context", False)),
                "us_index_source": str(settings.get("us_index_source", "akshare")),
                "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
                "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
                "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
                "model_hashes": model_hashes,
                "data_window": {
                    "start": str(settings.get("start", "")),
                    "end": str(settings.get("end", "")),
                },
                "regime_counts": regime_counts,
                "frozen_bundle_ready": bool(frozen_forecast_bundle),
            },
            artifact_type="forecast_models_manifest",
        )

    dataset_path = base_dir / "dataset_manifest.json"
    calibration_path = base_dir / "policy_calibration.json"
    learning_path = base_dir / "learned_policy_model.json"
    forecast_models_path = base_dir / "forecast_models_manifest.json"
    frozen_forecast_bundle_path = base_dir / "frozen_forecast_bundle.json"
    frozen_state_path = base_dir / "frozen_daily_state.json"
    backtest_path = base_dir / "backtest_summary.json"
    consistency_path = base_dir / "consistency_checklist.json"
    rolling_oos_path = base_dir / "rolling_oos_report.json"
    info_manifest_path = base_dir / "info_manifest.json"
    info_shadow_report_path = base_dir / "info_shadow_report.json"
    external_signal_manifest_path = ensure_external_signal_manifest_path(base_dir)
    manifest_path = base_dir / "research_manifest.json"
    latest_policy_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    latest_manifest_path = Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json"
    tier_latest_policy_path = dependencies.tier_latest_policy_path_fn(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        universe_tier=universe_tier_value,
    )
    tier_latest_manifest_path = dependencies.tier_latest_manifest_path_fn(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        universe_tier=universe_tier_value,
    )

    info_artifacts = _build_info_publish_artifacts(
        dependencies=dependencies,
        settings=settings,
        learned_meta=learned_meta,
        baseline_meta=baseline_meta,
        trajectory=trajectory,
        frozen_daily_state=frozen_daily_state,
        frozen_forecast_bundle=frozen_forecast_bundle,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        cache_root=cache_root,
        forecast_backend=forecast_backend,
        publish_forecast_models=publish_forecast_models,
        split_mode=split_mode,
        embargo_days=embargo_days,
        config_hash=config_hash,
    )
    info_shadow_enabled = info_artifacts.info_shadow_enabled
    info_shadow_report = info_artifacts.info_shadow_report
    info_file_path = info_artifacts.info_file_path
    info_manifest = info_artifacts.info_manifest
    info_hash = info_artifacts.info_hash
    external_signal_package = info_artifacts.external_signal_package
    external_signal_manifest = info_artifacts.external_signal_manifest
    frozen_daily_state = info_artifacts.frozen_daily_state
    frozen_forecast_bundle = info_artifacts.frozen_forecast_bundle

    gate_artifacts = _evaluate_publish_gates(
        dependencies=dependencies,
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        tier_latest_manifest_path=tier_latest_manifest_path,
        baseline_meta=baseline_meta,
        learned_meta=learned_meta,
        baseline_reference_run_id=baseline_reference_run_id,
        universe_tier_value=universe_tier_value,
        candidate_default_universe_tier=candidate_default_universe_tier,
    )
    release_gate_passed = gate_artifacts.release_gate_passed
    default_switch_gate_passed = gate_artifacts.default_switch_gate_passed
    manifests = _build_publish_manifests(
        dependencies=dependencies,
        strategy_id=strategy_id,
        settings=settings,
        calibration=calibration,
        run_id=run_id,
        created_at=created_at,
        universe_tier_value=universe_tier_value,
        universe_id=universe_id,
        universe_size=universe_size,
        universe_generation_rule=universe_generation_rule,
        source_universe_manifest_path=source_universe_manifest_path,
        active_default_universe_tier=active_default_universe_tier,
        candidate_default_universe_tier=candidate_default_universe_tier,
        baseline_reference_run_id=baseline_reference_run_id,
        config_hash=config_hash,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        model_hashes=model_hashes,
        snapshot_hash=snapshot_hash,
        baseline_meta=baseline_meta,
        calibrated_meta=calibrated_meta,
        learned_meta=learned_meta,
        symbols=symbols,
        train_window=train_window,
        validation_window=validation_window,
        holdout_window=holdout_window,
        regime_counts=regime_counts,
        info_artifacts=info_artifacts,
        split_mode=split_mode,
        embargo_days=embargo_days,
        publish_forecast_models=publish_forecast_models,
        dataset_path=dataset_path,
        calibration_path=calibration_path,
        learning_path=learning_path,
        forecast_models_path=forecast_models_path,
        frozen_forecast_bundle_path=frozen_forecast_bundle_path,
        frozen_state_path=frozen_state_path,
        backtest_path=backtest_path,
        consistency_path=consistency_path,
        rolling_oos_path=rolling_oos_path,
        info_manifest_path=info_manifest_path,
        info_shadow_report_path=info_shadow_report_path,
        external_signal_manifest_path=external_signal_manifest_path,
        latest_policy_path=latest_policy_path,
        latest_manifest_path=latest_manifest_path,
        tier_latest_policy_path=tier_latest_policy_path,
        tier_latest_manifest_path=tier_latest_manifest_path,
        release_gate=gate_artifacts.release_gate,
        default_switch_gate=gate_artifacts.default_switch_gate,
    )

    _write_and_publish_artifacts(
        dependencies=dependencies,
        dataset_path=dataset_path,
        calibration_path=calibration_path,
        learning_path=learning_path,
        forecast_models_path=forecast_models_path,
        frozen_forecast_bundle_path=frozen_forecast_bundle_path,
        frozen_state_path=frozen_state_path,
        backtest_path=backtest_path,
        consistency_path=consistency_path,
        rolling_oos_path=rolling_oos_path,
        info_manifest_path=info_manifest_path,
        info_shadow_report_path=info_shadow_report_path,
        external_signal_manifest_path=external_signal_manifest_path,
        manifest_path=manifest_path,
        latest_policy_path=latest_policy_path,
        latest_manifest_path=latest_manifest_path,
        tier_latest_policy_path=tier_latest_policy_path,
        tier_latest_manifest_path=tier_latest_manifest_path,
        learning_manifest=learning_manifest,
        forecast_models_manifest=forecast_models_manifest,
        info_artifacts=info_artifacts,
        manifests=manifests,
        gate_artifacts=gate_artifacts,
        publish_forecast_models=publish_forecast_models,
        update_latest=update_latest,
        universe_tier_value=universe_tier_value,
        active_default_universe_tier=active_default_universe_tier,
    )

    memory_path = dependencies.remember_research_run_fn(
        memory_root=Path(str(artifact_root)) / "memory",
        strategy_id=strategy_id,
        run_id=run_id,
        baseline=baseline_meta,
        calibration=calibration,
        learning=learning,
        release_gate_passed=release_gate_passed,
        universe_id=universe_id,
        universe_tier=universe_tier_value,
        universe_size=int(universe_size),
        external_signal_version=str(settings.get("external_signal_version", "v1")),
        external_signal_enabled=bool(settings.get("external_signals", True)),
    )

    return {
        "run_dir": str(base_dir),
        "run_id": run_id,
        "baseline_reference_run_id": baseline_reference_run_id,
        "universe_tier": universe_tier_value,
        "universe_id": universe_id,
        "universe_size": str(universe_size),
        "source_universe_manifest_path": source_universe_manifest_path,
        "info_manifest": str(info_manifest_path),
        "info_shadow_report": str(info_shadow_report_path),
        "info_hash": info_hash,
        "info_item_count": str(info_manifest.get("info_item_count", 0)),
        "info_shadow_enabled": "true" if info_shadow_enabled else "false",
        "external_signal_manifest": str(external_signal_manifest_path),
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
        "dataset_manifest": str(dataset_path),
        "policy_calibration": str(calibration_path),
        "learned_policy_model": str(learning_path),
        "forecast_models_manifest": str(forecast_models_path) if publish_forecast_models else "",
        "frozen_forecast_bundle": str(frozen_forecast_bundle_path) if publish_forecast_models else "",
        "frozen_daily_state": str(frozen_state_path) if publish_forecast_models else "",
        "backtest_summary": str(backtest_path),
        "consistency_checklist": str(consistency_path),
        "rolling_oos_report": str(rolling_oos_path),
        "research_manifest": str(manifest_path),
        "published_policy_model": str(latest_policy_path),
        "strategy_memory": str(memory_path),
        "capital_flow_snapshot": json.dumps(external_signal_package.get("capital_flow_snapshot", {}), ensure_ascii=False),
        "macro_context_snapshot": json.dumps(external_signal_package.get("macro_context_snapshot", {}), ensure_ascii=False),
        "release_gate_passed": "true" if release_gate_passed else "false",
        "default_switch_gate_passed": "true" if default_switch_gate_passed else "false",
        "snapshot_hash": snapshot_hash,
        "config_hash": config_hash,
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
    learned_meta: V2BacktestSummary,
    baseline_meta: V2BacktestSummary,
    trajectory: Any | None,
    frozen_daily_state: dict[str, object],
    frozen_forecast_bundle: dict[str, object],
    config_path: str,
    source: str | None,
    universe_file: str | None,
    universe_limit: int | None,
    universe_tier: str | None,
    cache_root: str,
    forecast_backend: str,
    publish_forecast_models: bool,
    split_mode: str,
    embargo_days: int,
    config_hash: str,
) -> InfoPublishArtifacts:
    info_shadow_enabled = False
    info_shadow_report = _default_info_shadow_report(settings)
    info_file_path = dependencies.resolve_info_file_from_settings_fn(settings)
    info_as_of_date = pd.Timestamp(
        learned_meta.end_date or baseline_meta.end_date or settings.get("end", "today")
    ).normalize()
    info_items: list[InfoItem] = []
    if bool(settings.get("use_info_fusion", False)):
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
                cache_root=cache_root,
                refresh_cache=False,
                forecast_backend=forecast_backend,
                use_us_index_context=bool(settings.get("use_us_index_context", False)),
                us_index_source=str(settings.get("us_index_source", "akshare")),
            )
        info_file_path, info_items = dependencies.load_v2_info_items_for_date_fn(
            settings=settings,
            as_of_date=info_as_of_date,
            learned_window=True,
        )
        if trajectory is not None and info_items:
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
        config_hash=config_hash,
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

    decorated_daily_state = dict(frozen_daily_state)
    decorated_forecast_bundle = dict(frozen_forecast_bundle)
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


def _evaluate_publish_gates(
    *,
    dependencies: ResearchPublishDependencies,
    artifact_root: str,
    strategy_id: str,
    tier_latest_manifest_path: Path,
    baseline_meta: V2BacktestSummary,
    learned_meta: V2BacktestSummary,
    baseline_reference_run_id: str,
    universe_tier_value: str,
    candidate_default_universe_tier: str,
) -> PublishGateArtifacts:
    gate_ok, gate_reasons = dependencies.pass_release_gate_fn(
        baseline=baseline_meta,
        candidate=learned_meta,
    )
    previous_manifest = dependencies.load_json_dict_fn(tier_latest_manifest_path)
    previous_gate_ok = False
    previous_reason = "missing previous same-tier latest manifest"
    if previous_manifest:
        previous_backtest = dependencies.load_backtest_payload_from_manifest_fn(previous_manifest, tier_latest_manifest_path)
        prev_baseline_payload = previous_backtest.get("baseline", {}) if isinstance(previous_backtest, dict) else {}
        prev_learned_payload = previous_backtest.get("learned", {}) if isinstance(previous_backtest, dict) else {}
        if isinstance(prev_baseline_payload, dict) and isinstance(prev_learned_payload, dict):
            prev_baseline = dependencies.summary_from_payload_fn(baseline_meta, prev_baseline_payload)
            prev_learned = dependencies.summary_from_payload_fn(learned_meta, prev_learned_payload)
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
        run_id=baseline_reference_run_id,
    )
    baseline_reference_learned_payload = (
        baseline_reference_payload.get("learned", {})
        if isinstance(baseline_reference_payload, dict)
        else {}
    )
    baseline_reference_summary = (
        dependencies.summary_from_payload_fn(learned_meta, baseline_reference_learned_payload)
        if isinstance(baseline_reference_learned_payload, dict) and baseline_reference_learned_payload
        else learned_meta
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
    if universe_tier_value == candidate_default_universe_tier and baseline_reference_run_id:
        switch_current_ok, switch_current_reasons, switch_deltas = dependencies.pass_default_switch_gate_fn(
            baseline_reference=baseline_reference_summary,
            candidate=learned_meta,
        )
        if previous_manifest:
            previous_switch_gate = previous_manifest.get("default_switch_gate", {})
            if isinstance(previous_switch_gate, dict):
                switch_previous_ok = bool(previous_switch_gate.get("current_passed", False))
                switch_previous_reason = "" if switch_previous_ok else str(previous_switch_gate.get("current_reasons", ""))
    default_switch_gate_passed = bool(release_gate_passed and switch_current_ok and switch_previous_ok)
    default_switch_gate = {
        "baseline_reference_run_id": baseline_reference_run_id,
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
    dataset_path: Path,
    calibration_path: Path,
    learning_path: Path,
    forecast_models_path: Path,
    frozen_forecast_bundle_path: Path,
    frozen_state_path: Path,
    backtest_path: Path,
    consistency_path: Path,
    rolling_oos_path: Path,
    info_manifest_path: Path,
    info_shadow_report_path: Path,
    external_signal_manifest_path: Path,
    manifest_path: Path,
    latest_policy_path: Path,
    latest_manifest_path: Path,
    tier_latest_policy_path: Path,
    tier_latest_manifest_path: Path,
    learning_manifest: dict[str, object],
    forecast_models_manifest: dict[str, object],
    info_artifacts: InfoPublishArtifacts,
    manifests: PublishManifestArtifacts,
    gate_artifacts: PublishGateArtifacts,
    publish_forecast_models: bool,
    update_latest: bool,
    universe_tier_value: str,
    active_default_universe_tier: str,
) -> None:
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)
    _write_publish_json(dataset_path, manifests.dataset_manifest)
    _write_publish_json(calibration_path, manifests.calibration_manifest)
    _write_publish_json(learning_path, learning_manifest)
    if publish_forecast_models:
        _write_publish_json(forecast_models_path, forecast_models_manifest)
        _write_publish_json(frozen_forecast_bundle_path, info_artifacts.frozen_forecast_bundle)
        _write_publish_json(frozen_state_path, info_artifacts.frozen_daily_state)
    _write_publish_json(backtest_path, manifests.backtest_manifest)
    _write_publish_json(consistency_path, manifests.consistency_manifest)
    _write_publish_json(rolling_oos_path, manifests.rolling_oos_manifest)
    _write_publish_json(info_manifest_path, info_artifacts.info_manifest)
    _write_publish_json(info_shadow_report_path, info_artifacts.info_shadow_report)
    _write_publish_json(external_signal_manifest_path, info_artifacts.external_signal_manifest)
    _write_publish_json(manifest_path, manifests.research_manifest)
    _write_publish_json(tier_latest_manifest_path, manifests.research_manifest)
    if gate_artifacts.gate_ok:
        _write_publish_json(tier_latest_policy_path, learning_manifest)
    allow_default_latest_update = bool(
        not universe_tier_value or universe_tier_value == active_default_universe_tier
    )
    if update_latest and gate_artifacts.release_gate_passed and allow_default_latest_update:
        _write_publish_json(latest_policy_path, learning_manifest)
        _write_publish_json(latest_manifest_path, manifests.research_manifest)
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
    run_id: str,
    created_at: str,
    universe_tier_value: str,
    universe_id: str,
    universe_size: int,
    universe_generation_rule: str,
    source_universe_manifest_path: str,
    active_default_universe_tier: str,
    candidate_default_universe_tier: str,
    baseline_reference_run_id: str,
    config_hash: str,
    policy_hash: str,
    universe_hash: str,
    model_hashes: dict[str, str],
    snapshot_hash: str,
    baseline_meta: V2BacktestSummary,
    calibrated_meta: V2BacktestSummary,
    learned_meta: V2BacktestSummary,
    symbols: list[str],
    train_window: dict[str, object],
    validation_window: dict[str, object],
    holdout_window: dict[str, object],
    regime_counts: dict[str, int],
    info_artifacts: InfoPublishArtifacts,
    split_mode: str,
    embargo_days: int,
    publish_forecast_models: bool,
    dataset_path: Path,
    calibration_path: Path,
    learning_path: Path,
    forecast_models_path: Path,
    frozen_forecast_bundle_path: Path,
    frozen_state_path: Path,
    backtest_path: Path,
    consistency_path: Path,
    rolling_oos_path: Path,
    info_manifest_path: Path,
    info_shadow_report_path: Path,
    external_signal_manifest_path: Path,
    latest_policy_path: Path,
    latest_manifest_path: Path,
    tier_latest_policy_path: Path,
    tier_latest_manifest_path: Path,
    release_gate: dict[str, object],
    default_switch_gate: dict[str, object],
) -> PublishManifestArtifacts:
    dataset_manifest = add_artifact_metadata(
        {
            "strategy_id": str(strategy_id),
            "config_path": str(settings.get("config_path", "")),
            "source": str(settings.get("source", "")),
            "watchlist": str(settings.get("watchlist", "")),
            "universe_tier": universe_tier_value,
            "universe_id": universe_id,
            "universe_size": int(universe_size),
            "universe_generation_rule": universe_generation_rule,
            "source_universe_manifest_path": source_universe_manifest_path,
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
            "symbols": symbols,
            "symbol_count": int(settings.get("symbol_count", len(symbols))),
            "universe_hash": universe_hash,
            "config_hash": config_hash,
            "info_file": str(info_artifacts.info_file_path),
            "event_file": str(settings.get("event_file", info_artifacts.info_file_path)),
            "info_hash": info_artifacts.info_hash,
            "info_shadow_enabled": bool(info_artifacts.info_shadow_enabled),
            "info_shadow_only": bool(settings.get("info_shadow_only", True)),
            "info_item_count": int(info_artifacts.info_manifest.get("info_item_count", 0)),
            "info_source_mode": str(settings.get("info_source_mode", "layered")),
            "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
            "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
            "capital_flow_file": str(settings.get("capital_flow_file", "")),
            "macro_file": str(settings.get("macro_file", "")),
            "external_signal_manifest": str(external_signal_manifest_path),
            "external_signal_version": str(settings.get("external_signal_version", "v1")),
            "external_signal_enabled": bool(settings.get("external_signals", True)),
            "event_lookback_days": int(settings.get("event_lookback_days", settings.get("info_lookback_days", 45))),
            "capital_flow_lookback_days": int(settings.get("capital_flow_lookback_days", 20)),
            "macro_lookback_days": int(settings.get("macro_lookback_days", 60)),
            "event_risk_cutoff": float(settings.get("event_risk_cutoff", 0.55)),
            "catalyst_boost_cap": float(settings.get("catalyst_boost_cap", 0.12)),
            "flow_exposure_cap": float(settings.get("flow_exposure_cap", 0.08)),
            "capital_flow_snapshot": dict(info_artifacts.external_signal_package.get("capital_flow_snapshot", {})),
            "macro_context_snapshot": dict(info_artifacts.external_signal_package.get("macro_context_snapshot", {})),
            "active_default_universe_tier": active_default_universe_tier,
            "candidate_default_universe_tier": candidate_default_universe_tier,
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
            "baseline": asdict(baseline_meta),
            "calibrated": asdict(calibrated_meta),
            "learned": asdict(learned_meta),
        },
        artifact_type="backtest_summary",
    )
    consistency_manifest = add_artifact_metadata(
        {
            "run_id": run_id,
            "universe_tier": universe_tier_value,
            "universe_id": universe_id,
            "universe_size": int(universe_size),
            "split_mode": str(split_mode),
            "embargo_days": int(embargo_days),
            "train_window": train_window,
            "validation_window": validation_window,
            "holdout_window": holdout_window,
            "snapshot_hash": snapshot_hash,
            "config_hash": config_hash,
            "policy_hash": policy_hash,
            "universe_hash": universe_hash,
            "model_hashes": model_hashes,
            "info_hash": info_artifacts.info_hash,
            "info_source_mode": str(settings.get("info_source_mode", "layered")),
            "external_signal_enabled": bool(settings.get("external_signals", True)),
            "external_signal_version": str(settings.get("external_signal_version", "v1")),
            "use_us_index_context": bool(settings.get("use_us_index_context", False)),
            "us_index_source": str(settings.get("us_index_source", "akshare")),
        },
        artifact_type="consistency_checklist",
    )
    rolling_oos_manifest = add_artifact_metadata(
        {
            "run_id": run_id,
            "universe_tier": universe_tier_value,
            "windows": [
                {
                    "name": "window_1",
                    "start": learned_meta.start_date,
                    "end": learned_meta.end_date,
                    "excess_annual_return": float(learned_meta.excess_annual_return),
                    "information_ratio": float(learned_meta.information_ratio),
                    "max_drawdown": float(learned_meta.max_drawdown),
                },
                {
                    "name": "window_2",
                    "start": calibrated_meta.start_date,
                    "end": calibrated_meta.end_date,
                    "excess_annual_return": float(calibrated_meta.excess_annual_return),
                    "information_ratio": float(calibrated_meta.information_ratio),
                    "max_drawdown": float(calibrated_meta.max_drawdown),
                },
            ],
            "regime_breakdown": regime_counts,
        },
        artifact_type="rolling_oos_report",
    )
    research_manifest = add_artifact_metadata(
        {
            "run_id": run_id,
            "strategy_id": str(strategy_id),
            "created_at": created_at,
            "baseline_reference_run_id": baseline_reference_run_id,
            "universe_tier": universe_tier_value,
            "universe_id": universe_id,
            "universe_size": int(universe_size),
            "universe_generation_rule": universe_generation_rule,
            "source_universe_manifest_path": source_universe_manifest_path,
            "info_hash": info_artifacts.info_hash,
            "info_shadow_enabled": bool(info_artifacts.info_shadow_enabled),
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
            "use_us_index_context": bool(settings.get("use_us_index_context", False)),
            "us_index_source": str(settings.get("us_index_source", "akshare")),
            "data_window": {
                "start": str(settings.get("start", "")),
                "end": str(settings.get("end", "")),
            },
            "config_hash": config_hash,
            "snapshot_hash": snapshot_hash,
            "policy_hash": policy_hash,
            "universe_hash": universe_hash,
            "model_hashes": model_hashes,
            "split_mode": str(split_mode),
            "embargo_days": int(embargo_days),
            "dataset_manifest": str(dataset_path),
            "policy_calibration": str(calibration_path),
            "learned_policy_model": str(learning_path),
            "forecast_models_manifest": str(forecast_models_path) if publish_forecast_models else "",
            "frozen_forecast_bundle": str(frozen_forecast_bundle_path) if publish_forecast_models else "",
            "frozen_daily_state": str(frozen_state_path) if publish_forecast_models else "",
            "backtest_summary": str(backtest_path),
            "consistency_checklist": str(consistency_path),
            "rolling_oos_report": str(rolling_oos_path),
            "info_manifest": str(info_manifest_path),
            "info_shadow_report": str(info_shadow_report_path),
            "external_signal_manifest": str(external_signal_manifest_path),
            "published_policy_model": str(latest_policy_path),
            "latest_research_manifest": str(latest_manifest_path),
            "tier_published_policy_model": str(tier_latest_policy_path),
            "tier_latest_research_manifest": str(tier_latest_manifest_path),
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
