from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from src.application.v2_contracts import V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult
from src.contracts.runtime import ResearchMatrixOptions, ResearchRunOptions

_LAST_RESEARCH_TRAJECTORY: object | None = None


@dataclass(frozen=True)
class ResearchWorkflowDependencies:
    emit_progress_fn: Callable[[str, str], None]
    load_or_build_v2_backtest_trajectory_fn: Callable[..., object]
    split_research_trajectory_fn: Callable[..., tuple[object, object, object]]
    trajectory_step_count_fn: Callable[[object], int]
    run_v2_backtest_live_fn: Callable[..., V2BacktestSummary]
    baseline_only_calibration_fn: Callable[[V2BacktestSummary], V2CalibrationResult]
    placeholder_learning_result_fn: Callable[[V2BacktestSummary], V2PolicyLearningResult]
    calibrate_v2_policy_fn: Callable[..., V2CalibrationResult]
    learn_v2_policy_model_fn: Callable[..., V2PolicyLearningResult]
    normalize_universe_tier_fn: Callable[[str], str]
    publish_v2_research_artifacts_fn: Callable[..., dict[str, str]]


def last_research_trajectory() -> object | None:
    return _LAST_RESEARCH_TRAJECTORY


def run_v2_research_workflow_impl(
    *,
    dependencies: ResearchWorkflowDependencies,
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
    skip_calibration: bool = False,
    skip_learning: bool = False,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
    training_window_days: int | None = 480,
    split_mode: str = "purged_wf",
    embargo_days: int = 20,
) -> tuple[V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult]:
    global _LAST_RESEARCH_TRAJECTORY
    _LAST_RESEARCH_TRAJECTORY = None
    dependencies.emit_progress_fn("research", f"载入研究轨迹: backend={forecast_backend}")
    trajectory = dependencies.load_or_build_v2_backtest_trajectory_fn(
        config_path=config_path,
        source=source,
        start_date=start_date,
        end_date=end_date,
        lookback_years=lookback_years,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
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
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        training_window_days=training_window_days,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    _LAST_RESEARCH_TRAJECTORY = trajectory
    if trajectory is None:
        empty_summary = dependencies.run_v2_backtest_live_fn(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
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
            dynamic_universe=dynamic_universe,
            generator_target_size=generator_target_size,
            generator_coarse_size=generator_coarse_size,
            generator_theme_aware=generator_theme_aware,
            generator_use_concepts=generator_use_concepts,
            retrain_days=retrain_days,
            trajectory=None,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        return (
            empty_summary,
            dependencies.baseline_only_calibration_fn(empty_summary),
            dependencies.placeholder_learning_result_fn(empty_summary),
        )
    _, validation_trajectory, holdout_trajectory = dependencies.split_research_trajectory_fn(
        trajectory,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )
    dependencies.emit_progress_fn(
        "research",
        f"样本切分完成(mode={split_mode}, embargo={embargo_days}d): validation={dependencies.trajectory_step_count_fn(validation_trajectory)}, holdout={dependencies.trajectory_step_count_fn(holdout_trajectory)}",
    )
    dependencies.emit_progress_fn("research", "开始回放 holdout 基线")
    baseline = dependencies.run_v2_backtest_live_fn(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        info_file=info_file,
        info_lookback_days=info_lookback_days,
        info_half_life_days=info_half_life_days,
        use_info_fusion=use_info_fusion,
        info_shadow_only=info_shadow_only,
        info_types=info_types,
        info_source_mode=info_source_mode,
        info_subsets=info_subsets,
        info_cutoff_time=info_cutoff_time,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        trajectory=holdout_trajectory,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    validation_baseline = None
    if not skip_calibration:
        dependencies.emit_progress_fn("research", "开始回放 validation 基线")
        validation_baseline = dependencies.run_v2_backtest_live_fn(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            info_file=info_file,
            info_lookback_days=info_lookback_days,
            info_half_life_days=info_half_life_days,
            use_info_fusion=use_info_fusion,
            info_shadow_only=info_shadow_only,
            info_types=info_types,
            info_source_mode=info_source_mode,
            info_subsets=info_subsets,
            info_cutoff_time=info_cutoff_time,
            external_signals=external_signals,
            event_file=event_file,
            capital_flow_file=capital_flow_file,
            macro_file=macro_file,
            dynamic_universe=dynamic_universe,
            generator_target_size=generator_target_size,
            generator_coarse_size=generator_coarse_size,
            generator_theme_aware=generator_theme_aware,
            generator_use_concepts=generator_use_concepts,
            trajectory=validation_trajectory,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
    calibration = (
        dependencies.baseline_only_calibration_fn(baseline)
        if skip_calibration
        else dependencies.calibrate_v2_policy_fn(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            info_file=info_file,
            info_lookback_days=info_lookback_days,
            info_half_life_days=info_half_life_days,
            use_info_fusion=use_info_fusion,
            info_shadow_only=info_shadow_only,
            info_types=info_types,
            info_source_mode=info_source_mode,
            info_subsets=info_subsets,
            info_cutoff_time=info_cutoff_time,
            external_signals=external_signals,
            event_file=event_file,
            capital_flow_file=capital_flow_file,
            macro_file=macro_file,
            dynamic_universe=dynamic_universe,
            generator_target_size=generator_target_size,
            generator_coarse_size=generator_coarse_size,
            generator_theme_aware=generator_theme_aware,
            generator_use_concepts=generator_use_concepts,
            baseline=validation_baseline if validation_baseline is not None else baseline,
            trajectory=validation_trajectory,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
    )
    if not skip_calibration:
        dependencies.emit_progress_fn("research", "参数搜索完成，开始 holdout 复核")
        holdout_calibrated = dependencies.run_v2_backtest_live_fn(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            info_file=info_file,
            info_lookback_days=info_lookback_days,
            info_half_life_days=info_half_life_days,
            use_info_fusion=use_info_fusion,
            info_shadow_only=info_shadow_only,
            info_types=info_types,
            info_source_mode=info_source_mode,
            info_subsets=info_subsets,
            info_cutoff_time=info_cutoff_time,
            external_signals=external_signals,
            event_file=event_file,
            capital_flow_file=capital_flow_file,
            macro_file=macro_file,
            policy_spec=calibration.best_policy,
            trajectory=holdout_trajectory,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        calibration = V2CalibrationResult(
            best_policy=calibration.best_policy,
            best_score=calibration.best_score,
            baseline=baseline,
            calibrated=holdout_calibrated,
            trials=calibration.trials,
        )
    else:
        dependencies.emit_progress_fn("research", "已跳过参数搜索")
    learning = (
        dependencies.placeholder_learning_result_fn(baseline)
        if skip_learning
        else dependencies.learn_v2_policy_model_fn(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            info_file=info_file,
            info_lookback_days=info_lookback_days,
            info_half_life_days=info_half_life_days,
            use_info_fusion=use_info_fusion,
            info_shadow_only=info_shadow_only,
            info_types=info_types,
            info_source_mode=info_source_mode,
            info_subsets=info_subsets,
            info_cutoff_time=info_cutoff_time,
            external_signals=external_signals,
            event_file=event_file,
            capital_flow_file=capital_flow_file,
            macro_file=macro_file,
            dynamic_universe=dynamic_universe,
            generator_target_size=generator_target_size,
            generator_coarse_size=generator_coarse_size,
            generator_theme_aware=generator_theme_aware,
            generator_use_concepts=generator_use_concepts,
            baseline=baseline,
            trajectory=holdout_trajectory,
            fit_trajectory=validation_trajectory,
            evaluation_trajectory=holdout_trajectory,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
    )
    if skip_learning:
        dependencies.emit_progress_fn("research", "已跳过学习型策略")
    else:
        dependencies.emit_progress_fn("research", "学习型策略评估完成")
    return baseline, calibration, learning


def run_v2_research_matrix_impl(
    *,
    dependencies: ResearchWorkflowDependencies,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    artifact_root: str = "artifacts/v2",
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
    training_window_days: int | None = 480,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    split_mode: str = "purged_wf",
    embargo_days: int = 20,
    universe_tiers: Iterable[str] = ("favorites_16", "generated_80", "generated_150", "generated_300"),
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    normalized_tiers = [dependencies.normalize_universe_tier_fn(item) for item in universe_tiers]
    for tier_id in normalized_tiers:
        dependencies.emit_progress_fn("matrix", f"开始研究矩阵档位: {tier_id}")
        baseline, calibration, learning = run_v2_research_workflow_impl(
            dependencies=dependencies,
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
            universe_tier=tier_id,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            training_window_days=training_window_days,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
            split_mode=split_mode,
            embargo_days=embargo_days,
        )
        artifacts = dependencies.publish_v2_research_artifacts_fn(
            strategy_id=strategy_id,
            artifact_root=artifact_root,
            config_path=config_path,
            source=source,
            start_date=start_date,
            end_date=end_date,
            lookback_years=lookback_years,
            universe_tier=tier_id,
            baseline=baseline,
            calibration=calibration,
            learning=learning,
            cache_root=cache_root,
            forecast_backend=forecast_backend,
            training_window_days=training_window_days,
            split_mode=split_mode,
            embargo_days=embargo_days,
            publish_forecast_models=True,
            trajectory=last_research_trajectory(),
        )
        rows.append(
            {
                "universe_tier": tier_id,
                "run_id": artifacts.get("run_id", ""),
                "release_gate_passed": artifacts.get("release_gate_passed", "false"),
                "default_switch_gate_passed": artifacts.get("default_switch_gate_passed", "false"),
                "annual_return": float(learning.learned.annual_return),
                "excess_annual_return": float(learning.learned.excess_annual_return),
                "information_ratio": float(learning.learned.information_ratio),
                "max_drawdown": float(learning.learned.max_drawdown),
                "avg_turnover": float(learning.learned.avg_turnover),
                "total_cost": float(learning.learned.total_cost),
                "baseline_annual_return": float(baseline.annual_return),
                "baseline_excess_annual_return": float(baseline.excess_annual_return),
                "baseline_information_ratio": float(baseline.information_ratio),
                "baseline_max_drawdown": float(baseline.max_drawdown),
                "research_manifest": artifacts.get("research_manifest", ""),
            }
        )
    return {
        "strategy_id": str(strategy_id),
        "split_mode": str(split_mode),
        "embargo_days": int(embargo_days),
        "forecast_backend": str(forecast_backend),
        "training_window_days": None if training_window_days is None else int(training_window_days),
        "rows": rows,
    }


def run_v2_research_workflow(
    *,
    options: ResearchRunOptions | None = None,
    **kwargs: object,
):
    from src.application import v2_services as legacy

    resolved = options or ResearchRunOptions.from_kwargs(**kwargs)
    return run_v2_research_workflow_impl(
        dependencies=legacy._research_workflow_dependencies(),
        **resolved.workflow_kwargs(),
    )


def run_v2_research_matrix(
    *,
    options: ResearchMatrixOptions | None = None,
    **kwargs: object,
):
    from src.application import v2_services as legacy

    resolved = options or ResearchMatrixOptions.from_kwargs(**kwargs)
    return run_v2_research_matrix_impl(
        dependencies=legacy._research_workflow_dependencies(),
        **resolved.workflow_kwargs(),
    )
