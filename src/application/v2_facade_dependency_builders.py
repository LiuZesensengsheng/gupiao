from __future__ import annotations

from types import ModuleType

from src.application import (
    v2_backtest_prepare_runtime as _v2_backtest_prepare_runtime,
    v2_backtest_runtime as _v2_backtest_runtime,
    v2_daily_state_runtime as _v2_daily_state_runtime,
    v2_feature_runtime as _v2_feature_runtime,
    v2_frozen_forecast_runtime as _v2_frozen_forecast_runtime,
    v2_info_shadow_runtime as _v2_info_shadow_runtime,
    v2_learning_target_runtime as _v2_learning_target_runtime,
    v2_policy_feature_runtime as _v2_policy_feature_runtime,
    v2_policy_learning_runtime as _v2_policy_learning_runtime,
    v2_policy_runtime as _v2_policy_runtime,
    v2_state_build_runtime as _v2_state_build_runtime,
)
from src.reporting import forecast_support as _reporting_forecast_support
from src.reporting import report_state_runtime as _report_state_runtime
from src.review_analytics import info_manifest as _info_manifest_runtime
from src.review_analytics import info_shadow_report as _info_shadow_report
from src.review_analytics import prediction_review as _prediction_review
from src.workflows.daily_workflow import DailyWorkflowDependencies
from src.workflows.research_workflow import ResearchWorkflowDependencies


def build_policy_feature_runtime_dependencies(
    module: ModuleType,
) -> _v2_policy_feature_runtime.PolicyFeatureRuntimeDependencies:
    return _v2_policy_feature_runtime.PolicyFeatureRuntimeDependencies(
        clip=module._clip,
        alpha_opportunity_metrics=module._alpha_opportunity_metrics,
        candidate_stocks_from_state=module._candidate_stocks_from_state_external,
        candidate_risk_snapshot=module._candidate_risk_snapshot_external,
    )


def build_policy_runtime_dependencies(module: ModuleType) -> _v2_policy_runtime.PolicyRuntimeDependencies:
    feature_deps = build_policy_feature_runtime_dependencies(module)
    return _v2_policy_runtime.PolicyRuntimeDependencies(
        clip=module._clip,
        alpha_score_components=module._alpha_score_components,
        candidate_stocks_from_state=module._candidate_stocks_from_state_external,
        candidate_risk_snapshot=module._candidate_risk_snapshot_external,
        dominant_mainline_sectors=module._dominant_mainline_sectors_external,
        ranked_sector_budgets_with_alpha_external=module._ranked_sector_budgets_with_alpha_external,
        cap_sector_budgets_external=module._cap_sector_budgets_external,
        allocate_sector_slots_external=module._allocate_sector_slots_external,
        allocate_with_sector_budgets_external=module._allocate_with_sector_budgets_external,
        is_actionable_status=module._is_actionable_status,
        policy_feature_vector=lambda state: _v2_policy_feature_runtime.policy_feature_vector(state, deps=feature_deps),
        normalize_coef_vector=_v2_policy_feature_runtime.normalize_coef_vector,
        predict_ridge=_v2_policy_feature_runtime.predict_ridge,
    )


def build_backtest_execution_dependencies(module: ModuleType) -> _v2_backtest_runtime.BacktestExecutionDependencies:
    feature_deps = build_policy_feature_runtime_dependencies(module)
    return _v2_backtest_runtime.BacktestExecutionDependencies(
        safe_float=module._safe_float,
        clip=module._clip,
        status_tradeability_limit=module._status_tradeability_limit,
        is_actionable_status=module._is_actionable_status,
        policy_spec_from_model=module._policy_spec_from_model,
        apply_policy=module.apply_policy,
        advance_holding_days=module._advance_holding_days,
        derive_learning_targets=module._derive_learning_targets,
        policy_feature_names=_v2_policy_feature_runtime.policy_feature_names,
        policy_feature_vector=lambda state: _v2_policy_feature_runtime.policy_feature_vector(state, deps=feature_deps),
        to_v2_backtest_summary=module._to_v2_backtest_summary,
    )


def build_backtest_core_dependencies(module: ModuleType) -> _v2_backtest_runtime.BacktestCoreDependencies:
    return _v2_backtest_runtime.BacktestCoreDependencies(
        load_or_build_v2_backtest_trajectory=module._load_or_build_v2_backtest_trajectory,
        empty_v2_backtest_result=module._empty_v2_backtest_result,
        execute_v2_backtest_trajectory=module._execute_v2_backtest_trajectory,
    )


def build_backtest_prepare_dependencies(module: ModuleType) -> _v2_backtest_prepare_runtime.BacktestPrepareDependencies:
    return _v2_backtest_prepare_runtime.BacktestPrepareDependencies(
        load_v2_runtime_settings=module._load_v2_runtime_settings,
        resolve_v2_universe_settings=module._resolve_v2_universe_settings,
        build_prepared_backtest_cache_key=module._build_prepared_backtest_cache_key_external,
        prepared_backtest_cache_path=module._prepared_backtest_cache_path_external,
        load_pickle_cache=module._load_pickle_cache_external,
        store_pickle_cache=module._store_pickle_cache_external,
        emit_progress=module._emit_progress,
        load_watchlist=module.load_watchlist,
        build_candidate_universe=module.build_candidate_universe,
        load_symbol_daily=module.load_symbol_daily,
        make_market_feature_frame=module.make_market_feature_frame,
        build_market_context_features=module.build_market_context_features,
        build_stock_panel_dataset=module.build_stock_panel_dataset,
        market_feature_columns=list(module.MARKET_FEATURE_COLUMNS),
        make_forecast_backend=module._make_forecast_backend,
        prepare_v2_backtest_data=module._prepare_v2_backtest_data,
        build_v2_backtest_trajectory_from_prepared=module._build_v2_backtest_trajectory_from_prepared,
        parse_boolish=module._parse_boolish,
    )


def build_state_build_runtime_dependencies(module: ModuleType) -> _v2_state_build_runtime.StateBuildRuntimeDependencies:
    return _v2_state_build_runtime.StateBuildRuntimeDependencies(
        predict_quantile_profiles=module._predict_quantile_profiles,
        distributional_score=module._distributional_score,
        status_tradeability_limit=module._status_tradeability_limit,
        build_horizon_forecasts=module._build_horizon_forecasts,
        return_quantile_profile_cls=module._ReturnQuantileProfile,
        safe_float=module._safe_float,
        clip=module._clip,
        stock_policy_score=module._stock_policy_score,
        market_facts_from_row=module._market_facts_from_row,
        load_symbol_daily=module.load_symbol_daily,
        make_market_feature_frame=module.make_market_feature_frame,
        build_market_context_features=module.build_market_context_features,
        decide_market_state=module.decide_market_state,
        forecast_cross_section_state=module.forecast_cross_section_state,
        build_mainline_states=module._build_mainline_states_external,
        build_candidate_selection_state=module._build_candidate_selection_state_external,
        apply_leader_candidate_overlay=module._apply_leader_candidate_overlay,
    )


def build_learning_target_dependencies(module: ModuleType) -> _v2_learning_target_runtime.LearningTargetDependencies:
    return _v2_learning_target_runtime.LearningTargetDependencies(
        stock_policy_score=module._stock_policy_score,
        safe_float=module._safe_float,
        alpha_opportunity_metrics=module._alpha_opportunity_metrics,
        signal_unit=module._signal_unit,
        normalize_universe_tier=module.normalize_universe_tier,
        clip=module._clip,
    )


def build_info_shadow_runtime_dependencies(module: ModuleType) -> _v2_info_shadow_runtime.InfoShadowRuntimeDependencies:
    return _v2_info_shadow_runtime.InfoShadowRuntimeDependencies(
        build_sector_map_from_state=module._build_sector_map_from_state,
        build_info_state_maps=module.build_info_state_maps,
        predict_info_shadow_prob=module._predict_info_shadow_prob,
        blend_probability=module.blend_probability,
        build_mainline_states=module._build_mainline_states_external,
        stock_policy_score=module._stock_policy_score,
        compose_shadow_stock_score=module._compose_shadow_stock_score,
        safe_float=module._safe_float,
        info_aggregate_state_cls=module.InfoAggregateState,
        info_feature_frame=module._info_feature_frame,
        fit_info_shadow_model=module._fit_info_shadow_model,
        info_shadow_feature_columns=list(module._INFO_SHADOW_FEATURE_COLUMNS),
        panel_slice_metrics=module._panel_slice_metrics,
    )


def build_frozen_forecast_runtime_dependencies(
    module: ModuleType,
) -> _v2_frozen_forecast_runtime.FrozenForecastRuntimeDependencies:
    return _v2_frozen_forecast_runtime.FrozenForecastRuntimeDependencies(
        load_symbol_daily=module.load_symbol_daily,
        make_market_feature_frame=module.make_market_feature_frame,
        build_market_context_features=module.build_market_context_features,
        deserialize_binary_model=module._deserialize_binary_model,
        deserialize_quantile_bundle=module._deserialize_quantile_bundle,
        predict_quantile_profile=module._predict_quantile_profile,
        build_market_and_cross_section_from_prebuilt_frame=module._build_market_and_cross_section_from_prebuilt_frame,
        build_stock_live_panel_dataset=module.build_stock_live_panel_dataset,
        build_stock_states_from_panel_slice=module._build_stock_states_from_panel_slice,
        build_sector_states=module._build_sector_states_external,
        stock_policy_score=module._stock_policy_score,
        compose_state=module.compose_state,
    )


def build_frozen_forecast_bundle_dependencies(
    module: ModuleType,
) -> _v2_frozen_forecast_runtime.FrozenForecastBundleDependencies:
    return _v2_frozen_forecast_runtime.FrozenForecastBundleDependencies(
        binary_model_cls=module.LogisticBinaryModel,
        serialize_binary_model=module._serialize_binary_model,
        fit_quantile_quintet=module._fit_quantile_quintet,
        serialize_quantile_bundle=module._serialize_quantile_bundle,
    )


def build_forecast_runtime_dependencies(module: ModuleType) -> _v2_feature_runtime.ForecastRuntimeDependencies:
    return _v2_feature_runtime.ForecastRuntimeDependencies(
        emit_progress=module._emit_progress,
        format_elapsed=module._format_elapsed,
        build_date_slice_index=module._build_date_slice_index,
        build_market_and_cross_section_from_prebuilt_frame=module._build_market_and_cross_section_from_prebuilt_frame,
        build_stock_states_from_panel_slice=module._build_stock_states_from_panel_slice,
        build_sector_states=module._build_sector_states_external,
        stock_policy_score=module._stock_policy_score,
        compose_state=module.compose_state,
        panel_horizon_metrics=module._panel_horizon_metrics,
        fit_quantile_quintet=module._fit_quantile_quintet,
        fit_mlp_quantile_quintet=module._fit_mlp_quantile_quintet,
        logistic_model_cls=module.LogisticBinaryModel,
        mlp_model_cls=module.MLPBinaryModel,
        quantile_model_cls=module.QuantileLinearModel,
        mlp_quantile_model_cls=module.MLPQuantileModel,
        trajectory_step_cls=module._TrajectoryStep,
        backtest_trajectory_cls=module._BacktestTrajectory,
    )


def build_prediction_review_dependencies(module: ModuleType) -> _prediction_review.PredictionReviewDependencies:
    return _prediction_review.PredictionReviewDependencies(
        path_from_manifest_entry=module._path_from_manifest_entry,
        load_json_dict=module._load_json_dict,
    )


def build_report_state_runtime_dependencies(
    module: ModuleType,
) -> _report_state_runtime.ReportStateRuntimeDependencies:
    return _report_state_runtime.ReportStateRuntimeDependencies(
        build_horizon_forecasts=module._build_horizon_forecasts,
        profile_from_horizon_map=module._profile_from_horizon_map,
        build_market_sentiment_state=module._build_market_sentiment_state,
        candidate_stocks_from_state=module._candidate_stocks_from_state_external,
        stock_reason_bundle=module._stock_reason_bundle,
        is_main_board_symbol=module._is_main_board_symbol,
        build_candidate_selection_state=module._build_candidate_selection_state_external,
        stock_policy_score=module._stock_policy_score,
        build_market_and_cross_section_states=module._build_market_and_cross_section_states,
    )


def build_forecast_support_dependencies(
    module: ModuleType,
) -> _reporting_forecast_support.ForecastSupportDependencies:
    return _reporting_forecast_support.ForecastSupportDependencies(
        clip=module._clip,
        return_quantile_profile_cls=module._ReturnQuantileProfile,
    )


def build_info_shadow_report_dependencies(
    module: ModuleType,
) -> _info_shadow_report.InfoShadowReportDependencies:
    return _info_shadow_report.InfoShadowReportDependencies(
        build_info_shadow_variant=module._build_info_shadow_variant,
        filter_info_items_by_source_subset=module._filter_info_items_by_source_subset,
        event_tag_counts=module.event_tag_counts,
        info_source_breakdown=module._info_source_breakdown,
    )


def build_info_manifest_dependencies(module: ModuleType) -> _info_manifest_runtime.InfoManifestDependencies:
    return _info_manifest_runtime.InfoManifestDependencies(
        sha256_file=module._sha256_file,
        stable_json_hash=module._stable_json_hash,
    )


def build_policy_learning_dependencies(
    module: ModuleType,
) -> _v2_policy_learning_runtime.PolicyLearningDependencies:
    return _v2_policy_learning_runtime.PolicyLearningDependencies(
        emit_progress=module._emit_progress,
        policy_objective_score=module._policy_objective_score,
        run_v2_backtest_live=module.run_v2_backtest_live,
        run_v2_backtest_core=module._run_v2_backtest_core,
        policy_feature_names=_v2_policy_feature_runtime.policy_feature_names,
        fit_ridge_regression=_v2_policy_feature_runtime.fit_ridge_regression,
        predict_ridge=_v2_policy_feature_runtime.predict_ridge,
        r2_score=_v2_policy_feature_runtime.r2_score,
    )


def build_daily_state_runtime_dependencies(
    module: ModuleType,
) -> _v2_daily_state_runtime.DailyStateRuntimeDependencies:
    return _v2_daily_state_runtime.DailyStateRuntimeDependencies(
        emit_progress=module._emit_progress,
        load_watchlist=module.load_watchlist,
        build_candidate_universe=module.build_candidate_universe,
        daily_universe_context_cls=module._DailyUniverseContext,
        run_quant_pipeline=module.run_quant_pipeline,
        load_symbol_daily=module.load_symbol_daily,
        build_market_and_cross_section_states=module._build_market_and_cross_section_states,
        safe_float=module._safe_float,
        return_quantile_profile_cls=module._ReturnQuantileProfile,
        build_sector_daily_frames=module.build_sector_daily_frames,
        run_sector_forecast=module.run_sector_forecast,
        build_stock_states_from_rows=module._build_stock_states_from_rows,
        compose_state=module.compose_state,
        path_from_manifest_entry=module._path_from_manifest_entry,
        load_json_dict=module._load_json_dict,
        decode_composite_state=module._decode_composite_state,
        load_frozen_forecast_bundle=module._load_frozen_forecast_bundle,
        score_live_composite_state_from_frozen_bundle=module._score_live_composite_state_from_frozen_bundle,
        load_v2_info_items_for_date=module._load_v2_info_items_for_date,
        enrich_state_with_info=module._enrich_state_with_info,
        sha256_file=module._sha256_file,
        stable_json_hash=module._stable_json_hash,
        top_negative_events=module.top_negative_events,
        top_positive_stock_signals=module.top_positive_stock_signals,
        quant_info_divergence_rows=module.quant_info_divergence_rows,
        attach_external_signals_to_composite_state=module._attach_external_signals_to_composite_state,
        attach_insight_memory_to_state=module._attach_insight_memory_to_state,
    )


def build_research_workflow_dependencies(module: ModuleType) -> ResearchWorkflowDependencies:
    return ResearchWorkflowDependencies(
        emit_progress_fn=module._emit_progress,
        load_or_build_v2_backtest_trajectory_fn=module._load_or_build_v2_backtest_trajectory,
        split_research_trajectory_fn=module._split_research_trajectory,
        trajectory_step_count_fn=module._trajectory_step_count,
        run_v2_backtest_live_fn=module.run_v2_backtest_live,
        baseline_only_calibration_fn=module._baseline_only_calibration,
        placeholder_learning_result_fn=module._placeholder_learning_result,
        calibrate_v2_policy_fn=module.calibrate_v2_policy,
        learn_v2_policy_model_fn=module.learn_v2_policy_model,
        normalize_universe_tier_fn=module.normalize_universe_tier,
        publish_v2_research_artifacts_fn=module._publish_v2_research_artifacts_impl,
    )


def build_daily_workflow_dependencies(module: ModuleType) -> DailyWorkflowDependencies:
    return DailyWorkflowDependencies(
        build_daily_snapshot_context_fn=module._build_daily_snapshot_context,
        daily_result_cache_key_fn=module._daily_result_cache_key,
        daily_result_cache_path_fn=module._daily_result_cache_path,
        load_daily_cached_result_fn=module._load_daily_cached_result,
        build_daily_universe_context_fn=module._build_daily_universe_context,
        build_daily_composite_state_fn=module._build_daily_composite_state,
        build_daily_symbol_names_fn=module._build_daily_symbol_names,
        attach_daily_info_overlay_fn=module._attach_daily_info_overlay,
        attach_daily_external_signal_overlay_fn=module._attach_daily_external_signal_overlay,
        attach_daily_insight_overlay_fn=module._attach_daily_insight_overlay,
        filter_state_for_recommendation_scope_fn=module._filter_state_for_recommendation_scope,
        apply_leader_candidate_overlay_fn=module._apply_leader_candidate_overlay,
        parse_boolish_fn=module._parse_boolish,
        resolve_daily_policy_model_fn=module._resolve_daily_policy_model,
        policy_spec_from_model_fn=module._policy_spec_from_model,
        apply_policy_fn=module.apply_policy,
        build_execution_plans_fn=module._build_execution_plans,
        build_trade_actions_fn=module.build_trade_actions,
        load_prediction_review_context_fn=module._load_prediction_review_context,
        build_live_market_reporting_overlay_fn=module._build_live_market_reporting_overlay,
        decorate_composite_state_for_reporting_fn=module._decorate_composite_state_for_reporting,
        remember_daily_run_fn=module.remember_daily_run,
        persist_daily_insight_artifacts_fn=module._persist_daily_insight_artifacts,
        emit_progress_fn=module._emit_progress,
    )
