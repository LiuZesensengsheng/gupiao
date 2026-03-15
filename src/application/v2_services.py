from __future__ import annotations

import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CapitalFlowState,
    CompositeState,
    CrossSectionForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
    DailyRunResult,
    InfoAggregateState,
    InfoDivergenceRecord,
    InfoItem,
    InfoSignalRecord,
    HorizonForecast,
    LearnedPolicyModel,
    MarketFactsState,
    MarketForecastState,
    MarketSentimentState,
    MacroContextState,
    PolicyDecision,
    PolicyInput,
    PolicySpec,
    PredictionReviewState,
    PredictionReviewWindow,
    SectorForecastState,
    StockForecastState,
    StrategySnapshot,
)
from src.contracts.artifacts import (
    ForecastBundle,
    add_artifact_metadata,
)
from src.application.v2_daily_runtime import (
    build_daily_symbol_names as _build_daily_symbol_names_external,
    load_daily_cached_result as _load_daily_cached_result_external,
)
from src.application.v2_backtest_cache_support import (
    build_prepared_backtest_cache_key as _build_prepared_backtest_cache_key_external,
    load_pickle_cache as _load_pickle_cache_external,
    prepared_backtest_cache_path as _prepared_backtest_cache_path_external,
    store_pickle_cache as _store_pickle_cache_external,
)
from src.application.v2_candidate_selection import (
    build_candidate_selection_state as _build_candidate_selection_state_external,
    candidate_risk_snapshot as _candidate_risk_snapshot_external,
    candidate_stocks_from_state as _candidate_stocks_from_state_external,
)
from src.application.v2_mainline_support import (
    build_mainline_states as _build_mainline_states_external,
    dominant_mainline_sectors as _dominant_mainline_sectors_external,
)
from src.application.v2_publish_support import (
    load_backtest_payload_for_run as _load_backtest_payload_for_run_external,
    load_backtest_payload_from_manifest as _load_backtest_payload_from_manifest_external,
    pass_default_switch_gate as _pass_default_switch_gate_external,
    pass_release_gate as _pass_release_gate_external,
    summary_from_payload as _summary_from_payload_external,
    tier_latest_manifest_path as _tier_latest_manifest_path_external,
    tier_latest_policy_path as _tier_latest_policy_path_external,
)
from src.application.v2_snapshot_support import (
    build_frozen_daily_state_payload as _build_frozen_daily_state_payload_external,
    decode_composite_state as _decode_composite_state_external,
    resolve_manifest_entry_path as _path_from_manifest_entry_external,
    serialize_composite_state as _serialize_composite_state_external,
)
from src.application.v2_daily_snapshot_runtime import (
    DailySnapshotContext as _DailySnapshotContextExternal,
    build_daily_snapshot_context as _build_daily_snapshot_context_external,
    build_snapshot_from_manifest as _build_snapshot_from_manifest_external,
    build_strategy_snapshot as _build_strategy_snapshot_external,
    hydrate_daily_settings_from_dataset_manifest as _hydrate_daily_settings_from_dataset_manifest_external,
    is_daily_universe_override_mismatch as _is_daily_universe_override_mismatch_external,
    load_research_manifest_for_daily as _load_research_manifest_for_daily_runtime_external,
    resolve_manifest_path as _resolve_manifest_path_external,
)
from src.application.v2_artifact_runtime import (
    load_published_v2_policy_model as _load_published_v2_policy_model_external,
    resolve_daily_policy_model as _resolve_daily_policy_model_external,
)
from src.application.v2_research_publish_runtime import (
    ResearchPublishDependencies as _ResearchPublishDependenciesExternal,
    publish_research_artifacts as _publish_research_artifacts_runtime_external,
)
from src.application import v2_backtest_runtime as _v2_backtest_runtime
from src.application import v2_backtest_metrics_runtime as _v2_backtest_metrics_runtime
from src.application import v2_external_signal_runtime as _v2_external_signal_runtime
from src.application import v2_facade_dependency_builders as _v2_facade_dependency_builders
from src.application import v2_facade_support_runtime as _v2_facade_support_runtime
from src.application import v2_feature_runtime as _v2_feature_runtime
from src.application import v2_backtest_prepare_runtime as _v2_backtest_prepare_runtime
from src.application import v2_daily_runtime as _v2_daily_runtime
from src.application import v2_daily_state_runtime as _v2_daily_state_runtime
from src.application import v2_forecast_model_runtime as _v2_forecast_model_runtime
from src.application import v2_frozen_forecast_runtime as _v2_frozen_forecast_runtime
from src.application import v2_info_shadow_runtime as _v2_info_shadow_runtime
from src.application import v2_learning_target_runtime as _v2_learning_target_runtime
from src.application import v2_policy_feature_runtime as _v2_policy_feature_runtime
from src.application import v2_policy_runtime as _v2_policy_runtime
from src.application import v2_policy_learning_runtime as _v2_policy_learning_runtime
from src.application import v2_runtime_primitives as _v2_runtime_primitives
from src.application import v2_state_build_runtime as _v2_state_build_runtime
from src.reporting import report_state_runtime as _report_state_runtime
from src.reporting import forecast_support as _reporting_forecast_support
from src.reporting import reason_bundles as _reporting_reason_bundles
from src.review_analytics import info_shadow_report as _info_shadow_report
from src.review_analytics import info_manifest as _info_manifest_runtime
from src.review_analytics import prediction_review as _prediction_review
from src.application import v2_registry_runtime as _v2_registry_runtime
from src.application.v2_registry_runtime import (
    build_research_publish_dependencies as _build_research_publish_dependencies_external,
    load_policy_model_from_path as _load_policy_model_from_path_runtime_external,
)
from src.application.v2_runtime_settings import (
    coalesce as _coalesce_external,
    configure_v2_tushare_token as _configure_v2_tushare_token_external,
    extract_universe_rows as _extract_universe_rows_external,
    hydrate_universe_metadata as _hydrate_universe_metadata_external,
    load_v2_runtime_settings as _load_v2_runtime_settings_external,
    parse_boolish as _parse_boolish_external,
    parse_csv_tokens as _parse_csv_tokens_external,
    resolve_v2_universe_settings as _resolve_v2_universe_settings_external,
)
from src.application.v2_universe_generator import generate_dynamic_universe
from src.application.v2_sector_support import (
    allocate_sector_slots as _allocate_sector_slots_external,
    allocate_with_sector_budgets as _allocate_with_sector_budgets_external,
    build_sector_states as _build_sector_states_external,
    cap_sector_budgets as _cap_sector_budgets_external,
    ranked_sector_budgets_with_alpha as _ranked_sector_budgets_with_alpha_external,
)
from src.application.watchlist import load_watchlist
from src.domain.entities import TradeAction
from src.domain.news import blend_probability
from src.domain.policies import blend_horizon_score, decide_market_state
from src.infrastructure.discovery import (
    build_candidate_universe,
    build_predefined_universe,
    normalize_universe_tier,
)
from src.infrastructure.cross_section_forecast import forecast_cross_section_state
from src.infrastructure.features import (
    MARKET_FEATURE_COLUMNS,
    make_market_feature_frame,
)
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.v2_info_fusion import (
    build_info_state_maps,
    event_tag_counts,
    quant_info_divergence_rows,
    top_negative_events,
    top_positive_stock_signals,
)
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import load_symbol_daily
from src.infrastructure.modeling import (
    LogisticBinaryModel,
    MLPBinaryModel,
    MLPQuantileModel,
    QuantileLinearModel,
)
from src.infrastructure.panel_dataset import build_stock_live_panel_dataset, build_stock_panel_dataset
from src.infrastructure.sector_data import build_sector_daily_frames
from src.infrastructure.sector_forecast import run_sector_forecast
from src.infrastructure.strategy_memory import remember_daily_run, remember_research_run


_clip = _v2_runtime_primitives.clip


_coalesce = _coalesce_external
_configure_v2_tushare_token = _configure_v2_tushare_token_external
_parse_boolish = _parse_boolish_external
_parse_csv_tokens = _parse_csv_tokens_external


_safe_float = _v2_runtime_primitives.safe_float
_signal_unit = _v2_runtime_primitives.signal_unit
_is_main_board_symbol = _v2_runtime_primitives.is_main_board_symbol


_DEFAULT_SPLIT_MODE = "purged_wf"
_DEFAULT_EMBARGO_DAYS = 20
_RELEASE_GATE_THRESHOLD = {
    "excess_annual_return_min": 0.0,
    "information_ratio_min": 0.30,
    "max_drawdown_worse_limit": 0.05,
}
_DEFAULT_SWITCH_GATE_THRESHOLD = {
    "excess_annual_return_delta_min": 0.02,
    "information_ratio_delta_min": 0.10,
    "max_drawdown_worse_limit": 0.02,
}
_INFO_SHADOW_FEATURE_COLUMNS = _v2_info_shadow_runtime.INFO_SHADOW_FEATURE_COLUMNS


_stable_json_hash = _v2_facade_support_runtime.stable_json_hash
_sha256_text = _v2_facade_support_runtime.sha256_text
_sha256_file = _v2_facade_support_runtime.sha256_file


_InfoShadowModel = _v2_info_shadow_runtime.InfoShadowModel
_clip_prob = _v2_info_shadow_runtime.clip_prob
_logit_prob = _v2_info_shadow_runtime.logit_prob
_resolve_info_file_from_settings = _v2_info_shadow_runtime.resolve_info_file_from_settings
_load_v2_info_items_for_date = _v2_info_shadow_runtime.load_v2_info_items_for_date
_info_feature_frame = _v2_info_shadow_runtime.info_feature_frame
_fit_info_shadow_model = _v2_info_shadow_runtime.fit_info_shadow_model
_predict_info_shadow_prob = _v2_info_shadow_runtime.predict_info_shadow_prob


_compose_shadow_stock_score = _v2_facade_support_runtime.compose_shadow_stock_score
_build_sector_map_from_state = _v2_facade_support_runtime.build_sector_map_from_state


def _enrich_state_with_info(
    *,
    state: CompositeState,
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
    settings: dict[str, object],
    stock_models: dict[str, _InfoShadowModel] | None = None,
    market_models: dict[str, _InfoShadowModel] | None = None,
) -> CompositeState:
    return _v2_info_shadow_runtime.enrich_state_with_info(
        state=state,
        as_of_date=as_of_date,
        info_items=info_items,
        settings=settings,
        stock_models=stock_models,
        market_models=market_models,
        deps=_info_shadow_runtime_dependencies(),
    )


def _external_signal_runtime_dependencies() -> _v2_external_signal_runtime.ExternalSignalRuntimeDependencies:
    return _v2_external_signal_runtime.ExternalSignalRuntimeDependencies(
        build_mainline_states=_build_mainline_states_external,
        stock_policy_score=_stock_policy_score,
    )


_build_external_signal_package_for_date = _v2_external_signal_runtime.build_external_signal_package_for_date


def _attach_external_signals_to_composite_state(
    *,
    state: CompositeState,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
) -> tuple[CompositeState, dict[str, object]]:
    return _v2_external_signal_runtime.attach_external_signals_to_composite_state(
        state=state,
        settings=settings,
        as_of_date=as_of_date,
        info_items=info_items,
        deps=_external_signal_runtime_dependencies(),
    )


_load_json_dict = _v2_registry_runtime.load_json_dict
_resolve_manifest_path = _resolve_manifest_path_external
_path_from_manifest_entry = _path_from_manifest_entry_external
_compose_run_snapshot_hash = _v2_registry_runtime.compose_run_snapshot_hash


_emit_progress = _v2_facade_support_runtime.emit_progress
_trajectory_step_count = _v2_facade_support_runtime.trajectory_step_count
_format_elapsed = _v2_facade_support_runtime.format_elapsed


_policy_feature_names = _v2_policy_feature_runtime.policy_feature_names


def _policy_feature_vector(state: CompositeState) -> np.ndarray:
    return _v2_policy_feature_runtime.policy_feature_vector(
        state,
        deps=_policy_feature_runtime_dependencies(),
    )


_fit_ridge_regression = _v2_policy_feature_runtime.fit_ridge_regression
_predict_ridge = _v2_policy_feature_runtime.predict_ridge
_normalize_coef_vector = _v2_policy_feature_runtime.normalize_coef_vector
_r2_score = _v2_policy_feature_runtime.r2_score


_ReturnQuantileProfile = _v2_forecast_model_runtime.ReturnQuantileProfile
_fit_quantile_quintet = _v2_forecast_model_runtime.fit_quantile_quintet
_fit_mlp_quantile_quintet = _v2_forecast_model_runtime.fit_mlp_quantile_quintet
_predict_quantile_profile = _v2_forecast_model_runtime.predict_quantile_profile
_predict_quantile_profiles = _v2_forecast_model_runtime.predict_quantile_profiles
_serialize_binary_model = _v2_forecast_model_runtime.serialize_binary_model
_deserialize_binary_model = _v2_forecast_model_runtime.deserialize_binary_model
_serialize_quantile_model = _v2_forecast_model_runtime.serialize_quantile_model
_deserialize_quantile_model = _v2_forecast_model_runtime.deserialize_quantile_model
_serialize_quantile_bundle = _v2_forecast_model_runtime.serialize_quantile_bundle
_deserialize_quantile_bundle = _v2_forecast_model_runtime.deserialize_quantile_bundle


_HORIZON_SCALE = {
    "1d": 0.035,
    "2d": 0.050,
    "3d": 0.065,
    "5d": 0.095,
    "10d": 0.145,
    "20d": 0.220,
}


_next_session_label = _v2_runtime_primitives.next_session_label


def _blend_quantile_profiles(
    left: _ReturnQuantileProfile,
    right: _ReturnQuantileProfile,
    *,
    left_weight: float,
) -> _ReturnQuantileProfile:
    return _reporting_forecast_support.blend_quantile_profiles(
        left,
        right,
        left_weight=left_weight,
        deps=_forecast_support_dependencies(),
    )


def _synthetic_quantile_profile(*, prob: float, horizon_key: str) -> _ReturnQuantileProfile:
    return _reporting_forecast_support.synthetic_quantile_profile(
        prob=prob,
        horizon_key=horizon_key,
        deps=_forecast_support_dependencies(),
    )


def _intrinsic_confidence(
    *,
    up_prob: float,
    horizon_probs: dict[str, float],
    info_state: InfoAggregateState | None = None,
    calibration_prior: dict[str, float] | None = None,
    tradability_status: str = "normal",
) -> tuple[float, str]:
    return _reporting_forecast_support.intrinsic_confidence(
        up_prob=up_prob,
        horizon_probs=horizon_probs,
        info_state=info_state,
        calibration_prior=calibration_prior,
        tradability_status=tradability_status,
    )


def _build_horizon_forecasts(
    *,
    latest_close: float,
    horizon_probs: dict[str, float],
    short_profile: _ReturnQuantileProfile | None,
    mid_profile: _ReturnQuantileProfile | None,
    info_state: InfoAggregateState | None = None,
    calibration_priors: dict[str, dict[str, float]] | None = None,
    tradability_status: str = "normal",
) -> dict[str, HorizonForecast]:
    return _reporting_forecast_support.build_horizon_forecasts(
        latest_close=latest_close,
        horizon_probs=horizon_probs,
        short_profile=short_profile,
        mid_profile=mid_profile,
        info_state=info_state,
        calibration_priors=calibration_priors,
        tradability_status=tradability_status,
        deps=_forecast_support_dependencies(),
    )


_sentiment_stage = _reporting_forecast_support.sentiment_stage
_pct_text = _reporting_forecast_support.pct_text
_num_text = _reporting_forecast_support.num_text
_build_market_sentiment_state = _reporting_forecast_support.build_market_sentiment_state


_market_facts_from_row = _v2_runtime_primitives.market_facts_from_row


_build_date_slice_index = _v2_backtest_metrics_runtime.build_date_slice_index
_distributional_score = _v2_backtest_metrics_runtime.distributional_score


_status_score_penalty = _reporting_forecast_support.status_score_penalty


def _alpha_score_components(stock: StockForecastState) -> dict[str, float]:
    return _reporting_forecast_support.alpha_score_components(
        stock,
        deps=_forecast_support_dependencies(),
    )


_is_actionable_status = _v2_runtime_primitives.is_actionable_status
_status_tradeability_limit = _v2_runtime_primitives.status_tradeability_limit


def _build_stock_states_from_panel_slice(
    *,
    panel_row: pd.DataFrame,
    feature_cols: list[str],
    short_model: LogisticBinaryModel | MLPBinaryModel,
    two_model: LogisticBinaryModel | MLPBinaryModel,
    three_model: LogisticBinaryModel | MLPBinaryModel,
    five_model: LogisticBinaryModel | MLPBinaryModel,
    mid_model: LogisticBinaryModel | MLPBinaryModel,
    short_q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
    mid_q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> tuple[list[StockForecastState], pd.DataFrame]:
    params = locals().copy()
    return _v2_state_build_runtime.build_stock_states_from_panel_slice(
        deps=_state_build_runtime_dependencies(),
        **params,
    )


_panel_slice_metrics = _v2_backtest_metrics_runtime.panel_slice_metrics
_panel_horizon_metrics = _v2_backtest_metrics_runtime.panel_horizon_metrics


build_strategy_snapshot = _build_strategy_snapshot_external
_load_v2_runtime_settings = _load_v2_runtime_settings_external


_extract_universe_rows = _extract_universe_rows_external
_hydrate_universe_metadata = _hydrate_universe_metadata_external


def _resolve_v2_universe_settings(
    *,
    settings: dict[str, object],
    cache_root: str,
) -> dict[str, object]:
    return _resolve_v2_universe_settings_external(
        settings=settings,
        cache_root=cache_root,
        generate_dynamic_universe_fn=generate_dynamic_universe,
        build_predefined_universe_fn=build_predefined_universe,
        normalize_universe_tier_fn=normalize_universe_tier,
    )


def _build_market_and_cross_section_states(**kwargs: object) -> tuple[MarketForecastState, CrossSectionForecastState]:
    return _v2_state_build_runtime.build_market_and_cross_section_states(
        deps=_state_build_runtime_dependencies(),
        **kwargs,
    )



def _build_stock_states_from_rows(*args: object, **kwargs: object) -> list[StockForecastState]:
    return _v2_state_build_runtime.build_stock_states_from_rows(
        *args,
        deps=_state_build_runtime_dependencies(),
        **kwargs,
    )


def compose_state(**kwargs: object) -> CompositeState:
    return _v2_state_build_runtime.compose_state(
        deps=_state_build_runtime_dependencies(),
        **kwargs,
    )


def _profile_from_horizon_map(
    forecasts: dict[str, HorizonForecast],
    key: str,
) -> _ReturnQuantileProfile | None:
    return _report_state_runtime.profile_from_horizon_map(
        forecasts,
        key,
        return_quantile_profile_cls=_ReturnQuantileProfile,
    )


def _load_prediction_review_context(
    *,
    manifest: dict[str, object],
    manifest_path: Path | None,
) -> tuple[PredictionReviewState, dict[str, dict[str, float]]]:
    return _prediction_review.load_prediction_review_context(
        manifest=manifest,
        manifest_path=manifest_path,
        deps=_prediction_review_dependencies(),
    )


def _stock_reason_bundle(**kwargs: object) -> tuple[list[str], list[str], list[str], str, str, str, str]:
    return _reporting_reason_bundles.stock_reason_bundle(
        alpha_score_components=_alpha_score_components,
        **kwargs,
    )


def _decorate_composite_state_for_reporting(**kwargs: object) -> CompositeState:
    return _report_state_runtime.decorate_composite_state_for_reporting(
        deps=_report_state_runtime_dependencies(),
        **kwargs,
    )


def _filter_state_for_recommendation_scope(
    *,
    state: CompositeState,
    main_board_only: bool,
) -> CompositeState:
    return _report_state_runtime.filter_state_for_recommendation_scope(
        state=state,
        main_board_only=main_board_only,
        deps=_report_state_runtime_dependencies(),
    )


def _build_live_market_reporting_overlay(
    *,
    settings: dict[str, object],
    universe_ctx: _DailyUniverseContext,
    state: CompositeState,
) -> tuple[MarketForecastState | None, CrossSectionForecastState | None]:
    return _report_state_runtime.build_live_market_reporting_overlay(
        settings=settings,
        universe_ctx=universe_ctx,
        state=state,
        deps=_report_state_runtime_dependencies(),
    )


def _policy_runtime_dependencies() -> _v2_policy_runtime.PolicyRuntimeDependencies:
    return _v2_facade_dependency_builders.build_policy_runtime_dependencies(sys.modules[__name__])


def _policy_feature_runtime_dependencies() -> _v2_policy_feature_runtime.PolicyFeatureRuntimeDependencies:
    return _v2_facade_dependency_builders.build_policy_feature_runtime_dependencies(sys.modules[__name__])


def _backtest_execution_dependencies() -> _v2_backtest_runtime.BacktestExecutionDependencies:
    return _v2_facade_dependency_builders.build_backtest_execution_dependencies(sys.modules[__name__])


def _backtest_core_dependencies() -> _v2_backtest_runtime.BacktestCoreDependencies:
    return _v2_facade_dependency_builders.build_backtest_core_dependencies(sys.modules[__name__])


def _backtest_prepare_dependencies() -> _v2_backtest_prepare_runtime.BacktestPrepareDependencies:
    return _v2_facade_dependency_builders.build_backtest_prepare_dependencies(sys.modules[__name__])


def _state_build_runtime_dependencies() -> _v2_state_build_runtime.StateBuildRuntimeDependencies:
    return _v2_facade_dependency_builders.build_state_build_runtime_dependencies(sys.modules[__name__])


def _learning_target_dependencies() -> _v2_learning_target_runtime.LearningTargetDependencies:
    return _v2_facade_dependency_builders.build_learning_target_dependencies(sys.modules[__name__])


def _info_shadow_runtime_dependencies() -> _v2_info_shadow_runtime.InfoShadowRuntimeDependencies:
    return _v2_facade_dependency_builders.build_info_shadow_runtime_dependencies(sys.modules[__name__])


def _frozen_forecast_runtime_dependencies() -> _v2_frozen_forecast_runtime.FrozenForecastRuntimeDependencies:
    return _v2_facade_dependency_builders.build_frozen_forecast_runtime_dependencies(sys.modules[__name__])


def _frozen_forecast_bundle_dependencies() -> _v2_frozen_forecast_runtime.FrozenForecastBundleDependencies:
    return _v2_facade_dependency_builders.build_frozen_forecast_bundle_dependencies(sys.modules[__name__])


def _forecast_runtime_dependencies() -> _v2_feature_runtime.ForecastRuntimeDependencies:
    return _v2_facade_dependency_builders.build_forecast_runtime_dependencies(sys.modules[__name__])


def _prediction_review_dependencies() -> _prediction_review.PredictionReviewDependencies:
    return _v2_facade_dependency_builders.build_prediction_review_dependencies(sys.modules[__name__])


def _report_state_runtime_dependencies() -> _report_state_runtime.ReportStateRuntimeDependencies:
    return _v2_facade_dependency_builders.build_report_state_runtime_dependencies(sys.modules[__name__])


def _forecast_support_dependencies() -> _reporting_forecast_support.ForecastSupportDependencies:
    return _v2_facade_dependency_builders.build_forecast_support_dependencies(sys.modules[__name__])


def _info_shadow_report_dependencies() -> _info_shadow_report.InfoShadowReportDependencies:
    return _v2_facade_dependency_builders.build_info_shadow_report_dependencies(sys.modules[__name__])


def _info_manifest_dependencies() -> _info_manifest_runtime.InfoManifestDependencies:
    return _v2_facade_dependency_builders.build_info_manifest_dependencies(sys.modules[__name__])


def _policy_learning_dependencies() -> _v2_policy_learning_runtime.PolicyLearningDependencies:
    return _v2_facade_dependency_builders.build_policy_learning_dependencies(sys.modules[__name__])


def _daily_state_runtime_dependencies() -> _v2_daily_state_runtime.DailyStateRuntimeDependencies:
    return _v2_facade_dependency_builders.build_daily_state_runtime_dependencies(sys.modules[__name__])


def _ranked_sector_budgets(sectors: Iterable[SectorForecastState], *, target_exposure: float) -> dict[str, float]:
    return _v2_policy_runtime.ranked_sector_budgets(
        sectors,
        target_exposure=target_exposure,
    )


def _alpha_opportunity_metrics(stocks: Iterable[StockForecastState]) -> dict[str, float]:
    return _v2_policy_runtime.alpha_opportunity_metrics(
        stocks,
        deps=_policy_runtime_dependencies(),
    )


def _mainline_preference_maps(
    mainlines: Iterable[MainlineState],
    *,
    risk_cutoff: float,
) -> tuple[dict[str, float], dict[str, float], list[MainlineState]]:
    return _v2_policy_runtime.mainline_preference_maps(
        mainlines,
        risk_cutoff=risk_cutoff,
        deps=_policy_runtime_dependencies(),
    )


def _ranked_sector_budgets_with_alpha(**kwargs: object) -> dict[str, float]:
    return _v2_policy_runtime.ranked_sector_budgets_with_alpha(
        deps=_policy_runtime_dependencies(),
        **kwargs,
    )


def _cap_sector_budgets(**kwargs: object) -> tuple[dict[str, float], list[str]]:
    return _v2_policy_runtime.cap_sector_budgets(
        deps=_policy_runtime_dependencies(),
        **kwargs,
    )


def _stock_policy_score(stock: StockForecastState) -> float:
    return _v2_policy_runtime.stock_policy_score(
        stock,
        deps=_policy_runtime_dependencies(),
    )


_policy_objective_score = _v2_facade_support_runtime.policy_objective_score


def _allocate_sector_slots(**kwargs: object) -> dict[str, int]:
    return _v2_policy_runtime.allocate_sector_slots(
        deps=_policy_runtime_dependencies(),
        **kwargs,
    )


def _allocate_with_sector_budgets(**kwargs: object) -> dict[str, float]:
    return _v2_policy_runtime.allocate_with_sector_budgets(
        deps=_policy_runtime_dependencies(),
        **kwargs,
    )


def _finalize_target_weights(**kwargs: object) -> tuple[dict[str, float], list[str]]:
    return _v2_policy_runtime.finalize_target_weights(
        deps=_policy_runtime_dependencies(),
        **kwargs,
    )


def _apply_external_signal_weight_tilts(**kwargs: object) -> tuple[dict[str, float], list[str]]:
    return _v2_policy_runtime.apply_external_signal_weight_tilts(
        deps=_policy_runtime_dependencies(),
        **kwargs,
    )


def _enforce_single_name_cap(
    *,
    weights: dict[str, float],
    max_single_position: float,
) -> dict[str, float]:
    return _v2_policy_runtime.enforce_single_name_cap(
        weights=weights,
        max_single_position=max_single_position,
    )


def _sector_budgets_from_weights(
    *,
    symbol_weights: dict[str, float],
    stocks: list[StockForecastState],
) -> dict[str, float]:
    return _v2_policy_runtime.sector_budgets_from_weights(
        symbol_weights=symbol_weights,
        stocks=stocks,
    )


def _advance_holding_days(
    *,
    prev_holding_days: dict[str, int],
    prev_weights: dict[str, float],
    next_weights: dict[str, float],
) -> dict[str, int]:
    return _v2_policy_runtime.advance_holding_days(
        prev_holding_days=prev_holding_days,
        prev_weights=prev_weights,
        next_weights=next_weights,
    )


def apply_policy(
    policy_input: PolicyInput,
    *,
    policy_spec: PolicySpec | None = None,
) -> PolicyDecision:
    return _v2_policy_runtime.apply_policy(
        policy_input,
        policy_spec=policy_spec,
        deps=_policy_runtime_dependencies(),
    )


def build_trade_actions(
    *,
    decision: PolicyDecision,
    current_weights: dict[str, float],
) -> list[TradeAction]:
    return _v2_policy_runtime.build_trade_actions(
        decision=decision,
        current_weights=current_weights,
    )


def _policy_spec_from_model(
    *,
    state: CompositeState,
    model: LearnedPolicyModel,
) -> PolicySpec:
    return _v2_policy_runtime.policy_spec_from_model(
        state=state,
        model=model,
        deps=_policy_runtime_dependencies(),
    )


def _simulate_execution_day(**kwargs: object) -> tuple[float, float, float, float, float, dict[str, float], float]:
    return _v2_backtest_runtime.simulate_execution_day(
        deps=_backtest_execution_dependencies(),
        **kwargs,
    )


def _to_v2_backtest_summary(**kwargs: object) -> V2BacktestSummary:
    return _v2_backtest_runtime.to_v2_backtest_summary(**kwargs)


def _build_market_and_cross_section_from_prebuilt_frame(
    **kwargs: object,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    return _v2_state_build_runtime.build_market_and_cross_section_from_prebuilt_frame(
        deps=_state_build_runtime_dependencies(),
        **kwargs,
    )



def _derive_learning_targets(**kwargs: object) -> tuple[float, float, float, float]:
    return _v2_learning_target_runtime.derive_learning_targets(
        deps=_learning_target_dependencies(),
        **kwargs,
    )


_PreparedV2BacktestData = _v2_backtest_prepare_runtime.PreparedV2BacktestData
_TrajectoryStep = _v2_backtest_prepare_runtime.TrajectoryStep
_BacktestTrajectory = _v2_backtest_prepare_runtime.BacktestTrajectory


ForecastBackend = _v2_feature_runtime.ForecastBackend


def _slice_backtest_trajectory(
    trajectory: _BacktestTrajectory,
    *,
    start: int,
    end: int,
) -> _BacktestTrajectory:
    return _v2_backtest_prepare_runtime.slice_backtest_trajectory(
        trajectory,
        start=start,
        end=end,
    )


def _split_research_trajectory(
    trajectory: _BacktestTrajectory,
    split_mode: str = _DEFAULT_SPLIT_MODE,
    embargo_days: int = _DEFAULT_EMBARGO_DAYS,
) -> tuple[_BacktestTrajectory, _BacktestTrajectory, _BacktestTrajectory]:
    train, validation, holdout = _v2_backtest_prepare_runtime.split_research_trajectory(
        trajectory,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )
    return train, validation, holdout


def _fit_v2_info_shadow_models(
    *,
    trajectory: _BacktestTrajectory,
    settings: dict[str, object],
    info_items: list[InfoItem],
) -> tuple[dict[str, _InfoShadowModel], dict[str, _InfoShadowModel]]:
    return _v2_info_shadow_runtime.fit_v2_info_shadow_models(
        trajectory=trajectory,
        settings=settings,
        info_items=info_items,
        deps=_info_shadow_runtime_dependencies(),
    )


def _build_shadow_scored_rows_for_step(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    return _v2_info_shadow_runtime.build_shadow_scored_rows_for_step(
        state=state,
        stock_frames=stock_frames,
        date=date,
        deps=_info_shadow_runtime_dependencies(),
    )


_filter_info_items_by_source_subset = _v2_facade_support_runtime.filter_info_items_by_source_subset


def _info_source_breakdown(items: Iterable[InfoItem]) -> dict[str, int]:
    return _info_manifest_runtime.info_source_breakdown(items)


def _build_info_shadow_variant(**kwargs: object) -> dict[str, object]:
    return _v2_info_shadow_runtime.build_info_shadow_variant(
        deps=_info_shadow_runtime_dependencies(),
        **kwargs,
    )


def _build_info_shadow_report(**kwargs: object) -> dict[str, object]:
    return _info_shadow_report.build_info_shadow_report(
        deps=_info_shadow_report_dependencies(),
        **kwargs,
    )


def _build_info_manifest_payload(**kwargs: object) -> dict[str, object]:
    return _info_manifest_runtime.build_info_manifest_payload(
        deps=_info_manifest_dependencies(),
        **kwargs,
    )


def _empty_v2_backtest_result() -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    return (
        _to_v2_backtest_summary(
            returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        ),
        [],
    )


def _prepare_v2_backtest_data(**kwargs: object) -> _PreparedV2BacktestData | None:
    return _v2_backtest_prepare_runtime.prepare_v2_backtest_data(
        prepared_dataclass=_PreparedV2BacktestData,
        deps=_backtest_prepare_dependencies(),
        **kwargs,
    )


def _build_frozen_linear_forecast_bundle(
    prepared: _PreparedV2BacktestData,
) -> dict[str, object]:
    return _v2_frozen_forecast_runtime.build_frozen_linear_forecast_bundle(
        prepared,
        deps=_frozen_forecast_bundle_dependencies(),
    )


def _load_frozen_forecast_bundle(path_like: object) -> dict[str, object]:
    return _v2_frozen_forecast_runtime.load_frozen_forecast_bundle(
        path_like,
        load_json_dict=_load_json_dict,
        forecast_bundle_cls=ForecastBundle,
    )


def _build_live_market_frame(
    *,
    settings: dict[str, object],
    market_symbol: str,
) -> pd.DataFrame:
    return _v2_frozen_forecast_runtime.build_live_market_frame(
        settings=settings,
        market_symbol=market_symbol,
        deps=_frozen_forecast_runtime_dependencies(),
    )



def _score_live_composite_state_from_frozen_bundle(
    **kwargs: object,
) -> tuple[CompositeState | None, list[object]]:
    return _v2_frozen_forecast_runtime.score_live_composite_state_from_frozen_bundle(
        deps=_frozen_forecast_runtime_dependencies(),
        **kwargs,
    )



def _tensorize_temporal_frame(
    frame: pd.DataFrame,
    **kwargs: object,
) -> tuple[pd.DataFrame, list[str]]:
    return _v2_feature_runtime.tensorize_temporal_frame(
        frame,
        **kwargs,
    )


def _make_forecast_backend(name: str | None) -> ForecastBackend:
    return _v2_feature_runtime.make_forecast_backend(
        name,
        deps=_forecast_runtime_dependencies(),
    )


_trajectory_cache_key = _v2_backtest_prepare_runtime.trajectory_cache_key
_trajectory_cache_path = _v2_backtest_prepare_runtime.trajectory_cache_path
_file_mtime_token = _v2_daily_runtime.file_mtime_token


def _daily_result_cache_key(**kwargs: object) -> str:
    return _v2_daily_runtime.daily_result_cache_key(
        deps=_v2_daily_runtime.DailyCacheKeyDependencies(
            resolve_manifest_path=_resolve_manifest_path,
            resolve_info_file_from_settings=_resolve_info_file_from_settings,
        ),
        **kwargs,
    )


_daily_result_cache_path = _v2_daily_runtime.daily_result_cache_path


def _load_or_build_v2_backtest_trajectory(**kwargs: object) -> _BacktestTrajectory | None:
    return _v2_backtest_prepare_runtime.load_or_build_v2_backtest_trajectory(
        deps=_backtest_prepare_dependencies(),
        **kwargs,
    )


def _build_v2_backtest_trajectory_from_prepared(
    prepared: _PreparedV2BacktestData,
    *,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
) -> _BacktestTrajectory:
    return _v2_backtest_prepare_runtime.build_v2_backtest_trajectory_from_prepared(
        prepared,
        retrain_days=retrain_days,
        forecast_backend=forecast_backend,
        deps=_backtest_prepare_dependencies(),
    )


def _execute_v2_backtest_trajectory(
    trajectory: _BacktestTrajectory,
    **kwargs: object,
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    return _v2_backtest_runtime.execute_v2_backtest_trajectory(
        trajectory,
        deps=_backtest_execution_dependencies(),
        **kwargs,
    )


def _run_v2_backtest_core(**kwargs: object) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    return _v2_backtest_runtime.run_v2_backtest_core(
        deps=_backtest_core_dependencies(),
        **kwargs,
    )


def run_v2_backtest_live(**kwargs: object) -> V2BacktestSummary:
    summary, _ = _run_v2_backtest_core(capture_learning_rows=False, **kwargs)
    return summary


def calibrate_v2_policy(**kwargs: object) -> V2CalibrationResult:
    return _v2_policy_learning_runtime.calibrate_v2_policy(
        deps=_policy_learning_dependencies(),
        **kwargs,
    )


def learn_v2_policy_model(**kwargs: object) -> V2PolicyLearningResult:
    return _v2_policy_learning_runtime.learn_v2_policy_model(
        deps=_policy_learning_dependencies(),
        **kwargs,
    )


def _baseline_only_calibration(baseline: V2BacktestSummary) -> V2CalibrationResult:
    return _v2_policy_learning_runtime.baseline_only_calibration(
        baseline,
        deps=_policy_learning_dependencies(),
    )


def _placeholder_learning_result(baseline: V2BacktestSummary) -> V2PolicyLearningResult:
    return _v2_policy_learning_runtime.placeholder_learning_result(
        baseline,
        deps=_policy_learning_dependencies(),
    )


def _run_v2_research_workflow_impl(
    **kwargs: object,
) -> tuple[V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult]:
    from src.workflows.research_workflow import run_v2_research_workflow_impl as _run_impl

    return _run_impl(
        dependencies=_research_workflow_dependencies(),
        **kwargs,
    )


def _run_v2_research_matrix_impl(**kwargs: object) -> dict[str, object]:
    from src.workflows.research_workflow import run_v2_research_matrix_impl as _run_impl

    return _run_impl(
        dependencies=_research_workflow_dependencies(),
        **kwargs,
    )


def _load_published_v2_policy_model_impl(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
) -> LearnedPolicyModel | None:
    return _load_published_v2_policy_model_external(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        load_policy_model_from_path_fn=_load_policy_model_from_path,
    )


_load_policy_model_from_path = _load_policy_model_from_path_runtime_external
_decode_composite_state = _decode_composite_state_external
_serialize_composite_state = _serialize_composite_state_external


def _build_frozen_daily_state_payload(
    *,
    trajectory: _BacktestTrajectory | None,
    split_mode: str,
    embargo_days: int,
) -> dict[str, object]:
    return _build_frozen_daily_state_payload_external(
        trajectory=trajectory,
        split_mode=split_mode,
        embargo_days=embargo_days,
        split_trajectory=_split_research_trajectory,
    )


def _pass_release_gate(
    *,
    baseline: V2BacktestSummary,
    candidate: V2BacktestSummary,
) -> tuple[bool, list[str]]:
    return _pass_release_gate_external(
        baseline=baseline,
        candidate=candidate,
        threshold=_RELEASE_GATE_THRESHOLD,
    )


_tier_latest_manifest_path = _tier_latest_manifest_path_external
_tier_latest_policy_path = _tier_latest_policy_path_external
_summary_from_payload = _summary_from_payload_external


def _load_backtest_payload_from_manifest(manifest_payload: dict[str, object], manifest_path: Path) -> dict[str, object]:
    return _load_backtest_payload_from_manifest_external(
        manifest_payload,
        manifest_path,
        path_from_manifest_entry=lambda entry: _path_from_manifest_entry(entry, run_dir=manifest_path.parent),
        load_json_dict=_load_json_dict,
    )


def _load_backtest_payload_for_run(
    *,
    artifact_root: str,
    strategy_id: str,
    run_id: str,
) -> dict[str, object]:
    return _load_backtest_payload_for_run_external(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        run_id=run_id,
        load_json_dict=_load_json_dict,
    )


def _pass_default_switch_gate(
    *,
    baseline_reference: V2BacktestSummary,
    candidate: V2BacktestSummary,
) -> tuple[bool, list[str], dict[str, float]]:
    return _pass_default_switch_gate_external(
        baseline_reference=baseline_reference,
        candidate=candidate,
        threshold=_DEFAULT_SWITCH_GATE_THRESHOLD,
    )


def _research_publish_dependencies() -> _ResearchPublishDependenciesExternal:
    return _build_research_publish_dependencies_external()


def _research_workflow_dependencies():
    return _v2_facade_dependency_builders.build_research_workflow_dependencies(sys.modules[__name__])


def _load_research_manifest_for_daily(
    *,
    strategy_id: str,
    artifact_root: str,
    run_id: str | None,
    snapshot_path: str | None,
) -> tuple[dict[str, object], Path]:
    return _load_research_manifest_for_daily_runtime_external(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        run_id=run_id,
        snapshot_path=snapshot_path,
        resolve_manifest_path_fn=_resolve_manifest_path,
        load_json_dict=_load_json_dict,
    )


def _build_snapshot_from_manifest(
    *,
    strategy_id: str,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
) -> StrategySnapshot:
    return _build_snapshot_from_manifest_external(
        strategy_id=strategy_id,
        settings=settings,
        manifest=manifest,
        manifest_path=manifest_path,
        path_from_manifest_entry=_path_from_manifest_entry,
        load_json_dict=_load_json_dict,
        parse_boolish=_parse_boolish,
        stable_json_hash=_stable_json_hash,
        compose_run_snapshot_hash=_compose_run_snapshot_hash,
    )


def _publish_v2_research_artifacts_impl(**kwargs: object) -> dict[str, str]:
    return _publish_research_artifacts_runtime_external(
        dependencies=_research_publish_dependencies(),
        **kwargs,
    )



_DailySnapshotContext = _DailySnapshotContextExternal
_DailyUniverseContext = _v2_daily_state_runtime.DailyUniverseContext


def _load_daily_cached_result(
    *,
    cache_path: Path,
    refresh_cache: bool,
    memory_root: Path,
) -> DailyRunResult | None:
    params = locals().copy()
    return _load_daily_cached_result_external(
        **params,
        emit_progress=_emit_progress,
    )


def _hydrate_daily_settings_from_dataset_manifest(
    *,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
    universe_tier: str | None,
    universe_file: str | None,
) -> dict[str, object]:
    params = locals().copy()
    return _hydrate_daily_settings_from_dataset_manifest_external(
        **params,
        path_from_manifest_entry=_path_from_manifest_entry,
        load_json_dict=_load_json_dict,
        parse_boolish=_parse_boolish,
    )


_is_daily_universe_override_mismatch = _is_daily_universe_override_mismatch_external


def _build_daily_snapshot_context(**kwargs: object) -> _DailySnapshotContext:
    runtime_kwargs = dict(kwargs)
    runtime_kwargs.update(
        {
            "load_v2_runtime_settings": _load_v2_runtime_settings,
            "resolve_v2_universe_settings": _resolve_v2_universe_settings,
            "load_research_manifest_for_daily_fn": _load_research_manifest_for_daily,
            "hydrate_daily_settings_from_dataset_manifest_fn": _hydrate_daily_settings_from_dataset_manifest,
            "build_snapshot_from_manifest_fn": _build_snapshot_from_manifest,
            "parse_boolish": _parse_boolish,
            "stable_json_hash": _stable_json_hash,
            "sha256_file": _sha256_file,
            "emit_progress": _emit_progress,
        }
    )
    return _build_daily_snapshot_context_external(
        **runtime_kwargs,
    )


def _build_daily_universe_context(settings: dict[str, object]) -> _DailyUniverseContext:
    return _v2_daily_state_runtime.build_daily_universe_context(
        settings,
        deps=_daily_state_runtime_dependencies(),
    )


def _build_daily_composite_state(
    *,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path | None,
    snapshot: StrategySnapshot,
    allow_retrain: bool,
    universe_ctx: _DailyUniverseContext,
) -> tuple[CompositeState, list[object]]:
    return _v2_daily_state_runtime.build_daily_composite_state(
        settings=settings,
        manifest=manifest,
        manifest_path=manifest_path,
        snapshot=snapshot,
        allow_retrain=allow_retrain,
        universe_ctx=universe_ctx,
        deps=_daily_state_runtime_dependencies(),
    )


def _build_daily_symbol_names(
    *,
    current_holdings: list[object],
    stocks: list[object],
    stock_rows: list[object],
    composite_state: CompositeState,
) -> dict[str, str]:
    return _build_daily_symbol_names_external(
        current_holdings=current_holdings,
        stocks=stocks,
        stock_rows=stock_rows,
        composite_state=composite_state,
    )


def _attach_daily_info_overlay(
    *,
    snapshot: StrategySnapshot,
    settings: dict[str, object],
    composite_state: CompositeState,
    symbol_names: dict[str, str],
) -> tuple[
    CompositeState,
    str,
    str,
    bool,
    int,
    list[InfoSignalRecord],
    list[InfoSignalRecord],
    list[InfoDivergenceRecord],
    list[InfoItem],
]:
    return _v2_daily_state_runtime.attach_daily_info_overlay(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        symbol_names=symbol_names,
        deps=_daily_state_runtime_dependencies(),
    )


def _attach_daily_external_signal_overlay(
    *,
    snapshot: StrategySnapshot,
    settings: dict[str, object],
    composite_state: CompositeState,
    info_items: list[InfoItem],
    allow_rebuild: bool,
) -> tuple[CompositeState, str, str, bool, dict[str, object], dict[str, object]]:
    return _v2_daily_state_runtime.attach_daily_external_signal_overlay(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        info_items=info_items,
        allow_rebuild=allow_rebuild,
        deps=_daily_state_runtime_dependencies(),
    )


def _resolve_daily_policy_model(
    *,
    strategy_id: str,
    artifact_root: str,
    manifest: dict[str, object],
    manifest_path: Path | None,
) -> LearnedPolicyModel | None:
    return _resolve_daily_policy_model_external(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        manifest=manifest,
        manifest_path=manifest_path,
        path_from_manifest_entry=_path_from_manifest_entry,
        load_policy_model_from_path_fn=_load_policy_model_from_path,
        load_published_v2_policy_model_fn=_load_published_v2_policy_model_impl,
    )


def _daily_workflow_dependencies():
    return _v2_facade_dependency_builders.build_daily_workflow_dependencies(sys.modules[__name__])


def _run_daily_v2_live_impl(**kwargs: object) -> DailyRunResult:
    from src.workflows.daily_workflow import run_daily_v2_live_impl as _run_impl

    return _run_impl(
        dependencies=_daily_workflow_dependencies(),
        **kwargs,
    )


def run_v2_research_workflow(**kwargs: object) -> tuple[V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult]:
    from src.workflows.research_workflow import run_v2_research_workflow as _run

    return _run(**kwargs)


def run_v2_research_matrix(**kwargs: object) -> dict[str, object]:
    from src.workflows.research_workflow import run_v2_research_matrix as _run

    return _run(**kwargs)


def load_published_v2_policy_model(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
) -> LearnedPolicyModel | None:
    from src.artifact_registry.v2_registry import load_published_v2_policy_model as _load

    return _load(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
    )


def publish_v2_research_artifacts(
    *,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    settings: dict[str, object] | None = None,
    **kwargs: object,
) -> dict[str, str]:
    from src.artifact_registry.v2_registry import publish_v2_research_artifacts as _publish

    return _publish(
        baseline=baseline,
        calibration=calibration,
        learning=learning,
        settings=settings,
        **kwargs,
    )


def run_daily_v2_live(**kwargs: object) -> DailyRunResult:
    from src.workflows.daily_workflow import run_daily_v2_live as _run

    return _run(**kwargs)


def summarize_daily_run(result: DailyRunResult) -> dict[str, object]:
    from src.review_analytics.summaries import summarize_daily_run as _summarize

    return _summarize(result)


def summarize_v2_backtest(
    result: V2BacktestSummary,
    *,
    run_id: str | None = None,
    snapshot_hash: str | None = None,
    config_hash: str | None = None,
) -> dict[str, object]:
    from src.review_analytics.summaries import summarize_v2_backtest as _summarize

    return _summarize(
        result,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
    )


def summarize_v2_calibration(result: V2CalibrationResult) -> dict[str, object]:
    from src.review_analytics.summaries import summarize_v2_calibration as _summarize

    return _summarize(result)


def summarize_v2_policy_learning(result: V2PolicyLearningResult) -> dict[str, object]:
    from src.review_analytics.summaries import summarize_v2_policy_learning as _summarize

    return _summarize(result)
