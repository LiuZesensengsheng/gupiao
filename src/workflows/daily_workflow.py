from __future__ import annotations

import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from src.application.v2_contracts import DailyRunResult, PolicyInput, PolicySpec
from src.contracts.runtime import DailyRunOptions


@dataclass(frozen=True)
class DailyWorkflowDependencies:
    build_daily_snapshot_context_fn: Callable[..., object]
    daily_result_cache_key_fn: Callable[..., str]
    daily_result_cache_path_fn: Callable[..., Path]
    load_daily_cached_result_fn: Callable[..., DailyRunResult | None]
    build_daily_universe_context_fn: Callable[..., object]
    build_daily_composite_state_fn: Callable[..., tuple[object, list[dict[str, object]]]]
    build_daily_symbol_names_fn: Callable[..., dict[str, str]]
    attach_daily_info_overlay_fn: Callable[..., tuple[object, str, str, bool, int, list[object], list[object], list[object], list[object], object | None]]
    attach_daily_external_signal_overlay_fn: Callable[..., tuple[object, str, str, bool, dict[str, object], dict[str, object]]]
    attach_daily_insight_overlay_fn: Callable[..., object]
    filter_state_for_recommendation_scope_fn: Callable[..., object]
    apply_leader_candidate_overlay_fn: Callable[..., object]
    parse_boolish_fn: Callable[[object, bool], bool]
    resolve_daily_policy_model_fn: Callable[..., object | None]
    resolve_daily_exit_behavior_model_fn: Callable[..., dict[str, object] | None]
    policy_spec_from_model_fn: Callable[..., PolicySpec]
    apply_policy_fn: Callable[..., object]
    build_execution_plans_fn: Callable[..., list[object]]
    build_trade_actions_fn: Callable[..., list[object]]
    load_prediction_review_context_fn: Callable[..., tuple[object | None, dict[str, object]]]
    build_live_market_reporting_overlay_fn: Callable[..., tuple[object | None, object | None]]
    decorate_composite_state_for_reporting_fn: Callable[..., object]
    remember_daily_run_fn: Callable[..., DailyRunResult]
    persist_daily_insight_artifacts_fn: Callable[..., dict[str, str]]
    emit_progress_fn: Callable[[str, str], None]


def run_daily_v2_live_impl(
    *,
    dependencies: DailyWorkflowDependencies,
    strategy_id: str = "swing_v2",
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
    run_id: str | None = None,
    snapshot_path: str | None = None,
    allow_retrain: bool = False,
    disable_learned_policy: bool = False,
) -> DailyRunResult:
    memory_root = Path(str(artifact_root)) / "memory"
    snapshot_ctx = dependencies.build_daily_snapshot_context_fn(
        strategy_id=strategy_id,
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
        info_cutoff_time=info_cutoff_time,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        artifact_root=artifact_root,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        run_id=run_id,
        snapshot_path=snapshot_path,
        allow_retrain=allow_retrain,
    )
    settings = snapshot_ctx.settings

    cache_key = dependencies.daily_result_cache_key_fn(
        strategy_id=strategy_id,
        settings=settings,
        artifact_root=artifact_root,
        run_id=snapshot_ctx.resolved_run_id,
        snapshot_path=str(snapshot_path or ""),
        allow_retrain=allow_retrain,
        disable_learned_policy=disable_learned_policy,
    )
    cache_path = dependencies.daily_result_cache_path_fn(
        cache_root=cache_root,
        cache_key=cache_key,
    )
    cached = dependencies.load_daily_cached_result_fn(
        cache_path=cache_path,
        refresh_cache=refresh_cache,
        memory_root=memory_root,
    )
    if cached is not None:
        return cached

    snapshot = snapshot_ctx.snapshot
    manifest = snapshot_ctx.manifest
    manifest_path = snapshot_ctx.manifest_path
    universe_ctx = dependencies.build_daily_universe_context_fn(settings)
    composite_state, stock_rows = dependencies.build_daily_composite_state_fn(
        settings=settings,
        manifest=manifest,
        manifest_path=manifest_path,
        snapshot=snapshot,
        allow_retrain=allow_retrain,
        universe_ctx=universe_ctx,
    )

    current_weights = {}
    if universe_ctx.current_holdings:
        equal_weight = 1.0 / float(len(universe_ctx.current_holdings))
        current_weights = {item.symbol: float(equal_weight) for item in universe_ctx.current_holdings}
    symbol_names = dependencies.build_daily_symbol_names_fn(
        current_holdings=universe_ctx.current_holdings,
        stocks=universe_ctx.stocks,
        stock_rows=stock_rows,
        composite_state=composite_state,
    )
    (
        composite_state,
        info_hash,
        info_manifest_path,
        info_shadow_enabled,
        info_item_count,
        top_negative_info_events,
        top_positive_info_signals,
        quant_info_divergence,
        info_items,
        shadow_info_state,
    ) = dependencies.attach_daily_info_overlay_fn(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        symbol_names=symbol_names,
    )
    (
        composite_state,
        external_signal_manifest_path,
        external_signal_version,
        external_signal_enabled,
        capital_flow_snapshot,
        macro_context_snapshot,
    ) = dependencies.attach_daily_external_signal_overlay_fn(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        info_items=info_items,
        allow_rebuild=allow_retrain,
    )
    composite_state = dependencies.attach_daily_insight_overlay_fn(
        settings=settings,
        composite_state=composite_state,
        info_items=info_items,
    )
    composite_state = dependencies.filter_state_for_recommendation_scope_fn(
        state=composite_state,
        main_board_only=dependencies.parse_boolish_fn(settings.get("main_board_only_recommendations", False), False),
    )
    composite_state = dependencies.apply_leader_candidate_overlay_fn(
        state=composite_state,
    )

    learned_policy = None
    if not disable_learned_policy:
        learned_policy = dependencies.resolve_daily_policy_model_fn(
            strategy_id=strategy_id,
            artifact_root=artifact_root,
            manifest=manifest,
            manifest_path=manifest_path,
        )
    exit_behavior_model = None
    if dependencies.parse_boolish_fn(settings.get("enable_exit_behavior_overlay", True), True):
        exit_behavior_model = dependencies.resolve_daily_exit_behavior_model_fn(
            manifest=manifest,
            manifest_path=manifest_path,
        )
    active_policy_spec = None
    if learned_policy is not None:
        active_policy_spec = dependencies.policy_spec_from_model_fn(
            state=composite_state,
            model=learned_policy,
        )
    active_policy_spec = replace(
        active_policy_spec or PolicySpec(),
        event_risk_cutoff=float(settings.get("event_risk_cutoff", 0.55)),
        catalyst_boost_cap=float(settings.get("catalyst_boost_cap", 0.12)),
        flow_exposure_cap=float(settings.get("flow_exposure_cap", 0.08)),
    )

    policy_decision = dependencies.apply_policy_fn(
        PolicyInput(
            composite_state=composite_state,
            current_weights=current_weights,
            current_cash=max(0.0, 1.0 - sum(current_weights.values())),
            total_equity=1.0,
            current_holding_days={symbol: 5 for symbol in current_weights},
            exit_behavior_model=dict(exit_behavior_model or {}),
        ),
        policy_spec=active_policy_spec,
    )
    if dependencies.parse_boolish_fn(settings.get("execution_overlay_enabled", False), False):
        execution_plans = dependencies.build_execution_plans_fn(
            state=composite_state,
            policy_decision=policy_decision,
            current_weights=current_weights,
            current_holding_days={symbol: 5 for symbol in current_weights},
            symbol_names=symbol_names,
            settings=settings,
        )
        composite_state = replace(composite_state, execution_plans=execution_plans)
    trade_actions = dependencies.build_trade_actions_fn(
        decision=policy_decision,
        current_weights=current_weights,
    )
    prediction_review, calibration_priors = dependencies.load_prediction_review_context_fn(
        manifest=manifest,
        manifest_path=manifest_path,
    )
    reporting_state = composite_state
    if shadow_info_state is not None and not dependencies.parse_boolish_fn(settings.get("use_info_fusion", False), False):
        reporting_state = replace(
            composite_state,
            market_info_state=getattr(shadow_info_state, "market_info_state", composite_state.market_info_state),
            sector_info_states=dict(getattr(shadow_info_state, "sector_info_states", {})),
            stock_info_states=dict(getattr(shadow_info_state, "stock_info_states", {})),
        )
    reporting_market, reporting_cross_section = dependencies.build_live_market_reporting_overlay_fn(
        settings=settings,
        universe_ctx=universe_ctx,
        state=reporting_state,
    )
    composite_state = dependencies.decorate_composite_state_for_reporting_fn(
        state=reporting_state,
        policy=policy_decision,
        calibration_priors=calibration_priors,
        reporting_market=reporting_market,
        reporting_cross_section=reporting_cross_section,
    )
    result = DailyRunResult(
        snapshot=snapshot,
        composite_state=composite_state,
        policy_decision=policy_decision,
        trade_actions=trade_actions,
        symbol_names=symbol_names,
        info_hash=info_hash,
        info_manifest_path=info_manifest_path,
        info_shadow_enabled=info_shadow_enabled,
        info_item_count=info_item_count,
        external_signal_manifest_path=external_signal_manifest_path,
        external_signal_version=external_signal_version,
        external_signal_enabled=external_signal_enabled,
        capital_flow_snapshot=capital_flow_snapshot,
        macro_context_snapshot=macro_context_snapshot,
        generator_manifest_path=snapshot.generator_manifest_path,
        generator_version=snapshot.generator_version,
        generator_hash=snapshot.generator_hash,
        coarse_pool_size=snapshot.coarse_pool_size,
        refined_pool_size=snapshot.refined_pool_size,
        selected_pool_size=snapshot.selected_pool_size,
        theme_allocations=list(snapshot.theme_allocations),
        top_negative_info_events=top_negative_info_events,
        top_positive_info_signals=top_positive_info_signals,
        quant_info_divergence=quant_info_divergence,
        run_id=snapshot.run_id,
        snapshot_hash=snapshot.snapshot_hash,
        config_hash=snapshot.config_hash,
        manifest_path=snapshot.manifest_path,
        prediction_review=prediction_review,
    )
    result = dependencies.remember_daily_run_fn(
        memory_root=memory_root,
        result=result,
    )
    try:
        dependencies.persist_daily_insight_artifacts_fn(
            result=result,
            settings=settings,
            artifact_root=artifact_root,
        )
    except Exception:
        pass
    try:
        with cache_path.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        dependencies.emit_progress_fn("daily", "日运行缓存已写入")
    except Exception:
        pass
    return result


def run_daily_v2_live(
    *,
    options: DailyRunOptions | None = None,
    **kwargs: object,
):
    from src.application import v2_services as legacy

    resolved = options or DailyRunOptions.from_kwargs(**kwargs)
    return run_daily_v2_live_impl(
        dependencies=legacy._daily_workflow_dependencies(),
        **resolved.workflow_kwargs(),
    )
