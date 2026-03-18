from __future__ import annotations

from dataclasses import asdict

from src.application.v2_contracts import DailyRunResult, V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult


def summarize_daily_run(result: DailyRunResult) -> dict[str, object]:
    def _display_name(symbol: str) -> str:
        return str(result.symbol_names.get(symbol, symbol))

    policy_payload = asdict(result.policy_decision)
    policy_payload["symbol_target_weights"] = {
        _display_name(symbol): weight
        for symbol, weight in result.policy_decision.symbol_target_weights.items()
    }
    policy_payload["desired_symbol_target_weights"] = {
        _display_name(symbol): weight
        for symbol, weight in result.policy_decision.desired_symbol_target_weights.items()
    }
    return {
        "strategy_id": result.snapshot.strategy_id,
        "universe_id": result.snapshot.universe_id,
        "universe_size": result.snapshot.universe_size,
        "universe_generation_rule": result.snapshot.universe_generation_rule,
        "source_universe_manifest_path": result.snapshot.source_universe_manifest_path,
        "info_hash": result.info_hash or result.snapshot.info_hash,
        "info_manifest_path": result.info_manifest_path or result.snapshot.info_manifest_path,
        "info_shadow_enabled": bool(result.info_shadow_enabled or result.snapshot.info_shadow_enabled),
        "info_item_count": int(result.info_item_count),
        "external_signal_manifest_path": result.external_signal_manifest_path or result.snapshot.external_signal_manifest_path,
        "external_signal_version": result.external_signal_version or result.snapshot.external_signal_version,
        "external_signal_enabled": bool(result.external_signal_enabled or result.snapshot.external_signal_enabled),
        "capital_flow_snapshot": dict(result.capital_flow_snapshot or result.snapshot.capital_flow_snapshot),
        "macro_context_snapshot": dict(result.macro_context_snapshot or result.snapshot.macro_context_snapshot),
        "generator_manifest_path": result.generator_manifest_path or result.snapshot.generator_manifest_path,
        "generator_version": result.generator_version or result.snapshot.generator_version,
        "generator_hash": result.generator_hash or result.snapshot.generator_hash,
        "coarse_pool_size": int(result.coarse_pool_size or result.snapshot.coarse_pool_size),
        "refined_pool_size": int(result.refined_pool_size or result.snapshot.refined_pool_size),
        "selected_pool_size": int(result.selected_pool_size or result.snapshot.selected_pool_size),
        "theme_allocations": [dict(item) for item in (result.theme_allocations or result.snapshot.theme_allocations)],
        "use_us_index_context": bool(result.snapshot.use_us_index_context),
        "us_index_source": str(result.snapshot.us_index_source),
        "run_id": result.run_id or result.snapshot.run_id,
        "snapshot_hash": result.snapshot_hash or result.snapshot.snapshot_hash,
        "config_hash": result.config_hash or result.snapshot.config_hash,
        "manifest_path": result.manifest_path or result.snapshot.manifest_path,
        "memory_path": result.memory_path,
        "memory_recall": asdict(result.memory_recall),
        "prediction_review": asdict(result.prediction_review),
        "strategy_mode": result.composite_state.strategy_mode,
        "risk_regime": result.composite_state.risk_regime,
        "market": asdict(result.composite_state.market),
        "market_info_state": asdict(result.composite_state.market_info_state),
        "capital_flow_state": asdict(result.composite_state.capital_flow_state),
        "macro_context_state": asdict(result.composite_state.macro_context_state),
        "mainlines": [asdict(item) for item in getattr(result.composite_state, "mainlines", [])],
        "viewpoints": [asdict(item) for item in getattr(result.composite_state, "viewpoints", [])],
        "theme_episodes": [asdict(item) for item in getattr(result.composite_state, "theme_episodes", [])],
        "stock_role_states": {
            str(key): asdict(value)
            for key, value in getattr(result.composite_state, "stock_role_states", {}).items()
        },
        "execution_plans": [asdict(item) for item in getattr(result.composite_state, "execution_plans", [])],
        "policy": policy_payload,
        "top_negative_info_events": [asdict(item) for item in result.top_negative_info_events],
        "top_positive_info_signals": [asdict(item) for item in result.top_positive_info_signals],
        "quant_info_divergence": [asdict(item) for item in result.quant_info_divergence],
        "trade_plan": [
            {
                "stock": _display_name(action.symbol),
                "action": action.action,
                "current_weight": action.current_weight,
                "target_weight": action.target_weight,
                "delta_weight": action.delta_weight,
                "note": action.note,
            }
            for action in result.trade_actions
        ],
    }


def summarize_v2_backtest(
    result: V2BacktestSummary,
    *,
    run_id: str | None = None,
    snapshot_hash: str | None = None,
    config_hash: str | None = None,
) -> dict[str, object]:
    payload = asdict(result)
    payload.pop("nav_curve", None)
    payload.pop("benchmark_nav_curve", None)
    payload.pop("excess_nav_curve", None)
    payload.pop("curve_dates", None)
    payload["run_id"] = str(run_id if run_id is not None else result.run_id)
    payload["snapshot_hash"] = str(snapshot_hash if snapshot_hash is not None else result.snapshot_hash)
    payload["config_hash"] = str(config_hash if config_hash is not None else result.config_hash)
    return payload


def summarize_v2_calibration(result: V2CalibrationResult) -> dict[str, object]:
    return {
        "best_score": float(result.best_score),
        "best_policy": asdict(result.best_policy),
        "baseline": summarize_v2_backtest(result.baseline),
        "calibrated": summarize_v2_backtest(result.calibrated),
        "trial_count": len(result.trials),
    }


def summarize_v2_policy_learning(result: V2PolicyLearningResult) -> dict[str, object]:
    return {
        "model": asdict(result.model),
        "baseline": summarize_v2_backtest(result.baseline),
        "learned": summarize_v2_backtest(result.learned),
    }
