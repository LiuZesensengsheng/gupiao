from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.application.v2_leader_runtime import top_leader_candidates
from src.application.v2_contracts import DailyRunResult, V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult
from src.contracts.reporting import DailyReportViewModel, ResearchReportViewModel
from src.review_analytics.summaries import (
    summarize_daily_run,
    summarize_v2_backtest,
    summarize_v2_calibration,
    summarize_v2_policy_learning,
)


def _load_json_payload(path_like: object) -> dict[str, object]:
    if not path_like:
        return {}
    path = Path(str(path_like))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json_items(path_like: object) -> list[dict[str, object]]:
    payload = _load_json_payload(path_like)
    items = payload.get("items", [])
    if not isinstance(items, list):
        return []
    return [dict(item) for item in items if isinstance(item, dict)]


def _next_business_day(date_text: str) -> str:
    ts = pd.Timestamp(date_text)
    if pd.isna(ts):
        return ""
    return str((ts + pd.offsets.BDay(1)).date())


def _flatten_horizon_forecasts(horizon_payload: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in ["1d", "2d", "3d", "5d", "10d", "20d"]:
        item = horizon_payload.get(key)
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "horizon": key,
                "label": str(item.get("label", key)),
                "up_prob": float(item.get("up_prob", float("nan"))),
                "q10": float(item.get("q10", float("nan"))),
                "q50": float(item.get("q50", float("nan"))),
                "q90": float(item.get("q90", float("nan"))),
                "price_low": float(item.get("price_low", float("nan"))),
                "price_high": float(item.get("price_high", float("nan"))),
                "confidence": float(item.get("confidence", float("nan"))),
            }
        )
    return rows


def build_daily_report_view_model(result: DailyRunResult) -> DailyReportViewModel:
    summary = summarize_daily_run(result)
    market_payload = dict(summary["market"])
    market_facts = dict(market_payload.get("market_facts", {}))
    sentiment = dict(market_payload.get("sentiment", {}))
    policy = dict(summary["policy"])
    selected_symbols = {
        str(symbol)
        for symbol, weight in result.policy_decision.symbol_target_weights.items()
        if float(weight) > 0.0
    }
    candidate_selection = asdict(result.composite_state.candidate_selection)
    name_map = dict(result.symbol_names)

    def _stock_name(symbol: str) -> str:
        return str(name_map.get(symbol, symbol))

    theme_episodes = [asdict(item) for item in getattr(result.composite_state, "theme_episodes", [])]
    leader_candidates = [
        {
            **asdict(item),
            "name": _stock_name(item.symbol),
        }
        for item in top_leader_candidates(state=result.composite_state, limit=10)
    ]
    role_states = {
        str(symbol): asdict(item)
        for symbol, item in getattr(result.composite_state, "stock_role_states", {}).items()
    }
    execution_plans = [asdict(item) for item in getattr(result.composite_state, "execution_plans", [])]

    candidate_order = {
        str(symbol): idx
        for idx, symbol in enumerate(result.composite_state.candidate_selection.shortlisted_symbols)
    }
    ranked_stocks = sorted(
        list(result.composite_state.stocks),
        key=lambda stock: candidate_order.get(stock.symbol, len(candidate_order) + 999),
    )
    top20 = ranked_stocks[:20]

    top_recommendations: list[dict[str, object]] = []
    explanation_cards: list[dict[str, object]] = []
    stock_reason_map: dict[str, str] = {}
    for idx, stock in enumerate(top20, start=1):
        forecasts = _flatten_horizon_forecasts(asdict(stock).get("horizon_forecasts", {}))
        forecast_by_horizon = {str(item["horizon"]): item for item in forecasts}
        one = forecast_by_horizon.get("1d", {})
        five = forecast_by_horizon.get("5d", {})
        ten = forecast_by_horizon.get("10d", {})
        twenty = forecast_by_horizon.get("20d", {})
        reason = (
            stock.action_reason
            if stock.symbol in selected_symbols and stock.action_reason
            else stock.blocked_reason or stock.weight_reason or "NA"
        )
        stock_reason_map[stock.symbol] = reason
        top_recommendations.append(
            {
                "rank": idx,
                "symbol": stock.symbol,
                "name": _stock_name(stock.symbol),
                "sector": stock.sector,
                "next_session_range": (
                    f"{one['price_low']:.2f} ~ {one['price_high']:.2f}"
                    if one and one["price_low"] == one["price_low"] and one["price_high"] == one["price_high"]
                    else "NA"
                ),
                "median_5d": float(five.get("q50", float("nan"))),
                "median_20d": float(twenty.get("q50", float("nan"))),
                "up_prob_1d": float(one.get("up_prob", float("nan"))),
                "confidence_1d": float(one.get("confidence", float("nan"))),
            }
        )
        explanation_cards.append(
            {
                "symbol": stock.symbol,
                "name": _stock_name(stock.symbol),
                "sector": stock.sector,
                "latest_close": float(stock.latest_close),
                "next_session_range": (
                    f"{one['price_low']:.2f} ~ {one['price_high']:.2f}"
                    if one and one["price_low"] == one["price_low"] and one["price_high"] == one["price_high"]
                    else "NA"
                ),
                "one_day_median": float(one.get("q50", float("nan"))),
                "one_day_up_prob": float(one.get("up_prob", float("nan"))),
                "one_day_confidence": float(one.get("confidence", float("nan"))),
                "median_5d": float(five.get("q50", float("nan"))),
                "median_10d": float(ten.get("q50", float("nan"))),
                "median_20d": float(twenty.get("q50", float("nan"))),
                "selection_reasons": list(stock.selection_reasons),
                "ranking_reasons": list(stock.ranking_reasons),
                "selected": stock.symbol in selected_symbols,
                "action_reason": stock.action_reason,
                "blocked_reason": stock.blocked_reason,
                "weight_reason": stock.weight_reason,
                "risk_flags": list(stock.risk_flags),
                "invalidation_rule": stock.invalidation_rule,
            }
        )

    trade_actions: list[dict[str, object]] = []
    for action in result.trade_actions:
        trade_actions.append(
            {
                "symbol": action.symbol,
                "name": _stock_name(action.symbol),
                "action": action.action,
                "current_weight": float(action.current_weight),
                "target_weight": float(action.target_weight),
                "delta_weight": float(action.delta_weight),
                "note": action.note,
                "reason": action.note or stock_reason_map.get(action.symbol, "NA"),
            }
        )

    review_payload = dict(summary["prediction_review"])
    review_windows = []
    for key in ["5d", "20d", "60d"]:
        item = review_payload.get("windows", {}).get(key)
        if not isinstance(item, dict):
            continue
        review_windows.append(
            {
                "horizon": key,
                "label": str(item.get("label", key)),
                "hit_rate": float(item.get("hit_rate", float("nan"))),
                "avg_edge": float(item.get("avg_edge", float("nan"))),
                "realized_return": float(item.get("realized_return", float("nan"))),
                "sample_size": int(item.get("sample_size", 0)),
                "note": str(item.get("note", "")),
            }
        )

    holding_role_changes: list[dict[str, object]] = []
    active_symbols = {
        str(action.symbol)
        for action in result.trade_actions
        if float(action.current_weight) > 0.0 or float(action.target_weight) > 0.0
    }
    for symbol in sorted(active_symbols):
        role_payload = role_states.get(symbol)
        if not role_payload:
            continue
        holding_role_changes.append(
            {
                "symbol": symbol,
                "name": _stock_name(symbol),
                "theme": str(role_payload.get("theme", "")),
                "role": str(role_payload.get("role", "")),
                "previous_role": str(role_payload.get("previous_role", "")),
                "role_downgrade": bool(role_payload.get("role_downgrade", False)),
                "note": str(role_payload.get("note", "")),
            }
        )

    return DailyReportViewModel(
        strategy_id=str(summary["strategy_id"]),
        run_id=str(summary["run_id"]),
        strategy_mode=str(summary["strategy_mode"]),
        risk_regime=str(summary["risk_regime"]),
        as_of_date=str(market_payload.get("as_of_date", "")),
        next_session=_next_business_day(str(market_payload.get("as_of_date", ""))),
        external_signal_enabled=bool(summary["external_signal_enabled"]),
        metadata={
            "universe_id": str(summary["universe_id"]),
            "universe_size": int(summary["universe_size"] or 0),
            "universe_generation_rule": str(summary["universe_generation_rule"] or ""),
            "source_universe_manifest_path": str(summary["source_universe_manifest_path"] or ""),
            "data_window": str(summary.get("data_window", "") or ""),
            "generator_manifest_path": str(summary["generator_manifest_path"] or ""),
            "generator_version": str(summary["generator_version"] or ""),
            "generator_hash": str(summary["generator_hash"] or ""),
            "info_manifest_path": str(summary["info_manifest_path"] or ""),
            "info_hash": str(summary["info_hash"] or ""),
            "info_shadow_enabled": bool(summary["info_shadow_enabled"]),
            "external_signal_manifest_path": str(summary["external_signal_manifest_path"] or ""),
            "external_signal_version": str(summary["external_signal_version"] or ""),
            "use_us_index_context": bool(summary["use_us_index_context"]),
            "us_index_source": str(summary["us_index_source"] or "NA"),
            "manifest_path": str(summary["manifest_path"] or ""),
            "snapshot_hash": str(summary["snapshot_hash"] or ""),
            "config_hash": str(summary["config_hash"] or ""),
        },
        market_summary={
            "sentiment": sentiment,
            "facts": market_facts,
            "risk_notes": list(result.policy_decision.risk_notes),
        },
        market_forecasts=_flatten_horizon_forecasts(dict(market_payload.get("horizon_forecasts", {}))),
        dynamic_universe={
            "coarse_pool_size": int(summary["coarse_pool_size"]),
            "refined_pool_size": int(summary["refined_pool_size"]),
            "selected_pool_size": int(summary["selected_pool_size"]),
            "generator_version": str(summary["generator_version"] or ""),
            "candidate_selection": candidate_selection,
            "theme_allocations": [dict(item) for item in summary["theme_allocations"]],
        },
        memory_summary={
            "memory_path": str(summary["memory_path"] or ""),
            "recall": dict(summary["memory_recall"]),
        },
        external_signals={
            "capital_flow": dict(summary["capital_flow_state"]),
            "macro_context": dict(summary["macro_context_state"]),
        },
        mainlines=[dict(item) for item in summary["mainlines"]],
        theme_episodes=theme_episodes,
        leader_candidates=leader_candidates,
        holding_role_changes=holding_role_changes,
        execution_plans=execution_plans,
        top_recommendations=top_recommendations,
        explanation_cards=explanation_cards,
        prediction_review={
            "windows": review_windows,
            "notes": list(review_payload.get("notes", [])),
        },
        trade_actions=trade_actions,
        policy=dict(summary["policy"]),
    )


def build_research_report_view_model(
    *,
    strategy_id: str,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    artifacts: dict[str, str] | None = None,
) -> ResearchReportViewModel:
    artifact_payload = dict(artifacts or {})
    release_gate_passed = str(artifact_payload.get("release_gate_passed", "false")).strip().lower() == "true"
    artifact_run_id = str(artifact_payload.get("run_id", "") or "")
    result_run_ids = {
        str(value)
        for value in [baseline.run_id, calibration.calibrated.run_id, learning.learned.run_id]
        if str(value or "").strip()
    }
    if artifact_run_id and result_run_ids and artifact_run_id not in result_run_ids:
        raise ValueError("artifact run_id mismatch")
    final_run_id = artifact_run_id or (next(iter(result_run_ids)) if result_run_ids else "")

    comparison_metrics = [
        {
            "label": "annual_return",
            "title": "年化收益",
            "baseline": float(baseline.annual_return),
            "calibrated": float(calibration.calibrated.annual_return),
            "learned": float(learning.learned.annual_return),
        },
        {
            "label": "excess_annual_return",
            "title": "超额年化",
            "baseline": float(baseline.excess_annual_return),
            "calibrated": float(calibration.calibrated.excess_annual_return),
            "learned": float(learning.learned.excess_annual_return),
        },
        {
            "label": "max_drawdown",
            "title": "最大回撤",
            "baseline": float(baseline.max_drawdown),
            "calibrated": float(calibration.calibrated.max_drawdown),
            "learned": float(learning.learned.max_drawdown),
        },
        {
            "label": "information_ratio",
            "title": "信息比率",
            "baseline": float(baseline.information_ratio),
            "calibrated": float(calibration.calibrated.information_ratio),
            "learned": float(learning.learned.information_ratio),
        },
    ]

    horizon_metrics: list[dict[str, object]] = []
    for stage_label, summary in [
        ("baseline", baseline),
        ("calibrated", calibration.calibrated),
        ("learned", learning.learned),
    ]:
        for horizon in ["1d", "2d", "3d", "5d", "20d"]:
            metrics = summary.horizon_metrics.get(horizon, {})
            if not metrics:
                continue
            horizon_metrics.append(
                {
                    "stage": stage_label,
                    "horizon": horizon,
                    "rank_ic": float(metrics.get("rank_ic", 0.0)),
                    "top_decile_return": float(metrics.get("top_decile_return", 0.0)),
                    "top_bottom_spread": float(metrics.get("top_bottom_spread", 0.0)),
                    "top_k_hit_rate": float(metrics.get("top_k_hit_rate", 0.0)),
                }
            )

    calibration_trials: list[dict[str, object]] = []
    sorted_trials = sorted(
        calibration.trials,
        key=lambda item: float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
        reverse=True,
    )
    for index, trial in enumerate(sorted_trials, start=1):
        if not isinstance(trial, dict):
            continue
        calibration_trials.append(
            {
                "rank": index,
                "score": float(trial.get("score", 0.0)),
                "policy": dict(trial.get("policy", {})),
                "summary": dict(trial.get("summary", {})),
            }
        )

    info_manifest = _load_json_payload(artifact_payload.get("info_manifest"))
    info_shadow = _load_json_payload(artifact_payload.get("info_shadow_report"))
    insight_manifest = _load_json_payload(artifact_payload.get("insight_manifest"))
    leader_manifest = _load_json_payload(artifact_payload.get("leader_manifest"))
    leader_candidates = _load_json_items(artifact_payload.get("leader_candidates"))
    theme_episodes = _load_json_items(artifact_payload.get("theme_episodes"))
    stock_roles = _load_json_items(artifact_payload.get("stock_roles"))
    phase_counts = dict(insight_manifest.get("phase_counts", {}))
    role_counts = dict(insight_manifest.get("role_counts", {}))
    fading_themes = [
        item for item in theme_episodes
        if str(item.get("phase", "")) == "fading"
    ]
    crowded_themes = [
        item for item in theme_episodes
        if str(item.get("phase", "")) == "crowded"
    ]
    role_downgrades = [
        item for item in stock_roles
        if bool(item.get("role_downgrade", False))
    ]
    return ResearchReportViewModel(
        strategy_id=str(strategy_id),
        run_id=final_run_id,
        release_gate_passed=release_gate_passed,
        artifacts={key: str(value) for key, value in artifact_payload.items()},
        baseline=summarize_v2_backtest(baseline, run_id=final_run_id or None),
        calibration=summarize_v2_calibration(calibration),
        learned=summarize_v2_policy_learning(learning),
        comparison_metrics=comparison_metrics,
        horizon_metrics=horizon_metrics,
        calibration_trials=calibration_trials,
        learning_model=asdict(learning.model),
        info_shadow_summary={
            "manifest": info_manifest,
            "shadow_report": info_shadow,
        },
        theme_lifecycle_summary={
            "manifest": insight_manifest,
            "phase_counts": phase_counts,
            "top_themes": theme_episodes[:6],
        },
        role_distribution_summary={
            "role_counts": role_counts,
            "top_roles": stock_roles[:12],
        },
        exit_contribution_summary={
            "fading_theme_count": len(fading_themes),
            "crowded_theme_count": len(crowded_themes),
            "role_downgrade_count": len(role_downgrades),
            "examples": role_downgrades[:8],
        },
        leader_summary={
            "manifest": leader_manifest,
            "evaluation": dict(leader_manifest.get("evaluation", {})),
            "top_candidates": leader_candidates[:12],
        },
    )
