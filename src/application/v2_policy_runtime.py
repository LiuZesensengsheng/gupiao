from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from src.application.v2_contracts import (
    CapitalFlowState,
    CompositeState,
    InfoAggregateState,
    LearnedPolicyModel,
    MacroContextState,
    MainlineState,
    PolicyDecision,
    PolicyInput,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
)
from src.application.v2_leader_runtime import build_leader_score_snapshots
from src.application.v2_signal_training_runtime import (
    build_exit_behavior_runtime_rows,
    rank_exit_candidates,
)
from src.domain.entities import TradeAction


@dataclass(frozen=True)
class PolicyRuntimeDependencies:
    clip: Callable[[float, float, float], float]
    alpha_score_components: Callable[[StockForecastState], dict[str, float]]
    candidate_stocks_from_state: Callable[[CompositeState], list[StockForecastState]]
    candidate_risk_snapshot: Callable[[Iterable[StockForecastState]], dict[str, float]]
    dominant_mainline_sectors: Callable[[Iterable[MainlineState]], Iterable[str]]
    ranked_sector_budgets_with_alpha_external: Callable[..., dict[str, float]]
    cap_sector_budgets_external: Callable[..., tuple[dict[str, float], list[str]]]
    allocate_sector_slots_external: Callable[..., dict[str, int]]
    allocate_with_sector_budgets_external: Callable[..., dict[str, float]]
    is_actionable_status: Callable[[str], bool]
    policy_feature_vector: Callable[[CompositeState], np.ndarray]
    normalize_coef_vector: Callable[[object, int], np.ndarray]
    predict_ridge: Callable[[np.ndarray, float, np.ndarray], float]


def ranked_sector_budgets(
    sectors: Iterable[SectorForecastState],
    *,
    target_exposure: float,
) -> dict[str, float]:
    rows = list(sectors)
    if not rows:
        return {}
    raw = [max(0.0, float(item.up_20d_prob) - 0.50) + max(0.0, float(item.relative_strength)) for item in rows]
    total = sum(raw)
    if total <= 1e-9:
        equal = float(target_exposure) / float(len(rows))
        return {item.sector: equal for item in rows}
    return {item.sector: float(target_exposure) * score / total for item, score in zip(rows, raw)}


def stock_policy_score(
    stock: StockForecastState,
    *,
    deps: PolicyRuntimeDependencies,
) -> float:
    return float(deps.alpha_score_components(stock)["alpha_score"])


def stock_signal_profile(
    stock: StockForecastState,
    *,
    deps: PolicyRuntimeDependencies,
) -> dict[str, float | bool]:
    components = deps.alpha_score_components(stock)
    alpha_score = float(components["alpha_score"])
    entry_score = float(
        deps.clip(
            alpha_score
            + 0.42 * float(components["swing_edge"])
            + 0.28 * float(components["medium_edge"])
            + 0.16 * float(components["sector_edge"])
            + 0.14 * float(components["trend_alignment"])
            - 0.26 * float(components["risk_penalty"])
            - 0.10 * float(components["status_penalty"]),
            0.0,
            1.0,
        )
    )
    hold_score = float(
        deps.clip(
            alpha_score
            + 0.46 * float(components["continuation_bonus"])
            + 0.18 * float(components["stability_bonus"])
            + 0.12 * float(components["quality_bonus"])
            + 0.10 * float(components["trend_alignment"])
            - 0.22 * float(components["swing_fade_penalty"])
            - 0.14 * float(components["reversal_penalty"]),
            0.0,
            1.0,
        )
    )
    exit_risk = float(
        deps.clip(
            0.72 * float(components["risk_penalty"])
            + 0.48 * float(components["swing_fade_penalty"])
            + 0.28 * float(components["reversal_penalty"])
            + 0.24 * float(components["weak_mid_penalty"])
            + 0.18 * float(components["status_penalty"])
            - 0.28 * float(components["continuation_bonus"])
            - 0.16 * float(components["quality_bonus"]),
            0.0,
            1.0,
        )
    )
    conviction_spread = float(max(0.0, entry_score - max(0.52, exit_risk + 0.44)))
    keep_strength = float(
        deps.clip(
            0.55 * hold_score
            + 0.20 * entry_score
            + 0.15 * float(components["quality_bonus"])
            + 0.10 * float(components["stability_bonus"])
            - 0.28 * exit_risk,
            0.0,
            1.0,
        )
    )
    return {
        "alpha_score": alpha_score,
        "entry_score": entry_score,
        "hold_score": hold_score,
        "exit_risk": exit_risk,
        "conviction_spread": conviction_spread,
        "keep_strength": keep_strength,
        "strong_entry": bool(entry_score >= 0.66 and exit_risk <= 0.12),
        "strong_hold": bool(hold_score >= 0.65 and exit_risk <= 0.10),
        "weakening": bool(hold_score <= 0.57 or exit_risk >= 0.11),
        "urgent_exit": bool(exit_risk >= 0.15 and hold_score < 0.60),
    }


def alpha_opportunity_metrics(
    stocks: Iterable[StockForecastState],
    *,
    deps: PolicyRuntimeDependencies,
) -> dict[str, float]:
    actionable = [
        stock for stock in stocks
        if deps.is_actionable_status(getattr(stock, "tradability_status", "normal"))
    ]
    if not actionable:
        return {
            "top_score": 0.0,
            "avg_top3": 0.0,
            "median_score": 0.0,
            "breadth_ratio": 0.0,
            "strong_count": 0.0,
            "alpha_headroom": 0.0,
            "entry_top_score": 0.0,
            "entry_avg_top3": 0.0,
            "entry_median_score": 0.0,
            "entry_headroom": 0.0,
            "entry_separation": 0.0,
            "hold_strength": 0.0,
            "exit_pressure": 0.0,
        }
    scores = sorted((stock_policy_score(stock, deps=deps) for stock in actionable), reverse=True)
    signal_profiles = [stock_signal_profile(stock, deps=deps) for stock in actionable]
    entry_scores = sorted((float(profile["entry_score"]) for profile in signal_profiles), reverse=True)
    hold_scores = sorted((float(profile["hold_score"]) for profile in signal_profiles), reverse=True)
    exit_risks = sorted((float(profile["exit_risk"]) for profile in signal_profiles))
    top_slice = scores[: min(3, len(scores))]
    top_score = float(scores[0])
    avg_top3 = float(sum(top_slice) / max(1, len(top_slice)))
    median_score = float(np.median(scores))
    strong_cut = max(0.56, median_score + 0.03)
    strong_count = int(sum(1 for score in scores if score >= strong_cut))
    breadth_ratio = float(strong_count / max(1, len(scores)))
    alpha_headroom = float(max(0.0, avg_top3 - max(0.54, median_score)))
    entry_top = float(entry_scores[0])
    entry_top3 = entry_scores[: min(3, len(entry_scores))]
    entry_avg_top3 = float(sum(entry_top3) / max(1, len(entry_top3)))
    entry_median = float(np.median(entry_scores))
    entry_headroom = float(max(0.0, entry_avg_top3 - max(0.55, entry_median)))
    entry_separation = float(
        deps.clip(
            1.15 * max(0.0, entry_top - entry_median)
            + 0.95 * max(0.0, entry_avg_top3 - entry_median),
            0.0,
            0.30,
        )
    )
    hold_strength = float(sum(hold_scores[: min(3, len(hold_scores))]) / max(1, min(3, len(hold_scores))))
    exit_pressure = float(np.mean(exit_risks[-min(5, len(exit_risks)) :])) if exit_risks else 0.0
    return {
        "top_score": top_score,
        "avg_top3": avg_top3,
        "median_score": median_score,
        "breadth_ratio": breadth_ratio,
        "strong_count": float(strong_count),
        "alpha_headroom": alpha_headroom,
        "entry_top_score": entry_top,
        "entry_avg_top3": entry_avg_top3,
        "entry_median_score": entry_median,
        "entry_headroom": entry_headroom,
        "entry_separation": entry_separation,
        "hold_strength": hold_strength,
        "exit_pressure": exit_pressure,
    }


def holding_alpha_profile(
    stock: StockForecastState,
    *,
    deps: PolicyRuntimeDependencies,
) -> dict[str, float | bool]:
    signal_profile = stock_signal_profile(stock, deps=deps)
    near_term_stack = float(
        0.22 * float(stock.up_1d_prob)
        + 0.24 * float(getattr(stock, "up_2d_prob", 0.5))
        + 0.26 * float(getattr(stock, "up_3d_prob", 0.5))
        + 0.28 * float(stock.up_5d_prob)
    )
    continuation_stack = float(
        0.34 * float(getattr(stock, "up_3d_prob", 0.5))
        + 0.36 * float(stock.up_5d_prob)
        + 0.12 * float(getattr(stock, "up_10d_prob", 0.5))
        + 0.18 * float(stock.up_20d_prob)
    )
    relative_edge = float(stock.excess_vs_sector_prob)
    return {
        "alpha_score": float(signal_profile["alpha_score"]),
        "near_term_stack": near_term_stack,
        "continuation_stack": continuation_stack,
        "relative_edge": relative_edge,
        "persistence_score": float(signal_profile["keep_strength"]),
        "breakdown_risk": float(signal_profile["exit_risk"]),
        "strong_persistence": bool(signal_profile["strong_hold"]),
        "weakening": bool(signal_profile["weakening"]),
        "urgent_exit": bool(signal_profile["urgent_exit"]),
    }


def _horizon_forecast_metric(
    stock: StockForecastState,
    *,
    horizon: str,
    field: str,
) -> tuple[float, bool]:
    forecasts = getattr(stock, "horizon_forecasts", {}) or {}
    payload = forecasts.get(str(horizon))
    if payload is None and str(horizon).endswith("d"):
        try:
            target_days = int(str(horizon)[:-1])
        except ValueError:
            target_days = -1
        if target_days > 0:
            for candidate in forecasts.values():
                candidate_days = (
                    candidate.get("horizon_days", -1)
                    if isinstance(candidate, dict)
                    else getattr(candidate, "horizon_days", -1)
                )
                if int(candidate_days) == target_days:
                    payload = candidate
                    break
    if payload is None:
        return 0.0, False
    value = payload.get(field, 0.0) if isinstance(payload, dict) else getattr(payload, field, 0.0)
    try:
        return float(value), True
    except (TypeError, ValueError):
        return 0.0, False


def stock_actionability_profile(
    stock: StockForecastState,
    *,
    composite_state: CompositeState,
    deps: PolicyRuntimeDependencies,
) -> dict[str, float | str | bool]:
    risk_regime = str(getattr(composite_state, "risk_regime", "") or "").strip().lower()
    signal_profile = stock_signal_profile(stock, deps=deps)
    confidence_1d, has_confidence_1d = _horizon_forecast_metric(stock, horizon="1d", field="confidence")
    expected_5d, has_expected_5d = _horizon_forecast_metric(stock, horizon="5d", field="expected_return")
    expected_20d, has_expected_20d = _horizon_forecast_metric(stock, horizon="20d", field="expected_return")
    q10_5d, has_q10_5d = _horizon_forecast_metric(stock, horizon="5d", field="q10")
    confidence_1d = confidence_1d if has_confidence_1d else 0.45
    expected_5d = expected_5d if has_expected_5d else 0.0
    expected_20d = expected_20d if has_expected_20d else 0.0
    q10_5d = q10_5d if has_q10_5d else 0.0
    downside_penalty = float(deps.clip(abs(min(0.0, q10_5d)) / 0.12, 0.0, 1.0))
    expected_5d_component = float(deps.clip(0.5 + expected_5d / 0.06, 0.0, 1.0))
    expected_20d_component = float(deps.clip(0.5 + expected_20d / 0.10, 0.0, 1.0))
    score = float(
        deps.clip(
            0.30 * float(signal_profile["entry_score"])
            + 0.12 * float(signal_profile["hold_score"])
            + 0.10 * float(stock.up_1d_prob)
            + 0.14 * float(stock.up_5d_prob)
            + 0.12 * float(stock.up_20d_prob)
            + 0.08 * float(stock.excess_vs_sector_prob)
            + 0.05 * float(confidence_1d)
            + 0.06 * expected_5d_component
            + 0.03 * expected_20d_component
            - 0.10 * float(signal_profile["exit_risk"])
            - 0.05 * downside_penalty,
            0.0,
            1.0,
        )
    )
    threshold = 0.51
    if risk_regime == "cautious":
        threshold = 0.54
    elif risk_regime == "risk_off":
        threshold = 0.57
    label = "blocked"
    if score >= threshold + 0.04:
        label = "high_conviction"
    elif score >= threshold:
        label = "actionable"
    elif score >= threshold - 0.03:
        label = "monitor"
    return {
        "score": score,
        "threshold": float(threshold),
        "label": label,
        "entry_score": float(signal_profile["entry_score"]),
        "hold_score": float(signal_profile["hold_score"]),
        "exit_risk": float(signal_profile["exit_risk"]),
        "confidence_1d": float(confidence_1d),
        "expected_5d": float(expected_5d),
        "expected_20d": float(expected_20d),
        "q10_5d": float(q10_5d),
        "has_expected_5d": bool(has_expected_5d),
        "has_expected_20d": bool(has_expected_20d),
    }


def evaluate_fresh_buy_candidate(
    stock: StockForecastState,
    *,
    composite_state: CompositeState,
    deps: PolicyRuntimeDependencies,
) -> tuple[bool, list[str]]:
    risk_regime = str(getattr(composite_state, "risk_regime", "") or "").strip().lower()
    actionability = stock_actionability_profile(
        stock,
        composite_state=composite_state,
        deps=deps,
    )
    score = float(actionability["score"])
    threshold = float(actionability["threshold"])
    reasons: list[str] = []
    expected_5d = float(actionability["expected_5d"])
    expected_20d = float(actionability["expected_20d"])
    has_expected_5d = bool(actionability.get("has_expected_5d", False))
    has_expected_20d = bool(actionability.get("has_expected_20d", False))
    exit_risk = float(actionability["exit_risk"])
    if score < threshold:
        reasons.append(f"actionability<{threshold:.2f}")
    if risk_regime == "risk_off":
        if float(stock.up_1d_prob) < 0.51:
            reasons.append("1d_prob<0.510")
        if float(stock.up_5d_prob) < 0.535:
            reasons.append("5d_prob<0.535")
        if float(stock.up_20d_prob) < 0.50:
            reasons.append("20d_prob<0.50")
        if has_expected_5d and expected_5d < 0.004:
            reasons.append("5d_exp_buffer<0.004")
        if has_expected_20d and expected_20d < 0.008:
            reasons.append("20d_exp_buffer<0.008")
        if has_expected_5d and expected_5d <= 0.0:
            reasons.append("5d_exp<=0")
        if has_expected_20d and expected_20d <= 0.0:
            reasons.append("20d_exp<=0")
        if exit_risk >= 0.12:
            reasons.append("exit_risk_high")
    elif risk_regime == "cautious":
        if has_expected_5d and expected_5d <= 0.0:
            reasons.append("5d_exp<=0")
        if has_expected_5d and has_expected_20d and expected_5d < 0.002 and expected_20d < 0.004:
            reasons.append("forward_edge_too_thin")
        if has_expected_20d and float(stock.up_20d_prob) < 0.49 and expected_20d <= 0.0:
            reasons.append("20d_trend_soft")
    elif (
        (has_expected_5d or has_expected_20d)
        and (not has_expected_5d or expected_5d <= 0.0)
        and (not has_expected_20d or expected_20d <= 0.0)
        and score < threshold + 0.02
    ):
        reasons.append("forward_edge<=0")

    return len(reasons) == 0, reasons


def _stock_info_support_score(
    *,
    symbol: str,
    composite_state: CompositeState,
    deps: PolicyRuntimeDependencies,
) -> float:
    stock_info_states = getattr(composite_state, "stock_info_states", {}) or {}
    info_state = stock_info_states.get(str(symbol), InfoAggregateState())
    return float(
        deps.clip(
            0.42 * float(getattr(info_state, "catalyst_strength", 0.0))
            + 0.18 * float(getattr(info_state, "coverage_confidence", 0.0))
            + 0.16 * max(0.0, float(getattr(info_state, "info_prob_5d", 0.5)) - 0.50) / 0.15
            + 0.12 * max(0.0, float(getattr(info_state, "info_prob_20d", 0.5)) - 0.50) / 0.12
            + 0.12 * max(0.0, float(getattr(info_state, "shadow_prob_20d", 0.5)) - 0.50) / 0.12,
            0.0,
            1.0,
        )
    )


def select_risk_off_pilot_candidate(
    *,
    blocked_candidates: dict[str, list[str]],
    raw_candidate_stocks: list[StockForecastState],
    actionability_profiles: dict[str, dict[str, float | str | bool]],
    leader_snapshots: list[LeaderScoreSnapshot],
    composite_state: CompositeState,
    deps: PolicyRuntimeDependencies,
) -> tuple[StockForecastState | None, dict[str, object] | None]:
    if str(getattr(composite_state, "risk_regime", "") or "").strip().lower() != "risk_off":
        return None, None
    if not blocked_candidates or not raw_candidate_stocks:
        return None, None

    leader_snapshot_map = {
        str(item.symbol).strip(): item
        for item in leader_snapshots
        if str(getattr(item, "symbol", "")).strip()
    }
    market_info = getattr(composite_state, "market_info_state", InfoAggregateState())
    structural_blockers = {"1d_prob<0.510", "5d_prob<0.535", "5d_exp<=0", "20d_exp<=0", "exit_risk_high"}
    thin_edge_blockers = {"20d_prob<0.50", "20d_exp_buffer<0.008", "5d_exp_buffer<0.004"}

    best_stock: StockForecastState | None = None
    best_payload: dict[str, object] | None = None
    best_score = -1.0

    for stock in raw_candidate_stocks:
        symbol = str(getattr(stock, "symbol", "")).strip()
        if not symbol or symbol not in blocked_candidates:
            continue
        reasons = list(blocked_candidates.get(symbol, []))
        if not reasons:
            continue
        if any(reason in structural_blockers for reason in reasons):
            continue

        non_actionability_reasons = {
            reason for reason in reasons
            if not str(reason).startswith("actionability<")
        }
        if non_actionability_reasons - thin_edge_blockers:
            continue

        snapshot = leader_snapshot_map.get(symbol)
        if snapshot is None or bool(snapshot.hard_negative):
            continue
        role = str(snapshot.role or "").strip().lower()
        if role not in {"leader", "core"}:
            continue
        if bool(snapshot.role_downgrade):
            continue
        max_negative = 0.18 if role == "leader" else 0.17
        if float(snapshot.negative_score) > max_negative:
            continue

        actionability = actionability_profiles.get(symbol, {})
        score = float(actionability.get("score", 0.0))
        threshold = float(actionability.get("threshold", 0.57))
        min_actionability = 0.62 if role == "leader" else 0.64
        if score < max(min_actionability, threshold - 0.01):
            continue

        if float(stock.up_1d_prob) < 0.53 or float(stock.up_5d_prob) < 0.57 or float(stock.excess_vs_sector_prob) < 0.56:
            continue

        info_support = _stock_info_support_score(
            symbol=symbol,
            composite_state=composite_state,
            deps=deps,
        )
        if info_support <= 1e-6:
            info_support = float(
                deps.clip(
                    0.55 * float(getattr(stock, "event_impact_score", 0.0))
                    + 0.25 * float(getattr(stock, "tradeability_score", 0.0))
                    + 0.20 * float(getattr(stock, "alpha_score", 0.0)),
                    0.0,
                    1.0,
                )
            )
        market_catalyst = float(getattr(market_info, "catalyst_strength", 0.0))
        market_negative = float(getattr(market_info, "negative_event_risk", 0.0))
        phase = str(snapshot.theme_phase or "").strip().lower()
        required_info_support = 0.54 if phase in {"", "emerging", "strengthening"} else 0.62
        if info_support < required_info_support:
            continue
        if market_negative >= 0.35 and market_catalyst < 0.60:
            continue

        pilot_score = float(
            deps.clip(
                0.36 * score
                + 0.22 * info_support
                + 0.12 * float(stock.excess_vs_sector_prob)
                + 0.10 * float(stock.up_5d_prob)
                + 0.08 * max(0.0, 0.20 - float(snapshot.negative_score)) / 0.20
                + 0.06 * float(snapshot.candidate_score)
                + 0.04 * float(snapshot.conviction_score)
                + (0.05 if role == "leader" else 0.03),
                0.0,
                1.0,
            )
        )
        if pilot_score < 0.63:
            continue

        if pilot_score > best_score:
            best_score = pilot_score
            best_stock = stock
            best_payload = {
                "symbol": symbol,
                "score": pilot_score,
                "info_support": info_support,
                "leader_role": str(snapshot.role),
                "theme_phase": str(snapshot.theme_phase),
                "blocked_reasons": reasons,
            }

    return best_stock, best_payload


def mainline_preference_maps(
    mainlines: Iterable[MainlineState],
    *,
    risk_cutoff: float,
    deps: PolicyRuntimeDependencies,
) -> tuple[dict[str, float], dict[str, float], list[MainlineState]]:
    confirmed: list[MainlineState] = []
    sector_boosts: dict[str, float] = {}
    symbol_boosts: dict[str, float] = {}
    rows = list(mainlines or [])[:3]
    top_conviction = max((float(getattr(item, "conviction", 0.0)) for item in rows), default=0.0)
    cutoff = max(0.30, top_conviction - 0.08)
    for rank, mainline in enumerate(rows):
        conviction = float(getattr(mainline, "conviction", 0.0))
        event_risk = float(getattr(mainline, "event_risk_level", 0.0))
        if conviction < cutoff or event_risk >= float(risk_cutoff):
            continue
        leadership = float(getattr(mainline, "leadership", 0.0))
        catalyst = float(getattr(mainline, "catalyst_strength", 0.0))
        boost = float(
            deps.clip(
                0.05
                + 0.28 * max(0.0, conviction - cutoff)
                + 0.06 * leadership
                + 0.05 * catalyst
                - 0.02 * rank,
                0.03,
                0.16,
            )
        )
        confirmed.append(mainline)
        for sector in getattr(mainline, "sectors", []):
            sector_key = str(sector)
            sector_boosts[sector_key] = max(sector_boosts.get(sector_key, 0.0), boost)
        for symbol in getattr(mainline, "representative_symbols", []):
            symbol_key = str(symbol)
            symbol_boosts[symbol_key] = max(symbol_boosts.get(symbol_key, 0.0), boost + 0.03)
    return sector_boosts, symbol_boosts, confirmed


def merge_symbol_score_adjustments(
    *maps: dict[str, float] | None,
    deps: PolicyRuntimeDependencies,
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for mapping in maps:
        if not mapping:
            continue
        for symbol, value in mapping.items():
            symbol_key = str(symbol).strip()
            if not symbol_key:
                continue
            merged[symbol_key] = float(
                deps.clip(
                    float(merged.get(symbol_key, 0.0)) + float(value),
                    -0.24,
                    0.20,
                )
            )
    return merged


def leader_symbol_preference_maps(
    *,
    state: CompositeState,
    candidate_stocks: Iterable[StockForecastState],
    snapshots: Iterable[LeaderScoreSnapshot] | None = None,
    deps: PolicyRuntimeDependencies,
) -> tuple[dict[str, float], list[str]]:
    candidate_symbols = {
        str(stock.symbol).strip()
        for stock in candidate_stocks
        if str(getattr(stock, "symbol", "")).strip()
    }
    source_snapshots = list(snapshots) if snapshots is not None else build_leader_score_snapshots(state=state)
    snapshots = [
        item
        for item in source_snapshots
        if not candidate_symbols or str(item.symbol).strip() in candidate_symbols
    ]
    if not snapshots:
        return {}, []

    adjustments: dict[str, float] = {}
    promoted: list[str] = []
    suppressed: list[str] = []
    hard_negative_count = 0
    for item in snapshots:
        role = str(item.role or "").strip().lower()
        phase = str(item.theme_phase or "").strip().lower()
        adjustment = 0.0
        adjustment += {
            "leader": 0.06,
            "core": 0.03,
            "follower": 0.00,
            "rebound": -0.01,
            "laggard": -0.08,
        }.get(role, 0.0)
        phase_adjustment = {
            "strengthening": 0.04,
            "emerging": 0.02,
            "crowded": -0.02,
            "diverging": -0.05,
            "fading": -0.10,
        }.get(phase, 0.0)
        if phase_adjustment >= 0.0:
            phase_multiplier = {
                "leader": 1.0,
                "core": 0.75,
                "follower": 0.0,
                "rebound": 0.25,
                "laggard": 0.0,
            }.get(role, 0.5)
        else:
            phase_multiplier = {
                "leader": 0.80,
                "core": 1.00,
                "follower": 1.00,
                "rebound": 1.00,
                "laggard": 1.10,
            }.get(role, 1.0)
        adjustment += phase_adjustment * phase_multiplier
        adjustment += 0.08 * max(0.0, float(item.conviction_score) - 0.62) / 0.38
        adjustment += 0.05 * max(0.0, float(item.candidate_score) - 0.60) / 0.40
        adjustment -= 0.12 * max(0.0, float(item.negative_score) - 0.46) / 0.54
        if bool(item.role_downgrade):
            adjustment -= 0.05
        if bool(item.hard_negative):
            hard_negative_count += 1
            adjustment = min(
                adjustment,
                -0.18 - 0.10 * max(0.0, float(item.negative_score) - 0.58) / 0.42,
            )
        adjustment = float(deps.clip(float(adjustment), -0.26, 0.14))
        if abs(adjustment) < 0.015:
            continue
        adjustments[str(item.symbol)] = adjustment
        if adjustment >= 0.06:
            promoted.append(str(item.symbol))
        elif adjustment <= -0.10:
            suppressed.append(str(item.symbol))

    if not adjustments:
        return {}, []

    notes: list[str] = []
    notes.append(
        "Leader weighting active: "
        f"{len(promoted)} promoted, {len(suppressed)} suppressed, {hard_negative_count} hard negatives."
    )
    if promoted:
        notes.append("Leader-promoted symbols: " + ", ".join(promoted[:3]))
    if suppressed:
        notes.append("Leader-suppressed symbols: " + ", ".join(suppressed[:3]))
    return adjustments, notes


def ranked_sector_budgets_with_alpha(
    *,
    sectors: Iterable[SectorForecastState],
    stocks: Iterable[StockForecastState],
    target_exposure: float,
    sector_score_adjustments: dict[str, float] | None = None,
    deps: PolicyRuntimeDependencies,
) -> dict[str, float]:
    return deps.ranked_sector_budgets_with_alpha_external(
        sectors=sectors,
        stocks=stocks,
        target_exposure=target_exposure,
        stock_score_fn=lambda stock: stock_policy_score(stock, deps=deps),
        sector_score_adjustments=sector_score_adjustments,
    )


def cap_sector_budgets(
    *,
    sector_budgets: dict[str, float],
    target_exposure: float,
    risk_regime: str,
    breadth_strength: float,
    deps: PolicyRuntimeDependencies,
) -> tuple[dict[str, float], list[str]]:
    return deps.cap_sector_budgets_external(
        sector_budgets=sector_budgets,
        target_exposure=target_exposure,
        risk_regime=risk_regime,
        breadth_strength=breadth_strength,
    )


def allocate_sector_slots(
    *,
    sector_budgets: dict[str, float],
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]],
    total_slots: int,
    sector_strengths: dict[str, float] | None = None,
    deps: PolicyRuntimeDependencies,
) -> dict[str, int]:
    return deps.allocate_sector_slots_external(
        sector_budgets=sector_budgets,
        available_by_sector=available_by_sector,
        total_slots=total_slots,
        sector_strengths=sector_strengths,
    )


def allocate_with_sector_budgets(
    *,
    stocks: list[StockForecastState],
    sector_budgets: dict[str, float],
    target_position_count: int,
    sector_strengths: dict[str, float] | None = None,
    max_single_position: float = 0.35,
    symbol_score_adjustments: dict[str, float] | None = None,
    deps: PolicyRuntimeDependencies,
) -> dict[str, float]:
    return deps.allocate_with_sector_budgets_external(
        stocks=stocks,
        sector_budgets=sector_budgets,
        target_position_count=target_position_count,
        stock_score_fn=lambda stock: stock_policy_score(stock, deps=deps),
        sector_strengths=sector_strengths,
        max_single_position=max_single_position,
        symbol_score_adjustments=symbol_score_adjustments,
    )


def _role_rank(role: str) -> int:
    return {
        "leader": 0,
        "core": 1,
        "follower": 2,
        "rebound": 3,
        "laggard": 4,
    }.get(str(role or "").strip().lower(), 99)


def _theme_episode_for_symbol(
    *,
    symbol: str,
    role_states: dict[str, object],
    theme_episodes: dict[str, object],
) -> object | None:
    role_state = role_states.get(str(symbol))
    if role_state is None:
        return None
    return theme_episodes.get(str(getattr(role_state, "theme", "")))


def _build_hold_buffer_rank_context(
    *,
    desired_weights: dict[str, float],
    current_weights: dict[str, float],
    state_map: dict[str, StockForecastState],
    holding_profiles: dict[str, dict[str, float | bool]],
    role_states: dict[str, object],
    theme_episodes: dict[str, object],
    deps: PolicyRuntimeDependencies,
) -> tuple[list[dict[str, object]], dict[str, int], int, int, int]:
    desired_active = [symbol for symbol, weight in desired_weights.items() if float(weight) > 1e-9]
    current_active = [symbol for symbol, weight in current_weights.items() if float(weight) > 1e-9]
    target_slot_count = max(1, len(desired_active) or len(current_active))
    entries: list[dict[str, object]] = []

    for symbol, stock_state in state_map.items():
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        desired = max(0.0, float(desired_weights.get(symbol, 0.0)))
        holding_profile = holding_profiles.get(symbol, {})
        role_state = role_states.get(symbol)
        theme_episode = _theme_episode_for_symbol(
            symbol=symbol,
            role_states=role_states,
            theme_episodes=theme_episodes,
        )
        role = str(getattr(role_state, "role", "")).strip().lower()
        status = str(getattr(stock_state, "tradability_status", "normal") or "normal")
        theme_phase = str(getattr(theme_episode, "phase", "")).strip().lower()
        theme_fading = theme_phase == "fading"
        role_downgrade = bool(getattr(role_state, "role_downgrade", False))
        laggard_role = role == "laggard"
        strong_theme_holder = bool(theme_phase == "strengthening" and role in {"leader", "core"})
        actionable = bool(deps.is_actionable_status(status))
        persistence_score = float(holding_profile.get("persistence_score", 0.0))
        exit_pressure = float(holding_profile.get("breakdown_risk", holding_profile.get("exit_risk", 0.0)))
        keep_score = float(
            np.clip(
                0.48 * stock_policy_score(stock_state, deps=deps)
                + 0.26 * persistence_score
                + 0.08 * float(getattr(stock_state, "excess_vs_sector_prob", 0.5))
                + 0.05 * (1.0 if desired > 1e-9 else 0.0)
                + 0.06 * (1.0 if current > 1e-9 else 0.0)
                + 0.04 * (1.0 if strong_theme_holder else 0.0)
                - 0.12 * (1.0 if theme_fading else 0.0)
                - 0.10 * (1.0 if role_downgrade else 0.0)
                - 0.10 * (1.0 if laggard_role else 0.0)
                - 0.14 * exit_pressure,
                0.0,
                1.0,
            )
        )
        buffer_hold_eligible = bool(
            actionable
            and current > 1e-9
            and not bool(holding_profile.get("urgent_exit", False))
            and not theme_fading
            and not role_downgrade
            and not laggard_role
            and keep_score >= 0.52
        )
        entries.append(
            {
                "symbol": symbol,
                "sector": str(getattr(stock_state, "sector", "") or "其他"),
                "score": keep_score,
                "current": current,
                "desired": desired,
                "role": role,
                "buffer_hold_eligible": buffer_hold_eligible,
            }
        )

    entries.sort(
        key=lambda item: (
            -float(item["score"]),
            -float(item["desired"]),
            -float(item["current"]),
            _role_rank(str(item["role"])),
            str(item["symbol"]),
        )
    )
    rank_map = {str(item["symbol"]): idx for idx, item in enumerate(entries, start=1)}
    for item in entries:
        item["rank"] = int(rank_map[str(item["symbol"])])

    hold_until_rank = min(len(entries), max(target_slot_count + 2, target_slot_count * 2))
    available_sectors = {str(item["sector"]) for item in entries if float(item["score"]) > 0.0}
    sector_name_cap = 0
    if len(available_sectors) > 1:
        sector_name_cap = 2 if target_slot_count >= 3 else 1
    return entries, rank_map, hold_until_rank, sector_name_cap, target_slot_count


def _apply_hold_buffer_retention(
    *,
    adjusted: dict[str, float],
    current_weights: dict[str, float],
    rank_entries: list[dict[str, object]],
    hold_until_rank: int,
    sector_name_cap: int,
    target_slot_count: int,
    target_exposure: float,
) -> tuple[dict[str, float], list[str]]:
    out = {str(symbol): max(0.0, float(weight)) for symbol, weight in adjusted.items()}
    notes: list[str] = []
    active_symbols = {symbol for symbol, weight in out.items() if float(weight) > 1e-9}
    sector_counts: dict[str, int] = {}
    for symbol in active_symbols:
        current_entry = next((item for item in rank_entries if str(item["symbol"]) == symbol), None)
        sector = str(current_entry["sector"]) if current_entry is not None else ""
        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    baseline_slot_weight = float(target_exposure) / max(1, int(target_slot_count))
    rank_entry_map = {str(item["symbol"]): item for item in rank_entries}
    for symbol, current_weight in current_weights.items():
        current = max(0.0, float(current_weight))
        if current <= 1e-9 or symbol in active_symbols:
            continue
        entry = rank_entry_map.get(str(symbol))
        if entry is None or not bool(entry.get("buffer_hold_eligible", False)):
            continue
        rank = int(entry.get("rank", 999999))
        if rank > int(hold_until_rank):
            continue
        sector = str(entry.get("sector", ""))
        if sector_name_cap > 0 and sector and sector_counts.get(sector, 0) >= int(sector_name_cap):
            continue
        retained_weight = min(current, baseline_slot_weight * (0.90 if rank <= target_slot_count else 0.75))
        retained_weight = max(retained_weight, min(current, baseline_slot_weight * 0.50))
        if retained_weight <= 1e-9:
            continue
        out[str(symbol)] = max(float(out.get(str(symbol), 0.0)), float(retained_weight))
        active_symbols.add(str(symbol))
        if sector:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        notes.append(f"{symbol}: retained by hold buffer (rank {rank}/{hold_until_rank}).")
    return out, notes


def _enforce_sector_name_cap_on_weights(
    *,
    adjusted: dict[str, float],
    current_weights: dict[str, float],
    state_map: dict[str, StockForecastState],
    rank_map: dict[str, int],
    sector_name_cap: int,
) -> tuple[dict[str, float], list[str]]:
    out = {
        str(symbol): max(0.0, float(weight))
        for symbol, weight in adjusted.items()
        if float(weight) > 1e-9
    }
    if sector_name_cap <= 0 or len(out) <= sector_name_cap:
        return out, []

    active_sectors = {
        str(getattr(state_map.get(symbol), "sector", "") or "其他")
        for symbol in out
        if state_map.get(symbol) is not None
    }
    if len(active_sectors) <= 1:
        return out, []

    notes: list[str] = []
    kept: dict[str, float] = {}
    sector_counts: dict[str, int] = {}
    ordered_symbols = sorted(
        out,
        key=lambda symbol: (
            int(rank_map.get(str(symbol), 999999)),
            -max(0.0, float(current_weights.get(str(symbol), 0.0))),
            -max(0.0, float(out.get(str(symbol), 0.0))),
            str(symbol),
        ),
    )
    for symbol in ordered_symbols:
        stock_state = state_map.get(str(symbol))
        sector = str(getattr(stock_state, "sector", "") or "其他")
        if sector_counts.get(sector, 0) < int(sector_name_cap):
            kept[str(symbol)] = float(out[str(symbol)])
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            continue
        notes.append(f"{symbol}: dropped by sector name cap ({sector_name_cap}) for {sector}.")
    return kept, notes


def finalize_target_weights(
    *,
    desired_weights: dict[str, float],
    current_weights: dict[str, float],
    current_holding_days: dict[str, int],
    stocks: list[StockForecastState],
    state: CompositeState | None = None,
    target_exposure: float,
    min_trade_delta: float,
    min_holding_days: int,
    deps: PolicyRuntimeDependencies,
) -> tuple[dict[str, float], list[str]]:
    adjusted = {symbol: max(0.0, float(weight)) for symbol, weight in desired_weights.items()}
    state_map = {item.symbol: item for item in stocks}
    holding_profiles = {
        symbol: holding_alpha_profile(stock_state, deps=deps)
        for symbol, stock_state in state_map.items()
    }
    composite_state = state
    theme_episodes = {
        str(item.theme): item
        for item in getattr(composite_state, "theme_episodes", []) or []
    } if composite_state is not None else {}
    role_states = {
        str(symbol): payload
        for symbol, payload in (getattr(composite_state, "stock_role_states", {}) or {}).items()
    } if composite_state is not None else {}
    rank_entries, rank_map, hold_until_rank, sector_name_cap, target_slot_count = _build_hold_buffer_rank_context(
        desired_weights=adjusted,
        current_weights=current_weights,
        state_map=state_map,
        holding_profiles=holding_profiles,
        role_states=role_states,
        theme_episodes=theme_episodes,
        deps=deps,
    )
    notes: list[str] = []
    locked_symbols: set[str] = set()
    adjusted, hold_buffer_notes = _apply_hold_buffer_retention(
        adjusted=adjusted,
        current_weights=current_weights,
        rank_entries=rank_entries,
        hold_until_rank=hold_until_rank,
        sector_name_cap=sector_name_cap,
        target_slot_count=target_slot_count,
        target_exposure=target_exposure,
    )
    notes.extend(hold_buffer_notes)
    adjusted, sector_cap_notes = _enforce_sector_name_cap_on_weights(
        adjusted=adjusted,
        current_weights=current_weights,
        state_map=state_map,
        rank_map=rank_map,
        sector_name_cap=sector_name_cap,
    )
    notes.extend(sector_cap_notes)

    all_symbols = sorted(set(adjusted) | set(current_weights))
    for symbol in all_symbols:
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        stock_state = state_map.get(symbol)
        status = "data_insufficient" if stock_state is None else str(getattr(stock_state, "tradability_status", "normal") or "normal")
        target = max(0.0, float(adjusted.get(symbol, 0.0)))

        if stock_state is None and current > 1e-9:
            adjusted[symbol] = current
            locked_symbols.add(symbol)
            notes.append(f"{symbol}: missing state, holding frozen.")
            continue
        if not deps.is_actionable_status(status):
            if current > 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
                notes.append(f"{symbol}: {status}, holding frozen.")
            else:
                adjusted.pop(symbol, None)
                notes.append(f"{symbol}: {status}, new entry blocked.")
            continue
        if status == "data_insufficient":
            if current <= 1e-9 and target > 1e-9:
                adjusted.pop(symbol, None)
                notes.append(f"{symbol}: data insufficient, new entry blocked.")
                continue
            if target > current + 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
                notes.append(f"{symbol}: data insufficient, add-on blocked.")
                continue
        if (
            current > 1e-9
            and target > current + 1e-9
            and stock_state is not None
            and composite_state is not None
        ):
            add_on_allowed, add_on_reasons = evaluate_fresh_buy_candidate(
                stock_state,
                composite_state=composite_state,
                deps=deps,
            )
            if not add_on_allowed:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
                notes.append(
                    f"{symbol}: add-on blocked by absolute buy gate ({', '.join(add_on_reasons[:2])})."
                )
                continue
        holding_days = int(max(0, current_holding_days.get(symbol, 0)))
        holding_profile = holding_profiles.get(symbol)
        role_state = role_states.get(symbol)
        theme_episode = theme_episodes.get(str(getattr(role_state, "theme", ""))) if role_state is not None else None
        theme_fading = bool(theme_episode is not None and str(theme_episode.phase) == "fading")
        theme_strengthening = bool(theme_episode is not None and str(theme_episode.phase) == "strengthening")
        theme_crowded = bool(
            theme_episode is not None
            and str(theme_episode.phase) == "crowded"
            and float(getattr(theme_episode, "event_risk", 0.0)) >= 0.55
        )
        role_downgrade = bool(getattr(role_state, "role_downgrade", False))
        laggard_role = bool(role_state is not None and str(getattr(role_state, "role", "")) == "laggard")
        strong_theme_holder = bool(
            theme_strengthening
            and role_state is not None
            and str(getattr(role_state, "role", "")) in {"leader", "core"}
        )
        if current > 1e-9 and holding_days < int(min_holding_days) and target < current - 1e-9:
            if (
                holding_profile is not None
                and (
                    bool(holding_profile["urgent_exit"])
                    or theme_fading
                    or role_downgrade
                    or laggard_role
                )
            ):
                notes.append(
                    f"{symbol}: early exit allowed inside holding window because exit pressure is confirmed."
                )
                continue
            adjusted[symbol] = current
            locked_symbols.add(symbol)
            notes.append(
                f"{symbol}: minimum holding window active ({holding_days}/{int(min_holding_days)}d), sell blocked."
            )
            continue
        if (
            current > 1e-9
            and target > 1e-9
            and target < current - 1e-9
            and holding_profile is not None
            and bool(holding_profile["strong_persistence"])
            and not bool(holding_profile["urgent_exit"])
            and target >= current * 0.35
        ):
            retention_ratio = float(
                np.clip(
                    (
                        0.72
                        + 0.25 * max(0.0, float(holding_profile["persistence_score"]) - 0.60)
                        + (0.04 if strong_theme_holder else 0.0)
                    ),
                    0.72,
                    0.86 if strong_theme_holder else 0.82,
                )
            )
            retained_weight = max(target, current * retention_ratio)
            if retained_weight > target + 1e-9:
                adjusted[symbol] = retained_weight
                locked_symbols.add(symbol)
                notes.append(
                    (
                        f"{symbol}: strengthening theme plus leader/core role softened the trim."
                        if strong_theme_holder
                        else f"{symbol}: strong alpha persistence, trim softened to keep winner running."
                    )
                )
        if current > 1e-9 and target > 1e-9 and target < current - 1e-9 and theme_crowded:
            notes.append(f"{symbol}: crowded theme with high event risk, reduce into strength instead of chasing.")

    for symbol in sorted(set(adjusted) | set(current_weights)):
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        target = max(0.0, float(adjusted.get(symbol, 0.0)))
        dynamic_trade_delta = float(min_trade_delta)
        holding_profile = holding_profiles.get(symbol)
        role_state = role_states.get(symbol)
        theme_episode = theme_episodes.get(str(getattr(role_state, "theme", ""))) if role_state is not None else None
        theme_fading = bool(theme_episode is not None and str(theme_episode.phase) == "fading")
        strong_theme_holder = bool(
            theme_episode is not None
            and str(theme_episode.phase) == "strengthening"
            and role_state is not None
            and str(getattr(role_state, "role", "")) in {"leader", "core"}
        )
        if current > 1e-9 and target < current - 1e-9 and holding_profile is not None:
            if strong_theme_holder and target > 1e-9:
                dynamic_trade_delta *= 1.90
            elif bool(holding_profile["strong_persistence"]) and target > 1e-9:
                dynamic_trade_delta *= 1.60
            if theme_fading or bool(getattr(role_state, "role_downgrade", False)) or str(getattr(role_state, "role", "")) == "laggard":
                dynamic_trade_delta *= 0.40
            elif bool(holding_profile["urgent_exit"]):
                dynamic_trade_delta *= 0.35
            elif bool(holding_profile["weakening"]):
                dynamic_trade_delta *= 0.45
        if abs(target - current) < dynamic_trade_delta:
            if abs(target - current) > 1e-9:
                if dynamic_trade_delta > float(min_trade_delta) + 1e-9:
                    notes.append(f"{symbol}: strong alpha hold threshold suppressed a small trim.")
                elif dynamic_trade_delta < float(min_trade_delta) - 1e-9:
                    notes.append(f"{symbol}: weakening alpha lowered the rebalance threshold.")
                else:
                    notes.append(f"{symbol}: rebalance gap below threshold.")
            if current > 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
            else:
                adjusted.pop(symbol, None)

    locked_total = float(sum(max(0.0, float(adjusted.get(symbol, 0.0))) for symbol in locked_symbols))
    free_symbols = [
        symbol for symbol, weight in adjusted.items()
        if symbol not in locked_symbols and float(weight) > 1e-9
    ]
    free_total = float(sum(float(adjusted[symbol]) for symbol in free_symbols))
    free_budget = max(0.0, float(target_exposure) - locked_total)
    if free_total > free_budget + 1e-9 and free_total > 1e-9:
        scale = float(free_budget / free_total) if free_budget > 1e-9 else 0.0
        for symbol in free_symbols:
            adjusted[symbol] = float(adjusted[symbol]) * scale
        notes.append("Actionable targets scaled down to respect target exposure after frozen holdings.")

    adjusted = {
        symbol: float(weight)
        for symbol, weight in adjusted.items()
        if float(weight) > 1e-6
    }
    return adjusted, notes


def apply_external_signal_weight_tilts(
    *,
    weights: dict[str, float],
    state: CompositeState,
    target_exposure: float,
    risk_cutoff: float,
    catalyst_boost_cap: float,
    deps: PolicyRuntimeDependencies,
) -> tuple[dict[str, float], list[str]]:
    adjusted = {str(symbol): max(0.0, float(weight)) for symbol, weight in weights.items() if float(weight) > 1e-9}
    if not adjusted:
        return adjusted, []
    notes: list[str] = []
    stock_map = {item.symbol: item for item in state.stocks}
    for symbol in list(adjusted):
        info_state = state.stock_info_states.get(symbol, InfoAggregateState())
        event_risk = float(info_state.event_risk_level)
        catalyst = float(info_state.catalyst_strength)
        alpha_advantage = 0.0
        stock = stock_map.get(symbol)
        if stock is not None:
            alpha_source = getattr(stock, "alpha_score", None)
            if alpha_source is None:
                try:
                    alpha_source = stock_policy_score(stock, deps=deps)
                except Exception:
                    alpha_source = 0.55
            alpha_advantage = max(0.0, float(alpha_source) - 0.55)
        if event_risk >= float(risk_cutoff):
            adjusted[symbol] *= max(0.0, 1.0 - min(0.85, event_risk))
            notes.append(f"{symbol}: event risk above cutoff, target trimmed.")
        elif catalyst > 0.0 and alpha_advantage > 0.0:
            boost = min(float(catalyst_boost_cap), 0.35 * catalyst + 0.80 * alpha_advantage)
            adjusted[symbol] *= 1.0 + boost
            notes.append(f"{symbol}: catalyst aligned with alpha, target boosted.")
    total = float(sum(adjusted.values()))
    if total <= 1e-9:
        return {}, notes
    scale = float(target_exposure) / total if target_exposure > 1e-9 else 0.0
    return (
        {
            symbol: float(weight) * scale
            for symbol, weight in adjusted.items()
            if float(weight) * scale > 1e-6
        },
        notes,
    )


def enforce_single_name_cap(
    *,
    weights: dict[str, float],
    max_single_position: float,
) -> dict[str, float]:
    adjusted = {
        str(symbol): max(0.0, float(weight))
        for symbol, weight in weights.items()
        if float(weight) > 1e-9
    }
    cap = max(0.0, float(max_single_position))
    if not adjusted or cap <= 1e-9:
        return adjusted
    for symbol in list(adjusted):
        adjusted[symbol] = min(adjusted[symbol], cap)
    return {
        symbol: float(weight)
        for symbol, weight in adjusted.items()
        if float(weight) > 1e-6
    }


def sector_budgets_from_weights(
    *,
    symbol_weights: dict[str, float],
    stocks: list[StockForecastState],
) -> dict[str, float]:
    state_map = {item.symbol: item for item in stocks}
    out: dict[str, float] = {}
    for symbol, weight in symbol_weights.items():
        if float(weight) <= 1e-9:
            continue
        sector = state_map.get(symbol).sector if state_map.get(symbol) is not None else "其他"
        out[sector] = out.get(sector, 0.0) + float(weight)
    return out


def advance_holding_days(
    *,
    prev_holding_days: dict[str, int],
    prev_weights: dict[str, float],
    next_weights: dict[str, float],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for symbol, weight in next_weights.items():
        if float(weight) <= 1e-9:
            continue
        prev_weight = max(0.0, float(prev_weights.get(symbol, 0.0)))
        if prev_weight > 1e-9:
            out[symbol] = int(max(1, int(prev_holding_days.get(symbol, 0)) + 1))
        else:
            out[symbol] = 1
    return out


def apply_policy(
    policy_input: PolicyInput,
    *,
    policy_spec: PolicySpec | None = None,
    deps: PolicyRuntimeDependencies,
) -> PolicyDecision:
    policy_spec = policy_spec or PolicySpec()
    state = policy_input.composite_state
    market = state.market
    cross = state.cross_section
    min_holding_days = 5

    target_position_count = 1
    turnover_cap = float(policy_spec.risk_off_turnover_cap)
    intraday_t_allowed = False
    risk_notes: list[str] = []
    raw_candidate_stocks = deps.candidate_stocks_from_state(state)
    current_live_symbols = {
        str(symbol)
        for symbol, weight in policy_input.current_weights.items()
        if max(0.0, float(weight)) > 1e-9
    }
    trim_candidate_scores: dict[str, float] = {}
    trim_candidate_ranks: dict[str, int] = {}
    trim_candidate_labels: dict[str, str] = {}
    actionability_profiles: dict[str, dict[str, float | str]] = {}
    actionability_scores: dict[str, float] = {}
    candidate_stocks: list[StockForecastState] = []
    blocked_fresh_candidates: dict[str, list[str]] = {}
    fresh_eligible_count = 0
    leader_snapshots = build_leader_score_snapshots(state=state)
    risk_off_pilot: dict[str, object] | None = None
    for stock in raw_candidate_stocks:
        symbol = str(getattr(stock, "symbol", "")).strip()
        actionability = stock_actionability_profile(
            stock,
            composite_state=state,
            deps=deps,
        )
        actionability_profiles[symbol] = actionability
        actionability_scores[symbol] = float(actionability["score"])
        if symbol in current_live_symbols:
            candidate_stocks.append(stock)
            continue
        buy_allowed, buy_reasons = evaluate_fresh_buy_candidate(
            stock,
            composite_state=state,
            deps=deps,
        )
        if buy_allowed:
            candidate_stocks.append(stock)
            fresh_eligible_count += 1
        else:
            blocked_fresh_candidates[symbol] = buy_reasons
    if state.risk_regime == "risk_off" and not current_live_symbols and fresh_eligible_count == 0:
        pilot_stock, risk_off_pilot = select_risk_off_pilot_candidate(
            blocked_candidates=blocked_fresh_candidates,
            raw_candidate_stocks=raw_candidate_stocks,
            actionability_profiles=actionability_profiles,
            leader_snapshots=leader_snapshots,
            composite_state=state,
            deps=deps,
        )
        if pilot_stock is not None and risk_off_pilot is not None:
            candidate_stocks.append(pilot_stock)
            fresh_eligible_count = 1
            blocked_fresh_candidates.pop(str(getattr(pilot_stock, "symbol", "")).strip(), None)
    fresh_scores = [
        float(actionability_scores.get(str(getattr(stock, "symbol", "")).strip(), 0.0))
        for stock in raw_candidate_stocks
        if str(getattr(stock, "symbol", "")).strip() not in current_live_symbols
    ]
    best_fresh_actionability = max(fresh_scores, default=0.0)
    candidate_selection = getattr(state, "candidate_selection", None)
    alpha_metrics = alpha_opportunity_metrics(candidate_stocks, deps=deps)
    candidate_risk = deps.candidate_risk_snapshot(raw_candidate_stocks or candidate_stocks)
    mainlines = list(getattr(state, "mainlines", []))
    dominant_mainline_sectors = deps.dominant_mainline_sectors(mainlines)
    mainline_sector_boosts, mainline_symbol_boosts, confirmed_mainlines = mainline_preference_maps(
        mainlines,
        risk_cutoff=float(policy_spec.event_risk_cutoff),
        deps=deps,
    )
    leader_symbol_boosts, leader_weight_notes = leader_symbol_preference_maps(
        state=state,
        candidate_stocks=candidate_stocks,
        snapshots=leader_snapshots,
        deps=deps,
    )
    risk_notes.extend(leader_weight_notes)
    if risk_off_pilot is not None:
        risk_notes.append(
            "Risk-off pilot unlocked: "
            f"{risk_off_pilot['symbol']} admitted with small size because leader and catalyst support offset a thin forward buffer."
        )
    if blocked_fresh_candidates:
        risk_notes.append(
            f"Absolute buy gate blocked {len(blocked_fresh_candidates)} fresh candidates without positive edge."
        )
        preview = ", ".join(
            f"{symbol}({', '.join(reasons[:2])})"
            for symbol, reasons in list(blocked_fresh_candidates.items())[:3]
        )
        if preview:
            risk_notes.append("Fresh-buy blocks: " + preview)
    if raw_candidate_stocks and fresh_eligible_count == 0:
        risk_notes.append("No fresh long passed the absolute edge gate; prefer cash or only manage existing holdings.")
    if policy_input.exit_behavior_model and current_live_symbols:
        exit_runtime_rows = build_exit_behavior_runtime_rows(state=state)
        ranked_trim_candidates = rank_exit_candidates(
            model_payload=policy_input.exit_behavior_model,
            rows_by_symbol=exit_runtime_rows,
            symbols=current_live_symbols,
        )
        for item in ranked_trim_candidates:
            symbol = str(item["symbol"]).upper()
            trim_candidate_scores[symbol] = float(item["score"])
            trim_candidate_ranks[symbol] = int(item["rank"])
            trim_candidate_labels[symbol] = str(item["label"])
        if ranked_trim_candidates:
            preview = ", ".join(
                f"{str(item['symbol']).upper()}({float(item['score']):.2f},{item['label']})"
                for item in ranked_trim_candidates[:3]
            )
            risk_notes.append("Exit ranking overlay: " + preview)
    alpha_headroom = float(alpha_metrics["alpha_headroom"])
    alpha_breadth = float(alpha_metrics["breadth_ratio"])
    top_alpha = float(alpha_metrics["top_score"])
    median_alpha = float(alpha_metrics["median_score"])
    entry_separation = float(alpha_metrics["entry_separation"])
    dominant_alpha_gap = float(max(0.0, top_alpha - median_alpha))
    dominant_alpha_setup = bool(
        alpha_headroom >= 0.05
        or entry_separation >= 0.10
        or dominant_alpha_gap >= 0.06
    )
    market_info = getattr(state, "market_info_state", InfoAggregateState())
    capital_flow = getattr(state, "capital_flow_state", CapitalFlowState())
    macro_context = getattr(state, "macro_context_state", MacroContextState())
    near_term_stack = float(
        0.20 * float(market.up_1d_prob)
        + 0.22 * float(getattr(market, "up_2d_prob", 0.5))
        + 0.24 * float(getattr(market, "up_3d_prob", 0.5))
        + 0.34 * float(market.up_5d_prob)
    )

    if state.risk_regime == "risk_on":
        regime_floor = 0.45
        target_position_count = int(policy_spec.risk_on_positions)
        turnover_cap = float(policy_spec.risk_on_turnover_cap)
        intraday_t_allowed = state.strategy_mode == "range_rotation"
    elif state.risk_regime == "cautious":
        regime_floor = 0.35
        target_position_count = int(policy_spec.cautious_positions)
        turnover_cap = float(policy_spec.cautious_turnover_cap)
    else:
        regime_floor = 0.25
        target_position_count = int(policy_spec.risk_off_positions)
        turnover_cap = float(policy_spec.risk_off_turnover_cap)
        risk_notes.append("Risk-off regime: hard floor reduced, but not forced into deep cash.")

    alpha_base_exposure = float(
        deps.clip(
            0.25
            + 1.60 * alpha_headroom
            + 0.55 * alpha_breadth
            + 0.35 * max(0.0, top_alpha - 0.55),
            regime_floor,
            0.95,
        )
    )
    target_exposure = float(alpha_base_exposure)

    if near_term_stack < 0.50:
        target_exposure *= 0.95
        risk_notes.append("Near-term market stack below 0.50: mild exposure trim.")
    if float(market_info.event_risk_level) >= float(policy_spec.event_risk_cutoff):
        target_exposure *= 0.90
        target_position_count = max(1, target_position_count - 1)
        turnover_cap = min(turnover_cap, 0.20)
        risk_notes.append("Event risk elevated: exposure trimmed and concentration reduced.")
    if macro_context.macro_risk_level == "high":
        target_exposure *= 0.88
        turnover_cap = min(turnover_cap, 0.18)
        risk_notes.append("Macro risk high: exposure trimmed and turnover capped.")
    elif macro_context.macro_risk_level == "elevated":
        target_exposure *= 0.94
        risk_notes.append("Macro risk elevated: mild exposure trim.")
    if capital_flow.flow_regime in {"outflow", "strong_outflow"}:
        flow_penalty = float(policy_spec.flow_exposure_cap) * (1.0 if capital_flow.flow_regime == "strong_outflow" else 0.65)
        target_exposure = max(regime_floor, target_exposure - flow_penalty)
        turnover_cap = min(turnover_cap, 0.22 if capital_flow.flow_regime == "outflow" else 0.18)
        risk_notes.append(f"Capital flow {capital_flow.flow_regime}: exposure trimmed.")
    elif capital_flow.flow_regime in {"inflow", "strong_inflow"} and state.risk_regime != "risk_off":
        flow_boost = float(policy_spec.flow_exposure_cap) * (0.60 if capital_flow.flow_regime == "inflow" else 1.0)
        target_exposure = min(1.0, target_exposure + flow_boost)
        turnover_cap = min(0.45, turnover_cap + 0.02)
        risk_notes.append(f"Capital flow {capital_flow.flow_regime}: measured exposure boost.")
    if market.drawdown_risk >= 0.50:
        target_exposure *= 0.90
        turnover_cap = min(turnover_cap, 0.22)
        risk_notes.append("Drawdown risk elevated: mild exposure trim.")
    elif market.drawdown_risk >= 0.35:
        target_exposure *= 0.94
        turnover_cap = min(turnover_cap, 0.20)
        risk_notes.append("Intermediate drawdown risk: extra exposure trim.")
    if cross.fund_flow_strength < 0.0:
        target_exposure *= 0.94
        risk_notes.append("Fund flow weak: mild exposure trim.")
    if market.volatility_regime == "high":
        target_exposure *= 0.90
        turnover_cap = min(turnover_cap, 0.24)
        risk_notes.append("High volatility regime: exposure trimmed, not capped aggressively.")
    elif market.volatility_regime == "low" and state.risk_regime == "risk_on" and cross.breadth_strength > 0.15 and near_term_stack >= 0.56:
        target_exposure = min(1.0, target_exposure * 1.05)
        risk_notes.append("Low volatility with strong near-term stack: exposure nudged up.")
    if cross.large_vs_small_bias < -0.05:
        target_position_count = min(target_position_count + 1, 5)
        risk_notes.append("Large-cap bias weak: diversify more positions.")
    if cross.growth_vs_value_bias < -0.08:
        turnover_cap = min(turnover_cap, 0.24)
        risk_notes.append("Growth style weak: turnover capped conservatively.")

    if (alpha_headroom <= 0.01 or alpha_breadth < 0.05) and not dominant_alpha_setup:
        target_exposure *= 0.90
        target_position_count = max(1, target_position_count - 1)
        turnover_cap = min(turnover_cap, 0.22)
        risk_notes.append("Cross-sectional alpha weak: exposure trimmed.")
    elif (
        alpha_headroom >= 0.02
        and alpha_breadth >= 0.08
        and cross.breadth_strength >= 0.10
        and market.liquidity_stress <= 0.60
    ):
        alpha_boost = min(0.12, 0.70 * alpha_headroom + 0.18 * alpha_breadth)
        target_exposure = min(1.0, target_exposure + alpha_boost)
        if top_alpha >= 0.62:
            target_position_count = min(5, target_position_count + 1)
        risk_notes.append("Cross-sectional alpha strong: exposure boosted.")

    if top_alpha >= 0.64 and cross.breadth_strength >= 0.12 and state.risk_regime != "risk_off":
        turnover_cap = min(0.45, turnover_cap + 0.03)
        risk_notes.append("Top alpha concentration supports measured rotation.")

    concentration_preference = bool(
        state.risk_regime != "risk_off"
        and top_alpha >= 0.69
        and dominant_alpha_setup
        and (alpha_breadth <= 0.22 or entry_separation >= 0.10 or dominant_alpha_gap >= 0.06)
        and cross.breadth_strength >= 0.08
        and float(candidate_risk["fragile_ratio"]) < 0.18
        and float(candidate_risk["durability_score"]) >= 0.58
        and float(market.drawdown_risk) <= 0.35
    )

    if mainlines:
        top_mainline = mainlines[0]
        if float(top_mainline.event_risk_level) >= float(policy_spec.event_risk_cutoff):
            target_exposure *= 0.94
            target_position_count = max(1, target_position_count - 1)
            turnover_cap = min(turnover_cap, 0.20)
            risk_notes.append(f"Mainline {top_mainline.name} is risk-watched: exposure trimmed.")
        elif (
            float(top_mainline.conviction) >= 0.62
            and float(top_mainline.catalyst_strength) >= 0.24
            and state.risk_regime != "risk_off"
        ):
            target_exposure = min(1.0, target_exposure + 0.03)
            if not concentration_preference:
                target_position_count = min(5, target_position_count + 1)
            risk_notes.append(f"Mainline {top_mainline.name} confirmed: measured exposure support.")
    theme_episodes = list(getattr(state, "theme_episodes", []) or [])
    if theme_episodes:
        top_theme = theme_episodes[0]
        if str(top_theme.phase) == "crowded" and float(top_theme.event_risk) >= float(policy_spec.event_risk_cutoff):
            risk_notes.append(
                f"Theme {top_theme.theme} is crowded with high event risk: favor reduce-into-strength over fresh chase."
            )
    if confirmed_mainlines:
        if concentration_preference:
            target_position_count = min(target_position_count, 3)
        else:
            target_position_count = max(target_position_count, min(4, len(confirmed_mainlines) + 1))
        if state.risk_regime != "risk_off":
            turnover_cap = min(0.40, turnover_cap + 0.02)
        risk_notes.append(
            "Mainline budgets prioritized: " + ", ".join(str(item.name) for item in confirmed_mainlines[:3])
        )

    if float(candidate_risk["fragile_ratio"]) >= 0.35:
        target_exposure *= 0.92
        turnover_cap = min(turnover_cap, 0.20)
        risk_notes.append("Candidate set fragile: exposure and turnover trimmed.")
    elif float(candidate_risk["fragile_ratio"]) >= 0.20 and state.risk_regime != "risk_on":
        target_exposure *= 0.95
        risk_notes.append("Candidate set mildly fragile under cautious regime: exposure trimmed.")
    if float(candidate_risk["reversal_ratio"]) >= 0.25:
        turnover_cap = min(turnover_cap, 0.18)
        risk_notes.append("Short-term reversal risk elevated across candidates: turnover capped.")
    if (
        float(candidate_risk["durability_score"]) <= 0.54
        and (float(market.drawdown_risk) >= 0.30 or float(cross.weak_stock_ratio) >= 0.48)
    ):
        target_exposure *= 0.94
        risk_notes.append("Candidate durability soft in a fragile tape: extra exposure trim.")
    if concentration_preference:
        target_exposure = min(1.0, target_exposure + min(0.05, 0.55 * alpha_headroom))
        target_position_count = max(2, min(target_position_count, 3))
        risk_notes.append("Top alpha concentrated and durable: allow tighter portfolio concentration.")

    if candidate_selection is not None and len(raw_candidate_stocks) < len(state.stocks):
        risk_notes.append(
            f"Candidate shortlist active: {len(raw_candidate_stocks)}/{len(state.stocks)} names after macro-sector screening."
        )
    if raw_candidate_stocks and len(candidate_stocks) < len(raw_candidate_stocks):
        risk_notes.append(
            f"Fresh-buy eligible after absolute gate: {len(candidate_stocks)}/{len(raw_candidate_stocks)} names."
        )
    if candidate_stocks:
        target_position_count = min(target_position_count, len(candidate_stocks))
    if (
        state.risk_regime == "risk_off"
        and not current_live_symbols
        and risk_off_pilot is None
        and (fresh_eligible_count == 0 or best_fresh_actionability < 0.57)
    ):
        target_exposure = 0.0
        target_position_count = 0
        turnover_cap = min(turnover_cap, float(policy_spec.risk_off_turnover_cap))
        risk_notes.append("Risk-off actionability weak: stay in cash instead of forcing a new long.")

    min_exposure_floor = 0.0 if target_position_count <= 0 else regime_floor
    target_exposure = deps.clip(target_exposure, min_exposure_floor, 1.0)
    max_single_position = 0.28
    if market.volatility_regime == "high":
        max_single_position = min(max_single_position, 0.24)
    if cross.large_vs_small_bias < -0.05:
        max_single_position = min(max_single_position, 0.22)
    if cross.growth_vs_value_bias < -0.08:
        max_single_position = min(max_single_position, 0.20)
    if float(candidate_risk["fragile_ratio"]) >= 0.20:
        max_single_position = min(max_single_position, 0.20 if state.risk_regime == "risk_on" else 0.18)
        risk_notes.append("Candidate fragility keeps single-name sizing conservative.")
    if (
        candidate_selection is not None
        and int(getattr(candidate_selection, "total_scored", 0)) >= 120
        and int(getattr(candidate_selection, "shortlist_size", 0)) >= 10
    ):
        if concentration_preference:
            target_position_count = min(target_position_count, 3)
            max_single_position = min(max_single_position, 0.24 if state.risk_regime == "risk_on" else 0.20)
            risk_notes.append("Large-universe shortlist respected, but top alpha supports selective concentration.")
        else:
            target_position_count = max(target_position_count, 3 if state.risk_regime != "risk_off" else 2)
            max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_on" else 0.16)
            risk_notes.append("Large-universe shortlist: concentration spread across more names.")
    if float(market.drawdown_risk) >= 0.35 or float(cross.weak_stock_ratio) >= 0.50:
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_on" else 0.16)
        risk_notes.append("Fragile tape: single-name cap tightened.")
    if alpha_breadth >= 0.12 and alpha_headroom >= 0.02 and not concentration_preference:
        target_position_count = max(target_position_count, 2)
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_off" else 0.22)
        risk_notes.append("Alpha breadth strong: concentration reduced across more names.")
    elif concentration_preference:
        max_single_position = min(max_single_position, 0.26 if state.risk_regime == "risk_on" else 0.22)
    if risk_off_pilot is not None and not current_live_symbols:
        target_exposure = min(float(target_exposure), 0.12)
        target_exposure = max(float(target_exposure), 0.08)
        target_position_count = 1
        turnover_cap = min(float(turnover_cap), 0.12)
        risk_notes.append("Risk-off pilot sizing active: cap exposure to a single small probing position.")
    min_position_floor = 0 if (state.risk_regime == "risk_off" and not current_live_symbols) else 1
    target_position_count = int(np.clip(target_position_count, min_position_floor, 5))
    candidate_sector_names = set(
        getattr(candidate_selection, "shortlisted_sectors", []) if candidate_selection is not None else []
    )
    if dominant_mainline_sectors:
        candidate_sector_names.update(dominant_mainline_sectors)
    policy_sectors = [
        sector for sector in state.sectors
        if not candidate_sector_names or str(sector.sector) in candidate_sector_names
    ]
    if candidate_stocks:
        candidate_stock_sectors = {str(stock.sector) for stock in candidate_stocks}
        policy_sectors = [
            sector for sector in policy_sectors
            if str(sector.sector) in candidate_stock_sectors
        ]
        if not policy_sectors:
            policy_sectors = [
                sector for sector in state.sectors
                if str(sector.sector) in candidate_stock_sectors
            ]
    if not policy_sectors and candidate_stocks:
        policy_sectors = list(state.sectors)
    if mainline_sector_boosts and policy_sectors:
        policy_sectors = sorted(
            policy_sectors,
            key=lambda sector: (
                float(mainline_sector_boosts.get(str(sector.sector), 0.0)),
                float(sector.up_20d_prob),
                float(sector.relative_strength),
            ),
            reverse=True,
        )
    desired_sector_budgets: dict[str, float] = {}
    sector_cap_notes: list[str] = []
    if candidate_stocks and policy_sectors:
        desired_sector_budgets = ranked_sector_budgets_with_alpha(
            sectors=policy_sectors[: max(1, target_position_count)],
            stocks=candidate_stocks,
            target_exposure=target_exposure,
            sector_score_adjustments=mainline_sector_boosts,
            deps=deps,
        )
        desired_sector_budgets, sector_cap_notes = cap_sector_budgets(
            sector_budgets=desired_sector_budgets,
            target_exposure=target_exposure,
            risk_regime=state.risk_regime,
            breadth_strength=float(cross.breadth_strength),
            deps=deps,
        )
    risk_notes.extend(sector_cap_notes)
    desired_symbol_target_weights: dict[str, float] = {}
    if candidate_stocks and desired_sector_budgets:
        actionability_adjustments = {
            symbol: float(
                deps.clip(
                    1.80
                    * (
                        float(actionability_profiles.get(symbol, {}).get("score", 0.0))
                        - float(actionability_profiles.get(symbol, {}).get("threshold", 0.5))
                    ),
                    -0.12,
                    0.12,
                )
            )
            for symbol in actionability_scores
            if symbol in {str(item.symbol) for item in candidate_stocks}
        }
        desired_symbol_target_weights = allocate_with_sector_budgets(
            stocks=candidate_stocks,
            sector_budgets=desired_sector_budgets,
            target_position_count=int(target_position_count),
            sector_strengths={
                sector: float(weight) / max(float(target_exposure), 1e-9)
                for sector, weight in desired_sector_budgets.items()
            },
            max_single_position=float(max_single_position),
            symbol_score_adjustments=merge_symbol_score_adjustments(
                mainline_symbol_boosts,
                leader_symbol_boosts,
                actionability_adjustments,
                deps=deps,
            ),
            deps=deps,
        )
    desired_symbol_target_weights, external_signal_notes = apply_external_signal_weight_tilts(
        weights=desired_symbol_target_weights,
        state=state,
        target_exposure=float(target_exposure),
        risk_cutoff=float(policy_spec.event_risk_cutoff),
        catalyst_boost_cap=float(policy_spec.catalyst_boost_cap),
        deps=deps,
    )
    desired_symbol_target_weights = enforce_single_name_cap(
        weights=desired_symbol_target_weights,
        max_single_position=float(max_single_position),
    )
    risk_notes.extend(external_signal_notes)
    symbol_target_weights, execution_notes = finalize_target_weights(
        desired_weights=desired_symbol_target_weights,
        current_weights=policy_input.current_weights,
        current_holding_days=policy_input.current_holding_days,
        stocks=state.stocks,
        state=state,
        target_exposure=target_exposure,
        min_trade_delta=min(0.02, 0.25 * float(turnover_cap)),
        min_holding_days=min_holding_days,
        deps=deps,
    )
    symbol_target_weights = enforce_single_name_cap(
        weights=symbol_target_weights,
        max_single_position=float(max_single_position),
    )
    risk_notes.extend(execution_notes)
    sector_budgets = sector_budgets_from_weights(
        symbol_weights=symbol_target_weights,
        stocks=state.stocks,
    )

    current_total = sum(max(0.0, float(v)) for v in policy_input.current_weights.values())
    rebalance_gap = abs(float(target_exposure) - float(current_total))
    rebalance_now = rebalance_gap >= 0.05
    rebalance_intensity = deps.clip(rebalance_gap / max(0.05, turnover_cap), 0.0, 1.0)

    return PolicyDecision(
        target_exposure=float(target_exposure),
        target_position_count=int(target_position_count),
        rebalance_now=bool(rebalance_now),
        rebalance_intensity=float(rebalance_intensity),
        intraday_t_allowed=bool(intraday_t_allowed),
        turnover_cap=float(turnover_cap),
        sector_budgets=sector_budgets,
        desired_sector_budgets=desired_sector_budgets,
        symbol_target_weights=symbol_target_weights,
        desired_symbol_target_weights=desired_symbol_target_weights,
        execution_notes=execution_notes,
        risk_notes=risk_notes,
        actionability_scores=actionability_scores,
        actionable_symbols=sorted(str(stock.symbol) for stock in candidate_stocks),
        blocked_fresh_candidates=blocked_fresh_candidates,
        trim_candidate_scores=trim_candidate_scores,
        trim_candidate_ranks=trim_candidate_ranks,
        trim_candidate_labels=trim_candidate_labels,
    )


def build_trade_actions(
    *,
    decision: PolicyDecision,
    current_weights: dict[str, float],
) -> list[TradeAction]:
    all_symbols = sorted(set(current_weights) | set(decision.symbol_target_weights))
    actions: list[TradeAction] = []
    for symbol in all_symbols:
        current_weight = max(0.0, float(current_weights.get(symbol, 0.0)))
        target_weight = max(0.0, float(decision.symbol_target_weights.get(symbol, 0.0)))
        delta_weight = float(target_weight - current_weight)
        if delta_weight > 0.02:
            action = "BUY"
        elif delta_weight < -0.02:
            action = "SELL"
        else:
            action = "HOLD"

        note = ""
        if action == "HOLD" and abs(delta_weight) > 1e-9:
            note = "below_rebalance_threshold"

        actions.append(
            TradeAction(
                symbol=symbol,
                name=symbol,
                action=action,
                current_weight=float(current_weight),
                target_weight=float(target_weight),
                delta_weight=float(delta_weight),
                note=note,
            )
        )
    actions.sort(key=lambda item: (abs(float(item.delta_weight)), float(item.target_weight)), reverse=True)
    return actions


def policy_spec_from_model(
    *,
    state: CompositeState,
    model: LearnedPolicyModel,
    deps: PolicyRuntimeDependencies,
) -> PolicySpec:
    features = deps.policy_feature_vector(state)
    exposure_coef = deps.normalize_coef_vector(model.exposure_coef, features.size)
    position_coef = deps.normalize_coef_vector(model.position_coef, features.size)
    turnover_coef = deps.normalize_coef_vector(model.turnover_coef, features.size)
    exposure = deps.clip(
        deps.predict_ridge(features, model.exposure_intercept, exposure_coef),
        0.20,
        0.95,
    )
    positions = int(
        round(
            deps.clip(
                deps.predict_ridge(features, model.position_intercept, position_coef),
                1.0,
                6.0,
            )
        )
    )
    turnover_cap = deps.clip(
        deps.predict_ridge(features, model.turnover_intercept, turnover_coef),
        0.10,
        0.45,
    )
    cautious_exposure = deps.clip(0.5 * (exposure + 0.35), 0.30, exposure)
    risk_off_exposure = deps.clip(0.5 * cautious_exposure, 0.20, 0.40)
    cautious_positions = min(positions, max(1, positions - 1))
    risk_off_positions = max(1, positions - 2)
    cautious_turnover = deps.clip(min(turnover_cap, 0.85 * turnover_cap), 0.10, turnover_cap)
    risk_off_turnover = deps.clip(min(cautious_turnover, 0.70 * turnover_cap), 0.08, cautious_turnover)
    return PolicySpec(
        risk_on_exposure=float(exposure),
        cautious_exposure=float(cautious_exposure),
        risk_off_exposure=float(risk_off_exposure),
        risk_on_positions=int(positions),
        cautious_positions=int(cautious_positions),
        risk_off_positions=int(risk_off_positions),
        risk_on_turnover_cap=float(turnover_cap),
        cautious_turnover_cap=float(cautious_turnover),
        risk_off_turnover_cap=float(risk_off_turnover),
    )
