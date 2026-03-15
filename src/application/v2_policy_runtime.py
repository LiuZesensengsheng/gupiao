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
        }
    scores = sorted((stock_policy_score(stock, deps=deps) for stock in actionable), reverse=True)
    top_slice = scores[: min(3, len(scores))]
    top_score = float(scores[0])
    avg_top3 = float(sum(top_slice) / max(1, len(top_slice)))
    median_score = float(np.median(scores))
    strong_cut = max(0.56, median_score + 0.03)
    strong_count = int(sum(1 for score in scores if score >= strong_cut))
    breadth_ratio = float(strong_count / max(1, len(scores)))
    alpha_headroom = float(max(0.0, avg_top3 - max(0.54, median_score)))
    return {
        "top_score": top_score,
        "avg_top3": avg_top3,
        "median_score": median_score,
        "breadth_ratio": breadth_ratio,
        "strong_count": float(strong_count),
        "alpha_headroom": alpha_headroom,
    }


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


def finalize_target_weights(
    *,
    desired_weights: dict[str, float],
    current_weights: dict[str, float],
    current_holding_days: dict[str, int],
    stocks: list[StockForecastState],
    target_exposure: float,
    min_trade_delta: float,
    min_holding_days: int,
    deps: PolicyRuntimeDependencies,
) -> tuple[dict[str, float], list[str]]:
    adjusted = {symbol: max(0.0, float(weight)) for symbol, weight in desired_weights.items()}
    state_map = {item.symbol: item for item in stocks}
    notes: list[str] = []
    locked_symbols: set[str] = set()

    all_symbols = sorted(set(adjusted) | set(current_weights))
    for symbol in all_symbols:
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        state = state_map.get(symbol)
        status = "data_insufficient" if state is None else str(getattr(state, "tradability_status", "normal") or "normal")
        target = max(0.0, float(adjusted.get(symbol, 0.0)))

        if state is None and current > 1e-9:
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
        holding_days = int(max(0, current_holding_days.get(symbol, 0)))
        if current > 1e-9 and holding_days < int(min_holding_days) and target < current - 1e-9:
            adjusted[symbol] = current
            locked_symbols.add(symbol)
            notes.append(
                f"{symbol}: minimum holding window active ({holding_days}/{int(min_holding_days)}d), sell blocked."
            )
            continue

    for symbol in sorted(set(adjusted) | set(current_weights)):
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        target = max(0.0, float(adjusted.get(symbol, 0.0)))
        if abs(target - current) < float(min_trade_delta):
            if abs(target - current) > 1e-9:
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
    candidate_stocks = deps.candidate_stocks_from_state(state)
    candidate_selection = getattr(state, "candidate_selection", None)
    alpha_metrics = alpha_opportunity_metrics(candidate_stocks, deps=deps)
    candidate_risk = deps.candidate_risk_snapshot(candidate_stocks)
    mainlines = list(getattr(state, "mainlines", []))
    dominant_mainline_sectors = deps.dominant_mainline_sectors(mainlines)
    mainline_sector_boosts, mainline_symbol_boosts, confirmed_mainlines = mainline_preference_maps(
        mainlines,
        risk_cutoff=float(policy_spec.event_risk_cutoff),
        deps=deps,
    )
    alpha_headroom = float(alpha_metrics["alpha_headroom"])
    alpha_breadth = float(alpha_metrics["breadth_ratio"])
    top_alpha = float(alpha_metrics["top_score"])
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

    if alpha_headroom <= 0.01 or alpha_breadth < 0.05:
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

    if mainlines:
        top_mainline = mainlines[0]
        if float(top_mainline.event_risk_level) >= float(policy_spec.event_risk_cutoff):
            target_exposure *= 0.94
            turnover_cap = min(turnover_cap, 0.20)
            risk_notes.append(f"Mainline {top_mainline.name} is risk-watched: exposure trimmed.")
        elif (
            float(top_mainline.conviction) >= 0.62
            and float(top_mainline.catalyst_strength) >= 0.24
            and state.risk_regime != "risk_off"
        ):
            target_exposure = min(1.0, target_exposure + 0.03)
            target_position_count = min(5, target_position_count + 1)
            risk_notes.append(f"Mainline {top_mainline.name} confirmed: measured exposure support.")
    if confirmed_mainlines:
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

    if candidate_selection is not None and len(candidate_stocks) < len(state.stocks):
        risk_notes.append(
            f"Candidate shortlist active: {len(candidate_stocks)}/{len(state.stocks)} names after macro-sector screening."
        )
    if candidate_stocks:
        target_position_count = min(target_position_count, len(candidate_stocks))

    target_exposure = deps.clip(target_exposure, regime_floor, 1.0)
    max_single_position = 0.35
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
        target_position_count = max(target_position_count, 3 if state.risk_regime != "risk_off" else 2)
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_on" else 0.16)
        risk_notes.append("Large-universe shortlist: concentration spread across more names.")
    if float(market.drawdown_risk) >= 0.35 or float(cross.weak_stock_ratio) >= 0.50:
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_on" else 0.16)
        risk_notes.append("Fragile tape: single-name cap tightened.")
    if alpha_breadth >= 0.12 and alpha_headroom >= 0.02:
        target_position_count = max(target_position_count, 2)
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_off" else 0.22)
        risk_notes.append("Alpha breadth strong: concentration reduced across more names.")
    target_position_count = int(np.clip(target_position_count, 1, 5))
    candidate_sector_names = set(
        getattr(candidate_selection, "shortlisted_sectors", []) if candidate_selection is not None else []
    )
    if dominant_mainline_sectors:
        candidate_sector_names.update(dominant_mainline_sectors)
    policy_sectors = [
        sector for sector in state.sectors
        if not candidate_sector_names or str(sector.sector) in candidate_sector_names
    ]
    if not policy_sectors:
        policy_sectors = list(state.sectors)
    if mainline_sector_boosts:
        policy_sectors = sorted(
            policy_sectors,
            key=lambda sector: (
                float(mainline_sector_boosts.get(str(sector.sector), 0.0)),
                float(sector.up_20d_prob),
                float(sector.relative_strength),
            ),
            reverse=True,
        )
    if not candidate_stocks:
        candidate_stocks = list(state.stocks)
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
    desired_symbol_target_weights = allocate_with_sector_budgets(
        stocks=candidate_stocks,
        sector_budgets=desired_sector_budgets,
        target_position_count=int(target_position_count),
        sector_strengths={
            sector: float(weight) / max(float(target_exposure), 1e-9)
            for sector, weight in desired_sector_budgets.items()
        },
        max_single_position=float(max_single_position),
        symbol_score_adjustments=mainline_symbol_boosts,
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
