from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    DailyRunResult,
    MarketForecastState,
    PolicyDecision,
    PolicyInput,
    SectorForecastState,
    StockForecastState,
    StrategySnapshot,
)
from src.domain.entities import TradeAction


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def build_demo_snapshot(strategy_id: str = "swing_v2") -> StrategySnapshot:
    return StrategySnapshot(
        strategy_id=str(strategy_id).strip() or "swing_v2",
        universe_id="top_liquid_200",
        feature_set_version="fset_v2_core",
        market_model_id="mkt_lr_v2",
        sector_model_id="sector_lr_v2",
        stock_model_id="stock_lr_v2",
        cross_section_model_id="cross_section_v2",
        policy_version="policy_v2_rules",
        execution_version="exec_t1_v2",
    )


def build_demo_forecast_states(as_of_date: str = "2026-03-01") -> tuple[
    MarketForecastState,
    list[SectorForecastState],
    list[StockForecastState],
    CrossSectionForecastState,
]:
    market = MarketForecastState(
        as_of_date=as_of_date,
        up_1d_prob=0.57,
        up_5d_prob=0.59,
        up_20d_prob=0.61,
        trend_state="trend",
        drawdown_risk=0.28,
        volatility_regime="normal",
        liquidity_stress=0.22,
    )
    sectors = [
        SectorForecastState("有色", 0.58, 0.62, 0.18, 0.42, 0.31),
        SectorForecastState("化工", 0.55, 0.60, 0.12, 0.36, 0.27),
        SectorForecastState("科技", 0.49, 0.53, -0.03, 0.51, 0.48),
    ]
    stocks = [
        StockForecastState("000630.SZ", "有色", 0.58, 0.60, 0.64, 0.55, 0.10, 0.88),
        StockForecastState("600160.SH", "化工", 0.56, 0.58, 0.62, 0.51, 0.07, 0.84),
        StockForecastState("603619.SH", "化工", 0.52, 0.56, 0.57, 0.49, 0.05, 0.79),
        StockForecastState("603516.SH", "科技", 0.47, 0.50, 0.54, 0.44, 0.03, 0.76),
    ]
    cross_section = CrossSectionForecastState(
        as_of_date=as_of_date,
        large_vs_small_bias=0.08,
        growth_vs_value_bias=-0.04,
        fund_flow_strength=0.16,
        margin_risk_on_score=0.14,
        breadth_strength=0.21,
        leader_participation=0.63,
        weak_stock_ratio=0.29,
    )
    return market, sectors, stocks, cross_section


def compose_state(
    *,
    market: MarketForecastState,
    sectors: list[SectorForecastState],
    stocks: list[StockForecastState],
    cross_section: CrossSectionForecastState,
) -> CompositeState:
    risk_score = max(
        float(market.drawdown_risk),
        float(market.liquidity_stress),
        float(cross_section.weak_stock_ratio),
    )
    if risk_score >= 0.60:
        risk_regime = "risk_off"
    elif risk_score >= 0.40:
        risk_regime = "cautious"
    else:
        risk_regime = "risk_on"

    if market.trend_state == "trend" and cross_section.breadth_strength >= 0.10:
        strategy_mode = "trend_follow"
    elif market.trend_state == "range":
        strategy_mode = "range_rotation"
    else:
        strategy_mode = "defensive"

    ordered_sectors = sorted(sectors, key=lambda item: (item.up_20d_prob, item.relative_strength), reverse=True)
    ordered_stocks = sorted(
        stocks,
        key=lambda item: (item.up_20d_prob, item.excess_vs_sector_prob, item.tradeability_score),
        reverse=True,
    )
    return CompositeState(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        strategy_mode=strategy_mode,
        risk_regime=risk_regime,
    )


def _ranked_sector_budgets(sectors: Iterable[SectorForecastState], *, target_exposure: float) -> dict[str, float]:
    rows = list(sectors)
    if not rows:
        return {}
    raw = [max(0.0, float(item.up_20d_prob) - 0.50) + max(0.0, float(item.relative_strength)) for item in rows]
    total = sum(raw)
    if total <= 1e-9:
        equal = float(target_exposure) / float(len(rows))
        return {item.sector: equal for item in rows}
    return {item.sector: float(target_exposure) * score / total for item, score in zip(rows, raw)}


def apply_policy(policy_input: PolicyInput) -> PolicyDecision:
    state = policy_input.composite_state
    market = state.market
    cross = state.cross_section

    target_exposure = 0.35
    target_position_count = 2
    turnover_cap = 0.20
    intraday_t_allowed = False
    risk_notes: list[str] = []

    if state.risk_regime == "risk_on":
        target_exposure = 0.85
        target_position_count = 4
        turnover_cap = 0.40
        intraday_t_allowed = state.strategy_mode == "range_rotation"
    elif state.risk_regime == "cautious":
        target_exposure = 0.60
        target_position_count = 3
        turnover_cap = 0.28
    else:
        risk_notes.append("Risk-off regime: exposure capped aggressively.")

    if market.up_20d_prob < 0.50:
        target_exposure = min(target_exposure, 0.45)
        risk_notes.append("Market mid-term probability below 0.50.")
    if market.drawdown_risk >= 0.50:
        target_exposure = min(target_exposure, 0.35)
        turnover_cap = min(turnover_cap, 0.18)
        risk_notes.append("Drawdown risk elevated.")
    if cross.fund_flow_strength < 0.0:
        target_exposure *= 0.85
        risk_notes.append("Fund flow weak.")

    target_exposure = _clip(target_exposure, 0.0, 1.0)
    sector_budgets = _ranked_sector_budgets(state.sectors[: max(1, target_position_count)], target_exposure=target_exposure)

    ranked_stocks = []
    for stock in state.stocks:
        score = (
            0.45 * float(stock.up_20d_prob)
            + 0.25 * float(stock.up_5d_prob)
            + 0.15 * float(stock.excess_vs_sector_prob)
            + 0.10 * float(stock.tradeability_score)
            + 0.05 * float(stock.event_impact_score)
        )
        ranked_stocks.append((stock, score))
    ranked_stocks.sort(key=lambda item: item[1], reverse=True)
    selected = ranked_stocks[: max(1, target_position_count)]

    symbol_target_weights: dict[str, float] = {}
    total_score = sum(max(0.0, score - 0.50) for _, score in selected)
    if total_score <= 1e-9:
        equal_weight = target_exposure / float(len(selected))
        for stock, _ in selected:
            symbol_target_weights[stock.symbol] = float(equal_weight)
    else:
        for stock, score in selected:
            symbol_target_weights[stock.symbol] = float(target_exposure) * max(0.0, score - 0.50) / total_score

    current_total = sum(max(0.0, float(v)) for v in policy_input.current_weights.values())
    rebalance_gap = abs(float(target_exposure) - float(current_total))
    rebalance_now = rebalance_gap >= 0.05
    rebalance_intensity = _clip(rebalance_gap / max(0.05, turnover_cap), 0.0, 1.0)

    return PolicyDecision(
        target_exposure=float(target_exposure),
        target_position_count=int(target_position_count),
        rebalance_now=bool(rebalance_now),
        rebalance_intensity=float(rebalance_intensity),
        intraday_t_allowed=bool(intraday_t_allowed),
        turnover_cap=float(turnover_cap),
        sector_budgets=sector_budgets,
        symbol_target_weights=symbol_target_weights,
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


def run_daily_v2(strategy_id: str = "swing_v2") -> DailyRunResult:
    snapshot = build_demo_snapshot(strategy_id=strategy_id)
    market, sectors, stocks, cross_section = build_demo_forecast_states()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    current_weights = {"000630.SZ": 0.20, "600160.SH": 0.15}
    policy_decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights=current_weights,
            current_cash=0.65,
            total_equity=1.0,
        )
    )
    trade_actions = build_trade_actions(
        decision=policy_decision,
        current_weights=current_weights,
    )
    return DailyRunResult(
        snapshot=snapshot,
        composite_state=composite_state,
        policy_decision=policy_decision,
        trade_actions=trade_actions,
    )


def summarize_daily_run(result: DailyRunResult) -> dict[str, object]:
    return {
        "strategy_id": result.snapshot.strategy_id,
        "strategy_mode": result.composite_state.strategy_mode,
        "risk_regime": result.composite_state.risk_regime,
        "market": asdict(result.composite_state.market),
        "policy": asdict(result.policy_decision),
        "trade_plan": [
            {
                "symbol": action.symbol,
                "action": action.action,
                "current_weight": action.current_weight,
                "target_weight": action.target_weight,
                "delta_weight": action.delta_weight,
                "note": action.note,
            }
            for action in result.trade_actions
        ],
    }
