from __future__ import annotations

from typing import Callable

from src.application.v2_contracts import CompositeState, InfoAggregateState, PolicyDecision, StockForecastState


def stock_reason_bundle(
    *,
    stock: StockForecastState,
    info_state: InfoAggregateState,
    state: CompositeState,
    rank: int,
    policy: PolicyDecision,
    alpha_score_components: Callable[[StockForecastState], dict[str, float]],
) -> tuple[list[str], list[str], list[str], str, str, str, str]:
    forecasts = dict(getattr(stock, "horizon_forecasts", {}))
    one_day = forecasts.get("1d")
    five_day = forecasts.get("5d")
    twenty_day = forecasts.get("20d")
    selection_reasons: list[str] = []
    if twenty_day is not None and float(twenty_day.up_prob) >= 0.60:
        selection_reasons.append(f"20日上涨概率 {twenty_day.up_prob * 100:.1f}%，中段趋势占优。")
    if five_day is not None and float(five_day.up_prob) >= 0.58:
        selection_reasons.append(f"5日上涨概率 {five_day.up_prob * 100:.1f}%，短波段延续性较好。")
    if float(stock.excess_vs_sector_prob) >= 0.55:
        selection_reasons.append(f"行业内相对强度 {stock.excess_vs_sector_prob * 100:.1f}%，位于板块前列。")
    if float(stock.tradeability_score) >= 0.80:
        selection_reasons.append(f"量价结构稳定，交易一致性 {stock.tradeability_score * 100:.1f}%。")
    if float(info_state.catalyst_strength) >= 0.55:
        selection_reasons.append(f"催化强度 {info_state.catalyst_strength * 100:.1f}%，对信号有加成。")
    if float(state.cross_section.breadth_strength) >= 0.08:
        selection_reasons.append("市场宽度配合度尚可，前排信号更容易兑现。")
    if not selection_reasons:
        selection_reasons.append("综合排序靠前，趋势、相对强弱和交易结构较均衡。")

    alpha_parts = alpha_score_components(stock)
    ranking_reasons: list[str] = []
    if float(alpha_parts.get("selection_bonus", 0.0)) > 0.08:
        ranking_reasons.append("趋势延续与稳定性加分较高。")
    if float(alpha_parts.get("sector_edge", 0.0)) > 0.03:
        ranking_reasons.append("行业内相对强度为排名提供支撑。")
    if float(alpha_parts.get("medium_edge", 0.0)) > 0.03:
        ranking_reasons.append("中期空间优于大部分候选。")
    if float(alpha_parts.get("stability_bonus", 0.0)) > 0.30:
        ranking_reasons.append("多周期信号一致，分歧不大。")
    if not ranking_reasons:
        ranking_reasons.append("综合排序稳定，处于当前候选前列。")

    risk_flags: list[str] = []
    if float(stock.up_1d_prob) > float(stock.up_5d_prob) + 0.06:
        risk_flags.append("短线偏热，次日容易先冲后分化。")
    if float(stock.tradeability_score) < 0.72:
        risk_flags.append("量价承接一般，追价性价比不高。")
    if float(info_state.negative_event_risk) >= 0.35:
        risk_flags.append("信息面负事件风险偏高。")
    if one_day is not None and float(one_day.confidence) < 0.48:
        risk_flags.append("次日预测置信度偏低，宜轻仓观察。")
    if str(stock.tradability_status) != "normal":
        risk_flags.append(f"交易状态受限: {stock.tradability_status}")
    if not risk_flags:
        risk_flags.append("当前未见明显硬性风险，但仍需服从仓位纪律。")

    invalidation = "若下一交易日"
    if one_day is not None and one_day.price_low == one_day.price_low:
        invalidation += f"收盘跌破 {one_day.price_low:.2f}"
    else:
        invalidation += "低于预期下沿"
    invalidation += "，且 5 日上涨概率回落到 50% 下方，则本次信号失效。"

    target_weight = float(policy.symbol_target_weights.get(stock.symbol, 0.0))
    desired_weight = float(policy.desired_symbol_target_weights.get(stock.symbol, target_weight))
    action_reason = ""
    weight_reason = ""
    blocked_reason = ""
    if target_weight > 0.0:
        action_reason = f"排序第 {rank}，进入当前 {policy.target_position_count} 个目标持仓。"
        seat_weight = float(policy.target_exposure / max(1, policy.target_position_count))
        weight_reason = (
            f"目标权重 {target_weight * 100:.2f}%，在总仓位 {policy.target_exposure * 100:.1f}% 下属于 "
            f"{'主仓' if target_weight >= max(0.15, seat_weight) else '辅助仓'} 配置。"
        )
        if desired_weight > target_weight + 1e-6:
            weight_reason += " 受风险约束后权重有所收缩。"
    else:
        if rank > int(policy.target_position_count):
            blocked_reason = (
                f"当前只开 {policy.target_position_count} 个仓位，这只排位靠前但未进入前 "
                f"{policy.target_position_count}。"
            )
        elif float(policy.sector_budgets.get(stock.sector, 0.0)) <= 0.0:
            blocked_reason = f"{stock.sector} 当前未分配预算，本次只保留跟踪。"
        else:
            blocked_reason = "综合排序仍然不错，但没有超过本轮实际执行门槛。"
    return (
        selection_reasons[:3],
        ranking_reasons[:3],
        risk_flags[:3],
        invalidation,
        action_reason,
        weight_reason,
        blocked_reason,
    )
