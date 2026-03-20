from __future__ import annotations

from html import escape

import pandas as pd

from src.contracts.reporting import DailyReportViewModel, ResearchReportViewModel


def _pct(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value * 100:.1f}%"


def _num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def _money(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:,.0f}"


def _intraday_signal_text(item: dict[str, object]) -> str:
    signal = str(item.get("intraday_signal", "") or "").strip()
    if not signal:
        return "NA"
    timeframe = str(item.get("intraday_timeframe", "") or "").strip()
    data_date = str(item.get("intraday_data_date", "") or "").strip()
    parts = [signal]
    if timeframe:
        parts.append(timeframe)
    if data_date:
        parts.append(data_date)
    return " | ".join(parts)


def _intraday_levels_text(item: dict[str, object]) -> str:
    stop_price = float(item.get("intraday_stop_price", float("nan")))
    take_profit_price = float(item.get("intraday_take_profit_price", float("nan")))
    parts: list[str] = []
    if not pd.isna(stop_price):
        parts.append(f"stop {_num(stop_price)}")
    if not pd.isna(take_profit_price):
        parts.append(f"tp {_num(take_profit_price)}")
    return " | ".join(parts) if parts else "NA"


def _intraday_reason_text(item: dict[str, object]) -> str:
    reason = str(item.get("intraday_reason", "") or "").strip()
    return reason or "NA"


def render_daily_markdown(view_model: DailyReportViewModel) -> str:
    metadata = view_model.metadata
    market_summary = view_model.market_summary
    info_summary = dict(view_model.info_summary)
    sentiment = dict(market_summary.get("sentiment", {}))
    facts = dict(market_summary.get("facts", {}))
    risk_notes = list(market_summary.get("risk_notes", []))
    recall = dict(view_model.memory_summary.get("recall", {}))
    market_info = dict(info_summary.get("market_info_state", {}))
    capital_flow = dict(view_model.external_signals.get("capital_flow", {}))
    macro_context = dict(view_model.external_signals.get("macro_context", {}))
    candidate_selection = dict(view_model.dynamic_universe.get("candidate_selection", {}))

    lines: list[str] = []
    lines.append("# V2 次日决策日报")
    lines.append("")
    lines.append(f"- 策略ID: {view_model.strategy_id}")
    lines.append(f"- artifact run_id: {view_model.run_id or 'NA'}")
    lines.append(f"- 股票池: {metadata.get('universe_id', 'NA')}")
    if metadata.get("universe_size"):
        lines.append(f"- 股票池规模: {metadata['universe_size']}")
    if metadata.get("generator_manifest_path"):
        lines.append(f"- generator manifest path: {metadata['generator_manifest_path']}")
    if metadata.get("generator_version"):
        lines.append(f"- generator version: {metadata['generator_version']}")
    if metadata.get("external_signal_manifest_path"):
        lines.append(f"- external signal manifest path: {metadata['external_signal_manifest_path']}")
    lines.append(f"- external signals enabled: {'true' if view_model.external_signal_enabled else 'false'}")
    if metadata.get("external_signal_version"):
        lines.append(f"- external signal version: {metadata['external_signal_version']}")
    lines.append(
        f"- US index context: {'enabled' if metadata.get('use_us_index_context') else 'disabled'}"
        f" ({metadata.get('us_index_source', 'NA')})"
    )
    lines.append(f"- 数据日期: {view_model.as_of_date}")
    lines.append(f"- 下一交易日: {view_model.next_session or 'NA'}")
    lines.append(f"- 策略模式: {view_model.strategy_mode}")
    lines.append(f"- 风险状态: {view_model.risk_regime}")
    lines.append("")

    lines.append("## 市场总览")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 情绪阶段 | {sentiment.get('stage', 'NA')} |")
    lines.append(f"| 情绪分 | {_num(float(sentiment.get('score', float('nan'))), 1)} / 100 |")
    lines.append(f"| 目标总仓位 | {_pct(float(view_model.policy.get('target_exposure', float('nan'))))} |")
    lines.append(f"| 目标持仓数 | {int(view_model.policy.get('target_position_count', 0))} |")
    lines.append(
        f"| 涨家数 / 跌家数 / 平家数 | {int(facts.get('advancers', 0))} / {int(facts.get('decliners', 0))} / {int(facts.get('flats', 0))} |"
    )
    lines.append(f"| 涨停 / 跌停 | {int(facts.get('limit_up_count', 0))} / {int(facts.get('limit_down_count', 0))} |")
    lines.append(f"| 新高 / 新低 | {int(facts.get('new_high_count', 0))} / {int(facts.get('new_low_count', 0))} |")
    lines.append(f"| 样本中位涨跌幅 | {_pct(float(facts.get('median_return', float('nan'))))} |")
    lines.append(f"| 样本覆盖数 | {int(facts.get('sample_coverage', 0))} |")
    lines.append(f"| 样本成交额 | {_money(float(facts.get('sample_amount', float('nan'))))} |")
    lines.append("")

    lines.append("## 大盘情绪")
    lines.append("")
    lines.append(f"- 情绪结论: {sentiment.get('summary', 'NA')}")
    if sentiment.get("drivers"):
        lines.append(f"- 主要驱动: {'；'.join(str(item) for item in sentiment['drivers'])}")
    if risk_notes:
        lines.append(f"- 风险提示: {'；'.join(str(item) for item in risk_notes)}")
    lines.append("")

    lines.append("## 大盘多周期预测")
    lines.append("")
    lines.append("| 周期 | 上涨概率 | 预期区间 | 中位预期 | 置信度 |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in view_model.market_forecasts:
        lines.append(
            f"| {item['label']} | {_pct(float(item['up_prob']))} | {_pct(float(item['q10']))} ~ {_pct(float(item['q90']))} | "
            f"{_pct(float(item['q50']))} | {_pct(float(item['confidence']))} |"
        )
    lines.append("")

    lines.append("## 动态股票池")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 粗排池 | {int(view_model.dynamic_universe.get('coarse_pool_size', 0))} |")
    lines.append(f"| 精排池 | {int(view_model.dynamic_universe.get('refined_pool_size', 0))} |")
    lines.append(f"| 最终动态池 | {int(view_model.dynamic_universe.get('selected_pool_size', 0))} |")
    lines.append("")
    if candidate_selection.get("selection_notes"):
        lines.append(f"- shortlist notes: {' | '.join(str(item) for item in candidate_selection['selection_notes'])}")
        lines.append("")

    if view_model.memory_summary.get("memory_path"):
        lines.append("## 策略记忆")
        lines.append("")
        lines.append(f"- memory path: {view_model.memory_summary['memory_path']}")
        if recall.get("latest_research_run_id"):
            gate_text = "通过" if recall.get("latest_research_release_gate_passed") else "未通过"
            lines.append(
                f"- 最近研究: run_id={recall.get('latest_research_run_id')}, 截止={recall.get('latest_research_end_date') or 'NA'}, "
                f"超额年化={_pct(float(recall.get('latest_research_excess_annual_return', float('nan'))))}, "
                f"IR={_num(float(recall.get('latest_research_information_ratio', float('nan'))), 2)}, release gate={gate_text}"
            )
        if recall.get("recent_daily_run_count"):
            lines.append(
                f"- 近期日运行 {int(recall.get('recent_daily_run_count', 0))} 次: 平均目标仓位={_pct(float(recall.get('average_target_exposure', float('nan'))))}, "
                f"调仓触发占比={_pct(float(recall.get('rebalance_ratio', float('nan'))))}, 仓位趋势={_pct(float(recall.get('exposure_trend', float('nan'))))}"
            )
        if recall.get("recurring_symbols"):
            lines.append(f"- 高频标的: {', '.join(str(item) for item in recall['recurring_symbols'])}")
        for item in recall.get("narrative", []):
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## 外部信号")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| 资金状态 | {capital_flow.get('flow_regime', 'NA')} |")
    lines.append(f"| 北向强度 | {_num(float(capital_flow.get('northbound_net_flow', float('nan'))), 3)} |")
    lines.append(f"| 两融变化 | {_num(float(capital_flow.get('margin_balance_change', float('nan'))), 3)} |")
    lines.append(f"| 成交热度 | {_pct(float(capital_flow.get('turnover_heat', float('nan'))))} |")
    lines.append(f"| 大单偏向 | {_num(float(capital_flow.get('large_order_bias', float('nan'))), 3)} |")
    lines.append(f"| 宏观风险 | {macro_context.get('macro_risk_level', 'NA')} |")
    lines.append(f"| 风格状态 | {macro_context.get('style_regime', 'NA')} |")
    lines.append(f"| 商品压力 | {_pct(float(macro_context.get('commodity_pressure', float('nan'))))} |")
    lines.append(f"| 汇率压力 | {_pct(float(macro_context.get('fx_pressure', float('nan'))))} |")
    lines.append(f"| 宽度代理 | {_pct(float(macro_context.get('index_breadth_proxy', float('nan'))))} |")
    lines.append("")

    lines.append("## 当日主线列表")
    lines.append("")
    lines.append("| 主线 | phase | conviction | breadth | leadership | event_risk | 说明 |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    if not view_model.theme_episodes:
        lines.append("| NA | NA | NA | NA | NA | NA | 暂无 insight 主线 |")
    else:
        for item in view_model.theme_episodes[:8]:
            lines.append(
                f"| {item.get('theme', 'NA')} | {item.get('phase', 'NA')} | {_pct(float(item.get('conviction', float('nan'))))} | "
                f"{_pct(float(item.get('breadth', float('nan'))))} | {_pct(float(item.get('leadership', float('nan'))))} | "
                f"{_pct(float(item.get('event_risk', float('nan'))))} | {item.get('phase_reason', 'NA')} |"
            )
    lines.append("")

    lines.append("## 龙头候选")
    lines.append("")
    lines.append("| 股票 | 主线 | phase | role | candidate | conviction | negative | 说明 |")
    lines.append("|---|---|---|---|---:|---:|---:|---|")
    if not view_model.leader_candidates:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | 暂无龙头候选 |")
    else:
        for item in view_model.leader_candidates[:8]:
            lines.append(
                f"| {item.get('name', 'NA')} ({item.get('symbol', 'NA')}) | {item.get('theme', 'NA')} | {item.get('theme_phase', 'NA')} | "
                f"{item.get('role', 'NA')} | {_pct(float(item.get('candidate_score', float('nan'))))} | "
                f"{_pct(float(item.get('conviction_score', float('nan'))))} | {_pct(float(item.get('negative_score', float('nan'))))} | "
                f"{' | '.join(str(v) for v in item.get('reasons', [])[:3]) or 'NA'} |"
            )
    lines.append("")

    lines.append("## 预测监控榜")
    lines.append("")
    lines.append("| 排名 | 股票 | 行业 | 下一交易日区间 | 5日中位预期 | 20日中位预期 | 1日上涨概率 | 可执行分 | 状态 |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
    if not view_model.monitor_recommendations:
        lines.append("| 1 | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for item in view_model.monitor_recommendations:
            lines.append(
                f"| {int(item['rank'])} | {item['name']} ({item['symbol']}) | {item['sector']} | {item['next_session_range']} | "
                f"{_pct(float(item['median_5d']))} | {_pct(float(item['median_20d']))} | {_pct(float(item['up_prob_1d']))} | "
                f"{_pct(float(item['actionability_score']))} | {item['execution_status']} |"
            )
    lines.append("")

    lines.append("## 可执行候选榜")
    lines.append("")
    lines.append("| 排名 | 股票 | 行业 | 可执行分 | 目标权重 | 下一交易日区间 | 1日上涨概率 | 备注 |")
    lines.append("|---:|---|---|---:|---:|---|---:|---|")
    if not view_model.actionable_recommendations:
        lines.append("| 1 | NA | NA | NA | NA | NA | NA | 当前无新开仓候选 |")
    else:
        for item in view_model.actionable_recommendations:
            lines.append(
                f"| {int(item['rank'])} | {item['name']} ({item['symbol']}) | {item['sector']} | {_pct(float(item['actionability_score']))} | "
                f"{_pct(float(item['target_weight']))} | {item['next_session_range']} | {_pct(float(item['up_prob_1d']))} | {item['execution_status']} |"
            )
    lines.append("")

    lines.append("## 实际操作")
    lines.append("")
    lines.append("| 股票 | 动作 | 当前权重 | 目标权重 | 权重变化 | 操作理由 |")
    lines.append("|---|---|---:|---:|---:|---|")
    if not view_model.trade_actions:
        lines.append("| 无 | HOLD | NA | NA | NA | 当前不触发调仓 |")
    else:
        for item in view_model.trade_actions:
            lines.append(
                f"| {item['name']} ({item['symbol']}) | {item['action']} | {_pct(float(item['current_weight']))} | {_pct(float(item['target_weight']))} | "
                f"{_pct(float(item['delta_weight']))} | {item['reason'] or 'NA'} |"
            )
    lines.append("")

    lines.append("## 持仓角色变化")
    lines.append("")
    lines.append("| 股票 | 主线 | 当前角色 | 前序角色 | 角色降级 | 备注 |")
    lines.append("|---|---|---|---|---|---|")
    if not view_model.holding_role_changes:
        lines.append("| NA | NA | NA | NA | NA | 暂无角色变化 |")
    else:
        for item in view_model.holding_role_changes:
            lines.append(
                f"| {item['name']} ({item['symbol']}) | {item['theme'] or 'NA'} | {item['role'] or 'NA'} | "
                f"{item['previous_role'] or 'NA'} | {'yes' if item['role_downgrade'] else 'no'} | {item['note'] or 'NA'} |"
            )
    lines.append("")

    lines.append("## 次日执行计划")
    lines.append("")
    lines.append("| 股票 | bias | buy_zone | avoid_zone | intraday | levels | reduce_if | exit_if | reason |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    if not view_model.execution_plans:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | 暂无执行计划 |")
    else:
        for item in view_model.execution_plans:
            lines.append(
                f"| {item.get('name', 'NA')} ({item.get('symbol', 'NA')}) | {item.get('bias', 'NA')} | {item.get('buy_zone', 'NA')} | "
                f"{item.get('avoid_zone', 'NA')} | {_intraday_signal_text(item)} | {_intraday_levels_text(item)} | "
                f"{item.get('reduce_if', 'NA')} | {item.get('exit_if', 'NA')} | {item.get('reason', 'NA')} |"
            )
    lines.append("")
    lines.append("## Intraday Overlay")
    lines.append("")
    intraday_items = [item for item in view_model.execution_plans if str(item.get("intraday_signal", "")).strip()]
    if not intraday_items:
        lines.append("- 暂无分钟级执行覆盖")
    else:
        for item in intraday_items:
            lines.append(
                f"- {item.get('name', 'NA')} ({item.get('symbol', 'NA')}): {_intraday_signal_text(item)} | "
                f"{_intraday_levels_text(item)} | {_intraday_reason_text(item)}"
            )
    lines.append("")

    lines.append("## 推荐解释卡")
    lines.append("")
    for item in view_model.explanation_cards[:5]:
        lines.append(f"### {item['name']} ({item['symbol']})")
        lines.append(f"- 行业: {item['sector']}")
        if not pd.isna(float(item["latest_close"])):
            lines.append(f"- 最新收盘价: {_num(float(item['latest_close']), 2)}")
        lines.append(
            f"- 下一交易日({view_model.next_session or 'NA'})预期区间: {item['next_session_range']}，中位预期 {_pct(float(item['one_day_median']))}，"
            f"上涨概率 {_pct(float(item['one_day_up_prob']))}，置信度 {_pct(float(item['one_day_confidence']))}"
        )
        lines.append(
            f"- 多周期判断: 5日 {_pct(float(item['median_5d']))} / 10日 {_pct(float(item['median_10d']))} / 20日 {_pct(float(item['median_20d']))}"
        )
        lines.append(
            f"- 可执行状态: {item['execution_status']}，可执行分 {_pct(float(item['actionability_score']))}，备注: {item['gate_reason']}"
        )
        if item["selection_reasons"]:
            lines.append(f"- 入池原因: {'；'.join(str(v) for v in item['selection_reasons'])}")
        if item["ranking_reasons"]:
            lines.append(f"- 排名原因: {'；'.join(str(v) for v in item['ranking_reasons'])}")
        if item["selected"] and item["action_reason"]:
            lines.append(f"- 操作原因: {item['action_reason']}")
        elif item["blocked_reason"]:
            lines.append(f"- 未入组合原因: {item['blocked_reason']}")
        if item["weight_reason"]:
            lines.append(f"- 仓位说明: {item['weight_reason']}")
        if item["risk_flags"]:
            lines.append(f"- 风险点: {'；'.join(str(v) for v in item['risk_flags'])}")
        if item["invalidation_rule"]:
            lines.append(f"- 失效条件: {item['invalidation_rule']}")
        lines.append("")

    lines.append("## 预测复盘")
    lines.append("")
    lines.append("| 窗口 | 命中参考 | 平均边际 | 近窗表现 | 样本数 | 说明 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    if not view_model.prediction_review.get("windows"):
        lines.append("| 近窗 | NA | NA | NA | 0 | 暂无复盘数据 |")
    else:
        for item in view_model.prediction_review["windows"]:
            lines.append(
                f"| {item['label']} | {_pct(float(item['hit_rate']))} | {_pct(float(item['avg_edge']))} | {_pct(float(item['realized_return']))} | "
                f"{int(item['sample_size'])} | {item['note'] or 'NA'} |"
            )
    for note in view_model.prediction_review.get("notes", []):
        lines.append(f"- {note}")
    if info_summary:
        lines.append("")
        lines.append("## 淇℃伅灞傛憳瑕?")
        lines.append("")
        lines.append("| 鎸囨爣 | 鏁板€?|")
        lines.append("|---|---:|")
        lines.append(f"| info shadow enabled | {'true' if info_summary.get('shadow_enabled') else 'false'} |")
        lines.append(f"| info items | {int(info_summary.get('item_count', 0))} |")
        lines.append(f"| market catalyst | {_pct(float(market_info.get('catalyst_strength', float('nan'))))} |")
        lines.append(f"| market event risk | {_pct(float(market_info.get('event_risk_level', float('nan'))))} |")
        lines.append(f"| market negative risk | {_pct(float(market_info.get('negative_event_risk', float('nan'))))} |")
        lines.append(f"| market coverage | {_pct(float(market_info.get('coverage_confidence', float('nan'))))} |")
        for item in info_summary.get("top_negative_events", []):
            lines.append(
                f"- negative: {item.get('target_name') or item.get('target')} | {item.get('title')} | "
                f"risk={_pct(float(item.get('negative_event_risk', float('nan'))))}"
            )
        for item in info_summary.get("top_positive_signals", []):
            lines.append(
                f"- positive: {item.get('target_name') or item.get('target')} | {item.get('title')} | "
                f"score={_num(float(item.get('score', float('nan'))), 2)}"
            )
        for item in info_summary.get("quant_info_divergence", []):
            lines.append(
                f"- divergence: {item.get('name') or item.get('symbol')} | "
                f"quant20={_pct(float(item.get('quant_prob_20d', float('nan'))))} | "
                f"shadow20={_pct(float(item.get('shadow_prob_20d', float('nan'))))} | "
                f"gap={_pct(float(item.get('gap', float('nan'))))}"
            )
    return "\n".join(lines)


def render_research_markdown(view_model: ResearchReportViewModel) -> str:
    lines: list[str] = []
    lines.append("# V2 研究回测报告")
    lines.append("")
    lines.append(f"- 策略ID: {view_model.strategy_id}")
    lines.append(f"- run_id: {view_model.run_id or 'NA'}")
    lines.append(f"- release gate: {'passed' if view_model.release_gate_passed else 'pending'}")
    lines.append(f"- best_score: {_num(float(view_model.calibration.get('best_score', 0.0)), 4)}")
    lines.append("")
    lines.append("## 核心指标")
    lines.append("")
    lines.append("| 指标 | baseline | calibrated | learned |")
    lines.append("|---|---:|---:|---:|")
    for item in view_model.comparison_metrics:
        is_ratio = "ratio" in str(item["label"])
        baseline_text = _num(float(item["baseline"]), 3) if is_ratio else _pct(float(item["baseline"]))
        calibrated_text = _num(float(item["calibrated"]), 3) if is_ratio else _pct(float(item["calibrated"]))
        learned_text = _num(float(item["learned"]), 3) if is_ratio else _pct(float(item["learned"]))
        lines.append(f"| {item['title']} | {baseline_text} | {calibrated_text} | {learned_text} |")
    lines.append("")
    lines.append("## Horizon Metrics")
    lines.append("")
    lines.append("| 方案 | 周期 | RankIC | 头部分层收益 | 头尾价差 | TopK命中率 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for item in view_model.horizon_metrics:
        lines.append(
            f"| {item['stage']} | {item['horizon']} | {_num(float(item['rank_ic']), 3)} | {_pct(float(item['top_decile_return']))} | "
            f"{_pct(float(item['top_bottom_spread']))} | {_pct(float(item['top_k_hit_rate']))} |"
        )
    lines.append("")
    lines.append("## Validation Trials")
    lines.append("")
    lines.append("| 排名 | risk_on_exposure | risk_on_positions | risk_on_turnover_cap | annual_return | benchmark_annual_return | excess_annual_return | information_ratio | max_drawdown | score |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    if not view_model.calibration_trials:
        lines.append("| 1 | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for item in view_model.calibration_trials:
            policy = dict(item.get("policy", {}))
            summary = dict(item.get("summary", {}))
            lines.append(
                f"| {int(item['rank'])} | {_pct(float(policy.get('risk_on_exposure', 0.0)))} | {int(policy.get('risk_on_positions', 0))} | "
                f"{_pct(float(policy.get('risk_on_turnover_cap', 0.0)))} | {_pct(float(summary.get('annual_return', 0.0)))} | "
                f"{_pct(float(summary.get('benchmark_annual_return', 0.0)))} | {_pct(float(summary.get('excess_annual_return', 0.0)))} | "
                f"{_num(float(summary.get('information_ratio', 0.0)), 3)} | {_pct(float(summary.get('max_drawdown', 0.0)))} | {_num(float(item['score']), 4)} |"
            )
    lines.append("")
    lines.append("## Learned Policy")
    lines.append("")
    lines.append(f"- feature_names: {', '.join(str(item) for item in view_model.learning_model.get('feature_names', [])) or 'NA'}")
    lines.append(f"- train_rows: {int(view_model.learning_model.get('train_rows', 0))}")
    lines.append(f"- train_r2_exposure: {_num(float(view_model.learning_model.get('train_r2_exposure', 0.0)), 3)}")
    lines.append(f"- train_r2_positions: {_num(float(view_model.learning_model.get('train_r2_positions', 0.0)), 3)}")
    lines.append(f"- train_r2_turnover: {_num(float(view_model.learning_model.get('train_r2_turnover', 0.0)), 3)}")
    lines.append("")
    lines.append("## 主线生命周期摘要")
    lines.append("")
    lines.append(f"- phase_counts: {view_model.theme_lifecycle_summary.get('phase_counts', {})}")
    for item in view_model.theme_lifecycle_summary.get("top_themes", [])[:6]:
        lines.append(
            f"- {item.get('theme', 'NA')}: {item.get('phase', 'NA')} | conviction={_pct(float(item.get('conviction', float('nan'))))} | "
            f"event_risk={_pct(float(item.get('event_risk', float('nan'))))}"
        )
    lines.append("")
    lines.append("## 龙头识别评估")
    lines.append("")
    leader_eval = dict(view_model.leader_summary.get("evaluation", {}))
    lines.append(
        f"- candidate_recall_at_k: {_pct(float(leader_eval.get('candidate_recall_at_k', 0.0)))}"
        f" | conviction_precision_at_1: {_pct(float(leader_eval.get('conviction_precision_at_1', 0.0)))}"
        f" | ndcg_at_k: {_num(float(leader_eval.get('ndcg_at_k', 0.0)), 3)}"
    )
    lines.append(
        f"- hard_negative_survival_recall: {_pct(float(leader_eval.get('hard_negative_survival_recall', 0.0)))}"
        f" | hard_negative_filter_rate: {_pct(float(leader_eval.get('hard_negative_filter_rate', 0.0)))}"
        f" | theme_groups: {int(leader_eval.get('theme_group_count', 0))}"
    )
    for item in view_model.leader_summary.get("top_candidates", [])[:6]:
        lines.append(
            f"- {item.get('symbol', 'NA')}: {item.get('theme', 'NA')} / {item.get('role', 'NA')} | "
            f"candidate={_pct(float(item.get('candidate_score', float('nan'))))} | "
            f"conviction={_pct(float(item.get('conviction_score', float('nan'))))}"
        )
    lines.append("")
    lines.append("## 角色分布摘要")
    lines.append("")
    lines.append(f"- role_counts: {view_model.role_distribution_summary.get('role_counts', {})}")
    for item in view_model.role_distribution_summary.get("top_roles", [])[:8]:
        lines.append(
            f"- {item.get('symbol', 'NA')}: {item.get('theme', 'NA')} / {item.get('role', 'NA')} | downgrade={item.get('role_downgrade', False)}"
        )
    lines.append("")
    lines.append("## 持仓退出决策贡献摘要")
    lines.append("")
    lines.append(
        f"- fading_theme_count: {int(view_model.exit_contribution_summary.get('fading_theme_count', 0))} | "
        f"crowded_theme_count: {int(view_model.exit_contribution_summary.get('crowded_theme_count', 0))} | "
        f"role_downgrade_count: {int(view_model.exit_contribution_summary.get('role_downgrade_count', 0))}"
    )
    for item in view_model.exit_contribution_summary.get("examples", [])[:6]:
        lines.append(
            f"- {item.get('symbol', 'NA')}: {item.get('theme', 'NA')} / {item.get('role', 'NA')} | note={item.get('note', 'NA')}"
        )
    lines.append("")
    if view_model.artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for key, value in view_model.artifacts.items():
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def render_daily_html(view_model: DailyReportViewModel) -> str:
    metadata = view_model.metadata
    sentiment = dict(view_model.market_summary.get("sentiment", {}))
    facts = dict(view_model.market_summary.get("facts", {}))
    info_summary = dict(view_model.info_summary)
    market_info = dict(info_summary.get("market_info_state", {}))
    candidate_selection = dict(view_model.dynamic_universe.get("candidate_selection", {}))
    selection_notes = " | ".join(str(item) for item in candidate_selection.get("selection_notes", [])[:2]) or "Full ranking in use."
    shortlist_text = (
        f"{int(candidate_selection.get('shortlist_size', 0))}/{int(candidate_selection.get('total_scored', 0))} shortlisted"
        if candidate_selection.get("shortlist_size") and candidate_selection.get("total_scored")
        else f"{len(view_model.monitor_recommendations)} names scored"
    )
    recurring_symbols = ", ".join(str(item) for item in view_model.memory_summary.get("recall", {}).get("recurring_symbols", [])[:4]) or "NA"
    capital_flow = dict(view_model.external_signals.get("capital_flow", {}))
    macro_context = dict(view_model.external_signals.get("macro_context", {}))
    market_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item['label']))}</td>"
        f"<td>{_pct(float(item['up_prob']))}</td>"
        f"<td>{_pct(float(item['q10']))} ~ {_pct(float(item['q90']))}</td>"
        f"<td>{_pct(float(item['q50']))}</td>"
        f"<td>{_pct(float(item['confidence']))}</td>"
        "</tr>"
        for item in view_model.market_forecasts
    )
    monitor_rows = "".join(
        "<tr>"
        f"<td>{int(item['rank'])}</td>"
        f"<td><div class='ticker'>{escape(str(item['name']))}</div><div class='ticker-sub'>{escape(str(item['symbol']))}</div></td>"
        f"<td>{escape(str(item['sector']))}</td>"
        f"<td>{escape(str(item['next_session_range']))}</td>"
        f"<td>{_pct(float(item['median_5d']))}</td>"
        f"<td>{_pct(float(item['median_20d']))}</td>"
        f"<td>{_pct(float(item['up_prob_1d']))}</td>"
        f"<td>{_pct(float(item['actionability_score']))}</td>"
        f"<td>{escape(str(item['execution_status']))}</td>"
        "</tr>"
        for item in view_model.monitor_recommendations
    )
    actionable_rows = "".join(
        "<tr>"
        f"<td>{int(item['rank'])}</td>"
        f"<td><div class='ticker'>{escape(str(item['name']))}</div><div class='ticker-sub'>{escape(str(item['symbol']))}</div></td>"
        f"<td>{escape(str(item['sector']))}</td>"
        f"<td>{_pct(float(item['actionability_score']))}</td>"
        f"<td>{_pct(float(item['target_weight']))}</td>"
        f"<td>{escape(str(item['next_session_range']))}</td>"
        f"<td>{_pct(float(item['up_prob_1d']))}</td>"
        f"<td>{escape(str(item['execution_status']))}</td>"
        "</tr>"
        for item in view_model.actionable_recommendations
    )
    action_rows = "".join(
        "<tr>"
        f"<td><div class='ticker'>{escape(str(item['name']))}</div><div class='ticker-sub'>{escape(str(item['symbol']))}</div></td>"
        f"<td>{escape(str(item['action']))}</td>"
        f"<td>{_pct(float(item['current_weight']))}</td>"
        f"<td>{_pct(float(item['target_weight']))}</td>"
        f"<td>{_pct(float(item['delta_weight']))}</td>"
        f"<td>{escape(str(item['reason'] or 'NA'))}</td>"
        "</tr>"
        for item in view_model.trade_actions
    )
    theme_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('theme', 'NA')))}</td>"
        f"<td>{escape(str(item.get('phase', 'NA')))}</td>"
        f"<td>{_pct(float(item.get('conviction', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('breadth', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('leadership', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('event_risk', float('nan'))))}</td>"
        f"<td>{escape(str(item.get('phase_reason', 'NA')))}</td>"
        "</tr>"
        for item in view_model.theme_episodes[:8]
    )
    leader_candidate_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('name', 'NA')))} ({escape(str(item.get('symbol', 'NA')) )})</td>"
        f"<td>{escape(str(item.get('theme', 'NA')))}</td>"
        f"<td>{escape(str(item.get('theme_phase', 'NA')))}</td>"
        f"<td>{escape(str(item.get('role', 'NA')))}</td>"
        f"<td>{_pct(float(item.get('candidate_score', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('conviction_score', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('negative_score', float('nan'))))}</td>"
        f"<td>{escape(' | '.join(str(v) for v in item.get('reasons', [])[:3]) or 'NA')}</td>"
        "</tr>"
        for item in view_model.leader_candidates[:8]
    )
    role_change_rows = "".join(
        "<tr>"
        f"<td><div class='ticker'>{escape(str(item['name']))}</div><div class='ticker-sub'>{escape(str(item['symbol']))}</div></td>"
        f"<td>{escape(str(item['theme'] or 'NA'))}</td>"
        f"<td>{escape(str(item['role'] or 'NA'))}</td>"
        f"<td>{escape(str(item['previous_role'] or 'NA'))}</td>"
        f"<td>{'yes' if item['role_downgrade'] else 'no'}</td>"
        f"<td>{escape(str(item['note'] or 'NA'))}</td>"
        "</tr>"
        for item in view_model.holding_role_changes
    )
    execution_plan_rows = "".join(
        "<tr>"
        f"<td><div class='ticker'>{escape(str(item.get('name', 'NA')))}</div><div class='ticker-sub'>{escape(str(item.get('symbol', 'NA')))}</div></td>"
        f"<td>{escape(str(item.get('bias', 'NA')))}</td>"
        f"<td>{escape(str(item.get('buy_zone', 'NA')))}</td>"
        f"<td>{escape(str(item.get('avoid_zone', 'NA')))}</td>"
        f"<td>{escape(_intraday_signal_text(item))}</td>"
        f"<td>{escape(_intraday_levels_text(item))}</td>"
        f"<td>{escape(str(item.get('reduce_if', 'NA')))}</td>"
        f"<td>{escape(str(item.get('exit_if', 'NA')))}</td>"
        f"<td>{escape(str(item.get('reason', 'NA')))}</td>"
        "</tr>"
        for item in view_model.execution_plans
    )
    intraday_plan_cards = "".join(
        "<article class='stock-card'>"
        f"<h3>{escape(str(item.get('name', 'NA')))} <span>{escape(str(item.get('symbol', 'NA')))}</span></h3>"
        f"<p class='meta'>{escape(_intraday_signal_text(item))} | {escape(_intraday_levels_text(item))}</p>"
        f"<p><strong>盘中说明</strong>: {escape(_intraday_reason_text(item))}</p>"
        f"<p><strong>Reduce If</strong>: {escape(str(item.get('reduce_if', 'NA')))}</p>"
        f"<p><strong>Exit If</strong>: {escape(str(item.get('exit_if', 'NA')))}</p>"
        "</article>"
        for item in view_model.execution_plans
        if str(item.get("intraday_signal", "")).strip()
    )
    cards_html = "".join(
        "<article class='stock-card'>"
        f"<h3>{escape(str(item['name']))} <span>{escape(str(item['symbol']))}</span></h3>"
        f"<p class='meta'>{escape(str(item['sector']))} | 最新收盘 {(_num(float(item['latest_close'])) if not pd.isna(float(item['latest_close'])) else 'NA')}</p>"
        f"<p><strong>下一交易日({escape(view_model.next_session or 'NA')})</strong>: {escape(str(item['next_session_range']))}，上涨概率 {_pct(float(item['one_day_up_prob']))}，置信度 {_pct(float(item['one_day_confidence']))}</p>"
        f"<p><strong>入池原因</strong>: {escape('；'.join(str(v) for v in item['selection_reasons']) or '综合排序靠前')}</p>"
        f"<p><strong>排名原因</strong>: {escape('；'.join(str(v) for v in item['ranking_reasons']) or '综合排序稳定')}</p>"
        f"<p><strong>可执行状态</strong>: {escape(str(item['execution_status']))} | 可执行分 {_pct(float(item['actionability_score']))} | {escape(str(item['gate_reason']))}</p>"
        f"<p><strong>{'操作原因' if item['selected'] else '未入组合原因'}</strong>: {escape(str(item['action_reason'] if item['selected'] else item['blocked_reason'] or item['weight_reason'] or 'NA'))}</p>"
        f"<p><strong>风险点</strong>: {escape('；'.join(str(v) for v in item['risk_flags']) or '暂无显著硬风险')}</p>"
        f"<p><strong>失效条件</strong>: {escape(str(item['invalidation_rule'] or '跌破预期下沿且5日概率转弱'))}</p>"
        "</article>"
        for item in view_model.explanation_cards[:6]
    )
    mainline_rows = "".join(
        "<li>"
        f"{escape(str(item.get('name', 'NA')))} | conviction {_pct(float(item.get('conviction', float('nan'))))} | sectors {escape(', '.join(str(v) for v in item.get('sectors', [])[:2]))}"
        "</li>"
        for item in view_model.mainlines[:4]
    )
    review_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item['label']))}</td>"
        f"<td>{_pct(float(item['hit_rate']))}</td>"
        f"<td>{_pct(float(item['avg_edge']))}</td>"
        f"<td>{_pct(float(item['realized_return']))}</td>"
        f"<td>{int(item['sample_size'])}</td>"
        f"<td>{escape(str(item['note'] or 'NA'))}</td>"
        "</tr>"
        for item in view_model.prediction_review.get("windows", [])
    )
    info_rows = "".join(
        "<tr>"
        f"<td>{escape(str(label))}</td>"
        f"<td>{escape(str(value))}</td>"
        "</tr>"
        for label, value in [
            ("info shadow enabled", "true" if info_summary.get("shadow_enabled") else "false"),
            ("info items", int(info_summary.get("item_count", 0))),
            ("market catalyst", _pct(float(market_info.get("catalyst_strength", float("nan"))))),
            ("market event risk", _pct(float(market_info.get("event_risk_level", float("nan"))))),
            ("market negative risk", _pct(float(market_info.get("negative_event_risk", float("nan"))))),
            ("market coverage", _pct(float(market_info.get("coverage_confidence", float("nan"))))),
        ]
    )
    info_note_rows = "".join(
        f"<li>negative | {escape(str(item.get('target_name') or item.get('target')))} | {escape(str(item.get('title')))} | risk {escape(_pct(float(item.get('negative_event_risk', float('nan')))))}</li>"
        for item in info_summary.get("top_negative_events", [])
    )
    info_note_rows += "".join(
        f"<li>positive | {escape(str(item.get('target_name') or item.get('target')))} | {escape(str(item.get('title')))} | score {escape(_num(float(item.get('score', float('nan'))), 2))}</li>"
        for item in info_summary.get("top_positive_signals", [])
    )
    info_note_rows += "".join(
        f"<li>divergence | {escape(str(item.get('name') or item.get('symbol')))} | quant20 {escape(_pct(float(item.get('quant_prob_20d', float('nan')))))} | shadow20 {escape(_pct(float(item.get('shadow_prob_20d', float('nan')))))} | gap {escape(_pct(float(item.get('gap', float('nan')))))}</li>"
        for item in info_summary.get("quant_info_divergence", [])
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>次日决策面板</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf2;
      --ink: #1e2730;
      --muted: #6f766d;
      --line: #ddd1bc;
      --shadow: 0 18px 46px rgba(45, 33, 12, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); font-family: "Avenir Next", "PingFang SC", "Microsoft YaHei", sans-serif; background: radial-gradient(circle at top right, rgba(197, 91, 53, 0.10), transparent 26%), radial-gradient(circle at left top, rgba(31, 111, 120, 0.10), transparent 24%), linear-gradient(180deg, #faf5ec 0%, var(--bg) 100%); }}
    .page {{ max-width: 1380px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 24px; box-shadow: var(--shadow); }}
    .hero {{ padding: 24px; }}
    .hero-grid, .triple, .stock-grid {{ display: grid; gap: 16px; }}
    .hero-grid {{ grid-template-columns: 1.2fr 1fr; align-items: start; }}
    .triple {{ grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 18px; }}
    .stock-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 18px; }}
    .panel {{ padding: 18px; margin-top: 18px; }}
    h1 {{ margin: 0; font-size: 36px; line-height: 1; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; }}
    h3 span {{ color: var(--muted); font-size: 12px; font-weight: 600; }}
    p, li {{ line-height: 1.6; }}
    .muted {{ color: var(--muted); }}
    .facts {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    .fact {{ padding: 14px; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,0.52); }}
    .fact .eyebrow {{ font-size: 11px; text-transform: uppercase; color: var(--muted); letter-spacing: .08em; }}
    .fact strong {{ display: block; margin-top: 8px; font-size: 28px; }}
    .pill {{ display: inline-flex; padding: 7px 12px; border-radius: 999px; border: 1px solid rgba(0,0,0,.08); background: rgba(255,255,255,.72); margin-right: 8px; font-size: 12px; font-weight: 700; }}
    .list {{ margin: 10px 0 0; padding-left: 18px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #e9dece; font-size: 13px; text-align: left; vertical-align: top; }}
    th {{ background: #faf5eb; font-size: 11px; text-transform: uppercase; color: #67706a; letter-spacing: .06em; }}
    .table-wrap {{ overflow: auto; border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.55); }}
    .ticker {{ font-weight: 800; }}
    .ticker-sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}
    .stock-card {{ border: 1px solid var(--line); border-radius: 18px; background: rgba(255,255,255,.62); padding: 16px; }}
    .stock-card .meta {{ color: var(--muted); margin-top: 0; }}
    @media (max-width: 1080px) {{ .hero-grid, .triple, .stock-grid, .facts {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div class="hero-grid">
        <div>
          <h1>次日决策面板</h1>
          <p class="muted">数据日期 {escape(view_model.as_of_date)}，下一交易日 {escape(view_model.next_session or 'NA')}。当前为 {escape(view_model.strategy_mode)} / {escape(view_model.risk_regime)}。</p>
          <p class="muted">strategy {escape(view_model.strategy_id)} | universe {escape(str(metadata.get('universe_id', 'NA')))} | generator {escape(str(metadata.get('generator_version', 'NA')))} | source {escape(str(metadata.get('us_index_source', 'NA')))}</p>
          <p class="muted">{escape(shortlist_text)} | {escape(selection_notes)}</p>
          <div>
            <span class="pill">情绪阶段: {escape(str(sentiment.get('stage', 'NA')))}</span>
            <span class="pill">情绪分: {_num(float(sentiment.get('score', float('nan'))), 1)}/100</span>
            <span class="pill">目标仓位: {_pct(float(view_model.policy.get('target_exposure', float('nan'))))}</span>
            <span class="pill">目标持仓: {int(view_model.policy.get('target_position_count', 0))}</span>
          </div>
          <ul class="list">{''.join(f"<li>{escape(str(item))}</li>" for item in sentiment.get('drivers', [])[:4])}</ul>
        </div>
        <div class="facts">
          <div class="fact"><div class="eyebrow">涨跌家数</div><strong>{int(facts.get('advancers', 0))}/{int(facts.get('decliners', 0))}</strong><div class="muted">平家数 {int(facts.get('flats', 0))}</div></div>
          <div class="fact"><div class="eyebrow">涨停 / 跌停</div><strong>{int(facts.get('limit_up_count', 0))}/{int(facts.get('limit_down_count', 0))}</strong><div class="muted">新高/新低 {int(facts.get('new_high_count', 0))}/{int(facts.get('new_low_count', 0))}</div></div>
          <div class="fact"><div class="eyebrow">样本中位涨跌幅</div><strong>{_pct(float(facts.get('median_return', float('nan'))))}</strong><div class="muted">样本覆盖 {int(facts.get('sample_coverage', 0))}</div></div>
        </div>
      </div>
      <div class="triple">
        <div class="panel"><h2>大盘情绪</h2><p>{escape(str(sentiment.get('summary', 'NA')))}</p><p class="muted">样本成交额 {_money(float(facts.get('sample_amount', float('nan'))))}，成交热度 {_pct(float(capital_flow.get('turnover_heat', float('nan'))))}，两融变化 {_num(float(capital_flow.get('margin_balance_change', float('nan'))), 3)}</p></div>
        <div class="panel"><h2>外部信号</h2><p>资金状态 {escape(str(capital_flow.get('flow_regime', 'NA')))}，北向强度 {_num(float(capital_flow.get('northbound_net_flow', float('nan'))), 3)}，大单偏向 {_num(float(capital_flow.get('large_order_bias', float('nan'))), 3)}</p><p class="muted">宏观风险 {escape(str(macro_context.get('macro_risk_level', 'NA')))}，风格 {escape(str(macro_context.get('style_regime', 'NA')))}</p></div>
        <div class="panel"><h2>策略记忆</h2><p>{escape('；'.join(str(item) for item in view_model.memory_summary.get('recall', {}).get('narrative', [])[:2]) or '暂无策略记忆摘要')}</p><p class="muted">高频标的: {escape(recurring_symbols)}</p></div>
      </div>
    </section>
    <section class="panel"><h2>大盘多周期预测</h2><div class="table-wrap"><table><thead><tr><th>周期</th><th>上涨概率</th><th>预期区间</th><th>中位预期</th><th>置信度</th></tr></thead><tbody>{market_rows}</tbody></table></div></section>
    <section class="panel"><h2>Dynamic Universe Funnel</h2><p class="muted">coarse {int(view_model.dynamic_universe.get('coarse_pool_size', 0))} | refined {int(view_model.dynamic_universe.get('refined_pool_size', 0))} | selected {int(view_model.dynamic_universe.get('selected_pool_size', 0))}</p><p class="muted">{escape(str(view_model.dynamic_universe.get('generator_version', 'NA')))} | {escape(shortlist_text)}</p><p class="muted">{escape(selection_notes)}</p></section>
    <section class="panel"><h2>当日主线列表</h2><div class="table-wrap"><table><thead><tr><th>主线</th><th>phase</th><th>conviction</th><th>breadth</th><th>leadership</th><th>event_risk</th><th>说明</th></tr></thead><tbody>{theme_rows or "<tr><td colspan='7'>暂无 insight 主线</td></tr>"}</tbody></table></div></section>
    <section class="panel"><h2>龙头候选</h2><div class="table-wrap"><table><thead><tr><th>股票</th><th>主线</th><th>phase</th><th>role</th><th>candidate</th><th>conviction</th><th>negative</th><th>说明</th></tr></thead><tbody>{leader_candidate_rows or "<tr><td colspan='8'>暂无龙头候选</td></tr>"}</tbody></table></div></section>
    <section class="panel"><h2>预测监控榜</h2><div class="table-wrap"><table><thead><tr><th>排名</th><th>股票</th><th>行业</th><th>下一交易日区间</th><th>5日中位预期</th><th>20日中位预期</th><th>1日上涨概率</th><th>可执行分</th><th>状态</th></tr></thead><tbody>{monitor_rows or "<tr><td colspan='9'>暂无监控候选</td></tr>"}</tbody></table></div></section>
    <section class="panel"><h2>可执行候选榜</h2><div class="table-wrap"><table><thead><tr><th>排名</th><th>股票</th><th>行业</th><th>可执行分</th><th>目标权重</th><th>下一交易日区间</th><th>1日上涨概率</th><th>备注</th></tr></thead><tbody>{actionable_rows or "<tr><td colspan='8'>当前无新开仓候选</td></tr>"}</tbody></table></div></section>
    <section class="panel"><h2>实际操作</h2><div class="table-wrap"><table><thead><tr><th>股票</th><th>动作</th><th>当前权重</th><th>目标权重</th><th>权重变化</th><th>操作理由</th></tr></thead><tbody>{action_rows or "<tr><td colspan='6'>当前不触发调仓</td></tr>"}</tbody></table></div></section>
    <section class="panel"><h2>持仓角色变化</h2><div class="table-wrap"><table><thead><tr><th>股票</th><th>主线</th><th>当前角色</th><th>前序角色</th><th>角色降级</th><th>备注</th></tr></thead><tbody>{role_change_rows or "<tr><td colspan='6'>暂无角色变化</td></tr>"}</tbody></table></div></section>
    <section class="panel"><h2>次日执行计划</h2><div class="table-wrap"><table><thead><tr><th>股票</th><th>bias</th><th>buy_zone</th><th>avoid_zone</th><th>intraday</th><th>levels</th><th>reduce_if</th><th>exit_if</th><th>reason</th></tr></thead><tbody>{execution_plan_rows or "<tr><td colspan='9'>暂无执行计划</td></tr>"}</tbody></table></div></section>
    <section class="panel"><h2>盘中执行覆盖</h2><div class="stock-grid">{intraday_plan_cards or "<div class='stock-card'>暂无分钟级执行覆盖</div>"}</div></section>
    <section class="stock-grid">{cards_html or "<div class='panel'>暂无解释卡</div>"}</section>
    <section class="panel"><h2>Mainline Radar</h2><ul class="list">{mainline_rows or "<li>暂无主线</li>"}</ul></section>
    <section class="panel"><h2>预测复盘</h2><div class="table-wrap"><table><thead><tr><th>窗口</th><th>命中参考</th><th>平均边际</th><th>近窗表现</th><th>样本数</th><th>说明</th></tr></thead><tbody>{review_rows or "<tr><td colspan='6'>暂无复盘数据</td></tr>"}</tbody></table></div><p class="muted">{escape('；'.join(str(item) for item in view_model.prediction_review.get('notes', [])))}</p></section>
    <section class="panel"><h2>淇℃伅灞傛憳瑕?</h2><div class="table-wrap"><table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>{info_rows}</tbody></table></div><ul class="list">{info_note_rows or "<li>NA</li>"}</ul></section>
  </main>
</body>
</html>
"""


def render_research_html(view_model: ResearchReportViewModel) -> str:
    comparison_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item['title']))}</td>"
        f"<td>{(_num(float(item['baseline']), 3) if 'ratio' in str(item['label']) else _pct(float(item['baseline'])))}</td>"
        f"<td>{(_num(float(item['calibrated']), 3) if 'ratio' in str(item['label']) else _pct(float(item['calibrated'])))}</td>"
        f"<td>{(_num(float(item['learned']), 3) if 'ratio' in str(item['label']) else _pct(float(item['learned'])))}</td>"
        "</tr>"
        for item in view_model.comparison_metrics
    )
    horizon_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item['stage']))}</td>"
        f"<td>{escape(str(item['horizon']))}</td>"
        f"<td>{_num(float(item['rank_ic']), 3)}</td>"
        f"<td>{_pct(float(item['top_decile_return']))}</td>"
        f"<td>{_pct(float(item['top_bottom_spread']))}</td>"
        f"<td>{_pct(float(item['top_k_hit_rate']))}</td>"
        "</tr>"
        for item in view_model.horizon_metrics
    )
    trial_rows = []
    for item in view_model.calibration_trials:
        policy = dict(item.get("policy", {}))
        summary = dict(item.get("summary", {}))
        trial_rows.append(
            "<tr>"
            f"<td>{int(item['rank'])}</td>"
            f"<td>{_pct(float(policy.get('risk_on_exposure', 0.0)))}</td>"
            f"<td>{int(policy.get('risk_on_positions', 0))}</td>"
            f"<td>{_pct(float(policy.get('risk_on_turnover_cap', 0.0)))}</td>"
            f"<td>{_pct(float(summary.get('annual_return', 0.0)))}</td>"
            f"<td>{_pct(float(summary.get('benchmark_annual_return', 0.0)))}</td>"
            f"<td>{_pct(float(summary.get('excess_annual_return', 0.0)))}</td>"
            f"<td>{_num(float(summary.get('information_ratio', 0.0)), 3)}</td>"
            f"<td>{_pct(float(summary.get('max_drawdown', 0.0)))}</td>"
            f"<td>{_num(float(item['score']), 4)}</td>"
            "</tr>"
        )
    artifact_rows = "".join(
        f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
        for key, value in view_model.artifacts.items()
    )
    lifecycle_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('theme', 'NA')))}</td>"
        f"<td>{escape(str(item.get('phase', 'NA')))}</td>"
        f"<td>{_pct(float(item.get('conviction', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('event_risk', float('nan'))))}</td>"
        "</tr>"
        for item in view_model.theme_lifecycle_summary.get("top_themes", [])[:8]
    )
    role_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('symbol', 'NA')))}</td>"
        f"<td>{escape(str(item.get('theme', 'NA')))}</td>"
        f"<td>{escape(str(item.get('role', 'NA')))}</td>"
        f"<td>{'yes' if item.get('role_downgrade', False) else 'no'}</td>"
        "</tr>"
        for item in view_model.role_distribution_summary.get("top_roles", [])[:10]
    )
    exit_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('symbol', 'NA')))}</td>"
        f"<td>{escape(str(item.get('theme', 'NA')))}</td>"
        f"<td>{escape(str(item.get('role', 'NA')))}</td>"
        f"<td>{escape(str(item.get('note', 'NA')))}</td>"
        "</tr>"
        for item in view_model.exit_contribution_summary.get("examples", [])[:8]
    )
    leader_eval = dict(view_model.leader_summary.get("evaluation", {}))
    leader_rows = "".join(
        "<tr>"
        f"<td>{escape(str(item.get('symbol', 'NA')))}</td>"
        f"<td>{escape(str(item.get('theme', 'NA')))}</td>"
        f"<td>{escape(str(item.get('role', 'NA')))}</td>"
        f"<td>{_pct(float(item.get('candidate_score', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('conviction_score', float('nan'))))}</td>"
        f"<td>{_pct(float(item.get('negative_score', float('nan'))))}</td>"
        "</tr>"
        for item in view_model.leader_summary.get("top_candidates", [])[:8]
    )
    feature_names = ", ".join(str(item) for item in view_model.learning_model.get("feature_names", [])) or "NA"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2 研究回测看板</title>
  <style>
    :root {{
      --bg: #eef3f1;
      --paper: #ffffff;
      --ink: #18242a;
      --muted: #60717a;
      --line: #d7e0e5;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Avenir Next", "PingFang SC", sans-serif; color: var(--ink); background: linear-gradient(180deg, #f7fbfc 0%, #eef3f1 100%); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    .hero, .card {{ background: var(--paper); border: 1px solid var(--line); border-radius: 22px; box-shadow: 0 14px 34px rgba(23,45,52,0.08); }}
    .hero {{ padding: 24px; }}
    .card {{ padding: 18px; margin-top: 18px; }}
    h1, h2 {{ margin: 0; }}
    h1 {{ font-size: 34px; letter-spacing: -0.02em; }}
    h2 {{ font-size: 18px; margin-bottom: 12px; }}
    .sub {{ margin-top: 8px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid #e8eef0; text-align: left; font-size: 13px; }}
    th {{ color: var(--muted); font-weight: 700; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>V2 研究回测看板</h1>
      <div class="sub">策略 {escape(view_model.strategy_id)} | run_id {escape(view_model.run_id or 'NA')} | release gate {'passed' if view_model.release_gate_passed else 'pending'}</div>
    </section>
    <section class="card"><h2>核心指标</h2><table><thead><tr><th>指标</th><th>baseline</th><th>calibrated</th><th>learned</th></tr></thead><tbody>{comparison_rows}</tbody></table></section>
    <section class="card"><h2>Horizon Metrics</h2><table><thead><tr><th>方案</th><th>周期</th><th>RankIC</th><th>头部分层收益</th><th>头尾价差</th><th>TopK命中率</th></tr></thead><tbody>{horizon_rows}</tbody></table></section>
    <section class="card"><h2>Validation Trials</h2><table><thead><tr><th>排名</th><th>risk_on_exposure</th><th>risk_on_positions</th><th>risk_on_turnover_cap</th><th>annual_return</th><th>benchmark_annual_return</th><th>excess_annual_return</th><th>information_ratio</th><th>max_drawdown</th><th>score</th></tr></thead><tbody>{''.join(trial_rows) or "<tr><td colspan='10'>暂无试验</td></tr>"}</tbody></table></section>
    <section class="card"><h2>Learned Policy</h2><p>feature_names: {escape(feature_names)}</p><p>train_rows: {int(view_model.learning_model.get('train_rows', 0))}</p><p>train_r2_exposure: {_num(float(view_model.learning_model.get('train_r2_exposure', 0.0)), 3)} | train_r2_positions: {_num(float(view_model.learning_model.get('train_r2_positions', 0.0)), 3)} | train_r2_turnover: {_num(float(view_model.learning_model.get('train_r2_turnover', 0.0)), 3)}</p></section>
    <section class="card"><h2>主线生命周期摘要</h2><p>phase_counts: {escape(str(view_model.theme_lifecycle_summary.get('phase_counts', {})))}</p><table><thead><tr><th>主线</th><th>phase</th><th>conviction</th><th>event_risk</th></tr></thead><tbody>{lifecycle_rows or "<tr><td colspan='4'>暂无主线摘要</td></tr>"}</tbody></table></section>
    <section class="card"><h2>龙头识别评估</h2><p>candidate_recall_at_k {_pct(float(leader_eval.get('candidate_recall_at_k', 0.0)))} | conviction_precision_at_1 {_pct(float(leader_eval.get('conviction_precision_at_1', 0.0)))} | ndcg_at_k {_num(float(leader_eval.get('ndcg_at_k', 0.0)), 3)} | hard_negative_survival_recall {_pct(float(leader_eval.get('hard_negative_survival_recall', 0.0)))} | theme_groups {int(leader_eval.get('theme_group_count', 0))}</p><table><thead><tr><th>symbol</th><th>theme</th><th>role</th><th>candidate</th><th>conviction</th><th>negative</th></tr></thead><tbody>{leader_rows or "<tr><td colspan='6'>暂无龙头评估</td></tr>"}</tbody></table></section>
    <section class="card"><h2>角色分布摘要</h2><p>role_counts: {escape(str(view_model.role_distribution_summary.get('role_counts', {})))}</p><table><thead><tr><th>symbol</th><th>theme</th><th>role</th><th>downgrade</th></tr></thead><tbody>{role_rows or "<tr><td colspan='4'>暂无角色摘要</td></tr>"}</tbody></table></section>
    <section class="card"><h2>持仓退出决策贡献摘要</h2><p>fading themes {int(view_model.exit_contribution_summary.get('fading_theme_count', 0))} | crowded themes {int(view_model.exit_contribution_summary.get('crowded_theme_count', 0))} | role downgrades {int(view_model.exit_contribution_summary.get('role_downgrade_count', 0))}</p><table><thead><tr><th>symbol</th><th>theme</th><th>role</th><th>note</th></tr></thead><tbody>{exit_rows or "<tr><td colspan='4'>暂无退出摘要</td></tr>"}</tbody></table></section>
    <section class="card"><h2>Artifacts</h2><table><tbody>{artifact_rows or "<tr><td>暂无产物</td></tr>"}</tbody></table></section>
  </div>
</body>
</html>
"""
