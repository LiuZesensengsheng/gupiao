from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.application.use_cases import DailyFusionResult, DiscoveryResult
from src.domain.entities import BacktestMetrics, BinaryMetrics, DiscoveryRow, ForecastRow, MarketForecast
from src.domain.policies import market_regime, target_exposure
from src.interfaces.presenters.driver_explainer import format_driver_list


def _to_percent(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 100:.1f}%"


def _to_bp(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 10000:.0f}bp"


def _to_float(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


def _to_money(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:,.0f}"


def _to_int(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return str(int(round(float(v))))


def _metrics_line(metrics: BinaryMetrics) -> str:
    auc_text = f"{metrics.auc:.3f}" if not pd.isna(metrics.auc) else "NA"
    brier_text = f"{metrics.brier:.3f}" if not pd.isna(metrics.brier) else "NA"
    return f"n={metrics.n}, acc={_to_percent(metrics.accuracy)}, auc={auc_text}, brier={brier_text}"


def _bt_line(metrics: BacktestMetrics) -> str:
    start_text = str(metrics.start_date.date()) if not pd.isna(metrics.start_date) else "NA"
    end_text = str(metrics.end_date.date()) if not pd.isna(metrics.end_date) else "NA"
    return (
        f"| {metrics.label} | {start_text} | {end_text} | {metrics.n_days} | "
        f"{_to_percent(metrics.total_return)} | {_to_percent(metrics.annual_return)} | {_to_percent(metrics.excess_annual_return)} | "
        f"{_to_percent(metrics.max_drawdown)} | {_to_float(metrics.sharpe)} | {_to_float(metrics.sortino)} | "
        f"{_to_float(metrics.calmar)} | {_to_float(metrics.information_ratio)} | {_to_percent(metrics.win_rate)} | "
        f"{_to_percent(metrics.annual_turnover)} | {_to_percent(metrics.total_cost)} | {_to_float(metrics.avg_trades_per_stock_per_week, 2)} |"
    )


def write_forecast_report(
    out_path: str | Path,
    market_forecast: MarketForecast,
    stock_rows: Sequence[ForecastRow],
) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    regime = market_regime(market_forecast.short_prob, market_forecast.mid_prob)
    exposure = target_exposure(market_forecast.short_prob, market_forecast.mid_prob)

    lines: list[str] = []
    lines.append("# A股多周期预测报告 (V1)")
    lines.append("")
    lines.append(f"- 数据截至: {market_forecast.latest_date.date()}")
    lines.append(f"- 市场基准: {market_forecast.name} ({market_forecast.symbol})")
    lines.append(f"- 市场状态: {regime}")
    lines.append(f"- 建议总仓位: {_to_percent(exposure)}")
    lines.append("")
    lines.append("## 大盘预测")
    lines.append("")
    lines.append("| 维度 | 上涨概率 | 说明 |")
    lines.append("|---|---:|---|")
    lines.append(f"| 短期(1日) | {_to_percent(market_forecast.short_prob)} | 次日方向概率 |")
    lines.append(f"| 中期(20日) | {_to_percent(market_forecast.mid_prob)} | 未来20日方向概率 |")
    lines.append("")
    lines.append(f"- 短期样本外: {_metrics_line(market_forecast.short_eval)}")
    lines.append(f"- 中期样本外: {_metrics_line(market_forecast.mid_eval)}")
    lines.append("")
    lines.append("## 个股预测")
    lines.append("")
    lines.append("| 个股 | 短期概率 | 中期概率 | 综合分数 | 建议权重 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in stock_rows:
        lines.append(
            f"| {row.name} ({row.symbol}) | {_to_percent(row.short_prob)} | {_to_percent(row.mid_prob)} | {row.score:.3f} | {_to_percent(row.suggested_weight)} |"
        )
    lines.append("")
    lines.append("## 因子解释 (最新截面)")
    lines.append("")
    for row in stock_rows:
        short_d = format_driver_list(row.short_drivers)
        mid_d = format_driver_list(row.mid_drivers)
        lines.append(f"### {row.name} ({row.symbol})")
        lines.append(f"- 短期驱动: {short_d}")
        lines.append(f"- 中期驱动: {mid_d}")
        lines.append(f"- 短期样本外: {_metrics_line(row.short_eval)}")
        lines.append(f"- 中期样本外: {_metrics_line(row.mid_eval)}")
        lines.append("")

    lines.append("## 风险提示")
    lines.append("")
    lines.append("- 本报告为统计预测结果，不构成投资建议。")
    lines.append("- 若市场处于 Risk-Off，建议优先执行仓位约束。")
    lines.append("- 建议叠加基本面与事件风控，不建议仅凭单模型交易。")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_daily_report(out_path: str | Path, result: DailyFusionResult) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    regime = result.market_state_label
    exposure = float(result.effective_total_exposure)

    lines: list[str] = []
    lines.append("# A股每日融合报告 (量化 + 学习型新闻融合)")
    lines.append("")
    lines.append(f"- 报告日期: {result.as_of_date.date()}")
    lines.append(f"- 数据源: {result.source}")
    lines.append(f"- 市场基准: {result.market_forecast.name} ({result.market_forecast.symbol})")
    lines.append(f"- 近窗新闻条数: {result.news_items_count}")
    lines.append(f"- 市场状态: {regime}")
    lines.append(f"- 建议总仓位: {_to_percent(exposure)}")
    lines.append(f"- 执行模板: {result.strategy_template} | 日内T强度: {result.intraday_t_level}")
    lines.append(
        f"- 状态参数: 权重阈值={result.effective_weight_threshold:.2f}, 持仓上限={int(result.effective_max_positions)}, "
        f"单股日交易上限={int(result.effective_max_trades_per_stock_per_day)}, 单股周交易上限={int(result.effective_max_trades_per_stock_per_week)}"
    )
    lines.append("")
    lines.append("## 大盘融合结果")
    lines.append("")
    lines.append("| 维度 | 模型概率 | 新闻净分 | 新闻模型概率 | 融合后概率 | 融合方式 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    lines.append(
        f"| 短期(1日) | {_to_percent(result.market_forecast.short_prob)} | {result.market_short_sent.score:+.3f} | {_to_percent(result.market_news_short_prob)} | {_to_percent(result.market_final_short)} | {result.market_fusion_mode_short} |"
    )
    lines.append(
        f"| 中期(20日) | {_to_percent(result.market_forecast.mid_prob)} | {result.market_mid_sent.score:+.3f} | {_to_percent(result.market_news_mid_prob)} | {_to_percent(result.market_final_mid)} | {result.market_fusion_mode_mid} |"
    )
    lines.append("")
    lines.append("## 个股融合结果")
    lines.append("")
    lines.append("| 个股 | 模型短期 | 模型中期 | 新闻短期净分 | 新闻中期净分 | 新闻短期概率 | 新闻中期概率 | 融合短期 | 融合中期 | 综合分数 | 建议权重 | 短期方式 | 中期方式 | 量价风险 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for row in result.blended_rows:
        risk_text = "高位巨量阴线风险" if row.volume_risk_flag else "正常"
        lines.append(
            f"| {row.name} ({row.symbol}) | {_to_percent(row.base_short)} | {_to_percent(row.base_mid)} | {row.short_sent.score:+.3f} | {row.mid_sent.score:+.3f} | {_to_percent(row.news_short_prob)} | {_to_percent(row.news_mid_prob)} | {_to_percent(row.final_short)} | {_to_percent(row.final_mid)} | {row.final_score:.3f} | {_to_percent(row.suggested_weight)} | {row.fusion_mode_short} | {row.fusion_mode_mid} | {risk_text} |"
        )
    lines.append("")
    lines.append("## 调仓执行建议")
    lines.append("")
    nav_text = _to_money(result.trade_plan_nav)
    lines.append(
        f"- 计算口径: `{result.trade_plan_basis}` | 组合净值: {nav_text if nav_text != 'NA' else 'NA(仅输出权重差)'} | 每手股数: {int(result.trade_plan_lot_size)}"
    )
    lines.append("| 个股 | 动作 | 当前权重 | 目标权重 | 权重变化 | 参考价格 | 估算金额 | 估算股数 | 估算手数 | 备注 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    if not result.trade_actions:
        lines.append("| 无数据 | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for action in result.trade_actions:
            lines.append(
                f"| {action.name} ({action.symbol}) | {action.action} | {_to_percent(action.current_weight)} | {_to_percent(action.target_weight)} | {_to_percent(action.delta_weight)} | {_to_float(action.est_price, 3)} | {_to_money(action.est_delta_value)} | {_to_int(action.est_delta_shares)} | {_to_float(action.est_delta_lots, 2)} | {action.note or 'NA'} |"
            )
    lines.append("")
    lines.append("## 因子解释 (最新截面)")
    lines.append("")
    for row in result.blended_rows:
        short_d = format_driver_list(row.short_drivers)
        mid_d = format_driver_list(row.mid_drivers)
        lines.append(f"### {row.name} ({row.symbol})")
        lines.append(f"- 短期驱动: {short_d}")
        lines.append(f"- 中期驱动: {mid_d}")
        lines.append(f"- 风险备注: {row.volume_risk_note or 'NA'}")
        lines.append("")
    lines.append("## 新闻模糊矩阵")
    lines.append("")
    lines.append("| 标的 | 维度 | 利好隶属 | 利空隶属 | 中性隶属 | 事件条数 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    lines.append(
        f"| {result.market_forecast.name} ({result.market_forecast.symbol}) | 短期 | {result.market_short_sent.bullish:.3f} | {result.market_short_sent.bearish:.3f} | {result.market_short_sent.neutral:.3f} | {result.market_short_sent.items} |"
    )
    lines.append(
        f"| {result.market_forecast.name} ({result.market_forecast.symbol}) | 中期 | {result.market_mid_sent.bullish:.3f} | {result.market_mid_sent.bearish:.3f} | {result.market_mid_sent.neutral:.3f} | {result.market_mid_sent.items} |"
    )
    for row in result.blended_rows:
        lines.append(
            f"| {row.name} ({row.symbol}) | 短期 | {row.short_sent.bullish:.3f} | {row.short_sent.bearish:.3f} | {row.short_sent.neutral:.3f} | {row.short_sent.items} |"
        )
        lines.append(
            f"| {row.name} ({row.symbol}) | 中期 | {row.mid_sent.bullish:.3f} | {row.mid_sent.bearish:.3f} | {row.mid_sent.neutral:.3f} | {row.mid_sent.items} |"
        )
    lines.append("")
    lines.append("## 学习型融合诊断")
    lines.append("")
    lines.append("| 标的 | 周期 | 模式 | 原因 | 样本数 | 验证样本 | 验证准确率 | 验证AUC | 验证Brier | 新闻系数(sent_score) | 融合系数(quant) | 融合系数(news) |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    if not result.learning_diagnostics:
        lines.append("| NA | NA | NA | NA | 0 | 0 | NA | NA | NA | NA | NA | NA |")
    else:
        for diag in result.learning_diagnostics:
            lines.append(
                f"| {diag.target} | {diag.horizon} | {diag.mode} | {diag.reason} | {diag.samples} | {diag.holdout_n} | {_to_percent(diag.holdout_accuracy)} | {_to_float(diag.holdout_auc, 3)} | {_to_float(diag.holdout_brier, 3)} | {_to_float(diag.news_coef_score, 3)} | {_to_float(diag.fusion_coef_quant, 3)} | {_to_float(diag.fusion_coef_news, 3)} |"
            )
    lines.append("")
    lines.append("## 市场效应分析")
    lines.append("")
    eff = result.effect_summary
    lines.append(f"- 样本覆盖: {eff.sample_size} 只股票")
    lines.append(
        f"- 赚钱效应: {eff.pnl_label} | 1日胜率 {_to_percent(eff.win_rate_1d)} | 5日胜率 {_to_percent(eff.win_rate_5d)} | 强势率(5日>3%) {_to_percent(eff.strong_rate_5d)}"
    )
    lines.append(
        f"- 亏钱效应: 风险{eff.risk_label} | 1日亏损率 {_to_percent(eff.loss_rate_1d)} | 深亏率 {_to_percent(eff.deep_loss_rate)} | 20日中位回撤 {_to_percent(eff.median_drawdown_20)}"
    )
    lines.append(
        f"- 资金状态: {eff.money_label} | 量能比(20日) {eff.avg_vol_ratio_20:.2f} | OBV Z值 {eff.avg_obv_z_20:.2f} | 资金分数 {eff.money_score:+.3f}"
    )
    lines.append(
        f"- 筹码结构: {eff.chip_label} | 20日位置 {eff.avg_price_pos_20:.2f} | 量能集中度(5/20) {eff.avg_vol_conc_5_20:.2f} | 筹码分数 {eff.chip_score:+.3f}"
    )
    lines.append("")
    lines.append("## 板块热度")
    lines.append("")
    sector_table = result.sector_table
    if sector_table.empty:
        lines.append("- 无可用板块数据")
    else:
        lines.append("| 板块 | 热度分数 | 1日胜率 | 5日中位收益 | 资金分数 | 筹码分数 | 样本数 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in sector_table.iterrows():
            lines.append(
                f"| {row['sector']} | {row['heat_score']:+.3f} | {_to_percent(float(row['win_rate_1d']))} | {_to_bp(float(row['median_ret_5d']))} | {float(row['money_score']):+.3f} | {float(row['chip_score']):+.3f} | {int(row['count'])} |"
            )
    lines.append("")
    lines.append("## 策略优化 (收益目标 + 换手约束)")
    lines.append("")
    lines.append(f"- 目标函数: `{result.strategy_objective_text}`")
    lines.append(f"- 评估窗口: `{result.strategy_target_metric_label}`")
    if result.strategy_selected is None:
        lines.append("- 结果: 使用默认参数（优化器关闭或未找到有效试验）")
    else:
        s = result.strategy_selected
        lines.append(
            f"- 选中参数: retrain={s.retrain_days}日, 阈值={s.weight_threshold:.2f}, 持仓上限={s.max_positions}, "
            f"市场新闻强度={s.market_news_strength:.2f}, 个股新闻强度={s.stock_news_strength:.2f}"
        )
        lines.append(
            f"- 选中表现: 年化超额 {_to_percent(s.excess_annual_return)} | 年化换手 {_to_percent(s.annual_turnover)} | "
            f"最大回撤 {_to_percent(s.max_drawdown)} | 单票周交易 {_to_float(s.avg_trades_per_stock_per_week, 2)} 次 | 目标得分 {_to_float(s.objective_score, 4)}"
        )
    lines.append("")
    lines.append("| 排名 | 指标窗口 | retrain(日) | 阈值 | 持仓上限 | 市场新闻强度 | 个股新闻强度 | 目标得分 | 年化收益 | 年化超额 | 最大回撤 | 年化换手 | 成本损耗 | 单票周交易 | Sharpe |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    if not result.strategy_trials:
        lines.append("| - | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for t in result.strategy_trials:
            lines.append(
                f"| {t.rank} | {t.metric_label} | {t.retrain_days} | {t.weight_threshold:.2f} | {t.max_positions} | "
                f"{t.market_news_strength:.2f} | {t.stock_news_strength:.2f} | {_to_float(t.objective_score, 4)} | "
                f"{_to_percent(t.annual_return)} | {_to_percent(t.excess_annual_return)} | {_to_percent(t.max_drawdown)} | "
                f"{_to_percent(t.annual_turnover)} | {_to_percent(t.total_cost)} | {_to_float(t.avg_trades_per_stock_per_week, 2)} | {_to_float(t.sharpe)} |"
            )
    lines.append("")
    lines.append("## 验收结果 (A/B + 约束审计)")
    lines.append("")
    lines.append(f"- 验收开关: {'开启' if result.acceptance_enabled else '关闭'}")
    lines.append(f"- A/B是否通过: {'是' if result.acceptance_ab_pass else '否'}")
    lines.append(f"- 约束审计是否通过: {'是' if result.acceptance_constraints_pass else '否'}")
    lines.append(f"- A/B超额年化差(新-旧): {_to_percent(result.acceptance_delta_excess_annual_return)}")
    lines.append(f"- A/B最大回撤差(新-旧): {_to_percent(result.acceptance_delta_max_drawdown)}")
    lines.append(f"- A/B年化换手差(新-旧): {_to_percent(result.acceptance_delta_annual_turnover)}")
    lines.append(f"- 周频次越界次数(监控项): {int(result.acceptance_limit_violations)}")
    lines.append(f"- 超卖违规次数: {int(result.acceptance_oversell_violations)}")
    lines.append(f"- 汇总: {result.acceptance_summary}")
    lines.append("")
    lines.append("## 使用说明")
    lines.append("")
    lines.append("- 新闻净分范围为 [-1, +1]，正值偏利好，负值偏利空。")
    lines.append("- `learned` 模式下先学习新闻影响，再校准 quant+news；`rule` 模式下回退为新闻净分非线性修正。")
    lines.append("- 若你希望更保守，可降低 `--stock-news-strength` 与 `--market-news-strength`。")
    lines.append("- 换手约束建议: `--max-trades-per-stock-per-day 1` + `--max-trades-per-stock-per-week 2~3`，并设置 `--min-weight-change-to-trade` 过滤微小调仓。")
    lines.append("- 本结果为研究支持，不构成投资建议。")
    lines.append("")
    lines.append("## 回测表现 (交易级, 含成本)")
    lines.append("")
    lines.append("| 窗口 | 开始 | 结束 | 交易日 | 策略总收益 | 年化收益 | 年化超额 | 最大回撤 | Sharpe | Sortino | Calmar | 信息比率 | 日胜率 | 年化换手 | 成本损耗 | 单票周交易 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    if not result.backtest_metrics:
        lines.append("| 无数据 | NA | NA | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for metrics in result.backtest_metrics:
            lines.append(_bt_line(metrics))
    lines.append("")
    lines.append("- 指标口径: 策略按日调仓，收益为次日收益，成本=换手*(佣金bps+滑点bps)。")
    lines.append("- 当启用新闻融合回测时，表格同时展示 `融合策略-*` 与 `量化基线-*` 对比。")
    lines.append("- 回测曲线已在 dashboard 中可视化展示（主曲线为当前融合策略）。")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _discovery_row_line(row: DiscoveryRow) -> str:
    risk = "高位巨量阴线风险" if row.volume_risk_flag else "正常"
    short_d = format_driver_list(row.short_drivers)
    mid_d = format_driver_list(row.mid_drivers)
    return (
        f"| {row.name} ({row.symbol}) | {_to_percent(row.short_prob)} | {_to_percent(row.mid_prob)} | "
        f"{row.score:.3f} | {_to_percent(row.suggested_weight)} | {risk} | {short_d} | {mid_d} |"
    )


def write_discovery_report(out_path: str | Path, result: DiscoveryResult) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    regime = market_regime(result.market_forecast.short_prob, result.market_forecast.mid_prob)
    exposure = target_exposure(result.market_forecast.short_prob, result.market_forecast.mid_prob)

    lines: list[str] = []
    lines.append("# A股候选发掘报告 (选股补充 + 择时辅助)")
    lines.append("")
    lines.append(f"- 数据截至: {result.as_of_date.date()}")
    lines.append(f"- 数据源: {result.source}")
    lines.append(f"- 市场基准: {result.market_forecast.name} ({result.market_forecast.symbol})")
    lines.append(f"- 市场状态: {regime}")
    lines.append(f"- 建议总仓位(择时): {_to_percent(exposure)}")
    lines.append(f"- 候选池来源: {result.universe_source}")
    lines.append(f"- 候选池规模: {result.universe_size}")
    lines.append("")
    if result.warnings:
        lines.append("## 提示")
        lines.append("")
        for w in result.warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## 候选排序")
    lines.append("")
    lines.append("| 个股 | 短期概率 | 中期概率 | 综合分数 | 建议权重 | 量价风险 | 短期驱动 | 中期驱动 |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    if not result.rows:
        lines.append("| 无数据 | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for row in result.rows:
            lines.append(_discovery_row_line(row))
    lines.append("")
    lines.append("## 使用建议")
    lines.append("")
    lines.append("- 这份列表用于补充选股池，不替代你的人工研究。")
    lines.append("- 你来选股、系统来择时是可行模式：把选股 alpha 与仓位/节奏 beta 分离。")
    lines.append("- 若出现 `高位巨量阴线风险`，建议降低单票仓位或等待二次确认。")
    lines.append("- 本结果为研究支持，不构成投资建议。")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
