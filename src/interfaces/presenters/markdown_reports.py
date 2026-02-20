from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from src.application.use_cases import DailyFusionResult
from src.domain.entities import BacktestMetrics, BinaryMetrics, ForecastRow, MarketForecast
from src.domain.policies import market_regime, target_exposure


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
        f"{_to_percent(metrics.annual_turnover)} | {_to_percent(metrics.total_cost)} |"
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
        short_d = ", ".join(row.short_drivers) if row.short_drivers else "NA"
        mid_d = ", ".join(row.mid_drivers) if row.mid_drivers else "NA"
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

    regime = market_regime(result.market_final_short, result.market_final_mid)
    exposure = target_exposure(result.market_final_short, result.market_final_mid)

    lines: list[str] = []
    lines.append("# A股每日融合报告 (量化 + 新闻模糊矩阵)")
    lines.append("")
    lines.append(f"- 报告日期: {result.as_of_date.date()}")
    lines.append(f"- 数据源: {result.source}")
    lines.append(f"- 市场基准: {result.market_forecast.name} ({result.market_forecast.symbol})")
    lines.append(f"- 近窗新闻条数: {result.news_items_count}")
    lines.append(f"- 市场状态: {regime}")
    lines.append(f"- 建议总仓位: {_to_percent(exposure)}")
    lines.append("")
    lines.append("## 大盘融合结果")
    lines.append("")
    lines.append("| 维度 | 模型概率 | 新闻净分 | 融合后概率 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| 短期(1日) | {_to_percent(result.market_forecast.short_prob)} | {result.market_short_sent.score:+.3f} | {_to_percent(result.market_final_short)} |"
    )
    lines.append(
        f"| 中期(20日) | {_to_percent(result.market_forecast.mid_prob)} | {result.market_mid_sent.score:+.3f} | {_to_percent(result.market_final_mid)} |"
    )
    lines.append("")
    lines.append("## 个股融合结果")
    lines.append("")
    lines.append("| 个股 | 模型短期 | 模型中期 | 新闻短期净分 | 新闻中期净分 | 融合短期 | 融合中期 | 综合分数 | 建议权重 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in result.blended_rows:
        lines.append(
            f"| {row.name} ({row.symbol}) | {_to_percent(row.base_short)} | {_to_percent(row.base_mid)} | {row.short_sent.score:+.3f} | {row.mid_sent.score:+.3f} | {_to_percent(row.final_short)} | {_to_percent(row.final_mid)} | {row.final_score:.3f} | {_to_percent(row.suggested_weight)} |"
        )
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
    lines.append("## 使用说明")
    lines.append("")
    lines.append("- 新闻净分范围为 [-1, +1]，正值偏利好，负值偏利空。")
    lines.append("- 融合后概率 = 模型概率经过新闻净分的非线性修正。")
    lines.append("- 若你希望更保守，可降低 `--stock-news-strength` 与 `--market-news-strength`。")
    lines.append("- 本结果为研究支持，不构成投资建议。")
    lines.append("")
    lines.append("## 回测表现 (交易级, 含成本)")
    lines.append("")
    lines.append("| 窗口 | 开始 | 结束 | 交易日 | 策略总收益 | 年化收益 | 年化超额 | 最大回撤 | Sharpe | Sortino | Calmar | 信息比率 | 日胜率 | 年化换手 | 成本损耗 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    if not result.backtest_metrics:
        lines.append("| 无数据 | NA | NA | 0 | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    else:
        for metrics in result.backtest_metrics:
            lines.append(_bt_line(metrics))
    lines.append("")
    lines.append("- 指标口径: 策略按日调仓，收益为次日收益，成本=换手*(佣金bps+滑点bps)。")
    lines.append("- 回测曲线已在 dashboard 中可视化展示。")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
