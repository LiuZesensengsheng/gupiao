from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pandas as pd

from src.application.use_cases import DailyFusionResult, DiscoveryResult
from src.contracts.reporting import DailyReportViewModel, ResearchReportViewModel
from src.domain.entities import BacktestMetrics, BinaryMetrics, DiscoveryRow, ForecastRow, MarketForecast
from src.domain.policies import market_regime, target_exposure
from src.interfaces.presenters.driver_explainer import format_driver_list
from src.interfaces.presenters.v2_view_model_renderers import render_daily_markdown, render_research_markdown
from src.reporting.view_models import build_daily_report_view_model, build_research_report_view_model

if TYPE_CHECKING:
    from src.application.v2_contracts import (
        DailyRunResult as V2DailyRunResult,
        V2BacktestSummary,
        V2CalibrationResult,
        V2PolicyLearningResult,
    )


def _to_percent(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 100:.1f}%"


def _to_float(v: float, digits: int = 2) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{digits}f}"


def _to_money(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:,.0f}"


def _metrics_line(metrics: BinaryMetrics) -> str:
    auc_text = f"{metrics.auc:.3f}" if not pd.isna(metrics.auc) else "NA"
    brier_text = f"{metrics.brier:.3f}" if not pd.isna(metrics.brier) else "NA"
    return f"n={metrics.n}, acc={_to_percent(metrics.accuracy)}, auc={auc_text}, brier={brier_text}"


def _bt_line(metrics: BacktestMetrics) -> str:
    start_text = str(metrics.start_date.date()) if not pd.isna(metrics.start_date) else "NA"
    end_text = str(metrics.end_date.date()) if not pd.isna(metrics.end_date) else "NA"
    return (
        f"| {metrics.label} | {start_text} | {end_text} | {metrics.n_days} | "
        f"{_to_percent(metrics.total_return)} | {_to_percent(metrics.annual_return)} | "
        f"{_to_percent(metrics.max_drawdown)} | {_to_float(metrics.sharpe)} |"
    )


def _write_lines(out_path: str | Path, lines: Sequence[str]) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def write_forecast_report(
    out_path: str | Path,
    market_forecast: MarketForecast,
    stock_rows: Sequence[ForecastRow],
) -> Path:
    regime = market_regime(market_forecast.short_prob, market_forecast.mid_prob)
    exposure = target_exposure(market_forecast.short_prob, market_forecast.mid_prob)

    lines: list[str] = [
        "# Forecast Report",
        "",
        f"- Date: {market_forecast.latest_date.date()}",
        f"- Market: {market_forecast.name} ({market_forecast.symbol})",
        f"- Regime: {regime}",
        f"- Suggested exposure: {_to_percent(exposure)}",
        "",
        "## Market",
        "",
        "| Horizon | Up Prob | Expected Return |",
        "|---|---:|---:|",
        f"| 1d | {_to_percent(market_forecast.short_prob)} | {_to_percent(market_forecast.short_expected_ret)} |",
        f"| 5d | {_to_percent(market_forecast.five_prob)} | NA |",
        f"| 20d | {_to_percent(market_forecast.mid_prob)} | {_to_percent(market_forecast.mid_expected_ret)} |",
        "",
        f"- Short eval: {_metrics_line(market_forecast.short_eval)}",
        f"- Mid eval: {_metrics_line(market_forecast.mid_eval)}",
        "",
        "## Stocks",
        "",
        "| Stock | 1d | 5d | 20d | Score | Weight |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if not stock_rows:
        lines.append("| NA | NA | NA | NA | NA | NA |")
    else:
        for row in stock_rows:
            lines.append(
                f"| {row.name} ({row.symbol}) | {_to_percent(row.short_prob)} | {_to_percent(row.five_prob)} | "
                f"{_to_percent(row.mid_prob)} | {_to_float(row.score, 3)} | {_to_percent(row.suggested_weight)} |"
            )
    lines.extend(["", "## Drivers", ""])
    for row in stock_rows[:10]:
        lines.append(f"### {row.name} ({row.symbol})")
        lines.append(f"- Short: {format_driver_list(row.short_drivers)}")
        lines.append(f"- Mid: {format_driver_list(row.mid_drivers)}")
        lines.append("")
    return _write_lines(out_path, lines)


def write_daily_report(out_path: str | Path, result: DailyFusionResult) -> Path:
    lines: list[str] = [
        "# Daily Fusion Report",
        "",
        f"- Date: {result.as_of_date.date()}",
        f"- Source: {result.source}",
        f"- Market state: {result.market_state_label}",
        f"- Strategy template: {result.strategy_template}",
        f"- Suggested exposure: {_to_percent(result.effective_total_exposure)}",
        f"- Weight threshold: {_to_percent(result.effective_weight_threshold)}",
        f"- News items: {result.news_items_count}",
        "",
        "## Market",
        "",
        f"- Short prob: {_to_percent(result.market_forecast.short_prob)}",
        f"- Mid prob: {_to_percent(result.market_forecast.mid_prob)}",
        f"- Final short: {_to_percent(result.market_final_short)}",
        f"- Final mid: {_to_percent(result.market_final_mid)}",
        "",
        "## Candidates",
        "",
        "| Stock | Final Short | Final Mid | Score | Weight | Volume Risk |",
        "|---|---:|---:|---:|---:|---|",
    ]
    if not result.blended_rows:
        lines.append("| NA | NA | NA | NA | NA | NA |")
    else:
        for row in result.blended_rows[:20]:
            volume_note = "yes" if row.volume_risk_flag else "no"
            lines.append(
                f"| {row.name} ({row.symbol}) | {_to_percent(row.final_short)} | {_to_percent(row.final_mid)} | "
                f"{_to_float(row.final_score, 3)} | {_to_percent(row.suggested_weight)} | {volume_note} |"
            )
    lines.extend(["", "## Trade Plan", "", "| Stock | Action | Delta Weight | Note |", "|---|---|---:|---|"])
    if not result.trade_actions:
        lines.append("| NA | HOLD | NA | No trades |")
    else:
        for action in result.trade_actions:
            lines.append(
                f"| {action.name or action.symbol} | {action.action} | {_to_percent(action.delta_weight)} | {action.note or 'NA'} |"
            )
    if result.backtest_metrics:
        lines.extend(
            [
                "",
                "## Backtest",
                "",
                "| Label | Start | End | Days | Total Return | Annual Return | Max DD | Sharpe |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metrics in result.backtest_metrics:
            lines.append(_bt_line(metrics))
    lines.extend(["", "## Effect Summary", ""])
    effect = result.effect_summary
    lines.extend(
        [
            f"- Sample size: {effect.sample_size}",
            f"- Win rate 1d: {_to_percent(effect.win_rate_1d)}",
            f"- Win rate 5d: {_to_percent(effect.win_rate_5d)}",
            f"- Median return 5d: {_to_percent(effect.median_ret_5d)}",
            f"- Money score: {_to_float(effect.money_score, 2)} ({effect.money_label})",
            f"- Chip score: {_to_float(effect.chip_score, 2)} ({effect.chip_label})",
        ]
    )
    return _write_lines(out_path, lines)


def write_discovery_report(out_path: str | Path, result: DiscoveryResult) -> Path:
    lines: list[str] = [
        "# Discovery Report",
        "",
        f"- Date: {result.as_of_date.date()}",
        f"- Source: {result.source}",
        f"- Universe size: {result.universe_size}",
        f"- Universe source: {result.universe_source}",
        "",
        "## Warnings",
        "",
    ]
    if result.warnings:
        lines.extend(f"- {warning}" for warning in result.warnings)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Stock | Short | Mid | Score | Weight | Volume Risk |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    if not result.rows:
        lines.append("| NA | NA | NA | NA | NA | NA |")
    else:
        for row in result.rows:
            note = row.volume_risk_note or ("yes" if row.volume_risk_flag else "no")
            lines.append(
                f"| {row.name} ({row.symbol}) | {_to_percent(row.short_prob)} | {_to_percent(row.mid_prob)} | "
                f"{_to_float(row.score, 3)} | {_to_percent(row.suggested_weight)} | {note} |"
            )
    lines.extend(["", "## Drivers", ""])
    for row in result.rows[:10]:
        lines.append(f"### {row.name} ({row.symbol})")
        lines.append(f"- Short: {format_driver_list(row.short_drivers)}")
        lines.append(f"- Mid: {format_driver_list(row.mid_drivers)}")
        lines.append("")
    return _write_lines(out_path, lines)


def write_v2_daily_report_from_view_model(out_path: str | Path, view_model: DailyReportViewModel) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_daily_markdown(view_model), encoding="utf-8")
    return report_path


def write_v2_research_report_from_view_model(out_path: str | Path, view_model: ResearchReportViewModel) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_research_markdown(view_model), encoding="utf-8")
    return report_path


def write_v2_research_report(
    out_path: str | Path,
    *,
    strategy_id: str,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    artifacts: dict[str, str] | None = None,
) -> Path:
    view_model = build_research_report_view_model(
        strategy_id=strategy_id,
        baseline=baseline,
        calibration=calibration,
        learning=learning,
        artifacts=artifacts,
    )
    return write_v2_research_report_from_view_model(out_path, view_model)


def write_v2_daily_report(out_path: str | Path, result: V2DailyRunResult) -> Path:
    view_model = build_daily_report_view_model(result)
    return write_v2_daily_report_from_view_model(out_path, view_model)
