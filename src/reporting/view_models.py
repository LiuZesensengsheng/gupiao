from __future__ import annotations

from src.application.v2_contracts import DailyRunResult, V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult
from src.contracts.reporting import DailyReportViewModel, ResearchReportViewModel
from src.review_analytics.summaries import (
    summarize_daily_run,
    summarize_v2_backtest,
    summarize_v2_calibration,
    summarize_v2_policy_learning,
)


def build_daily_report_view_model(result: DailyRunResult) -> DailyReportViewModel:
    summary = summarize_daily_run(result)
    return DailyReportViewModel(
        strategy_id=str(summary["strategy_id"]),
        run_id=str(summary["run_id"]),
        strategy_mode=str(summary["strategy_mode"]),
        risk_regime=str(summary["risk_regime"]),
        external_signal_enabled=bool(summary["external_signal_enabled"]),
        trade_plan=list(summary["trade_plan"]),
        policy=dict(summary["policy"]),
        market=dict(summary["market"]),
    )


def build_research_report_view_model(
    *,
    strategy_id: str,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    artifacts: dict[str, str] | None = None,
) -> ResearchReportViewModel:
    release_gate_passed = str((artifacts or {}).get("release_gate_passed", "false")).strip().lower() == "true"
    return ResearchReportViewModel(
        strategy_id=str(strategy_id),
        run_id=str((artifacts or {}).get("run_id", "")),
        release_gate_passed=release_gate_passed,
        baseline=summarize_v2_backtest(baseline),
        calibration=summarize_v2_calibration(calibration),
        learned=summarize_v2_policy_learning(learning),
    )
