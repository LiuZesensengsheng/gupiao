from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.contracts.reporting import DailyReportViewModel, ResearchReportViewModel
from src.interfaces.presenters.v2_view_model_renderers import render_daily_markdown, render_research_markdown
from src.reporting.view_models import build_daily_report_view_model, build_research_report_view_model

if TYPE_CHECKING:
    from src.application.v2_contracts import (
        DailyRunResult as V2DailyRunResult,
        V2BacktestSummary,
        V2CalibrationResult,
        V2PolicyLearningResult,
    )


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
