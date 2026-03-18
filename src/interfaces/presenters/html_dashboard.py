from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.contracts.reporting import ResearchReportViewModel
from src.interfaces.presenters.v2_daily_dashboard_modern import (
    write_v2_daily_dashboard as write_v2_daily_dashboard,
    write_v2_daily_dashboard_from_view_model as write_v2_daily_dashboard_from_view_model,
)
from src.interfaces.presenters.v2_view_model_renderers import render_research_html
from src.reporting.view_models import build_research_report_view_model

if TYPE_CHECKING:
    from src.application.v2_contracts import (
        V2BacktestSummary,
        V2CalibrationResult,
        V2PolicyLearningResult,
    )


def write_v2_research_dashboard_from_view_model(
    out_path: str | Path,
    view_model: ResearchReportViewModel,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_research_html(view_model), encoding="utf-8")
    return path


def write_v2_research_dashboard(
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
    return write_v2_research_dashboard_from_view_model(out_path, view_model)
