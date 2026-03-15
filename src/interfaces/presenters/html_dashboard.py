from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from src.application.use_cases import DailyFusionResult
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


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def write_daily_dashboard(out_path: str | Path, result: DailyFusionResult) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    action_rows = "".join(
        "<tr>"
        f"<td>{escape(action.name or action.symbol)}</td>"
        f"<td>{escape(action.action)}</td>"
        f"<td>{_pct(action.current_weight)}</td>"
        f"<td>{_pct(action.target_weight)}</td>"
        f"<td>{_pct(action.delta_weight)}</td>"
        f"<td>{escape(action.note or 'NA')}</td>"
        "</tr>"
        for action in result.trade_actions
    ) or "<tr><td colspan='6'>No trade action.</td></tr>"

    candidate_rows = "".join(
        "<tr>"
        f"<td>{escape(row.name)} ({escape(row.symbol)})</td>"
        f"<td>{_pct(row.final_short)}</td>"
        f"<td>{_pct(row.final_mid)}</td>"
        f"<td>{row.final_score:.3f}</td>"
        f"<td>{_pct(row.suggested_weight)}</td>"
        "</tr>"
        for row in result.blended_rows[:20]
    ) or "<tr><td colspan='5'>No candidates.</td></tr>"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daily Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #f5f5f5; color: #1f2937; }}
    .page {{ max-width: 1100px; margin: 0 auto; }}
    .card {{ background: #fff; border: 1px solid #d1d5db; border-radius: 16px; padding: 20px; margin-bottom: 16px; }}
    .grid {{ display: grid; gap: 16px; grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #e5e7eb; font-size: 14px; }}
    th {{ font-size: 12px; text-transform: uppercase; color: #6b7280; }}
    @media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h1>Daily Dashboard</h1>
      <p>Date: {escape(str(result.as_of_date.date()))}</p>
      <p>Market state: {escape(result.market_state_label)}</p>
      <p>Strategy template: {escape(result.strategy_template)}</p>
    </div>
    <div class="grid">
      <div class="card"><strong>Exposure</strong><div>{_pct(result.effective_total_exposure)}</div></div>
      <div class="card"><strong>Market Short</strong><div>{_pct(result.market_final_short)}</div></div>
      <div class="card"><strong>Market Mid</strong><div>{_pct(result.market_final_mid)}</div></div>
    </div>
    <div class="card">
      <h2>Trade Plan</h2>
      <table>
        <thead><tr><th>Name</th><th>Action</th><th>Current</th><th>Target</th><th>Delta</th><th>Note</th></tr></thead>
        <tbody>{action_rows}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>Candidates</h2>
      <table>
        <thead><tr><th>Name</th><th>Final Short</th><th>Final Mid</th><th>Score</th><th>Weight</th></tr></thead>
        <tbody>{candidate_rows}</tbody>
      </table>
    </div>
  </div>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    return path


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
