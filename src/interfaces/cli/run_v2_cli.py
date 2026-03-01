from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.application.v2_services import (
    calibrate_v2_policy,
    run_daily_v2_live,
    run_v2_backtest_live,
    summarize_daily_run,
    summarize_v2_backtest,
    summarize_v2_calibration,
)
from src.application.v2_workflow import (
    build_daily_run_blueprint,
    build_research_run_blueprint,
    describe_v2_stack,
)
from src.interfaces.presenters.html_dashboard import write_v2_daily_dashboard, write_v2_research_dashboard
from src.interfaces.presenters.markdown_reports import write_v2_daily_report, write_v2_research_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant architecture V2 CLI scaffold")
    sub = parser.add_subparsers(dest="task", required=True)

    sub.add_parser("describe", help="Print the V2 architecture summary")

    daily = sub.add_parser("daily-run", help="Print the V2 production workflow stages")
    daily.add_argument("--strategy", default="swing_v2", help="Strategy snapshot id")
    daily.add_argument("--config", default="config/api.json", help="Runtime config path for live mode")
    daily.add_argument("--source", default=None, help="Optional source override")
    daily.add_argument("--universe-file", dest="universe_file", default=None, help="Optional universe file override")
    daily.add_argument("--universe-limit", dest="universe_limit", type=int, default=None, help="Optional universe size override")
    daily.add_argument("--report", default="reports/v2_daily_report.md", help="Markdown report output path")
    daily.add_argument("--dashboard", default="reports/v2_daily_dashboard.html", help="HTML dashboard output path")

    research = sub.add_parser("research-run", help="Print the V2 research workflow stages")
    research.add_argument("--strategy", default="swing_v2", help="Target strategy id")
    research.add_argument("--config", default="config/api.json", help="Runtime config path")
    research.add_argument("--source", default=None, help="Optional source override")
    research.add_argument("--universe-file", dest="universe_file", default=None, help="Optional universe file override")
    research.add_argument("--universe-limit", dest="universe_limit", type=int, default=None, help="Optional universe size override")
    research.add_argument("--report", default="reports/v2_research_report.md", help="Markdown report output path")
    research.add_argument("--dashboard", default="reports/v2_research_dashboard.html", help="HTML dashboard output path")

    return parser


def _print_blueprint(title: str, strategy: str, stages: list[tuple[str, str, str]]) -> None:
    print(f"[V2] {title}: strategy={strategy}")
    for idx, (name, purpose, produces) in enumerate(stages, start=1):
        print(f"{idx}. {name}")
        print(f"   purpose: {purpose}")
        print(f"   produces: {produces}")


def main() -> int:
    args = build_parser().parse_args()

    if args.task == "describe":
        print(describe_v2_stack())
        print("Use `research-run` to build artifacts and `daily-run` to consume published strategy snapshots.")
        return 0

    if args.task == "daily-run":
        bp = build_daily_run_blueprint()
        _print_blueprint(
            bp.name,
            str(args.strategy),
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        result = run_daily_v2_live(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
        )
        report_path = write_v2_daily_report(str(args.report), result)
        dashboard_path = write_v2_daily_dashboard(str(args.dashboard), result)
        print(f"[V2] daily-run report: {Path(report_path).resolve()}")
        print(f"[V2] daily-run dashboard: {Path(dashboard_path).resolve()}")
        print("[V2] daily-run summary:")
        print(json.dumps(summarize_daily_run(result), ensure_ascii=False, indent=2))
        return 0

    if args.task == "research-run":
        bp = build_research_run_blueprint()
        _print_blueprint(
            bp.name,
            str(args.strategy),
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        baseline = run_v2_backtest_live(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
        )
        calibration = calibrate_v2_policy(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
        )
        report_path = write_v2_research_report(
            str(args.report),
            strategy_id=str(args.strategy),
            baseline=baseline,
            calibration=calibration,
        )
        dashboard_path = write_v2_research_dashboard(
            str(args.dashboard),
            strategy_id=str(args.strategy),
            baseline=baseline,
            calibration=calibration,
        )
        print(f"[V2] research report: {Path(report_path).resolve()}")
        print(f"[V2] research dashboard: {Path(dashboard_path).resolve()}")
        print("[V2] research baseline backtest:")
        print(json.dumps(summarize_v2_backtest(baseline), ensure_ascii=False, indent=2))
        print("[V2] policy calibration:")
        print(json.dumps(summarize_v2_calibration(calibration), ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unsupported task: {args.task}")
