from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.application.v2_services import (
    load_published_v2_policy_model,
    publish_v2_research_artifacts,
    run_daily_v2_live,
    run_v2_research_matrix,
    run_v2_research_workflow,
    summarize_daily_run,
    summarize_v2_backtest,
    summarize_v2_calibration,
    summarize_v2_policy_learning,
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
    daily.add_argument("--universe-tier", dest="universe_tier", default=None, help="Optional predefined universe tier override")
    daily.add_argument("--universe-file", dest="universe_file", default=None, help="Optional universe file override")
    daily.add_argument("--universe-limit", dest="universe_limit", type=int, default=None, help="Optional universe size override")
    daily.add_argument("--report", default="reports/v2_daily_report.md", help="Markdown report output path")
    daily.add_argument("--dashboard", default="reports/v2_daily_dashboard.html", help="HTML dashboard output path")
    daily.add_argument("--artifact-root", default="artifacts/v2", help="Published artifact root for learned policy snapshots")
    daily.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for daily-run results")
    daily.add_argument("--refresh-cache", action="store_true", help="Ignore existing daily-run cache and rebuild")
    daily.add_argument("--run-id", default=None, help="Pinned research run_id to consume")
    daily.add_argument("--snapshot-path", default=None, help="Pinned research manifest path (file or directory)")
    daily.add_argument("--allow-retrain", action="store_true", help="Allow daily-run to retrain forecasts (default: false)")

    research = sub.add_parser("research-run", help="Print the V2 research workflow stages")
    research.add_argument("--strategy", default="swing_v2", help="Target strategy id")
    research.add_argument("--config", default="config/api.json", help="Runtime config path")
    research.add_argument("--source", default=None, help="Optional source override")
    research.add_argument("--universe-tier", dest="universe_tier", default=None, help="Optional predefined universe tier override")
    research.add_argument("--universe-file", dest="universe_file", default=None, help="Optional universe file override")
    research.add_argument("--universe-limit", dest="universe_limit", type=int, default=None, help="Optional universe size override")
    research.add_argument("--report", default="reports/v2_research_report.md", help="Markdown report output path")
    research.add_argument("--dashboard", default="reports/v2_research_dashboard.html", help="HTML dashboard output path")
    research.add_argument("--artifact-root", default="artifacts/v2", help="Artifact output root for research runs")
    research.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for prepared data and trajectories")
    research.add_argument("--refresh-cache", action="store_true", help="Ignore existing cached trajectory and rebuild it")
    research.add_argument("--forecast-backend", default="linear", help="Forecast backend id for research backtests (linear/deep)")
    research.add_argument("--light", action="store_true", help="Run baseline-only light research mode")
    research.add_argument("--skip-calibration", action="store_true", help="Skip policy calibration stage")
    research.add_argument("--skip-learning", action="store_true", help="Skip learned policy stage")
    research.add_argument("--split-mode", default="purged_wf", choices=["purged_wf", "simple"], help="Research split mode")
    research.add_argument("--embargo-days", type=int, default=20, help="Embargo days for purged walk-forward split")
    research.add_argument("--publish-forecast-models", dest="publish_forecast_models", action="store_true", default=True, help="Publish forecast-layer metadata and frozen state snapshot")
    research.add_argument("--no-publish-forecast-models", dest="publish_forecast_models", action="store_false", help="Skip publishing forecast-layer metadata")

    matrix = sub.add_parser("research-matrix", help="Run the fixed 16/80/150/300 universe matrix")
    matrix.add_argument("--strategy", default="swing_v2", help="Target strategy id")
    matrix.add_argument("--config", default="config/api.json", help="Runtime config path")
    matrix.add_argument("--source", default=None, help="Optional source override")
    matrix.add_argument("--report", default="reports/v2_research_report.md", help="Markdown report output path for the last run")
    matrix.add_argument("--dashboard", default="reports/v2_research_dashboard.html", help="HTML dashboard output path for the last run")
    matrix.add_argument("--artifact-root", default="artifacts/v2", help="Artifact output root for research runs")
    matrix.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for prepared data and trajectories")
    matrix.add_argument("--refresh-cache", action="store_true", help="Ignore existing cached trajectory and rebuild it")
    matrix.add_argument("--forecast-backend", default="linear", help="Forecast backend id for research backtests (linear/deep)")
    matrix.add_argument("--split-mode", default="purged_wf", choices=["purged_wf", "simple"], help="Research split mode")
    matrix.add_argument("--embargo-days", type=int, default=20, help="Embargo days for purged walk-forward split")
    matrix.add_argument("--tiers", nargs="*", default=["favorites_16", "generated_80", "generated_150", "generated_300"], help="Universe tiers to evaluate")

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
            universe_tier=args.universe_tier,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
            artifact_root=str(args.artifact_root),
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            run_id=args.run_id,
            snapshot_path=args.snapshot_path,
            allow_retrain=bool(args.allow_retrain),
        )
        published_model = load_published_v2_policy_model(
            strategy_id=str(args.strategy),
            artifact_root=str(args.artifact_root),
        )
        report_path = write_v2_daily_report(str(args.report), result)
        dashboard_path = write_v2_daily_dashboard(str(args.dashboard), result)
        print(f"[V2] daily-run report: {Path(report_path).resolve()}")
        print(f"[V2] daily-run dashboard: {Path(dashboard_path).resolve()}")
        if published_model is not None:
            print(f"[V2] daily-run policy snapshot: {Path(str(args.artifact_root)).resolve() / str(args.strategy) / 'latest_policy_model.json'}")
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
        skip_calibration = bool(args.light or args.skip_calibration)
        skip_learning = bool(args.light or args.skip_learning)
        baseline, calibration, learning = run_v2_research_workflow(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            universe_tier=args.universe_tier,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
            skip_calibration=skip_calibration,
            skip_learning=skip_learning,
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            forecast_backend=str(args.forecast_backend),
            split_mode=str(args.split_mode),
            embargo_days=int(args.embargo_days),
        )
        artifacts = None
        if not skip_learning:
            artifacts = publish_v2_research_artifacts(
                strategy_id=str(args.strategy),
                artifact_root=str(args.artifact_root),
                config_path=str(args.config),
                source=args.source,
                universe_tier=args.universe_tier,
                universe_file=args.universe_file,
                universe_limit=args.universe_limit,
                baseline=baseline,
                calibration=calibration,
                learning=learning,
                cache_root=str(args.cache_root),
                forecast_backend=str(args.forecast_backend),
                publish_forecast_models=bool(args.publish_forecast_models),
                split_mode=str(args.split_mode),
                embargo_days=int(args.embargo_days),
            )
        report_path = write_v2_research_report(
            str(args.report),
            strategy_id=str(args.strategy),
            baseline=baseline,
            calibration=calibration,
            learning=learning,
            artifacts=artifacts,
        )
        dashboard_path = write_v2_research_dashboard(
            str(args.dashboard),
            strategy_id=str(args.strategy),
            baseline=baseline,
            calibration=calibration,
            learning=learning,
            artifacts=artifacts,
        )
        print(f"[V2] research report: {Path(report_path).resolve()}")
        print(f"[V2] research dashboard: {Path(dashboard_path).resolve()}")
        if artifacts is not None:
            print(f"[V2] research artifacts: {Path(str(artifacts['run_dir'])).resolve()}")
            print(f"[V2] research run_id: {artifacts.get('run_id', '')}")
            print(f"[V2] release gate passed: {artifacts.get('release_gate_passed', 'false')}")
        else:
            print("[V2] 轻量模式: 已跳过研究产物发布")
        print("[V2] research baseline backtest:")
        print(
            json.dumps(
                summarize_v2_backtest(
                    baseline,
                    run_id=None if artifacts is None else str(artifacts.get("run_id", "")),
                    snapshot_hash=None if artifacts is None else str(artifacts.get("snapshot_hash", "")),
                    config_hash=None if artifacts is None else str(artifacts.get("config_hash", "")),
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        print("[V2] policy calibration:")
        print(json.dumps(summarize_v2_calibration(calibration), ensure_ascii=False, indent=2))
        print("[V2] learned policy:")
        print(json.dumps(summarize_v2_policy_learning(learning), ensure_ascii=False, indent=2))
        return 0

    if args.task == "research-matrix":
        results = run_v2_research_matrix(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            cache_root=str(args.cache_root),
            artifact_root=str(args.artifact_root),
            refresh_cache=bool(args.refresh_cache),
            forecast_backend=str(args.forecast_backend),
            split_mode=str(args.split_mode),
            embargo_days=int(args.embargo_days),
            universe_tiers=[str(item) for item in args.tiers],
        )
        print("[V2] research matrix:")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unsupported task: {args.task}")
