from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.application.v2_services import (
    persist_v2_info_research_artifacts,
    persist_v2_ranking_research_artifacts,
    run_v2_info_research,
    run_v2_ranking_research,
    summarize_daily_run,
    summarize_v2_backtest,
    summarize_v2_calibration,
    summarize_v2_info_research,
    summarize_v2_ranking_research,
    summarize_v2_policy_learning,
)
from src.application.v2_workflow import (
    build_daily_run_blueprint,
    build_info_research_blueprint,
    build_ranking_research_blueprint,
    build_research_run_blueprint,
    describe_v2_stack,
)
from src.artifact_registry.v2_registry import (
    load_published_v2_policy_model,
    publish_v2_research_artifacts,
)
from src.contracts.runtime import DailyRunOptions, InfoResearchOptions, RankingResearchOptions, ResearchMatrixOptions, ResearchRunOptions
from src.infrastructure.market_data import set_tushare_token
from src.interfaces.presenters.html_dashboard import write_v2_daily_dashboard, write_v2_research_dashboard
from src.interfaces.presenters.markdown_reports import write_v2_daily_report, write_v2_research_report
from src.interfaces.presenters.v2_ranking_research_presenters import (
    write_v2_ranking_research_dashboard,
    write_v2_ranking_research_report,
)
from src.interfaces.presenters.v2_info_research_presenters import (
    write_v2_info_research_dashboard,
    write_v2_info_research_report,
)
from src.workflows.daily_workflow import run_daily_v2_live
from src.workflows.research_workflow import last_research_trajectory, run_v2_research_matrix, run_v2_research_workflow


def _add_runtime_identity_args(parser: argparse.ArgumentParser, *, strategy_help: str) -> None:
    parser.add_argument("--strategy", default="swing_v2", help=strategy_help)
    parser.add_argument("--config", default="config/api.json", help="Runtime config path")
    parser.add_argument("--source", default=None, help="Optional source override")
    parser.add_argument("--tushare-token", dest="tushare_token", default=None, help="Optional Tushare token override")


def _add_universe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--universe-tier", dest="universe_tier", default=None, help="Optional predefined universe tier override")
    parser.add_argument("--universe-file", dest="universe_file", default=None, help="Optional universe file override")
    parser.add_argument("--universe-limit", dest="universe_limit", type=int, default=None, help="Optional universe size override")
    parser.add_argument("--dynamic-universe", dest="dynamic_universe", action="store_true", default=None, help="Enable dynamic universe generation")
    parser.add_argument("--no-dynamic-universe", dest="dynamic_universe", action="store_false", help="Disable dynamic universe generation")
    parser.add_argument("--generator-target-size", dest="generator_target_size", type=int, default=None, help="Dynamic universe target size")
    parser.add_argument("--generator-coarse-size", dest="generator_coarse_size", type=int, default=None, help="Dynamic universe coarse pool size")
    parser.add_argument("--generator-theme-aware", dest="generator_theme_aware", action="store_true", default=None, help="Enable theme-aware generator quotas")
    parser.add_argument("--no-generator-theme-aware", dest="generator_theme_aware", action="store_false", help="Disable theme-aware generator quotas")
    parser.add_argument("--generator-use-concepts", dest="generator_use_concepts", action="store_true", default=None, help="Use concept metadata in dynamic generator")
    parser.add_argument("--no-generator-use-concepts", dest="generator_use_concepts", action="store_false", help="Disable concept metadata in dynamic generator")


def _add_info_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--info-file", dest="info_file", default=None, help="Optional structured info file or directory override")
    parser.add_argument("--info-lookback-days", dest="info_lookback_days", type=int, default=None, help="Info lookback window")
    parser.add_argument("--info-half-life-days", dest="info_half_life_days", type=float, default=None, help="Info half life")
    parser.add_argument(
        "--info-cutoff-time",
        dest="info_cutoff_time",
        default=None,
        help="Decision cutoff time for same-day info availability, e.g. 15:00:00 or 23:59:59",
    )
    parser.add_argument("--use-info-fusion", dest="use_info_fusion", action="store_true", default=None, help="Enable info shadow evaluation")
    parser.add_argument("--no-use-info-fusion", dest="use_info_fusion", action="store_false", help="Disable info shadow evaluation")
    parser.add_argument(
        "--use-learned-info-fusion",
        dest="use_learned_info_fusion",
        action="store_true",
        default=None,
        help="Enable learned info shadow fitting and reporting",
    )
    parser.add_argument(
        "--no-use-learned-info-fusion",
        dest="use_learned_info_fusion",
        action="store_false",
        help="Disable learned info shadow fitting and reporting",
    )
    parser.add_argument("--info-shadow-only", dest="info_shadow_only", action="store_true", default=None, help="Keep info in shadow-only mode")
    parser.add_argument("--no-info-shadow-only", dest="info_shadow_only", action="store_false", help="Disable shadow-only flag")
    parser.add_argument("--info-types", dest="info_types", default=None, help="Comma-separated info types")
    parser.add_argument("--info-source-mode", dest="info_source_mode", default=None, choices=["layered", "legacy"], help="Info input mode")
    parser.add_argument("--info-subsets", dest="info_subsets", default=None, help="Comma-separated info subsets")


def _add_external_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--external-signals", dest="external_signals", action="store_true", default=None, help="Enable external signal overlay")
    parser.add_argument("--no-external-signals", dest="external_signals", action="store_false", help="Disable external signal overlay")
    parser.add_argument("--event-file", dest="event_file", default=None, help="Optional event/news/announcement input override")
    parser.add_argument("--capital-flow-file", dest="capital_flow_file", default=None, help="Optional capital flow input override")
    parser.add_argument("--macro-file", dest="macro_file", default=None, help="Optional macro context input override")
    parser.add_argument("--use-us-index-context", dest="use_us_index_context", action="store_true", default=None, help="Enable US index context features")
    parser.add_argument("--no-use-us-index-context", dest="use_us_index_context", action="store_false", help="Disable US index context features")
    parser.add_argument("--us-index-source", dest="us_index_source", default=None, choices=["akshare"], help="US index feature source")


def _add_history_window_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", dest="start_date", default=None, help="Optional inclusive backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", dest="end_date", default=None, help="Optional inclusive backtest end date (YYYY-MM-DD)")
    parser.add_argument("--lookback-years", dest="lookback_years", type=int, default=None, help="Optional rolling history length in years; resolves against end-date or today when end is open-ended")


def _add_output_args(parser: argparse.ArgumentParser, *, report_default: str, dashboard_default: str, report_help: str) -> None:
    parser.add_argument("--report", default=report_default, help=report_help)
    parser.add_argument("--dashboard", default=dashboard_default, help="HTML dashboard output path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quant architecture V2 CLI scaffold")
    sub = parser.add_subparsers(dest="task", required=True)

    sub.add_parser("describe", help="Print the V2 architecture summary")

    daily = sub.add_parser("daily-run", help="Print the V2 production workflow stages")
    _add_runtime_identity_args(daily, strategy_help="Strategy snapshot id")
    _add_universe_args(daily)
    _add_info_args(daily)
    _add_external_context_args(daily)
    _add_output_args(
        daily,
        report_default="reports/v2_daily_report.md",
        dashboard_default="reports/v2_daily_dashboard.html",
        report_help="Markdown report output path",
    )
    daily.add_argument("--artifact-root", default="artifacts/v2", help="Published artifact root for learned policy snapshots")
    daily.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for daily-run results")
    daily.add_argument("--refresh-cache", action="store_true", help="Ignore existing daily-run cache and rebuild")
    daily.add_argument("--run-id", default=None, help="Pinned research run_id to consume")
    daily.add_argument("--snapshot-path", default=None, help="Pinned research manifest path (file or directory)")
    daily.add_argument("--allow-retrain", action="store_true", help="Allow daily-run to retrain forecasts (default: false)")
    daily.add_argument(
        "--disable-learned-policy",
        action="store_true",
        help="Use the rule-based policy only and ignore published learned policy models",
    )

    research = sub.add_parser("research-run", help="Print the V2 research workflow stages")
    _add_runtime_identity_args(research, strategy_help="Target strategy id")
    _add_universe_args(research)
    _add_info_args(research)
    _add_external_context_args(research)
    _add_history_window_args(research)
    _add_output_args(
        research,
        report_default="reports/v2_research_report.md",
        dashboard_default="reports/v2_research_dashboard.html",
        report_help="Markdown report output path",
    )
    research.add_argument("--artifact-root", default="artifacts/v2", help="Artifact output root for research runs")
    research.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for prepared data and trajectories")
    research.add_argument("--refresh-cache", action="store_true", help="Ignore existing cached trajectory and rebuild it")
    research.add_argument("--retrain-days", type=int, default=20, help="Trajectory retraining cadence in trading days")
    research.add_argument("--forecast-backend", default="linear", help="Forecast backend id for research backtests (linear/deep/hybrid)")
    research.add_argument("--training-window-days", type=int, default=480, help="Rolling training window in trading days; use 0 to keep expanding history")
    research.add_argument("--light", action="store_true", help="Run baseline-only light research mode")
    research.add_argument("--skip-calibration", action="store_true", help="Skip policy calibration stage")
    research.add_argument("--skip-learning", action="store_true", help="Skip learned policy stage")
    research.add_argument("--split-mode", default="purged_wf", choices=["purged_wf", "simple"], help="Research split mode")
    research.add_argument("--embargo-days", type=int, default=20, help="Embargo days for purged walk-forward split")
    research.add_argument("--publish-forecast-models", dest="publish_forecast_models", action="store_true", default=True, help="Publish forecast-layer metadata and frozen state snapshot")
    research.add_argument("--no-publish-forecast-models", dest="publish_forecast_models", action="store_false", help="Skip publishing forecast-layer metadata")

    ranking = sub.add_parser("ranking-research-run", help="Run fast leader/ranking diagnostics without full policy backtest")
    _add_runtime_identity_args(ranking, strategy_help="Target strategy id")
    _add_universe_args(ranking)
    _add_info_args(ranking)
    _add_external_context_args(ranking)
    _add_history_window_args(ranking)
    _add_output_args(
        ranking,
        report_default="reports/v2_ranking_research_report.md",
        dashboard_default="reports/v2_ranking_research_dashboard.html",
        report_help="Markdown ranking report output path",
    )
    ranking.add_argument("--artifact-root", default="artifacts/v2", help="Artifact output root for ranking research runs")
    ranking.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for prepared data and trajectories")
    ranking.add_argument("--refresh-cache", action="store_true", help="Ignore existing cached trajectory and rebuild it")
    ranking.add_argument("--retrain-days", type=int, default=20, help="Trajectory retraining cadence in trading days")
    ranking.add_argument("--forecast-backend", default="linear", help="Forecast backend id for ranking diagnostics")
    ranking.add_argument("--training-window-days", type=int, default=480, help="Rolling training window in trading days; use 0 to keep expanding history")
    ranking.add_argument("--split-mode", default="purged_wf", choices=["purged_wf", "simple"], help="Research split mode")
    ranking.add_argument("--embargo-days", type=int, default=20, help="Embargo days for purged walk-forward split")
    ranking.add_argument("--top-k", type=int, default=3, help="Top-k used for leader ranking metrics")
    ranking.add_argument("--candidate-limit", type=int, default=16, help="Number of leader candidates to render")
    ranking.add_argument("--leader-min-theme-size", type=int, default=3, help="Minimum theme size for leader labels")
    ranking.add_argument("--exit-min-theme-size", type=int, default=2, help="Minimum theme size for exit labels")
    ranking.add_argument("--exit-candidate-limit", type=int, default=8, help="Candidate cap when building exit labels")
    ranking.add_argument("--signal-l2", type=float, default=1.0, help="L2 strength for lightweight ridge signal models")

    info_research = sub.add_parser("info-research-run", help="Run standalone info-layer diagnostics without policy backtest")
    _add_runtime_identity_args(info_research, strategy_help="Target strategy id")
    _add_universe_args(info_research)
    _add_info_args(info_research)
    _add_external_context_args(info_research)
    _add_history_window_args(info_research)
    _add_output_args(
        info_research,
        report_default="reports/v2_info_research_report.md",
        dashboard_default="reports/v2_info_research_dashboard.html",
        report_help="Markdown info research report output path",
    )
    info_research.add_argument("--artifact-root", default="artifacts/v2", help="Artifact output root for info research runs")
    info_research.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for prepared data and trajectories")
    info_research.add_argument("--refresh-cache", action="store_true", help="Ignore existing cached trajectory and rebuild it")
    info_research.add_argument("--retrain-days", type=int, default=20, help="Trajectory retraining cadence in trading days")
    info_research.add_argument("--forecast-backend", default="linear", help="Forecast backend id for info diagnostics")
    info_research.add_argument("--training-window-days", type=int, default=480, help="Rolling training window in trading days; use 0 to keep expanding history")
    info_research.add_argument("--split-mode", default="purged_wf", choices=["purged_wf", "simple"], help="Research split mode")
    info_research.add_argument("--embargo-days", type=int, default=20, help="Embargo days for purged walk-forward split")
    info_research.add_argument("--horizons", default="1d,2d,3d,5d,10d,20d", help="Comma-separated horizons to evaluate")
    info_research.add_argument("--min-tag-count", type=int, default=5, help="Minimum item count required for event tag diagnostics")
    info_research.add_argument("--max-tag-count", type=int, default=12, help="Maximum number of event tags to render")

    matrix = sub.add_parser("research-matrix", help="Run the fixed 16/80/150/300 universe matrix")
    _add_runtime_identity_args(matrix, strategy_help="Target strategy id")
    _add_history_window_args(matrix)
    _add_output_args(
        matrix,
        report_default="reports/v2_research_report.md",
        dashboard_default="reports/v2_research_dashboard.html",
        report_help="Markdown report output path for the last run",
    )
    matrix.add_argument("--artifact-root", default="artifacts/v2", help="Artifact output root for research runs")
    matrix.add_argument("--cache-root", default="artifacts/v2/cache", help="On-disk cache root for prepared data and trajectories")
    matrix.add_argument("--refresh-cache", action="store_true", help="Ignore existing cached trajectory and rebuild it")
    matrix.add_argument("--retrain-days", type=int, default=20, help="Trajectory retraining cadence in trading days")
    matrix.add_argument("--forecast-backend", default="linear", help="Forecast backend id for research backtests (linear/deep/hybrid)")
    matrix.add_argument("--training-window-days", type=int, default=480, help="Rolling training window in trading days; use 0 to keep expanding history")
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
    token_override = getattr(args, "tushare_token", None)
    if token_override is not None and str(token_override).strip():
        set_tushare_token(str(token_override))

    if args.task == "describe":
        print(describe_v2_stack())
        print("Use `research-run` to build artifacts and `daily-run` to consume published strategy snapshots.")
        return 0

    if args.task == "daily-run":
        options = DailyRunOptions.from_namespace(args)
        bp = build_daily_run_blueprint()
        _print_blueprint(
            bp.name,
            options.strategy_id,
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        result = run_daily_v2_live(options=options)
        published_model = load_published_v2_policy_model(
            strategy_id=options.strategy_id,
            artifact_root=options.artifact_root,
        )
        report_path = write_v2_daily_report(str(args.report), result)
        dashboard_path = write_v2_daily_dashboard(str(args.dashboard), result)
        print(f"[V2] daily-run report: {Path(report_path).resolve()}")
        print(f"[V2] daily-run dashboard: {Path(dashboard_path).resolve()}")
        if published_model is not None:
            snapshot_path = Path(options.artifact_root).resolve() / options.strategy_id / "latest_policy_model.json"
            print(f"[V2] daily-run policy snapshot: {snapshot_path}")
        print("[V2] daily-run summary:")
        print(json.dumps(summarize_daily_run(result), ensure_ascii=False, indent=2))
        return 0

    if args.task == "research-run":
        options = ResearchRunOptions.from_namespace(args)
        bp = build_research_run_blueprint()
        _print_blueprint(
            bp.name,
            options.strategy_id,
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        baseline, calibration, learning = run_v2_research_workflow(options=options)
        artifacts = None
        if not options.skip_learning:
            artifacts = publish_v2_research_artifacts(
                options=options,
                baseline=baseline,
                calibration=calibration,
                learning=learning,
                trajectory=last_research_trajectory(),
            )
        report_path = write_v2_research_report(
            str(args.report),
            strategy_id=options.strategy_id,
            baseline=baseline,
            calibration=calibration,
            learning=learning,
            artifacts=artifacts,
        )
        dashboard_path = write_v2_research_dashboard(
            str(args.dashboard),
            strategy_id=options.strategy_id,
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
            print("[V2] light mode: skipped research artifact publish")
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

    if args.task == "info-research-run":
        options = InfoResearchOptions.from_namespace(args)
        bp = build_info_research_blueprint()
        _print_blueprint(
            bp.name,
            options.strategy_id,
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        result = run_v2_info_research(**options.workflow_kwargs())
        artifacts = persist_v2_info_research_artifacts(result, artifact_root=options.artifact_root)
        report_path = write_v2_info_research_report(str(args.report), result)
        dashboard_path = write_v2_info_research_dashboard(str(args.dashboard), result)
        print(f"[V2] info research report: {Path(report_path).resolve()}")
        print(f"[V2] info research dashboard: {Path(dashboard_path).resolve()}")
        print(f"[V2] info research artifacts: {Path(str(artifacts['run_dir'])).resolve()}")
        print(f"[V2] info research run_id: {artifacts.get('run_id', '')}")
        print("[V2] info research summary:")
        print(json.dumps(summarize_v2_info_research(result), ensure_ascii=False, indent=2))
        return 0

    if args.task == "ranking-research-run":
        options = RankingResearchOptions.from_namespace(args)
        bp = build_ranking_research_blueprint()
        _print_blueprint(
            bp.name,
            options.strategy_id,
            [(stage.name, stage.purpose, stage.produces) for stage in bp.stages],
        )
        result = run_v2_ranking_research(**options.workflow_kwargs())
        artifacts = persist_v2_ranking_research_artifacts(
            result,
            artifact_root=options.artifact_root,
        )
        report_path = write_v2_ranking_research_report(str(args.report), result)
        dashboard_path = write_v2_ranking_research_dashboard(str(args.dashboard), result)
        print(f"[V2] ranking research report: {Path(report_path).resolve()}")
        print(f"[V2] ranking research dashboard: {Path(dashboard_path).resolve()}")
        print(f"[V2] ranking research artifacts: {Path(str(artifacts['run_dir'])).resolve()}")
        print(f"[V2] ranking research run_id: {artifacts.get('run_id', '')}")
        print("[V2] ranking research summary:")
        print(json.dumps(summarize_v2_ranking_research(result), ensure_ascii=False, indent=2))
        return 0

    if args.task == "research-matrix":
        options = ResearchMatrixOptions.from_namespace(args)
        results = run_v2_research_matrix(options=options)
        print("[V2] research matrix:")
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    raise ValueError(f"Unsupported task: {args.task}")
