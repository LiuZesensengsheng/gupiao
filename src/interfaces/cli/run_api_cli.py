from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src.application.use_cases import generate_daily_fusion, generate_discovery, generate_forecast
from src.application.watchlist import load_watchlist
from src.domain.symbols import normalize_symbol
from src.infrastructure.data_sync import sync_market_data
from src.infrastructure.discovery import build_candidate_universe
from src.infrastructure.discovery import _load_universe_file as _load_discovery_universe_file
from src.infrastructure.margin_sync import sync_margin_data
from src.infrastructure.market_data import DataError, set_tushare_token
from src.interfaces.cli import legacy_cli_tasks as _legacy_cli_tasks
from src.interfaces.cli.legacy_cli_options import (
    DEFAULT_COMMON,
    DEFAULT_TASK,
    _coalesce,
    _masked_settings,
    _parse_bool,
    _parse_float_list,
    _parse_int_list,
    _parse_symbol_list,
    _parse_years,
    _read_config_section,
    _read_json_config,
    _resolve_settings,
)
from src.interfaces.presenters.html_dashboard import write_daily_dashboard
from src.interfaces.presenters.markdown_reports import write_daily_report, write_discovery_report, write_forecast_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified A-share API CLI")
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument("--config", default="config/api.json", help="Path to unified API JSON config")
    config_parent.add_argument(
        "--tushare-token",
        dest="tushare_token",
        default=None,
        help="Tushare token (or set env TUSHARE_TOKEN)",
    )
    config_parent.add_argument(
        "--print-effective-config",
        action="store_true",
        help="Print merged runtime config and exit",
    )
    config_parent.add_argument(
        "--max-positions",
        dest="max_positions",
        type=int,
        default=None,
        help="Max simultaneous stock positions for weight allocation",
    )
    config_parent.add_argument(
        "--use-margin-features",
        dest="use_margin_features",
        choices=["true", "false"],
        default=None,
        help="Enable margin financing/securities lending feature module",
    )
    config_parent.add_argument(
        "--margin-market-file",
        dest="margin_market_file",
        default=None,
        help="CSV path for market-level margin data",
    )
    config_parent.add_argument(
        "--margin-stock-file",
        dest="margin_stock_file",
        default=None,
        help="CSV path for stock-level margin data",
    )
    config_parent.add_argument(
        "--use-us-index-context",
        dest="use_us_index_context",
        choices=["true", "false"],
        default=None,
        help="Enable US index context features for A-share training",
    )
    config_parent.add_argument(
        "--us-index-source",
        dest="us_index_source",
        choices=["akshare"],
        default=None,
        help="US index feature source",
    )

    sub = parser.add_subparsers(dest="task", required=True)

    daily = sub.add_parser("daily", parents=[config_parent], help="Generate daily fusion report (quant + news)")
    daily.add_argument(
        "--source",
        default=None,
        help="Data source: eastmoney/tushare/akshare/baostock/local/auto or comma chain",
    )
    daily.add_argument("--watchlist", default=None, help="Watchlist JSON path")
    daily.add_argument("--data-dir", dest="data_dir", default=None, help="Directory for local CSV when source=local")
    daily.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    daily.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    daily.add_argument("--min-train-days", dest="min_train_days", type=int, default=None, help="Min train days")
    daily.add_argument("--step-days", dest="step_days", type=int, default=None, help="Walk-forward test block size")
    daily.add_argument("--l2", type=float, default=None, help="L2 regularization strength")
    daily.add_argument("--universe-file", dest="universe_file", default=None, help="Universe file (csv/json) for large stock pool")
    daily.add_argument(
        "--universe-limit",
        dest="universe_limit",
        type=int,
        default=None,
        help="Maximum symbols loaded from universe file for daily run",
    )
    daily.add_argument(
        "--positions-file",
        dest="positions_file",
        default=None,
        help="Optional current positions file (csv/json) for trade action details",
    )
    daily.add_argument(
        "--portfolio-nav",
        dest="portfolio_nav",
        type=float,
        default=None,
        help="Optional portfolio net asset value used for amount/share estimation",
    )
    daily.add_argument(
        "--trade-lot-size",
        dest="trade_lot_size",
        type=int,
        default=None,
        help="Lot size for share estimation (A-share default 100)",
    )
    daily.add_argument(
        "--news-file",
        dest="news_file",
        default=None,
        help="News source path: single CSV file or directory of CSV partitions",
    )
    daily.add_argument("--news-lookback-days", dest="news_lookback_days", type=int, default=None, help="Lookback")
    daily.add_argument(
        "--learned-news-lookback-days",
        dest="learned_news_lookback_days",
        type=int,
        default=None,
        help="Lookback window used by learned news models",
    )
    daily.add_argument("--news-half-life-days", dest="news_half_life_days", type=float, default=None, help="Half-life")
    daily.add_argument("--market-news-strength", dest="market_news_strength", type=float, default=None, help="Blend")
    daily.add_argument("--stock-news-strength", dest="stock_news_strength", type=float, default=None, help="Blend")
    daily.add_argument(
        "--use-learned-news-fusion",
        dest="use_learned_news_fusion",
        choices=["true", "false"],
        default=None,
        help="Enable learned news impact + fusion calibration",
    )
    daily.add_argument(
        "--learned-news-min-samples",
        dest="learned_news_min_samples",
        type=int,
        default=None,
        help="Minimum samples required for learned news model",
    )
    daily.add_argument(
        "--learned-holdout-ratio",
        dest="learned_holdout_ratio",
        type=float,
        default=None,
        help="Holdout ratio for diagnostics",
    )
    daily.add_argument(
        "--learned-news-l2",
        dest="learned_news_l2",
        type=float,
        default=None,
        help="L2 regularization for news model",
    )
    daily.add_argument(
        "--learned-fusion-l2",
        dest="learned_fusion_l2",
        type=float,
        default=None,
        help="L2 regularization for fusion calibrator",
    )
    daily.add_argument("--report-date", dest="report_date", default=None, help="Override report date YYYY-MM-DD")
    daily.add_argument("--report", default=None, help="Output markdown report path")
    daily.add_argument("--dashboard", default=None, help="Output HTML dashboard path")
    daily.add_argument("--backtest-years", dest="backtest_years", default=None, help="Years list, e.g. 3,5")
    daily.add_argument("--backtest-retrain-days", dest="backtest_retrain_days", type=int, default=None, help="Model retrain interval in days")
    daily.add_argument(
        "--backtest-weight-threshold",
        dest="backtest_weight_threshold",
        type=float,
        default=None,
        help="Score threshold for portfolio weights",
    )
    daily.add_argument(
        "--backtest-time-budget-minutes",
        dest="backtest_time_budget_minutes",
        type=float,
        default=None,
        help="Max backtest runtime in minutes; <=0 means no time limit",
    )
    daily.add_argument("--commission-bps", dest="commission_bps", type=float, default=None, help="Commission in bps")
    daily.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=None, help="Slippage in bps")
    daily.add_argument(
        "--use-turnover-control",
        dest="use_turnover_control",
        choices=["true", "false"],
        default=None,
        help="Enable turnover/frequency guardrail in backtest execution",
    )
    daily.add_argument(
        "--max-trades-per-stock-per-day",
        dest="max_trades_per_stock_per_day",
        type=int,
        default=None,
        help="Maximum rebalance trades per stock per day (recommended 1 for A-share T style)",
    )
    daily.add_argument(
        "--max-trades-per-stock-per-week",
        dest="max_trades_per_stock_per_week",
        type=int,
        default=None,
        help="Maximum rebalance trades per stock in a rolling 5-day window",
    )
    daily.add_argument(
        "--min-weight-change-to-trade",
        dest="min_weight_change_to_trade",
        type=float,
        default=None,
        help="Minimum absolute weight change required to execute a trade",
    )
    daily.add_argument(
        "--range-t-sell-ret-1-min",
        dest="range_t_sell_ret_1_min",
        type=float,
        default=None,
        help="Range-state T whitelist: minimum daily return to allow reduce/sell action",
    )
    daily.add_argument(
        "--range-t-sell-price-pos-20-min",
        dest="range_t_sell_price_pos_20_min",
        type=float,
        default=None,
        help="Range-state T whitelist: minimum 20-day price position to allow reduce/sell action",
    )
    daily.add_argument(
        "--range-t-buy-ret-1-max",
        dest="range_t_buy_ret_1_max",
        type=float,
        default=None,
        help="Range-state T whitelist: maximum daily return to allow add/buy action",
    )
    daily.add_argument(
        "--range-t-buy-price-pos-20-max",
        dest="range_t_buy_price_pos_20_max",
        type=float,
        default=None,
        help="Range-state T whitelist: maximum 20-day price position to allow add/buy action",
    )
    daily.add_argument(
        "--use-tradeability-guard",
        dest="use_tradeability_guard",
        choices=["true", "false"],
        default=None,
        help="Enable tradability guard: suspension and one-price limit-up/limit-down trade block",
    )
    daily.add_argument(
        "--tradeability-limit-tolerance",
        dest="tradeability_limit_tolerance",
        type=float,
        default=None,
        help="Tolerance around daily limit rate used by tradability guard (e.g. 0.002 means 0.2%%)",
    )
    daily.add_argument(
        "--tradeability-min-volume",
        dest="tradeability_min_volume",
        type=float,
        default=None,
        help="Treat bars with volume <= this threshold as non-tradable",
    )
    daily.add_argument(
        "--limit-rule-file",
        dest="limit_rule_file",
        default=None,
        help="JSON file for price-limit rules by board/time",
    )
    daily.add_argument(
        "--use-index-constituent-guard",
        dest="use_index_constituent_guard",
        choices=["true", "false"],
        default=None,
        help="Restrict daily eligible universe by index constituent snapshots",
    )
    daily.add_argument(
        "--index-constituent-file",
        dest="index_constituent_file",
        default=None,
        help="CSV file with constituent snapshots: date,symbol[,index_symbol]",
    )
    daily.add_argument(
        "--index-constituent-symbol",
        dest="index_constituent_symbol",
        default=None,
        help="Index symbol used to filter constituent snapshot rows (default 000300.SH)",
    )
    daily.add_argument(
        "--enable-acceptance-checks",
        dest="enable_acceptance_checks",
        choices=["true", "false"],
        default=None,
        help="Run acceptance checks: state-engine A/B backtest + execution-constraint audit",
    )
    daily.add_argument(
        "--acceptance-target-years",
        dest="acceptance_target_years",
        type=int,
        default=None,
        help="Target backtest window (years) used for A/B acceptance comparison",
    )
    daily.add_argument(
        "--use-strategy-optimizer",
        dest="use_strategy_optimizer",
        choices=["true", "false"],
        default=None,
        help="Enable strategy objective search (maximize excess return with turnover/drawdown penalties)",
    )
    daily.add_argument(
        "--optimizer-retrain-days",
        dest="optimizer_retrain_days",
        default=None,
        help="Grid of retrain days, e.g. 20,30,40",
    )
    daily.add_argument(
        "--optimizer-weight-thresholds",
        dest="optimizer_weight_thresholds",
        default=None,
        help="Grid of weight thresholds, e.g. 0.50,0.55,0.60",
    )
    daily.add_argument(
        "--optimizer-max-positions",
        dest="optimizer_max_positions",
        default=None,
        help="Grid of max positions, e.g. 3,5",
    )
    daily.add_argument(
        "--optimizer-market-news-strengths",
        dest="optimizer_market_news_strengths",
        default=None,
        help="Grid of market news strengths, e.g. 0.8,1.0",
    )
    daily.add_argument(
        "--optimizer-stock-news-strengths",
        dest="optimizer_stock_news_strengths",
        default=None,
        help="Grid of stock news strengths, e.g. 1.0,1.2",
    )
    daily.add_argument(
        "--optimizer-force-full-news-strength-grid",
        dest="optimizer_force_full_news_strength_grid",
        choices=["true", "false"],
        default=None,
        help="Force full market/stock news-strength grid search even when learned news fusion is enabled",
    )
    daily.add_argument(
        "--optimizer-turnover-penalty",
        dest="optimizer_turnover_penalty",
        type=float,
        default=None,
        help="Penalty coefficient on annual turnover in objective",
    )
    daily.add_argument(
        "--optimizer-drawdown-penalty",
        dest="optimizer_drawdown_penalty",
        type=float,
        default=None,
        help="Penalty coefficient on absolute max drawdown in objective",
    )
    daily.add_argument(
        "--optimizer-target-years",
        dest="optimizer_target_years",
        type=int,
        default=None,
        help="Target window years for objective evaluation (default 3)",
    )
    daily.add_argument(
        "--optimizer-top-trials",
        dest="optimizer_top_trials",
        type=int,
        default=None,
        help="How many top optimizer trials to keep in report",
    )
    daily.add_argument(
        "--optimizer-time-budget-minutes",
        dest="optimizer_time_budget_minutes",
        type=float,
        default=None,
        help="Max optimizer runtime in minutes; <=0 means no time limit",
    )

    forecast = sub.add_parser("forecast", parents=[config_parent], help="Generate base quant forecast report")
    forecast.add_argument(
        "--source",
        default=None,
        help="Data source: eastmoney/tushare/akshare/baostock/local/auto or comma chain",
    )
    forecast.add_argument("--watchlist", default=None, help="Watchlist JSON path")
    forecast.add_argument("--data-dir", dest="data_dir", default=None, help="Directory for local CSV when source=local")
    forecast.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    forecast.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    forecast.add_argument("--min-train-days", dest="min_train_days", type=int, default=None, help="Min train days")
    forecast.add_argument("--step-days", dest="step_days", type=int, default=None, help="Walk-forward test block size")
    forecast.add_argument("--l2", type=float, default=None, help="L2 regularization strength")
    forecast.add_argument("--report", default=None, help="Output report path")

    discover = sub.add_parser("discover", parents=[config_parent], help="Discover candidate stocks for your pool")
    discover.add_argument(
        "--source",
        default=None,
        help="Data source: eastmoney/tushare/akshare/baostock/local/auto or comma chain",
    )
    discover.add_argument("--watchlist", default=None, help="Watchlist JSON path (market index + optional exclusions)")
    discover.add_argument("--data-dir", dest="data_dir", default=None, help="Data directory")
    discover.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    discover.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    discover.add_argument("--min-train-days", dest="min_train_days", type=int, default=None, help="Min train days")
    discover.add_argument("--step-days", dest="step_days", type=int, default=None, help="Walk-forward test block size")
    discover.add_argument("--l2", type=float, default=None, help="L2 regularization strength")
    discover.add_argument("--universe-file", dest="universe_file", default=None, help="Universe file (csv/json)")
    discover.add_argument("--candidate-limit", dest="candidate_limit", type=int, default=None, help="Candidate pool size before ranking")
    discover.add_argument("--top-k", dest="top_k", type=int, default=None, help="Output top-k candidates")
    discover.add_argument(
        "--exclude-watchlist",
        dest="exclude_watchlist",
        choices=["true", "false"],
        default=None,
        help="Exclude current watchlist symbols from discovered candidates",
    )
    discover.add_argument("--report", default=None, help="Output discovery report path")

    sync = sub.add_parser("sync-data", parents=[config_parent], help="Sync local A-share universe data (300-1000 stocks)")
    sync.add_argument(
        "--source",
        default=None,
        help="Data source chain used for bar download: eastmoney/tushare/akshare/baostock/local/auto",
    )
    sync.add_argument("--data-dir", dest="data_dir", default=None, help="Local data directory")
    sync.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    sync.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    sync.add_argument("--universe-size", dest="universe_size", type=int, default=None, help="Universe size target")
    sync.add_argument("--universe-file", dest="universe_file", default=None, help="Optional universe file (csv/json)")
    sync.add_argument(
        "--universe-min-amount",
        dest="universe_min_amount",
        type=float,
        default=None,
        help="Minimum成交额 filter for auto universe fetch; set 0 for full A-share coverage",
    )
    sync.add_argument(
        "--universe-exclude-st",
        dest="universe_exclude_st",
        choices=["true", "false"],
        default=None,
        help="Whether to exclude ST names in auto universe fetch",
    )
    sync.add_argument(
        "--include-indices",
        dest="include_indices",
        choices=["true", "false"],
        default=None,
        help="Also sync main index files (000300/000001/399001/399006)",
    )
    sync.add_argument(
        "--force-refresh",
        dest="force_refresh",
        choices=["true", "false"],
        default=None,
        help="Refresh files even if local data is recent",
    )
    sync.add_argument("--sleep-ms", dest="sleep_ms", type=int, default=None, help="Sleep between requests in ms")
    sync.add_argument(
        "--parallel-workers",
        dest="parallel_workers",
        type=int,
        default=None,
        help="Parallel download workers for remote sync; useful for large tushare runs",
    )
    sync.add_argument("--max-failures", dest="max_failures", type=int, default=None, help="Stop after N failures")
    sync.add_argument(
        "--write-universe-file",
        dest="write_universe_file",
        default=None,
        help="Write discovered universe json for discovery task",
    )

    sync_margin = sub.add_parser("sync-margin", parents=[config_parent], help="Sync margin financing/securities lending (两融) CSV data")
    sync_margin.add_argument(
        "--source",
        default=None,
        help="Margin source chain: akshare/tushare/auto or comma chain",
    )
    sync_margin.add_argument("--watchlist", default=None, help="Watchlist JSON path when --symbols is not provided")
    sync_margin.add_argument("--universe-file", dest="universe_file", default=None, help="Universe file (csv/json) used when --symbols is not provided")
    sync_margin.add_argument("--universe-limit", dest="universe_limit", type=int, default=None, help="Optional limit on symbols loaded from universe file")
    sync_margin.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    sync_margin.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    sync_margin.add_argument(
        "--symbols",
        default=None,
        help="Comma separated symbols, e.g. 600160.SH,000630.SZ (overrides watchlist stocks)",
    )
    sync_margin.add_argument("--sleep-ms", dest="sleep_ms", type=int, default=None, help="Sleep between requests in ms")

    return parser



def _task_dependencies() -> Any:
    return sys.modules[__name__]


def run_daily(settings: dict[str, Any]) -> int:
    return _legacy_cli_tasks.run_daily(settings, dependencies=_task_dependencies())


def run_forecast(settings: dict[str, Any]) -> int:
    return _legacy_cli_tasks.run_forecast(settings, dependencies=_task_dependencies())


def run_discover(settings: dict[str, Any]) -> int:
    return _legacy_cli_tasks.run_discover(settings, dependencies=_task_dependencies())


def run_sync_data(settings: dict[str, Any]) -> int:
    return _legacy_cli_tasks.run_sync_data(settings, dependencies=_task_dependencies())


def run_sync_margin(settings: dict[str, Any]) -> int:
    return _legacy_cli_tasks.run_sync_margin(settings, dependencies=_task_dependencies())


def _task_handlers() -> dict[str, Any]:
    return {
        "daily": run_daily,
        "forecast": run_forecast,
        "discover": run_discover,
        "sync-data": run_sync_data,
        "sync-margin": run_sync_margin,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = _read_json_config(args.config)
        settings = _resolve_settings(args, payload)
        if args.print_effective_config:
            print(json.dumps({"task": args.task, "settings": _masked_settings(settings)}, indent=2, ensure_ascii=False))
            return 0
        if args.task in {"daily", "forecast"}:
            print("[V1] Compatibility entrypoint: prefer `python3 run_v2.py daily-run` or `python3 run_v2.py research-run`.")
        handler = _task_handlers().get(args.task)
        if handler is None:
            parser.error(f"Unknown task: {args.task}")
        return int(handler(settings))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] Config failure: {exc}")
        return 4
    except DataError as exc:
        print(f"[ERROR] {exc}")
        print("Hint: try `--source auto` or chain fallback like `--source eastmoney,tushare,akshare,baostock`.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
