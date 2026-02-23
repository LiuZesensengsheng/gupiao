from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.application.config import DailyConfig, DiscoverConfig, ForecastConfig
from src.application.use_cases import generate_daily_fusion, generate_discovery, generate_forecast
from src.application.watchlist import load_watchlist
from src.domain.symbols import SymbolError, normalize_symbol
from src.infrastructure.data_sync import sync_market_data
from src.infrastructure.discovery import build_candidate_universe
from src.infrastructure.margin_sync import sync_margin_data
from src.infrastructure.market_data import DataError, set_tushare_token
from src.interfaces.presenters.html_dashboard import write_daily_dashboard
from src.interfaces.presenters.markdown_reports import write_daily_report, write_discovery_report, write_forecast_report

DEFAULT_COMMON: dict[str, Any] = {
    "source": "auto",
    "tushare_token": "",
    "watchlist": "config/watchlist.json",
    "data_dir": "data",
    "start": "2018-01-01",
    "end": "2099-12-31",
    "min_train_days": 240,
    "step_days": 20,
    "l2": 0.8,
    "max_positions": 5,
    "use_margin_features": True,
    "margin_market_file": "input/margin_market.csv",
    "margin_stock_file": "input/margin_stock.csv",
}

DEFAULT_TASK: dict[str, dict[str, Any]] = {
    "forecast": {
        "report": "reports/latest_report.md",
    },
    "discover": {
        "universe_file": "",
        "candidate_limit": 120,
        "top_k": 20,
        "exclude_watchlist": False,
        "report": "reports/discovery_report.md",
    },
    "sync-data": {
        "universe_size": 500,
        "universe_file": "",
        "universe_min_amount": 50000000.0,
        "universe_exclude_st": True,
        "include_indices": True,
        "force_refresh": False,
        "sleep_ms": 80,
        "max_failures": 100,
        "write_universe_file": "config/universe_auto.json",
    },
    "sync-margin": {
        "symbols": "",
        "sleep_ms": 80,
    },
    "daily": {
        "universe_file": "config/universe_auto_longtrain.json",
        "universe_limit": 500,
        "news_file": "input/news_parts",
        "news_lookback_days": 45,
        "learned_news_lookback_days": 720,
        "news_half_life_days": 10.0,
        "market_news_strength": 0.9,
        "stock_news_strength": 1.1,
        "use_learned_news_fusion": True,
        "learned_news_min_samples": 80,
        "learned_holdout_ratio": 0.2,
        "learned_news_l2": 0.8,
        "learned_fusion_l2": 0.6,
        "report_date": "",
        "report": "reports/daily_report.md",
        "dashboard": "reports/daily_dashboard.html",
        "backtest_years": [3, 5],
        "backtest_retrain_days": 20,
        "backtest_weight_threshold": 0.50,
        "backtest_time_budget_minutes": 0.0,
        "commission_bps": 1.5,
        "slippage_bps": 2.0,
        "use_turnover_control": True,
        "max_trades_per_stock_per_day": 1,
        "max_trades_per_stock_per_week": 3,
        "min_weight_change_to_trade": 0.03,
        "range_t_sell_ret_1_min": 0.015,
        "range_t_sell_price_pos_20_min": 0.75,
        "range_t_buy_ret_1_max": -0.015,
        "range_t_buy_price_pos_20_max": 0.30,
        "use_tradeability_guard": True,
        "tradeability_limit_tolerance": 0.002,
        "tradeability_min_volume": 0.0,
        "limit_rule_file": "",
        "use_index_constituent_guard": False,
        "index_constituent_file": "",
        "index_constituent_symbol": "000300.SH",
        "enable_acceptance_checks": True,
        "acceptance_target_years": 3,
        "use_strategy_optimizer": True,
        "optimizer_retrain_days": [20, 40],
        "optimizer_weight_thresholds": [0.50, 0.60],
        "optimizer_max_positions": [3, 5],
        "optimizer_market_news_strengths": [0.8, 1.0],
        "optimizer_stock_news_strengths": [1.0, 1.2],
        "optimizer_turnover_penalty": 0.0015,
        "optimizer_drawdown_penalty": 0.20,
        "optimizer_target_years": 3,
        "optimizer_top_trials": 12,
        "optimizer_time_budget_minutes": 0.0,
    },
}


def _read_json_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be an object: {config_path}")
    return payload


def _read_config_section(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = payload.get(section_name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section `{section_name}` must be an object")
    return section


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_years(value: Any) -> tuple[int, ...]:
    if value is None:
        return (3, 5)
    if isinstance(value, (list, tuple)):
        parsed = [int(v) for v in value if int(v) > 0]
        return tuple(sorted(set(parsed))) if parsed else (3, 5)
    text = str(value).strip()
    if not text:
        return (3, 5)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    parsed = [int(p) for p in parts if int(p) > 0]
    return tuple(sorted(set(parsed))) if parsed else (3, 5)


def _parse_int_list(value: Any, *, min_value: int = 1) -> tuple[int, ...]:
    if value is None:
        return ()
    raw: list[Any]
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw = [p.strip() for p in text.split(",") if str(p).strip()]

    out: list[int] = []
    for item in raw:
        try:
            v = int(item)
        except (TypeError, ValueError):
            continue
        if v >= int(min_value):
            out.append(v)
    return tuple(sorted(set(out)))


def _parse_float_list(value: Any, *, min_value: float | None = None) -> tuple[float, ...]:
    if value is None:
        return ()
    raw: list[Any]
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw = [p.strip() for p in text.split(",") if str(p).strip()]

    out: list[float] = []
    for item in raw:
        try:
            v = float(item)
        except (TypeError, ValueError):
            continue
        if min_value is None or v >= float(min_value):
            out.append(v)
    uniq = sorted({round(x, 6) for x in out})
    return tuple(float(x) for x in uniq)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_symbol_list(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    out: list[str] = []
    for token in text.replace("\n", ",").replace(";", ",").split(","):
        code = token.strip()
        if not code:
            continue
        try:
            out.append(normalize_symbol(code).symbol)
        except SymbolError:
            continue
    return sorted(set(out))


def _resolve_settings(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    common_cfg = _read_config_section(payload, "common")
    task_cfg = _read_config_section(payload, args.task)
    defaults = DEFAULT_TASK[args.task]
    resolved: dict[str, Any] = {}

    for key, default in DEFAULT_COMMON.items():
        resolved[key] = _coalesce(
            getattr(args, key, None),
            task_cfg.get(key),
            common_cfg.get(key),
            default,
        )

    for key, default in defaults.items():
        resolved[key] = _coalesce(
            getattr(args, key, None),
            task_cfg.get(key),
            common_cfg.get(key),
            default,
        )

    return resolved


def _masked_settings(settings: dict[str, Any]) -> dict[str, Any]:
    out = dict(settings)
    if "tushare_token" in out and str(out["tushare_token"]).strip():
        out["tushare_token"] = "***"
    return out


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
    sync_margin.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    sync_margin.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    sync_margin.add_argument(
        "--symbols",
        default=None,
        help="Comma separated symbols, e.g. 600160.SH,000630.SZ (overrides watchlist stocks)",
    )
    sync_margin.add_argument("--sleep-ms", dest="sleep_ms", type=int, default=None, help="Sleep between requests in ms")

    return parser


def run_daily(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    market_security, stocks, sector_map = load_watchlist(settings["watchlist"])
    universe_file = str(settings.get("universe_file", "")).strip()
    if not universe_file:
        raise ValueError(
            "Daily requires `--universe-file` (large pool). "
            "The 5-stock watchlist fallback has been disabled."
        )
    universe = build_candidate_universe(
        source=settings["source"],
        data_dir=settings["data_dir"],
        universe_file=universe_file,
        candidate_limit=max(5, int(settings.get("universe_limit", 500))),
        exclude_symbols=[market_security.symbol],
    )
    if not universe.rows:
        raise ValueError(f"Daily universe is empty: {universe_file}")
    stocks = universe.rows
    sector_map = {normalize_symbol(s.symbol).symbol: (s.sector or "其他") for s in stocks}
    print(f"[OK] Daily universe source: {universe.source_label}")
    print(f"[OK] Daily universe size: {len(stocks)}")
    for warning in universe.warnings:
        print(f"[WARN] {warning}")

    config = DailyConfig(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        min_train_days=settings["min_train_days"],
        step_days=settings["step_days"],
        l2=settings["l2"],
        max_positions=int(settings["max_positions"]),
        use_margin_features=_parse_bool(settings["use_margin_features"]),
        margin_market_file=settings["margin_market_file"],
        margin_stock_file=settings["margin_stock_file"],
        news_file=settings["news_file"],
        news_lookback_days=settings["news_lookback_days"],
        learned_news_lookback_days=int(settings["learned_news_lookback_days"]),
        news_half_life_days=settings["news_half_life_days"],
        market_news_strength=settings["market_news_strength"],
        stock_news_strength=settings["stock_news_strength"],
        use_learned_news_fusion=_parse_bool(settings["use_learned_news_fusion"]),
        learned_news_min_samples=int(settings["learned_news_min_samples"]),
        learned_holdout_ratio=float(settings["learned_holdout_ratio"]),
        learned_news_l2=float(settings["learned_news_l2"]),
        learned_fusion_l2=float(settings["learned_fusion_l2"]),
        backtest_years=_parse_years(settings["backtest_years"]),
        backtest_retrain_days=int(settings["backtest_retrain_days"]),
        backtest_weight_threshold=float(settings["backtest_weight_threshold"]),
        backtest_time_budget_minutes=float(settings["backtest_time_budget_minutes"]),
        commission_bps=float(settings["commission_bps"]),
        slippage_bps=float(settings["slippage_bps"]),
        use_turnover_control=_parse_bool(settings["use_turnover_control"]),
        max_trades_per_stock_per_day=max(1, int(settings["max_trades_per_stock_per_day"])),
        max_trades_per_stock_per_week=max(1, int(settings["max_trades_per_stock_per_week"])),
        min_weight_change_to_trade=max(0.0, float(settings["min_weight_change_to_trade"])),
        range_t_sell_ret_1_min=float(settings["range_t_sell_ret_1_min"]),
        range_t_sell_price_pos_20_min=float(settings["range_t_sell_price_pos_20_min"]),
        range_t_buy_ret_1_max=float(settings["range_t_buy_ret_1_max"]),
        range_t_buy_price_pos_20_max=float(settings["range_t_buy_price_pos_20_max"]),
        use_tradeability_guard=_parse_bool(settings["use_tradeability_guard"]),
        tradeability_limit_tolerance=max(0.0, float(settings["tradeability_limit_tolerance"])),
        tradeability_min_volume=max(0.0, float(settings["tradeability_min_volume"])),
        limit_rule_file=str(settings["limit_rule_file"]).strip(),
        use_index_constituent_guard=_parse_bool(settings["use_index_constituent_guard"]),
        index_constituent_file=str(settings["index_constituent_file"]).strip(),
        index_constituent_symbol=str(settings["index_constituent_symbol"]).strip() or "000300.SH",
        enable_acceptance_checks=_parse_bool(settings["enable_acceptance_checks"]),
        acceptance_target_years=max(1, int(settings["acceptance_target_years"])),
        use_strategy_optimizer=_parse_bool(settings["use_strategy_optimizer"]),
        optimizer_retrain_days=_parse_int_list(settings["optimizer_retrain_days"], min_value=1) or (20, 40),
        optimizer_weight_thresholds=_parse_float_list(settings["optimizer_weight_thresholds"], min_value=0.0) or (0.50, 0.60),
        optimizer_max_positions=_parse_int_list(settings["optimizer_max_positions"], min_value=1) or (3, 5),
        optimizer_market_news_strengths=_parse_float_list(settings["optimizer_market_news_strengths"], min_value=0.0) or (0.8, 1.0),
        optimizer_stock_news_strengths=_parse_float_list(settings["optimizer_stock_news_strengths"], min_value=0.0) or (1.0, 1.2),
        optimizer_turnover_penalty=float(settings["optimizer_turnover_penalty"]),
        optimizer_drawdown_penalty=float(settings["optimizer_drawdown_penalty"]),
        optimizer_target_years=max(1, int(settings["optimizer_target_years"])),
        optimizer_top_trials=max(1, int(settings["optimizer_top_trials"])),
        optimizer_time_budget_minutes=float(settings["optimizer_time_budget_minutes"]),
        report_date=settings["report_date"],
    )

    result = generate_daily_fusion(
        config=config,
        market_security=market_security,
        stocks=stocks,
        sector_map=sector_map,
    )
    report_path = write_daily_report(settings["report"], result)
    print(f"[OK] Daily report generated: {report_path.resolve()}")

    if settings["dashboard"].strip():
        dashboard_path = write_daily_dashboard(settings["dashboard"], result)
        print(f"[OK] Daily dashboard generated: {dashboard_path.resolve()}")
    return 0


def run_forecast(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    market_security, stocks, _ = load_watchlist(settings["watchlist"])
    config = ForecastConfig(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        min_train_days=settings["min_train_days"],
        step_days=settings["step_days"],
        l2=settings["l2"],
        max_positions=int(settings["max_positions"]),
        use_margin_features=_parse_bool(settings["use_margin_features"]),
        margin_market_file=settings["margin_market_file"],
        margin_stock_file=settings["margin_stock_file"],
    )
    result = generate_forecast(config=config, market_security=market_security, stocks=stocks)
    path = write_forecast_report(settings["report"], result.market_forecast, result.stock_rows)
    print(f"[OK] Report generated: {path.resolve()}")
    return 0


def run_discover(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    market_security, stocks, _ = load_watchlist(settings["watchlist"])
    config = DiscoverConfig(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        min_train_days=settings["min_train_days"],
        step_days=settings["step_days"],
        l2=settings["l2"],
        max_positions=int(settings["max_positions"]),
        use_margin_features=_parse_bool(settings["use_margin_features"]),
        margin_market_file=settings["margin_market_file"],
        margin_stock_file=settings["margin_stock_file"],
        universe_file=settings["universe_file"],
        candidate_limit=int(settings["candidate_limit"]),
        top_k=int(settings["top_k"]),
        exclude_watchlist=_parse_bool(settings["exclude_watchlist"]),
    )
    result = generate_discovery(config=config, market_security=market_security, watchlist_stocks=stocks)
    path = write_discovery_report(settings["report"], result)
    print(f"[OK] Discovery report generated: {path.resolve()}")
    return 0


def run_sync_data(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    result = sync_market_data(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        universe_size=int(settings["universe_size"]),
        universe_file=settings["universe_file"],
        include_indices=_parse_bool(settings["include_indices"]),
        force_refresh=_parse_bool(settings["force_refresh"]),
        sleep_ms=int(settings["sleep_ms"]),
        max_failures=int(settings["max_failures"]),
        write_universe_file=settings["write_universe_file"],
        universe_min_amount=max(0.0, float(settings["universe_min_amount"])),
        universe_exclude_st=_parse_bool(settings["universe_exclude_st"]),
    )
    print(f"[OK] Universe source: {result.universe_source}")
    print(f"[OK] Universe size: {result.universe_size} (requested {result.requested_universe_size})")
    print(f"[OK] Downloaded: {result.downloaded}, skipped: {result.skipped}, failed: {result.failed}, attempted: {result.attempted}")
    if result.universe_file:
        print(f"[OK] Universe file written: {Path(result.universe_file).resolve()}")
    if result.failed_symbols:
        print(f"[WARN] Failed symbols (first 20): {', '.join(result.failed_symbols)}")
    if result.downloaded <= 0 and result.skipped <= 0:
        print("[ERROR] No symbols were synced.")
        return 2
    return 0


def run_sync_margin(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    symbols = _parse_symbol_list(settings.get("symbols", ""))
    if not symbols:
        _, stocks, _ = load_watchlist(settings["watchlist"])
        symbols = [normalize_symbol(sec.symbol).symbol for sec in stocks]
    if not symbols:
        print("[ERROR] No valid symbols for margin sync. Set --symbols or check watchlist.")
        return 2

    result = sync_margin_data(
        source=settings["source"],
        symbols=symbols,
        start=settings["start"],
        end=settings["end"],
        market_out=settings["margin_market_file"],
        stock_out=settings["margin_stock_file"],
        tushare_token=str(settings.get("tushare_token", "")),
        sleep_ms=int(settings["sleep_ms"]),
    )
    print(f"[OK] Margin source used: {result.source_used}")
    print(f"[OK] Symbols: {len(symbols)}")
    print(f"[OK] Market rows: {result.market_rows} -> {Path(result.market_path).resolve()}")
    print(f"[OK] Stock rows: {result.stock_rows} -> {Path(result.stock_path).resolve()}")
    if result.notes:
        for note in result.notes:
            print(f"[WARN] {note}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        payload = _read_json_config(args.config)
        settings = _resolve_settings(args, payload)
        if args.print_effective_config:
            print(json.dumps({"task": args.task, "settings": _masked_settings(settings)}, indent=2, ensure_ascii=False))
            return 0
        if args.task == "daily":
            return run_daily(settings)
        if args.task == "forecast":
            return run_forecast(settings)
        if args.task == "discover":
            return run_discover(settings)
        if args.task == "sync-data":
            return run_sync_data(settings)
        if args.task == "sync-margin":
            return run_sync_margin(settings)
        parser.error(f"Unknown task: {args.task}")
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
