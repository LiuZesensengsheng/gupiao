from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.application.config import DailyConfig, ForecastConfig
from src.application.use_cases import generate_daily_fusion, generate_forecast
from src.application.watchlist import load_watchlist
from src.infrastructure.market_data import DataError, set_tushare_token
from src.interfaces.presenters.html_dashboard import write_daily_dashboard
from src.interfaces.presenters.markdown_reports import write_daily_report, write_forecast_report

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
}

DEFAULT_TASK: dict[str, dict[str, Any]] = {
    "forecast": {
        "report": "reports/latest_report.md",
    },
    "daily": {
        "news_file": "input/news.csv",
        "news_lookback_days": 45,
        "news_half_life_days": 10.0,
        "market_news_strength": 0.9,
        "stock_news_strength": 1.1,
        "report_date": "",
        "report": "reports/daily_report.md",
        "dashboard": "reports/daily_dashboard.html",
        "backtest_years": [3, 5],
        "backtest_retrain_days": 20,
        "backtest_weight_threshold": 0.50,
        "commission_bps": 1.5,
        "slippage_bps": 2.0,
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


def _resolve_settings(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    common_cfg = _read_config_section(payload, "common")
    task_cfg = _read_config_section(payload, args.task)
    defaults = DEFAULT_TASK[args.task]
    resolved: dict[str, Any] = {}

    for key, default in DEFAULT_COMMON.items():
        resolved[key] = _coalesce(
            getattr(args, key),
            task_cfg.get(key),
            common_cfg.get(key),
            default,
        )

    for key, default in defaults.items():
        resolved[key] = _coalesce(
            getattr(args, key),
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
    daily.add_argument("--news-file", dest="news_file", default=None, help="CSV file for news events")
    daily.add_argument("--news-lookback-days", dest="news_lookback_days", type=int, default=None, help="Lookback")
    daily.add_argument("--news-half-life-days", dest="news_half_life_days", type=float, default=None, help="Half-life")
    daily.add_argument("--market-news-strength", dest="market_news_strength", type=float, default=None, help="Blend")
    daily.add_argument("--stock-news-strength", dest="stock_news_strength", type=float, default=None, help="Blend")
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
    daily.add_argument("--commission-bps", dest="commission_bps", type=float, default=None, help="Commission in bps")
    daily.add_argument("--slippage-bps", dest="slippage_bps", type=float, default=None, help="Slippage in bps")

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

    return parser


def run_daily(settings: dict[str, Any]) -> int:
    set_tushare_token(settings.get("tushare_token", ""))
    market_security, stocks, sector_map = load_watchlist(settings["watchlist"])
    config = DailyConfig(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        min_train_days=settings["min_train_days"],
        step_days=settings["step_days"],
        l2=settings["l2"],
        news_file=settings["news_file"],
        news_lookback_days=settings["news_lookback_days"],
        news_half_life_days=settings["news_half_life_days"],
        market_news_strength=settings["market_news_strength"],
        stock_news_strength=settings["stock_news_strength"],
        backtest_years=_parse_years(settings["backtest_years"]),
        backtest_retrain_days=int(settings["backtest_retrain_days"]),
        backtest_weight_threshold=float(settings["backtest_weight_threshold"]),
        commission_bps=float(settings["commission_bps"]),
        slippage_bps=float(settings["slippage_bps"]),
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
    )
    result = generate_forecast(config=config, market_security=market_security, stocks=stocks)
    path = write_forecast_report(settings["report"], result.market_forecast, result.stock_rows)
    print(f"[OK] Report generated: {path.resolve()}")
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
