from __future__ import annotations

import argparse

from src.application.config import DailyConfig, ForecastConfig
from src.application.use_cases import generate_daily_fusion, generate_forecast
from src.application.watchlist import load_watchlist
from src.infrastructure.market_data import DataError
from src.interfaces.presenters.html_dashboard import write_daily_dashboard
from src.interfaces.presenters.markdown_reports import write_daily_report, write_forecast_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified A-share API CLI")
    sub = parser.add_subparsers(dest="task", required=True)

    daily = sub.add_parser("daily", help="Generate daily fusion report (quant + news)")
    daily.add_argument("--source", default="eastmoney", choices=["eastmoney", "local"], help="Data source")
    daily.add_argument("--watchlist", default="config/watchlist.json", help="Watchlist JSON path")
    daily.add_argument("--data-dir", default="data", help="Directory for local CSV when source=local")
    daily.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    daily.add_argument("--end", default="2099-12-31", help="End date YYYY-MM-DD")
    daily.add_argument("--min-train-days", type=int, default=240, help="Min train days for walk-forward")
    daily.add_argument("--step-days", type=int, default=20, help="Walk-forward test block size")
    daily.add_argument("--l2", type=float, default=0.8, help="L2 regularization strength")
    daily.add_argument("--news-file", default="input/news.csv", help="CSV file for news events")
    daily.add_argument("--news-lookback-days", type=int, default=45, help="News lookback window in days")
    daily.add_argument("--news-half-life-days", type=float, default=10.0, help="Decay half-life for news")
    daily.add_argument("--market-news-strength", type=float, default=0.9, help="Market news blend strength")
    daily.add_argument("--stock-news-strength", type=float, default=1.1, help="Stock news blend strength")
    daily.add_argument("--report-date", default="", help="Override report date YYYY-MM-DD")
    daily.add_argument("--report", default="reports/daily_report.md", help="Output markdown report path")
    daily.add_argument("--dashboard", default="reports/daily_dashboard.html", help="Output HTML dashboard path")

    forecast = sub.add_parser("forecast", help="Generate base quant forecast report")
    forecast.add_argument("--source", default="eastmoney", choices=["eastmoney", "local"], help="Data source")
    forecast.add_argument("--watchlist", default="config/watchlist.json", help="Watchlist JSON path")
    forecast.add_argument("--data-dir", default="data", help="Directory for local CSV when source=local")
    forecast.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    forecast.add_argument("--end", default="2099-12-31", help="End date YYYY-MM-DD")
    forecast.add_argument("--min-train-days", type=int, default=240, help="Min train days for walk-forward")
    forecast.add_argument("--step-days", type=int, default=20, help="Walk-forward test block size")
    forecast.add_argument("--l2", type=float, default=0.8, help="L2 regularization strength")
    forecast.add_argument("--report", default="reports/latest_report.md", help="Output report path")

    return parser


def run_daily(args: argparse.Namespace) -> int:
    market_security, stocks, sector_map = load_watchlist(args.watchlist)
    config = DailyConfig(
        source=args.source,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        min_train_days=args.min_train_days,
        step_days=args.step_days,
        l2=args.l2,
        news_file=args.news_file,
        news_lookback_days=args.news_lookback_days,
        news_half_life_days=args.news_half_life_days,
        market_news_strength=args.market_news_strength,
        stock_news_strength=args.stock_news_strength,
        report_date=args.report_date,
    )

    result = generate_daily_fusion(
        config=config,
        market_security=market_security,
        stocks=stocks,
        sector_map=sector_map,
    )
    report_path = write_daily_report(args.report, result)
    print(f"[OK] Daily report generated: {report_path.resolve()}")

    if args.dashboard.strip():
        dashboard_path = write_daily_dashboard(args.dashboard, result)
        print(f"[OK] Daily dashboard generated: {dashboard_path.resolve()}")
    return 0


def run_forecast(args: argparse.Namespace) -> int:
    market_security, stocks, _ = load_watchlist(args.watchlist)
    config = ForecastConfig(
        source=args.source,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        min_train_days=args.min_train_days,
        step_days=args.step_days,
        l2=args.l2,
    )
    result = generate_forecast(config=config, market_security=market_security, stocks=stocks)
    path = write_forecast_report(args.report, result.market_forecast, result.stock_rows)
    print(f"[OK] Report generated: {path.resolve()}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.task == "daily":
            return run_daily(args)
        if args.task == "forecast":
            return run_forecast(args)
        parser.error(f"Unknown task: {args.task}")
    except DataError as exc:
        print(f"[ERROR] {exc}")
        print("Hint: if online source fails, use `--source local --data-dir data`.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

