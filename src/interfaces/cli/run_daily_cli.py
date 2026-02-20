from __future__ import annotations

import argparse

from src.application.config import DailyConfig
from src.application.use_cases import generate_daily_fusion
from src.application.watchlist import load_watchlist
from src.infrastructure.market_data import DataError
from src.interfaces.presenters.html_dashboard import write_daily_dashboard
from src.interfaces.presenters.markdown_reports import write_daily_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily A-share report with quant baseline + fuzzy news matrix")
    parser.add_argument("--source", default="eastmoney", choices=["eastmoney", "local"], help="Data source")
    parser.add_argument("--watchlist", default="config/watchlist.json", help="Watchlist JSON path")
    parser.add_argument("--data-dir", default="data", help="Directory for local CSV when source=local")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2099-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--min-train-days", type=int, default=240, help="Min train days for walk-forward")
    parser.add_argument("--step-days", type=int, default=20, help="Walk-forward test block size")
    parser.add_argument("--l2", type=float, default=0.8, help="L2 regularization strength")
    parser.add_argument("--news-file", default="input/news.csv", help="CSV file for news events")
    parser.add_argument("--news-lookback-days", type=int, default=45, help="News lookback window in days")
    parser.add_argument("--news-half-life-days", type=float, default=10.0, help="Decay half-life for news")
    parser.add_argument("--market-news-strength", type=float, default=0.9, help="Market news blend strength")
    parser.add_argument("--stock-news-strength", type=float, default=1.1, help="Stock news blend strength")
    parser.add_argument("--report-date", default="", help="Override report date YYYY-MM-DD")
    parser.add_argument("--report", default="reports/daily_report.md", help="Output markdown report path")
    parser.add_argument("--dashboard", default="reports/daily_dashboard.html", help="Output HTML dashboard path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    try:
        result = generate_daily_fusion(
            config=config,
            market_security=market_security,
            stocks=stocks,
            sector_map=sector_map,
        )
    except DataError as exc:
        print(f"[ERROR] {exc}")
        print("Hint: if online source fails, use `--source local --data-dir data`.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        return 3

    report_path = write_daily_report(args.report, result)
    print(f"[OK] Daily report generated: {report_path.resolve()}")

    if args.dashboard.strip():
        dashboard_path = write_daily_dashboard(args.dashboard, result)
        print(f"[OK] Daily dashboard generated: {dashboard_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

