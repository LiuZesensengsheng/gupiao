from __future__ import annotations

import argparse

from src.application.config import ForecastConfig
from src.application.use_cases import generate_forecast
from src.application.watchlist import load_watchlist
from src.infrastructure.market_data import DataError
from src.interfaces.presenters.markdown_reports import write_forecast_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-share market + stock multi-horizon forecast")
    parser.add_argument("--source", default="eastmoney", choices=["eastmoney", "local"], help="Data source")
    parser.add_argument("--watchlist", default="config/watchlist.json", help="Watchlist JSON path")
    parser.add_argument("--data-dir", default="data", help="Directory for local CSV when source=local")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2099-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--min-train-days", type=int, default=240, help="Min train days for walk-forward")
    parser.add_argument("--step-days", type=int, default=20, help="Walk-forward test block size")
    parser.add_argument("--l2", type=float, default=0.8, help="L2 regularization strength")
    parser.add_argument("--report", default="reports/latest_report.md", help="Output report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    try:
        result = generate_forecast(config=config, market_security=market_security, stocks=stocks)
    except DataError as exc:
        print(f"[ERROR] {exc}")
        print(
            "Hint: if online source fails, prepare local CSV files then run with "
            "`--source local --data-dir data`."
        )
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        return 3

    path = write_forecast_report(args.report, result.market_forecast, result.stock_rows)
    print(f"[OK] Report generated: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

