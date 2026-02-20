"""Compatibility wrapper for forecast pipeline APIs.

Use `src.infrastructure.forecast_engine` and `src.interfaces.presenters.markdown_reports`
for new code.
"""

from pathlib import Path
from typing import Sequence

from src.domain.entities import ForecastRow, MarketForecast, Security
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.interfaces.presenters.markdown_reports import write_forecast_report


def run_pipeline(
    market_security: Security,
    stock_securities: Sequence[Security],
    source: str,
    data_dir: str,
    start: str,
    end: str,
    min_train_days: int,
    step_days: int,
    l2: float,
) -> tuple[MarketForecast, list[ForecastRow]]:
    return run_quant_pipeline(
        market_security=market_security,
        stock_securities=stock_securities,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        min_train_days=min_train_days,
        step_days=step_days,
        l2=l2,
    )


def write_report(out_path: str | Path, market_forecast: MarketForecast, stock_rows: Sequence[ForecastRow]) -> Path:
    return write_forecast_report(out_path, market_forecast, stock_rows)


__all__ = ["Security", "ForecastRow", "MarketForecast", "run_pipeline", "write_report"]

