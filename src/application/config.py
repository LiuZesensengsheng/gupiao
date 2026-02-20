from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastConfig:
    source: str
    data_dir: str
    start: str
    end: str
    min_train_days: int
    step_days: int
    l2: float


@dataclass(frozen=True)
class DailyConfig(ForecastConfig):
    news_file: str
    news_lookback_days: int
    news_half_life_days: float
    market_news_strength: float
    stock_news_strength: float
    report_date: str = ""

