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
    learned_news_lookback_days: int
    news_half_life_days: float
    market_news_strength: float
    stock_news_strength: float
    use_learned_news_fusion: bool
    learned_news_min_samples: int
    learned_holdout_ratio: float
    learned_news_l2: float
    learned_fusion_l2: float
    backtest_years: tuple[int, ...]
    backtest_retrain_days: int
    backtest_weight_threshold: float
    commission_bps: float
    slippage_bps: float
    report_date: str = ""
