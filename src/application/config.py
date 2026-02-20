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
    max_positions: int
    use_margin_features: bool
    margin_market_file: str
    margin_stock_file: str


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
    backtest_time_budget_minutes: float
    commission_bps: float
    slippage_bps: float
    use_turnover_control: bool
    max_trades_per_stock_per_week: int
    min_weight_change_to_trade: float
    use_strategy_optimizer: bool
    optimizer_retrain_days: tuple[int, ...]
    optimizer_weight_thresholds: tuple[float, ...]
    optimizer_max_positions: tuple[int, ...]
    optimizer_market_news_strengths: tuple[float, ...]
    optimizer_stock_news_strengths: tuple[float, ...]
    optimizer_turnover_penalty: float
    optimizer_drawdown_penalty: float
    optimizer_target_years: int
    optimizer_top_trials: int
    optimizer_time_budget_minutes: float
    report_date: str = ""


@dataclass(frozen=True)
class DiscoverConfig(ForecastConfig):
    universe_file: str
    candidate_limit: int
    top_k: int
    exclude_watchlist: bool
