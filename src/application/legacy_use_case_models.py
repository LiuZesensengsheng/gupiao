from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from src.domain.entities import (
    BacktestMetrics,
    BlendedRow,
    DiscoveryRow,
    EffectSummary,
    ForecastRow,
    FusionDiagnostics,
    MarketForecast,
    NewsItem,
    StrategyTrial,
    TradeAction,
)


@dataclass(frozen=True)
class ForecastResult:
    market_forecast: MarketForecast
    stock_rows: List[ForecastRow]


@dataclass(frozen=True)
class DailyFusionResult:
    as_of_date: pd.Timestamp
    source: str
    market_forecast: MarketForecast
    market_news_short_prob: float
    market_news_mid_prob: float
    market_final_short: float
    market_final_mid: float
    market_state_code: str
    market_state_label: str
    strategy_template: str
    intraday_t_level: str
    effective_total_exposure: float
    effective_weight_threshold: float
    effective_max_positions: int
    effective_max_trades_per_stock_per_day: int
    effective_max_trades_per_stock_per_week: int
    market_short_sent: object
    market_mid_sent: object
    market_fusion_mode_short: str
    market_fusion_mode_mid: str
    blended_rows: List[BlendedRow]
    learning_diagnostics: List[FusionDiagnostics]
    effect_summary: EffectSummary
    sector_table: pd.DataFrame
    backtest_metrics: List[BacktestMetrics]
    backtest_curve: pd.DataFrame
    strategy_objective_text: str
    strategy_target_metric_label: str
    strategy_selected: StrategyTrial | None
    strategy_trials: List[StrategyTrial]
    acceptance_enabled: bool
    acceptance_ab_pass: bool
    acceptance_constraints_pass: bool
    acceptance_summary: str
    acceptance_delta_excess_annual_return: float
    acceptance_delta_max_drawdown: float
    acceptance_delta_annual_turnover: float
    acceptance_limit_violations: int
    acceptance_oversell_violations: int
    trade_plan_basis: str
    trade_plan_nav: float
    trade_plan_lot_size: int
    trade_actions: List[TradeAction]
    news_items_count: int
    news_items: List[NewsItem]


@dataclass(frozen=True)
class DiscoveryResult:
    as_of_date: pd.Timestamp
    source: str
    market_forecast: MarketForecast
    universe_size: int
    universe_source: str
    warnings: List[str]
    rows: List[DiscoveryRow]
