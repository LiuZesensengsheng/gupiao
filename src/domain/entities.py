from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Security:
    symbol: str
    name: str
    sector: str = "其他"


@dataclass
class BinaryMetrics:
    n: int
    accuracy: float
    brier: float
    auc: float
    base_rate: float

    @staticmethod
    def empty() -> "BinaryMetrics":
        return BinaryMetrics(n=0, accuracy=np.nan, brier=np.nan, auc=np.nan, base_rate=np.nan)


@dataclass
class ForecastRow:
    symbol: str
    name: str
    latest_date: pd.Timestamp
    short_prob: float
    mid_prob: float
    score: float
    short_drivers: list[str]
    mid_drivers: list[str]
    short_eval: BinaryMetrics
    mid_eval: BinaryMetrics
    suggested_weight: float = 0.0


@dataclass
class MarketForecast:
    symbol: str
    name: str
    latest_date: pd.Timestamp
    short_prob: float
    mid_prob: float
    short_eval: BinaryMetrics
    mid_eval: BinaryMetrics


@dataclass(frozen=True)
class NewsItem:
    date: pd.Timestamp
    target_type: str
    target: str
    horizon: str
    direction: str
    strength: float
    confidence: float
    source_weight: float
    title: str = ""
    source_url: str = ""


@dataclass
class SentimentAggregate:
    bullish: float
    bearish: float
    neutral: float
    score: float
    items: int


@dataclass
class BlendedRow:
    symbol: str
    name: str
    base_short: float
    base_mid: float
    news_short_prob: float
    news_mid_prob: float
    final_short: float
    final_mid: float
    final_score: float
    short_sent: SentimentAggregate
    mid_sent: SentimentAggregate
    fusion_mode_short: str = "rule"
    fusion_mode_mid: str = "rule"
    volume_risk_flag: bool = False
    volume_risk_note: str = ""
    short_drivers: list[str] = field(default_factory=list)
    mid_drivers: list[str] = field(default_factory=list)
    suggested_weight: float = 0.0


@dataclass
class FusionDiagnostics:
    target: str
    horizon: str
    mode: str
    reason: str
    samples: int
    holdout_n: int
    holdout_accuracy: float
    holdout_brier: float
    holdout_auc: float
    news_coef_score: float
    fusion_coef_quant: float
    fusion_coef_news: float


@dataclass
class DiscoveryRow:
    symbol: str
    name: str
    short_prob: float
    mid_prob: float
    score: float
    suggested_weight: float
    volume_risk_flag: bool
    volume_risk_note: str
    short_drivers: list[str]
    mid_drivers: list[str]


@dataclass
class EffectSummary:
    sample_size: int
    win_rate_1d: float
    win_rate_5d: float
    strong_rate_5d: float
    median_ret_1d: float
    median_ret_5d: float
    loss_rate_1d: float
    deep_loss_rate: float
    median_drawdown_20: float
    avg_price_pos_20: float
    avg_vol_ratio_20: float
    avg_obv_z_20: float
    avg_vol_conc_5_20: float
    money_score: float
    chip_score: float
    money_label: str
    chip_label: str
    pnl_label: str
    risk_label: str

    @staticmethod
    def empty() -> "EffectSummary":
        return EffectSummary(
            sample_size=0,
            win_rate_1d=np.nan,
            win_rate_5d=np.nan,
            strong_rate_5d=np.nan,
            median_ret_1d=np.nan,
            median_ret_5d=np.nan,
            loss_rate_1d=np.nan,
            deep_loss_rate=np.nan,
            median_drawdown_20=np.nan,
            avg_price_pos_20=np.nan,
            avg_vol_ratio_20=np.nan,
            avg_obv_z_20=np.nan,
            avg_vol_conc_5_20=np.nan,
            money_score=np.nan,
            chip_score=np.nan,
            money_label="NA",
            chip_label="NA",
            pnl_label="NA",
            risk_label="NA",
        )


@dataclass
class BacktestMetrics:
    label: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_days: int
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    benchmark_total_return: float
    benchmark_annual_return: float
    excess_total_return: float
    excess_annual_return: float
    information_ratio: float
    tracking_error: float
    win_rate: float
    avg_turnover: float
    annual_turnover: float
    total_cost: float
    avg_trade_count_per_day: float
    avg_trades_per_stock_per_week: float

    @staticmethod
    def empty(label: str) -> "BacktestMetrics":
        nat = pd.NaT
        return BacktestMetrics(
            label=label,
            start_date=nat,
            end_date=nat,
            n_days=0,
            total_return=np.nan,
            annual_return=np.nan,
            annual_vol=np.nan,
            sharpe=np.nan,
            sortino=np.nan,
            max_drawdown=np.nan,
            calmar=np.nan,
            benchmark_total_return=np.nan,
            benchmark_annual_return=np.nan,
            excess_total_return=np.nan,
            excess_annual_return=np.nan,
            information_ratio=np.nan,
            tracking_error=np.nan,
            win_rate=np.nan,
            avg_turnover=np.nan,
            annual_turnover=np.nan,
            total_cost=np.nan,
            avg_trade_count_per_day=np.nan,
            avg_trades_per_stock_per_week=np.nan,
        )


@dataclass
class StrategyTrial:
    rank: int
    metric_label: str
    retrain_days: int
    weight_threshold: float
    max_positions: int
    market_news_strength: float
    stock_news_strength: float
    objective_score: float
    annual_return: float
    excess_annual_return: float
    max_drawdown: float
    annual_turnover: float
    total_cost: float
    sharpe: float
    avg_trades_per_stock_per_week: float
