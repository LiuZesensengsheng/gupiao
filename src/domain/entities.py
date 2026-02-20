from __future__ import annotations

from dataclasses import dataclass

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
    final_short: float
    final_mid: float
    final_score: float
    short_sent: SentimentAggregate
    mid_sent: SentimentAggregate
    suggested_weight: float = 0.0


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

