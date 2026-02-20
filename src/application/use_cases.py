from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.application.config import DailyConfig, ForecastConfig
from src.domain.entities import (
    BacktestMetrics,
    BlendedRow,
    EffectSummary,
    ForecastRow,
    MarketForecast,
    NewsItem,
    Security,
    SentimentAggregate,
)
from src.domain.news import aggregate_sentiment, blend_probability
from src.domain.policies import allocate_weights, blend_horizon_score, target_exposure
from src.infrastructure.backtesting import run_portfolio_backtest
from src.infrastructure.effect_analysis import build_latest_snapshot, compute_effect_summary, compute_sector_table
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.news_repository import load_news_items


@dataclass(frozen=True)
class ForecastResult:
    market_forecast: MarketForecast
    stock_rows: List[ForecastRow]


@dataclass(frozen=True)
class DailyFusionResult:
    as_of_date: pd.Timestamp
    source: str
    market_forecast: MarketForecast
    market_final_short: float
    market_final_mid: float
    market_short_sent: SentimentAggregate
    market_mid_sent: SentimentAggregate
    blended_rows: List[BlendedRow]
    effect_summary: EffectSummary
    sector_table: pd.DataFrame
    backtest_metrics: List[BacktestMetrics]
    backtest_curve: pd.DataFrame
    news_items_count: int
    news_items: List[NewsItem]


def generate_forecast(
    config: ForecastConfig,
    market_security: Security,
    stocks: List[Security],
) -> ForecastResult:
    market_forecast, stock_rows = run_quant_pipeline(
        market_security=market_security,
        stock_securities=stocks,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        min_train_days=config.min_train_days,
        step_days=config.step_days,
        l2=config.l2,
    )
    return ForecastResult(market_forecast=market_forecast, stock_rows=stock_rows)


def _blend_stock_rows(
    rows: List[ForecastRow],
    news_items: List[NewsItem],
    as_of_date: pd.Timestamp,
    stock_news_strength: float,
    news_half_life_days: float,
) -> List[BlendedRow]:
    out: List[BlendedRow] = []
    for row in rows:
        short_sent = aggregate_sentiment(
            news_items=news_items,
            as_of_date=as_of_date,
            target=row.symbol,
            horizon="short",
            half_life_days=news_half_life_days,
        )
        mid_sent = aggregate_sentiment(
            news_items=news_items,
            as_of_date=as_of_date,
            target=row.symbol,
            horizon="mid",
            half_life_days=news_half_life_days,
        )

        final_short = blend_probability(row.short_prob, short_sent.score, sentiment_strength=stock_news_strength)
        final_mid = blend_probability(row.mid_prob, mid_sent.score, sentiment_strength=stock_news_strength)
        final_score = blend_horizon_score(final_short, final_mid, short_weight=0.55)
        out.append(
            BlendedRow(
                symbol=row.symbol,
                name=row.name,
                base_short=row.short_prob,
                base_mid=row.mid_prob,
                final_short=final_short,
                final_mid=final_mid,
                final_score=final_score,
                short_sent=short_sent,
                mid_sent=mid_sent,
            )
        )
    return out


def generate_daily_fusion(
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
    sector_map: Dict[str, str],
) -> DailyFusionResult:
    forecast = generate_forecast(
        config=ForecastConfig(
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            l2=config.l2,
        ),
        market_security=market_security,
        stocks=stocks,
    )
    market_forecast = forecast.market_forecast
    stock_rows = forecast.stock_rows

    as_of_date = pd.Timestamp(market_forecast.latest_date).normalize()
    if config.report_date:
        as_of_date = pd.Timestamp(config.report_date).normalize()

    news_items = load_news_items(config.news_file, as_of_date=as_of_date, lookback_days=config.news_lookback_days)

    market_short_sent = aggregate_sentiment(
        news_items=news_items,
        as_of_date=as_of_date,
        target="MARKET",
        horizon="short",
        half_life_days=config.news_half_life_days,
    )
    market_mid_sent = aggregate_sentiment(
        news_items=news_items,
        as_of_date=as_of_date,
        target="MARKET",
        horizon="mid",
        half_life_days=config.news_half_life_days,
    )

    market_final_short = blend_probability(
        market_forecast.short_prob,
        market_short_sent.score,
        sentiment_strength=config.market_news_strength,
    )
    market_final_mid = blend_probability(
        market_forecast.mid_prob,
        market_mid_sent.score,
        sentiment_strength=config.market_news_strength,
    )

    blended_rows = _blend_stock_rows(
        rows=stock_rows,
        news_items=news_items,
        as_of_date=as_of_date,
        stock_news_strength=config.stock_news_strength,
        news_half_life_days=config.news_half_life_days,
    )
    total = target_exposure(market_final_short, market_final_mid)
    weights = allocate_weights([row.final_score for row in blended_rows], total_exposure=total, threshold=0.50)
    for row, weight in zip(blended_rows, weights):
        row.suggested_weight = float(weight)
    blended_rows.sort(key=lambda x: x.final_score, reverse=True)

    snapshot = build_latest_snapshot(
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        stocks=stocks,
        sector_map=sector_map,
    )
    effect_summary = compute_effect_summary(snapshot)
    sector_table = compute_sector_table(snapshot)
    backtest = run_portfolio_backtest(
        market_security=market_security,
        stock_securities=stocks,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        min_train_days=config.min_train_days,
        l2=config.l2,
        retrain_days=config.backtest_retrain_days,
        weight_threshold=config.backtest_weight_threshold,
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps,
        window_years=config.backtest_years,
    )

    return DailyFusionResult(
        as_of_date=as_of_date,
        source=config.source,
        market_forecast=market_forecast,
        market_final_short=market_final_short,
        market_final_mid=market_final_mid,
        market_short_sent=market_short_sent,
        market_mid_sent=market_mid_sent,
        blended_rows=blended_rows,
        effect_summary=effect_summary,
        sector_table=sector_table,
        backtest_metrics=backtest.metrics,
        backtest_curve=backtest.curve_frame,
        news_items_count=len(news_items),
        news_items=news_items,
    )
