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
    FusionDiagnostics,
    MarketForecast,
    NewsItem,
    Security,
    SentimentAggregate,
)
from src.domain.symbols import normalize_symbol
from src.domain.policies import allocate_weights, blend_horizon_score, target_exposure
from src.infrastructure.backtesting import run_portfolio_backtest
from src.infrastructure.effect_analysis import build_latest_snapshot, compute_effect_summary, compute_sector_table
from src.infrastructure.features import MARKET_FEATURE_COLUMNS, make_market_feature_frame, make_stock_feature_frame, stock_feature_columns
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.market_data import load_symbol_daily
from src.infrastructure.news_fusion import predict_with_learned_fusion
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
    market_news_short_prob: float
    market_news_mid_prob: float
    market_final_short: float
    market_final_mid: float
    market_short_sent: SentimentAggregate
    market_mid_sent: SentimentAggregate
    market_fusion_mode_short: str
    market_fusion_mode_mid: str
    blended_rows: List[BlendedRow]
    learning_diagnostics: List[FusionDiagnostics]
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


def _prepare_learning_frames(
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
    )
    market_feat = make_market_feature_frame(market_raw)
    stock_frames: Dict[str, pd.DataFrame] = {}
    for security in stocks:
        symbol = normalize_symbol(security.symbol).symbol
        stock_raw = load_symbol_daily(
            symbol=symbol,
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
        )
        stock_frames[symbol] = make_stock_feature_frame(stock_raw, market_feat)
    return market_feat, stock_frames


def _blend_stock_rows(
    rows: List[ForecastRow],
    news_items_train: List[NewsItem],
    news_items_live: List[NewsItem],
    as_of_date: pd.Timestamp,
    config: DailyConfig,
    stock_news_strength: float,
    stock_feature_frames: Dict[str, pd.DataFrame],
) -> tuple[List[BlendedRow], List[FusionDiagnostics]]:
    out: List[BlendedRow] = []
    diagnostics: List[FusionDiagnostics] = []
    feature_cols = stock_feature_columns()
    for row in rows:
        feature_frame = stock_feature_frames.get(row.symbol)
        short_pred = predict_with_learned_fusion(
            enabled=config.use_learned_news_fusion,
            base_prob=row.short_prob,
            target=row.symbol,
            horizon="short",
            feature_frame=feature_frame,
            feature_cols=feature_cols,
            target_col="target_1d_up",
            news_items_train=news_items_train,
            news_items_live=news_items_live,
            as_of_date=as_of_date,
            half_life_days=config.news_half_life_days,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            quant_l2=config.l2,
            news_l2=config.learned_news_l2,
            fusion_l2=config.learned_fusion_l2,
            min_samples=config.learned_news_min_samples,
            holdout_ratio=config.learned_holdout_ratio,
            fallback_strength=stock_news_strength,
        )
        mid_pred = predict_with_learned_fusion(
            enabled=config.use_learned_news_fusion,
            base_prob=row.mid_prob,
            target=row.symbol,
            horizon="mid",
            feature_frame=feature_frame,
            feature_cols=feature_cols,
            target_col="target_20d_up",
            news_items_train=news_items_train,
            news_items_live=news_items_live,
            as_of_date=as_of_date,
            half_life_days=config.news_half_life_days,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            quant_l2=config.l2,
            news_l2=config.learned_news_l2,
            fusion_l2=config.learned_fusion_l2,
            min_samples=config.learned_news_min_samples,
            holdout_ratio=config.learned_holdout_ratio,
            fallback_strength=stock_news_strength,
        )
        diagnostics.extend([short_pred.diagnostics, mid_pred.diagnostics])
        final_short = short_pred.final_prob
        final_mid = mid_pred.final_prob
        final_score = blend_horizon_score(final_short, final_mid, short_weight=0.55)
        out.append(
            BlendedRow(
                symbol=row.symbol,
                name=row.name,
                base_short=row.short_prob,
                base_mid=row.mid_prob,
                news_short_prob=short_pred.news_prob,
                news_mid_prob=mid_pred.news_prob,
                final_short=final_short,
                final_mid=final_mid,
                final_score=final_score,
                short_sent=short_pred.sentiment,
                mid_sent=mid_pred.sentiment,
                fusion_mode_short=short_pred.mode,
                fusion_mode_mid=mid_pred.mode,
            )
        )
    return out, diagnostics


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

    news_items_live = load_news_items(
        config.news_file,
        as_of_date=as_of_date,
        lookback_days=config.news_lookback_days,
    )
    news_items_train = load_news_items(
        config.news_file,
        as_of_date=as_of_date,
        lookback_days=max(config.news_lookback_days, config.learned_news_lookback_days),
    )

    market_feat = pd.DataFrame()
    stock_feature_frames: Dict[str, pd.DataFrame] = {}
    if config.use_learned_news_fusion:
        market_feat, stock_feature_frames = _prepare_learning_frames(
            config=config,
            market_security=market_security,
            stocks=stocks,
        )
    market_short_pred = predict_with_learned_fusion(
        enabled=config.use_learned_news_fusion,
        base_prob=market_forecast.short_prob,
        target="MARKET",
        horizon="short",
        feature_frame=market_feat,
        feature_cols=MARKET_FEATURE_COLUMNS,
        target_col="mkt_target_1d_up",
        news_items_train=news_items_train,
        news_items_live=news_items_live,
        as_of_date=as_of_date,
        half_life_days=config.news_half_life_days,
        min_train_days=config.min_train_days,
        step_days=config.step_days,
        quant_l2=config.l2,
        news_l2=config.learned_news_l2,
        fusion_l2=config.learned_fusion_l2,
        min_samples=config.learned_news_min_samples,
        holdout_ratio=config.learned_holdout_ratio,
        fallback_strength=config.market_news_strength,
    )
    market_mid_pred = predict_with_learned_fusion(
        enabled=config.use_learned_news_fusion,
        base_prob=market_forecast.mid_prob,
        target="MARKET",
        horizon="mid",
        feature_frame=market_feat,
        feature_cols=MARKET_FEATURE_COLUMNS,
        target_col="mkt_target_20d_up",
        news_items_train=news_items_train,
        news_items_live=news_items_live,
        as_of_date=as_of_date,
        half_life_days=config.news_half_life_days,
        min_train_days=config.min_train_days,
        step_days=config.step_days,
        quant_l2=config.l2,
        news_l2=config.learned_news_l2,
        fusion_l2=config.learned_fusion_l2,
        min_samples=config.learned_news_min_samples,
        holdout_ratio=config.learned_holdout_ratio,
        fallback_strength=config.market_news_strength,
    )

    market_final_short = market_short_pred.final_prob
    market_final_mid = market_mid_pred.final_prob

    blended_rows, stock_diagnostics = _blend_stock_rows(
        rows=stock_rows,
        news_items_train=news_items_train,
        news_items_live=news_items_live,
        as_of_date=as_of_date,
        config=config,
        stock_news_strength=config.stock_news_strength,
        stock_feature_frames=stock_feature_frames,
    )
    total = target_exposure(market_final_short, market_final_mid)
    weights = allocate_weights([row.final_score for row in blended_rows], total_exposure=total, threshold=0.50)
    for row, weight in zip(blended_rows, weights):
        row.suggested_weight = float(weight)
    blended_rows.sort(key=lambda x: x.final_score, reverse=True)
    learning_diagnostics = [market_short_pred.diagnostics, market_mid_pred.diagnostics] + stock_diagnostics

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
        market_news_short_prob=market_short_pred.news_prob,
        market_news_mid_prob=market_mid_pred.news_prob,
        market_final_short=market_final_short,
        market_final_mid=market_final_mid,
        market_short_sent=market_short_pred.sentiment,
        market_mid_sent=market_mid_pred.sentiment,
        market_fusion_mode_short=market_short_pred.mode,
        market_fusion_mode_mid=market_mid_pred.mode,
        blended_rows=blended_rows,
        learning_diagnostics=learning_diagnostics,
        effect_summary=effect_summary,
        sector_table=sector_table,
        backtest_metrics=backtest.metrics,
        backtest_curve=backtest.curve_frame,
        news_items_count=len(news_items_live),
        news_items=news_items_live,
    )
