from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List

import numpy as np
import pandas as pd

from src.application.config import DailyConfig, DiscoverConfig, ForecastConfig
from src.domain.entities import (
    BacktestMetrics,
    BlendedRow,
    DiscoveryRow,
    EffectSummary,
    ForecastRow,
    FusionDiagnostics,
    MarketForecast,
    NewsItem,
    Security,
    SentimentAggregate,
    StrategyTrial,
)
from src.domain.symbols import normalize_symbol
from src.domain.policies import allocate_weights, blend_horizon_score, target_exposure
from src.infrastructure.backtesting import BacktestResult, run_portfolio_backtest
from src.infrastructure.discovery import build_candidate_universe, compute_volume_risk
from src.infrastructure.effect_analysis import build_latest_snapshot, compute_effect_summary, compute_sector_table
from src.infrastructure.features import MARKET_FEATURE_COLUMNS, make_market_feature_frame, make_stock_feature_frame, stock_feature_columns
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_context import build_market_context_features
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
    strategy_objective_text: str
    strategy_target_metric_label: str
    strategy_selected: StrategyTrial | None
    strategy_trials: List[StrategyTrial]
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


@dataclass(frozen=True)
class _StrategySelection:
    retrain_days: int
    weight_threshold: float
    max_positions: int
    market_news_strength: float
    stock_news_strength: float
    objective_text: str
    target_metric_label: str
    selected_trial: StrategyTrial | None
    trials: List[StrategyTrial]
    best_backtest: BacktestResult | None


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
        max_positions=config.max_positions,
        use_margin_features=config.use_margin_features,
        margin_market_file=config.margin_market_file,
        margin_stock_file=config.margin_stock_file,
    )
    return ForecastResult(market_forecast=market_forecast, stock_rows=stock_rows)


def generate_discovery(
    config: DiscoverConfig,
    market_security: Security,
    watchlist_stocks: List[Security],
) -> DiscoveryResult:
    exclude_symbols = [market_security.symbol]
    if config.exclude_watchlist:
        exclude_symbols.extend(s.symbol for s in watchlist_stocks)

    universe = build_candidate_universe(
        source=config.source,
        data_dir=config.data_dir,
        universe_file=config.universe_file,
        candidate_limit=config.candidate_limit,
        exclude_symbols=exclude_symbols,
    )
    if not universe.rows:
        raise ValueError("Discovery universe is empty. Provide `--universe-file` or prepare more symbols in data directory.")

    forecast = generate_forecast(
        config=ForecastConfig(
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            l2=config.l2,
            max_positions=config.max_positions,
            use_margin_features=config.use_margin_features,
            margin_market_file=config.margin_market_file,
            margin_stock_file=config.margin_stock_file,
        ),
        market_security=market_security,
        stocks=universe.rows,
    )
    as_of_date = pd.Timestamp(forecast.market_forecast.latest_date).normalize()
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
    )
    market_feat = make_market_feature_frame(market_raw)

    top_rows = sorted(forecast.stock_rows, key=lambda x: x.score, reverse=True)[: max(1, int(config.top_k))]
    out: List[DiscoveryRow] = []
    for row in top_rows:
        stock_raw = load_symbol_daily(
            symbol=row.symbol,
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
        )
        stock_feat = make_stock_feature_frame(stock_raw, market_feat)
        risk_flag, risk_note = compute_volume_risk(stock_feat, as_of_date=as_of_date)
        out.append(
            DiscoveryRow(
                symbol=row.symbol,
                name=row.name,
                short_prob=row.short_prob,
                mid_prob=row.mid_prob,
                score=row.score,
                suggested_weight=row.suggested_weight,
                volume_risk_flag=risk_flag,
                volume_risk_note=risk_note,
                short_drivers=row.short_drivers,
                mid_drivers=row.mid_drivers,
            )
        )

    return DiscoveryResult(
        as_of_date=as_of_date,
        source=config.source,
        market_forecast=forecast.market_forecast,
        universe_size=len(universe.rows),
        universe_source=universe.source_label,
        warnings=universe.warnings,
        rows=out,
    )


def _prepare_learning_frames(
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
) -> tuple[pd.DataFrame, List[str], Dict[str, List[str]], Dict[str, pd.DataFrame]]:
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        market_dates=market_feat_base["date"],
        use_margin_features=config.use_margin_features,
        margin_market_file=config.margin_market_file,
    )
    market_feat = market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")
    market_feature_cols = MARKET_FEATURE_COLUMNS + market_context.feature_columns
    stock_feature_cols_map: Dict[str, List[str]] = {}
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
        stock_feat = make_stock_feature_frame(stock_raw, market_feat)
        stock_margin_cols: list[str] = []
        if config.use_margin_features:
            margin_frame, margin_cols, _ = build_stock_margin_features(
                margin_stock_file=config.margin_stock_file,
                symbol=symbol,
                start=config.start,
                end=config.end,
            )
            if margin_cols:
                stock_feat = stock_feat.merge(margin_frame, on="date", how="left", validate="1:1")
                stock_margin_cols = list(margin_cols)
        stock_frames[symbol] = stock_feat
        stock_feature_cols_map[symbol] = stock_feature_columns(
            extra_market_cols=market_context.feature_columns,
            extra_stock_cols=stock_margin_cols,
        )
    return market_feat, market_feature_cols, stock_feature_cols_map, stock_frames


def _blend_stock_rows(
    rows: List[ForecastRow],
    news_items_train: List[NewsItem],
    news_items_live: List[NewsItem],
    as_of_date: pd.Timestamp,
    config: DailyConfig,
    stock_news_strength: float,
    stock_feature_cols_map: Dict[str, List[str]],
    stock_feature_frames: Dict[str, pd.DataFrame],
) -> tuple[List[BlendedRow], List[FusionDiagnostics]]:
    out: List[BlendedRow] = []
    diagnostics: List[FusionDiagnostics] = []
    for row in rows:
        feature_frame = stock_feature_frames.get(row.symbol)
        volume_risk_flag = False
        volume_risk_note = ""
        if feature_frame is not None and not feature_frame.empty:
            history = feature_frame[feature_frame["date"] <= as_of_date].sort_values("date")
            if not history.empty:
                latest = history.iloc[-1]
                hvbd_recent = float(latest.get("hvbd_recent_5", 0.0))
                vol_ratio = float(latest.get("vol_ratio_20", np.nan))
                price_pos = float(latest.get("price_pos_20", np.nan))
                if hvbd_recent >= 0.5:
                    volume_risk_flag = True
                    volume_risk_note = f"高位巨量大阴线(5日内), 量能比={vol_ratio:.2f}, 位置={price_pos:.2f}"
                else:
                    volume_risk_note = f"量能比={vol_ratio:.2f}, 位置={price_pos:.2f}"
        feature_cols = stock_feature_cols_map.get(row.symbol, stock_feature_columns())
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
                volume_risk_flag=volume_risk_flag,
                volume_risk_note=volume_risk_note,
                short_drivers=list(row.short_drivers),
                mid_drivers=list(row.mid_drivers),
            )
        )
    return out, diagnostics


def _sanitize_int_grid(values: tuple[int, ...], *, fallback: int, min_value: int = 1, max_value: int | None = None) -> list[int]:
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value < int(min_value):
            continue
        if max_value is not None and value > int(max_value):
            continue
        out.append(value)
    uniq = sorted(set(out))
    return uniq if uniq else [int(fallback)]


def _sanitize_float_grid(values: tuple[float, ...], *, fallback: float, min_value: float = 0.0) -> list[float]:
    out: list[float] = []
    for raw in values:
        value = float(raw)
        if value < float(min_value):
            continue
        out.append(value)
    uniq = sorted({round(v, 6) for v in out})
    return [float(v) for v in uniq] if uniq else [float(fallback)]


def _run_daily_backtest(
    *,
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
    news_items_train: List[NewsItem],
    retrain_days: int,
    weight_threshold: float,
    max_positions: int,
    market_news_strength: float,
    stock_news_strength: float,
) -> BacktestResult:
    return run_portfolio_backtest(
        market_security=market_security,
        stock_securities=stocks,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        min_train_days=config.min_train_days,
        l2=config.l2,
        retrain_days=int(retrain_days),
        weight_threshold=float(weight_threshold),
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps,
        window_years=config.backtest_years,
        news_items=news_items_train,
        apply_news_fusion=True,
        news_half_life_days=config.news_half_life_days,
        market_news_strength=float(market_news_strength),
        stock_news_strength=float(stock_news_strength),
        use_learned_news_fusion=config.use_learned_news_fusion,
        learned_news_min_samples=config.learned_news_min_samples,
        learned_news_l2=config.learned_news_l2,
        learned_fusion_l2=config.learned_fusion_l2,
        max_positions=int(max_positions),
        use_turnover_control=config.use_turnover_control,
        max_trades_per_stock_per_week=int(config.max_trades_per_stock_per_week),
        min_weight_change_to_trade=float(config.min_weight_change_to_trade),
        use_margin_features=config.use_margin_features,
        margin_market_file=config.margin_market_file,
        margin_stock_file=config.margin_stock_file,
    )


def _pick_target_metric(metrics: List[BacktestMetrics], target_years: int) -> BacktestMetrics | None:
    labels = [
        f"融合策略-近{int(target_years)}年",
        "融合策略-全样本",
        f"近{int(target_years)}年",
        "全样本",
    ]
    for label in labels:
        for item in metrics:
            if item.label == label:
                return item
    return metrics[0] if metrics else None


def _strategy_objective(
    metrics: BacktestMetrics | None,
    *,
    turnover_penalty: float,
    drawdown_penalty: float,
) -> float:
    if metrics is None:
        return float("-inf")
    if pd.isna(metrics.excess_annual_return):
        return float("-inf")
    annual_turnover = float(max(0.0, metrics.annual_turnover)) if not pd.isna(metrics.annual_turnover) else 0.0
    max_dd = float(abs(metrics.max_drawdown)) if not pd.isna(metrics.max_drawdown) else 0.0
    return float(
        float(metrics.excess_annual_return)
        - float(turnover_penalty) * annual_turnover
        - float(drawdown_penalty) * max_dd
    )


def _optimize_strategy_selection(
    *,
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
    news_items_train: List[NewsItem],
) -> _StrategySelection:
    objective_text = (
        f"score = excess_annual_return - {float(config.optimizer_turnover_penalty):.4f}*annual_turnover "
        f"- {float(config.optimizer_drawdown_penalty):.3f}*abs(max_drawdown)"
    )
    target_metric_label = f"融合策略-近{int(config.optimizer_target_years)}年"
    baseline = _StrategySelection(
        retrain_days=int(config.backtest_retrain_days),
        weight_threshold=float(config.backtest_weight_threshold),
        max_positions=int(config.max_positions),
        market_news_strength=float(config.market_news_strength),
        stock_news_strength=float(config.stock_news_strength),
        objective_text=objective_text,
        target_metric_label=target_metric_label,
        selected_trial=None,
        trials=[],
        best_backtest=None,
    )
    if not config.use_strategy_optimizer:
        return baseline

    retrain_grid = _sanitize_int_grid(
        config.optimizer_retrain_days,
        fallback=int(config.backtest_retrain_days),
        min_value=1,
    )
    threshold_grid = _sanitize_float_grid(
        config.optimizer_weight_thresholds,
        fallback=float(config.backtest_weight_threshold),
        min_value=0.0,
    )
    max_pos_grid = _sanitize_int_grid(
        config.optimizer_max_positions,
        fallback=int(config.max_positions),
        min_value=1,
        max_value=int(config.max_positions),
    )
    market_strength_grid = _sanitize_float_grid(
        config.optimizer_market_news_strengths,
        fallback=float(config.market_news_strength),
        min_value=0.0,
    )
    stock_strength_grid = _sanitize_float_grid(
        config.optimizer_stock_news_strengths,
        fallback=float(config.stock_news_strength),
        min_value=0.0,
    )

    trials: list[StrategyTrial] = []
    best_trial: StrategyTrial | None = None
    best_backtest: BacktestResult | None = None
    best_score = float("-inf")

    for retrain_days, threshold, max_pos, market_strength, stock_strength in product(
        retrain_grid,
        threshold_grid,
        max_pos_grid,
        market_strength_grid,
        stock_strength_grid,
    ):
        try:
            backtest = _run_daily_backtest(
                config=config,
                market_security=market_security,
                stocks=stocks,
                news_items_train=news_items_train,
                retrain_days=int(retrain_days),
                weight_threshold=float(threshold),
                max_positions=int(max_pos),
                market_news_strength=float(market_strength),
                stock_news_strength=float(stock_strength),
            )
        except Exception:
            continue

        metric = _pick_target_metric(backtest.metrics, target_years=int(config.optimizer_target_years))
        score = _strategy_objective(
            metric,
            turnover_penalty=float(config.optimizer_turnover_penalty),
            drawdown_penalty=float(config.optimizer_drawdown_penalty),
        )
        if metric is None:
            continue
        trial = StrategyTrial(
            rank=0,
            metric_label=metric.label,
            retrain_days=int(retrain_days),
            weight_threshold=float(threshold),
            max_positions=int(max_pos),
            market_news_strength=float(market_strength),
            stock_news_strength=float(stock_strength),
            objective_score=float(score),
            annual_return=float(metric.annual_return),
            excess_annual_return=float(metric.excess_annual_return),
            max_drawdown=float(metric.max_drawdown),
            annual_turnover=float(metric.annual_turnover),
            total_cost=float(metric.total_cost),
            sharpe=float(metric.sharpe),
            avg_trades_per_stock_per_week=float(metric.avg_trades_per_stock_per_week),
        )
        trials.append(trial)
        if score > best_score:
            best_score = float(score)
            best_trial = trial
            best_backtest = backtest

    if not trials:
        return baseline

    trials.sort(key=lambda x: x.objective_score, reverse=True)
    top_n = max(1, int(config.optimizer_top_trials))
    trials = trials[:top_n]
    for idx, trial in enumerate(trials, start=1):
        trial.rank = int(idx)

    if best_trial is None:
        return baseline

    return _StrategySelection(
        retrain_days=int(best_trial.retrain_days),
        weight_threshold=float(best_trial.weight_threshold),
        max_positions=int(best_trial.max_positions),
        market_news_strength=float(best_trial.market_news_strength),
        stock_news_strength=float(best_trial.stock_news_strength),
        objective_text=objective_text,
        target_metric_label=target_metric_label,
        selected_trial=best_trial,
        trials=trials,
        best_backtest=best_backtest,
    )


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
            max_positions=config.max_positions,
            use_margin_features=config.use_margin_features,
            margin_market_file=config.margin_market_file,
            margin_stock_file=config.margin_stock_file,
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
    history_lookback_days = max(
        config.news_lookback_days,
        config.learned_news_lookback_days,
        int(max(30, (as_of_date - pd.Timestamp(config.start)).days + 7)),
    )
    news_items_train = load_news_items(
        config.news_file,
        as_of_date=as_of_date,
        lookback_days=history_lookback_days,
    )

    market_feat = pd.DataFrame()
    market_feature_cols = MARKET_FEATURE_COLUMNS
    stock_feature_cols_map: Dict[str, List[str]] = {}
    stock_feature_frames: Dict[str, pd.DataFrame] = {}
    if config.use_learned_news_fusion:
        market_feat, market_feature_cols, stock_feature_cols_map, stock_feature_frames = _prepare_learning_frames(
            config=config,
            market_security=market_security,
            stocks=stocks,
        )

    strategy_selection = _optimize_strategy_selection(
        config=config,
        market_security=market_security,
        stocks=stocks,
        news_items_train=news_items_train,
    )
    market_news_strength_live = float(strategy_selection.market_news_strength)
    stock_news_strength_live = float(strategy_selection.stock_news_strength)
    weight_threshold_live = float(strategy_selection.weight_threshold)
    max_positions_live = int(strategy_selection.max_positions)

    market_short_pred = predict_with_learned_fusion(
        enabled=config.use_learned_news_fusion,
        base_prob=market_forecast.short_prob,
        target="MARKET",
        horizon="short",
        feature_frame=market_feat,
        feature_cols=market_feature_cols,
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
        fallback_strength=market_news_strength_live,
    )
    market_mid_pred = predict_with_learned_fusion(
        enabled=config.use_learned_news_fusion,
        base_prob=market_forecast.mid_prob,
        target="MARKET",
        horizon="mid",
        feature_frame=market_feat,
        feature_cols=market_feature_cols,
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
        fallback_strength=market_news_strength_live,
    )

    market_final_short = market_short_pred.final_prob
    market_final_mid = market_mid_pred.final_prob

    blended_rows, stock_diagnostics = _blend_stock_rows(
        rows=stock_rows,
        news_items_train=news_items_train,
        news_items_live=news_items_live,
        as_of_date=as_of_date,
        config=config,
        stock_news_strength=stock_news_strength_live,
        stock_feature_cols_map=stock_feature_cols_map,
        stock_feature_frames=stock_feature_frames,
    )
    total = target_exposure(market_final_short, market_final_mid)
    weights = allocate_weights(
        [row.final_score for row in blended_rows],
        total_exposure=total,
        threshold=weight_threshold_live,
        max_positions=max_positions_live,
    )
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
    backtest = strategy_selection.best_backtest
    if backtest is None:
        backtest = _run_daily_backtest(
            config=config,
            market_security=market_security,
            stocks=stocks,
            news_items_train=news_items_train,
            retrain_days=strategy_selection.retrain_days,
            weight_threshold=strategy_selection.weight_threshold,
            max_positions=strategy_selection.max_positions,
            market_news_strength=strategy_selection.market_news_strength,
            stock_news_strength=strategy_selection.stock_news_strength,
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
        strategy_objective_text=strategy_selection.objective_text,
        strategy_target_metric_label=strategy_selection.target_metric_label,
        strategy_selected=strategy_selection.selected_trial,
        strategy_trials=strategy_selection.trials,
        news_items_count=len(news_items_live),
        news_items=news_items_live,
    )
