from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from typing import List

import numpy as np
import pandas as pd

from src.application.config import DailyConfig
from src.domain.entities import BacktestMetrics, NewsItem, Security, StrategyTrial
from src.infrastructure.backtesting import BacktestResult, run_portfolio_backtest


@dataclass(frozen=True)
class StrategySelection:
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


def sanitize_int_grid(
    values: tuple[int, ...],
    *,
    fallback: int,
    min_value: int = 1,
    max_value: int | None = None,
) -> list[int]:
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


def sanitize_float_grid(
    values: tuple[float, ...],
    *,
    fallback: float,
    min_value: float = 0.0,
) -> list[float]:
    out: list[float] = []
    for raw in values:
        value = float(raw)
        if value < float(min_value):
            continue
        out.append(value)
    uniq = sorted({round(value, 6) for value in out})
    return [float(value) for value in uniq] if uniq else [float(fallback)]


def run_daily_backtest(
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
    max_trades_per_stock_per_day: int,
    max_trades_per_stock_per_week: int,
    use_state_engine: bool = True,
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
        max_runtime_seconds=max(0.0, float(config.backtest_time_budget_minutes) * 60.0),
        news_half_life_days=config.news_half_life_days,
        market_news_strength=float(market_news_strength),
        stock_news_strength=float(stock_news_strength),
        use_learned_news_fusion=config.use_learned_news_fusion,
        learned_news_min_samples=config.learned_news_min_samples,
        learned_news_l2=config.learned_news_l2,
        learned_fusion_l2=config.learned_fusion_l2,
        max_positions=int(max_positions),
        use_turnover_control=config.use_turnover_control,
        max_trades_per_stock_per_day=int(max_trades_per_stock_per_day),
        max_trades_per_stock_per_week=int(max_trades_per_stock_per_week),
        min_weight_change_to_trade=float(config.min_weight_change_to_trade),
        range_t_sell_ret_1_min=float(config.range_t_sell_ret_1_min),
        range_t_sell_price_pos_20_min=float(config.range_t_sell_price_pos_20_min),
        range_t_buy_ret_1_max=float(config.range_t_buy_ret_1_max),
        range_t_buy_price_pos_20_max=float(config.range_t_buy_price_pos_20_max),
        use_tradeability_guard=bool(config.use_tradeability_guard),
        tradeability_limit_tolerance=float(config.tradeability_limit_tolerance),
        tradeability_min_volume=float(config.tradeability_min_volume),
        limit_rule_file=str(config.limit_rule_file),
        use_index_constituent_guard=bool(config.use_index_constituent_guard),
        index_constituent_file=str(config.index_constituent_file),
        index_constituent_symbol=str(config.index_constituent_symbol),
        use_margin_features=config.use_margin_features,
        margin_market_file=config.margin_market_file,
        margin_stock_file=config.margin_stock_file,
        use_state_engine=bool(use_state_engine),
    )


def metric_delta(new: float, old: float) -> float:
    if pd.isna(new) or pd.isna(old):
        return np.nan
    return float(new - old)


def pick_target_metric(metrics: List[BacktestMetrics], target_years: int) -> BacktestMetrics | None:
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


def strategy_objective(
    metrics: BacktestMetrics | None,
    *,
    turnover_penalty: float,
    drawdown_penalty: float,
) -> float:
    if metrics is None or pd.isna(metrics.excess_annual_return):
        return float("-inf")
    annual_turnover = float(max(0.0, metrics.annual_turnover)) if not pd.isna(metrics.annual_turnover) else 0.0
    max_drawdown = float(abs(metrics.max_drawdown)) if not pd.isna(metrics.max_drawdown) else 0.0
    return float(
        float(metrics.excess_annual_return)
        - float(turnover_penalty) * annual_turnover
        - float(drawdown_penalty) * max_drawdown
    )


def optimize_strategy_selection(
    *,
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
    news_items_train: List[NewsItem],
) -> StrategySelection:
    objective_text = (
        f"score = excess_annual_return - {float(config.optimizer_turnover_penalty):.4f}*annual_turnover "
        f"- {float(config.optimizer_drawdown_penalty):.3f}*abs(max_drawdown)"
    )
    target_metric_label = f"融合策略-近{int(config.optimizer_target_years)}年"
    baseline = StrategySelection(
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

    retrain_grid = sanitize_int_grid(config.optimizer_retrain_days, fallback=int(config.backtest_retrain_days))
    threshold_grid = sanitize_float_grid(
        config.optimizer_weight_thresholds,
        fallback=float(config.backtest_weight_threshold),
    )
    max_pos_grid = sanitize_int_grid(
        config.optimizer_max_positions,
        fallback=int(config.max_positions),
        max_value=int(config.max_positions),
    )
    market_strength_grid = sanitize_float_grid(
        config.optimizer_market_news_strengths,
        fallback=float(config.market_news_strength),
    )
    stock_strength_grid = sanitize_float_grid(
        config.optimizer_stock_news_strengths,
        fallback=float(config.stock_news_strength),
    )

    trials: list[StrategyTrial] = []
    best_trial: StrategyTrial | None = None
    best_backtest: BacktestResult | None = None
    best_score = float("-inf")
    total_trials = (
        len(retrain_grid)
        * len(threshold_grid)
        * len(max_pos_grid)
        * len(market_strength_grid)
        * len(stock_strength_grid)
    )
    time_budget_sec = max(0.0, float(config.optimizer_time_budget_minutes) * 60.0)
    start_ts = time.monotonic()
    budget_text = "unlimited" if time_budget_sec <= 0 else f"{time_budget_sec:.0f}s"
    print(f"[OPT] strategy search started: trials={total_trials}, budget={budget_text}")

    for trial_idx, (retrain_days, threshold, max_pos, market_strength, stock_strength) in enumerate(
        product(retrain_grid, threshold_grid, max_pos_grid, market_strength_grid, stock_strength_grid),
        start=1,
    ):
        elapsed = time.monotonic() - start_ts
        if time_budget_sec > 0 and elapsed >= time_budget_sec:
            print(f"[OPT] time budget reached at {elapsed:.1f}s, stop search ({trial_idx - 1}/{total_trials} trials).")
            break
        print(
            f"[OPT] trial {trial_idx}/{total_trials} "
            f"(retrain={int(retrain_days)}, threshold={float(threshold):.2f}, "
            f"max_pos={int(max_pos)}, m_news={float(market_strength):.2f}, s_news={float(stock_strength):.2f})"
        )
        try:
            backtest = run_daily_backtest(
                config=config,
                market_security=market_security,
                stocks=stocks,
                news_items_train=news_items_train,
                retrain_days=int(retrain_days),
                weight_threshold=float(threshold),
                max_positions=int(max_pos),
                market_news_strength=float(market_strength),
                stock_news_strength=float(stock_strength),
                max_trades_per_stock_per_day=int(config.max_trades_per_stock_per_day),
                max_trades_per_stock_per_week=int(config.max_trades_per_stock_per_week),
            )
        except Exception as exc:
            print(f"[OPT] trial {trial_idx}/{total_trials} failed: {exc}")
            continue

        metric = pick_target_metric(backtest.metrics, target_years=int(config.optimizer_target_years))
        score = strategy_objective(
            metric,
            turnover_penalty=float(config.optimizer_turnover_penalty),
            drawdown_penalty=float(config.optimizer_drawdown_penalty),
        )
        if metric is None:
            print(f"[OPT] trial {trial_idx}/{total_trials} skipped: missing target metric")
            continue
        print(
            f"[OPT] trial {trial_idx}/{total_trials} result: "
            f"objective={float(score):.4f}, annual={float(metric.annual_return):.2%}, "
            f"excess={float(metric.excess_annual_return):.2%}, max_dd={float(metric.max_drawdown):.2%}"
        )
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
            print(f"[OPT] new best at trial {trial_idx}/{total_trials}: score={best_score:.4f}, label={metric.label}")

    if not trials:
        return baseline
    trials.sort(key=lambda item: item.objective_score, reverse=True)
    trials = trials[: max(1, int(config.optimizer_top_trials))]
    for idx, trial in enumerate(trials, start=1):
        trial.rank = int(idx)
    if best_trial is None:
        return baseline
    return StrategySelection(
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
