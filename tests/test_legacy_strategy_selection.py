from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.application.legacy_strategy_selection import optimize_strategy_selection
from src.domain.entities import BacktestMetrics, Security
from src.infrastructure.backtesting import BacktestResult


def _make_metric(*, excess_annual_return: float) -> BacktestMetrics:
    return BacktestMetrics(
        label="test",
        start_date=pd.Timestamp("2024-01-01"),
        end_date=pd.Timestamp("2024-12-31"),
        n_days=252,
        total_return=0.10,
        annual_return=0.10,
        annual_vol=0.20,
        sharpe=1.0,
        sortino=1.0,
        max_drawdown=-0.10,
        calmar=1.0,
        benchmark_total_return=0.02,
        benchmark_annual_return=0.02,
        excess_total_return=0.08,
        excess_annual_return=float(excess_annual_return),
        information_ratio=0.8,
        tracking_error=0.1,
        win_rate=0.55,
        avg_turnover=0.2,
        annual_turnover=1.0,
        total_cost=0.01,
        avg_trade_count_per_day=1.0,
        avg_trades_per_stock_per_week=0.5,
    )


def _make_config(**overrides: object) -> SimpleNamespace:
    config = {
        "optimizer_turnover_penalty": 0.0,
        "optimizer_drawdown_penalty": 0.0,
        "backtest_retrain_days": 20,
        "backtest_weight_threshold": 0.5,
        "max_positions": 5,
        "market_news_strength": 0.9,
        "stock_news_strength": 1.1,
        "use_learned_news_fusion": False,
        "use_strategy_optimizer": True,
        "optimizer_retrain_days": (20, 40),
        "optimizer_weight_thresholds": (0.5, 0.6),
        "optimizer_max_positions": (3, 5),
        "optimizer_market_news_strengths": (0.8, 1.0),
        "optimizer_stock_news_strengths": (1.0, 1.2),
        "optimizer_force_full_news_strength_grid": False,
        "optimizer_target_years": 3,
        "optimizer_top_trials": 12,
        "optimizer_time_budget_minutes": 0.0,
    }
    config.update(overrides)
    return SimpleNamespace(**config)


def _patch_optimizer_tape_reuse(monkeypatch):
    prepare_calls: list[tuple[int, float, float]] = []
    replay_calls: list[tuple[int, float, float, float, int]] = []

    def fake_prepare_daily_backtest_tape(**kwargs: object):
        key = (
            int(kwargs["retrain_days"]),
            float(kwargs["market_news_strength"]),
            float(kwargs["stock_news_strength"]),
        )
        prepare_calls.append(key)
        return {"key": key}

    def fake_replay_daily_backtest_tape(**kwargs: object) -> BacktestResult:
        prepared_tape = kwargs["prepared_tape"]
        retrain_days, market_strength, stock_strength = prepared_tape["key"]
        weight_threshold = float(kwargs["weight_threshold"])
        max_positions = int(kwargs["max_positions"])
        replay_calls.append((retrain_days, market_strength, stock_strength, weight_threshold, max_positions))
        excess = 0.01 * retrain_days + 0.1 * market_strength + 0.05 * stock_strength - 0.02 * weight_threshold + 0.005 * max_positions
        return BacktestResult(
            daily_frame=pd.DataFrame(),
            curve_frame=pd.DataFrame(),
            metrics=[_make_metric(excess_annual_return=excess)],
            audit={},
        )

    monkeypatch.setattr(
        "src.application.legacy_strategy_selection.prepare_daily_backtest_tape",
        fake_prepare_daily_backtest_tape,
    )
    monkeypatch.setattr(
        "src.application.legacy_strategy_selection.replay_daily_backtest_tape",
        fake_replay_daily_backtest_tape,
    )

    return prepare_calls, replay_calls


def test_optimize_strategy_selection_reuses_prepared_tapes(monkeypatch) -> None:
    prepare_calls, replay_calls = _patch_optimizer_tape_reuse(monkeypatch)

    config = _make_config()

    selection = optimize_strategy_selection(
        config=config,
        market_security=Security(symbol="000300.SH", name="CSI300"),
        stocks=[Security(symbol="000001.SZ", name="Ping An")],
        news_items_train=[],
    )

    assert len(prepare_calls) == 8
    assert len(set(prepare_calls)) == 8
    assert len(replay_calls) == 32
    assert selection.selected_trial is not None
    assert selection.best_backtest is not None


def test_optimize_strategy_selection_collapses_news_strength_grid_for_learned_fusion(monkeypatch) -> None:
    prepare_calls, replay_calls = _patch_optimizer_tape_reuse(monkeypatch)

    selection = optimize_strategy_selection(
        config=_make_config(use_learned_news_fusion=True),
        market_security=Security(symbol="000300.SH", name="CSI300"),
        stocks=[Security(symbol="000001.SZ", name="Ping An")],
        news_items_train=[],
    )

    assert len(prepare_calls) == 2
    assert set(prepare_calls) == {(20, 0.9, 1.1), (40, 0.9, 1.1)}
    assert len(replay_calls) == 8
    assert selection.selected_trial is not None
    assert selection.best_backtest is not None


def test_optimize_strategy_selection_can_force_full_news_strength_grid(monkeypatch) -> None:
    prepare_calls, replay_calls = _patch_optimizer_tape_reuse(monkeypatch)

    selection = optimize_strategy_selection(
        config=_make_config(
            use_learned_news_fusion=True,
            optimizer_force_full_news_strength_grid=True,
        ),
        market_security=Security(symbol="000300.SH", name="CSI300"),
        stocks=[Security(symbol="000001.SZ", name="Ping An")],
        news_items_train=[],
    )

    assert len(prepare_calls) == 8
    assert len(set(prepare_calls)) == 8
    assert len(replay_calls) == 32
    assert selection.selected_trial is not None
    assert selection.best_backtest is not None
