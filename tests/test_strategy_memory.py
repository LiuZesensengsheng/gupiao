from __future__ import annotations

from pathlib import Path

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    DailyRunResult,
    LearnedPolicyModel,
    MarketForecastState,
    PolicyDecision,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    StrategySnapshot,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
from src.domain.entities import TradeAction
from src.infrastructure.strategy_memory import remember_daily_run, remember_research_run


def _backtest(excess_annual_return: float) -> V2BacktestSummary:
    return V2BacktestSummary(
        start_date="2025-01-01",
        end_date="2026-03-10",
        n_days=240,
        total_return=0.20,
        annual_return=0.18,
        max_drawdown=-0.09,
        avg_turnover=0.14,
        total_cost=0.01,
        excess_annual_return=excess_annual_return,
        information_ratio=0.66,
    )


def _daily_result() -> DailyRunResult:
    return DailyRunResult(
        snapshot=StrategySnapshot(
            strategy_id="swing_v2",
            universe_id="smoke",
            feature_set_version="v2",
            market_model_id="market_v2",
            sector_model_id="sector_v2",
            stock_model_id="stock_v2",
            cross_section_model_id="cross_v2",
            policy_version="policy_v2",
            execution_version="exec_v2",
            run_id="20260310_210000",
        ),
        composite_state=CompositeState(
            market=MarketForecastState(
                as_of_date="2026-03-11",
                up_1d_prob=0.55,
                up_5d_prob=0.60,
                up_20d_prob=0.63,
                trend_state="trend",
                drawdown_risk=0.20,
                volatility_regime="normal",
                liquidity_stress=0.15,
            ),
            cross_section=CrossSectionForecastState(
                as_of_date="2026-03-11",
                large_vs_small_bias=0.08,
                growth_vs_value_bias=-0.02,
                fund_flow_strength=0.10,
                margin_risk_on_score=0.06,
                breadth_strength=0.18,
                leader_participation=0.61,
                weak_stock_ratio=0.22,
            ),
            sectors=[SectorForecastState("有色", 0.58, 0.62, 0.10, 0.12, 0.20)],
            stocks=[StockForecastState("AAA", "有色", 0.56, 0.61, 0.66, 0.55, 0.02, 0.90)],
            strategy_mode="trend_follow",
            risk_regime="risk_on",
        ),
        policy_decision=PolicyDecision(
            target_exposure=0.72,
            target_position_count=1,
            rebalance_now=True,
            rebalance_intensity=0.20,
            intraday_t_allowed=False,
            turnover_cap=0.24,
            sector_budgets={"有色": 0.72},
            symbol_target_weights={"AAA": 0.72},
            risk_notes=["earnings_negative"],
        ),
        trade_actions=[
            TradeAction(
                symbol="AAA",
                name="示例股",
                action="BUY",
                current_weight=0.20,
                target_weight=0.72,
                delta_weight=0.52,
                note="加仓",
            )
        ],
        run_id="20260310_210000",
    )


def test_strategy_memory_persists_research_and_daily_recall(tmp_path: Path) -> None:
    memory_root = tmp_path / "memory"
    baseline = _backtest(0.03)
    calibrated = _backtest(0.05)
    learned = _backtest(0.08)

    memory_path = remember_research_run(
        memory_root=memory_root,
        strategy_id="swing_v2",
        run_id="20260310_210000",
        baseline=baseline,
        calibration=V2CalibrationResult(
            best_policy=PolicySpec(),
            best_score=0.12,
            baseline=baseline,
            calibrated=calibrated,
        ),
        learning=V2PolicyLearningResult(
            model=LearnedPolicyModel(
                feature_names=["x1"],
                exposure_intercept=0.5,
                exposure_coef=[0.1],
                position_intercept=3.0,
                position_coef=[0.2],
                turnover_intercept=0.2,
                turnover_coef=[0.05],
                train_rows=64,
                train_r2_exposure=0.12,
                train_r2_positions=0.10,
                train_r2_turnover=0.08,
            ),
            baseline=baseline,
            learned=learned,
        ),
        release_gate_passed=True,
        universe_id="smoke",
        universe_tier="favorites_16",
        universe_size=16,
    )

    result = remember_daily_run(
        memory_root=memory_root,
        result=_daily_result(),
    )

    assert memory_path.exists()
    assert result.memory_path.endswith("swing_v2_memory.json")
    assert result.memory_recall.latest_research_run_id == "20260310_210000"
    assert result.memory_recall.latest_research_release_gate_passed is True
    assert result.memory_recall.recent_daily_run_count == 1
    assert result.memory_recall.recurring_symbols == ["AAA"]
    assert "最近一次研究" in " ".join(result.memory_recall.narrative)
