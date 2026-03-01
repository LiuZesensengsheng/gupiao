from __future__ import annotations

from pathlib import Path

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    LearnedPolicyModel,
    MarketForecastState,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
from src.application.v2_services import (
    _policy_spec_from_model,
    load_published_v2_policy_model,
    publish_v2_research_artifacts,
)


def _make_state() -> CompositeState:
    return CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-01",
            up_1d_prob=0.58,
            up_5d_prob=0.60,
            up_20d_prob=0.63,
            trend_state="trend",
            drawdown_risk=0.18,
            volatility_regime="normal",
            liquidity_stress=0.22,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-01",
            large_vs_small_bias=0.08,
            growth_vs_value_bias=-0.03,
            fund_flow_strength=0.12,
            margin_risk_on_score=0.10,
            breadth_strength=0.20,
            leader_participation=0.61,
            weak_stock_ratio=0.24,
        ),
        sectors=[
            SectorForecastState("有色", 0.57, 0.64, 0.18, 0.22, 0.18),
            SectorForecastState("化工", 0.54, 0.58, 0.10, 0.25, 0.16),
        ],
        stocks=[
            StockForecastState("AAA", "有色", 0.58, 0.60, 0.66, 0.56, 0.04, 0.88),
            StockForecastState("BBB", "化工", 0.54, 0.56, 0.60, 0.51, 0.02, 0.82),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )


def _make_backtest(total_return: float, annual_return: float) -> V2BacktestSummary:
    return V2BacktestSummary(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_days=200,
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=-0.10,
        avg_turnover=0.22,
        total_cost=0.02,
        gross_total_return=total_return + 0.02,
        annual_vol=0.18,
        win_rate=0.54,
        trade_days=140,
        avg_fill_ratio=0.76,
        avg_slippage_bps=2.4,
        nav_curve=[1.0, 1.05, 1.10],
        curve_dates=["2024-01-01", "2024-06-01", "2024-12-31"],
    )


def test_policy_model_projects_state_into_valid_policy_spec() -> None:
    state = _make_state()
    model = LearnedPolicyModel(
        feature_names=[
            "mkt_up_1d",
            "mkt_up_20d",
            "mkt_drawdown_risk",
            "mkt_liquidity_stress",
            "cross_fund_flow",
            "cross_margin_risk_on",
            "cross_breadth",
            "cross_leader_participation",
            "cross_weak_ratio",
            "top_sector_up_20d",
            "top_sector_relative_strength",
            "top_stock_up_20d",
            "top_stock_tradeability",
            "top_stock_excess_vs_sector",
        ],
        exposure_intercept=0.15,
        exposure_coef=[0.05] * 14,
        position_intercept=1.0,
        position_coef=[0.2] * 14,
        turnover_intercept=0.08,
        turnover_coef=[0.01] * 14,
        train_rows=120,
        train_r2_exposure=0.30,
        train_r2_positions=0.22,
        train_r2_turnover=0.18,
    )

    spec = _policy_spec_from_model(state=state, model=model)

    assert isinstance(spec, PolicySpec)
    assert 0.20 <= spec.risk_on_exposure <= 0.95
    assert 1 <= spec.risk_on_positions <= 6
    assert 0.10 <= spec.risk_on_turnover_cap <= 0.45


def test_publish_artifacts_writes_and_loads_latest_policy(tmp_path: Path) -> None:
    baseline = _make_backtest(0.20, 0.18)
    calibrated = _make_backtest(0.22, 0.20)
    learned = _make_backtest(0.24, 0.22)
    learning_result = V2PolicyLearningResult(
        model=LearnedPolicyModel(
            feature_names=["x1", "x2"],
            exposure_intercept=0.55,
            exposure_coef=[0.1, 0.2],
            position_intercept=2.5,
            position_coef=[0.05, 0.08],
            turnover_intercept=0.20,
            turnover_coef=[0.01, 0.02],
            train_rows=88,
            train_r2_exposure=0.33,
            train_r2_positions=0.25,
            train_r2_turnover=0.19,
        ),
        baseline=baseline,
        learned=learned,
    )
    calibration = V2CalibrationResult(
        best_policy=PolicySpec(),
        best_score=0.12,
        baseline=baseline,
        calibrated=calibrated,
        trials=[],
    )

    paths = publish_v2_research_artifacts(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path),
        settings={
            "config_path": "config/api.json",
            "source": "local",
            "watchlist": "config/watchlist.json",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 5,
            "start": "2024-01-01",
            "end": "2024-12-31",
        },
        baseline=baseline,
        calibration=calibration,
        learning=learning_result,
    )

    assert Path(paths["research_manifest"]).exists()
    loaded = load_published_v2_policy_model(strategy_id="swing_v2", artifact_root=str(tmp_path))
    assert loaded is not None
    assert loaded.train_rows == 88
    assert loaded.exposure_coef == [0.1, 0.2]


def test_backtest_summary_carries_cross_section_metrics() -> None:
    summary = _make_backtest(0.18, 0.16)

    assert summary.avg_rank_ic == 0.0
    assert summary.avg_top_decile_return == 0.0
    assert summary.avg_top_bottom_spread == 0.0
    assert summary.avg_top_k_hit_rate == 0.0
