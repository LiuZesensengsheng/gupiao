from __future__ import annotations

import numpy as np

from src.application.v2_contracts import (
    CapitalFlowState,
    CompositeState,
    CrossSectionForecastState,
    HorizonForecast,
    InfoAggregateState,
    MacroContextState,
    MarketFactsState,
    MarketForecastState,
    PolicyDecision,
    SectorForecastState,
    StockForecastState,
)
from src.application.v2_forecast_model_runtime import ReturnQuantileProfile
from src.reporting.forecast_support import (
    ForecastSupportDependencies,
    alpha_score_components,
    build_horizon_forecasts,
    build_market_sentiment_state,
)
from src.reporting.reason_bundles import stock_reason_bundle


def _forecast_support_dependencies() -> ForecastSupportDependencies:
    return ForecastSupportDependencies(
        clip=lambda value, lo, hi: float(np.clip(float(value), float(lo), float(hi))),
        return_quantile_profile_cls=ReturnQuantileProfile,
    )


def test_forecast_support_builds_horizons_and_sentiment() -> None:
    deps = _forecast_support_dependencies()
    short_profile = ReturnQuantileProfile(
        expected_return=0.02,
        q10=-0.03,
        q30=-0.01,
        q20=-0.02,
        q50=0.01,
        q70=0.03,
        q80=0.04,
        q90=0.06,
    )
    mid_profile = ReturnQuantileProfile(
        expected_return=0.08,
        q10=-0.06,
        q30=-0.02,
        q20=-0.04,
        q50=0.05,
        q70=0.09,
        q80=0.11,
        q90=0.14,
    )
    forecasts = build_horizon_forecasts(
        latest_close=10.0,
        horizon_probs={"1d": 0.58, "2d": 0.57, "3d": 0.59, "5d": 0.61, "10d": 0.63, "20d": 0.66},
        short_profile=short_profile,
        mid_profile=mid_profile,
        info_state=InfoAggregateState(coverage_confidence=0.72),
        calibration_priors={"5d": {"rank_ic": 0.12, "top_k_hit_rate": 0.64}},
        tradability_status="normal",
        deps=deps,
    )
    assert set(forecasts) == {"1d", "2d", "3d", "5d", "10d", "20d"}
    assert forecasts["1d"].up_prob == 0.58
    assert forecasts["20d"].price_high > forecasts["20d"].price_low
    assert 0.0 < forecasts["10d"].confidence <= 1.0

    market = MarketForecastState(
        as_of_date="2026-03-16",
        up_1d_prob=0.56,
        up_5d_prob=0.60,
        up_20d_prob=0.64,
        trend_state="trend",
        drawdown_risk=0.14,
        volatility_regime="normal",
        liquidity_stress=0.10,
        market_facts=MarketFactsState(
            sample_coverage=100,
            advancers=62,
            decliners=30,
            flats=8,
            limit_up_count=5,
            limit_down_count=1,
            new_high_count=12,
            new_low_count=4,
            median_return=0.01,
            sample_amount=1.2e9,
            amount_z20=0.5,
        ),
    )
    sentiment = build_market_sentiment_state(
        market=market,
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-16",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.01,
            fund_flow_strength=0.03,
            margin_risk_on_score=0.04,
            breadth_strength=0.15,
            leader_participation=0.16,
            weak_stock_ratio=0.18,
        ),
        capital_flow=CapitalFlowState(
            northbound_net_flow=0.12,
            margin_balance_change=0.05,
            turnover_heat=0.65,
            large_order_bias=0.04,
            flow_regime="risk_on",
        ),
        macro=MacroContextState(
            style_regime="growth",
            commodity_pressure=0.02,
            fx_pressure=0.01,
            index_breadth_proxy=0.6,
            macro_risk_level="neutral",
        ),
    )
    assert sentiment.score > 50.0
    assert sentiment.stage
    assert len(sentiment.drivers) == 4


def test_stock_reason_bundle_uses_alpha_components_and_policy_context() -> None:
    deps = _forecast_support_dependencies()
    stock = StockForecastState(
        symbol="AAA",
        sector="科技",
        up_1d_prob=0.67,
        up_5d_prob=0.59,
        up_20d_prob=0.64,
        excess_vs_sector_prob=0.58,
        event_impact_score=0.12,
        tradeability_score=0.86,
        latest_close=12.3,
        horizon_forecasts={
            "1d": HorizonForecast(1, "1日", up_prob=0.67, price_low=11.8, price_high=12.9, confidence=0.52),
            "5d": HorizonForecast(5, "5日", up_prob=0.59),
            "20d": HorizonForecast(20, "20日", up_prob=0.64),
        },
    )
    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-15",
            up_1d_prob=0.57,
            up_5d_prob=0.59,
            up_20d_prob=0.61,
            trend_state="trend",
            drawdown_risk=0.16,
            volatility_regime="normal",
            liquidity_stress=0.12,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-15",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.03,
            fund_flow_strength=0.04,
            margin_risk_on_score=0.05,
            breadth_strength=0.11,
            leader_participation=0.16,
            weak_stock_ratio=0.21,
        ),
        sectors=[SectorForecastState("科技", 0.60, 0.63, 0.10, 0.18, 0.14)],
        stocks=[stock],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    policy = PolicyDecision(
        target_exposure=0.85,
        target_position_count=5,
        rebalance_now=True,
        rebalance_intensity=0.8,
        intraday_t_allowed=False,
        turnover_cap=0.25,
        sector_budgets={"科技": 0.25},
        desired_sector_budgets={"科技": 0.30},
        symbol_target_weights={"AAA": 0.18},
        desired_symbol_target_weights={"AAA": 0.22},
    )

    alpha_parts = alpha_score_components(
        StockForecastState(
            symbol="AAA",
            sector="科技",
            up_1d_prob=0.62,
            up_5d_prob=0.64,
            up_20d_prob=0.68,
            excess_vs_sector_prob=0.58,
            event_impact_score=0.12,
            tradeability_score=0.86,
            alpha_score=0.0,
            tradability_status="normal",
            up_2d_prob=0.63,
            up_3d_prob=0.65,
        ),
        deps=deps,
    )
    assert alpha_parts["alpha_score"] == alpha_parts["alpha_score"]
    assert alpha_parts["selection_bonus"] == alpha_parts["selection_bonus"]

    bundle = stock_reason_bundle(
        stock=stock,
        info_state=InfoAggregateState(catalyst_strength=0.60, negative_event_risk=0.12),
        state=state,
        rank=2,
        policy=policy,
        alpha_score_components=lambda candidate: alpha_score_components(candidate, deps=deps),
    )

    assert bundle[0]
    assert bundle[1]
    assert bundle[2]
    assert "11.80" in bundle[3]
    assert "50%" in bundle[3]
    assert "18.00%" in bundle[5]
    assert "失效" in bundle[3]
    assert "目标权重" in bundle[5]
