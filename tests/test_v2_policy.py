from __future__ import annotations

from src.application.v2_contracts import DailyRunResult, StrategySnapshot
from src.application.v2_contracts import PolicyInput
from src.application.v2_services import (
    apply_policy,
    build_trade_actions,
    build_demo_forecast_states,
    build_demo_snapshot,
    compose_state,
    run_daily_v2,
    summarize_daily_run,
)


def test_v2_policy_returns_bounded_exposure_and_weights() -> None:
    market, sectors, stocks, cross_section = build_demo_forecast_states()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights={"000630.SZ": 0.10},
            current_cash=0.90,
            total_equity=1.0,
        )
    )

    assert 0.0 <= decision.target_exposure <= 1.0
    assert 1 <= decision.target_position_count <= 4
    assert sum(decision.symbol_target_weights.values()) <= decision.target_exposure + 1e-9
    assert all(weight >= 0.0 for weight in decision.symbol_target_weights.values())


def test_v2_policy_reduces_exposure_under_risk_off_inputs() -> None:
    market, sectors, stocks, cross_section = build_demo_forecast_states()
    stressed_market = market.__class__(
        as_of_date=market.as_of_date,
        up_1d_prob=0.42,
        up_5d_prob=0.44,
        up_20d_prob=0.45,
        trend_state="risk_off",
        drawdown_risk=0.72,
        volatility_regime=market.volatility_regime,
        liquidity_stress=0.68,
    )
    stressed_cross = cross_section.__class__(
        as_of_date=cross_section.as_of_date,
        large_vs_small_bias=cross_section.large_vs_small_bias,
        growth_vs_value_bias=cross_section.growth_vs_value_bias,
        fund_flow_strength=-0.10,
        margin_risk_on_score=cross_section.margin_risk_on_score,
        breadth_strength=0.02,
        leader_participation=cross_section.leader_participation,
        weak_stock_ratio=0.70,
    )
    composite_state = compose_state(
        market=stressed_market,
        sectors=sectors,
        stocks=stocks,
        cross_section=stressed_cross,
    )
    decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights={"000630.SZ": 0.40, "600160.SH": 0.25},
            current_cash=0.35,
            total_equity=1.0,
        )
    )

    assert composite_state.risk_regime == "risk_off"
    assert decision.target_exposure <= 0.35
    assert decision.rebalance_now is True


def test_compose_state_sorts_best_sector_and_stock_first() -> None:
    market, sectors, stocks, cross_section = build_demo_forecast_states()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    assert composite_state.strategy_mode == "trend_follow"
    assert composite_state.risk_regime == "risk_on"
    assert composite_state.sectors[0].sector == "有色"
    assert composite_state.stocks[0].symbol == "000630.SZ"


def test_v2_policy_sector_budgets_are_bounded_by_target_exposure() -> None:
    market, sectors, stocks, cross_section = build_demo_forecast_states()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        )
    )

    assert sum(decision.sector_budgets.values()) <= decision.target_exposure + 1e-9
    assert set(decision.symbol_target_weights).issubset(
        {stock.symbol for stock in composite_state.stocks}
    )


def test_run_daily_v2_returns_structured_summary() -> None:
    result = run_daily_v2("swing_v2")

    assert isinstance(result, DailyRunResult)
    assert isinstance(result.snapshot, StrategySnapshot)
    assert result.snapshot == build_demo_snapshot("swing_v2")

    summary = summarize_daily_run(result)
    assert summary["strategy_id"] == "swing_v2"
    assert summary["strategy_mode"] == result.composite_state.strategy_mode
    assert summary["risk_regime"] == result.composite_state.risk_regime
    assert "market" in summary
    assert "policy" in summary
    assert "trade_plan" in summary
    assert len(summary["trade_plan"]) == len(result.trade_actions)


def test_build_trade_actions_marks_buy_sell_and_hold() -> None:
    actions = build_trade_actions(
        decision=type(
            "DecisionStub",
            (),
            {
                "symbol_target_weights": {
                    "AAA": 0.40,
                    "BBB": 0.10,
                    "CCC": 0.005,
                }
            },
        )(),
        current_weights={
            "AAA": 0.10,
            "BBB": 0.20,
            "CCC": 0.0,
            "DDD": 0.01,
        },
    )

    action_map = {item.symbol: item for item in actions}
    assert action_map["AAA"].action == "BUY"
    assert action_map["BBB"].action == "SELL"
    assert action_map["CCC"].action == "HOLD"
    assert action_map["DDD"].action == "HOLD"
