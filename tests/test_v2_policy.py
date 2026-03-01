from __future__ import annotations

import pandas as pd

from src.application.v2_contracts import DailyRunResult, StrategySnapshot
from src.application.v2_contracts import PolicyInput
from src.application.v2_services import (
    _stock_policy_score,
    apply_policy,
    build_strategy_snapshot,
    build_trade_actions,
    compose_state,
    summarize_daily_run,
)
from src.application.v2_contracts import (
    CrossSectionForecastState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
)


def _make_demo_state() -> tuple[
    MarketForecastState,
    list[SectorForecastState],
    list[StockForecastState],
    CrossSectionForecastState,
]:
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.57,
        up_5d_prob=0.59,
        up_20d_prob=0.61,
        trend_state="trend",
        drawdown_risk=0.28,
        volatility_regime="normal",
        liquidity_stress=0.22,
    )
    sectors = [
        SectorForecastState("有色", 0.58, 0.62, 0.18, 0.42, 0.31),
        SectorForecastState("化工", 0.55, 0.60, 0.12, 0.36, 0.27),
        SectorForecastState("科技", 0.49, 0.53, -0.03, 0.51, 0.48),
    ]
    stocks = [
        StockForecastState("000630.SZ", "有色", 0.58, 0.60, 0.64, 0.55, 0.10, 0.88),
        StockForecastState("600160.SH", "化工", 0.56, 0.58, 0.62, 0.51, 0.07, 0.84),
        StockForecastState("603619.SH", "化工", 0.52, 0.56, 0.57, 0.49, 0.05, 0.79),
        StockForecastState("603516.SH", "科技", 0.47, 0.50, 0.54, 0.44, 0.03, 0.76),
    ]
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.08,
        growth_vs_value_bias=-0.04,
        fund_flow_strength=0.16,
        margin_risk_on_score=0.14,
        breadth_strength=0.21,
        leader_participation=0.63,
        weak_stock_ratio=0.29,
    )
    return market, sectors, stocks, cross_section


def test_v2_policy_returns_bounded_exposure_and_weights() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
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
    market, sectors, stocks, cross_section = _make_demo_state()
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
    market, sectors, stocks, cross_section = _make_demo_state()
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
    market, sectors, stocks, cross_section = _make_demo_state()
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
    by_sector = {}
    for stock in composite_state.stocks:
        if stock.symbol not in decision.symbol_target_weights:
            continue
        by_sector[stock.sector] = by_sector.get(stock.sector, 0.0) + decision.symbol_target_weights[stock.symbol]
    for sector, total in by_sector.items():
        assert total <= decision.sector_budgets.get(sector, 0.0) + 1e-9


def test_summarize_daily_run_returns_structured_summary() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights={"000630.SZ": 0.20},
            current_cash=0.80,
            total_equity=1.0,
        )
    )
    result = DailyRunResult(
        snapshot=build_strategy_snapshot(strategy_id="swing_v2", universe_id="demo_universe"),
        composite_state=composite_state,
        policy_decision=decision,
        trade_actions=build_trade_actions(decision=decision, current_weights={"000630.SZ": 0.20}),
    )

    assert isinstance(result, DailyRunResult)
    assert isinstance(result.snapshot, StrategySnapshot)
    assert result.snapshot == build_strategy_snapshot(strategy_id="swing_v2", universe_id="demo_universe")

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


def test_v2_policy_allocates_within_sector_budgets_only() -> None:
    market, _, _, cross_section = _make_demo_state()
    sectors = [
        type("Sector", (), {"sector": "强", "up_5d_prob": 0.60, "up_20d_prob": 0.66, "relative_strength": 0.20, "rotation_speed": 0.2, "crowding_score": 0.3})(),
        type("Sector", (), {"sector": "弱", "up_5d_prob": 0.45, "up_20d_prob": 0.48, "relative_strength": -0.05, "rotation_speed": 0.4, "crowding_score": 0.2})(),
    ]
    stocks = [
        type("Stock", (), {"symbol": "A1", "sector": "强", "up_1d_prob": 0.55, "up_5d_prob": 0.60, "up_20d_prob": 0.68, "excess_vs_sector_prob": 0.62, "event_impact_score": 0.0, "tradeability_score": 0.9})(),
        type("Stock", (), {"symbol": "A2", "sector": "强", "up_1d_prob": 0.53, "up_5d_prob": 0.57, "up_20d_prob": 0.63, "excess_vs_sector_prob": 0.56, "event_impact_score": 0.0, "tradeability_score": 0.85})(),
        type("Stock", (), {"symbol": "B1", "sector": "弱", "up_1d_prob": 0.61, "up_5d_prob": 0.64, "up_20d_prob": 0.70, "excess_vs_sector_prob": 0.65, "event_impact_score": 0.0, "tradeability_score": 0.95})(),
    ]
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

    weak_total = sum(
        weight for symbol, weight in decision.symbol_target_weights.items()
        if symbol.startswith("B")
    )
    assert weak_total <= decision.sector_budgets.get("弱", 0.0) + 1e-9


def test_stock_policy_score_uses_explicit_five_day_head() -> None:
    slow_burn = StockForecastState(
        "AAA",
        "强",
        0.50,
        0.46,
        0.70,
        0.55,
        0.02,
        0.86,
    )
    fast_follow = StockForecastState(
        "BBB",
        "强",
        0.56,
        0.72,
        0.62,
        0.55,
        0.02,
        0.86,
    )

    assert _stock_policy_score(fast_follow) > _stock_policy_score(slow_burn)


def test_compose_state_prefers_stronger_five_day_stock_when_other_signals_close() -> None:
    market, sectors, _, cross_section = _make_demo_state()
    stocks = [
        StockForecastState("AAA", "有色", 0.52, 0.48, 0.66, 0.54, 0.01, 0.84),
        StockForecastState("BBB", "有色", 0.56, 0.71, 0.62, 0.53, 0.01, 0.84),
    ]

    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    assert composite_state.stocks[0].symbol == "BBB"


def test_v2_policy_filters_halted_stock_and_freezes_existing_position() -> None:
    market, sectors, _, cross_section = _make_demo_state()
    stocks = [
        StockForecastState("AAA", "有色", 0.58, 0.63, 0.68, 0.57, 0.02, 0.90, tradability_status="halted"),
        StockForecastState("BBB", "有色", 0.54, 0.59, 0.63, 0.55, 0.02, 0.86),
    ]
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights={"AAA": 0.18},
            current_cash=0.82,
            total_equity=1.0,
        )
    )

    assert decision.symbol_target_weights["AAA"] == 0.18
    assert "BBB" in decision.symbol_target_weights
    assert any("holding frozen" in note for note in decision.risk_notes)


def test_v2_policy_blocks_new_entry_when_data_is_insufficient() -> None:
    market, sectors, _, cross_section = _make_demo_state()
    stocks = [
        StockForecastState("AAA", "有色", 0.60, 0.66, 0.70, 0.60, 0.03, 0.88, tradability_status="data_insufficient"),
        StockForecastState("BBB", "化工", 0.55, 0.58, 0.62, 0.53, 0.02, 0.84),
    ]
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

    assert "AAA" not in decision.symbol_target_weights
    assert any("new entry blocked" in note for note in decision.risk_notes)


def test_v2_policy_suppresses_small_rebalance_gap() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights={
                "000630.SZ": 0.20,
                "600160.SH": 0.19,
                "603619.SH": 0.18,
                "603516.SH": 0.18,
            },
            current_cash=0.25,
            total_equity=1.0,
        )
    )

    deltas = [
        abs(decision.symbol_target_weights.get(symbol, 0.0) - weight)
        for symbol, weight in {
            "000630.SZ": 0.20,
            "600160.SH": 0.19,
            "603619.SH": 0.18,
            "603516.SH": 0.18,
        }.items()
    ]
    assert any(delta < 0.02 for delta in deltas)
    assert any("below threshold" in note for note in decision.risk_notes)


def test_v2_policy_caps_single_stock_weight() -> None:
    market, sectors, _, cross_section = _make_demo_state()
    stocks = [
        StockForecastState("AAA", "有色", 0.70, 0.78, 0.84, 0.70, 0.05, 0.96),
        StockForecastState("BBB", "有色", 0.62, 0.68, 0.72, 0.58, 0.03, 0.90),
    ]
    composite_state = compose_state(
        market=market,
        sectors=sectors[:1],
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

    assert max(decision.symbol_target_weights.values()) <= 0.35 + 1e-9
