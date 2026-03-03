from __future__ import annotations

import pandas as pd
import pytest

from src.application.v2_contracts import DailyRunResult, StrategySnapshot
from src.application.v2_contracts import PolicyInput
from src.application.v2_services import (
    _stock_policy_score,
    apply_policy,
    build_strategy_snapshot,
    build_trade_actions,
    compose_state,
    run_daily_v2_live,
    summarize_daily_run,
)
from src.application.v2_contracts import (
    CrossSectionForecastState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
)
from src.domain.entities import BinaryMetrics, ForecastRow, MarketForecast, Security


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
    assert 1 <= decision.target_position_count <= 5
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


def test_stock_policy_score_improves_when_two_and_three_day_signal_improve() -> None:
    weak = StockForecastState(
        symbol="AAA",
        sector="科技",
        up_1d_prob=0.55,
        up_5d_prob=0.58,
        up_20d_prob=0.61,
        excess_vs_sector_prob=0.54,
        event_impact_score=0.08,
        tradeability_score=0.86,
        up_2d_prob=0.49,
        up_3d_prob=0.50,
    )
    strong = StockForecastState(
        symbol="AAA",
        sector="科技",
        up_1d_prob=0.55,
        up_5d_prob=0.58,
        up_20d_prob=0.61,
        excess_vs_sector_prob=0.54,
        event_impact_score=0.08,
        tradeability_score=0.86,
        up_2d_prob=0.62,
        up_3d_prob=0.66,
    )

    assert _stock_policy_score(strong) > _stock_policy_score(weak)


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


def test_v2_policy_caps_single_sector_concentration() -> None:
    market, _, _, cross_section = _make_demo_state()
    sectors = [
        SectorForecastState("单一", 0.62, 0.74, 0.26, 0.18, 0.12),
    ]
    stocks = [
        StockForecastState("A1", "单一", 0.58, 0.62, 0.70, 0.60, 0.30, 0.92),
        StockForecastState("A2", "单一", 0.55, 0.60, 0.67, 0.56, 0.24, 0.88),
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

    assert decision.sector_budgets["单一"] < decision.target_exposure
    assert any("sector budget capped" in note for note in decision.risk_notes)


def test_v2_policy_does_not_cap_unknown_fallback_sector_bucket() -> None:
    market, _, _, cross_section = _make_demo_state()
    sectors = [
        SectorForecastState("其他", 0.62, 0.74, 0.26, 0.18, 0.12),
    ]
    stocks = [
        StockForecastState("A1", "其他", 0.58, 0.62, 0.70, 0.60, 0.30, 0.92),
        StockForecastState("A2", "其他", 0.55, 0.60, 0.67, 0.56, 0.24, 0.88),
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

    assert decision.desired_sector_budgets["其他"] == pytest.approx(decision.target_exposure)
    assert decision.sector_budgets["其他"] <= decision.desired_sector_budgets["其他"]
    assert not any("sector budget capped" in note for note in decision.risk_notes)


def test_v2_policy_tightens_single_name_cap_under_high_volatility() -> None:
    _, sectors, _, cross_section = _make_demo_state()
    stressed_market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.56,
        up_5d_prob=0.58,
        up_20d_prob=0.61,
        trend_state="trend",
        drawdown_risk=0.18,
        volatility_regime="high",
        liquidity_stress=0.20,
    )
    stocks = [
        StockForecastState("A1", "有色", 0.60, 0.62, 0.70, 0.60, 0.24, 0.95),
        StockForecastState("A2", "化工", 0.58, 0.60, 0.67, 0.58, 0.20, 0.92),
        StockForecastState("A3", "科技", 0.56, 0.58, 0.64, 0.55, 0.18, 0.90),
    ]
    composite_state = compose_state(
        market=stressed_market,
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

    assert all(weight <= 0.24 + 1e-9 for weight in decision.symbol_target_weights.values())
    assert any("High volatility regime" in note for note in decision.risk_notes)


def test_stock_policy_score_penalizes_fragile_high_risk_setup() -> None:
    steady = StockForecastState(
        "AAA",
        "有色",
        0.58,
        0.60,
        0.63,
        0.56,
        0.62,
        0.92,
        alpha_score=0.78,
    )
    fragile = StockForecastState(
        "BBB",
        "有色",
        0.78,
        0.46,
        0.28,
        0.56,
        0.18,
        0.24,
        alpha_score=0.78,
    )

    assert _stock_policy_score(steady) > _stock_policy_score(fragile)


def test_v2_policy_trims_exposure_when_cross_sectional_alpha_is_weak() -> None:
    market, sectors, _, cross_section = _make_demo_state()
    weak_stocks = [
        StockForecastState("A1", "有色", 0.50, 0.51, 0.52, 0.49, 0.02, 0.78, alpha_score=0.53),
        StockForecastState("A2", "化工", 0.49, 0.50, 0.51, 0.48, 0.01, 0.76, alpha_score=0.52),
        StockForecastState("A3", "科技", 0.48, 0.49, 0.50, 0.47, 0.00, 0.74, alpha_score=0.51),
    ]
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=weak_stocks,
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

    assert decision.target_exposure < 0.85
    assert any("Cross-sectional alpha weak" in note for note in decision.risk_notes)


def test_v2_policy_boosts_exposure_and_sector_budget_for_strong_alpha() -> None:
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.48,
        up_5d_prob=0.49,
        up_20d_prob=0.48,
        trend_state="range",
        drawdown_risk=0.30,
        volatility_regime="normal",
        liquidity_stress=0.25,
    )
    sectors = [
        SectorForecastState("有色", 0.54, 0.57, 0.06, 0.28, 0.18),
        SectorForecastState("化工", 0.54, 0.57, 0.06, 0.28, 0.18),
    ]
    strong_stocks = [
        StockForecastState("A1", "有色", 0.62, 0.66, 0.70, 0.62, 0.72, 0.93, alpha_score=0.68),
        StockForecastState("A2", "有色", 0.60, 0.64, 0.67, 0.60, 0.68, 0.90, alpha_score=0.64),
        StockForecastState("B1", "化工", 0.54, 0.56, 0.58, 0.51, 0.45, 0.84, alpha_score=0.56),
        StockForecastState("B2", "化工", 0.53, 0.55, 0.57, 0.50, 0.42, 0.82, alpha_score=0.55),
    ]
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.02,
        growth_vs_value_bias=0.01,
        fund_flow_strength=0.10,
        margin_risk_on_score=0.08,
        breadth_strength=0.18,
        leader_participation=0.61,
        weak_stock_ratio=0.28,
    )
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=strong_stocks,
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

    assert decision.target_exposure > 0.45
    assert decision.target_position_count >= 4
    assert decision.desired_sector_budgets["有色"] > decision.desired_sector_budgets["化工"]
    assert any("Cross-sectional alpha strong" in note for note in decision.risk_notes)


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
        symbol_names={"000630.SZ": "铜陵有色"},
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
    assert "铜陵有色" in summary["policy"]["symbol_target_weights"]
    assert all("symbol" not in item for item in summary["trade_plan"])


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

    assert decision.desired_symbol_target_weights.get("AAA", 0.0) > 0.0
    assert "AAA" not in decision.symbol_target_weights
    assert "AAA" not in decision.sector_budgets
    assert any("new entry blocked" in note for note in decision.risk_notes)
    assert any("new entry blocked" in note for note in decision.execution_notes)


def test_v2_policy_blocks_sell_within_minimum_holding_window() -> None:
    market, _, _, cross_section = _make_demo_state()
    sectors = [
        SectorForecastState("强", 0.60, 0.66, 0.18, 0.22, 0.18),
        SectorForecastState("弱", 0.45, 0.47, -0.06, 0.40, 0.16),
    ]
    stocks = [
        StockForecastState("AAA", "弱", 0.48, 0.49, 0.50, 0.46, 0.20, 0.84, alpha_score=0.51),
        StockForecastState("BBB", "强", 0.62, 0.66, 0.70, 0.60, 0.72, 0.92, alpha_score=0.68),
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
            current_holding_days={"AAA": 3},
        )
    )

    assert decision.symbol_target_weights["AAA"] == pytest.approx(0.18)
    assert any("minimum holding window active" in note for note in decision.risk_notes)


def test_v2_policy_suppresses_small_rebalance_gap() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    baseline_decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        )
    )
    nearly_aligned_weights = {
        symbol: max(0.0, weight - 0.005)
        for symbol, weight in baseline_decision.symbol_target_weights.items()
    }

    decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights=nearly_aligned_weights,
            current_cash=max(0.0, 1.0 - sum(nearly_aligned_weights.values())),
            total_equity=1.0,
        )
    )

    deltas = [
        abs(decision.symbol_target_weights.get(symbol, 0.0) - weight)
        for symbol, weight in nearly_aligned_weights.items()
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


def test_run_daily_v2_live_reuses_cache_without_retraining(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
    watchlist_path = tmp_path / "watchlist.json"
    universe_path = tmp_path / "universe.json"
    margin_market_path = tmp_path / "margin_market.csv"
    margin_stock_path = tmp_path / "margin_stock.csv"
    watchlist_path.write_text("[]", encoding="utf-8")
    universe_path.write_text("[]", encoding="utf-8")
    margin_market_path.write_text("", encoding="utf-8")
    margin_stock_path.write_text("", encoding="utf-8")
    settings = {
        "config_path": "config/api.json",
        "watchlist": str(watchlist_path),
        "source": "local",
        "data_dir": "data",
        "start": "2024-01-01",
        "end": "2024-12-31",
        "min_train_days": 240,
        "step_days": 20,
        "l2": 0.8,
        "max_positions": 5,
        "use_margin_features": False,
        "margin_market_file": str(margin_market_path),
        "margin_stock_file": str(margin_stock_path),
        "universe_file": str(universe_path),
        "universe_limit": 5,
    }
    calls = {"quant": 0}
    progress: list[tuple[str, str]] = []

    def fake_quant(**_: object) -> tuple[MarketForecast, list[ForecastRow]]:
        progress_callback = _.get("progress_callback")
        assert callable(progress_callback)
        calls["quant"] += 1
        progress_callback("量化预测测试进度")
        return (
            MarketForecast(
                symbol="000001.SH",
                name="指数",
                latest_date=pd.Timestamp("2026-02-26"),
                short_prob=0.55,
                five_prob=0.56,
                mid_prob=0.60,
                short_eval=BinaryMetrics.empty(),
                mid_eval=BinaryMetrics.empty(),
            ),
            [
                ForecastRow(
                    symbol="000630.SZ",
                    name="样例股",
                    latest_date=pd.Timestamp("2026-02-26"),
                    short_prob=0.58,
                    five_prob=0.60,
                    mid_prob=0.64,
                    score=0.70,
                    short_drivers=[],
                    mid_drivers=[],
                    short_eval=BinaryMetrics.empty(),
                    mid_eval=BinaryMetrics.empty(),
                )
            ],
        )

    monkeypatch.setattr("src.application.v2_services._emit_progress", lambda stage, message: progress.append((stage, message)))
    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: settings)
    monkeypatch.setattr(
        "src.application.v2_services.load_watchlist",
        lambda *_: (Security("000001.SH", "指数"), [], {}),
    )
    monkeypatch.setattr(
        "src.application.v2_services.build_candidate_universe",
        lambda **_: type("Universe", (), {"rows": [Security("000630.SZ", "样例股", "有色")]})(),
    )
    monkeypatch.setattr("src.application.v2_services.run_quant_pipeline", fake_quant)
    monkeypatch.setattr(
        "src.application.v2_services.load_symbol_daily",
        lambda **_: pd.DataFrame({"date": pd.to_datetime(["2026-02-26"]), "close": [1.0]}),
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_market_and_cross_section_states",
        lambda **_: (market, cross_section),
    )
    monkeypatch.setattr("src.application.v2_services.build_sector_daily_frames", lambda **_: {})
    monkeypatch.setattr("src.application.v2_services.run_sector_forecast", lambda **_: [])
    monkeypatch.setattr("src.application.v2_services._build_stock_states_from_rows", lambda *_, **__: stocks)
    monkeypatch.setattr("src.application.v2_services.compose_state", lambda **_: composite_state)
    monkeypatch.setattr("src.application.v2_services.load_published_v2_policy_model", lambda **_: None)
    monkeypatch.setattr("src.application.v2_services.apply_policy", lambda *_, **__: decision)

    first = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache"),
        refresh_cache=True,
    )
    second = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache"),
        refresh_cache=False,
    )

    assert calls["quant"] == 1
    assert first.policy_decision.target_exposure == second.policy_decision.target_exposure
    assert len(second.trade_actions) == len(first.trade_actions)
    assert any(stage == "daily" and "量化预测测试进度" in message for stage, message in progress)
    assert any(stage == "daily" and "命中日运行缓存" in message for stage, message in progress)
