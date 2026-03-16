from __future__ import annotations

import json
import pandas as pd
import pytest

import src.application.v2_services as legacy_services
from src.application import v2_policy_feature_runtime as policy_feature_runtime
from src.application import v2_policy_runtime as policy_runtime
from src.application.v2_leader_runtime import LeaderScoreSnapshot
from src.application.v2_contracts import DailyRunResult, StrategySnapshot
from src.application.v2_contracts import PolicyInput
from src.application.v2_sector_support import build_sector_states
from src.application.v2_candidate_selection import build_candidate_selection_state
from src.application.v2_candidate_selection import candidate_stocks_from_state
from src.application.v2_services import (
    _decode_composite_state,
    _serialize_composite_state,
    apply_policy,
    build_strategy_snapshot,
    build_trade_actions,
    compose_state,
    run_daily_v2_live,
    summarize_daily_run,
)
from src.application.v2_contracts import (
    CapitalFlowState,
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    InfoAggregateState,
    MainlineState,
    MarketForecastState,
    MacroContextState,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    StockRoleSnapshot,
    ThemeEpisode,
)
from src.domain.entities import BinaryMetrics, ForecastRow, MarketForecast, Security


def _policy_score(stock: StockForecastState) -> float:
    return policy_runtime.stock_policy_score(
        stock,
        deps=legacy_services._policy_runtime_dependencies(),
    )


def _allocate_sector_budget_weights(**kwargs: object) -> dict[str, float]:
    return policy_runtime.allocate_with_sector_budgets(
        deps=legacy_services._policy_runtime_dependencies(),
        **kwargs,
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


def _make_leader_policy_state(
    *,
    stocks: list[StockForecastState],
    theme_episodes: list[ThemeEpisode] | None = None,
    stock_role_states: dict[str, StockRoleSnapshot] | None = None,
) -> CompositeState:
    return CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-12",
            up_1d_prob=0.58,
            up_5d_prob=0.61,
            up_20d_prob=0.64,
            trend_state="trend",
            drawdown_risk=0.18,
            volatility_regime="normal",
            liquidity_stress=0.16,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-12",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.04,
            fund_flow_strength=0.12,
            margin_risk_on_score=0.10,
            breadth_strength=0.20,
            leader_participation=0.64,
            weak_stock_ratio=0.18,
        ),
        sectors=[SectorForecastState("chips", 0.61, 0.67, 0.18, 0.20, 0.18)],
        stocks=stocks,
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=[stock.symbol for stock in stocks],
            shortlisted_sectors=["chips"],
            sector_slots={"chips": 1},
            total_scored=len(stocks),
            shortlist_size=len(stocks),
            shortlist_ratio=1.0,
            selection_mode="macro_sector_ranking",
            selection_notes=["test shortlist"],
        ),
        theme_episodes=list(theme_episodes or []),
        stock_role_states=dict(stock_role_states or {}),
    )


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


def test_v2_policy_trims_for_event_and_macro_risk() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    base_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    stressed_state = base_state.__class__(
        market=base_state.market,
        cross_section=base_state.cross_section,
        sectors=base_state.sectors,
        stocks=base_state.stocks,
        strategy_mode=base_state.strategy_mode,
        risk_regime=base_state.risk_regime,
        market_info_state=InfoAggregateState(event_risk_level=0.82, negative_event_risk=0.70),
        sector_info_states=base_state.sector_info_states,
        stock_info_states={
            "000630.SZ": InfoAggregateState(event_risk_level=0.78, catalyst_strength=0.10),
            "600160.SH": InfoAggregateState(catalyst_strength=0.42, coverage_confidence=0.80),
        },
        capital_flow_state=CapitalFlowState(
            northbound_net_flow=-0.30,
            margin_balance_change=-0.18,
            turnover_heat=0.36,
            large_order_bias=-0.24,
            flow_regime="strong_outflow",
        ),
        macro_context_state=MacroContextState(
            style_regime="defensive",
            commodity_pressure=0.72,
            fx_pressure=0.68,
            index_breadth_proxy=0.28,
            macro_risk_level="high",
        ),
    )

    decision = apply_policy(
        PolicyInput(
            composite_state=stressed_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        )
    )

    assert decision.target_exposure < 0.60
    assert any("Event risk elevated" in note for note in decision.risk_notes)
    assert any("Macro risk high" in note for note in decision.risk_notes)
    assert any("strong_outflow" in note for note in decision.risk_notes)


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


def test_compose_state_applies_leader_overlay_in_shared_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def _fake_overlay(*, state: object) -> object:
        calls.append(list(getattr(getattr(state, "candidate_selection", None), "shortlisted_symbols", [])))
        selection = getattr(state, "candidate_selection")
        return state.__class__(
            **{
                **state.__dict__,
                "candidate_selection": selection.__class__(
                    **{
                        **selection.__dict__,
                        "selection_notes": list(getattr(selection, "selection_notes", []) or []) + ["overlay called"],
                    }
                ),
            }
        )

    monkeypatch.setattr("src.application.v2_services._apply_leader_candidate_overlay", _fake_overlay)
    market, sectors, stocks, cross_section = _make_demo_state()

    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    assert len(calls) == 1
    assert any(note == "overlay called" for note in composite_state.candidate_selection.selection_notes)


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

    assert _policy_score(strong) > _policy_score(weak)


def test_stock_policy_score_rewards_five_day_continuation_and_sector_edge() -> None:
    weak = StockForecastState(
        symbol="AAA",
        sector="科技",
        up_1d_prob=0.54,
        up_5d_prob=0.56,
        up_20d_prob=0.61,
        excess_vs_sector_prob=0.53,
        event_impact_score=0.08,
        tradeability_score=0.87,
        up_2d_prob=0.55,
        up_3d_prob=0.57,
    )
    strong = StockForecastState(
        symbol="AAA",
        sector="科技",
        up_1d_prob=0.54,
        up_5d_prob=0.67,
        up_20d_prob=0.63,
        excess_vs_sector_prob=0.61,
        event_impact_score=0.08,
        tradeability_score=0.87,
        up_2d_prob=0.57,
        up_3d_prob=0.63,
    )

    assert _policy_score(strong) > _policy_score(weak)


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

    assert max(decision.symbol_target_weights.values()) < decision.target_exposure
    assert len(decision.symbol_target_weights) >= 2


def test_build_sector_states_rewards_broad_sector_leadership() -> None:
    stocks = [
        StockForecastState("S1", "强", 0.58, 0.61, 0.67, 0.58, 0.16, 0.90, alpha_score=0.66),
        StockForecastState("S2", "强", 0.57, 0.60, 0.65, 0.57, 0.12, 0.88, alpha_score=0.64),
        StockForecastState("S3", "强", 0.56, 0.59, 0.64, 0.55, 0.10, 0.87, alpha_score=0.62),
        StockForecastState("W1", "弱", 0.59, 0.62, 0.68, 0.60, 0.14, 0.89, alpha_score=0.65),
        StockForecastState("W2", "弱", 0.49, 0.50, 0.51, 0.47, 0.02, 0.78, alpha_score=0.52),
    ]

    sector_states = build_sector_states(stocks, stock_score_fn=lambda stock: float(stock.alpha_score))
    sector_map = {item.sector: item for item in sector_states}

    assert sector_map["强"].relative_strength > sector_map["弱"].relative_strength
    assert sector_map["强"].rotation_speed >= sector_map["弱"].rotation_speed


def test_allocate_with_sector_budgets_filters_marginal_weak_sector_names() -> None:
    stocks = [
        StockForecastState("S1", "强", 0.58, 0.61, 0.67, 0.58, 0.12, 0.92, alpha_score=0.66),
        StockForecastState("S2", "强", 0.56, 0.59, 0.64, 0.55, 0.10, 0.89, alpha_score=0.62),
        StockForecastState("W1", "弱", 0.55, 0.57, 0.60, 0.52, 0.08, 0.84, alpha_score=0.55),
    ]

    weights = _allocate_sector_budget_weights(
        stocks=stocks,
        sector_budgets={"强": 0.24, "弱": 0.11},
        target_position_count=2,
        sector_strengths={"强": 0.70, "弱": 0.30},
        max_single_position=0.18,
    )

    assert "W1" not in weights


def test_allocate_with_sector_budgets_prefers_mainline_representative_symbol() -> None:
    stocks = [
        StockForecastState("L1", "光模块", 0.56, 0.61, 0.67, 0.58, 0.10, 0.90, alpha_score=0.61),
        StockForecastState("L2", "光模块", 0.57, 0.62, 0.68, 0.59, 0.12, 0.91, alpha_score=0.64),
    ]

    weights = _allocate_sector_budget_weights(
        stocks=stocks,
        sector_budgets={"光模块": 0.22},
        target_position_count=1,
        sector_strengths={"光模块": 0.75},
        max_single_position=0.22,
        symbol_score_adjustments={"L1": 0.08},
    )

    assert set(weights) == {"L1"}


def test_compose_state_builds_macro_sector_shortlist_for_large_universe() -> None:
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.56,
        up_5d_prob=0.60,
        up_20d_prob=0.63,
        trend_state="trend",
        drawdown_risk=0.24,
        volatility_regime="normal",
        liquidity_stress=0.18,
    )
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.05,
        growth_vs_value_bias=0.03,
        fund_flow_strength=0.12,
        margin_risk_on_score=0.10,
        breadth_strength=0.18,
        leader_participation=0.62,
        weak_stock_ratio=0.22,
    )
    sectors = [
        SectorForecastState("Alpha", 0.62, 0.68, 0.24, 0.22, 0.18),
        SectorForecastState("Beta", 0.58, 0.64, 0.16, 0.20, 0.22),
        SectorForecastState("Gamma", 0.49, 0.50, -0.04, 0.12, 0.54),
        SectorForecastState("Delta", 0.47, 0.48, -0.08, 0.10, 0.58),
    ]
    stocks: list[StockForecastState] = []
    for idx in range(1, 9):
        stocks.append(StockForecastState(f"A{idx}", "Alpha", 0.56, 0.61, 0.68, 0.60, 0.18, 0.90, alpha_score=0.66))
        stocks.append(StockForecastState(f"B{idx}", "Beta", 0.54, 0.58, 0.63, 0.57, 0.10, 0.86, alpha_score=0.62))
        stocks.append(StockForecastState(f"G{idx}", "Gamma", 0.49, 0.51, 0.50, 0.48, 0.03, 0.78, alpha_score=0.51))
        stocks.append(StockForecastState(f"D{idx}", "Delta", 0.47, 0.49, 0.48, 0.46, 0.02, 0.76, alpha_score=0.49))

    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    assert composite_state.candidate_selection.selection_mode == "macro_sector_ranking"
    assert composite_state.candidate_selection.shortlist_size < len(composite_state.stocks)
    assert len(composite_state.candidate_selection.shortlisted_symbols) == len(composite_state.stocks)
    assert set(composite_state.candidate_selection.shortlisted_sectors).issubset({"Alpha", "Beta"})
    assert all(
        symbol.startswith(("A", "B"))
        for symbol in composite_state.candidate_selection.shortlisted_symbols[
            : composite_state.candidate_selection.shortlist_size
        ]
    )


def test_candidate_shortlist_blends_sector_signal_with_stock_support() -> None:
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.50,
        up_5d_prob=0.52,
        up_20d_prob=0.54,
        trend_state="down",
        drawdown_risk=0.42,
        volatility_regime="normal",
        liquidity_stress=0.26,
    )
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.01,
        growth_vs_value_bias=-0.01,
        fund_flow_strength=0.03,
        margin_risk_on_score=0.04,
        breadth_strength=0.08,
        leader_participation=0.55,
        weak_stock_ratio=0.46,
    )
    sectors = [
        SectorForecastState("Main1", 0.58, 0.64, 0.16, 0.14, 0.18),
        SectorForecastState("Main2", 0.57, 0.63, 0.14, 0.12, 0.18),
        SectorForecastState("Challenger", 0.50, 0.54, 0.02, 0.08, 0.12),
    ]
    stocks: list[StockForecastState] = []
    for idx in range(1, 7):
        stocks.append(StockForecastState(f"M1_{idx}", "Main1", 0.53, 0.56, 0.59, 0.54, 0.05, 0.88, alpha_score=0.57))
        stocks.append(StockForecastState(f"M2_{idx}", "Main2", 0.52, 0.55, 0.58, 0.53, 0.05, 0.87, alpha_score=0.56))
        stocks.append(
            StockForecastState(
                f"C_{idx}",
                "Challenger",
                0.56,
                0.61,
                0.66,
                0.60,
                0.10,
                0.91,
                alpha_score=0.66 - 0.01 * idx,
            )
        )

    selection = build_candidate_selection_state(
        market=market,
        cross_section=cross_section,
        sectors=sectors,
        stocks=stocks,
        mainlines=[
            MainlineState(name="Main1", conviction=0.45, sectors=["Main1"], representative_symbols=["M1_1"]),
            MainlineState(name="Main2", conviction=0.42, sectors=["Main2"], representative_symbols=["M2_1"]),
        ],
        strategy_mode="defensive",
        risk_regime="risk_off",
        stock_score_fn=lambda stock: float(stock.alpha_score),
    )

    assert "Challenger" in selection.shortlisted_sectors
    assert any(symbol.startswith("C_") for symbol in selection.shortlisted_symbols[: selection.shortlist_size])


def test_policy_feature_vector_captures_shortlist_shape() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks * 2,
        cross_section=cross_section,
    )
    narrowed_state = state.__class__(
        market=state.market,
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=state.stocks,
        strategy_mode=state.strategy_mode,
        risk_regime=state.risk_regime,
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["000630.SZ", "600160.SH"],
            shortlisted_sectors=["MainA", "MainB"],
            sector_slots={"MainA": 1, "MainB": 1},
            total_scored=len(state.stocks),
            shortlist_size=2,
            shortlist_ratio=0.25,
            selection_mode="macro_sector_shortlist",
            selection_notes=["test shortlist"],
        ),
        mainlines=state.mainlines,
        market_info_state=state.market_info_state,
        sector_info_states=state.sector_info_states,
        stock_info_states=state.stock_info_states,
        capital_flow_state=state.capital_flow_state,
        macro_context_state=state.macro_context_state,
    )

    features = policy_feature_runtime.policy_feature_vector(
        narrowed_state,
        deps=legacy_services._policy_feature_runtime_dependencies(),
    )

    assert len(features) == 23
    assert features[19] == pytest.approx(0.25)
    assert 0.0 < features[20] <= 1.0


def test_apply_policy_prefers_candidate_shortlist_pool() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    constrained_state = composite_state.__class__(
        market=composite_state.market,
        cross_section=composite_state.cross_section,
        sectors=composite_state.sectors,
        stocks=composite_state.stocks,
        strategy_mode=composite_state.strategy_mode,
        risk_regime=composite_state.risk_regime,
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["000630.SZ"],
            shortlisted_sectors=["鏈夎壊"],
            sector_slots={"鏈夎壊": 1},
            total_scored=len(composite_state.stocks),
            shortlist_size=1,
            shortlist_ratio=1.0 / len(composite_state.stocks),
            selection_mode="macro_sector_shortlist",
            selection_notes=["Macro shortlist active for test."],
        ),
        market_info_state=composite_state.market_info_state,
        sector_info_states=composite_state.sector_info_states,
        stock_info_states=composite_state.stock_info_states,
        capital_flow_state=composite_state.capital_flow_state,
        macro_context_state=composite_state.macro_context_state,
    )

    decision = apply_policy(
        PolicyInput(
            composite_state=constrained_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        )
    )

    assert set(decision.desired_symbol_target_weights).issubset({"000630.SZ"})
    assert any("Candidate shortlist active" in note for note in decision.risk_notes)


def test_apply_policy_leader_weighting_can_promote_theme_leader_into_buy_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_alpha_components(stock: StockForecastState) -> dict[str, float]:
        base_alpha = {
            "AAA": 0.76,
            "BBB": 0.72,
        }[stock.symbol]
        return {
            "alpha_score": base_alpha,
            "swing_edge": 0.0,
            "medium_edge": 0.0,
            "sector_edge": 0.0,
            "trend_alignment": 0.0,
            "risk_penalty": 0.0,
            "status_penalty": 0.0,
            "continuation_bonus": 0.0,
            "stability_bonus": 0.0,
            "quality_bonus": 0.0,
            "swing_fade_penalty": 0.0,
            "reversal_penalty": 0.0,
            "weak_mid_penalty": 0.0,
        }

    monkeypatch.setattr("src.application.v2_services._alpha_score_components", _fake_alpha_components)
    monkeypatch.setattr(
        policy_runtime,
        "build_leader_score_snapshots",
        lambda *, state: []
        if not getattr(state, "theme_episodes", [])
        else [
            LeaderScoreSnapshot(
                symbol="AAA",
                sector="chips",
                theme="chips",
                theme_phase="strengthening",
                role="follower",
                negative_score=0.18,
                candidate_score=0.58,
                conviction_score=0.57,
            ),
            LeaderScoreSnapshot(
                symbol="BBB",
                sector="chips",
                theme="chips",
                theme_phase="strengthening",
                role="leader",
                negative_score=0.08,
                candidate_score=0.78,
                conviction_score=0.82,
            ),
        ],
    )

    stocks = [
        StockForecastState("AAA", "chips", 0.54, 0.56, 0.58, 0.52, 0.08, 0.86, alpha_score=0.76),
        StockForecastState("BBB", "chips", 0.61, 0.68, 0.72, 0.60, 0.13, 0.92, alpha_score=0.72),
    ]
    baseline_state = _make_leader_policy_state(stocks=stocks)
    leader_state = _make_leader_policy_state(
        stocks=stocks,
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="strengthening",
                conviction=0.72,
                breadth=0.36,
                leadership=0.34,
                event_risk=0.18,
                sectors=["chips"],
                representative_symbols=["BBB"],
            )
        ],
        stock_role_states={
            "AAA": StockRoleSnapshot(symbol="AAA", theme="chips", role="follower"),
            "BBB": StockRoleSnapshot(symbol="BBB", theme="chips", role="leader"),
        },
    )
    spec = PolicySpec(risk_on_positions=1, cautious_positions=1, risk_off_positions=1)

    baseline = apply_policy(
        PolicyInput(
            composite_state=baseline_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        ),
        policy_spec=spec,
    )
    leader = apply_policy(
        PolicyInput(
            composite_state=leader_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        ),
        policy_spec=spec,
    )

    assert max(baseline.desired_symbol_target_weights, key=baseline.desired_symbol_target_weights.get) == "AAA"
    assert max(leader.desired_symbol_target_weights, key=leader.desired_symbol_target_weights.get) == "BBB"
    assert any("Leader weighting active" in note for note in leader.risk_notes)


def test_apply_policy_leader_weighting_suppresses_hard_negative_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_alpha_components(stock: StockForecastState) -> dict[str, float]:
        base_alpha = {
            "AAA": 0.80,
            "BBB": 0.72,
        }[stock.symbol]
        return {
            "alpha_score": base_alpha,
            "swing_edge": 0.0,
            "medium_edge": 0.0,
            "sector_edge": 0.0,
            "trend_alignment": 0.0,
            "risk_penalty": 0.0,
            "status_penalty": 0.0,
            "continuation_bonus": 0.0,
            "stability_bonus": 0.0,
            "quality_bonus": 0.0,
            "swing_fade_penalty": 0.0,
            "reversal_penalty": 0.0,
            "weak_mid_penalty": 0.0,
        }

    monkeypatch.setattr("src.application.v2_services._alpha_score_components", _fake_alpha_components)
    monkeypatch.setattr(
        policy_runtime,
        "build_leader_score_snapshots",
        lambda *, state: []
        if not getattr(state, "theme_episodes", [])
        else [
            LeaderScoreSnapshot(
                symbol="AAA",
                sector="chips",
                theme="chips",
                theme_phase="fading",
                role="laggard",
                role_downgrade=True,
                negative_score=0.74,
                candidate_score=0.38,
                conviction_score=0.30,
                hard_negative=True,
            ),
            LeaderScoreSnapshot(
                symbol="BBB",
                sector="chips",
                theme="chips",
                theme_phase="strengthening",
                role="core",
                negative_score=0.10,
                candidate_score=0.70,
                conviction_score=0.73,
            ),
        ],
    )

    stocks = [
        StockForecastState("AAA", "chips", 0.49, 0.45, 0.47, 0.43, 0.03, 0.72, alpha_score=0.80),
        StockForecastState("BBB", "chips", 0.58, 0.64, 0.69, 0.58, 0.11, 0.90, alpha_score=0.72),
    ]
    baseline_state = _make_leader_policy_state(stocks=stocks)
    leader_state = _make_leader_policy_state(
        stocks=stocks,
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="fading",
                conviction=0.44,
                breadth=0.18,
                leadership=0.16,
                event_risk=0.62,
                sectors=["chips"],
                representative_symbols=["BBB"],
            )
        ],
        stock_role_states={
            "AAA": StockRoleSnapshot(
                symbol="AAA",
                theme="chips",
                role="laggard",
                previous_role="core",
                role_downgrade=True,
            ),
            "BBB": StockRoleSnapshot(symbol="BBB", theme="chips", role="core"),
        },
    )
    spec = PolicySpec(risk_on_positions=1, cautious_positions=1, risk_off_positions=1)

    baseline = apply_policy(
        PolicyInput(
            composite_state=baseline_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        ),
        policy_spec=spec,
    )
    leader = apply_policy(
        PolicyInput(
            composite_state=leader_state,
            current_weights={},
            current_cash=1.0,
            total_equity=1.0,
        ),
        policy_spec=spec,
    )

    assert max(baseline.desired_symbol_target_weights, key=baseline.desired_symbol_target_weights.get) == "AAA"
    assert max(leader.desired_symbol_target_weights, key=leader.desired_symbol_target_weights.get) == "BBB"
    assert leader.desired_symbol_target_weights.get("AAA", 0.0) == pytest.approx(0.0)
    assert any("Leader-suppressed symbols: AAA" in note for note in leader.risk_notes)


def test_candidate_stocks_from_state_only_uses_shortlist_core() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    narrowed_state = composite_state.__class__(
        market=composite_state.market,
        cross_section=composite_state.cross_section,
        sectors=composite_state.sectors,
        stocks=composite_state.stocks,
        strategy_mode=composite_state.strategy_mode,
        risk_regime=composite_state.risk_regime,
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["600160.SH", "000630.SZ", "603619.SH", "603516.SH"],
            shortlisted_sectors=["鍖栧伐", "鏈夎壊"],
            sector_slots={"鍖栧伐": 1, "鏈夎壊": 1},
            total_scored=len(composite_state.stocks),
            shortlist_size=2,
            shortlist_ratio=0.5,
            selection_mode="macro_sector_shortlist",
            selection_notes=["Shortlist core should constrain downstream policy inputs."],
        ),
        mainlines=composite_state.mainlines,
        market_info_state=composite_state.market_info_state,
        sector_info_states=composite_state.sector_info_states,
        stock_info_states=composite_state.stock_info_states,
        capital_flow_state=composite_state.capital_flow_state,
        macro_context_state=composite_state.macro_context_state,
    )

    candidate_symbols = [stock.symbol for stock in candidate_stocks_from_state(narrowed_state)]

    assert candidate_symbols == ["600160.SH", "000630.SZ"]


def test_compose_state_builds_explicit_mainline_layer() -> None:
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.57,
        up_5d_prob=0.61,
        up_20d_prob=0.65,
        trend_state="trend",
        drawdown_risk=0.18,
        volatility_regime="normal",
        liquidity_stress=0.16,
    )
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.04,
        growth_vs_value_bias=0.02,
        fund_flow_strength=0.12,
        margin_risk_on_score=0.11,
        breadth_strength=0.18,
        leader_participation=0.64,
        weak_stock_ratio=0.22,
    )
    sectors = [
        SectorForecastState("有色资源", 0.62, 0.69, 0.24, 0.18, 0.18),
        SectorForecastState("黄金", 0.60, 0.67, 0.20, 0.16, 0.20),
        SectorForecastState("科技", 0.52, 0.56, 0.06, 0.24, 0.32),
    ]
    stocks = [
        StockForecastState("R1", "有色资源", 0.56, 0.61, 0.69, 0.60, 0.18, 0.90, alpha_score=0.67),
        StockForecastState("R2", "黄金", 0.55, 0.60, 0.67, 0.58, 0.16, 0.88, alpha_score=0.64),
        StockForecastState("T1", "科技", 0.53, 0.56, 0.58, 0.54, 0.10, 0.86, alpha_score=0.58),
    ]

    state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    assert state.mainlines
    assert state.mainlines[0].name == "资源"
    assert "有色资源" in state.mainlines[0].sectors
    assert state.mainlines[0].representative_symbols


def test_compose_state_preserves_fine_grained_mainline_labels() -> None:
    market, _, _, cross_section = _make_demo_state()
    sectors = [
        SectorForecastState("能源石油", 0.61, 0.66, 0.16, 0.14, 0.12),
        SectorForecastState("光模块", 0.62, 0.68, 0.20, 0.12, 0.18),
        SectorForecastState("煤化工", 0.58, 0.63, 0.14, 0.10, 0.16),
    ]
    stocks = [
        StockForecastState("O1", "能源石油", 0.55, 0.60, 0.66, 0.57, 0.14, 0.89, alpha_score=0.63),
        StockForecastState("L1", "光模块", 0.56, 0.62, 0.69, 0.59, 0.18, 0.90, alpha_score=0.67),
        StockForecastState("C1", "煤化工", 0.54, 0.59, 0.65, 0.56, 0.13, 0.88, alpha_score=0.61),
    ]

    state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    mainline_names = {item.name for item in state.mainlines}
    assert "能源石油" in mainline_names
    assert "光模块" in mainline_names
    assert "煤化工" in mainline_names


def test_apply_policy_respects_mainline_risk_and_conviction() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    bullish_mainline = MainlineState(
        name="科技",
        driver="catalyst_confirmed",
        conviction=0.66,
        breadth=0.40,
        leadership=0.30,
        catalyst_strength=0.36,
        event_risk_level=0.18,
        sectors=["化工"],
        representative_symbols=["600160.SH"],
    )
    risky_mainline = MainlineState(
        name="资源",
        driver="risk_watched",
        conviction=0.64,
        breadth=0.32,
        leadership=0.28,
        catalyst_strength=0.12,
        event_risk_level=0.68,
        sectors=["有色"],
        representative_symbols=["000630.SZ"],
    )
    support_state = composite_state.__class__(
        market=composite_state.market,
        cross_section=composite_state.cross_section,
        sectors=composite_state.sectors,
        stocks=composite_state.stocks,
        strategy_mode=composite_state.strategy_mode,
        risk_regime=composite_state.risk_regime,
        candidate_selection=composite_state.candidate_selection,
        mainlines=[bullish_mainline],
        market_info_state=composite_state.market_info_state,
        sector_info_states=composite_state.sector_info_states,
        stock_info_states=composite_state.stock_info_states,
        capital_flow_state=composite_state.capital_flow_state,
        macro_context_state=composite_state.macro_context_state,
    )
    risk_state = composite_state.__class__(
        market=composite_state.market,
        cross_section=composite_state.cross_section,
        sectors=composite_state.sectors,
        stocks=composite_state.stocks,
        strategy_mode=composite_state.strategy_mode,
        risk_regime=composite_state.risk_regime,
        candidate_selection=composite_state.candidate_selection,
        mainlines=[risky_mainline],
        market_info_state=composite_state.market_info_state,
        sector_info_states=composite_state.sector_info_states,
        stock_info_states=composite_state.stock_info_states,
        capital_flow_state=composite_state.capital_flow_state,
        macro_context_state=composite_state.macro_context_state,
    )

    support_decision = apply_policy(
        PolicyInput(composite_state=support_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )
    risk_decision = apply_policy(
        PolicyInput(composite_state=risk_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )

    assert support_decision.target_position_count > risk_decision.target_position_count
    assert any("confirmed" in note for note in support_decision.risk_notes)
    assert any("risk-watched" in note for note in risk_decision.risk_notes)


def test_apply_policy_allocates_budget_toward_confirmed_mainline() -> None:
    market, _, _, cross_section = _make_demo_state()
    sectors = [
        SectorForecastState("光模块", 0.60, 0.67, 0.18, 0.18, 0.18),
        SectorForecastState("建筑工程", 0.59, 0.66, 0.17, 0.16, 0.20),
    ]
    stocks = [
        StockForecastState("L1", "光模块", 0.56, 0.61, 0.69, 0.60, 0.15, 0.90, alpha_score=0.63),
        StockForecastState("L2", "光模块", 0.55, 0.60, 0.67, 0.58, 0.12, 0.88, alpha_score=0.61),
        StockForecastState("B1", "建筑工程", 0.56, 0.61, 0.68, 0.59, 0.12, 0.89, alpha_score=0.64),
        StockForecastState("B2", "建筑工程", 0.55, 0.60, 0.66, 0.57, 0.10, 0.87, alpha_score=0.60),
    ]
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    base_state = composite_state.__class__(
        market=composite_state.market,
        cross_section=composite_state.cross_section,
        sectors=composite_state.sectors,
        stocks=composite_state.stocks,
        strategy_mode=composite_state.strategy_mode,
        risk_regime=composite_state.risk_regime,
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["L1", "L2", "B1", "B2"],
            shortlisted_sectors=["光模块", "建筑工程"],
            sector_slots={"光模块": 2, "建筑工程": 2},
            total_scored=4,
            shortlist_size=4,
            shortlist_ratio=1.0,
            selection_mode="macro_sector_shortlist",
            selection_notes=["macro shortlist test"],
        ),
        mainlines=[],
        market_info_state=composite_state.market_info_state,
        sector_info_states=composite_state.sector_info_states,
        stock_info_states=composite_state.stock_info_states,
        capital_flow_state=composite_state.capital_flow_state,
        macro_context_state=composite_state.macro_context_state,
    )
    focused_state = base_state.__class__(
        market=base_state.market,
        cross_section=base_state.cross_section,
        sectors=base_state.sectors,
        stocks=base_state.stocks,
        strategy_mode=base_state.strategy_mode,
        risk_regime=base_state.risk_regime,
        candidate_selection=base_state.candidate_selection,
        mainlines=[
            MainlineState(
                name="光模块",
                driver="catalyst_confirmed",
                conviction=0.74,
                breadth=0.44,
                leadership=0.34,
                catalyst_strength=0.32,
                event_risk_level=0.10,
                sectors=["光模块"],
                representative_symbols=["L1"],
            )
        ],
        market_info_state=base_state.market_info_state,
        sector_info_states=base_state.sector_info_states,
        stock_info_states=base_state.stock_info_states,
        capital_flow_state=base_state.capital_flow_state,
        macro_context_state=base_state.macro_context_state,
    )

    base_decision = apply_policy(
        PolicyInput(composite_state=base_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )
    focused_decision = apply_policy(
        PolicyInput(composite_state=focused_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )

    assert focused_decision.desired_sector_budgets["光模块"] > base_decision.desired_sector_budgets["光模块"]
    assert "L1" in focused_decision.desired_symbol_target_weights
    assert any("Mainline budgets prioritized" in note for note in focused_decision.risk_notes)


def test_high_volatility_shortlist_is_tighter_and_filters_fragile_names() -> None:
    sectors = [
        SectorForecastState("Stable", 0.61, 0.68, 0.22, 0.18, 0.18),
        SectorForecastState("Momentum", 0.60, 0.65, 0.20, 0.26, 0.24),
    ]
    stocks: list[StockForecastState] = []
    for idx in range(1, 10):
        stocks.append(StockForecastState(f"S{idx}", "Stable", 0.55, 0.61, 0.69, 0.60, 0.16, 0.91, alpha_score=0.65))
        stocks.append(StockForecastState(f"M{idx}", "Momentum", 0.69, 0.57, 0.53, 0.59, 0.05, 0.83, alpha_score=0.64))

    normal_market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.56,
        up_5d_prob=0.60,
        up_20d_prob=0.63,
        trend_state="trend",
        drawdown_risk=0.20,
        volatility_regime="normal",
        liquidity_stress=0.18,
    )
    high_vol_market = normal_market.__class__(
        as_of_date=normal_market.as_of_date,
        up_1d_prob=normal_market.up_1d_prob,
        up_5d_prob=normal_market.up_5d_prob,
        up_20d_prob=normal_market.up_20d_prob,
        trend_state=normal_market.trend_state,
        drawdown_risk=0.42,
        volatility_regime="high",
        liquidity_stress=normal_market.liquidity_stress,
    )
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.02,
        growth_vs_value_bias=0.01,
        fund_flow_strength=0.08,
        margin_risk_on_score=0.08,
        breadth_strength=0.06,
        leader_participation=0.54,
        weak_stock_ratio=0.52,
    )

    normal_state = compose_state(
        market=normal_market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    high_vol_state = compose_state(
        market=high_vol_market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )

    assert high_vol_state.candidate_selection.shortlist_size < normal_state.candidate_selection.shortlist_size
    assert sum(
        1
        for symbol in high_vol_state.candidate_selection.shortlisted_symbols[
            : high_vol_state.candidate_selection.shortlist_size
        ]
        if symbol.startswith("M")
    ) <= 1


def test_apply_policy_trims_exposure_when_candidate_set_is_fragile() -> None:
    sectors = [SectorForecastState("Stable", 0.60, 0.66, 0.18, 0.20, 0.18)]
    stable_stocks = [
        StockForecastState("S1", "Stable", 0.55, 0.61, 0.68, 0.58, 0.16, 0.90, alpha_score=0.64),
        StockForecastState("S2", "Stable", 0.54, 0.60, 0.66, 0.57, 0.12, 0.88, alpha_score=0.62),
        StockForecastState("S3", "Stable", 0.53, 0.59, 0.65, 0.56, 0.10, 0.87, alpha_score=0.60),
    ]
    fragile_stocks = [
        StockForecastState("F1", "Stable", 0.70, 0.56, 0.51, 0.51, 0.02, 0.82, alpha_score=0.63),
        StockForecastState("F2", "Stable", 0.68, 0.55, 0.50, 0.50, 0.01, 0.80, alpha_score=0.61),
        StockForecastState("F3", "Stable", 0.67, 0.54, 0.49, 0.49, 0.01, 0.79, alpha_score=0.60),
    ]
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.52,
        up_5d_prob=0.54,
        up_20d_prob=0.56,
        trend_state="range",
        drawdown_risk=0.34,
        volatility_regime="normal",
        liquidity_stress=0.24,
    )
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.01,
        growth_vs_value_bias=-0.01,
        fund_flow_strength=0.02,
        margin_risk_on_score=0.04,
        breadth_strength=0.08,
        leader_participation=0.52,
        weak_stock_ratio=0.50,
    )

    stable_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stable_stocks,
        cross_section=cross_section,
    )
    fragile_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=fragile_stocks,
        cross_section=cross_section,
    )

    stable_decision = apply_policy(
        PolicyInput(composite_state=stable_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )
    fragile_decision = apply_policy(
        PolicyInput(composite_state=fragile_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )

    assert fragile_decision.target_exposure < stable_decision.target_exposure
    assert fragile_decision.turnover_cap <= stable_decision.turnover_cap
    assert any("Candidate set fragile" in note for note in fragile_decision.risk_notes)


def test_large_universe_shortlist_spreads_concentration() -> None:
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.55,
        up_5d_prob=0.58,
        up_20d_prob=0.62,
        trend_state="trend",
        drawdown_risk=0.22,
        volatility_regime="normal",
        liquidity_stress=0.20,
    )
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.04,
        growth_vs_value_bias=0.02,
        fund_flow_strength=0.10,
        margin_risk_on_score=0.10,
        breadth_strength=0.16,
        leader_participation=0.60,
        weak_stock_ratio=0.26,
    )
    sectors = [
        SectorForecastState("Alpha", 0.61, 0.67, 0.20, 0.18, 0.18),
        SectorForecastState("Beta", 0.58, 0.64, 0.16, 0.16, 0.20),
    ]
    stocks: list[StockForecastState] = []
    for idx in range(1, 11):
        stocks.append(StockForecastState(f"A{idx}", "Alpha", 0.56, 0.61, 0.68, 0.59, 0.14, 0.90, alpha_score=0.66))
        stocks.append(StockForecastState(f"B{idx}", "Beta", 0.54, 0.59, 0.64, 0.56, 0.10, 0.88, alpha_score=0.62))

    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    widened_state = composite_state.__class__(
        market=composite_state.market,
        cross_section=composite_state.cross_section,
        sectors=composite_state.sectors,
        stocks=composite_state.stocks,
        strategy_mode=composite_state.strategy_mode,
        risk_regime=composite_state.risk_regime,
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=composite_state.candidate_selection.shortlisted_symbols,
            shortlisted_sectors=composite_state.candidate_selection.shortlisted_sectors,
            sector_slots=composite_state.candidate_selection.sector_slots,
            total_scored=300,
            shortlist_size=max(12, composite_state.candidate_selection.shortlist_size),
            shortlist_ratio=max(12, composite_state.candidate_selection.shortlist_size) / 300.0,
            selection_mode="macro_sector_shortlist",
            selection_notes=["Large universe shortlist for diversification test."],
        ),
        market_info_state=composite_state.market_info_state,
        sector_info_states=composite_state.sector_info_states,
        stock_info_states=composite_state.stock_info_states,
        capital_flow_state=composite_state.capital_flow_state,
        macro_context_state=composite_state.macro_context_state,
    )

    decision = apply_policy(
        PolicyInput(composite_state=widened_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )

    assert decision.target_position_count >= 3
    assert all(weight <= 0.18 + 1e-9 for weight in decision.desired_symbol_target_weights.values())
    assert any("Large-universe shortlist" in note for note in decision.risk_notes)


def test_large_universe_shortlist_allows_selective_concentration_for_dominant_alpha() -> None:
    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.57,
        up_5d_prob=0.63,
        up_20d_prob=0.66,
        trend_state="trend",
        drawdown_risk=0.18,
        volatility_regime="normal",
        liquidity_stress=0.16,
    )
    cross_section = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.03,
        growth_vs_value_bias=0.01,
        fund_flow_strength=0.12,
        margin_risk_on_score=0.11,
        breadth_strength=0.12,
        leader_participation=0.60,
        weak_stock_ratio=0.24,
    )
    sectors = [
        SectorForecastState("Alpha", 0.63, 0.69, 0.22, 0.18, 0.16),
        SectorForecastState("Beta", 0.55, 0.60, 0.10, 0.16, 0.18),
    ]
    stocks: list[StockForecastState] = [
        StockForecastState("A1", "Alpha", 0.60, 0.72, 0.71, 0.66, 0.18, 0.93, alpha_score=0.76, up_2d_prob=0.63, up_3d_prob=0.69),
        StockForecastState("A2", "Alpha", 0.59, 0.69, 0.69, 0.63, 0.16, 0.92, alpha_score=0.72, up_2d_prob=0.62, up_3d_prob=0.66),
        StockForecastState("A3", "Alpha", 0.55, 0.58, 0.60, 0.55, 0.10, 0.89, alpha_score=0.55, up_2d_prob=0.56, up_3d_prob=0.57),
    ]
    for idx in range(4, 9):
        stocks.append(
            StockForecastState(
                f"A{idx}",
                "Alpha",
                0.51,
                0.53,
                0.56,
                0.51,
                0.06,
                0.86,
                alpha_score=0.50,
                up_2d_prob=0.51,
                up_3d_prob=0.52,
            )
        )
    for idx in range(1, 7):
        stocks.append(
            StockForecastState(
                f"B{idx}",
                "Beta",
                0.50,
                0.52,
                0.55,
                0.50,
                0.05,
                0.85,
                alpha_score=0.49,
                up_2d_prob=0.50,
                up_3d_prob=0.51,
            )
        )

    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=stocks,
        cross_section=cross_section,
    )
    concentrated_state = composite_state.__class__(
        market=composite_state.market,
        cross_section=composite_state.cross_section,
        sectors=composite_state.sectors,
        stocks=composite_state.stocks,
        strategy_mode=composite_state.strategy_mode,
        risk_regime=composite_state.risk_regime,
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=[stock.symbol for stock in composite_state.stocks],
            shortlisted_sectors=["Alpha", "Beta"],
            sector_slots={"Alpha": 2, "Beta": 1},
            total_scored=300,
            shortlist_size=14,
            shortlist_ratio=14 / 300.0,
            selection_mode="macro_sector_shortlist",
            selection_notes=["Large universe shortlist with dominant top alpha."],
        ),
        mainlines=composite_state.mainlines,
        market_info_state=composite_state.market_info_state,
        sector_info_states=composite_state.sector_info_states,
        stock_info_states=composite_state.stock_info_states,
        capital_flow_state=composite_state.capital_flow_state,
        macro_context_state=composite_state.macro_context_state,
    )

    decision = apply_policy(
        PolicyInput(composite_state=concentrated_state, current_weights={}, current_cash=1.0, total_equity=1.0)
    )

    assert decision.target_position_count <= 3
    assert max(decision.desired_symbol_target_weights.values()) > 0.18
    assert any("Top alpha concentrated and durable" in note for note in decision.risk_notes)
    assert any("selective concentration" in note for note in decision.risk_notes)


def test_composite_state_roundtrip_preserves_candidate_selection() -> None:
    market, sectors, stocks, cross_section = _make_demo_state()
    expanded_stocks = stocks * 8
    composite_state = compose_state(
        market=market,
        sectors=sectors,
        stocks=expanded_stocks,
        cross_section=cross_section,
    )

    payload = _serialize_composite_state(composite_state)
    restored = _decode_composite_state(payload)

    assert restored is not None
    assert restored.candidate_selection.shortlisted_symbols == composite_state.candidate_selection.shortlisted_symbols
    assert restored.candidate_selection.shortlisted_sectors == composite_state.candidate_selection.shortlisted_sectors
    assert [item.name for item in restored.mainlines] == [item.name for item in composite_state.mainlines]


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
    assert decision.sector_budgets["其他"] <= decision.desired_sector_budgets["其他"] + 1e-9
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

    assert _policy_score(steady) > _policy_score(fragile)


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
    assert decision.target_position_count >= 3
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
    assert "run_id" in summary
    assert "snapshot_hash" in summary
    assert "config_hash" in summary
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

    assert _policy_score(fast_follow) > _policy_score(slow_burn)


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


def test_v2_policy_signal_profile_prefers_entry_and_hold_quality() -> None:
    profile = policy_runtime.stock_signal_profile(
        StockForecastState(
            "AAA",
            "寮?",
            0.60,
            0.66,
            0.72,
            0.60,
            0.15,
            0.91,
            alpha_score=0.67,
            up_2d_prob=0.62,
            up_3d_prob=0.65,
            up_10d_prob=0.69,
        ),
        deps=legacy_services._policy_runtime_dependencies(),
    )

    assert float(profile["entry_score"]) > 0.70
    assert float(profile["hold_score"]) > 0.72
    assert float(profile["exit_risk"]) < 0.10
    assert bool(profile["strong_entry"])
    assert bool(profile["strong_hold"])


def test_v2_policy_signal_profile_flags_exit_risk() -> None:
    profile = policy_runtime.stock_signal_profile(
        StockForecastState(
            "AAA",
            "寮?",
            0.40,
            0.41,
            0.45,
            0.42,
            0.10,
            0.80,
            alpha_score=0.46,
            up_2d_prob=0.41,
            up_3d_prob=0.39,
            up_10d_prob=0.44,
        ),
        deps=legacy_services._policy_runtime_dependencies(),
    )

    assert float(profile["hold_score"]) < 0.58
    assert float(profile["exit_risk"]) > 0.11
    assert bool(profile["weakening"])
    assert not bool(profile["strong_hold"])


def test_v2_policy_alpha_metrics_capture_entry_signal_separation() -> None:
    metrics = policy_runtime.alpha_opportunity_metrics(
        [
            StockForecastState(
                "AAA",
                "鏈夎壊",
                0.58,
                0.65,
                0.71,
                0.60,
                0.10,
                0.90,
                alpha_score=0.60,
                up_2d_prob=0.61,
                up_3d_prob=0.64,
                up_10d_prob=0.69,
            ),
            StockForecastState(
                "BBB",
                "鏈夎壊",
                0.61,
                0.53,
                0.56,
                0.49,
                0.10,
                0.88,
                alpha_score=0.60,
                up_2d_prob=0.55,
                up_3d_prob=0.50,
                up_10d_prob=0.54,
            ),
            StockForecastState(
                "CCC",
                "鍖栧伐",
                0.55,
                0.57,
                0.59,
                0.52,
                0.08,
                0.85,
                alpha_score=0.58,
                up_2d_prob=0.56,
                up_3d_prob=0.57,
                up_10d_prob=0.60,
            ),
        ],
        deps=legacy_services._policy_runtime_dependencies(),
    )

    assert float(metrics["entry_top_score"]) > float(metrics["entry_median_score"])
    assert float(metrics["entry_separation"]) > 0.05
    assert float(metrics["hold_strength"]) > 0.65


def test_v2_policy_allows_early_exit_for_weakening_alpha_within_holding_window() -> None:
    adjusted, notes = policy_runtime.finalize_target_weights(
        desired_weights={"AAA": 0.0},
        current_weights={"AAA": 0.18},
        current_holding_days={"AAA": 3},
        stocks=[
            StockForecastState(
                "AAA",
                "寮?",
                0.32,
                0.35,
                0.39,
                0.38,
                0.05,
                0.75,
                alpha_score=0.40,
                up_2d_prob=0.33,
                up_3d_prob=0.31,
                up_10d_prob=0.37,
            )
        ],
        target_exposure=0.18,
        min_trade_delta=0.02,
        min_holding_days=5,
        deps=legacy_services._policy_runtime_dependencies(),
    )

    assert adjusted.get("AAA", 0.0) == pytest.approx(0.0)
    assert any("early exit allowed" in note for note in notes)


def test_v2_policy_softens_trim_for_strong_alpha_holding() -> None:
    adjusted, notes = policy_runtime.finalize_target_weights(
        desired_weights={"AAA": 0.10},
        current_weights={"AAA": 0.18},
        current_holding_days={"AAA": 7},
        stocks=[
            StockForecastState(
                "AAA",
                "寮?",
                0.63,
                0.66,
                0.70,
                0.58,
                0.15,
                0.91,
                alpha_score=0.70,
                up_2d_prob=0.64,
                up_3d_prob=0.65,
                up_10d_prob=0.68,
            ),
        ]
        ,
        target_exposure=0.18,
        min_trade_delta=0.02,
        min_holding_days=5,
        deps=legacy_services._policy_runtime_dependencies(),
    )

    assert adjusted["AAA"] > 0.12
    assert any("trim softened" in note for note in notes)


def test_v2_policy_allows_small_trim_when_alpha_breaks_down() -> None:
    adjusted, _ = policy_runtime.finalize_target_weights(
        desired_weights={"AAA": 0.169},
        current_weights={"AAA": 0.18},
        current_holding_days={"AAA": 8},
        stocks=[
                StockForecastState(
                    "AAA",
                    "寮?",
                    0.40,
                    0.41,
                    0.45,
                    0.42,
                    0.10,
                    0.80,
                    alpha_score=0.46,
                    up_2d_prob=0.41,
                    up_3d_prob=0.39,
                    up_10d_prob=0.44,
                )
            ],
        target_exposure=0.20,
        min_trade_delta=0.02,
        min_holding_days=5,
        deps=legacy_services._policy_runtime_dependencies(),
    )

    assert adjusted["AAA"] == pytest.approx(0.169)


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
        allow_retrain=True,
    )
    second = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache"),
        refresh_cache=False,
        allow_retrain=True,
    )

    assert calls["quant"] == 1
    assert first.policy_decision.target_exposure == second.policy_decision.target_exposure
    assert len(second.trade_actions) == len(first.trade_actions)
    assert any(stage == "daily" and "量化预测测试进度" in message for stage, message in progress)
    assert any(stage == "daily" and "命中日运行缓存" in message for stage, message in progress)


def test_run_daily_v2_live_default_mode_uses_snapshot_without_retraining(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
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
    run_id = "20260303_124920"
    run_dir = tmp_path / "artifacts" / "swing_v2" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    learned_policy = {
        "feature_names": [],
        "exposure_intercept": 0.6,
        "exposure_coef": [],
        "position_intercept": 3.0,
        "position_coef": [],
        "turnover_intercept": 0.2,
        "turnover_coef": [],
        "train_rows": 0,
        "train_r2_exposure": 0.0,
        "train_r2_positions": 0.0,
        "train_r2_turnover": 0.0,
    }
    (run_dir / "learned_policy_model.json").write_text(json.dumps(learned_policy), encoding="utf-8")
    frozen_state = {
        "as_of_date": "2026-03-01",
        "next_date": "2026-03-02",
        "composite_state": {
            "market": {
                "as_of_date": "2026-03-01",
                "up_1d_prob": 0.57,
                "up_5d_prob": 0.59,
                "up_20d_prob": 0.61,
                "trend_state": "trend",
                "drawdown_risk": 0.28,
                "volatility_regime": "normal",
                "liquidity_stress": 0.22,
                "up_2d_prob": 0.55,
                "up_3d_prob": 0.56,
            },
            "cross_section": {
                "as_of_date": "2026-03-01",
                "large_vs_small_bias": 0.08,
                "growth_vs_value_bias": -0.04,
                "fund_flow_strength": 0.16,
                "margin_risk_on_score": 0.14,
                "breadth_strength": 0.21,
                "leader_participation": 0.63,
                "weak_stock_ratio": 0.29,
            },
            "sectors": [
                {
                    "sector": "有色",
                    "up_5d_prob": 0.58,
                    "up_20d_prob": 0.62,
                    "relative_strength": 0.18,
                    "rotation_speed": 0.42,
                    "crowding_score": 0.31,
                }
            ],
            "stocks": [
                {
                    "symbol": "000630.SZ",
                    "sector": "有色",
                    "up_1d_prob": 0.58,
                    "up_5d_prob": 0.60,
                    "up_20d_prob": 0.64,
                    "excess_vs_sector_prob": 0.55,
                    "event_impact_score": 0.10,
                    "tradeability_score": 0.88,
                    "alpha_score": 0.75,
                    "tradability_status": "normal",
                    "up_2d_prob": 0.59,
                    "up_3d_prob": 0.60,
                }
            ],
            "strategy_mode": "trend_follow",
            "risk_regime": "risk_on",
        },
    }
    (run_dir / "frozen_daily_state.json").write_text(json.dumps(frozen_state), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "strategy_id": "swing_v2",
        "config_hash": "cfg_hash_1",
        "snapshot_hash": "snap_hash_1",
        "policy_hash": "policy_hash_1",
        "universe_hash": "universe_hash_1",
        "model_hashes": {"stock_model": "stock_hash_1"},
        "dataset_manifest": str(run_dir / "dataset_manifest.json"),
        "learned_policy_model": str(run_dir / "learned_policy_model.json"),
        "frozen_daily_state": str(run_dir / "frozen_daily_state.json"),
    }
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps({"universe_file": "config/universe_smoke_5.json", "start": "2024-01-01", "end": "2026-03-01"}),
        encoding="utf-8",
    )
    (tmp_path / "artifacts" / "swing_v2" / "latest_research_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (run_dir / "research_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

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
        "end": "2026-03-01",
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

    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: settings)
    monkeypatch.setattr(
        "src.application.v2_services.load_watchlist",
        lambda *_: (Security("000001.SH", "指数"), [], {}),
    )
    monkeypatch.setattr(
        "src.application.v2_services.build_candidate_universe",
        lambda **_: type("Universe", (), {"rows": [Security("000630.SZ", "样例股", "有色")]})(),
    )
    monkeypatch.setattr(
        "src.application.v2_services.run_quant_pipeline",
        lambda **_: (_ for _ in ()).throw(AssertionError("run_quant_pipeline should not be called in snapshot mode")),
    )
    monkeypatch.setattr("src.application.v2_services.apply_policy", lambda *_, **__: decision)

    result = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache"),
        run_id=run_id,
    )

    assert result.run_id == run_id
    assert result.snapshot.run_id == run_id
    assert result.snapshot_hash == "snap_hash_1"


def test_run_daily_v2_live_fails_when_run_id_mismatches_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
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
        "end": "2026-03-01",
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
    manifest_path = tmp_path / "artifacts" / "swing_v2" / "mismatch" / "research_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "strategy_id": "swing_v2",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: settings)

    with pytest.raises(ValueError, match="run_id mismatch"):
        run_daily_v2_live(
            strategy_id="swing_v2",
            artifact_root=str(tmp_path / "artifacts"),
            run_id="run_b",
            snapshot_path=str(manifest_path),
        )


def test_run_daily_v2_live_fails_when_universe_tier_mismatches_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    watchlist_path = tmp_path / "watchlist.json"
    margin_market_path = tmp_path / "margin_market.csv"
    margin_stock_path = tmp_path / "margin_stock.csv"
    watchlist_path.write_text("[]", encoding="utf-8")
    margin_market_path.write_text("", encoding="utf-8")
    margin_stock_path.write_text("", encoding="utf-8")
    run_id = "run_tier_mismatch"
    run_dir = tmp_path / "artifacts" / "swing_v2" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "strategy_id": "swing_v2",
        "dataset_manifest": str(run_dir / "dataset_manifest.json"),
        "frozen_daily_state": str(run_dir / "frozen_daily_state.json"),
    }
    (run_dir / "research_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "universe_tier": "favorites_16",
                "universe_file": str(run_dir / "favorites.json"),
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "frozen_daily_state.json").write_text(
        json.dumps(
            {
                "composite_state": {
                    "market": {
                        "as_of_date": "2026-03-01",
                        "up_1d_prob": 0.57,
                        "up_5d_prob": 0.59,
                        "up_20d_prob": 0.61,
                        "trend_state": "trend",
                        "drawdown_risk": 0.28,
                        "volatility_regime": "normal",
                        "liquidity_stress": 0.22,
                        "up_2d_prob": 0.55,
                        "up_3d_prob": 0.56,
                    },
                    "cross_section": {
                        "as_of_date": "2026-03-01",
                        "large_vs_small_bias": 0.08,
                        "growth_vs_value_bias": -0.04,
                        "fund_flow_strength": 0.16,
                        "margin_risk_on_score": 0.14,
                        "breadth_strength": 0.21,
                        "leader_participation": 0.63,
                        "weak_stock_ratio": 0.29,
                    },
                    "sectors": [],
                    "stocks": [],
                    "strategy_mode": "trend_follow",
                    "risk_regime": "risk_on",
                }
            }
        ),
        encoding="utf-8",
    )
    settings = {
        "config_path": "config/api.json",
        "watchlist": str(watchlist_path),
        "source": "local",
        "data_dir": "data",
        "start": "2024-01-01",
        "end": "2026-03-01",
        "min_train_days": 240,
        "step_days": 20,
        "l2": 0.8,
        "max_positions": 5,
        "use_margin_features": False,
        "margin_market_file": str(margin_market_path),
        "margin_stock_file": str(margin_stock_path),
        "universe_tier": "generated_80",
        "universe_file": str(run_dir / "generated_80.json"),
        "universe_limit": 80,
        "source_universe_manifest_path": str(run_dir / "generated_80.json"),
        "universe_generation_rule": "test",
        "universe_id": "generated_80",
        "universe_size": 80,
        "symbol_count": 80,
        "symbols": [],
        "universe_hash": "u",
    }
    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: settings)
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda **kwargs: dict(kwargs["settings"]))

    with pytest.raises(ValueError, match="universe tier mismatch"):
        run_daily_v2_live(
            strategy_id="swing_v2",
            artifact_root=str(tmp_path / "artifacts"),
            run_id=run_id,
            universe_tier="generated_80",
        )


def test_run_daily_v2_live_info_shadow_only_keeps_trade_plan_stable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
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
    info_path = tmp_path / "info.csv"
    watchlist_path.write_text("[]", encoding="utf-8")
    universe_path.write_text("[]", encoding="utf-8")
    margin_market_path.write_text("", encoding="utf-8")
    margin_stock_path.write_text("", encoding="utf-8")
    info_path.write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,info_type,title,event_tag",
                "2026-03-01,market,MARKET,mid,bearish,news,macro caution,regulatory_negative",
                "2026-03-01,stock,000630.SZ,short,bullish,announcement,positive contract,contract_win",
            ]
        ),
        encoding="utf-8",
    )
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
        "info_file": str(info_path),
        "use_info_fusion": False,
        "info_shadow_only": True,
        "info_types": ["news", "announcement", "research"],
    }

    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: dict(settings))
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda **kwargs: dict(kwargs["settings"]))
    monkeypatch.setattr(
        "src.application.v2_services.load_watchlist",
        lambda *_: (Security("000001.SH", "指数"), [], {}),
    )
    monkeypatch.setattr(
        "src.application.v2_services.build_candidate_universe",
        lambda **_: type("Universe", (), {"rows": [Security("000630.SZ", "样例股", "有色")]})(),
    )
    monkeypatch.setattr(
        "src.application.v2_services.run_quant_pipeline",
        lambda **_: (
            MarketForecast(
                symbol="000001.SH",
                name="指数",
                latest_date=pd.Timestamp("2026-03-01"),
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
                    latest_date=pd.Timestamp("2026-03-01"),
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
        ),
    )
    monkeypatch.setattr(
        "src.application.v2_services.load_symbol_daily",
        lambda **_: pd.DataFrame({"date": pd.to_datetime(["2026-03-01"]), "close": [1.0]}),
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

    baseline = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache_base"),
        refresh_cache=True,
        allow_retrain=True,
    )
    settings["use_info_fusion"] = True
    shadow = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache_shadow"),
        refresh_cache=True,
        allow_retrain=True,
    )

    assert baseline.policy_decision.target_exposure == shadow.policy_decision.target_exposure
    assert baseline.trade_actions == shadow.trade_actions
    assert shadow.info_shadow_enabled is True
    assert shadow.info_item_count == 2
    assert shadow.top_negative_info_events
