from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from src.application.v2_contracts import (
    CapitalFlowState,
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    DailyRunResult,
    LearnedPolicyModel,
    MainlineState,
    MarketForecastState,
    MacroContextState,
    PolicyDecision,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    StrategyMemoryRecall,
    StrategySnapshot,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
from src.application.v2_services import _simulate_execution_day
from src.domain.entities import TradeAction
from src.interfaces.presenters.html_dashboard import write_v2_daily_dashboard, write_v2_research_dashboard
from src.interfaces.presenters.markdown_reports import write_v2_daily_report, write_v2_research_report


def _make_daily_result() -> DailyRunResult:
    composite = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-01",
            up_1d_prob=0.56,
            up_5d_prob=0.61,
            up_20d_prob=0.64,
            trend_state="trend",
            drawdown_risk=0.22,
            volatility_regime="normal",
            liquidity_stress=0.18,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-01",
            large_vs_small_bias=0.08,
            growth_vs_value_bias=-0.03,
            fund_flow_strength=0.12,
            margin_risk_on_score=0.10,
            breadth_strength=0.20,
            leader_participation=0.62,
            weak_stock_ratio=0.24,
        ),
        sectors=[
            SectorForecastState("有色", 0.58, 0.64, 0.18, 0.24, 0.20),
        ],
        stocks=[
            StockForecastState("AAA", "有色", 0.55, 0.62, 0.66, 0.57, 0.03, 0.88),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["AAA"],
            shortlisted_sectors=["鏈夎壊"],
            sector_slots={"鏈夎壊": 1},
            total_scored=12,
            shortlist_size=1,
            shortlist_ratio=1.0 / 12.0,
            selection_mode="macro_sector_shortlist",
            selection_notes=["Macro shortlist active: 1 sector prioritized before fine timing."],
        ),
        mainlines=[
            MainlineState(
                name="资源",
                driver="flow_supported",
                conviction=0.64,
                breadth=0.38,
                leadership=0.26,
                catalyst_strength=0.22,
                event_risk_level=0.12,
                sectors=["鏈夎壊"],
                representative_symbols=["AAA"],
            )
        ],
        capital_flow_state=CapitalFlowState(
            northbound_net_flow=0.24,
            margin_balance_change=0.10,
            turnover_heat=0.66,
            large_order_bias=0.18,
            flow_regime="inflow",
        ),
        macro_context_state=MacroContextState(
            style_regime="quality",
            commodity_pressure=0.18,
            fx_pressure=0.14,
            index_breadth_proxy=0.64,
            macro_risk_level="neutral",
        ),
    )
    decision = PolicyDecision(
        target_exposure=0.70,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.20,
        intraday_t_allowed=False,
        turnover_cap=0.25,
        sector_budgets={"有色": 0.70},
        symbol_target_weights={"AAA": 0.70},
        risk_notes=["测试备注"],
    )
    return DailyRunResult(
        snapshot=StrategySnapshot(
            strategy_id="swing_v2",
            universe_id="smoke",
            feature_set_version="v2",
            market_model_id="market_v2",
            sector_model_id="sector_v2",
            stock_model_id="stock_panel_v2",
            cross_section_model_id="cross_v2",
            policy_version="policy_v2",
            execution_version="exec_v2",
            use_us_index_context=True,
            us_index_source="akshare",
            external_signal_manifest_path="artifacts/v2/swing_v2/20260310_210000/external_signal_manifest.json",
            external_signal_version="v1",
            external_signal_enabled=True,
            generator_manifest_path="artifacts/v2/cache/universe_catalog/dynamic_300.generator.json",
            generator_version="dynamic_universe_v1",
            generator_hash="generatorhash123",
            coarse_pool_size=1000,
            refined_pool_size=300,
            selected_pool_size=300,
            theme_allocations=[
                {"theme": "资源", "selected_count": 6, "refined_count": 12, "coarse_count": 22, "theme_strength": 0.71},
                {"theme": "能源石油", "selected_count": 4, "refined_count": 8, "coarse_count": 16, "theme_strength": 0.65},
            ],
            capital_flow_snapshot={
                "flow_regime": "inflow",
                "northbound_net_flow": 0.24,
                "margin_balance_change": 0.10,
                "turnover_heat": 0.66,
                "large_order_bias": 0.18,
            },
            macro_context_snapshot={
                "macro_risk_level": "neutral",
                "style_regime": "quality",
                "commodity_pressure": 0.18,
                "fx_pressure": 0.14,
                "index_breadth_proxy": 0.64,
            },
        ),
        composite_state=composite,
        policy_decision=decision,
        trade_actions=[
            TradeAction(
                symbol="AAA",
                name="样例股",
                action="BUY",
                current_weight=0.20,
                target_weight=0.70,
                delta_weight=0.50,
                note="加仓",
            )
        ],
        memory_path="artifacts/v2/memory/swing_v2_memory.json",
        memory_recall=StrategyMemoryRecall(
            memory_path="artifacts/v2/memory/swing_v2_memory.json",
            latest_research_run_id="20260310_210000",
            latest_research_end_date="2026-03-10",
            latest_research_release_gate_passed=True,
            latest_research_excess_annual_return=0.08,
            latest_research_information_ratio=0.72,
            recent_daily_run_count=3,
            average_target_exposure=0.66,
            exposure_trend=0.10,
            rebalance_ratio=2 / 3,
            recurring_symbols=["AAA", "BBB"],
            recurring_risk_tags=["earnings_negative"],
            recurring_positive_tags=["AAA"],
            recurring_event_risk_tags=["earnings_negative", "regulatory_warning"],
            recurring_catalyst_tags=["research_positive"],
            recent_flow_regimes=["inflow"],
            recurring_macro_risk_levels=["neutral"],
            narrative=[
                "最近一次研究 run_id=20260310_210000，超额年化 8.0%，IR 0.72，release gate 通过。",
                "近 3 次日运行平均目标仓位 66.0%，调仓触发占比 66.7%，近几次仓位有上调倾向。",
            ],
        ),
        external_signal_manifest_path="artifacts/v2/swing_v2/20260310_210000/external_signal_manifest.json",
        external_signal_version="v1",
        external_signal_enabled=True,
        generator_manifest_path="artifacts/v2/cache/universe_catalog/dynamic_300.generator.json",
        generator_version="dynamic_universe_v1",
        generator_hash="generatorhash123",
        coarse_pool_size=1000,
        refined_pool_size=300,
        selected_pool_size=300,
        theme_allocations=[
            {"theme": "资源", "selected_count": 6, "refined_count": 12, "coarse_count": 22, "theme_strength": 0.71},
            {"theme": "能源石油", "selected_count": 4, "refined_count": 8, "coarse_count": 16, "theme_strength": 0.65},
        ],
        capital_flow_snapshot={
            "flow_regime": "inflow",
            "northbound_net_flow": 0.24,
            "margin_balance_change": 0.10,
            "turnover_heat": 0.66,
            "large_order_bias": 0.18,
        },
        macro_context_snapshot={
            "macro_risk_level": "neutral",
            "style_regime": "quality",
            "commodity_pressure": 0.18,
            "fx_pressure": 0.14,
            "index_breadth_proxy": 0.64,
        },
    )


def _make_backtest(annual_return: float) -> V2BacktestSummary:
    return V2BacktestSummary(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_days=120,
        total_return=0.18,
        annual_return=annual_return,
        max_drawdown=-0.08,
        avg_turnover=0.16,
        total_cost=0.01,
        avg_rank_ic=0.11,
        avg_top_decile_return=0.008,
        avg_top_bottom_spread=0.012,
        avg_top_k_hit_rate=0.56,
        benchmark_total_return=0.10,
        benchmark_annual_return=0.09,
        excess_total_return=0.07,
        excess_annual_return=0.08,
        excess_max_drawdown=-0.04,
        information_ratio=0.72,
        horizon_metrics={
            "1d": {"rank_ic": 0.12, "top_decile_return": 0.003, "top_bottom_spread": 0.005, "top_k_hit_rate": 0.52},
            "5d": {"rank_ic": 0.10, "top_decile_return": 0.007, "top_bottom_spread": 0.012, "top_k_hit_rate": 0.56},
            "20d": {"rank_ic": 0.08, "top_decile_return": 0.018, "top_bottom_spread": 0.026, "top_k_hit_rate": 0.59},
        },
        nav_curve=[1.0, 1.08, 1.18],
        benchmark_nav_curve=[1.0, 1.04, 1.10],
        excess_nav_curve=[1.0, 1.038, 1.073],
        curve_dates=["2024-01-01", "2024-06-30", "2024-12-31"],
    )


def test_simulate_execution_day_respects_turnover_and_liquidity_caps() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.60,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.40,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        sector_budgets={"有色": 0.60},
        symbol_target_weights={"AAA": 0.60},
    )
    stock_states = [
        StockForecastState("AAA", "有色", 0.55, 0.60, 0.64, 0.56, 0.0, 0.10),
    ]
    stock_frames = {
        "AAA": pd.DataFrame(
            {
                "date": [date],
                "fwd_ret_1": [0.0],
            }
        )
    }

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={},
        current_cash=1.0,
        stock_states=stock_states,
        stock_frames=stock_frames,
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert round(turnover, 3) == 0.042
    assert 0.0 < fill_ratio < 1.0
    assert slip_bps > 0.0
    assert cost > 0.0
    assert next_weights["AAA"] < 0.05
    assert 0.95 < next_cash < 1.0
    assert daily_ret < 0.0


def test_simulate_execution_day_tolerates_missing_price_row_for_held_position() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.40,
        target_position_count=1,
        rebalance_now=False,
        rebalance_intensity=0.0,
        intraday_t_allowed=False,
        turnover_cap=0.0,
        sector_budgets={"有色": 0.40},
        symbol_target_weights={"AAA": 0.40},
    )

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"AAA": 0.40},
        current_cash=0.60,
        stock_states=[StockForecastState("AAA", "有色", 0.50, 0.52, 0.55, 0.50, 0.0, 0.80)],
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [pd.Timestamp("2024-01-05")],
                    "fwd_ret_1": [0.03],
                }
            )
        },
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert daily_ret == 0.0
    assert turnover == 0.0
    assert cost == 0.0
    assert fill_ratio == 0.0
    assert slip_bps == 0.0
    assert next_weights["AAA"] == 0.40
    assert next_cash == 0.60


def test_simulate_execution_day_tolerates_missing_stock_frame_for_held_position() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.30,
        target_position_count=1,
        rebalance_now=False,
        rebalance_intensity=0.0,
        intraday_t_allowed=False,
        turnover_cap=0.0,
        sector_budgets={"有色": 0.30},
        symbol_target_weights={"AAA": 0.30},
    )

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"AAA": 0.30},
        current_cash=0.70,
        stock_states=[],
        stock_frames={},
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert daily_ret == 0.0
    assert turnover == 0.0
    assert cost == 0.0
    assert fill_ratio == 0.0
    assert slip_bps == 0.0
    assert next_weights["AAA"] == 0.30
    assert next_cash == 0.70


def test_simulate_execution_day_uses_fallback_tradeability_when_state_missing() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.50,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.30,
        intraday_t_allowed=False,
        turnover_cap=0.30,
        sector_budgets={"有色": 0.50},
        symbol_target_weights={"AAA": 0.50},
    )
    stock_frames = {
        "AAA": pd.DataFrame(
            {
                "date": [date],
                "fwd_ret_1": [0.0],
            }
        )
    }

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={},
        current_cash=1.0,
        stock_states=[],
        stock_frames=stock_frames,
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert round(turnover, 3) == 0.084
    assert fill_ratio < 1.0
    assert slip_bps > 0.0
    assert cost > 0.0
    assert next_weights["AAA"] < 0.09
    assert 0.90 < next_cash < 1.0
    assert daily_ret < 0.0


def test_simulate_execution_day_skips_trades_for_halted_status() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.40,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.30,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        sector_budgets={"有色": 0.40},
        symbol_target_weights={"AAA": 0.40},
    )

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"AAA": 0.10},
        current_cash=0.90,
        stock_states=[StockForecastState("AAA", "有色", 0.55, 0.60, 0.64, 0.56, 0.0, 0.90, tradability_status="halted")],
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [date],
                    "fwd_ret_1": [0.02],
                }
            )
        },
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert turnover == 0.0
    assert cost == 0.0
    assert fill_ratio == 0.0
    assert slip_bps == 0.0
    assert next_weights["AAA"] > 0.10
    assert next_cash < 0.90


def test_simulate_execution_day_blocks_add_on_for_data_insufficient_status() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.30,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.20,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        sector_budgets={"有色": 0.30},
        symbol_target_weights={"AAA": 0.30},
    )

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"AAA": 0.10},
        current_cash=0.90,
        stock_states=[StockForecastState("AAA", "有色", 0.55, 0.57, 0.61, 0.54, 0.20, 0.35, tradability_status="data_insufficient")],
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [date],
                    "fwd_ret_1": [0.0],
                }
            )
        },
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert daily_ret == 0.0
    assert turnover == 0.0
    assert cost == 0.0
    assert fill_ratio == 0.0
    assert slip_bps == 0.0
    assert next_weights["AAA"] == 0.10
    assert next_cash == 0.90


def test_simulate_execution_day_charges_more_slippage_on_adverse_gap_open() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.20,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.10,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        sector_budgets={"有色": 0.20},
        symbol_target_weights={"AAA": 0.20},
    )
    base_kwargs = dict(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={},
        current_cash=1.0,
        stock_states=[StockForecastState("AAA", "有色", 0.58, 0.60, 0.64, 0.55, 0.0, 0.90)],
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )
    flat = _simulate_execution_day(
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [date],
                    "open": [10.0],
                    "close": [10.0],
                    "low": [9.9],
                    "high": [10.1],
                    "ret_1": [0.0],
                    "fwd_ret_1": [0.0],
                }
            )
        },
        **base_kwargs,
    )
    adverse = _simulate_execution_day(
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [date],
                    "open": [10.6],
                    "close": [10.0],
                    "low": [9.9],
                    "high": [10.7],
                    "ret_1": [0.0],
                    "fwd_ret_1": [0.0],
                }
            )
        },
        **base_kwargs,
    )

    assert adverse[2] > flat[2]
    assert adverse[4] > flat[4]


def test_simulate_execution_day_blocks_buy_when_limit_up_pinned() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.40,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.20,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        sector_budgets={"有色": 0.40},
        symbol_target_weights={"AAA": 0.40},
    )

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={},
        current_cash=1.0,
        stock_states=[StockForecastState("AAA", "有色", 0.60, 0.62, 0.66, 0.55, 0.0, 0.90)],
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [date],
                    "close": [11.0],
                    "low": [11.0],
                    "high": [11.0],
                    "ret_1": [0.10],
                    "fwd_ret_1": [0.0],
                }
            )
        },
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert turnover == 0.0
    assert cost == 0.0
    assert fill_ratio == 0.0
    assert slip_bps == 0.0
    assert next_weights == {}
    assert next_cash == 1.0
    assert daily_ret == 0.0


def test_simulate_execution_day_blocks_sell_when_limit_down_pinned() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.0,
        target_position_count=0,
        rebalance_now=True,
        rebalance_intensity=0.20,
        intraday_t_allowed=False,
        turnover_cap=0.30,
        sector_budgets={},
        symbol_target_weights={},
    )

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"AAA": 0.30},
        current_cash=0.70,
        stock_states=[StockForecastState("AAA", "有色", 0.45, 0.48, 0.50, 0.45, 0.0, 0.90)],
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [date],
                    "close": [9.0],
                    "low": [9.0],
                    "high": [9.0],
                    "ret_1": [-0.10],
                    "fwd_ret_1": [0.0],
                }
            )
        },
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert turnover == 0.0
    assert cost == 0.0
    assert fill_ratio == 0.0
    assert slip_bps == 0.0
    assert next_weights["AAA"] == 0.30
    assert next_cash == 0.70
    assert daily_ret == 0.0


def test_simulate_execution_day_marks_down_delisted_holding_without_price() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.10,
        target_position_count=1,
        rebalance_now=False,
        rebalance_intensity=0.0,
        intraday_t_allowed=False,
        turnover_cap=0.0,
        sector_budgets={"有色": 0.10},
        symbol_target_weights={"AAA": 0.10},
    )

    daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"AAA": 0.10},
        current_cash=0.90,
        stock_states=[StockForecastState("AAA", "有色", 0.40, 0.42, 0.45, 0.40, 0.0, 0.0, tradability_status="delisted")],
        stock_frames={},
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert turnover == 0.0
    assert cost == 0.0
    assert fill_ratio == 0.0
    assert slip_bps == 0.0
    assert daily_ret < 0.0
    assert next_weights["AAA"] < 0.08
    assert next_cash > 0.92


def test_simulate_execution_day_caps_delisted_positive_forward_return() -> None:
    date = pd.Timestamp("2024-01-02")
    next_date = pd.Timestamp("2024-01-03")
    decision = PolicyDecision(
        target_exposure=0.10,
        target_position_count=1,
        rebalance_now=False,
        rebalance_intensity=0.0,
        intraday_t_allowed=False,
        turnover_cap=0.0,
        sector_budgets={"有色": 0.10},
        symbol_target_weights={"AAA": 0.10},
    )

    daily_ret, *_rest, next_weights, next_cash = _simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"AAA": 0.10},
        current_cash=0.90,
        stock_states=[StockForecastState("AAA", "有色", 0.40, 0.42, 0.45, 0.40, 0.0, 0.0, tradability_status="delisted")],
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [date],
                    "fwd_ret_1": [0.15],
                }
            )
        },
        total_commission_rate=0.001,
        base_slippage_rate=0.0005,
    )

    assert daily_ret < 0.0
    assert next_weights["AAA"] == pytest.approx(0.08 / 0.98)
    assert next_cash == pytest.approx(0.90 / 0.98)


def test_v2_markdown_reports_keep_key_chinese_sections(tmp_path: Path) -> None:
    daily_result = _make_daily_result()
    daily_path = write_v2_daily_report(tmp_path / "daily.md", daily_result)
    daily_text = daily_path.read_text(encoding="utf-8")

    assert "swing_v2" in daily_text
    assert "smoke" in daily_text
    assert "external signal version: v1" in daily_text
    assert "US index context: enabled (akshare)" in daily_text
    assert "AAA, BBB" in daily_text
    assert "inflow" in daily_text
    assert "quality" in daily_text
    assert "generator manifest path" in daily_text
    assert "动态股票池" in daily_text

    baseline = _make_backtest(0.24)
    calibrated = _make_backtest(0.26)
    learned = _make_backtest(0.20)
    research_path = write_v2_research_report(
        tmp_path / "research.md",
        strategy_id="swing_v2",
        baseline=baseline,
        calibration=V2CalibrationResult(
            best_policy=PolicySpec(),
            best_score=0.12,
            baseline=baseline,
            calibrated=calibrated,
            trials=[
                {
                    "policy": {
                        "risk_on_exposure": 0.85,
                        "risk_on_positions": 4,
                        "risk_on_turnover_cap": 0.40,
                    },
                    "summary": {
                        "annual_return": 0.22,
                        "benchmark_annual_return": 0.09,
                        "excess_annual_return": 0.12,
                        "information_ratio": 0.61,
                        "max_drawdown": -0.07,
                    },
                    "score": 0.185,
                }
            ],
        ),
        learning=V2PolicyLearningResult(
            model=LearnedPolicyModel(
                feature_names=["x1"],
                exposure_intercept=0.5,
                exposure_coef=[0.1],
                position_intercept=2.0,
                position_coef=[0.1],
                turnover_intercept=0.2,
                turnover_coef=[0.05],
                train_rows=64,
                train_r2_exposure=0.20,
                train_r2_positions=0.18,
                train_r2_turnover=0.12,
            ),
            baseline=baseline,
            learned=learned,
        ),
    )
    research_text = research_path.read_text(encoding="utf-8")

    assert "swing_v2" in research_text
    assert "0.1850" in research_text
    assert "5d" in research_text
    assert "64" in research_text
    assert "0.720" in research_text
    assert "0.1200" in research_text


def test_v2_research_report_fails_on_artifact_run_id_mismatch(tmp_path: Path) -> None:
    baseline = _make_backtest(0.24)
    baseline = baseline.__class__(**{**asdict(baseline), "run_id": "run_a"})
    calibrated = _make_backtest(0.26)
    calibrated = calibrated.__class__(**{**asdict(calibrated), "run_id": "run_a"})
    learned = _make_backtest(0.20)
    learned = learned.__class__(**{**asdict(learned), "run_id": "run_a"})

    with pytest.raises(ValueError, match="artifact run_id mismatch"):
        write_v2_research_report(
            tmp_path / "research_mismatch.md",
            strategy_id="swing_v2",
            baseline=baseline,
            calibration=V2CalibrationResult(
                best_policy=PolicySpec(),
                best_score=0.12,
                baseline=baseline,
                calibrated=calibrated,
                trials=[],
            ),
            learning=V2PolicyLearningResult(
                model=LearnedPolicyModel(
                    feature_names=["x1"],
                    exposure_intercept=0.5,
                    exposure_coef=[0.1],
                    position_intercept=2.0,
                    position_coef=[0.1],
                    turnover_intercept=0.2,
                    turnover_coef=[0.05],
                    train_rows=64,
                    train_r2_exposure=0.20,
                    train_r2_positions=0.18,
                    train_r2_turnover=0.12,
                ),
                baseline=baseline,
                learned=learned,
            ),
            artifacts={"run_id": "run_b"},
        )


def test_v2_html_dashboards_keep_key_chinese_sections(tmp_path: Path) -> None:
    daily_result = _make_daily_result()
    daily_path = write_v2_daily_dashboard(tmp_path / "daily.html", daily_result)
    daily_html = daily_path.read_text(encoding="utf-8")

    assert "swing_v2" in daily_html
    assert "smoke" in daily_html
    assert "akshare" in daily_html
    assert "AAA, BBB" in daily_html
    assert "quality" in daily_html
    assert "Dynamic Universe Funnel" in daily_html
    assert "dynamic_universe_v1" in daily_html
    assert "inflow" in daily_html
    assert "1/12 shortlisted" in daily_html
    assert "Macro shortlist active" in daily_html
    assert "Mainline Radar" in daily_html
    assert "资源" in daily_html

    baseline = _make_backtest(0.24)
    calibrated = _make_backtest(0.26)
    learned = _make_backtest(0.20)
    research_path = write_v2_research_dashboard(
        tmp_path / "research.html",
        strategy_id="swing_v2",
        baseline=baseline,
        calibration=V2CalibrationResult(
            best_policy=PolicySpec(),
            best_score=0.12,
            baseline=baseline,
            calibrated=calibrated,
            trials=[
                {
                    "policy": {
                        "risk_on_exposure": 0.85,
                        "risk_on_positions": 4,
                        "risk_on_turnover_cap": 0.40,
                    },
                    "summary": {
                        "annual_return": 0.22,
                        "benchmark_annual_return": 0.09,
                        "excess_annual_return": 0.12,
                        "information_ratio": 0.61,
                        "max_drawdown": -0.07,
                    },
                    "score": 0.185,
                }
            ],
        ),
        learning=V2PolicyLearningResult(
            model=LearnedPolicyModel(
                feature_names=["x1"],
                exposure_intercept=0.5,
                exposure_coef=[0.1],
                position_intercept=2.0,
                position_coef=[0.1],
                turnover_intercept=0.2,
                turnover_coef=[0.05],
                train_rows=64,
                train_r2_exposure=0.20,
                train_r2_positions=0.18,
                train_r2_turnover=0.12,
            ),
            baseline=baseline,
            learned=learned,
        ),
    )
    research_html = research_path.read_text(encoding="utf-8")

    assert "swing_v2" in research_html
    assert ">5d<" in research_html
    assert "0.185" in research_html
    assert "x1" in research_html
    assert "64" in research_html
    assert "0.12" in research_html

