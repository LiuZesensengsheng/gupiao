from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.application.v2_backtest_prepare_runtime import BacktestTrajectory, TrajectoryStep
from src.application.v2_contracts import (
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
    StockRoleSnapshot,
    ThemeEpisode,
)
from src.application.v2_leader_runtime import (
    build_exit_training_labels,
    build_leader_training_labels,
    build_research_label_artifact_payloads,
)


def _make_step_state() -> CompositeState:
    return CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-01",
            up_1d_prob=0.60,
            up_5d_prob=0.64,
            up_20d_prob=0.68,
            trend_state="trend",
            drawdown_risk=0.20,
            volatility_regime="normal",
            liquidity_stress=0.18,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-01",
            large_vs_small_bias=0.06,
            growth_vs_value_bias=0.04,
            fund_flow_strength=0.18,
            margin_risk_on_score=0.12,
            breadth_strength=0.24,
            leader_participation=0.66,
            weak_stock_ratio=0.18,
        ),
        sectors=[SectorForecastState("chips", 0.62, 0.70, 0.24, 0.16, 0.14)],
        stocks=[
            StockForecastState("AAA", "chips", 0.66, 0.70, 0.76, 0.68, 0.14, 0.94, alpha_score=0.82),
            StockForecastState("BBB", "chips", 0.58, 0.60, 0.64, 0.58, 0.12, 0.90, alpha_score=0.66),
            StockForecastState("CCC", "chips", 0.46, 0.44, 0.42, 0.40, 0.06, 0.72, alpha_score=0.32),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["AAA", "BBB", "CCC"],
            shortlist_size=3,
            total_scored=3,
        ),
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="strengthening",
                conviction=0.74,
                breadth=0.38,
                leadership=0.32,
                catalyst_strength=0.28,
                event_risk=0.18,
            )
        ],
        stock_role_states={
            "AAA": StockRoleSnapshot(symbol="AAA", theme="chips", role="leader", theme_rank=1, theme_size=3),
            "BBB": StockRoleSnapshot(symbol="BBB", theme="chips", role="core", theme_rank=2, theme_size=3),
            "CCC": StockRoleSnapshot(
                symbol="CCC",
                theme="chips",
                role="laggard",
                previous_role="core",
                role_downgrade=True,
                theme_rank=3,
                theme_size=3,
            ),
        },
    )


def _make_trajectory() -> BacktestTrajectory:
    state = _make_step_state()
    step = TrajectoryStep(
        date=pd.Timestamp("2026-03-01"),
        next_date=pd.Timestamp("2026-03-02"),
        composite_state=state,
        stock_states=list(state.stocks),
        horizon_metrics={},
    )
    stock_frames = {
        "AAA": pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-01"),
                    "fwd_ret_1": 0.030,
                    "excess_ret_1_vs_mkt": 0.020,
                    "excess_ret_5_vs_sector": 0.080,
                    "excess_ret_20_vs_sector": 0.120,
                }
            ]
        ),
        "BBB": pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-01"),
                    "fwd_ret_1": 0.010,
                    "excess_ret_1_vs_mkt": 0.005,
                    "excess_ret_5_vs_sector": 0.020,
                    "excess_ret_20_vs_sector": 0.040,
                }
            ]
        ),
        "CCC": pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-01"),
                    "fwd_ret_1": -0.030,
                    "excess_ret_1_vs_mkt": -0.025,
                    "excess_ret_5_vs_sector": -0.080,
                    "excess_ret_20_vs_sector": -0.120,
                }
            ]
        ),
    }
    return BacktestTrajectory(
        prepared=SimpleNamespace(stock_frames=stock_frames),
        steps=[step],
    )


def test_build_leader_training_labels_marks_true_leader_and_negative_tail() -> None:
    rows = build_leader_training_labels(trajectory=_make_trajectory())

    assert len(rows) == 3
    row_map = {item.symbol: item for item in rows}
    assert row_map["AAA"].is_true_leader is True
    assert row_map["AAA"].leader_bucket == "true_leader"
    assert row_map["CCC"].hard_negative is True
    assert row_map["CCC"].future_excess_5d_vs_sector < 0.0
    assert row_map["CCC"].leader_bucket in {"hard_negative", "neutral"}


def test_build_exit_training_labels_generates_keep_and_exit_cases() -> None:
    rows = build_exit_training_labels(trajectory=_make_trajectory())

    assert len(rows) == 3
    row_map = {item.symbol: item for item in rows}
    assert row_map["AAA"].sample_source == "shortlist"
    assert row_map["AAA"].exit_label == "keep"
    assert row_map["AAA"].exit_severity_label == "keep"
    assert row_map["CCC"].should_exit_early is True
    assert row_map["CCC"].should_watch is True
    assert row_map["CCC"].should_reduce is True
    assert row_map["CCC"].exit_label in {"reduce", "exit_fast"}
    assert row_map["CCC"].future_drag_score == row_map["CCC"].exit_pressure_score
    assert row_map["CCC"].exit_pressure_score > row_map["AAA"].exit_pressure_score
    assert row_map["CCC"].sample_weight > row_map["AAA"].sample_weight


def test_build_research_label_artifact_payloads_handles_empty_and_populated_trajectory() -> None:
    empty_payload = build_research_label_artifact_payloads(trajectory=None)
    assert empty_payload["training_label_manifest"]["leader_row_count"] == 0
    assert empty_payload["training_label_manifest"]["exit_row_count"] == 0

    payload = build_research_label_artifact_payloads(trajectory=_make_trajectory())
    assert payload["training_label_manifest"]["leader_row_count"] == 3
    assert payload["training_label_manifest"]["leader_true_count"] >= 1
    assert payload["training_label_manifest"]["exit_row_count"] == 3
    assert payload["training_label_manifest"]["exit_watch_count"] >= 0
    assert payload["training_label_manifest"]["exit_fast_count"] + payload["training_label_manifest"]["exit_reduce_count"] >= 1
