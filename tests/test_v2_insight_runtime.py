from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import pytest

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    DailyRunResult,
    ExecutionPlan,
    HorizonForecast,
    InfoAggregateState,
    InfoDivergenceRecord,
    InfoItem,
    InfoSignalRecord,
    LearnedPolicyModel,
    MainlineState,
    MarketForecastState,
    PolicyDecision,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    StockRoleSnapshot,
    StrategySnapshot,
    ThemeEpisode,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
    Viewpoint,
)
from src.application.v2_execution_overlay_runtime import build_execution_plans
from src.application.v2_insight_runtime import (
    attach_insight_memory_to_state,
    build_viewpoints,
    write_insight_artifacts,
)
from src.application.v2_leader_runtime import build_leader_artifact_payloads
from src.application.v2_snapshot_support import decode_composite_state, serialize_composite_state
from src.application.v2_stock_role_runtime import build_stock_role_snapshots
from src.application.v2_theme_episode_runtime import build_theme_episodes
from src.domain.entities import TradeAction
from src.reporting.view_models import build_daily_report_view_model, build_research_report_view_model


def _base_composite_state() -> CompositeState:
    return CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-12",
            up_1d_prob=0.56,
            up_5d_prob=0.60,
            up_20d_prob=0.63,
            trend_state="trend",
            drawdown_risk=0.20,
            volatility_regime="normal",
            liquidity_stress=0.18,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-12",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.01,
            fund_flow_strength=0.12,
            margin_risk_on_score=0.10,
            breadth_strength=0.20,
            leader_participation=0.62,
            weak_stock_ratio=0.20,
        ),
        sectors=[],
        stocks=[],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )


def _make_backtest_summary(*, annual_return: float, excess_annual_return: float) -> V2BacktestSummary:
    return V2BacktestSummary(
        start_date="2025-01-01",
        end_date="2025-12-31",
        n_days=120,
        total_return=0.20,
        annual_return=annual_return,
        max_drawdown=-0.08,
        avg_turnover=0.12,
        total_cost=0.01,
        avg_rank_ic=0.07,
        avg_top_decile_return=0.01,
        avg_top_bottom_spread=0.02,
        avg_top_k_hit_rate=0.55,
        horizon_metrics={
            "1d": {"rank_ic": 0.06, "top_decile_return": 0.002, "top_bottom_spread": 0.004, "top_k_hit_rate": 0.51},
            "5d": {"rank_ic": 0.08, "top_decile_return": 0.008, "top_bottom_spread": 0.012, "top_k_hit_rate": 0.56},
            "20d": {"rank_ic": 0.09, "top_decile_return": 0.015, "top_bottom_spread": 0.022, "top_k_hit_rate": 0.58},
        },
        benchmark_total_return=0.10,
        benchmark_annual_return=0.09,
        excess_total_return=0.08,
        excess_annual_return=excess_annual_return,
        excess_max_drawdown=-0.05,
        information_ratio=0.75,
    )


def test_build_viewpoints_from_notes_deduplicates_and_keeps_conflicts(tmp_path) -> None:
    note_dir = tmp_path / "notes"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "2026-03-10.md").write_text(
        "\n".join(
            [
                "---",
                "effective_time: 2026-03-10",
                "ingest_time: 2026-03-11T09:30:00",
                "source: manual_note",
                "---",
                "## stronger duplicate",
                "- target_type: stock",
                "- target: AAA",
                "- theme: chips",
                "- direction: bullish",
                "- confidence: 0.85",
                "- importance: 0.90",
                "- horizon: mid",
                "- reason: demand improving",
                "- invalid_if: loses breakout",
                "- event_tag: catalyst",
                "## weaker duplicate",
                "- target_type: stock",
                "- target: AAA",
                "- theme: chips",
                "- direction: bullish",
                "- confidence: 0.70",
                "- importance: 0.80",
                "- horizon: mid",
                "- reason: demand improving",
                "## conflict",
                "- target_type: stock",
                "- target: AAA",
                "- theme: chips",
                "- direction: bearish",
                "- confidence: 0.60",
                "- importance: 0.70",
                "- horizon: short",
                "- reason: earnings risk",
                "",
            ]
        ),
        encoding="utf-8",
    )

    viewpoints = build_viewpoints(
        settings={
            "insight_notes_dir": str(note_dir),
            "insight_lookback_days": 10,
            "info_half_life_days": 10.0,
        },
        as_of_date=pd.Timestamp("2026-03-12"),
        info_items=[],
    )

    assert len(viewpoints) == 2
    assert {item.direction for item in viewpoints} == {"bullish", "bearish"}

    bullish = next(item for item in viewpoints if item.direction == "bullish")
    expected_recency = 0.5 ** (2.0 / 10.0)
    expected_weight = expected_recency * 0.85 * 0.90 * 1.15
    assert bullish.confidence == 0.85
    assert bullish.importance == 0.90
    assert bullish.reason == "demand improving"
    assert bullish.invalid_if == "loses breakout"
    assert bullish.weight == expected_weight


def test_build_viewpoints_from_notes_skips_future_ingest_time(tmp_path) -> None:
    note_dir = tmp_path / "notes"
    note_dir.mkdir(parents=True, exist_ok=True)
    (note_dir / "2026-03-10.md").write_text(
        "\n".join(
            [
                "---",
                "effective_time: 2026-03-10",
                "ingest_time: 2026-03-13T09:30:00",
                "---",
                "- target_type: stock",
                "- target: AAA",
                "- direction: bullish",
                "- confidence: 0.8",
                "- importance: 0.8",
                "- horizon: mid",
                "- reason: backfilled after the fact",
            ]
        ),
        encoding="utf-8",
    )

    viewpoints = build_viewpoints(
        settings={
            "insight_notes_dir": str(note_dir),
            "insight_lookback_days": 10,
            "info_half_life_days": 10.0,
        },
        as_of_date=pd.Timestamp("2026-03-12"),
        info_items=[],
    )

    assert viewpoints == []


def test_build_viewpoints_from_info_items_skips_future_publish_datetime() -> None:
    viewpoints = build_viewpoints(
        settings={
            "insight_notes_dir": "missing",
            "insight_lookback_days": 10,
            "info_half_life_days": 10.0,
        },
        as_of_date=pd.Timestamp("2026-03-12"),
        info_items=[
            InfoItem(
                date="2026-03-12",
                publish_datetime="2026-03-13T08:00:00",
                target_type="stock",
                target="AAA",
                horizon="mid",
                direction="bullish",
                info_type="news",
                title="future-published item",
            )
        ],
    )

    assert viewpoints == []


def test_attach_insight_memory_to_state_returns_empty_overlay_when_disabled() -> None:
    state = _base_composite_state()
    enriched = attach_insight_memory_to_state(
        state=state,
        settings={"enable_insight_memory": False},
        as_of_date=pd.Timestamp("2026-03-12"),
        info_items=[],
    )

    assert enriched.viewpoints == []
    assert enriched.theme_episodes == []
    assert enriched.stock_role_states == {}
    assert enriched.execution_plans == []


def test_build_theme_episodes_assigns_strengthening_and_fading_phases() -> None:
    state = CompositeState(
        market=_base_composite_state().market,
        cross_section=_base_composite_state().cross_section,
        sectors=[
            SectorForecastState("chips", 0.60, 0.67, 0.18, 0.22, 0.18),
            SectorForecastState("biotech", 0.57, 0.62, 0.14, 0.20, 0.22),
        ],
        stocks=[],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        mainlines=[
            MainlineState(
                name="chips",
                conviction=0.62,
                breadth=0.36,
                leadership=0.32,
                catalyst_strength=0.22,
                sectors=["chips"],
                representative_symbols=["AAA"],
            ),
            MainlineState(
                name="biotech",
                conviction=0.60,
                breadth=0.34,
                leadership=0.28,
                catalyst_strength=0.20,
                sectors=["biotech"],
                representative_symbols=["BBB"],
            ),
        ],
        market_info_state=InfoAggregateState(catalyst_strength=0.18, coverage_ratio=0.20),
        sector_info_states={
            "chips": InfoAggregateState(catalyst_strength=0.24, event_risk_level=0.12),
            "biotech": InfoAggregateState(catalyst_strength=0.18, event_risk_level=0.68),
        },
        viewpoints=[
            Viewpoint(
                target_type="sector",
                target="chips",
                theme="chips",
                direction="bullish",
                confidence=0.80,
                importance=0.75,
                horizon="mid",
                reason="flow improving",
                weight=0.35,
            )
        ],
    )

    episodes = build_theme_episodes(state=state)
    episode_map = {item.theme: item for item in episodes}

    assert episode_map["chips"].phase == "strengthening"
    assert episode_map["chips"].viewpoint_count == 1
    assert episode_map["biotech"].phase == "fading"
    assert episode_map["biotech"].event_risk >= 0.55


def test_build_stock_role_snapshots_assigns_role_buckets_and_downgrade() -> None:
    state = CompositeState(
        market=_base_composite_state().market,
        cross_section=_base_composite_state().cross_section,
        sectors=[SectorForecastState("chips", 0.60, 0.67, 0.18, 0.22, 0.18)],
        stocks=[
            StockForecastState("S1", "chips", 0.62, 0.66, 0.80, 0.70, 0.05, 0.92, alpha_score=0.90),
            StockForecastState("S2", "chips", 0.57, 0.61, 0.72, 0.55, 0.05, 0.90, alpha_score=0.75),
            StockForecastState("S3", "chips", 0.60, 0.56, 0.62, 0.53, 0.04, 0.88, alpha_score=0.60, up_2d_prob=0.61, up_3d_prob=0.62),
            StockForecastState("S4", "chips", 0.52, 0.57, 0.60, 0.57, 0.04, 0.88, alpha_score=0.54),
            StockForecastState("S5", "chips", 0.45, 0.40, 0.50, 0.45, 0.02, 0.60, alpha_score=0.40),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    episodes = [
        ThemeEpisode(
            theme="chips",
            phase="strengthening",
            conviction=0.68,
            sectors=["chips"],
            representative_symbols=[],
        )
    ]
    previous_roles = {
        "S5": StockRoleSnapshot(symbol="S5", theme="chips", role="core"),
    }

    roles = build_stock_role_snapshots(
        state=state,
        theme_episodes=episodes,
        previous_roles=previous_roles,
    )

    assert roles["S1"].role == "leader"
    assert roles["S2"].role == "core"
    assert roles["S3"].role == "rebound"
    assert roles["S4"].role == "follower"
    assert roles["S5"].role == "laggard"
    assert roles["S5"].role_downgrade is True


def test_build_execution_plans_uses_theme_phase_and_role_downgrade() -> None:
    state = CompositeState(
        market=_base_composite_state().market,
        cross_section=_base_composite_state().cross_section,
        sectors=[SectorForecastState("chips", 0.58, 0.62, 0.10, 0.18, 0.25)],
        stocks=[
            StockForecastState(
                "AAA",
                "chips",
                0.49,
                0.45,
                0.47,
                0.46,
                0.02,
                0.90,
                alpha_score=0.42,
                latest_close=10.0,
                horizon_forecasts={
                    "1d": HorizonForecast(
                        horizon_days=1,
                        label="1d",
                        price_low=9.8,
                        price_high=10.2,
                    )
                },
            )
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        market_info_state=InfoAggregateState(
            item_count=4,
            catalyst_strength=0.28,
            event_risk_level=0.14,
            negative_event_risk=0.10,
            coverage_confidence=0.75,
        ),
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="fading",
                conviction=0.46,
                event_risk=0.60,
                sectors=["chips"],
            )
        ],
        stock_role_states={
            "AAA": StockRoleSnapshot(
                symbol="AAA",
                theme="chips",
                role="laggard",
                previous_role="core",
                role_downgrade=True,
            )
        },
    )
    decision = PolicyDecision(
        target_exposure=0.05,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.5,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        symbol_target_weights={"AAA": 0.05},
        trim_candidate_scores={"AAA": 0.81},
        trim_candidate_ranks={"AAA": 1},
        trim_candidate_labels={"AAA": "reduce"},
    )

    plans = build_execution_plans(
        state=state,
        policy_decision=decision,
        current_weights={"AAA": 0.15},
        current_holding_days={"AAA": 7},
        symbol_names={"AAA": "Alpha"},
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.bias == "reduce"
    assert plan.buy_zone == "9.80 ~ 10.20"
    assert "fading" in plan.reduce_if
    assert "underperforming" in plan.exit_if
    assert "laggard in chips" in plan.reason
    assert "role downgrade active" in plan.reason
    assert "held 7d" in plan.reason
    assert plan.trim_rank == 1
    assert plan.trim_label == "reduce"
    assert plan.trim_score == 0.81


def test_snapshot_round_trip_preserves_insight_fields() -> None:
    state = CompositeState(
        market=_base_composite_state().market,
        cross_section=_base_composite_state().cross_section,
        sectors=[SectorForecastState("chips", 0.60, 0.66, 0.18, 0.22, 0.18)],
        stocks=[StockForecastState("AAA", "chips", 0.56, 0.62, 0.67, 0.58, 0.03, 0.88)],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        viewpoints=[
            Viewpoint(
                target_type="stock",
                target="AAA",
                theme="chips",
                direction="bullish",
                confidence=0.80,
                importance=0.75,
                horizon="mid",
                reason="trend intact",
                weight=0.30,
            )
        ],
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="strengthening",
                conviction=0.62,
                sectors=["chips"],
                representative_symbols=["AAA"],
            )
        ],
        stock_role_states={
            "AAA": StockRoleSnapshot(symbol="AAA", theme="chips", role="leader")
        },
        execution_plans=[
            ExecutionPlan(
                symbol="AAA",
                name="Alpha",
                theme="chips",
                role="leader",
                bias="hold",
                buy_zone="9.80 ~ 10.20",
                avoid_zone="gap-up chase",
                reduce_if="breakout fails",
                exit_if="trend breaks",
                reason="leader in chips",
                intraday_signal="hold_strong",
                intraday_timeframe="15m",
                intraday_data_date="2026-03-12",
                intraday_stop_price=9.92,
                intraday_take_profit_price=10.35,
                intraday_vwap_gap=0.004,
                intraday_drawdown_from_high=0.011,
                intraday_break_state="trend_intact",
                intraday_reason="15m trend intact",
            )
        ],
    )

    payload = serialize_composite_state(state)
    restored = decode_composite_state(payload)

    assert restored is not None
    assert len(restored.viewpoints) == 1
    assert restored.viewpoints[0].reason == "trend intact"
    assert restored.theme_episodes[0].phase == "strengthening"
    assert restored.stock_role_states["AAA"].role == "leader"
    assert restored.execution_plans[0].bias == "hold"
    assert restored.execution_plans[0].intraday_signal == "hold_strong"


def test_build_execution_plans_merges_intraday_exit_overlay_from_local_bars(tmp_path) -> None:
    intraday_dir = tmp_path / "intraday" / "15m"
    intraday_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-03-12 09:30:00", periods=12, freq="15min"),
            "open": [10.40, 10.32, 10.24, 10.18, 10.12, 10.08, 10.02, 9.98, 9.94, 9.90, 9.88, 9.86],
            "high": [10.42, 10.33, 10.26, 10.20, 10.14, 10.09, 10.03, 9.99, 9.95, 9.91, 9.89, 9.87],
            "low": [10.30, 10.22, 10.16, 10.10, 10.04, 9.99, 9.95, 9.90, 9.86, 9.83, 9.80, 9.78],
            "close": [10.31, 10.24, 10.17, 10.11, 10.05, 10.00, 9.96, 9.92, 9.88, 9.85, 9.82, 9.80],
            "volume": [1200, 1300, 1100, 1000, 950, 980, 1020, 990, 970, 940, 930, 920],
        }
    )
    frame.to_csv(intraday_dir / "AAA.csv", index=False)
    state = CompositeState(
        market=_base_composite_state().market,
        cross_section=_base_composite_state().cross_section,
        sectors=[SectorForecastState("chips", 0.58, 0.62, 0.10, 0.18, 0.25)],
        stocks=[
            StockForecastState(
                "AAA",
                "chips",
                0.49,
                0.45,
                0.47,
                0.46,
                0.02,
                0.90,
                alpha_score=0.42,
                latest_close=9.8,
            )
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    decision = PolicyDecision(
        target_exposure=0.05,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.5,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        symbol_target_weights={"AAA": 0.10},
    )

    plans = build_execution_plans(
        state=state,
        policy_decision=decision,
        current_weights={"AAA": 0.15},
        current_holding_days={"AAA": 4},
        symbol_names={"AAA": "Alpha"},
        settings={
            "enable_intraday_execution_overlay": True,
            "intraday_data_dir": str(tmp_path / "intraday"),
            "intraday_primary_timeframe": "15m",
            "intraday_secondary_timeframe": "",
            "intraday_lookback_bars": 32,
            "intraday_symbol_limit": 10,
        },
    )

    assert len(plans) == 1
    plan = plans[0]
    assert plan.intraday_signal == "exit_on_weak_rebound"
    assert plan.intraday_timeframe == "15m"
    assert plan.intraday_data_date == "2026-03-12"
    assert plan.intraday_stop_price == pytest.approx(9.8, rel=1e-6)
    assert "15m" in plan.reduce_if
    assert "exit" in plan.exit_if.lower()
    assert "breakdown" in plan.intraday_reason


def test_build_execution_plans_auto_fetches_intraday_and_writes_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetched_calls: list[tuple[str, str]] = []

    def _fake_fetch_tushare_intraday(*, symbol: str, timeframe: str, start, end, timeout: int):
        fetched_calls.append((symbol, timeframe))
        return pd.DataFrame(
            {
                "datetime": pd.date_range("2026-03-12 09:30:00", periods=12, freq="15min"),
                "open": [10.00, 10.02, 10.03, 10.05, 10.06, 10.08, 10.09, 10.10, 10.12, 10.13, 10.14, 10.15],
                "high": [10.03, 10.04, 10.06, 10.07, 10.08, 10.10, 10.11, 10.13, 10.14, 10.15, 10.17, 10.19],
                "low": [9.99, 10.00, 10.02, 10.03, 10.05, 10.07, 10.08, 10.09, 10.10, 10.12, 10.13, 10.14],
                "close": [10.02, 10.03, 10.05, 10.06, 10.08, 10.09, 10.10, 10.12, 10.13, 10.14, 10.16, 10.18],
                "volume": [1000, 1100, 1050, 1200, 1150, 1220, 1180, 1260, 1300, 1320, 1350, 1380],
                "amount": [10020, 11033, 10552, 12072, 11592, 12309, 11918, 12751, 13169, 13385, 13716, 14048],
                "symbol": ["AAA"] * 12,
            }
        )

    monkeypatch.setattr(
        "src.application.v2_intraday_execution_runtime._fetch_tushare_intraday",
        _fake_fetch_tushare_intraday,
    )

    state = CompositeState(
        market=_base_composite_state().market,
        cross_section=_base_composite_state().cross_section,
        sectors=[SectorForecastState("chips", 0.58, 0.62, 0.10, 0.18, 0.25)],
        stocks=[
            StockForecastState(
                "AAA",
                "chips",
                0.53,
                0.58,
                0.61,
                0.57,
                0.01,
                0.92,
                alpha_score=0.45,
                latest_close=10.18,
            )
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    decision = PolicyDecision(
        target_exposure=0.10,
        target_position_count=1,
        rebalance_now=True,
        rebalance_intensity=0.5,
        intraday_t_allowed=False,
        turnover_cap=0.20,
        symbol_target_weights={"AAA": 0.12},
    )

    plans = build_execution_plans(
        state=state,
        policy_decision=decision,
        current_weights={"AAA": 0.12},
        current_holding_days={"AAA": 3},
        symbol_names={"AAA": "Alpha"},
        settings={
            "enable_intraday_execution_overlay": True,
            "intraday_data_dir": str(tmp_path / "intraday"),
            "intraday_source": "tushare",
            "intraday_auto_fetch": True,
            "intraday_primary_timeframe": "15m",
            "intraday_secondary_timeframe": "",
            "intraday_lookback_bars": 32,
            "intraday_fetch_lookback_days": 3,
            "intraday_fetch_timeout_seconds": 6,
            "intraday_symbol_limit": 10,
        },
    )

    assert fetched_calls == [("AAA", "15m")]
    assert len(plans) == 1
    plan = plans[0]
    assert plan.intraday_signal == "hold_strong"
    assert plan.intraday_timeframe == "15m"
    assert "trend intact" in plan.intraday_reason
    assert 10.0 < float(plan.intraday_stop_price) < 10.18
    assert (tmp_path / "intraday" / "15m" / "AAA.csv").exists()


def test_write_and_report_view_models_surface_insight_outputs(tmp_path) -> None:
    state = CompositeState(
        market=_base_composite_state().market,
        cross_section=_base_composite_state().cross_section,
        sectors=[SectorForecastState("chips", 0.60, 0.66, 0.18, 0.22, 0.18)],
        stocks=[StockForecastState("AAA", "chips", 0.56, 0.62, 0.67, 0.58, 0.03, 0.88)],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        market_info_state=InfoAggregateState(
            item_count=4,
            catalyst_strength=0.28,
            event_risk_level=0.14,
            negative_event_risk=0.10,
            coverage_confidence=0.75,
        ),
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="strengthening",
                conviction=0.64,
                event_risk=0.18,
                sectors=["chips"],
            )
        ],
        stock_role_states={
            "AAA": StockRoleSnapshot(
                symbol="AAA",
                theme="chips",
                role="leader",
                previous_role="core",
                role_downgrade=False,
                note="leader inside chips.",
            )
        },
        execution_plans=[
            ExecutionPlan(
                symbol="AAA",
                name="Alpha",
                theme="chips",
                role="leader",
                bias="hold",
                buy_zone="9.80 ~ 10.20",
                avoid_zone="gap-up chase",
                reduce_if="breakout fails",
                exit_if="trend breaks",
                reason="leader in chips",
            )
        ],
    )
    result = DailyRunResult(
        snapshot=StrategySnapshot(
            strategy_id="swing_v2",
            universe_id="demo",
            feature_set_version="v2",
            market_model_id="m",
            sector_model_id="s",
            stock_model_id="st",
            cross_section_model_id="c",
            policy_version="p",
            execution_version="e",
            run_id="run_1",
        ),
        composite_state=state,
        policy_decision=PolicyDecision(
            target_exposure=0.60,
            target_position_count=1,
            rebalance_now=True,
            rebalance_intensity=0.20,
            intraday_t_allowed=False,
            turnover_cap=0.20,
            symbol_target_weights={"AAA": 0.60},
        ),
        trade_actions=[
            TradeAction(
                symbol="AAA",
                name="Alpha",
                action="BUY",
                current_weight=0.20,
                target_weight=0.60,
                delta_weight=0.40,
                note="scale in",
            )
        ],
        symbol_names={"AAA": "Alpha"},
        info_shadow_enabled=True,
        info_item_count=4,
        top_negative_info_events=[
            InfoSignalRecord(
                target="AAA",
                target_name="Alpha",
                title="earnings delay",
                info_type="announcement",
                direction="bearish",
                horizon="short",
                negative_event_risk=0.62,
            )
        ],
        top_positive_info_signals=[
            InfoSignalRecord(
                target="AAA",
                target_name="Alpha",
                title="new contract",
                info_type="announcement",
                direction="bullish",
                horizon="mid",
                score=0.58,
            )
        ],
        quant_info_divergence=[
            InfoDivergenceRecord(
                symbol="AAA",
                name="Alpha",
                quant_prob_20d=0.52,
                info_prob_20d=0.66,
                shadow_prob_20d=0.63,
                gap=0.11,
            )
        ],
        run_id="run_1",
    )
    daily_view_model = build_daily_report_view_model(result)

    assert daily_view_model.theme_episodes[0]["theme"] == "chips"
    assert daily_view_model.leader_candidates[0]["symbol"] == "AAA"
    assert daily_view_model.holding_role_changes[0]["symbol"] == "AAA"
    assert daily_view_model.execution_plans[0]["symbol"] == "AAA"
    assert daily_view_model.info_summary["item_count"] == 4
    assert daily_view_model.info_summary["market_info_state"]["catalyst_strength"] == 0.28
    assert daily_view_model.info_summary["top_negative_events"][0]["title"] == "earnings delay"

    insight_dir = tmp_path / "insight"
    payloads = {
        "insight_manifest": {
            "phase_counts": {"strengthening": 1},
            "role_counts": {"leader": 1},
            "role_downgrade_count": 0,
        },
        "viewpoints": [],
        "theme_episodes": [asdict(state.theme_episodes[0])],
        "stock_roles": [asdict(state.stock_role_states["AAA"])],
        "execution_plan": [asdict(state.execution_plans[0])],
    }
    written = write_insight_artifacts(base_dir=insight_dir, payloads=payloads)
    theme_payload = json.loads((insight_dir / "theme_episodes.json").read_text(encoding="utf-8"))
    role_payload = json.loads((insight_dir / "stock_roles.json").read_text(encoding="utf-8"))
    leader_payloads = build_leader_artifact_payloads(state=state, trajectory=None, limit=4)
    leader_manifest_path = insight_dir / "leader_manifest.json"
    leader_candidates_path = insight_dir / "leader_candidates.json"
    leader_manifest_path.write_text(
        json.dumps(leader_payloads["leader_manifest"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    leader_candidates_path.write_text(
        json.dumps({"items": leader_payloads["leader_candidates"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    assert isinstance(theme_payload.get("items"), list)
    assert isinstance(role_payload.get("items"), list)

    baseline = _make_backtest_summary(annual_return=0.24, excess_annual_return=0.12)
    calibrated = _make_backtest_summary(annual_return=0.26, excess_annual_return=0.14)
    learned = _make_backtest_summary(annual_return=0.28, excess_annual_return=0.16)
    research_view_model = build_research_report_view_model(
        strategy_id="swing_v2",
        baseline=baseline,
        calibration=V2CalibrationResult(
            best_policy=PolicySpec(),
            best_score=0.15,
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
                train_rows=32,
                train_r2_exposure=0.20,
                train_r2_positions=0.18,
                train_r2_turnover=0.15,
            ),
            baseline=baseline,
            learned=learned,
        ),
        artifacts={
            "run_id": "run_1",
            "release_gate_passed": "true",
            "insight_manifest": written["insight_manifest"],
            "theme_episodes": written["theme_episodes"],
            "stock_roles": written["stock_roles"],
            "leader_manifest": str(leader_manifest_path),
            "leader_candidates": str(leader_candidates_path),
        },
    )

    assert research_view_model.theme_lifecycle_summary["phase_counts"]["strengthening"] == 1
    assert research_view_model.role_distribution_summary["role_counts"]["leader"] == 1
    assert research_view_model.exit_contribution_summary["role_downgrade_count"] == 0
    assert research_view_model.leader_summary["top_candidates"][0]["symbol"] == "AAA"
