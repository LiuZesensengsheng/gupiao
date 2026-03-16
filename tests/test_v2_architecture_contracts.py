from __future__ import annotations

import json
import ast
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.application.v2_contracts import (
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    HorizonForecast,
    InfoAggregateState,
    InfoItem,
    LearnedPolicyModel,
    MarketForecastState,
    PolicyDecision,
    PolicySpec,
    PolicyInput,
    SectorForecastState,
    StockForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
from src.application.v2_policy_runtime import (
    apply_policy as apply_policy_runtime,
    build_trade_actions as build_trade_actions_runtime,
    policy_spec_from_model as policy_spec_from_model_runtime,
)
from src.application.v2_backtest_runtime import (
    execute_v2_backtest_trajectory as execute_v2_backtest_trajectory_runtime,
    run_v2_backtest_core as run_v2_backtest_core_runtime,
    simulate_execution_day as simulate_execution_day_runtime,
    to_v2_backtest_summary as to_v2_backtest_summary_runtime,
)
from src.application.v2_backtest_prepare_runtime import (
    load_or_build_v2_backtest_trajectory as load_or_build_v2_backtest_trajectory_runtime,
    prepare_v2_backtest_data as prepare_v2_backtest_data_runtime,
    split_research_trajectory as split_research_trajectory_runtime,
)
from src.application.v2_daily_runtime import (
    DailyCacheKeyDependencies,
    daily_result_cache_key as daily_result_cache_key_runtime,
    daily_result_cache_path as daily_result_cache_path_runtime,
    file_mtime_token as file_mtime_token_runtime,
)
from src.application.v2_feature_runtime import (
    make_forecast_backend as make_forecast_backend_runtime,
    tensorize_temporal_frame as tensorize_temporal_frame_runtime,
)
from src.application import v2_policy_feature_runtime as policy_feature_runtime
from src.application.v2_policy_learning_runtime import (
    baseline_only_calibration as baseline_only_calibration_runtime,
    calibrate_v2_policy as calibrate_v2_policy_runtime,
    learn_v2_policy_model as learn_v2_policy_model_runtime,
    placeholder_learning_result as placeholder_learning_result_runtime,
)
from src.application.v2_daily_state_runtime import (
    attach_daily_external_signal_overlay as attach_daily_external_signal_overlay_runtime,
    attach_daily_info_overlay as attach_daily_info_overlay_runtime,
    build_daily_composite_state as build_daily_composite_state_runtime,
    build_daily_universe_context as build_daily_universe_context_runtime,
)
from src.application.v2_state_build_runtime import (
    compose_state as compose_state_runtime,
    build_market_and_cross_section_from_prebuilt_frame as build_market_and_cross_section_from_prebuilt_frame_runtime,
    build_market_and_cross_section_states as build_market_and_cross_section_states_runtime,
    build_stock_states_from_panel_slice as build_stock_states_from_panel_slice_runtime,
    build_stock_states_from_rows as build_stock_states_from_rows_runtime,
)
from src.application.v2_frozen_forecast_runtime import (
    build_frozen_linear_forecast_bundle as build_frozen_linear_forecast_bundle_runtime,
    build_live_market_frame as build_live_market_frame_runtime,
    score_live_composite_state_from_frozen_bundle as score_live_composite_state_from_frozen_bundle_runtime,
)
from src.application.v2_learning_target_runtime import (
    derive_learning_targets as derive_learning_targets_runtime,
)
from src.application.v2_info_shadow_runtime import (
    build_info_shadow_variant as build_info_shadow_variant_runtime,
    enrich_state_with_info as enrich_state_with_info_runtime,
    fit_v2_info_shadow_models as fit_v2_info_shadow_models_runtime,
)
from src.reporting.report_state_runtime import (
    build_live_market_reporting_overlay as build_live_market_reporting_overlay_runtime,
    decorate_composite_state_for_reporting as decorate_composite_state_for_reporting_runtime,
    filter_state_for_recommendation_scope as filter_state_for_recommendation_scope_runtime,
)
from src.reporting.forecast_support import (
    alpha_score_components as alpha_score_components_runtime,
    build_horizon_forecasts as build_horizon_forecasts_runtime,
    build_market_sentiment_state as build_market_sentiment_state_runtime,
)
from src.reporting.reason_bundles import stock_reason_bundle as stock_reason_bundle_runtime
from src.review_analytics.info_shadow_report import build_info_shadow_report as build_info_shadow_report_runtime
from src.review_analytics.info_manifest import build_info_manifest_payload as build_info_manifest_payload_runtime
from src.review_analytics.prediction_review import load_prediction_review_context as load_prediction_review_context_runtime
from src.application.v2_daily_snapshot_runtime import (
    build_strategy_snapshot as build_strategy_snapshot_runtime,
    resolve_manifest_path,
)
from src.application.v2_artifact_runtime import (
    load_policy_model_from_path as load_policy_model_from_path_runtime,
)
from src.application.v2_research_publish_runtime import publish_research_artifacts as publish_research_artifacts_runtime
from src.application.v2_runtime_settings import (
    load_v2_runtime_settings as load_v2_runtime_settings_runtime,
    resolve_v2_universe_settings as resolve_v2_universe_settings_runtime,
)
from src.artifact_registry.v2_registry import publish_v2_research_artifacts
from src.contracts.artifacts import (
    CURRENT_ARTIFACT_VERSION,
    ArtifactValidationError,
    DatasetManifest,
    LearnedPolicyArtifact,
    ResearchManifest,
)
from src.contracts.runtime import DailyRunOptions, ResearchMatrixOptions, ResearchRunOptions
from src.interfaces.cli.run_v2_cli import build_parser
from src.reporting.view_models import build_daily_report_view_model, build_research_report_view_model
from src.review_analytics.summaries import summarize_daily_run
from src.workflows.daily_workflow import run_daily_v2_live_impl as run_daily_v2_live_runtime
from src.workflows.research_workflow import run_v2_research_workflow_impl as run_v2_research_workflow_runtime


def _make_backtest(total_return: float = 0.20, annual_return: float = 0.18) -> V2BacktestSummary:
    return V2BacktestSummary(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_days=200,
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=-0.10,
        avg_turnover=0.22,
        total_cost=0.02,
    )


def test_runtime_options_are_built_from_cli_namespace() -> None:
    parser = build_parser()

    research_args = parser.parse_args(
        [
            "research-run",
            "--strategy",
            "alpha_v2",
            "--dynamic-universe",
            "--generator-target-size",
            "120",
            "--light",
            "--forecast-backend",
            "linear",
        ]
    )
    research_options = ResearchRunOptions.from_namespace(research_args)
    assert research_options.strategy_id == "alpha_v2"
    assert research_options.dynamic_universe is True
    assert research_options.generator_target_size == 120
    assert research_options.training_window_days == 480
    assert research_options.skip_calibration is True
    assert research_options.skip_learning is True

    daily_args = parser.parse_args(
        [
            "daily-run",
            "--strategy",
            "alpha_v2",
            "--snapshot-path",
            "artifacts/v2/alpha_v2/latest_research_manifest.json",
            "--allow-retrain",
            "--disable-learned-policy",
        ]
    )
    daily_options = DailyRunOptions.from_namespace(daily_args)
    assert daily_options.strategy_id == "alpha_v2"
    assert daily_options.snapshot_path
    assert daily_options.allow_retrain is True
    assert daily_options.disable_learned_policy is True

    matrix_args = parser.parse_args(
        [
            "research-matrix",
            "--strategy",
            "alpha_v2",
            "--tiers",
            "favorites_16",
            "generated_80",
        ]
    )
    matrix_options = ResearchMatrixOptions.from_namespace(matrix_args)
    assert matrix_options.strategy_id == "alpha_v2"
    assert matrix_options.training_window_days == 480
    assert matrix_options.universe_tiers == ("favorites_16", "generated_80")


def test_backtest_core_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.application.v2_backtest_runtime import BacktestCoreDependencies
    import src.application.v2_services as legacy

    trajectory = SimpleNamespace(name="trajectory")
    summary = _make_backtest(total_return=0.18, annual_return=0.16)
    learning_rows = [{"target_exposure": 0.75}]
    deps = BacktestCoreDependencies(
        load_or_build_v2_backtest_trajectory=lambda **_: trajectory,
        empty_v2_backtest_result=lambda: (_make_backtest(0.0, 0.0), []),
        execute_v2_backtest_trajectory=lambda *args, **kwargs: (summary, learning_rows),
    )

    monkeypatch.setattr(legacy, "_backtest_core_dependencies", lambda: deps)

    runtime_result = run_v2_backtest_core_runtime(
        strategy_id="alpha_v2",
        config_path="config/api.json",
        forecast_backend="linear",
        deps=deps,
    )
    facade_result = legacy._run_v2_backtest_core(
        strategy_id="alpha_v2",
        config_path="config/api.json",
        forecast_backend="linear",
    )

    assert facade_result == runtime_result


def test_backtest_summary_runtime_keeps_facade_contract() -> None:
    import pandas as pd
    import src.application.v2_services as legacy

    dates = [pd.Timestamp("2026-03-14"), pd.Timestamp("2026-03-17"), pd.Timestamp("2026-03-18")]
    kwargs = {
        "returns": [0.01, -0.005, 0.012],
        "benchmark_returns": [0.003, -0.002, 0.004],
        "turnovers": [0.10, 0.15, 0.08],
        "costs": [0.001, 0.0015, 0.0008],
        "gross_returns": [0.011, -0.004, 0.013],
        "fill_ratios": [0.95, 0.90, 0.92],
        "slippage_bps": [4.0, 6.0, 5.0],
        "rank_ics": [0.08, 0.04, 0.09],
        "top_decile_returns": [0.02, 0.01, 0.03],
        "top_bottom_spreads": [0.04, 0.03, 0.05],
        "top_k_hit_rates": [0.60, 0.55, 0.66],
        "horizon_metrics": {
            "5d": {
                "rank_ic": [0.06, 0.07],
                "top_decile_return": [0.012, 0.015],
                "top_bottom_spread": [0.025, 0.030],
                "top_k_hit_rate": [0.58, 0.61],
            }
        },
        "dates": dates,
    }

    runtime_summary = to_v2_backtest_summary_runtime(**kwargs)
    facade_summary = legacy._to_v2_backtest_summary(**kwargs)

    assert facade_summary == runtime_summary


def test_report_state_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    import src.application.v2_services as legacy

    stock_a = StockForecastState("AAA", "科技", 0.60, 0.62, 0.65, 0.58, 0.10, 0.84, latest_close=12.3)
    stock_b = StockForecastState("BBB", "消费", 0.55, 0.57, 0.60, 0.54, 0.08, 0.80, latest_close=9.8)
    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-15",
            up_1d_prob=0.57,
            up_5d_prob=0.59,
            up_20d_prob=0.61,
            trend_state="trend",
            drawdown_risk=0.15,
            volatility_regime="normal",
            liquidity_stress=0.11,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-15",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.03,
            fund_flow_strength=0.04,
            margin_risk_on_score=0.05,
            breadth_strength=0.12,
            leader_participation=0.16,
            weak_stock_ratio=0.20,
        ),
        sectors=[
            SectorForecastState("科技", 0.60, 0.63, 0.10, 0.18, 0.14),
            SectorForecastState("消费", 0.56, 0.58, 0.08, 0.16, 0.12),
        ],
        stocks=[stock_a, stock_b],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        stock_info_states={
            "AAA": InfoAggregateState(catalyst_strength=0.60),
            "BBB": InfoAggregateState(catalyst_strength=0.30),
        },
    )
    policy = PolicyDecision(
        target_exposure=0.85,
        target_position_count=4,
        rebalance_now=True,
        rebalance_intensity=0.8,
        intraday_t_allowed=False,
        turnover_cap=0.25,
        sector_budgets={"科技": 0.25, "消费": 0.20},
        desired_sector_budgets={"科技": 0.30, "消费": 0.22},
        symbol_target_weights={"AAA": 0.18, "BBB": 0.12},
        desired_symbol_target_weights={"AAA": 0.20, "BBB": 0.13},
    )

    deps = replace(
        legacy._report_state_runtime_dependencies(),
        build_horizon_forecasts=lambda **kwargs: {
            "1d": {"up_prob": float(kwargs["horizon_probs"]["1d"]), "tag": kwargs.get("tradability_status", "market")}
        },
        profile_from_horizon_map=lambda horizon_map, key: horizon_map.get(key, {"picked": key}),
        build_market_sentiment_state=lambda **_: SimpleNamespace(score=77.0, stage="risk_on"),
        candidate_stocks_from_state=lambda state: [state.stocks[1], state.stocks[0]],
        stock_reason_bundle=lambda **kwargs: (
            [f"pick-{kwargs['stock'].symbol}"],
            [f"rank-{kwargs['rank']}"],
            [f"risk-{kwargs['stock'].symbol}"],
            f"invalid-{kwargs['stock'].symbol}",
            f"action-{kwargs['stock'].symbol}",
            f"weight-{kwargs['stock'].symbol}",
            f"blocked-{kwargs['stock'].symbol}",
        ),
    )
    monkeypatch.setattr(legacy, "_report_state_runtime_dependencies", lambda: deps)

    runtime_state = decorate_composite_state_for_reporting_runtime(
        state=state,
        policy=policy,
        calibration_priors={"20d": {"rank_ic": 0.1}},
        deps=deps,
    )
    facade_state = legacy._decorate_composite_state_for_reporting(
        state=state,
        policy=policy,
        calibration_priors={"20d": {"rank_ic": 0.1}},
    )

    assert facade_state == runtime_state
    assert [stock.symbol for stock in runtime_state.stocks] == ["BBB", "AAA"]

    scope_state = CompositeState(
        market=state.market,
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=[
            StockForecastState("600000.SH", "绉戞妧", 0.60, 0.62, 0.65, 0.58, 0.10, 0.84),
            StockForecastState("300001.SZ", "绉戞妧", 0.58, 0.60, 0.63, 0.56, 0.08, 0.80),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        stock_info_states={
            "600000.SH": InfoAggregateState(item_count=1),
            "300001.SZ": InfoAggregateState(item_count=1),
        },
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_candidate_selection_state_external",
        lambda **kwargs: CandidateSelectionState(
            selection_notes=["seed"],
            shortlisted_sectors=["绉戞妧"],
            total_scored=len(kwargs["stocks"]),
            shortlist_size=len(kwargs["stocks"]),
        ),
    )
    deps = replace(
        deps,
        build_candidate_selection_state=lambda **kwargs: CandidateSelectionState(
            selection_notes=["seed"],
            shortlisted_sectors=["科技"],
            total_scored=len(kwargs["stocks"]),
            shortlist_size=len(kwargs["stocks"]),
        ),
    )
    monkeypatch.setattr(legacy, "_report_state_runtime_dependencies", lambda: deps)
    runtime_filtered = filter_state_for_recommendation_scope_runtime(
        state=scope_state,
        main_board_only=True,
        deps=deps,
    )
    facade_filtered = legacy._filter_state_for_recommendation_scope(
        state=scope_state,
        main_board_only=True,
    )
    assert facade_filtered == runtime_filtered
    assert [stock.symbol for stock in runtime_filtered.stocks] == ["600000.SH"]

    overlay_state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-15",
            up_1d_prob=0.57,
            up_5d_prob=0.59,
            up_20d_prob=0.61,
            trend_state="trend",
            drawdown_risk=0.15,
            volatility_regime="normal",
            liquidity_stress=0.11,
            up_2d_prob=0.58,
            up_3d_prob=0.59,
            horizon_forecasts={"1d": {"picked": "1d"}, "20d": {"picked": "20d"}},
        ),
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=state.stocks,
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_market_and_cross_section_states",
        lambda **kwargs: (
            MarketForecastState(
                as_of_date="2026-03-15",
                up_1d_prob=float(kwargs["market_short_prob"]),
                up_5d_prob=float(kwargs["market_five_prob"]),
                up_20d_prob=float(kwargs["market_mid_prob"]),
                trend_state="trend",
                drawdown_risk=0.12,
                volatility_regime="normal",
                liquidity_stress=0.09,
            ),
            CrossSectionForecastState(
                as_of_date="2026-03-15",
                large_vs_small_bias=0.01,
                growth_vs_value_bias=0.02,
                fund_flow_strength=0.03,
                margin_risk_on_score=0.04,
                breadth_strength=0.15,
                leader_participation=0.16,
                weak_stock_ratio=0.18,
            ),
        ),
    )
    deps = replace(
        deps,
        build_market_and_cross_section_states=lambda **kwargs: (
            MarketForecastState(
                as_of_date="2026-03-15",
                up_1d_prob=float(kwargs["market_short_prob"]),
                up_5d_prob=float(kwargs["market_five_prob"]),
                up_20d_prob=float(kwargs["market_mid_prob"]),
                trend_state="trend",
                drawdown_risk=0.12,
                volatility_regime="normal",
                liquidity_stress=0.09,
            ),
            CrossSectionForecastState(
                as_of_date="2026-03-15",
                large_vs_small_bias=0.01,
                growth_vs_value_bias=0.02,
                fund_flow_strength=0.03,
                margin_risk_on_score=0.04,
                breadth_strength=0.15,
                leader_participation=0.16,
                weak_stock_ratio=0.18,
            ),
        ),
    )
    monkeypatch.setattr(legacy, "_report_state_runtime_dependencies", lambda: deps)
    runtime_overlay = build_live_market_reporting_overlay_runtime(
        settings={
            "source": "local",
            "data_dir": "data",
            "start": "2026-03-01",
            "end": "2026-03-15",
            "use_margin_features": False,
            "margin_market_file": "margin.csv",
            "use_us_index_context": False,
            "us_index_source": "akshare",
        },
        universe_ctx=SimpleNamespace(market_security=SimpleNamespace(symbol="000300.SH")),
        state=overlay_state,
        deps=deps,
    )
    facade_overlay = legacy._build_live_market_reporting_overlay(
        settings={
            "source": "local",
            "data_dir": "data",
            "start": "2026-03-01",
            "end": "2026-03-15",
            "use_margin_features": False,
            "margin_market_file": "margin.csv",
            "use_us_index_context": False,
            "us_index_source": "akshare",
        },
        universe_ctx=SimpleNamespace(market_security=SimpleNamespace(symbol="000300.SH")),
        state=overlay_state,
    )
    assert facade_overlay == runtime_overlay


def test_backtest_prepare_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dataclasses import replace
    import pandas as pd
    import src.application.v2_services as legacy

    watchlist = tmp_path / "watchlist.json"
    universe_file = tmp_path / "universe.json"
    margin_market = tmp_path / "margin_market.csv"
    margin_stock = tmp_path / "margin_stock.csv"
    for path in (watchlist, universe_file, margin_market, margin_stock):
        path.write_text("{}", encoding="utf-8")

    settings = {
        "config_path": "config/api.json",
        "source": "local",
        "data_dir": str(tmp_path / "data"),
        "watchlist": str(watchlist),
        "universe_file": str(universe_file),
        "universe_limit": 5,
        "universe_tier": "",
        "source_universe_manifest_path": str(universe_file),
        "start": "2024-01-01",
        "end": "2024-01-31",
        "min_train_days": 2,
        "use_margin_features": False,
        "margin_market_file": str(margin_market),
        "margin_stock_file": str(margin_stock),
        "use_us_index_context": False,
        "us_index_source": "akshare",
    }
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    deps = replace(
        legacy._backtest_prepare_dependencies(),
        load_v2_runtime_settings=lambda **_: dict(settings),
        resolve_v2_universe_settings=lambda settings, cache_root: dict(settings),
        load_pickle_cache=lambda *_: None,
        store_pickle_cache=lambda *_: None,
        emit_progress=lambda *_: None,
        load_watchlist=lambda _: (SimpleNamespace(symbol="MKT"), None, None),
        build_candidate_universe=lambda **_: SimpleNamespace(rows=[SimpleNamespace(symbol="AAA"), SimpleNamespace(symbol="BBB")]),
        load_symbol_daily=lambda **_: pd.DataFrame({"date": dates}),
        make_market_feature_frame=lambda _: pd.DataFrame(
            {
                "date": dates,
                "mkt_feature": [0.1, 0.2, 0.3, 0.4, 0.5],
                "mkt_target_1d_up": [1, 0, 1, 1, 0],
                "mkt_target_2d_up": [1, 1, 0, 1, 0],
                "mkt_target_3d_up": [1, 1, 1, 0, 0],
                "mkt_target_5d_up": [1, 0, 1, 0, 1],
                "mkt_target_20d_up": [1, 1, 1, 1, 0],
            }
        ),
        build_market_context_features=lambda **_: SimpleNamespace(frame=pd.DataFrame({"date": dates}), feature_columns=[]),
        build_stock_panel_dataset=lambda **_: SimpleNamespace(
            frame=pd.DataFrame(
                {
                    "date": list(dates) * 2,
                    "symbol": ["AAA"] * len(dates) + ["BBB"] * len(dates),
                    "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6],
                }
            ),
            feature_columns=["feature_a"],
        ),
        market_feature_columns=["mkt_feature"],
    )
    monkeypatch.setattr(legacy, "_backtest_prepare_dependencies", lambda: deps)

    runtime_prepared = prepare_v2_backtest_data_runtime(
        config_path="config/api.json",
        cache_root=str(tmp_path),
        prepared_dataclass=legacy._PreparedV2BacktestData,
        deps=deps,
    )
    facade_prepared = legacy._prepare_v2_backtest_data(
        config_path="config/api.json",
        cache_root=str(tmp_path),
    )
    assert facade_prepared is not None
    assert runtime_prepared is not None
    assert facade_prepared.dates == runtime_prepared.dates
    assert facade_prepared.feature_cols == runtime_prepared.feature_cols
    assert facade_prepared.market_feature_cols == runtime_prepared.market_feature_cols
    assert facade_prepared.settings == runtime_prepared.settings
    assert facade_prepared.panel.equals(runtime_prepared.panel)
    assert facade_prepared.market_valid.equals(runtime_prepared.market_valid)


def test_backtest_trajectory_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dataclasses import replace
    import src.application.v2_services as legacy

    settings = {
        "config_path": "config/api.json",
        "source": "local",
        "universe_file": str(tmp_path / "universe.json"),
        "universe_limit": 5,
        "universe_tier": "",
        "use_us_index_context": False,
        "us_index_source": "akshare",
        "use_us_sector_etf_context": False,
        "use_cn_etf_context": False,
        "cn_etf_source": "akshare",
        "dynamic_universe_enabled": False,
        "generator_target_size": 5,
        "generator_coarse_size": 10,
        "generator_theme_aware": True,
        "generator_use_concepts": True,
    }
    prepared_sentinel = SimpleNamespace(prepared=True)
    trajectory_sentinel = SimpleNamespace(steps=["ok"])
    deps = replace(
        legacy._backtest_prepare_dependencies(),
        load_v2_runtime_settings=lambda **_: dict(settings),
        resolve_v2_universe_settings=lambda settings, cache_root: dict(settings),
        load_pickle_cache=lambda *_: None,
        store_pickle_cache=lambda *_: None,
        emit_progress=lambda *_: None,
        make_forecast_backend=lambda name: SimpleNamespace(name=str(name or "linear")),
        prepare_v2_backtest_data=lambda **_: prepared_sentinel,
        build_v2_backtest_trajectory_from_prepared=lambda *_, **__: trajectory_sentinel,
    )
    monkeypatch.setattr(legacy, "_backtest_prepare_dependencies", lambda: deps)

    runtime_trajectory = load_or_build_v2_backtest_trajectory_runtime(
        config_path="config/api.json",
        cache_root=str(tmp_path),
        refresh_cache=True,
        deps=deps,
    )
    facade_trajectory = legacy._load_or_build_v2_backtest_trajectory(
        config_path="config/api.json",
        cache_root=str(tmp_path),
        refresh_cache=True,
    )
    assert facade_trajectory == runtime_trajectory


def test_policy_learning_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import src.application.v2_services as legacy

    baseline = _make_backtest(0.18, 0.16)
    calibrated = _make_backtest(0.20, 0.18)
    learned = _make_backtest(0.19, 0.17)

    def fake_backtest(**kwargs: object) -> V2BacktestSummary:
        policy_spec = kwargs.get("policy_spec")
        learned_policy = kwargs.get("learned_policy")
        if isinstance(learned_policy, LearnedPolicyModel):
            return learned
        if isinstance(policy_spec, PolicySpec):
            return calibrated
        return baseline

    learning_rows = [
        {
            name: 0.1
            for name in legacy._policy_feature_names()
        }
    ]
    learning_rows[0].update(
        {
            "target_exposure": 0.62,
            "target_positions": 3.0,
            "target_turnover": 0.24,
            "sample_weight": 1.0,
        }
    )

    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", fake_backtest)
    monkeypatch.setattr("src.application.v2_services._run_v2_backtest_core", lambda **_: (baseline, learning_rows))

    runtime_calibration = calibrate_v2_policy_runtime(
        strategy_id="swing_v2",
        baseline=baseline,
        trajectory=object(),
        cache_root=str(tmp_path),
        deps=legacy._policy_learning_dependencies(),
    )
    facade_calibration = legacy.calibrate_v2_policy(
        strategy_id="swing_v2",
        baseline=baseline,
        trajectory=object(),
        cache_root=str(tmp_path),
    )
    assert facade_calibration == runtime_calibration

    runtime_learning = learn_v2_policy_model_runtime(
        strategy_id="swing_v2",
        baseline=baseline,
        trajectory=object(),
        fit_trajectory=object(),
        evaluation_trajectory=object(),
        cache_root=str(tmp_path),
        deps=legacy._policy_learning_dependencies(),
    )
    facade_learning = legacy.learn_v2_policy_model(
        strategy_id="swing_v2",
        baseline=baseline,
        trajectory=object(),
        fit_trajectory=object(),
        evaluation_trajectory=object(),
        cache_root=str(tmp_path),
    )
    assert facade_learning == runtime_learning

    assert legacy._baseline_only_calibration(baseline) == baseline_only_calibration_runtime(
        baseline,
        deps=legacy._policy_learning_dependencies(),
    )
    assert legacy._placeholder_learning_result(baseline) == placeholder_learning_result_runtime(
        baseline,
        deps=legacy._policy_learning_dependencies(),
    )


def test_artifact_contract_rejects_unsupported_version(tmp_path: Path) -> None:
    payload_path = tmp_path / "dataset_manifest.json"
    payload_path.write_text(
        json.dumps(
            {
                "artifact_type": "dataset_manifest",
                "artifact_version": CURRENT_ARTIFACT_VERSION + 1,
                "strategy_id": "swing_v2",
                "universe_file": "config/universe_smoke_5.json",
                "universe_limit": 5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactValidationError, match="unsupported artifact_version"):
        DatasetManifest.from_path(payload_path)


def test_publish_artifacts_include_contract_metadata(tmp_path: Path) -> None:
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
        publish_forecast_models=False,
        settings={
            "config_path": "config/api.json",
            "source": "local",
            "watchlist": "config/watchlist.json",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 5,
            "symbols": ["000001.SZ", "000002.SZ"],
            "symbol_count": 2,
            "start": "2024-01-01",
            "end": "2024-12-31",
        },
        baseline=baseline,
        calibration=calibration,
        learning=learning_result,
    )

    dataset_manifest = DatasetManifest.from_path(paths["dataset_manifest"])
    research_manifest = ResearchManifest.from_path(paths["research_manifest"])
    learned_policy = LearnedPolicyArtifact.from_path(paths["learned_policy_model"])

    assert dataset_manifest.artifact_version == CURRENT_ARTIFACT_VERSION
    assert research_manifest.artifact_version == CURRENT_ARTIFACT_VERSION
    assert learned_policy.artifact_version == CURRENT_ARTIFACT_VERSION

    dataset_payload = json.loads(Path(paths["dataset_manifest"]).read_text(encoding="utf-8"))
    manifest_payload = json.loads(Path(paths["research_manifest"]).read_text(encoding="utf-8"))
    learning_payload = json.loads(Path(paths["learned_policy_model"]).read_text(encoding="utf-8"))

    assert dataset_payload["artifact_type"] == "dataset_manifest"
    assert manifest_payload["artifact_type"] == "research_manifest"
    assert learning_payload["artifact_type"] == "learned_policy_model"


def test_research_publish_runtime_keeps_facade_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return cls(2026, 3, 14, 9, 30, 15, tzinfo=tz)

    import src.application.v2_research_publish_runtime as publish_runtime_module
    from src.application import v2_services as legacy

    monkeypatch.setattr(publish_runtime_module, "datetime", _FrozenDatetime)

    baseline = _make_backtest(0.20, 0.18)
    calibrated = _make_backtest(0.22, 0.20)
    learned = _make_backtest(0.24, 0.22)
    learning_result = V2PolicyLearningResult(
        model=LearnedPolicyModel(
            feature_names=["x1"],
            exposure_intercept=0.5,
            exposure_coef=[0.1],
            position_intercept=2.0,
            position_coef=[0.1],
            turnover_intercept=0.2,
            turnover_coef=[0.05],
            train_rows=64,
            train_r2_exposure=0.2,
            train_r2_positions=0.18,
            train_r2_turnover=0.12,
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
    settings = {
        "config_path": "config/api.json",
        "source": "local",
        "watchlist": "config/watchlist.json",
        "universe_file": "config/universe_smoke_5.json",
        "universe_limit": 5,
        "symbols": ["000001.SZ", "000002.SZ"],
        "symbol_count": 2,
        "start": "2024-01-01",
        "end": "2024-12-31",
    }

    runtime_paths = publish_research_artifacts_runtime(
        dependencies=legacy._research_publish_dependencies(),
        strategy_id="swing_v2",
        artifact_root=str(tmp_path),
        publish_forecast_models=False,
        settings=settings,
        baseline=baseline,
        calibration=calibration,
        learning=learning_result,
    )
    facade_paths = legacy._publish_v2_research_artifacts_impl(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path),
        publish_forecast_models=False,
        settings=settings,
        baseline=baseline,
        calibration=calibration,
        learning=learning_result,
    )

    assert facade_paths == runtime_paths
    assert Path(facade_paths["research_manifest"]).exists()


def test_research_workflow_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    trajectory_sentinel = object()
    train_sentinel = object()
    validation_sentinel = object()
    holdout_sentinel = object()

    from src.application import v2_services as legacy

    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", lambda **_: trajectory_sentinel)
    monkeypatch.setattr(
        "src.application.v2_services._split_research_trajectory",
        lambda trajectory, *args, **kwargs: (train_sentinel, validation_sentinel, holdout_sentinel),
    )
    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", lambda **_: baseline)
    monkeypatch.setattr("src.application.v2_services.calibrate_v2_policy", lambda **_: (_ for _ in ()).throw(AssertionError("skip")))
    monkeypatch.setattr("src.application.v2_services.learn_v2_policy_model", lambda **_: (_ for _ in ()).throw(AssertionError("skip")))

    runtime_result = run_v2_research_workflow_runtime(
        dependencies=legacy._research_workflow_dependencies(),
        strategy_id="swing_v2",
        skip_calibration=True,
        skip_learning=True,
        cache_root=str(tmp_path),
    )
    facade_result = legacy._run_v2_research_workflow_impl(
        strategy_id="swing_v2",
        skip_calibration=True,
        skip_learning=True,
        cache_root=str(tmp_path),
    )

    assert facade_result == runtime_result


def test_daily_workflow_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.application import v2_services as legacy
    from src.application.v2_contracts import CompositeState, CrossSectionForecastState, DailyRunResult, MarketForecastState, PolicyDecision

    snapshot = build_strategy_snapshot_runtime(strategy_id="swing_v2", universe_id="demo")
    composite_state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-01",
            up_1d_prob=0.5,
            up_5d_prob=0.5,
            up_20d_prob=0.5,
            trend_state="trend",
            drawdown_risk=0.1,
            volatility_regime="normal",
            liquidity_stress=0.1,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-01",
            large_vs_small_bias=0.0,
            growth_vs_value_bias=0.0,
            fund_flow_strength=0.0,
            margin_risk_on_score=0.0,
            breadth_strength=0.0,
            leader_participation=0.0,
            weak_stock_ratio=0.0,
        ),
        sectors=[],
        stocks=[],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    decision = PolicyDecision(
        target_exposure=0.8,
        target_position_count=3,
        rebalance_now=True,
        rebalance_intensity=0.5,
        intraday_t_allowed=False,
        turnover_cap=0.2,
    )
    settings = {"event_risk_cutoff": 0.55, "catalyst_boost_cap": 0.12, "flow_exposure_cap": 0.08}

    monkeypatch.setattr(
        "src.application.v2_services._build_daily_snapshot_context",
        lambda **_: SimpleNamespace(
            settings=settings,
            manifest={},
            manifest_path=tmp_path / "research_manifest.json",
            snapshot=snapshot,
            resolved_run_id="20260314_120000",
        ),
    )
    monkeypatch.setattr("src.application.v2_services._daily_result_cache_key", lambda **_: "cache_key")
    monkeypatch.setattr("src.application.v2_services._daily_result_cache_path", lambda **_: tmp_path / "daily.pkl")
    monkeypatch.setattr("src.application.v2_services._load_daily_cached_result", lambda **_: None)
    monkeypatch.setattr(
        "src.application.v2_services._build_daily_universe_context",
        lambda settings: SimpleNamespace(current_holdings=[], stocks=[], sector_map={}),
    )
    monkeypatch.setattr("src.application.v2_services._build_daily_composite_state", lambda **_: (composite_state, []))
    monkeypatch.setattr("src.application.v2_services._build_daily_symbol_names", lambda **_: {})
    monkeypatch.setattr(
        "src.application.v2_services._attach_daily_info_overlay",
        lambda **_: (composite_state, "", "", False, 0, [], [], [], []),
    )
    monkeypatch.setattr(
        "src.application.v2_services._attach_daily_external_signal_overlay",
        lambda **_: (composite_state, "", "v1", False, {}, {}),
    )
    monkeypatch.setattr("src.application.v2_services._filter_state_for_recommendation_scope", lambda **kwargs: kwargs["state"])
    monkeypatch.setattr("src.application.v2_services._resolve_daily_policy_model", lambda **_: None)
    monkeypatch.setattr("src.application.v2_services.apply_policy", lambda *_, **__: decision)
    monkeypatch.setattr("src.application.v2_services.build_trade_actions", lambda **_: [])
    monkeypatch.setattr("src.application.v2_services._load_prediction_review_context", lambda **_: (None, {}))
    monkeypatch.setattr("src.application.v2_services._build_live_market_reporting_overlay", lambda **_: (None, None))
    monkeypatch.setattr("src.application.v2_services._decorate_composite_state_for_reporting", lambda **kwargs: kwargs["state"])
    monkeypatch.setattr("src.application.v2_services.remember_daily_run", lambda **kwargs: kwargs["result"])

    runtime_result = run_daily_v2_live_runtime(
        dependencies=legacy._daily_workflow_dependencies(),
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache"),
        allow_retrain=True,
    )
    facade_result = legacy._run_daily_v2_live_impl(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts"),
        cache_root=str(tmp_path / "cache"),
        allow_retrain=True,
    )

    assert isinstance(runtime_result, DailyRunResult)
    assert facade_result == runtime_result


def test_daily_state_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dataclasses import replace

    import src.application.v2_services as legacy

    market = MarketForecastState(
        as_of_date="2026-03-01",
        up_1d_prob=0.56,
        up_5d_prob=0.58,
        up_20d_prob=0.60,
        trend_state="trend",
        drawdown_risk=0.18,
        volatility_regime="normal",
        liquidity_stress=0.16,
    )
    cross = CrossSectionForecastState(
        as_of_date="2026-03-01",
        large_vs_small_bias=0.02,
        growth_vs_value_bias=0.01,
        fund_flow_strength=0.04,
        margin_risk_on_score=0.10,
        breadth_strength=0.12,
        leader_participation=0.15,
        weak_stock_ratio=0.20,
    )
    composite_state = CompositeState(
        market=market,
        cross_section=cross,
        sectors=[],
        stocks=[],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    snapshot = build_strategy_snapshot_runtime(strategy_id="swing_v2", universe_id="demo")
    settings = {
        "watchlist": str(tmp_path / "watchlist.json"),
        "source": "local",
        "data_dir": str(tmp_path / "data"),
        "universe_file": str(tmp_path / "universe.json"),
        "universe_limit": 5,
        "use_info_fusion": True,
        "info_half_life_days": 10.0,
        "external_signals": True,
    }
    frozen_state_path = tmp_path / "frozen_state.json"
    frozen_state_path.write_text(
        json.dumps({"composite_state": legacy._serialize_composite_state(composite_state)}, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "frozen_daily_state": "frozen_state.json",
    }
    symbol_names = {"AAA": "Alpha"}
    info_items = [
        InfoItem(
            date="2026-03-01",
            target_type="stock",
            target="AAA",
            horizon="20d",
            direction="up",
            info_type="news",
            title="positive",
        )
    ]
    enriched_state = CompositeState(
        market=market,
        cross_section=cross,
        sectors=[],
        stocks=[],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        stock_info_states={"AAA": InfoAggregateState(item_count=1)},
    )

    monkeypatch.setattr("src.application.v2_services.load_watchlist", lambda _: (SimpleNamespace(symbol="MKT"), [], {"AAA": "科技"}))
    monkeypatch.setattr(
        "src.application.v2_services.build_candidate_universe",
        lambda **_: SimpleNamespace(rows=[SimpleNamespace(symbol="AAA", sector="科技")]),
    )
    monkeypatch.setattr("src.application.v2_services._load_v2_info_items_for_date", lambda **_: ("info.json", info_items))
    monkeypatch.setattr("src.application.v2_services._enrich_state_with_info", lambda **_: enriched_state)
    monkeypatch.setattr("src.application.v2_services.top_negative_events", lambda *_, **__: [])
    monkeypatch.setattr("src.application.v2_services.top_positive_stock_signals", lambda *_, **__: [])
    monkeypatch.setattr("src.application.v2_services.quant_info_divergence_rows", lambda *_, **__: [])
    monkeypatch.setattr(
        "src.application.v2_services._attach_external_signals_to_composite_state",
        lambda **_: (
            enriched_state,
            {
                "manifest": {
                    "external_signal_version": "v2",
                    "external_signal_enabled": True,
                },
                "capital_flow_snapshot": {"flow_regime": "inflow"},
                "macro_context_snapshot": {"macro_risk_level": "neutral"},
            },
        ),
    )
    deps = replace(
        legacy._daily_state_runtime_dependencies(),
        emit_progress=lambda *_: None,
        load_watchlist=lambda _: (SimpleNamespace(symbol="MKT"), [], {"AAA": "科技"}),
        build_candidate_universe=lambda **_: SimpleNamespace(rows=[SimpleNamespace(symbol="AAA", sector="科技")]),
        load_v2_info_items_for_date=lambda **_: ("info.json", info_items),
        enrich_state_with_info=lambda **_: enriched_state,
        top_negative_events=lambda *_, **__: [],
        top_positive_stock_signals=lambda *_, **__: [],
        quant_info_divergence_rows=lambda *_, **__: [],
        attach_external_signals_to_composite_state=lambda **_: (
            enriched_state,
            {
                "manifest": {
                    "external_signal_version": "v2",
                    "external_signal_enabled": True,
                },
                "capital_flow_snapshot": {"flow_regime": "inflow"},
                "macro_context_snapshot": {"macro_risk_level": "neutral"},
            },
        ),
    )
    monkeypatch.setattr(legacy, "_daily_state_runtime_dependencies", lambda: deps)

    runtime_universe = build_daily_universe_context_runtime(
        settings,
        deps=deps,
    )
    facade_universe = legacy._build_daily_universe_context(settings)
    assert runtime_universe == facade_universe

    runtime_composite, runtime_rows = build_daily_composite_state_runtime(
        settings=settings,
        manifest=manifest,
        manifest_path=tmp_path / "research_manifest.json",
        snapshot=snapshot,
        allow_retrain=False,
        universe_ctx=runtime_universe,
        deps=deps,
    )
    facade_composite, facade_rows = legacy._build_daily_composite_state(
        settings=settings,
        manifest=manifest,
        manifest_path=tmp_path / "research_manifest.json",
        snapshot=snapshot,
        allow_retrain=False,
        universe_ctx=facade_universe,
    )
    assert runtime_composite == facade_composite
    assert runtime_rows == facade_rows

    runtime_info = attach_daily_info_overlay_runtime(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        symbol_names=symbol_names,
        deps=deps,
    )
    facade_info = legacy._attach_daily_info_overlay(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        symbol_names=symbol_names,
    )
    assert facade_info == runtime_info

    runtime_external = attach_daily_external_signal_overlay_runtime(
        snapshot=snapshot,
        settings=settings,
        composite_state=enriched_state,
        info_items=info_items,
        allow_rebuild=True,
        deps=deps,
    )
    facade_external = legacy._attach_daily_external_signal_overlay(
        snapshot=snapshot,
        settings=settings,
        composite_state=enriched_state,
        info_items=info_items,
        allow_rebuild=True,
    )
    assert facade_external == runtime_external


def test_state_build_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    import pandas as pd
    import src.application.v2_services as legacy

    monkeypatch.setattr(
        "src.application.v2_services._predict_quantile_profiles",
        lambda frame, **_: pd.DataFrame(
            {
                "expected_return": [0.03 + 0.01 * i for i in range(len(frame))],
                "q10": [-0.05] * len(frame),
                "q30": [-0.02] * len(frame),
                "q20": [-0.03] * len(frame),
                "q50": [0.01] * len(frame),
                "q70": [0.04] * len(frame),
                "q80": [0.05] * len(frame),
                "q90": [0.08] * len(frame),
            }
        ),
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_horizon_forecasts",
        lambda **kwargs: {
            horizon: {"up_prob": float(prob)}
            for horizon, prob in dict(kwargs["horizon_probs"]).items()
        },
    )
    monkeypatch.setattr(
        "src.application.v2_services._market_facts_from_row",
        lambda row: {"sample_coverage": int(row.get("breadth_coverage", 0))},
    )
    monkeypatch.setattr(
        "src.application.v2_services._stock_policy_score",
        lambda stock: float(getattr(stock, "alpha_score", 0.0)),
    )
    deps = replace(
        legacy._state_build_runtime_dependencies(),
        predict_quantile_profiles=lambda frame, **_: pd.DataFrame(
            {
                "expected_return": [0.03 + 0.01 * i for i in range(len(frame))],
                "q10": [-0.05] * len(frame),
                "q30": [-0.02] * len(frame),
                "q20": [-0.03] * len(frame),
                "q50": [0.01] * len(frame),
                "q70": [0.04] * len(frame),
                "q80": [0.05] * len(frame),
                "q90": [0.08] * len(frame),
            }
        ),
        build_horizon_forecasts=lambda **kwargs: {
            horizon: {"up_prob": float(prob)}
            for horizon, prob in dict(kwargs["horizon_probs"]).items()
        },
        market_facts_from_row=lambda row: {"sample_coverage": int(row.get("breadth_coverage", 0))},
        stock_policy_score=lambda stock: float(getattr(stock, "alpha_score", 0.0)),
        build_mainline_states=lambda **_: [SimpleNamespace(name="mainline", conviction=0.72)],
        build_candidate_selection_state=lambda **kwargs: SimpleNamespace(
            total_scored=len(kwargs["stocks"]),
            shortlist_size=len(kwargs["stocks"]),
            selection_notes=["ok"],
        ),
    )
    monkeypatch.setattr(legacy, "_state_build_runtime_dependencies", lambda: deps)

    panel_row = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "sector": "科技",
                "tradability_status": "normal",
                "close": 10.0,
                "excess_ret_1_vs_mkt": 0.01,
                "excess_ret_2_vs_mkt": 0.015,
                "excess_ret_3_vs_mkt": 0.02,
                "excess_ret_5_vs_mkt": 0.03,
                "excess_ret_20_vs_sector": 0.05,
                "feature_a": 1.0,
            },
            {
                "symbol": "BBB",
                "sector": "医药",
                "tradability_status": "normal",
                "close": 20.0,
                "excess_ret_1_vs_mkt": -0.01,
                "excess_ret_2_vs_mkt": 0.0,
                "excess_ret_3_vs_mkt": 0.01,
                "excess_ret_5_vs_mkt": 0.015,
                "excess_ret_20_vs_sector": 0.02,
                "feature_a": 2.0,
            },
        ]
    )

    class FakeBinaryModel:
        def __init__(self, probs: list[float]) -> None:
            self._probs = probs

        def predict_proba(self, frame: pd.DataFrame, feature_cols: list[str]) -> list[float]:
            assert feature_cols == ["feature_a"]
            assert len(frame) == len(self._probs)
            return self._probs

    runtime_panel_states, runtime_scored = build_stock_states_from_panel_slice_runtime(
        panel_row=panel_row,
        feature_cols=["feature_a"],
        short_model=FakeBinaryModel([0.60, 0.48]),
        two_model=FakeBinaryModel([0.62, 0.49]),
        three_model=FakeBinaryModel([0.64, 0.50]),
        five_model=FakeBinaryModel([0.66, 0.52]),
        mid_model=FakeBinaryModel([0.68, 0.55]),
        short_q_models=(object(), object(), object(), object(), object()),
        mid_q_models=(object(), object(), object(), object(), object()),
        deps=deps,
    )
    facade_panel_states, facade_scored = legacy._build_stock_states_from_panel_slice(
        panel_row=panel_row,
        feature_cols=["feature_a"],
        short_model=FakeBinaryModel([0.60, 0.48]),
        two_model=FakeBinaryModel([0.62, 0.49]),
        three_model=FakeBinaryModel([0.64, 0.50]),
        five_model=FakeBinaryModel([0.66, 0.52]),
        mid_model=FakeBinaryModel([0.68, 0.55]),
        short_q_models=(object(), object(), object(), object(), object()),
        mid_q_models=(object(), object(), object(), object(), object()),
    )
    assert facade_panel_states == runtime_panel_states
    assert facade_scored.equals(runtime_scored)

    rows = [
        SimpleNamespace(
            symbol="AAA",
            short_prob=0.60,
            two_prob=0.62,
            three_prob=0.64,
            five_prob=0.66,
            mid_prob=0.68,
            short_expected_ret=0.03,
            mid_expected_ret=0.08,
            latest_close=10.0,
            tradability_status="normal",
        ),
        SimpleNamespace(
            symbol="BBB",
            short_prob=0.48,
            two_prob=0.49,
            three_prob=0.50,
            five_prob=0.52,
            mid_prob=0.55,
            short_expected_ret=0.01,
            mid_expected_ret=0.03,
            latest_close=20.0,
            tradability_status="normal",
        ),
    ]
    sector_map = {"AAA": "科技", "BBB": "医药"}
    sector_strength_map = {"科技": 0.08, "医药": 0.03}

    runtime_row_states = build_stock_states_from_rows_runtime(
        rows,
        sector_map,
        sector_strength_map=sector_strength_map,
        deps=deps,
    )
    facade_row_states = legacy._build_stock_states_from_rows(
        rows,
        sector_map,
        sector_strength_map=sector_strength_map,
    )
    assert facade_row_states == runtime_row_states

    market_raw = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-13"),
                "close": 3000.0,
                "mkt_volatility_20": 0.12,
                "mkt_volatility_60": 0.10,
                "mkt_drawdown_20": -0.08,
                "breadth_coverage": 120,
            }
        ]
    )
    deps = replace(
        deps,
        load_symbol_daily=lambda **_: market_raw,
        make_market_feature_frame=lambda frame: frame.copy(),
        build_market_context_features=lambda **_: SimpleNamespace(frame=pd.DataFrame([{"date": pd.Timestamp("2026-03-13")}])),
        forecast_cross_section_state=lambda frame: SimpleNamespace(
            as_of_date=pd.Timestamp("2026-03-13"),
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.03,
            fund_flow_strength=0.04,
            margin_risk_on_score=0.05,
            breadth_strength=0.16,
            leader_participation=0.18,
            weak_stock_ratio=0.20,
        ),
        decide_market_state=lambda short_prob, mid_prob: SimpleNamespace(state_code="trend" if mid_prob >= 0.5 else "range"),
        build_mainline_states=lambda **_: [SimpleNamespace(name="mainline", conviction=0.72)],
        build_candidate_selection_state=lambda **kwargs: SimpleNamespace(
            total_scored=len(kwargs["stocks"]),
            shortlist_size=len(kwargs["stocks"]),
            selection_notes=["ok"],
        ),
    )
    monkeypatch.setattr(legacy, "_state_build_runtime_dependencies", lambda: deps)

    monkeypatch.setattr("src.application.v2_services.load_symbol_daily", lambda **_: market_raw)
    monkeypatch.setattr("src.application.v2_services.make_market_feature_frame", lambda frame: frame.copy())
    monkeypatch.setattr(
        "src.application.v2_services.build_market_context_features",
        lambda **_: SimpleNamespace(frame=pd.DataFrame([{"date": pd.Timestamp("2026-03-13")}]))
    )
    monkeypatch.setattr(
        "src.application.v2_services.forecast_cross_section_state",
        lambda frame: SimpleNamespace(
            as_of_date=pd.Timestamp("2026-03-13"),
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.03,
            fund_flow_strength=0.04,
            margin_risk_on_score=0.05,
            breadth_strength=0.16,
            leader_participation=0.18,
            weak_stock_ratio=0.20,
        ),
    )
    monkeypatch.setattr(
        "src.application.v2_services.decide_market_state",
        lambda short_prob, mid_prob: SimpleNamespace(state_code="trend" if mid_prob >= 0.5 else "range"),
    )

    runtime_market_states = build_market_and_cross_section_states_runtime(
        market_symbol="000300.SH",
        source="local",
        data_dir="data",
        start="2026-03-01",
        end="2026-03-13",
        use_margin_features=False,
        margin_market_file="margin.csv",
        use_us_index_context=False,
        us_index_source="akshare",
        market_short_prob=0.57,
        market_two_prob=0.58,
        market_three_prob=0.59,
        market_five_prob=0.60,
        market_mid_prob=0.62,
        deps=deps,
    )
    facade_market_states = legacy._build_market_and_cross_section_states(
        market_symbol="000300.SH",
        source="local",
        data_dir="data",
        start="2026-03-01",
        end="2026-03-13",
        use_margin_features=False,
        margin_market_file="margin.csv",
        use_us_index_context=False,
        us_index_source="akshare",
        market_short_prob=0.57,
        market_two_prob=0.58,
        market_three_prob=0.59,
        market_five_prob=0.60,
        market_mid_prob=0.62,
    )
    assert facade_market_states == runtime_market_states

    market_frame = market_raw.copy()
    runtime_prebuilt_states = build_market_and_cross_section_from_prebuilt_frame_runtime(
        market_frame=market_frame,
        market_short_prob=0.57,
        market_two_prob=0.58,
        market_three_prob=0.59,
        market_five_prob=0.60,
        market_mid_prob=0.62,
        deps=deps,
    )
    facade_prebuilt_states = legacy._build_market_and_cross_section_from_prebuilt_frame(
        market_frame=market_frame,
        market_short_prob=0.57,
        market_two_prob=0.58,
        market_three_prob=0.59,
        market_five_prob=0.60,
        market_mid_prob=0.62,
    )
    assert facade_prebuilt_states == runtime_prebuilt_states

    monkeypatch.setattr(
        "src.application.v2_services._build_mainline_states_external",
        lambda **_: [SimpleNamespace(name="绉戞妧涓荤嚎", conviction=0.72)],
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_candidate_selection_state_external",
        lambda **kwargs: SimpleNamespace(
            total_scored=len(kwargs["stocks"]),
            shortlist_size=len(kwargs["stocks"]),
            selection_notes=["ok"],
        ),
    )

    sectors = [
        SectorForecastState("绉戞妧", 0.58, 0.62, 0.10, 0.24, 0.18),
        SectorForecastState("鍖昏嵂", 0.54, 0.57, 0.06, 0.18, 0.14),
    ]

    runtime_state = compose_state_runtime(
        market=runtime_market_states[0],
        sectors=sectors,
        stocks=list(runtime_row_states),
        cross_section=runtime_market_states[1],
        deps=deps,
    )
    facade_state = legacy.compose_state(
        market=runtime_market_states[0],
        sectors=sectors,
        stocks=list(runtime_row_states),
        cross_section=runtime_market_states[1],
    )
    assert facade_state == runtime_state


def test_frozen_bundle_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    import src.application.v2_services as legacy

    class FakeBinaryModel:
        def __init__(self, *, l2: float) -> None:
            self.l2 = l2
            self.target_col = ""
            self.feature_cols: list[str] = []

        def fit(self, frame: pd.DataFrame, feature_cols: list[str], target_col: str) -> "FakeBinaryModel":
            self.target_col = target_col
            self.feature_cols = list(feature_cols)
            return self

    monkeypatch.setattr("src.application.v2_services.LogisticBinaryModel", FakeBinaryModel)
    monkeypatch.setattr(
        "src.application.v2_services._serialize_binary_model",
        lambda model: {"target": model.target_col, "features": list(model.feature_cols), "l2": model.l2},
    )
    monkeypatch.setattr(
        "src.application.v2_services._fit_quantile_quintet",
        lambda frame, *, feature_cols, target_col, l2: {
            "target": target_col,
            "features": list(feature_cols),
            "rows": len(frame),
            "l2": l2,
        },
    )
    monkeypatch.setattr(
        "src.application.v2_services._serialize_quantile_bundle",
        lambda payload: {"serialized": payload},
    )

    prepared = SimpleNamespace(
        settings={"l2": 0.25},
        market_valid=pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-03-14"), "mkt_target_1d_up": 1, "mkt_fwd_ret_1": 0.01, "mkt_fwd_ret_20": 0.04}
            ]
        ),
        panel=pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-03-14"),
                    "symbol": "AAA",
                    "target_1d_excess_mkt_up": 1,
                    "target_2d_excess_mkt_up": 1,
                    "target_3d_excess_mkt_up": 1,
                    "target_5d_excess_mkt_up": 0,
                    "target_20d_excess_sector_up": 1,
                    "excess_ret_1_vs_mkt": 0.01,
                    "excess_ret_20_vs_sector": 0.05,
                }
            ]
        ),
        market_feature_cols=["mkt_factor"],
        feature_cols=["alpha_factor"],
        dates=[pd.Timestamp("2026-03-14"), pd.Timestamp("2026-03-17")],
    )

    runtime_bundle = build_frozen_linear_forecast_bundle_runtime(
        prepared,
        deps=legacy._frozen_forecast_bundle_dependencies(),
    )
    facade_bundle = legacy._build_frozen_linear_forecast_bundle(prepared)

    assert facade_bundle == runtime_bundle
    assert runtime_bundle["backend"] == "linear"
    assert runtime_bundle["market_models"]["1d"]["target"] == "mkt_target_1d_up"
    assert runtime_bundle["stock_quantiles"]["20d"]["serialized"]["target"] == "excess_ret_20_vs_sector"


def test_frozen_forecast_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    import pandas as pd
    import src.application.v2_services as legacy

    market_raw = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-14"),
                "close": 3200.0,
                "mkt_volatility_20": 0.10,
                "mkt_volatility_60": 0.09,
                "mkt_drawdown_20": -0.05,
            }
        ]
    )
    live_panel = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-03-14"),
                "symbol": "AAA",
                "feature_a": 1.0,
            }
        ]
    )
    composite_state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-14",
            up_1d_prob=0.58,
            up_5d_prob=0.60,
            up_20d_prob=0.62,
            trend_state="trend",
            drawdown_risk=0.12,
            volatility_regime="normal",
            liquidity_stress=0.10,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-14",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.01,
            fund_flow_strength=0.03,
            margin_risk_on_score=0.04,
            breadth_strength=0.18,
            leader_participation=0.16,
            weak_stock_ratio=0.20,
        ),
        sectors=[SectorForecastState("科技", 0.58, 0.61, 0.08, 0.20, 0.12)],
        stocks=[StockForecastState("AAA", "科技", 0.60, 0.63, 0.66, 0.59, 0.10, 0.88)],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )

    class FakeBinaryModel:
        def __init__(self, prob: float) -> None:
            self.prob = prob

        def predict_proba(self, frame: pd.DataFrame, feature_cols: list[str]) -> list[float]:
            assert feature_cols
            return [self.prob] * len(frame)

    monkeypatch.setattr("src.application.v2_services.load_symbol_daily", lambda **_: market_raw)
    monkeypatch.setattr("src.application.v2_services.make_market_feature_frame", lambda frame: frame.copy())
    monkeypatch.setattr(
        "src.application.v2_services.build_market_context_features",
        lambda **_: SimpleNamespace(frame=pd.DataFrame([{"date": pd.Timestamp("2026-03-14"), "us_x": 1.0}])),
    )
    monkeypatch.setattr(
        "src.application.v2_services._deserialize_binary_model",
        lambda payload: FakeBinaryModel(float(payload.get("prob", 0.5))),
    )
    monkeypatch.setattr(
        "src.application.v2_services._deserialize_quantile_bundle",
        lambda payload: payload,
    )
    monkeypatch.setattr(
        "src.application.v2_services._predict_quantile_profile",
        lambda frame, **_: SimpleNamespace(
            expected_return=0.03,
            q10=-0.05,
            q30=-0.02,
            q20=-0.03,
            q50=0.01,
            q70=0.04,
            q80=0.05,
            q90=0.08,
        ),
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_market_and_cross_section_from_prebuilt_frame",
        lambda **_: (composite_state.market, composite_state.cross_section),
    )
    monkeypatch.setattr(
        "src.application.v2_services.build_stock_live_panel_dataset",
        lambda **_: SimpleNamespace(frame=live_panel),
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_stock_states_from_panel_slice",
        lambda **_: (composite_state.stocks, pd.DataFrame([{"symbol": "AAA", "score": 0.8}])),
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_sector_states_external",
        lambda *args, **kwargs: composite_state.sectors,
    )
    monkeypatch.setattr(
        "src.application.v2_services.compose_state",
        lambda **_: composite_state,
    )

    settings = {
        "source": "local",
        "data_dir": "data",
        "start": "2026-03-01",
        "end": "2026-03-14",
        "use_margin_features": False,
        "margin_market_file": "margin_market.csv",
        "margin_stock_file": "margin_stock.csv",
        "use_us_index_context": False,
        "us_index_source": "akshare",
        "use_us_sector_etf_context": False,
        "use_cn_etf_context": False,
        "cn_etf_source": "akshare",
    }
    universe_ctx = SimpleNamespace(
        market_security=SimpleNamespace(symbol="000300.SH"),
        stocks=[SimpleNamespace(symbol="AAA")],
    )
    bundle = {
        "backend": "linear",
        "market_feature_cols": ["close"],
        "panel_feature_cols": ["feature_a"],
        "market_models": {
            "1d": {"prob": 0.58},
            "2d": {"prob": 0.59},
            "3d": {"prob": 0.60},
            "5d": {"prob": 0.61},
            "20d": {"prob": 0.62},
        },
        "market_quantiles": {"1d": {}, "20d": {}},
        "stock_models": {
            "1d": {"prob": 0.60},
            "2d": {"prob": 0.61},
            "3d": {"prob": 0.62},
            "5d": {"prob": 0.63},
            "20d": {"prob": 0.64},
        },
        "stock_quantiles": {"1d": {}, "20d": {}},
    }

    runtime_market_frame = build_live_market_frame_runtime(
        settings=settings,
        market_symbol="000300.SH",
        deps=legacy._frozen_forecast_runtime_dependencies(),
    )
    facade_market_frame = legacy._build_live_market_frame(
        settings=settings,
        market_symbol="000300.SH",
    )
    assert facade_market_frame.equals(runtime_market_frame)

    runtime_scored = score_live_composite_state_from_frozen_bundle_runtime(
        bundle=bundle,
        settings=settings,
        universe_ctx=universe_ctx,
        deps=legacy._frozen_forecast_runtime_dependencies(),
    )
    facade_scored = legacy._score_live_composite_state_from_frozen_bundle(
        bundle=bundle,
        settings=settings,
        universe_ctx=universe_ctx,
    )
    assert facade_scored == runtime_scored


def test_info_shadow_report_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    import src.application.v2_services as legacy

    validation = SimpleNamespace(steps=[])
    holdout = SimpleNamespace(
        steps=[
            SimpleNamespace(horizon_metrics={"20d": {"rank_ic": 0.10, "top_bottom_spread": 0.04}}),
            SimpleNamespace(horizon_metrics={"20d": {"rank_ic": 0.14, "top_bottom_spread": 0.06}}),
        ]
    )
    info_items = [
        InfoItem(
            date="2026-03-15",
            target_type="stock",
            target="AAA",
            horizon="20d",
            direction="up",
            info_type="announcement",
            title="ann",
            source_subset="announcements",
        ),
        InfoItem(
            date="2026-03-15",
            target_type="market",
            target="all",
            horizon="5d",
            direction="up",
            info_type="news",
            title="news",
            source_subset="market_news",
        ),
    ]

    deps = replace(
        legacy._info_shadow_report_dependencies(),
        build_info_shadow_variant=lambda **kwargs: {
            "avg_1d_rank_ic": float(len(kwargs["info_items"])),
            "avg_5d_rank_ic": float(len(kwargs["info_items"])) + 0.1,
            "avg_20d_rank_ic": float(len(kwargs["info_items"])) + 0.2,
            "avg_20d_top_bottom_spread": float(len(kwargs["info_items"])) + 0.3,
            "event_day_hit_rate": float(len(kwargs["info_items"])) + 0.4,
            "quant_event_day_hit_rate": 0.55,
            "coverage_summary": {"count": len(kwargs["info_items"])},
            "top_positive_stock_deltas": [{"symbol": "AAA"}],
            "top_negative_stock_deltas": [{"symbol": "BBB"}],
            "last_market_info_state": {"score": len(kwargs["info_items"])},
            "last_date": "2026-03-15",
            "market_shadow_modes": {"1d": "blend"},
            "stock_shadow_modes": {"20d": "shadow"},
            "model_samples": {"market": {"1d": len(kwargs["info_items"])}, "stock": {}},
        },
        filter_info_items_by_source_subset=lambda items, subset: [
            item for item in items if getattr(item, "source_subset", "") == subset
        ],
        event_tag_counts=lambda items: {"all": len(list(items))},
        info_source_breakdown=lambda items: {"sources": len(list(items))},
    )
    monkeypatch.setattr(legacy, "_info_shadow_report_dependencies", lambda: deps)

    runtime_report = build_info_shadow_report_runtime(
        validation_trajectory=validation,
        holdout_trajectory=holdout,
        settings={"use_info_fusion": True, "info_shadow_only": False},
        info_items=info_items,
        deps=deps,
    )
    facade_report = legacy._build_info_shadow_report(
        validation_trajectory=validation,
        holdout_trajectory=holdout,
        settings={"use_info_fusion": True, "info_shadow_only": False},
        info_items=info_items,
    )

    assert facade_report == runtime_report
    assert runtime_report["quant_only"]["avg_20d_rank_ic"] == pytest.approx(0.12)


def test_info_shadow_runtime_keeps_facade_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace

    import pandas as pd
    import src.application.v2_services as legacy

    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-15",
            up_1d_prob=0.56,
            up_5d_prob=0.58,
            up_20d_prob=0.60,
            trend_state="trend",
            drawdown_risk=0.18,
            volatility_regime="normal",
            liquidity_stress=0.14,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-15",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.01,
            fund_flow_strength=0.05,
            margin_risk_on_score=0.06,
            breadth_strength=0.12,
            leader_participation=0.15,
            weak_stock_ratio=0.24,
        ),
        sectors=[SectorForecastState("科技", 0.58, 0.61, 0.08, 0.20, 0.15)],
        stocks=[StockForecastState("AAA", "科技", 0.60, 0.63, 0.67, 0.59, 0.10, 0.88, alpha_score=0.72)],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    info_items = [
        InfoItem(
            date="2026-03-15",
            target_type="stock",
            target="AAA",
            horizon="20d",
            direction="up",
            info_type="news",
            title="positive",
        )
    ]

    monkeypatch.setattr(
        "src.application.v2_services.build_info_state_maps",
        lambda **_: (
            InfoAggregateState(item_count=1, info_prob_1d=0.61, info_prob_5d=0.62, info_prob_20d=0.66, short_score=0.63, mid_score=0.67),
            {"科技": InfoAggregateState(item_count=1, short_score=0.60, mid_score=0.64)},
            {"AAA": InfoAggregateState(item_count=1, info_prob_1d=0.64, info_prob_5d=0.65, info_prob_20d=0.68, short_score=0.66, mid_score=0.70)},
        ),
    )
    monkeypatch.setattr(
        "src.application.v2_services._predict_info_shadow_prob",
        lambda **kwargs: (float(0.5 * kwargs["quant_prob"] + 0.5 * kwargs["info_prob"]), {"mode": "stub"}),
    )
    monkeypatch.setattr(
        "src.application.v2_services.blend_probability",
        lambda quant_prob, info_prob, sentiment_strength=0.0: float(0.5 * float(quant_prob) + 0.5 * float(info_prob)),
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_mainline_states_external",
        lambda **_: [],
    )
    monkeypatch.setattr(
        "src.application.v2_services._info_feature_frame",
        lambda **kwargs: pd.DataFrame(
            {
                "q_logit": [0.1] * len(kwargs["quant_prob"]),
                "i_logit": [0.2] * len(kwargs["quant_prob"]),
                "q_minus_i": [0.1] * len(kwargs["quant_prob"]),
                "negative_event_risk": list(kwargs["negative_event_risk"]),
                "item_count_log": [0.5] * len(kwargs["quant_prob"]),
            }
        ),
    )
    monkeypatch.setattr(
        "src.application.v2_services._fit_info_shadow_model",
        lambda frame, **_: SimpleNamespace(mode="learned", samples=len(frame), feature_cols=list(frame.columns)),
    )
    monkeypatch.setattr(
        "src.application.v2_services._panel_slice_metrics",
        lambda frame, **_: (0.11, 0.02, 0.03, 0.64),
    )
    deps = replace(
        legacy._info_shadow_runtime_dependencies(),
        build_info_state_maps=lambda **_: (
            InfoAggregateState(item_count=1, info_prob_1d=0.61, info_prob_5d=0.62, info_prob_20d=0.66, short_score=0.63, mid_score=0.67),
            {"绉戞妧": InfoAggregateState(item_count=1, short_score=0.60, mid_score=0.64)},
            {"AAA": InfoAggregateState(item_count=1, info_prob_1d=0.64, info_prob_5d=0.65, info_prob_20d=0.68, short_score=0.66, mid_score=0.70)},
        ),
        predict_info_shadow_prob=lambda **kwargs: (
            float(0.5 * kwargs["quant_prob"] + 0.5 * kwargs["info_prob"]),
            {"mode": "stub"},
        ),
        blend_probability=lambda quant_prob, info_prob, sentiment_strength=0.0: float(
            0.5 * float(quant_prob) + 0.5 * float(info_prob)
        ),
        build_mainline_states=lambda **_: [],
        info_feature_frame=lambda **kwargs: pd.DataFrame(
            {
                "q_logit": [0.1] * len(kwargs["quant_prob"]),
                "i_logit": [0.2] * len(kwargs["quant_prob"]),
                "q_minus_i": [0.1] * len(kwargs["quant_prob"]),
                "negative_event_risk": list(kwargs["negative_event_risk"]),
                "item_count_log": [0.5] * len(kwargs["quant_prob"]),
            }
        ),
        fit_info_shadow_model=lambda frame, **_: SimpleNamespace(
            mode="learned",
            samples=len(frame),
            feature_cols=list(frame.columns),
        ),
        panel_slice_metrics=lambda frame, **_: (0.11, 0.02, 0.03, 0.64),
    )
    monkeypatch.setattr(legacy, "_info_shadow_runtime_dependencies", lambda: deps)

    settings = {
        "info_half_life_days": 10.0,
        "market_info_strength": 0.9,
        "stock_info_strength": 1.1,
        "learned_info_l2": 0.8,
        "learned_info_min_samples": 1,
    }
    trajectory = SimpleNamespace(
        prepared=SimpleNamespace(
            market_valid=pd.DataFrame([{"date": pd.Timestamp("2026-03-15"), "mkt_fwd_ret_1": 0.01, "mkt_fwd_ret_5": 0.02, "mkt_fwd_ret_20": 0.03}]),
            stock_frames={
                "AAA": pd.DataFrame([{"date": pd.Timestamp("2026-03-15"), "excess_ret_1_vs_mkt": 0.01, "excess_ret_5_vs_mkt": 0.02, "excess_ret_20_vs_sector": 0.04}])
            },
        ),
        steps=[
            SimpleNamespace(
                date=pd.Timestamp("2026-03-15"),
                composite_state=state,
                horizon_metrics={"20d": {"top_k_hit_rate": 0.60}},
            )
        ],
    )

    runtime_enriched = enrich_state_with_info_runtime(
        state=state,
        as_of_date=pd.Timestamp("2026-03-15"),
        info_items=info_items,
        settings=settings,
        deps=deps,
    )
    facade_enriched = legacy._enrich_state_with_info(
        state=state,
        as_of_date=pd.Timestamp("2026-03-15"),
        info_items=info_items,
        settings=settings,
    )
    assert facade_enriched == runtime_enriched

    runtime_models = fit_v2_info_shadow_models_runtime(
        trajectory=trajectory,
        settings=settings,
        info_items=info_items,
        deps=deps,
    )
    facade_models = legacy._fit_v2_info_shadow_models(
        trajectory=trajectory,
        settings=settings,
        info_items=info_items,
    )
    assert facade_models == runtime_models

    runtime_variant = build_info_shadow_variant_runtime(
        validation_trajectory=trajectory,
        holdout_trajectory=trajectory,
        settings=settings,
        info_items=info_items,
        deps=deps,
    )
    facade_variant = legacy._build_info_shadow_variant(
        validation_trajectory=trajectory,
        holdout_trajectory=trajectory,
        settings=settings,
        info_items=info_items,
    )
    assert facade_variant == runtime_variant
