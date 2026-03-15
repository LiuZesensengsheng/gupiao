from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.application.v2_services as legacy_services
from src.application import v2_policy_runtime as policy_runtime
from src.application import v2_learning_target_runtime as learning_target_runtime
from src.application import v2_policy_learning_runtime as policy_learning_runtime
from src.application.v2_contracts import (
    CapitalFlowState,
    CompositeState,
    CrossSectionForecastState,
    InfoAggregateState,
    LearnedPolicyModel,
    MarketFactsState,
    MarketForecastState,
    MacroContextState,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
from src.application.v2_backtest_metrics_runtime import build_date_slice_index as _build_date_slice_index
from src.application.v2_backtest_prepare_runtime import (
    BacktestTrajectory as _BacktestTrajectory,
    TrajectoryStep as _TrajectoryStep,
    split_research_trajectory as _split_research_trajectory,
)
from src.application.v2_facade_support_runtime import (
    policy_objective_score as _policy_objective_score,
    sha256_file as _sha256_file,
)
from src.application.v2_feature_runtime import (
    make_forecast_backend as _make_forecast_backend,
    tensorize_temporal_frame as _tensorize_temporal_frame,
)
from src.application.v2_forecast_model_runtime import predict_quantile_profiles as _predict_quantile_profiles
from src.application.v2_runtime_primitives import is_main_board_symbol as _is_main_board_symbol
from src.reporting import report_state_runtime
from src.application.v2_runtime_settings import (
    load_v2_runtime_settings as _load_v2_runtime_settings,
    resolve_v2_universe_settings as _resolve_v2_universe_settings,
)
from src.application.v2_services import (
    _build_daily_snapshot_context,
    _prepare_v2_backtest_data,
    calibrate_v2_policy,
    _load_or_build_v2_backtest_trajectory,
    load_published_v2_policy_model,
    publish_v2_research_artifacts,
    run_daily_v2_live,
)
from src.workflows.research_workflow import run_v2_research_workflow_impl as _run_v2_research_workflow_runtime
from src.infrastructure import market_data
from src.interfaces.cli.run_v2_cli import build_parser


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
        market_info_state=InfoAggregateState(catalyst_strength=0.20, coverage_confidence=0.65),
        stock_info_states={
            "AAA": InfoAggregateState(catalyst_strength=0.35, event_risk_level=0.12, coverage_confidence=0.80),
            "BBB": InfoAggregateState(catalyst_strength=0.10, event_risk_level=0.18, coverage_confidence=0.55),
        },
        capital_flow_state=CapitalFlowState(
            northbound_net_flow=0.25,
            margin_balance_change=0.12,
            turnover_heat=0.68,
            large_order_bias=0.18,
            flow_regime="inflow",
        ),
        macro_context_state=MacroContextState(
            style_regime="quality",
            commodity_pressure=0.22,
            fx_pressure=0.18,
            index_breadth_proxy=0.61,
            macro_risk_level="neutral",
        ),
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
        excess_annual_return=0.06,
        information_ratio=0.55,
        nav_curve=[1.0, 1.05, 1.10],
        curve_dates=["2024-01-01", "2024-06-01", "2024-12-31"],
    )


def _research_workflow_dependencies(**overrides: object):
    return replace(legacy_services._research_workflow_dependencies(), **overrides)


def _policy_learning_dependencies(**overrides: object):
    return replace(legacy_services._policy_learning_dependencies(), **overrides)


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

    spec = policy_runtime.policy_spec_from_model(
        state=state,
        model=model,
        deps=legacy_services._policy_runtime_dependencies(),
    )

    assert isinstance(spec, PolicySpec)
    assert 0.20 <= spec.risk_on_exposure <= 0.95
    assert 1 <= spec.risk_on_positions <= 6
    assert 0.10 <= spec.risk_on_turnover_cap <= 0.45


def test_sha256_file_supports_directory_inputs(tmp_path: Path) -> None:
    info_dir = tmp_path / "info_parts"
    info_dir.mkdir(parents=True, exist_ok=True)
    (info_dir / "a.csv").write_text("x\n1\n", encoding="utf-8")
    nested = info_dir / "nested"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "b.csv").write_text("y\n2\n", encoding="utf-8")

    first = _sha256_file(info_dir)
    second = _sha256_file(info_dir)

    assert first
    assert first == second


def test_load_v2_runtime_settings_applies_tushare_token_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "api.json"
    config_path.write_text(
        json.dumps(
            {
                "common": {
                    "source": "tushare",
                    "tushare_token": "token-from-config",
                }
            }
        ),
        encoding="utf-8",
    )
    previous = market_data._resolve_tushare_token()
    market_data.set_tushare_token(None)
    try:
        settings = _load_v2_runtime_settings(config_path=str(config_path))
        assert settings["source"] == "tushare"
        assert market_data._resolve_tushare_token() == "token-from-config"
    finally:
        market_data.set_tushare_token(previous)


def test_load_v2_runtime_settings_prefers_generated_base_file_for_dynamic_universe(tmp_path: Path) -> None:
    config_path = tmp_path / "api.json"
    config_path.write_text(
        json.dumps(
            {
                "common": {"source": "local"},
                "daily": {
                    "dynamic_universe_enabled": True,
                    "universe_limit": 300,
                    "generated_universe_base_file": "config/universe_all_a_full.json",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = _load_v2_runtime_settings(config_path=str(config_path))

    assert settings["dynamic_universe_enabled"] is True
    assert settings["universe_file"] == "config/universe_all_a_full.json"


def test_load_v2_runtime_settings_reads_main_board_recommendation_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "api.json"
    config_path.write_text(
        json.dumps(
            {
                "common": {"source": "local"},
                "daily": {"main_board_only_recommendations": True},
            }
        ),
        encoding="utf-8",
    )

    settings = _load_v2_runtime_settings(config_path=str(config_path))

    assert settings["main_board_only_recommendations"] is True


def test_load_v2_runtime_settings_reads_main_board_universe_flag(tmp_path: Path) -> None:
    config_path = tmp_path / "api.json"
    config_path.write_text(
        json.dumps(
            {
                "common": {"source": "local"},
                "daily": {"main_board_only_universe": True},
            }
        ),
        encoding="utf-8",
    )

    settings = _load_v2_runtime_settings(config_path=str(config_path))

    assert settings["main_board_only_universe"] is True


def test_main_board_symbol_classifier() -> None:
    assert _is_main_board_symbol("600000.SH") is True
    assert _is_main_board_symbol("002001.SZ") is True
    assert _is_main_board_symbol("300001.SZ") is False
    assert _is_main_board_symbol("688001.SH") is False


def test_filter_state_for_recommendation_scope_keeps_only_main_board() -> None:
    state = _make_state()
    state = replace(
        state,
        stocks=[
            replace(state.stocks[0], symbol="600000.SH", sector="银行"),
            replace(state.stocks[1], symbol="300001.SZ", sector="电气设备"),
        ],
    )

    filtered = report_state_runtime.filter_state_for_recommendation_scope(
        state=state,
        main_board_only=True,
        deps=legacy_services._report_state_runtime_dependencies(),
    )

    assert [stock.symbol for stock in filtered.stocks] == ["600000.SH"]
    assert filtered.candidate_selection.shortlisted_symbols == ["600000.SH"]
    assert any("main-board" in note for note in filtered.candidate_selection.selection_notes)


def test_run_v2_cli_accepts_tushare_token_override() -> None:
    parser = build_parser()
    args = parser.parse_args(["daily-run", "--tushare-token", "demo-token"])
    assert args.tushare_token == "demo-token"


def test_run_v2_cli_accepts_dynamic_universe_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "research-run",
            "--dynamic-universe",
            "--generator-target-size",
            "300",
            "--generator-coarse-size",
            "1000",
            "--generator-theme-aware",
            "--generator-use-concepts",
        ]
    )
    assert args.dynamic_universe is True
    assert args.generator_target_size == 300
    assert args.generator_coarse_size == 1000
    assert args.generator_theme_aware is True
    assert args.generator_use_concepts is True


def test_explicit_universe_file_disables_default_universe_tier(tmp_path: Path) -> None:
    config_path = tmp_path / "api.json"
    config_path.write_text(
        json.dumps(
            {
                "daily": {
                    "universe_tier": "favorites_16",
                    "universe_file": "config/universe_auto_longtrain.json",
                    "universe_limit": 16,
                }
            }
        ),
        encoding="utf-8",
    )
    settings = _load_v2_runtime_settings(
        config_path=str(config_path),
        universe_file="config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json",
        universe_limit=300,
    )
    assert settings["universe_file"] == "config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json"
    assert settings["universe_limit"] == 300
    assert settings["universe_tier"] == ""


def test_explicit_universe_file_overrides_generated_base_for_dynamic_generated_tier(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate_dynamic_universe(**kwargs):
        captured.update(kwargs)
        manifest_dir = tmp_path / "cache" / "universe_catalog"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "dynamic_300_test.generator.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            selected_300=[{"symbol": "600000.SH"}, {"symbol": "000001.SZ"}],
            generator_manifest=SimpleNamespace(
                source_universe_path=str(kwargs["universe_file"]),
                manifest_path=str(manifest_path),
                generator_version="dynamic_universe_v2_leaders",
                generator_hash="test-hash",
                coarse_pool_size=1000,
                refined_pool_size=600,
                selected_pool_size=2,
                theme_allocations=[],
            ),
        )

    resolved = _resolve_v2_universe_settings(
        settings={
            "universe_tier": "generated_300",
            "dynamic_universe_enabled": True,
            "generated_universe_base_file": "config/universe_smoke_5.json",
            "universe_file": "config/universe_all_a_tushare_live_20260313.json",
            "generator_target_size": 300,
            "generator_coarse_size": 1000,
            "data_dir": "data",
            "end": "2026-03-13",
        },
        cache_root=str(tmp_path / "cache"),
        generate_dynamic_universe_fn=fake_generate_dynamic_universe,
    )

    assert captured["universe_file"] == "config/universe_all_a_tushare_live_20260313.json"
    assert resolved["source_universe_manifest_path"] == "config/universe_all_a_tushare_live_20260313.json"
    assert resolved["selected_pool_size"] == 2


def test_resolve_v2_universe_settings_passes_main_board_only_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_generate_dynamic_universe(**kwargs):
        captured.update(kwargs)
        manifest_dir = tmp_path / "cache" / "universe_catalog"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "dynamic_300_test.generator.json"
        manifest_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            selected_300=[{"symbol": "600000.SH"}],
            generator_manifest=SimpleNamespace(
                source_universe_path=str(kwargs["universe_file"]),
                manifest_path=str(manifest_path),
                generator_version="dynamic_universe_v2_leaders",
                generator_hash="test-hash",
                coarse_pool_size=1000,
                refined_pool_size=600,
                selected_pool_size=1,
                theme_allocations=[],
            ),
        )

    _resolve_v2_universe_settings(
        settings={
            "universe_tier": "generated_300",
            "dynamic_universe_enabled": True,
            "generated_universe_base_file": "config/universe_all_a_tushare_live_20260313.json",
            "generator_target_size": 300,
            "generator_coarse_size": 1000,
            "data_dir": "data",
            "end": "2026-03-13",
            "main_board_only_universe": True,
        },
        cache_root=str(tmp_path / "cache"),
        generate_dynamic_universe_fn=fake_generate_dynamic_universe,
    )

    assert captured["main_board_only"] is True


def test_generated_80_learning_targets_prefer_realizable_alpha_and_ranking() -> None:
    date = pd.Timestamp("2026-03-01")
    strong_state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-01",
            up_1d_prob=0.56,
            up_5d_prob=0.58,
            up_20d_prob=0.60,
            trend_state="trend",
            drawdown_risk=0.20,
            volatility_regime="normal",
            liquidity_stress=0.18,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-01",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=-0.01,
            fund_flow_strength=0.10,
            margin_risk_on_score=0.09,
            breadth_strength=0.18,
            leader_participation=0.62,
            weak_stock_ratio=0.24,
        ),
        sectors=[
            SectorForecastState("有色", 0.58, 0.63, 0.18, 0.22, 0.18),
            SectorForecastState("化工", 0.55, 0.59, 0.10, 0.24, 0.16),
        ],
        stocks=[
            StockForecastState("A1", "有色", 0.60, 0.65, 0.72, 0.62, 0.30, 0.92, alpha_score=0.72),
            StockForecastState("A2", "有色", 0.58, 0.62, 0.68, 0.60, 0.24, 0.90, alpha_score=0.67),
            StockForecastState("B1", "化工", 0.55, 0.58, 0.62, 0.55, 0.14, 0.86, alpha_score=0.58),
            StockForecastState("B2", "化工", 0.53, 0.56, 0.60, 0.53, 0.10, 0.84, alpha_score=0.56),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    weak_state = CompositeState(
        market=strong_state.market,
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-01",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=-0.01,
            fund_flow_strength=0.02,
            margin_risk_on_score=0.04,
            breadth_strength=0.06,
            leader_participation=0.48,
            weak_stock_ratio=0.58,
        ),
        sectors=strong_state.sectors,
        stocks=[
            StockForecastState("C1", "有色", 0.68, 0.71, 0.74, 0.51, 0.01, 0.82, alpha_score=0.53),
            StockForecastState("C2", "有色", 0.67, 0.70, 0.73, 0.50, 0.00, 0.80, alpha_score=0.52),
            StockForecastState("D1", "化工", 0.66, 0.69, 0.72, 0.49, -0.01, 0.80, alpha_score=0.51),
            StockForecastState("D2", "化工", 0.65, 0.68, 0.71, 0.48, -0.02, 0.78, alpha_score=0.50),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    strong_frames = {
        "A1": pd.DataFrame([{"date": date, "fwd_ret_1": 0.020, "excess_ret_1_vs_mkt": 0.015, "excess_ret_5_vs_mkt": 0.030, "excess_ret_20_vs_sector": 0.050}]),
        "A2": pd.DataFrame([{"date": date, "fwd_ret_1": 0.016, "excess_ret_1_vs_mkt": 0.012, "excess_ret_5_vs_mkt": 0.024, "excess_ret_20_vs_sector": 0.040}]),
        "B1": pd.DataFrame([{"date": date, "fwd_ret_1": 0.010, "excess_ret_1_vs_mkt": 0.006, "excess_ret_5_vs_mkt": 0.014, "excess_ret_20_vs_sector": 0.022}]),
        "B2": pd.DataFrame([{"date": date, "fwd_ret_1": 0.008, "excess_ret_1_vs_mkt": 0.004, "excess_ret_5_vs_mkt": 0.010, "excess_ret_20_vs_sector": 0.018}]),
    }
    weak_frames = {
        "C1": pd.DataFrame([{"date": date, "fwd_ret_1": 0.030, "excess_ret_1_vs_mkt": -0.003, "excess_ret_5_vs_mkt": -0.004, "excess_ret_20_vs_sector": -0.006}]),
        "C2": pd.DataFrame([{"date": date, "fwd_ret_1": 0.028, "excess_ret_1_vs_mkt": -0.004, "excess_ret_5_vs_mkt": -0.006, "excess_ret_20_vs_sector": -0.008}]),
        "D1": pd.DataFrame([{"date": date, "fwd_ret_1": 0.026, "excess_ret_1_vs_mkt": -0.005, "excess_ret_5_vs_mkt": -0.007, "excess_ret_20_vs_sector": -0.010}]),
        "D2": pd.DataFrame([{"date": date, "fwd_ret_1": 0.024, "excess_ret_1_vs_mkt": -0.006, "excess_ret_5_vs_mkt": -0.008, "excess_ret_20_vs_sector": -0.012}]),
    }
    strong_targets = learning_target_runtime.derive_learning_targets(
        state=strong_state,
        stock_frames=strong_frames,
        date=date,
        horizon_metrics={"20d": {"rank_ic": 0.16, "top_bottom_spread": 0.09, "top_k_hit_rate": 0.72}},
        universe_tier="generated_80",
        deps=legacy_services._learning_target_dependencies(),
    )
    weak_targets = learning_target_runtime.derive_learning_targets(
        state=weak_state,
        stock_frames=weak_frames,
        date=date,
        horizon_metrics={"20d": {"rank_ic": -0.03, "top_bottom_spread": -0.01, "top_k_hit_rate": 0.48}},
        universe_tier="generated_80",
        deps=legacy_services._learning_target_dependencies(),
    )

    assert strong_targets[0] > weak_targets[0]
    assert strong_targets[1] >= weak_targets[1]
    assert strong_targets[2] >= weak_targets[2]
    assert strong_targets[3] > weak_targets[3]


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
    previous_run = tmp_path / "swing_v2" / "20260301_101010"
    previous_run.mkdir(parents=True, exist_ok=True)
    prev_backtest = {
        "baseline": asdict(baseline),
        "learned": asdict(learned),
    }
    (previous_run / "backtest_summary.json").write_text(json.dumps(prev_backtest), encoding="utf-8")
    (tmp_path / "swing_v2" / "latest_research_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "20260301_101010",
                "strategy_id": "swing_v2",
                "backtest_summary": str(previous_run / "backtest_summary.json"),
            }
        ),
        encoding="utf-8",
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
    assert paths["release_gate_passed"] == "true"


def test_publish_artifacts_records_universe_metadata_and_keeps_non_default_latest_isolated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = _make_backtest(0.20, 0.18)
    calibrated = _make_backtest(0.22, 0.20)
    learned = _make_backtest(0.24, 0.22)
    learning_result = V2PolicyLearningResult(
        model=LearnedPolicyModel(
            feature_names=["x1"],
            exposure_intercept=0.55,
            exposure_coef=[0.1],
            position_intercept=2.5,
            position_coef=[0.05],
            turnover_intercept=0.20,
            turnover_coef=[0.01],
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
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2022-01-01", periods=520, freq="B")
    for symbol, amount in [("000001.SZ", 8.0e7), ("000002.SZ", 6.0e7), ("000003.SZ", 4.0e7)]:
        pd.DataFrame(
            {
                "date": dates,
                "open": 10.0,
                "high": 10.2,
                "low": 9.8,
                "close": 10.0,
                "volume": 1000000,
                "amount": amount,
                "symbol": symbol,
            }
        ).to_csv(data_dir / f"{symbol}.csv", index=False)
    favorites = tmp_path / "favorites.json"
    favorites.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "000001.SZ", "name": "A", "sector": "其他"},
                    {"symbol": "000002.SZ", "name": "B", "sector": "其他"},
                ]
            }
        ),
        encoding="utf-8",
    )
    generated_base = tmp_path / "base.json"
    generated_base.write_text(
        json.dumps(
            {
                "stocks": [
                    {"symbol": "000001.SZ", "name": "A", "sector": "其他"},
                    {"symbol": "000002.SZ", "name": "B", "sector": "其他"},
                    {"symbol": "000003.SZ", "name": "C", "sector": "其他"},
                ]
            }
        ),
        encoding="utf-8",
    )
    baseline_ref_dir = tmp_path / "swing_v2" / "20260308_211808"
    baseline_ref_dir.mkdir(parents=True, exist_ok=True)
    (baseline_ref_dir / "backtest_summary.json").write_text(
        json.dumps({"learned": asdict(_make_backtest(0.30, 0.28))}),
        encoding="utf-8",
    )
    info_dir = tmp_path / "info_parts"
    (info_dir / "market_news").mkdir(parents=True, exist_ok=True)
    (info_dir / "announcements").mkdir(parents=True, exist_ok=True)
    (info_dir / "research").mkdir(parents=True, exist_ok=True)
    (info_dir / "market_news" / "market.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title",
                "2024-12-20,market,MARKET,mid,bullish,macro support",
            ]
        ),
        encoding="utf-8",
    )
    (info_dir / "announcements" / "ann.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title,event_tag",
                "2024-12-22,stock,000001.SZ,short,bearish,risk event,earnings_negative",
            ]
        ),
        encoding="utf-8",
    )
    (info_dir / "research" / "research.csv").write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,title",
                "2024-12-23,stock,000001.SZ,mid,bullish,券商首次覆盖并给出买入评级",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", lambda **_: None)

    paths = publish_v2_research_artifacts(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path),
        cache_root=str(tmp_path / "cache"),
        publish_forecast_models=False,
        settings={
            "config_path": "config/api.json",
            "source": "local",
            "watchlist": "config/watchlist.json",
            "data_dir": str(data_dir),
            "start": "2024-01-01",
            "end": "2024-12-31",
            "universe_tier": "generated_80",
            "active_default_universe_tier": "favorites_16",
            "candidate_default_universe_tier": "generated_80",
            "favorites_universe_file": str(favorites),
            "generated_universe_base_file": str(generated_base),
            "baseline_reference_run_id": "20260308_211808",
            "dynamic_universe_enabled": True,
            "generator_manifest_path": str(tmp_path / "cache" / "dynamic.generator.json"),
            "generator_version": "dynamic_universe_v1",
            "generator_hash": "genhash",
            "coarse_pool_size": 12,
            "refined_pool_size": 6,
            "selected_pool_size": 3,
            "theme_allocations": [
                {"theme": "资源", "selected_count": 2, "refined_count": 3, "coarse_count": 4, "theme_strength": 0.72},
                {"theme": "能源石油", "selected_count": 1, "refined_count": 2, "coarse_count": 3, "theme_strength": 0.65},
            ],
            "info_file": str(info_dir),
            "use_info_fusion": True,
            "info_shadow_only": True,
            "info_source_mode": "layered",
            "info_subsets": ["market_news", "announcements", "research"],
            "announcement_event_tags": ["earnings_negative"],
            "use_us_index_context": True,
            "us_index_source": "akshare",
        },
        baseline=baseline,
        calibration=calibration,
        learning=learning_result,
    )

    dataset_manifest = json.loads(Path(paths["dataset_manifest"]).read_text(encoding="utf-8"))
    assert dataset_manifest["universe_tier"] == "generated_80"
    assert dataset_manifest["universe_id"] == "dynamic_80"
    assert dataset_manifest["symbol_count"] == 3
    assert len(dataset_manifest["symbols"]) == 3
    assert dataset_manifest["source_universe_manifest_path"]
    assert dataset_manifest["dynamic_universe_enabled"] is True
    assert dataset_manifest["generator_version"] == "dynamic_universe_v2_leaders"
    assert dataset_manifest["generator_hash"]
    assert dataset_manifest["coarse_pool_size"] >= 3
    assert dataset_manifest["refined_pool_size"] >= 3
    assert dataset_manifest["selected_pool_size"] == 3
    assert len(dataset_manifest["theme_allocations"]) >= 1
    assert dataset_manifest["info_file"] == str(info_dir)
    assert dataset_manifest["info_hash"]
    assert dataset_manifest["use_us_index_context"] is True
    assert dataset_manifest["us_index_source"] == "akshare"
    assert dataset_manifest["info_item_count"] == 3
    assert dataset_manifest["info_source_mode"] == "layered"
    assert dataset_manifest["info_subsets"] == ["market_news", "announcements", "research"]
    assert dataset_manifest["announcement_event_tags"] == ["earnings_negative"]

    manifest = json.loads(Path(paths["research_manifest"]).read_text(encoding="utf-8"))
    assert manifest["default_switch_gate"]["passed"] is False
    assert manifest["info_manifest"]
    assert manifest["info_shadow_report"]
    assert manifest["info_hash"]
    assert manifest["use_us_index_context"] is True
    assert manifest["us_index_source"] == "akshare"
    assert manifest["generator_version"] == "dynamic_universe_v2_leaders"
    assert manifest["generator_hash"]
    assert manifest["selected_pool_size"] == 3
    assert len(manifest["theme_allocations"]) >= 1
    assert not (tmp_path / "swing_v2" / "latest_research_manifest.json").exists()
    assert (tmp_path / "swing_v2" / "latest_research_manifest.generated_80.json").exists()
    info_manifest = json.loads(Path(paths["info_manifest"]).read_text(encoding="utf-8"))
    assert info_manifest["info_item_count"] == 3
    assert info_manifest["info_type_counts"]["news"] == 1
    assert info_manifest["info_type_counts"]["announcement"] == 1
    assert info_manifest["info_type_counts"]["research"] == 1
    assert info_manifest["info_source_breakdown"]["market_news"] == 1
    assert info_manifest["info_source_breakdown"]["announcements"] == 1
    assert info_manifest["info_source_breakdown"]["research"] == 1


def test_daily_run_can_consume_published_relative_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "universe_smoke_5.json").write_text('["AAA","BBB"]', encoding="utf-8")

    state = _make_state()
    step = _TrajectoryStep(
        date=pd.Timestamp("2026-03-01"),
        next_date=pd.Timestamp("2026-03-02"),
        composite_state=state,
        stock_states=list(state.stocks),
        horizon_metrics={},
    )
    trajectory = _BacktestTrajectory(
        prepared=SimpleNamespace(
            settings={},
            market_valid=pd.DataFrame(),
            panel=pd.DataFrame(),
            market_feature_cols=[],
            feature_cols=[],
            dates=[pd.Timestamp("2026-03-01")],
        ),
        steps=[step],
    )
    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", lambda **_: trajectory)
    monkeypatch.setattr(
        "src.application.v2_services._split_research_trajectory",
        lambda trajectory, *args, **kwargs: (trajectory, trajectory, trajectory),
    )
    monkeypatch.setattr("src.application.v2_services._build_frozen_linear_forecast_bundle", lambda prepared: {})

    baseline = _make_backtest(0.20, 0.18)
    calibrated = _make_backtest(0.22, 0.20)
    learned = _make_backtest(0.24, 0.22)
    paths = publish_v2_research_artifacts(
        strategy_id="swing_v2",
        artifact_root="artifacts/v2",
        cache_root="artifacts/v2/cache",
        settings={
            "config_path": "config/api.json",
            "source": "local",
            "watchlist": "config/watchlist.json",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 5,
            "universe_tier": "favorites_16",
            "universe_id": "favorites_16",
            "universe_size": 2,
            "universe_generation_rule": "manual_favorites_locked",
            "source_universe_manifest_path": "config/universe_smoke_5.json",
            "symbols": ["AAA", "BBB"],
            "symbol_count": 2,
            "use_us_index_context": True,
            "us_index_source": "akshare",
            "start": "2024-01-01",
            "end": "2026-03-01",
        },
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
                exposure_intercept=0.55,
                exposure_coef=[0.1],
                position_intercept=2.5,
                position_coef=[0.05],
                turnover_intercept=0.20,
                turnover_coef=[0.01],
                train_rows=88,
                train_r2_exposure=0.33,
                train_r2_positions=0.25,
                train_r2_turnover=0.19,
            ),
            baseline=baseline,
            learned=learned,
        ),
        publish_forecast_models=True,
    )

    monkeypatch.setattr(
        "src.application.v2_services._load_v2_runtime_settings",
        lambda **_: {
            "config_path": "config/api.json",
            "watchlist": "config/watchlist.json",
            "source": "local",
            "data_dir": "data",
            "start": "2024-01-01",
            "end": "2026-03-01",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 5,
            "universe_tier": "favorites_16",
            "use_info_fusion": False,
            "use_us_index_context": True,
            "us_index_source": "akshare",
        },
    )
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda settings, cache_root: settings)
    monkeypatch.setattr(
        "src.application.v2_services.load_watchlist",
        lambda _: (SimpleNamespace(symbol="000001.SH"), [], {}),
    )
    monkeypatch.setattr(
        "src.application.v2_services.build_candidate_universe",
        lambda **_: SimpleNamespace(rows=[]),
    )

    result = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root="artifacts/v2",
        cache_root="artifacts/v2/cache",
        snapshot_path=paths["research_manifest"],
    )

    assert result.snapshot.run_id == paths["run_id"]
    assert str(result.composite_state.market.as_of_date)
    assert result.policy_decision.symbol_target_weights


def test_daily_run_reports_clear_error_when_frozen_state_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "v2" / "swing_v2" / "20260311_100000"
    run_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = run_dir / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "universe_file": "config/universe_smoke_5.json",
                "universe_limit": 5,
                "universe_tier": "favorites_16",
                "universe_id": "favorites_16",
                "universe_size": 2,
                "universe_generation_rule": "manual_favorites_locked",
                "source_universe_manifest_path": "config/universe_smoke_5.json",
                "symbols": ["AAA", "BBB"],
                "symbol_count": 2,
                "use_us_index_context": True,
                "us_index_source": "akshare",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "research_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "20260311_100000",
                "strategy_id": "swing_v2",
                "dataset_manifest": str(dataset_path),
                "frozen_daily_state": "artifacts/v2/swing_v2/20260311_100000/missing_frozen_daily_state.json",
                "config_hash": "cfg",
                "snapshot_hash": "snap",
                "policy_hash": "policy",
                "universe_hash": "uni",
                "model_hashes": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.application.v2_services._load_v2_runtime_settings",
        lambda **_: {
            "config_path": "config/api.json",
            "watchlist": "config/watchlist.json",
            "source": "local",
            "data_dir": "data",
            "start": "2024-01-01",
            "end": "2026-03-01",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 5,
            "universe_tier": "favorites_16",
            "use_info_fusion": False,
            "use_us_index_context": True,
            "us_index_source": "akshare",
        },
    )
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda settings, cache_root: settings)
    monkeypatch.setattr(
        "src.application.v2_services.load_watchlist",
        lambda _: (SimpleNamespace(symbol="000001.SH"), [], {}),
    )
    monkeypatch.setattr(
        "src.application.v2_services.build_candidate_universe",
        lambda **_: SimpleNamespace(rows=[]),
    )

    with pytest.raises(RuntimeError, match="does not contain usable frozen daily state"):
        run_daily_v2_live(
            strategy_id="swing_v2",
            artifact_root=str(tmp_path / "artifacts" / "v2"),
            cache_root=str(tmp_path / "artifacts" / "v2" / "cache"),
            snapshot_path=str(manifest_path),
        )


def test_daily_snapshot_context_bypasses_manifest_on_explicit_universe_override_with_retrain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested_universe = tmp_path / "universe_300.json"
    requested_universe.write_text(json.dumps({"stocks": []}, ensure_ascii=False), encoding="utf-8")
    manifest_universe = tmp_path / "favorites_16.json"
    manifest_universe.write_text(json.dumps({"stocks": []}, ensure_ascii=False), encoding="utf-8")
    dataset_path = tmp_path / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "universe_file": str(manifest_universe),
                "universe_limit": 16,
                "universe_tier": "favorites_16",
                "universe_id": "favorites_16",
                "universe_size": 16,
                "symbol_count": 16,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "research_manifest.json"
    manifest = {
        "run_id": "20260312_100000",
        "strategy_id": "swing_v2",
        "dataset_manifest": str(dataset_path),
        "config_hash": "cfg",
        "snapshot_hash": "snap",
        "policy_hash": "policy",
        "universe_hash": "uni",
        "model_hashes": {},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        "src.application.v2_services._load_v2_runtime_settings",
        lambda **_: {
            "config_path": "config/api.json",
            "watchlist": "config/watchlist.json",
            "source": "local",
            "data_dir": "data",
            "start": "2024-01-01",
            "end": "2026-03-01",
            "universe_file": str(requested_universe),
            "universe_limit": 300,
            "universe_tier": "",
            "universe_id": "universe_300",
            "universe_size": 300,
            "symbol_count": 300,
            "use_info_fusion": False,
            "use_us_index_context": False,
            "external_signals": True,
        },
    )
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda settings, cache_root: settings)
    monkeypatch.setattr(
        "src.application.v2_services._load_research_manifest_for_daily",
        lambda **_: (manifest, manifest_path),
    )

    ctx = _build_daily_snapshot_context(
        strategy_id="swing_v2",
        config_path="config/api.json",
        source="local",
        universe_file=str(requested_universe),
        universe_limit=300,
        universe_tier=None,
        info_file=None,
        info_lookback_days=None,
        info_half_life_days=None,
        use_info_fusion=None,
        info_shadow_only=None,
        info_types=None,
        info_source_mode=None,
        info_subsets=None,
        external_signals=True,
        event_file=None,
        capital_flow_file=None,
        macro_file=None,
        use_us_index_context=False,
        us_index_source=None,
        artifact_root=str(tmp_path),
        cache_root=str(tmp_path / "cache"),
        run_id=None,
        snapshot_path=str(manifest_path),
        allow_retrain=True,
    )

    assert ctx.manifest == {}
    assert ctx.manifest_path is None
    assert ctx.snapshot.universe_id == "universe_300"
    assert ctx.snapshot.run_id == ""


def test_daily_snapshot_context_still_raises_on_explicit_universe_override_without_retrain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    requested_universe = tmp_path / "universe_300.json"
    requested_universe.write_text(json.dumps({"stocks": []}, ensure_ascii=False), encoding="utf-8")
    manifest_universe = tmp_path / "favorites_16.json"
    manifest_universe.write_text(json.dumps({"stocks": []}, ensure_ascii=False), encoding="utf-8")
    dataset_path = tmp_path / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "universe_file": str(manifest_universe),
                "universe_limit": 16,
                "universe_tier": "favorites_16",
                "universe_id": "favorites_16",
                "universe_size": 16,
                "symbol_count": 16,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "research_manifest.json"
    manifest = {
        "run_id": "20260312_100000",
        "strategy_id": "swing_v2",
        "dataset_manifest": str(dataset_path),
        "config_hash": "cfg",
        "snapshot_hash": "snap",
        "policy_hash": "policy",
        "universe_hash": "uni",
        "model_hashes": {},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    monkeypatch.setattr(
        "src.application.v2_services._load_v2_runtime_settings",
        lambda **_: {
            "config_path": "config/api.json",
            "watchlist": "config/watchlist.json",
            "source": "local",
            "data_dir": "data",
            "start": "2024-01-01",
            "end": "2026-03-01",
            "universe_file": str(requested_universe),
            "universe_limit": 300,
            "universe_tier": "",
            "universe_id": "universe_300",
            "universe_size": 300,
            "symbol_count": 300,
            "use_info_fusion": False,
            "use_us_index_context": False,
            "external_signals": True,
        },
    )
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda settings, cache_root: settings)
    monkeypatch.setattr(
        "src.application.v2_services._load_research_manifest_for_daily",
        lambda **_: (manifest, manifest_path),
    )

    with pytest.raises(ValueError, match="universe file mismatch"):
        _build_daily_snapshot_context(
            strategy_id="swing_v2",
            config_path="config/api.json",
            source="local",
            universe_file=str(requested_universe),
            universe_limit=300,
            universe_tier=None,
            info_file=None,
            info_lookback_days=None,
            info_half_life_days=None,
            use_info_fusion=None,
            info_shadow_only=None,
            info_types=None,
            info_source_mode=None,
            info_subsets=None,
            external_signals=True,
            event_file=None,
            capital_flow_file=None,
            macro_file=None,
            use_us_index_context=False,
            us_index_source=None,
            artifact_root=str(tmp_path),
            cache_root=str(tmp_path / "cache"),
            run_id=None,
            snapshot_path=str(manifest_path),
            allow_retrain=False,
        )


def test_publish_research_artifacts_freezes_external_signal_states(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    info_root = tmp_path / "input" / "info_parts"
    (info_root / "market_news").mkdir(parents=True, exist_ok=True)
    (info_root / "market_news" / "core.csv").write_text(
        "date,target_type,target,direction,title,info_type\n"
        "2024-12-28,market,MARKET,bullish,市场情绪修复,news\n",
        encoding="utf-8",
    )
    capital_flow_path = tmp_path / "capital_flow.csv"
    capital_flow_path.write_text(
        "date,northbound_net_flow,margin_balance_change,turnover_heat,large_order_bias\n"
        "2024-12-28,0.35,0.20,0.72,0.25\n",
        encoding="utf-8",
    )
    macro_path = tmp_path / "macro.csv"
    macro_path.write_text(
        "date,style_regime,commodity_pressure,fx_pressure,index_breadth_proxy\n"
        "2024-12-28,quality,0.20,0.15,0.64\n",
        encoding="utf-8",
    )

    state = _make_state()
    trajectory = _BacktestTrajectory(
        prepared=SimpleNamespace(
            stock_frames={},
            settings={"source": "local", "universe_tier": "favorites_16"},
        ),
        steps=[
            _TrajectoryStep(
                date=pd.Timestamp("2026-02-28"),
                next_date=pd.Timestamp("2026-03-03"),
                composite_state=state,
                stock_states=list(state.stocks),
                horizon_metrics={},
            )
        ],
    )
    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", lambda **_: trajectory)
    monkeypatch.setattr("src.application.v2_services._split_research_trajectory", lambda *args, **kwargs: (trajectory, trajectory, trajectory))

    paths = publish_v2_research_artifacts(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts" / "v2"),
        cache_root=str(tmp_path / "artifacts" / "v2" / "cache"),
        baseline=_make_backtest(0.10, 0.08),
        calibration=V2CalibrationResult(best_policy=PolicySpec(), best_score=0.2, baseline=_make_backtest(0.10, 0.08), calibrated=_make_backtest(0.11, 0.09)),
        learning=V2PolicyLearningResult(model=LearnedPolicyModel(feature_names=["x1"], exposure_intercept=0.5, exposure_coef=[0.0], position_intercept=2.0, position_coef=[0.0], turnover_intercept=0.2, turnover_coef=[0.0], train_rows=1, train_r2_exposure=0.0, train_r2_positions=0.0, train_r2_turnover=0.0), baseline=_make_backtest(0.10, 0.08), learned=_make_backtest(0.12, 0.10)),
        settings={
            "config_path": "config/api.json",
            "source": "local",
            "watchlist": "config/watchlist.json",
            "universe_tier": "favorites_16",
            "universe_id": "favorites_16",
            "universe_size": 2,
            "universe_generation_rule": "manual",
            "source_universe_manifest_path": "config/universe_smoke_5.json",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 2,
            "start": "2024-01-01",
            "end": "2026-03-01",
            "symbols": ["AAA", "BBB"],
            "symbol_count": 2,
            "info_file": str(info_root),
            "event_file": str(info_root),
            "use_info_fusion": False,
            "info_shadow_only": False,
            "info_source_mode": "layered",
            "info_subsets": ["market_news"],
            "announcement_event_tags": ["earnings_negative"],
            "capital_flow_file": str(capital_flow_path),
            "macro_file": str(macro_path),
            "external_signals": True,
            "external_signal_version": "v1",
            "use_us_index_context": False,
            "us_index_source": "akshare",
        },
    )

    frozen_payload = json.loads(Path(paths["frozen_daily_state"]).read_text(encoding="utf-8"))
    composite_payload = frozen_payload["composite_state"]
    assert composite_payload["capital_flow_state"]["flow_regime"] == "strong_inflow"
    assert composite_payload["macro_context_state"]["style_regime"] == "quality"

    manifest_payload = json.loads(Path(paths["research_manifest"]).read_text(encoding="utf-8"))
    assert manifest_payload["external_signal_enabled"] is True
    assert manifest_payload["external_signal_manifest"]


def test_daily_run_prefers_frozen_external_signal_states(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "v2" / "swing_v2" / "20260311_100100"
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_state = {
        "as_of_date": "2026-03-01",
        "next_date": "2026-03-02",
        "composite_state": {
            "market": asdict(_make_state().market),
            "cross_section": asdict(_make_state().cross_section),
            "sectors": [asdict(item) for item in _make_state().sectors],
            "stocks": [asdict(item) for item in _make_state().stocks],
            "strategy_mode": "trend_follow",
            "risk_regime": "risk_on",
            "market_info_state": asdict(_make_state().market_info_state),
            "sector_info_states": {},
            "stock_info_states": {key: asdict(value) for key, value in _make_state().stock_info_states.items()},
            "capital_flow_state": {"northbound_net_flow": 0.42, "margin_balance_change": 0.16, "turnover_heat": 0.75, "large_order_bias": 0.22, "flow_regime": "strong_inflow"},
            "macro_context_state": {"style_regime": "quality", "commodity_pressure": 0.18, "fx_pressure": 0.12, "index_breadth_proxy": 0.67, "macro_risk_level": "neutral"},
        },
    }
    frozen_path = run_dir / "frozen_daily_state.json"
    frozen_path.write_text(json.dumps(frozen_state, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_path = run_dir / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "universe_tier": "favorites_16",
                "universe_id": "favorites_16",
                "universe_size": 2,
                "universe_generation_rule": "manual",
                "source_universe_manifest_path": "config/universe_smoke_5.json",
                "universe_file": "config/universe_smoke_5.json",
                "universe_limit": 2,
                "symbols": ["AAA", "BBB"],
                "symbol_count": 2,
                "external_signal_enabled": True,
                "external_signal_version": "v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "research_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "20260311_100100",
                "strategy_id": "swing_v2",
                "universe_tier": "favorites_16",
                "dataset_manifest": str(dataset_path),
                "frozen_daily_state": str(frozen_path),
                "learned_policy_model": "",
                "external_signal_enabled": True,
                "external_signal_version": "v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: {"watchlist": "config/watchlist.json", "source": "local", "data_dir": "data", "universe_file": "config/universe_smoke_5.json", "universe_limit": 2, "use_info_fusion": False, "external_signals": True, "universe_tier": "favorites_16"})
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda **kwargs: dict(kwargs["settings"]))
    monkeypatch.setattr("src.application.v2_services.load_watchlist", lambda *_: (SimpleNamespace(symbol="000001.SH"), [], {}))
    monkeypatch.setattr("src.application.v2_services.build_candidate_universe", lambda **_: SimpleNamespace(rows=[]))

    result = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts" / "v2"),
        cache_root=str(tmp_path / "artifacts" / "v2" / "cache"),
        snapshot_path=str(manifest_path),
    )

    assert result.composite_state.capital_flow_state.flow_regime == "strong_inflow"
    assert result.composite_state.macro_context_state.style_regime == "quality"
    assert result.external_signal_enabled is True


def test_daily_run_refreshes_market_snapshot_for_reporting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "v2" / "swing_v2" / "20260311_100101"
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_state = {
        "as_of_date": "2026-03-01",
        "next_date": "2026-03-02",
        "composite_state": {
            "market": asdict(_make_state().market),
            "cross_section": asdict(_make_state().cross_section),
            "sectors": [asdict(item) for item in _make_state().sectors],
            "stocks": [asdict(item) for item in _make_state().stocks],
            "strategy_mode": "trend_follow",
            "risk_regime": "risk_on",
            "market_info_state": asdict(_make_state().market_info_state),
            "sector_info_states": {},
            "stock_info_states": {key: asdict(value) for key, value in _make_state().stock_info_states.items()},
            "capital_flow_state": asdict(_make_state().capital_flow_state),
            "macro_context_state": asdict(_make_state().macro_context_state),
        },
    }
    frozen_path = run_dir / "frozen_daily_state.json"
    frozen_path.write_text(json.dumps(frozen_state, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_path = run_dir / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "universe_tier": "favorites_16",
                "universe_id": "favorites_16",
                "universe_size": 2,
                "universe_generation_rule": "manual",
                "source_universe_manifest_path": "config/universe_smoke_5.json",
                "universe_file": "config/universe_smoke_5.json",
                "universe_limit": 2,
                "symbols": ["AAA", "BBB"],
                "symbol_count": 2,
                "external_signal_enabled": False,
                "external_signal_version": "v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "research_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "20260311_100101",
                "strategy_id": "swing_v2",
                "universe_tier": "favorites_16",
                "dataset_manifest": str(dataset_path),
                "frozen_daily_state": str(frozen_path),
                "learned_policy_model": "",
                "external_signal_enabled": False,
                "external_signal_version": "v1",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.application.v2_services._load_v2_runtime_settings",
        lambda **_: {
            "watchlist": "config/watchlist.json",
            "source": "local",
            "data_dir": "data",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 2,
            "use_info_fusion": False,
            "external_signals": False,
            "universe_tier": "favorites_16",
            "start": "2024-01-01",
            "end": "2026-03-12",
            "use_margin_features": False,
            "use_us_index_context": False,
        },
    )
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda **kwargs: dict(kwargs["settings"]))
    monkeypatch.setattr("src.application.v2_services.load_watchlist", lambda *_: (SimpleNamespace(symbol="000001.SH"), [], {}))
    monkeypatch.setattr("src.application.v2_services.build_candidate_universe", lambda **_: SimpleNamespace(rows=[]))
    monkeypatch.setattr(
        "src.application.v2_services._build_market_and_cross_section_states",
        lambda **_: (
            MarketForecastState(
                as_of_date="2026-03-12",
                up_1d_prob=0.58,
                up_2d_prob=0.59,
                up_3d_prob=0.60,
                up_5d_prob=0.61,
                up_10d_prob=0.62,
                up_20d_prob=0.63,
                trend_state="trend",
                drawdown_risk=0.11,
                volatility_regime="normal",
                liquidity_stress=0.18,
                latest_close=3250.0,
                market_facts=MarketFactsState(
                    sample_coverage=3000,
                    advancers=2100,
                    decliners=760,
                    flats=140,
                    limit_up_count=82,
                    limit_down_count=7,
                    new_high_count=224,
                    new_low_count=26,
                    median_return=0.008,
                    sample_amount=1.42e12,
                    amount_z20=0.55,
                ),
            ),
            CrossSectionForecastState(
                as_of_date="2026-03-12",
                large_vs_small_bias=0.10,
                growth_vs_value_bias=0.04,
                fund_flow_strength=0.16,
                margin_risk_on_score=0.12,
                breadth_strength=0.31,
                leader_participation=0.66,
                weak_stock_ratio=0.18,
            ),
        ),
    )

    result = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts" / "v2"),
        cache_root=str(tmp_path / "artifacts" / "v2" / "cache"),
        snapshot_path=str(manifest_path),
    )

    assert result.composite_state.market.as_of_date == "2026-03-12"
    assert result.composite_state.market.market_facts.advancers == 2100
    assert result.composite_state.market.market_facts.limit_up_count == 82
    assert result.composite_state.market.sentiment.score > 55.0


def test_publish_artifacts_write_frozen_forecast_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "universe_smoke_5.json").write_text('["AAA","BBB"]', encoding="utf-8")

    state = _make_state()
    step = _TrajectoryStep(
        date=pd.Timestamp("2026-03-01"),
        next_date=pd.Timestamp("2026-03-02"),
        composite_state=state,
        stock_states=list(state.stocks),
        horizon_metrics={},
    )
    trajectory = _BacktestTrajectory(
        prepared=SimpleNamespace(
            settings={},
            market_valid=pd.DataFrame(),
            panel=pd.DataFrame(),
            market_feature_cols=[],
            feature_cols=[],
            dates=[pd.Timestamp("2026-03-01")],
        ),
        steps=[step],
    )
    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", lambda **_: trajectory)
    monkeypatch.setattr(
        "src.application.v2_services._split_research_trajectory",
        lambda trajectory, *args, **kwargs: (trajectory, trajectory, trajectory),
    )
    monkeypatch.setattr(
        "src.application.v2_services._build_frozen_linear_forecast_bundle",
        lambda prepared: {
            "backend": "linear",
            "market_models": {"1d": {"model_type": "logistic_linear"}},
            "stock_models": {"1d": {"model_type": "logistic_linear"}},
        },
    )

    paths = publish_v2_research_artifacts(
        strategy_id="swing_v2",
        artifact_root="artifacts/v2",
        cache_root="artifacts/v2/cache",
        settings={
            "config_path": "config/api.json",
            "source": "local",
            "watchlist": "config/watchlist.json",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 5,
            "universe_tier": "favorites_16",
            "universe_id": "favorites_16",
            "universe_size": 2,
            "universe_generation_rule": "manual_favorites_locked",
            "source_universe_manifest_path": "config/universe_smoke_5.json",
            "symbols": ["AAA", "BBB"],
            "symbol_count": 2,
            "use_us_index_context": True,
            "us_index_source": "akshare",
            "start": "2024-01-01",
            "end": "2026-03-01",
        },
        baseline=_make_backtest(0.20, 0.18),
        calibration=V2CalibrationResult(
            best_policy=PolicySpec(),
            best_score=0.12,
            baseline=_make_backtest(0.20, 0.18),
            calibrated=_make_backtest(0.22, 0.20),
            trials=[],
        ),
        learning=V2PolicyLearningResult(
            model=LearnedPolicyModel(
                feature_names=["x1"],
                exposure_intercept=0.55,
                exposure_coef=[0.1],
                position_intercept=2.5,
                position_coef=[0.05],
                turnover_intercept=0.20,
                turnover_coef=[0.01],
                train_rows=88,
                train_r2_exposure=0.33,
                train_r2_positions=0.25,
                train_r2_turnover=0.19,
            ),
            baseline=_make_backtest(0.20, 0.18),
            learned=_make_backtest(0.24, 0.22),
        ),
        publish_forecast_models=True,
    )

    bundle_path = Path(paths["frozen_forecast_bundle"])
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "linear"
    assert "market_models" in payload
    assert "stock_models" in payload


def test_daily_run_prefers_frozen_forecast_bundle_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts" / "v2" / "swing_v2" / "20260311_100102"
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen_state = {
        "as_of_date": "2026-03-01",
        "next_date": "2026-03-02",
        "composite_state": {
            "market": asdict(_make_state().market),
            "cross_section": asdict(_make_state().cross_section),
            "sectors": [asdict(item) for item in _make_state().sectors],
            "stocks": [asdict(item) for item in _make_state().stocks],
            "strategy_mode": "trend_follow",
            "risk_regime": "risk_on",
        },
    }
    frozen_path = run_dir / "frozen_daily_state.json"
    frozen_path.write_text(json.dumps(frozen_state, ensure_ascii=False, indent=2), encoding="utf-8")
    frozen_bundle_path = run_dir / "frozen_forecast_bundle.json"
    frozen_bundle_path.write_text(json.dumps({"backend": "linear"}, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset_path = run_dir / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "universe_tier": "favorites_16",
                "universe_id": "favorites_16",
                "universe_size": 2,
                "universe_generation_rule": "manual",
                "source_universe_manifest_path": "config/universe_smoke_5.json",
                "universe_file": "config/universe_smoke_5.json",
                "universe_limit": 2,
                "symbols": ["AAA", "BBB"],
                "symbol_count": 2,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path = run_dir / "research_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "20260311_100102",
                "strategy_id": "swing_v2",
                "universe_tier": "favorites_16",
                "dataset_manifest": str(dataset_path),
                "frozen_daily_state": str(frozen_path),
                "frozen_forecast_bundle": str(frozen_bundle_path),
                "learned_policy_model": "",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.application.v2_services._load_v2_runtime_settings",
        lambda **_: {
            "watchlist": "config/watchlist.json",
            "source": "local",
            "data_dir": "data",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 2,
            "use_info_fusion": False,
            "external_signals": False,
            "universe_tier": "favorites_16",
        },
    )
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda **kwargs: dict(kwargs["settings"]))
    monkeypatch.setattr("src.application.v2_services.load_watchlist", lambda *_: (SimpleNamespace(symbol="000001.SH"), [], {}))
    monkeypatch.setattr("src.application.v2_services.build_candidate_universe", lambda **_: SimpleNamespace(rows=[]))
    monkeypatch.setattr("src.application.v2_services._load_frozen_forecast_bundle", lambda *_: {"backend": "linear"})
    monkeypatch.setattr(
        "src.application.v2_services._score_live_composite_state_from_frozen_bundle",
        lambda **_: (
            CompositeState(
                market=replace(_make_state().market, as_of_date="2026-03-12"),
                cross_section=_make_state().cross_section,
                sectors=_make_state().sectors,
                stocks=_make_state().stocks,
                strategy_mode="trend_follow",
                risk_regime="risk_on",
            ),
            [],
        ),
    )

    result = run_daily_v2_live(
        strategy_id="swing_v2",
        artifact_root=str(tmp_path / "artifacts" / "v2"),
        cache_root=str(tmp_path / "artifacts" / "v2" / "cache"),
        snapshot_path=str(manifest_path),
    )

    assert result.composite_state.market.as_of_date == "2026-03-12"


def test_backtest_summary_carries_cross_section_metrics() -> None:
    summary = _make_backtest(0.18, 0.16)

    assert summary.avg_rank_ic == 0.0
    assert summary.avg_top_decile_return == 0.0
    assert summary.avg_top_bottom_spread == 0.0
    assert summary.avg_top_k_hit_rate == 0.0
    assert summary.horizon_metrics == {}


def test_backtest_summary_accepts_multi_horizon_metrics_payload() -> None:
    summary = V2BacktestSummary(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_days=120,
        total_return=0.12,
        annual_return=0.10,
        max_drawdown=-0.08,
        avg_turnover=0.18,
        total_cost=0.01,
        horizon_metrics={
            "1d": {"rank_ic": 0.11, "top_decile_return": 0.002, "top_bottom_spread": 0.004, "top_k_hit_rate": 0.52},
            "5d": {"rank_ic": 0.09, "top_decile_return": 0.006, "top_bottom_spread": 0.010, "top_k_hit_rate": 0.55},
            "20d": {"rank_ic": 0.07, "top_decile_return": 0.018, "top_bottom_spread": 0.025, "top_k_hit_rate": 0.57},
        },
    )

    assert set(summary.horizon_metrics) == {"1d", "5d", "20d"}
    assert summary.horizon_metrics["5d"]["top_k_hit_rate"] == 0.55


def test_policy_objective_prefers_excess_and_ir_over_raw_return() -> None:
    high_beta = V2BacktestSummary(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_days=120,
        total_return=0.20,
        annual_return=0.28,
        max_drawdown=-0.10,
        avg_turnover=0.18,
        total_cost=0.02,
        excess_annual_return=-0.01,
        information_ratio=-0.10,
    )
    cleaner_alpha = V2BacktestSummary(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_days=120,
        total_return=0.16,
        annual_return=0.20,
        max_drawdown=-0.08,
        avg_turnover=0.12,
        total_cost=0.01,
        excess_annual_return=0.08,
        information_ratio=0.70,
    )

    assert _policy_objective_score(cleaner_alpha) > _policy_objective_score(high_beta)


def test_calibrate_v2_policy_runs_expanded_validation_grid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    seen: dict[str, object] = {"calls": 0}

    def fake_backtest(**kwargs: object) -> V2BacktestSummary:
        policy_spec = kwargs.get("policy_spec")
        seen["calls"] = int(seen["calls"]) + 1
        if not isinstance(policy_spec, PolicySpec):
            return baseline
        annual = 0.10 + 0.08 * float(policy_spec.risk_on_exposure) + 0.01 * float(policy_spec.risk_on_positions)
        dd = -0.05 - 0.02 * float(policy_spec.risk_off_exposure)
        return V2BacktestSummary(
            start_date="2024-01-01",
            end_date="2024-12-31",
            n_days=120,
            total_return=annual,
            annual_return=annual,
            max_drawdown=dd,
            avg_turnover=float(policy_spec.risk_on_turnover_cap),
            total_cost=0.01,
            excess_annual_return=0.5 * annual - 0.1 * float(policy_spec.risk_on_turnover_cap),
            information_ratio=0.6 * annual,
        )

    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", fake_backtest)

    result = calibrate_v2_policy(
        strategy_id="swing_v2",
        baseline=baseline,
        trajectory=object(),
        cache_root=str(tmp_path),
    )

    assert len(result.trials) == 27
    assert seen["calls"] == 26
    assert result.best_score >= max(float(item["score"]) for item in result.trials)


def test_calibrate_v2_policy_emits_progress_updates(tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    progress: list[tuple[str, str]] = []

    def fake_backtest(**kwargs: object) -> V2BacktestSummary:
        policy_spec = kwargs.get("policy_spec")
        annual = 0.12
        if isinstance(policy_spec, PolicySpec):
            annual += 0.02 * float(policy_spec.risk_on_exposure)
        return V2BacktestSummary(
            start_date="2024-01-01",
            end_date="2024-12-31",
            n_days=120,
            total_return=annual,
            annual_return=annual,
            max_drawdown=-0.06,
            avg_turnover=0.18,
            total_cost=0.01,
            excess_annual_return=0.08,
            information_ratio=0.50,
        )

    deps = _policy_learning_dependencies(
        run_v2_backtest_live=fake_backtest,
        emit_progress=lambda stage, message: progress.append((stage, message)),
    )

    policy_learning_runtime.calibrate_v2_policy(
        strategy_id="swing_v2",
        baseline=baseline,
        trajectory=object(),
        cache_root=str(tmp_path),
        deps=deps,
    )

    assert any(stage == "calibration" and "开始参数搜索" in message for stage, message in progress)
    assert any(stage == "calibration" and "评估候选" in message for stage, message in progress)


def test_research_workflow_light_mode_skips_heavy_stages(tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    trajectory_sentinel = object()
    train_sentinel = object()
    validation_sentinel = object()
    holdout_sentinel = object()
    seen: dict[str, object] = {}

    def fake_load(**_: object) -> object:
        return trajectory_sentinel

    def fake_split(trajectory: object, *args: object, **kwargs: object) -> tuple[object, object, object]:
        assert trajectory is trajectory_sentinel
        return train_sentinel, validation_sentinel, holdout_sentinel

    def fake_baseline(**kwargs: object) -> V2BacktestSummary:
        seen["baseline_trajectory"] = kwargs.get("trajectory")
        return baseline

    def fail_calibration(**_: object) -> V2CalibrationResult:
        raise AssertionError("light mode should skip calibration")

    def fail_learning(**_: object) -> V2PolicyLearningResult:
        raise AssertionError("light mode should skip learning")

    deps = _research_workflow_dependencies(
        load_or_build_v2_backtest_trajectory_fn=fake_load,
        split_research_trajectory_fn=fake_split,
        run_v2_backtest_live_fn=fake_baseline,
        calibrate_v2_policy_fn=fail_calibration,
        learn_v2_policy_model_fn=fail_learning,
    )

    got_baseline, calibration, learning = _run_v2_research_workflow_runtime(
        dependencies=deps,
        strategy_id="swing_v2",
        skip_calibration=True,
        skip_learning=True,
        cache_root=str(tmp_path),
    )

    assert got_baseline == baseline
    assert seen["baseline_trajectory"] is holdout_sentinel
    assert calibration.calibrated == baseline
    assert calibration.trials
    assert learning.learned == baseline
    assert learning.model.train_rows == 0


def test_research_workflow_emits_stage_progress(tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    trajectory_sentinel = object()
    train_sentinel = object()
    validation_sentinel = object()
    holdout_sentinel = object()
    progress: list[tuple[str, str]] = []

    deps = _research_workflow_dependencies(
        emit_progress_fn=lambda stage, message: progress.append((stage, message)),
        load_or_build_v2_backtest_trajectory_fn=lambda **_: trajectory_sentinel,
        split_research_trajectory_fn=lambda trajectory, *args, **kwargs: (
            train_sentinel,
            validation_sentinel,
            holdout_sentinel,
        ),
        run_v2_backtest_live_fn=lambda **_: baseline,
    )

    _run_v2_research_workflow_runtime(
        dependencies=deps,
        strategy_id="swing_v2",
        skip_calibration=True,
        skip_learning=True,
        cache_root=str(tmp_path),
    )

    assert any(stage == "research" and "载入研究轨迹" in message for stage, message in progress)
    assert any(stage == "research" and "样本切分完成" in message for stage, message in progress)
    assert any(stage == "research" and "已跳过参数搜索" in message for stage, message in progress)
    assert any(stage == "research" and "已跳过学习型策略" in message for stage, message in progress)


def test_research_workflow_reuses_single_trajectory_for_all_stages(tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    calibrated = _make_backtest(0.20, 0.18)
    learned = _make_backtest(0.19, 0.17)
    trajectory_sentinel = object()
    train_sentinel = object()
    validation_sentinel = object()
    holdout_sentinel = object()
    seen: dict[str, object] = {}

    def fake_load(**_: object) -> object:
        return trajectory_sentinel

    def fake_split(trajectory: object, *args: object, **kwargs: object) -> tuple[object, object, object]:
        assert trajectory is trajectory_sentinel
        return train_sentinel, validation_sentinel, holdout_sentinel

    def fake_baseline(**kwargs: object) -> V2BacktestSummary:
        trajectory = kwargs.get("trajectory")
        seen.setdefault("baseline_trajectories", []).append(trajectory)
        if trajectory is holdout_sentinel and len(seen["baseline_trajectories"]) >= 2:
            return calibrated
        if trajectory is validation_sentinel:
            return _make_backtest(0.17, 0.15)
        return baseline

    def fake_calibration(**kwargs: object) -> V2CalibrationResult:
        seen["calibration_trajectory"] = kwargs.get("trajectory")
        seen["calibration_baseline"] = kwargs.get("baseline")
        return V2CalibrationResult(
            best_policy=PolicySpec(),
            best_score=0.11,
            baseline=kwargs.get("baseline"),
            calibrated=calibrated,
            trials=[],
        )

    def fake_learning(**kwargs: object) -> V2PolicyLearningResult:
        seen["learning_trajectory"] = kwargs.get("trajectory")
        seen["learning_fit_trajectory"] = kwargs.get("fit_trajectory")
        seen["learning_eval_trajectory"] = kwargs.get("evaluation_trajectory")
        seen["learning_baseline"] = kwargs.get("baseline")
        return V2PolicyLearningResult(
            model=LearnedPolicyModel(
                feature_names=["x1"],
                exposure_intercept=0.5,
                exposure_coef=[0.1],
                position_intercept=3.0,
                position_coef=[0.0],
                turnover_intercept=0.2,
                turnover_coef=[0.0],
                train_rows=10,
                train_r2_exposure=0.1,
                train_r2_positions=0.1,
                train_r2_turnover=0.1,
            ),
            baseline=baseline,
            learned=learned,
        )

    deps = _research_workflow_dependencies(
        load_or_build_v2_backtest_trajectory_fn=fake_load,
        split_research_trajectory_fn=fake_split,
        run_v2_backtest_live_fn=fake_baseline,
        calibrate_v2_policy_fn=fake_calibration,
        learn_v2_policy_model_fn=fake_learning,
    )

    got_baseline, got_calibration, got_learning = _run_v2_research_workflow_runtime(
        dependencies=deps,
        strategy_id="swing_v2",
        cache_root=str(tmp_path),
    )

    assert got_baseline == baseline
    assert got_calibration.calibrated == calibrated
    assert got_learning.learned == learned
    assert seen["baseline_trajectories"] == [holdout_sentinel, validation_sentinel, holdout_sentinel]
    assert seen["calibration_trajectory"] is validation_sentinel
    assert seen["learning_trajectory"] is holdout_sentinel
    assert seen["learning_fit_trajectory"] is validation_sentinel
    assert seen["learning_eval_trajectory"] is holdout_sentinel
    assert seen["calibration_baseline"] == _make_backtest(0.17, 0.15)
    assert seen["learning_baseline"] == baseline


def test_research_workflow_passes_deep_backend(tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    trajectory_sentinel = object()
    train_sentinel = object()
    validation_sentinel = object()
    holdout_sentinel = object()
    seen: dict[str, object] = {}

    def fake_load(**_: object) -> object:
        return trajectory_sentinel

    def fake_split(trajectory: object, *args: object, **kwargs: object) -> tuple[object, object, object]:
        assert trajectory is trajectory_sentinel
        return train_sentinel, validation_sentinel, holdout_sentinel

    def fake_baseline(**kwargs: object) -> V2BacktestSummary:
        seen["forecast_backend"] = kwargs.get("forecast_backend")
        seen["trajectory"] = kwargs.get("trajectory")
        return baseline

    deps = _research_workflow_dependencies(
        load_or_build_v2_backtest_trajectory_fn=fake_load,
        split_research_trajectory_fn=fake_split,
        run_v2_backtest_live_fn=fake_baseline,
    )

    got_baseline, _, _ = _run_v2_research_workflow_runtime(
        dependencies=deps,
        strategy_id="swing_v2",
        forecast_backend="deep",
        skip_calibration=True,
        skip_learning=True,
        cache_root=str(tmp_path),
    )

    assert got_baseline == baseline
    assert seen["forecast_backend"] == "deep"
    assert seen["trajectory"] is holdout_sentinel


def test_load_or_build_v2_backtest_trajectory_uses_disk_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    built_trajectory = {"steps": 3}
    seen = {"build_calls": 0}

    def fake_prepare(**_: object) -> object:
        return {"prepared": True}

    def fake_build(prepared: object, *, retrain_days: int = 20, forecast_backend: str = "linear") -> object:
        assert prepared == {"prepared": True}
        assert retrain_days == 20
        assert forecast_backend == "linear"
        seen["build_calls"] += 1
        return built_trajectory

    monkeypatch.setattr("src.application.v2_services._prepare_v2_backtest_data", fake_prepare)
    monkeypatch.setattr("src.application.v2_services._build_v2_backtest_trajectory_from_prepared", fake_build)

    first = _load_or_build_v2_backtest_trajectory(
        config_path="config/api.json",
        cache_root=str(tmp_path),
        forecast_backend="linear",
    )
    second = _load_or_build_v2_backtest_trajectory(
        config_path="config/api.json",
        cache_root=str(tmp_path),
        forecast_backend="linear",
    )

    assert first == built_trajectory
    assert second == built_trajectory
    assert seen["build_calls"] == 1


def test_prepare_v2_backtest_data_uses_prepared_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    calls = {"universe": 0, "market": 0, "panel": 0}

    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: dict(settings))
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda settings, cache_root: dict(settings))
    monkeypatch.setattr("src.application.v2_services.load_watchlist", lambda _: (SimpleNamespace(symbol="MKT"), None, None))

    def fake_universe(**_: object) -> SimpleNamespace:
        calls["universe"] += 1
        return SimpleNamespace(rows=[SimpleNamespace(symbol="AAA"), SimpleNamespace(symbol="BBB")])

    def fake_market_raw(**_: object) -> pd.DataFrame:
        calls["market"] += 1
        return pd.DataFrame({"date": dates})

    def fake_market_frame(_: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": dates,
                "mkt_feature": [0.1, 0.2, 0.3, 0.4, 0.5],
                "mkt_target_1d_up": [1, 0, 1, 1, 0],
                "mkt_target_2d_up": [1, 1, 0, 1, 0],
                "mkt_target_3d_up": [1, 1, 1, 0, 0],
                "mkt_target_5d_up": [1, 0, 1, 0, 1],
                "mkt_target_20d_up": [1, 1, 1, 1, 0],
            }
        )

    def fake_panel(**_: object) -> SimpleNamespace:
        calls["panel"] += 1
        frame = pd.DataFrame(
            {
                "date": list(dates) * 2,
                "symbol": ["AAA"] * len(dates) + ["BBB"] * len(dates),
                "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        return SimpleNamespace(frame=frame, feature_columns=["feature_a"])

    monkeypatch.setattr("src.application.v2_services.build_candidate_universe", fake_universe)
    monkeypatch.setattr("src.application.v2_services.load_symbol_daily", fake_market_raw)
    monkeypatch.setattr("src.application.v2_services.make_market_feature_frame", fake_market_frame)
    monkeypatch.setattr(
        "src.application.v2_services.build_market_context_features",
        lambda **_: SimpleNamespace(frame=pd.DataFrame({"date": dates}), feature_columns=[]),
    )
    monkeypatch.setattr("src.application.v2_services.build_stock_panel_dataset", fake_panel)
    monkeypatch.setattr("src.application.v2_services.MARKET_FEATURE_COLUMNS", ["mkt_feature"])

    first = _prepare_v2_backtest_data(config_path="config/api.json", cache_root=str(tmp_path))
    second = _prepare_v2_backtest_data(config_path="config/api.json", cache_root=str(tmp_path))

    assert first is not None
    assert second is not None
    assert first.dates == second.dates
    assert calls["universe"] == 1
    assert calls["market"] == 1
    assert calls["panel"] == 1


def test_prepare_v2_backtest_data_refresh_cache_rebuilds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    calls = {"panel": 0}

    monkeypatch.setattr("src.application.v2_services._load_v2_runtime_settings", lambda **_: dict(settings))
    monkeypatch.setattr("src.application.v2_services._resolve_v2_universe_settings", lambda settings, cache_root: dict(settings))
    monkeypatch.setattr("src.application.v2_services.load_watchlist", lambda _: (SimpleNamespace(symbol="MKT"), None, None))
    monkeypatch.setattr(
        "src.application.v2_services.build_candidate_universe",
        lambda **_: SimpleNamespace(rows=[SimpleNamespace(symbol="AAA")]),
    )
    monkeypatch.setattr("src.application.v2_services.load_symbol_daily", lambda **_: pd.DataFrame({"date": dates}))
    monkeypatch.setattr(
        "src.application.v2_services.make_market_feature_frame",
        lambda _: pd.DataFrame(
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
    )
    monkeypatch.setattr(
        "src.application.v2_services.build_market_context_features",
        lambda **_: SimpleNamespace(frame=pd.DataFrame({"date": dates}), feature_columns=[]),
    )

    def fake_panel(**_: object) -> SimpleNamespace:
        calls["panel"] += 1
        return SimpleNamespace(
            frame=pd.DataFrame({"date": dates, "symbol": ["AAA"] * len(dates), "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5]}),
            feature_columns=["feature_a"],
        )

    monkeypatch.setattr("src.application.v2_services.build_stock_panel_dataset", fake_panel)
    monkeypatch.setattr("src.application.v2_services.MARKET_FEATURE_COLUMNS", ["mkt_feature"])

    first = _prepare_v2_backtest_data(config_path="config/api.json", cache_root=str(tmp_path))
    second = _prepare_v2_backtest_data(config_path="config/api.json", cache_root=str(tmp_path), refresh_cache=True)

    assert first is not None
    assert second is not None
    assert calls["panel"] == 2


def test_split_research_trajectory_purged_mode_applies_embargo() -> None:
    state = _make_state()
    steps = []
    base_date = pd.Timestamp("2024-01-01")
    for idx in range(120):
        step_date = base_date + pd.Timedelta(days=idx)
        next_date = step_date + pd.Timedelta(days=1)
        steps.append(
            _TrajectoryStep(
                date=step_date,
                next_date=next_date,
                composite_state=state,
                stock_states=[],
                horizon_metrics={},
            )
        )
    trajectory = _BacktestTrajectory(prepared=object(), steps=steps)

    train, validation, holdout = _split_research_trajectory(
        trajectory,
        split_mode="purged_wf",
        embargo_days=20,
    )

    assert train.steps
    assert validation.steps
    assert holdout.steps
    assert (validation.steps[0].date - train.steps[-1].date).days >= 20
    assert (holdout.steps[0].date - validation.steps[-1].date).days >= 20


def test_predict_quantile_profiles_vectorizes_and_keeps_monotonic_order() -> None:
    frame = pd.DataFrame({"feature": [1.0, 2.0]})

    class _DummyModel:
        def __init__(self, scale: float) -> None:
            self.scale = scale

        def predict(self, row: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
            return row[feature_cols[0]].astype(float) * self.scale

    out = _predict_quantile_profiles(
        frame,
        feature_cols=["feature"],
        q_models=(
            _DummyModel(0.60),
            _DummyModel(0.30),
            _DummyModel(0.80),
            _DummyModel(0.70),
            _DummyModel(0.90),
        ),
    )

    assert list(out.columns) == ["expected_return", "q10", "q30", "q20", "q50", "q70", "q80", "q90"]
    assert out.loc[0, "q10"] <= out.loc[0, "q30"] <= out.loc[0, "q50"] <= out.loc[0, "q70"] <= out.loc[0, "q90"]
    assert out.loc[1, "q10"] == pytest.approx(1.2)
    assert out.loc[1, "q30"] == pytest.approx(1.2)
    assert out.loc[1, "q50"] == pytest.approx(1.6)
    assert out.loc[1, "q70"] == pytest.approx(1.6)
    assert out.loc[1, "q90"] == pytest.approx(1.8)
    assert out.loc[1, "expected_return"] == pytest.approx(1.5)


def test_build_date_slice_index_returns_contiguous_bounds() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"]),
            "symbol": ["BBB", "AAA", "AAA", "BBB"],
            "value": [2, 1, 3, 4],
        }
    )

    sorted_frame, bounds = _build_date_slice_index(frame, sort_cols=["date", "symbol"])

    first_start, first_end = bounds[pd.Timestamp("2024-01-01")]
    second_start, second_end = bounds[pd.Timestamp("2024-01-02")]

    assert list(sorted_frame["date"]) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-02"),
    ]
    assert list(sorted_frame.iloc[first_start:first_end]["symbol"]) == ["AAA", "BBB"]
    assert list(sorted_frame.iloc[second_start:second_end]["symbol"]) == ["AAA", "BBB"]


def test_make_forecast_backend_accepts_linear_and_deep() -> None:
    linear_backend = _make_forecast_backend("linear", deps=legacy_services._forecast_runtime_dependencies())
    deep_backend = _make_forecast_backend("deep", deps=legacy_services._forecast_runtime_dependencies())

    assert linear_backend.name == "linear"
    assert deep_backend.name == "deep"


def test_tensorize_temporal_frame_builds_group_lags() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "f1": [1.0, 2.0, 10.0, 20.0],
            "f2": [3.0, 4.0, 30.0, 40.0],
        }
    )

    out, cols = _tensorize_temporal_frame(
        frame,
        feature_cols=["f1", "f2"],
        group_col="symbol",
        lag_depth=2,
    )

    assert cols == ["f1__lag0", "f2__lag0", "f1__lag1", "f2__lag1"]
    latest_aaa = out[(out["symbol"] == "AAA") & (out["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    latest_bbb = out[(out["symbol"] == "BBB") & (out["date"] == pd.Timestamp("2024-01-02"))].iloc[0]
    assert latest_aaa["f1__lag0"] == 2.0
    assert latest_aaa["f1__lag1"] == 1.0
    assert latest_bbb["f2__lag0"] == 40.0
    assert latest_bbb["f2__lag1"] == 30.0
