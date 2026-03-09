from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pytest

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    LearnedPolicyModel,
    MarketForecastState,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
from src.application.v2_services import (
    _BacktestTrajectory,
    _TrajectoryStep,
    _build_date_slice_index,
    _derive_learning_targets,
    calibrate_v2_policy,
    _load_or_build_v2_backtest_trajectory,
    _make_forecast_backend,
    _policy_objective_score,
    _policy_spec_from_model,
    _predict_quantile_profiles,
    _split_research_trajectory,
    _tensorize_temporal_frame,
    load_published_v2_policy_model,
    publish_v2_research_artifacts,
    run_v2_research_workflow,
)


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

    spec = _policy_spec_from_model(state=state, model=model)

    assert isinstance(spec, PolicySpec)
    assert 0.20 <= spec.risk_on_exposure <= 0.95
    assert 1 <= spec.risk_on_positions <= 6
    assert 0.10 <= spec.risk_on_turnover_cap <= 0.45


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
    strong_targets = _derive_learning_targets(
        state=strong_state,
        stock_frames=strong_frames,
        date=date,
        horizon_metrics={"20d": {"rank_ic": 0.16, "top_bottom_spread": 0.09, "top_k_hit_rate": 0.72}},
        universe_tier="generated_80",
    )
    weak_targets = _derive_learning_targets(
        state=weak_state,
        stock_frames=weak_frames,
        date=date,
        horizon_metrics={"20d": {"rank_ic": -0.03, "top_bottom_spread": -0.01, "top_k_hit_rate": 0.48}},
        universe_tier="generated_80",
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
    info_file = tmp_path / "info.csv"
    info_file.write_text(
        "\n".join(
            [
                "date,target_type,target,horizon,direction,info_type,title,event_tag",
                "2024-12-20,market,MARKET,mid,bullish,news,macro support,regulatory_positive",
                "2024-12-22,stock,000001.SZ,short,bearish,announcement,risk event,earnings_negative",
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
            "info_file": str(info_file),
            "use_info_fusion": True,
            "info_shadow_only": True,
        },
        baseline=baseline,
        calibration=calibration,
        learning=learning_result,
    )

    dataset_manifest = json.loads(Path(paths["dataset_manifest"]).read_text(encoding="utf-8"))
    assert dataset_manifest["universe_tier"] == "generated_80"
    assert dataset_manifest["universe_id"] == "generated_80"
    assert dataset_manifest["symbol_count"] == 3
    assert len(dataset_manifest["symbols"]) == 3
    assert dataset_manifest["source_universe_manifest_path"]
    assert dataset_manifest["info_file"] == str(info_file)
    assert dataset_manifest["info_hash"]
    assert dataset_manifest["info_item_count"] == 2

    manifest = json.loads(Path(paths["research_manifest"]).read_text(encoding="utf-8"))
    assert manifest["default_switch_gate"]["passed"] is False
    assert manifest["info_manifest"]
    assert manifest["info_shadow_report"]
    assert manifest["info_hash"]
    assert not (tmp_path / "swing_v2" / "latest_research_manifest.json").exists()
    assert (tmp_path / "swing_v2" / "latest_research_manifest.generated_80.json").exists()
    info_manifest = json.loads(Path(paths["info_manifest"]).read_text(encoding="utf-8"))
    assert info_manifest["info_item_count"] == 2
    assert info_manifest["info_type_counts"]["news"] == 1
    assert info_manifest["info_type_counts"]["announcement"] == 1


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


def test_calibrate_v2_policy_emits_progress_updates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", fake_backtest)
    monkeypatch.setattr("src.application.v2_services._emit_progress", lambda stage, message: progress.append((stage, message)))

    calibrate_v2_policy(
        strategy_id="swing_v2",
        baseline=baseline,
        trajectory=object(),
        cache_root=str(tmp_path),
    )

    assert any(stage == "calibration" and "开始参数搜索" in message for stage, message in progress)
    assert any(stage == "calibration" and "评估候选" in message for stage, message in progress)


def test_research_workflow_light_mode_skips_heavy_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", fake_load)
    monkeypatch.setattr("src.application.v2_services._split_research_trajectory", fake_split)
    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", fake_baseline)
    monkeypatch.setattr("src.application.v2_services.calibrate_v2_policy", fail_calibration)
    monkeypatch.setattr("src.application.v2_services.learn_v2_policy_model", fail_learning)

    got_baseline, calibration, learning = run_v2_research_workflow(
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


def test_research_workflow_emits_stage_progress(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    baseline = _make_backtest(0.18, 0.16)
    trajectory_sentinel = object()
    train_sentinel = object()
    validation_sentinel = object()
    holdout_sentinel = object()
    progress: list[tuple[str, str]] = []

    monkeypatch.setattr("src.application.v2_services._emit_progress", lambda stage, message: progress.append((stage, message)))
    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", lambda **_: trajectory_sentinel)
    monkeypatch.setattr(
        "src.application.v2_services._split_research_trajectory",
        lambda trajectory, *args, **kwargs: (train_sentinel, validation_sentinel, holdout_sentinel),
    )
    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", lambda **_: baseline)

    run_v2_research_workflow(
        strategy_id="swing_v2",
        skip_calibration=True,
        skip_learning=True,
        cache_root=str(tmp_path),
    )

    assert any(stage == "research" and "载入研究轨迹" in message for stage, message in progress)
    assert any(stage == "research" and "样本切分完成" in message for stage, message in progress)
    assert any(stage == "research" and "已跳过参数搜索" in message for stage, message in progress)
    assert any(stage == "research" and "已跳过学习型策略" in message for stage, message in progress)


def test_research_workflow_reuses_single_trajectory_for_all_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", fake_load)
    monkeypatch.setattr("src.application.v2_services._split_research_trajectory", fake_split)
    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", fake_baseline)
    monkeypatch.setattr("src.application.v2_services.calibrate_v2_policy", fake_calibration)
    monkeypatch.setattr("src.application.v2_services.learn_v2_policy_model", fake_learning)

    got_baseline, got_calibration, got_learning = run_v2_research_workflow(
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


def test_research_workflow_passes_deep_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("src.application.v2_services._load_or_build_v2_backtest_trajectory", fake_load)
    monkeypatch.setattr("src.application.v2_services._split_research_trajectory", fake_split)
    monkeypatch.setattr("src.application.v2_services.run_v2_backtest_live", fake_baseline)

    got_baseline, _, _ = run_v2_research_workflow(
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
    linear_backend = _make_forecast_backend("linear")
    deep_backend = _make_forecast_backend("deep")

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
