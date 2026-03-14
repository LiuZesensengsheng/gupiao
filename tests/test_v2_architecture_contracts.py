from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.application.v2_contracts import (
    LearnedPolicyModel,
    PolicySpec,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)
from src.application.v2_daily_snapshot_runtime import (
    build_strategy_snapshot as build_strategy_snapshot_runtime,
    resolve_manifest_path,
)
from src.application.v2_artifact_runtime import (
    load_policy_model_from_path as load_policy_model_from_path_runtime,
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
        ]
    )
    daily_options = DailyRunOptions.from_namespace(daily_args)
    assert daily_options.strategy_id == "alpha_v2"
    assert daily_options.snapshot_path
    assert daily_options.allow_retrain is True

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
    assert matrix_options.universe_tiers == ("favorites_16", "generated_80")


def test_daily_snapshot_runtime_keeps_facade_contract() -> None:
    runtime_snapshot = build_strategy_snapshot_runtime(strategy_id="alpha_v2", universe_id="demo")

    from src.application.v2_services import build_strategy_snapshot

    facade_snapshot = build_strategy_snapshot(strategy_id="alpha_v2", universe_id="demo")

    assert facade_snapshot == runtime_snapshot
    assert (
        resolve_manifest_path(
            strategy_id="alpha_v2",
            artifact_root="artifacts/v2",
            run_id="20260314_010203",
            snapshot_path=None,
        )
        == Path("artifacts/v2") / "alpha_v2" / "20260314_010203" / "research_manifest.json"
    )


def test_policy_artifact_runtime_keeps_facade_contract(tmp_path: Path) -> None:
    payload_path = tmp_path / "latest_policy_model.json"
    payload_path.write_text(
        json.dumps(
            {
                "artifact_type": "learned_policy_model",
                "artifact_version": CURRENT_ARTIFACT_VERSION,
                "feature_names": ["x1"],
                "exposure_intercept": 0.5,
                "exposure_coef": [0.1],
                "position_intercept": 2.0,
                "position_coef": [0.1],
                "turnover_intercept": 0.2,
                "turnover_coef": [0.05],
                "train_rows": 64,
                "train_r2_exposure": 0.2,
                "train_r2_positions": 0.18,
                "train_r2_turnover": 0.12,
            }
        ),
        encoding="utf-8",
    )

    runtime_model = load_policy_model_from_path_runtime(
        payload_path,
        load_json_dict=lambda path_like: json.loads(Path(path_like).read_text(encoding="utf-8")),
    )

    from src.application.v2_services import _load_policy_model_from_path
    from src.artifact_registry.v2_registry import load_published_v2_policy_model

    facade_model = _load_policy_model_from_path(payload_path)

    strategy_dir = tmp_path / "alpha_v2"
    strategy_dir.mkdir()
    (strategy_dir / "latest_policy_model.json").write_text(payload_path.read_text(encoding="utf-8"), encoding="utf-8")
    registry_model = load_published_v2_policy_model(strategy_id="alpha_v2", artifact_root=str(tmp_path))

    assert runtime_model == facade_model
    assert registry_model == facade_model


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


def test_daily_report_view_model_uses_summary_shape() -> None:
    baseline = _make_backtest()
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
        learned=_make_backtest(0.24, 0.22),
    )
    calibration = V2CalibrationResult(
        best_policy=PolicySpec(),
        best_score=0.12,
        baseline=baseline,
        calibrated=_make_backtest(0.22, 0.20),
        trials=[],
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = publish_v2_research_artifacts(
        strategy_id="swing_v2",
        artifact_root=tmp_dir,
        publish_forecast_models=False,
        settings={
            "config_path": "config/api.json",
            "source": "local",
            "watchlist": "config/watchlist.json",
            "universe_file": "config/universe_smoke_5.json",
            "universe_limit": 5,
            "symbols": ["000001.SZ"],
            "symbol_count": 1,
            "start": "2024-01-01",
            "end": "2024-12-31",
        },
        baseline=baseline,
        calibration=calibration,
        learning=learning_result,
        )
        from src.application.v2_contracts import (
            CompositeState,
            CrossSectionForecastState,
            DailyRunResult,
            MarketForecastState,
            PolicyDecision,
        )
        from src.application.v2_services import build_strategy_snapshot

        result = DailyRunResult(
            snapshot=build_strategy_snapshot(strategy_id="swing_v2", universe_id="demo_universe"),
            composite_state=CompositeState(
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
            ),
            policy_decision=PolicyDecision(
                target_exposure=0.8,
                target_position_count=3,
                rebalance_now=True,
                rebalance_intensity=0.5,
                intraday_t_allowed=False,
                turnover_cap=0.2,
            ),
            trade_actions=[],
        )

        daily_vm = build_daily_report_view_model(result)
        research_vm = build_research_report_view_model(
            strategy_id="swing_v2",
            baseline=baseline,
            calibration=calibration,
            learning=learning_result,
            artifacts=paths,
        )

        assert daily_vm.strategy_id == summarize_daily_run(result)["strategy_id"]
        assert daily_vm.strategy_mode == "trend_follow"
        assert daily_vm.market_forecasts == []
        assert "candidate_selection" in daily_vm.dynamic_universe
        assert research_vm.strategy_id == "swing_v2"
        assert research_vm.release_gate_passed is (
            str(paths.get("release_gate_passed", "false")).strip().lower() == "true"
        )
        assert research_vm.learning_model["train_rows"] == 88
