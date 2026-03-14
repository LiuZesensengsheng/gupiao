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
