from __future__ import annotations

from pathlib import Path

from src.application import v2_services as legacy
from src.application.v2_artifact_runtime import load_published_v2_policy_model as _load_published_v2_policy_model
from src.contracts.artifacts import DatasetManifest, ForecastBundle, LearnedPolicyArtifact, ResearchManifest
from src.contracts.runtime import ResearchRunOptions


def load_published_v2_policy_model(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
):
    return _load_published_v2_policy_model(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        load_policy_model_from_path_fn=legacy._load_policy_model_from_path,
    )


def publish_v2_research_artifacts(
    *,
    options: ResearchRunOptions | None = None,
    baseline,
    calibration,
    learning,
    settings: dict[str, object] | None = None,
    **kwargs: object,
):
    resolved = options or ResearchRunOptions.from_kwargs(**kwargs)
    paths = legacy._publish_v2_research_artifacts_impl(
        settings=settings,
        baseline=baseline,
        calibration=calibration,
        learning=learning,
        **resolved.publish_kwargs(),
    )
    _validate_published_paths(paths)
    return paths


def _validate_published_paths(paths: dict[str, str]) -> None:
    DatasetManifest.from_path(paths["dataset_manifest"])
    ResearchManifest.from_path(paths["research_manifest"])
    LearnedPolicyArtifact.from_path(paths["learned_policy_model"])

    forecast_bundle_path = str(paths.get("frozen_forecast_bundle", "")).strip()
    if forecast_bundle_path:
        bundle_path = Path(forecast_bundle_path)
        if bundle_path.exists() and bundle_path.read_text(encoding="utf-8").strip() not in {"", "{}"}:
            ForecastBundle.from_path(bundle_path)
