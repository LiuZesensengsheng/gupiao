from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.application.v2_contracts import LearnedPolicyModel

CURRENT_ARTIFACT_VERSION = 1


class ArtifactValidationError(ValueError):
    pass


def add_artifact_metadata(payload: Mapping[str, object], *, artifact_type: str) -> dict[str, object]:
    out = dict(payload)
    out.setdefault("artifact_type", str(artifact_type))
    out.setdefault("artifact_version", CURRENT_ARTIFACT_VERSION)
    return out


def _artifact_version(payload: Mapping[str, object], artifact_type: str) -> int:
    raw = payload.get("artifact_version", CURRENT_ARTIFACT_VERSION)
    try:
        version = int(raw)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{artifact_type}: invalid artifact_version={raw!r}") from exc
    if version != CURRENT_ARTIFACT_VERSION:
        raise ArtifactValidationError(
            f"{artifact_type}: unsupported artifact_version={version}, expected={CURRENT_ARTIFACT_VERSION}"
        )
    return version


def _require(payload: Mapping[str, object], artifact_type: str, *field_names: str) -> None:
    missing = [name for name in field_names if not str(payload.get(name, "")).strip()]
    if missing:
        raise ArtifactValidationError(f"{artifact_type}: missing required fields: {', '.join(missing)}")


def _load_payload(path_like: str | Path) -> dict[str, object]:
    path = Path(path_like)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"{path}: expected JSON object payload")
    return payload


@dataclass(frozen=True)
class DatasetManifest:
    artifact_version: int
    strategy_id: str
    universe_file: str
    universe_limit: int
    symbol_count: int
    symbols: tuple[str, ...]
    raw: dict[str, object]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "DatasetManifest":
        artifact_type = str(payload.get("artifact_type", "dataset_manifest"))
        version = _artifact_version(payload, artifact_type)
        _require(payload, artifact_type, "universe_file")
        symbols = tuple(str(item) for item in payload.get("symbols", []))
        return cls(
            artifact_version=version,
            strategy_id=str(payload.get("strategy_id", "")),
            universe_file=str(payload.get("universe_file", "")),
            universe_limit=int(payload.get("universe_limit", 0)),
            symbol_count=int(payload.get("symbol_count", len(symbols))),
            symbols=symbols,
            raw=dict(payload),
        )

    @classmethod
    def from_path(cls, path_like: str | Path) -> "DatasetManifest":
        return cls.from_payload(_load_payload(path_like))


@dataclass(frozen=True)
class ForecastBundle:
    artifact_version: int
    backend: str
    market_feature_cols: tuple[str, ...]
    panel_feature_cols: tuple[str, ...]
    raw: dict[str, object]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ForecastBundle":
        artifact_type = str(payload.get("artifact_type", "forecast_bundle"))
        version = _artifact_version(payload, artifact_type)
        _require(payload, artifact_type, "backend")
        return cls(
            artifact_version=version,
            backend=str(payload.get("backend", "")),
            market_feature_cols=tuple(str(item) for item in payload.get("market_feature_cols", [])),
            panel_feature_cols=tuple(str(item) for item in payload.get("panel_feature_cols", [])),
            raw=dict(payload),
        )

    @classmethod
    def from_path(cls, path_like: str | Path) -> "ForecastBundle":
        return cls.from_payload(_load_payload(path_like))


@dataclass(frozen=True)
class LearnedPolicyArtifact:
    artifact_version: int
    model: LearnedPolicyModel
    raw: dict[str, object]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "LearnedPolicyArtifact":
        artifact_type = str(payload.get("artifact_type", "learned_policy_model"))
        version = _artifact_version(payload, artifact_type)
        _require(
            payload,
            artifact_type,
            "feature_names",
            "exposure_intercept",
            "position_intercept",
            "turnover_intercept",
        )
        model = LearnedPolicyModel(
            feature_names=[str(item) for item in payload.get("feature_names", [])],
            exposure_intercept=float(payload.get("exposure_intercept", 0.60)),
            exposure_coef=[float(item) for item in payload.get("exposure_coef", [])],
            position_intercept=float(payload.get("position_intercept", 3.0)),
            position_coef=[float(item) for item in payload.get("position_coef", [])],
            turnover_intercept=float(payload.get("turnover_intercept", 0.22)),
            turnover_coef=[float(item) for item in payload.get("turnover_coef", [])],
            train_rows=int(payload.get("train_rows", 0)),
            train_r2_exposure=float(payload.get("train_r2_exposure", 0.0)),
            train_r2_positions=float(payload.get("train_r2_positions", 0.0)),
            train_r2_turnover=float(payload.get("train_r2_turnover", 0.0)),
        )
        return cls(
            artifact_version=version,
            model=model,
            raw=dict(payload),
        )

    @classmethod
    def from_path(cls, path_like: str | Path) -> "LearnedPolicyArtifact":
        return cls.from_payload(_load_payload(path_like))


@dataclass(frozen=True)
class ResearchManifest:
    artifact_version: int
    run_id: str
    strategy_id: str
    dataset_manifest: str
    backtest_summary: str
    raw: dict[str, object]

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ResearchManifest":
        artifact_type = str(payload.get("artifact_type", "research_manifest"))
        version = _artifact_version(payload, artifact_type)
        _require(payload, artifact_type, "run_id", "strategy_id", "dataset_manifest")
        return cls(
            artifact_version=version,
            run_id=str(payload.get("run_id", "")),
            strategy_id=str(payload.get("strategy_id", "")),
            dataset_manifest=str(payload.get("dataset_manifest", "")),
            backtest_summary=str(payload.get("backtest_summary", "")),
            raw=dict(payload),
        )

    @classmethod
    def from_path(cls, path_like: str | Path) -> "ResearchManifest":
        return cls.from_payload(_load_payload(path_like))
