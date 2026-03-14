from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.application.v2_contracts import LearnedPolicyModel
from src.contracts.artifacts import LearnedPolicyArtifact


def load_policy_model_from_path(
    model_path: Path,
    *,
    load_json_dict: Callable[[object], dict[str, object]],
) -> LearnedPolicyModel | None:
    if not model_path.exists():
        return None
    payload = load_json_dict(model_path)
    if not isinstance(payload, dict):
        return None
    return LearnedPolicyArtifact.from_payload(payload).model


def load_published_v2_policy_model(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
    load_policy_model_from_path_fn: Callable[[Path], LearnedPolicyModel | None],
) -> LearnedPolicyModel | None:
    model_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    return load_policy_model_from_path_fn(model_path)


def resolve_daily_policy_model(
    *,
    strategy_id: str,
    artifact_root: str,
    manifest: dict[str, object],
    manifest_path: Path | None,
    path_from_manifest_entry: Callable[..., Path | None],
    load_policy_model_from_path_fn: Callable[[Path], LearnedPolicyModel | None],
    load_published_v2_policy_model_fn: Callable[..., LearnedPolicyModel | None],
) -> LearnedPolicyModel | None:
    learned_policy = None
    if manifest and manifest_path is not None:
        model_path = path_from_manifest_entry(
            manifest.get("learned_policy_model"),
            run_dir=manifest_path.parent,
        )
        if model_path is not None:
            learned_policy = load_policy_model_from_path_fn(model_path)
    if learned_policy is None:
        learned_policy = load_published_v2_policy_model_fn(
            strategy_id=strategy_id,
            artifact_root=artifact_root,
        )
    return learned_policy
