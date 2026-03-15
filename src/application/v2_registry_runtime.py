from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.application.v2_artifact_runtime import load_policy_model_from_path as _load_policy_model_from_path_external
from src.application.v2_research_publish_runtime import ResearchPublishDependencies


def load_json_dict(path_like: object) -> dict[str, object]:
    path = Path(str(path_like))
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def load_policy_model_from_path(model_path: Path):
    return _load_policy_model_from_path_external(
        model_path,
        load_json_dict=load_json_dict,
    )


def compose_run_snapshot_hash(
    *,
    run_id: str,
    strategy_id: str,
    config_hash: str,
    policy_hash: str,
    universe_hash: str,
    model_hashes: dict[str, str],
) -> str:
    payload = {
        "run_id": str(run_id),
        "strategy_id": str(strategy_id),
        "config_hash": str(config_hash),
        "policy_hash": str(policy_hash),
        "universe_hash": str(universe_hash),
        "model_hashes": {str(k): str(v) for k, v in sorted(model_hashes.items())},
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_research_publish_dependencies() -> ResearchPublishDependencies:
    from src.application import v2_services as legacy

    return ResearchPublishDependencies(
        load_v2_runtime_settings_fn=legacy._load_v2_runtime_settings,
        resolve_v2_universe_settings_fn=legacy._resolve_v2_universe_settings,
        stable_json_hash_fn=legacy._stable_json_hash,
        sha256_file_fn=legacy._sha256_file,
        sha256_text_fn=legacy._sha256_text,
        compose_run_snapshot_hash_fn=legacy._compose_run_snapshot_hash,
        load_or_build_v2_backtest_trajectory_fn=legacy._load_or_build_v2_backtest_trajectory,
        split_research_trajectory_fn=legacy._split_research_trajectory,
        build_frozen_daily_state_payload_fn=legacy._build_frozen_daily_state_payload,
        build_frozen_linear_forecast_bundle_fn=legacy._build_frozen_linear_forecast_bundle,
        resolve_info_file_from_settings_fn=legacy._resolve_info_file_from_settings,
        load_v2_info_items_for_date_fn=legacy._load_v2_info_items_for_date,
        build_info_shadow_report_fn=legacy._build_info_shadow_report,
        build_info_manifest_payload_fn=legacy._build_info_manifest_payload,
        build_external_signal_package_for_date_fn=legacy._build_external_signal_package_for_date,
        parse_boolish_fn=legacy._parse_boolish,
        decode_composite_state_fn=legacy._decode_composite_state,
        enrich_state_with_info_fn=legacy._enrich_state_with_info,
        attach_external_signals_to_composite_state_fn=legacy._attach_external_signals_to_composite_state,
        serialize_composite_state_fn=legacy._serialize_composite_state,
        load_json_dict_fn=legacy._load_json_dict,
        tier_latest_manifest_path_fn=legacy._tier_latest_manifest_path,
        tier_latest_policy_path_fn=legacy._tier_latest_policy_path,
        load_backtest_payload_from_manifest_fn=legacy._load_backtest_payload_from_manifest,
        load_backtest_payload_for_run_fn=legacy._load_backtest_payload_for_run,
        summary_from_payload_fn=legacy._summary_from_payload,
        pass_release_gate_fn=legacy._pass_release_gate,
        pass_default_switch_gate_fn=legacy._pass_default_switch_gate,
        emit_progress_fn=legacy._emit_progress,
        remember_research_run_fn=legacy.remember_research_run,
    )
