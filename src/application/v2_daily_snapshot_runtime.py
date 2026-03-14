from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from src.application.v2_contracts import StrategySnapshot
from src.contracts.artifacts import DatasetManifest, ResearchManifest


@dataclass(frozen=True)
class DailySnapshotContext:
    settings: dict[str, object]
    manifest: dict[str, object]
    manifest_path: Path | None
    snapshot: StrategySnapshot
    resolved_run_id: str


def resolve_manifest_path(
    *,
    strategy_id: str,
    artifact_root: str,
    run_id: str | None,
    snapshot_path: str | None,
) -> Path:
    if snapshot_path is not None and str(snapshot_path).strip():
        path = Path(str(snapshot_path))
        if path.is_dir():
            return path / "research_manifest.json"
        return path
    if run_id is not None and str(run_id).strip():
        return Path(str(artifact_root)) / str(strategy_id) / str(run_id).strip() / "research_manifest.json"
    return Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json"


def build_strategy_snapshot(
    *,
    strategy_id: str,
    universe_id: str = "top_liquid_200",
    universe_size: int = 0,
    universe_generation_rule: str = "",
    source_universe_manifest_path: str = "",
    info_manifest_path: str = "",
    info_hash: str = "",
    info_shadow_enabled: bool = False,
    external_signal_manifest_path: str = "",
    external_signal_version: str = "",
    external_signal_enabled: bool = False,
    capital_flow_snapshot: dict[str, object] | None = None,
    macro_context_snapshot: dict[str, object] | None = None,
    generator_manifest_path: str = "",
    generator_version: str = "",
    generator_hash: str = "",
    coarse_pool_size: int = 0,
    refined_pool_size: int = 0,
    selected_pool_size: int = 0,
    theme_allocations: list[dict[str, object]] | None = None,
    run_id: str = "",
    data_window: str = "",
    model_hashes: dict[str, str] | None = None,
    policy_hash: str = "",
    universe_hash: str = "",
    created_at: str = "",
    snapshot_hash: str = "",
    config_hash: str = "",
    manifest_path: str = "",
    use_us_index_context: bool = False,
    us_index_source: str = "",
) -> StrategySnapshot:
    return StrategySnapshot(
        strategy_id=str(strategy_id).strip() or "swing_v2",
        universe_id=str(universe_id).strip() or "top_liquid_200",
        feature_set_version="fset_v2_core",
        market_model_id="mkt_lr_v2",
        sector_model_id="sector_lr_v2",
        stock_model_id="stock_lr_v2",
        cross_section_model_id="cross_section_v2",
        policy_version="policy_v2_rules",
        execution_version="exec_t1_v2",
        universe_size=int(universe_size),
        universe_generation_rule=str(universe_generation_rule),
        source_universe_manifest_path=str(source_universe_manifest_path),
        info_manifest_path=str(info_manifest_path),
        info_hash=str(info_hash),
        info_shadow_enabled=bool(info_shadow_enabled),
        external_signal_manifest_path=str(external_signal_manifest_path),
        external_signal_version=str(external_signal_version),
        external_signal_enabled=bool(external_signal_enabled),
        capital_flow_snapshot=dict(capital_flow_snapshot or {}),
        macro_context_snapshot=dict(macro_context_snapshot or {}),
        generator_manifest_path=str(generator_manifest_path),
        generator_version=str(generator_version),
        generator_hash=str(generator_hash),
        coarse_pool_size=int(coarse_pool_size),
        refined_pool_size=int(refined_pool_size),
        selected_pool_size=int(selected_pool_size),
        theme_allocations=list(theme_allocations or []),
        run_id=str(run_id),
        data_window=str(data_window),
        model_hashes=dict(model_hashes or {}),
        policy_hash=str(policy_hash),
        universe_hash=str(universe_hash),
        created_at=str(created_at),
        snapshot_hash=str(snapshot_hash),
        config_hash=str(config_hash),
        manifest_path=str(manifest_path),
        use_us_index_context=bool(use_us_index_context),
        us_index_source=str(us_index_source),
    )


def load_research_manifest_for_daily(
    *,
    strategy_id: str,
    artifact_root: str,
    run_id: str | None,
    snapshot_path: str | None,
    resolve_manifest_path_fn: Callable[..., Path],
    load_json_dict: Callable[[object], dict[str, object]],
) -> tuple[dict[str, object], Path]:
    manifest_path = resolve_manifest_path_fn(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        run_id=run_id,
        snapshot_path=snapshot_path,
    )
    payload = load_json_dict(manifest_path)
    if not payload:
        raise FileNotFoundError(
            f"Missing research manifest for daily-run: {manifest_path}. "
            "Run `research-run` first or pass `--allow-retrain`."
        )
    manifest_run_id = str(payload.get("run_id", "")).strip()
    requested_run_id = "" if run_id is None else str(run_id).strip()
    if requested_run_id and manifest_run_id and manifest_run_id != requested_run_id:
        raise ValueError(
            f"run_id mismatch: requested={requested_run_id}, manifest={manifest_run_id} ({manifest_path})"
        )
    manifest_strategy = str(payload.get("strategy_id", "")).strip()
    if manifest_strategy and manifest_strategy != str(strategy_id):
        raise ValueError(
            f"strategy mismatch in manifest: requested={strategy_id}, manifest={manifest_strategy}"
        )
    return payload, manifest_path


def build_snapshot_from_manifest(
    *,
    strategy_id: str,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
    path_from_manifest_entry: Callable[..., Path | None],
    load_json_dict: Callable[[object], dict[str, object]],
    parse_boolish: Callable[[object, bool], bool],
    stable_json_hash: Callable[[object], str],
    compose_run_snapshot_hash: Callable[..., str],
) -> StrategySnapshot:
    dataset_path = path_from_manifest_entry(manifest.get("dataset_manifest"), run_dir=manifest_path.parent)
    dataset_manifest = load_json_dict(dataset_path) if dataset_path is not None else {}
    config_hash = str(manifest.get("config_hash", "")) or stable_json_hash(settings)
    model_hashes_raw = manifest.get("model_hashes", {})
    model_hashes = (
        {str(k): str(v) for k, v in model_hashes_raw.items()}
        if isinstance(model_hashes_raw, dict)
        else {}
    )
    policy_hash = str(manifest.get("policy_hash", ""))
    universe_hash = str(manifest.get("universe_hash", ""))
    run_id = str(manifest.get("run_id", ""))
    snapshot_hash = str(manifest.get("snapshot_hash", "")) or compose_run_snapshot_hash(
        run_id=run_id,
        strategy_id=strategy_id,
        config_hash=config_hash,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        model_hashes=model_hashes,
    )
    universe_file = str(dataset_manifest.get("universe_file", settings.get("universe_file", "")))
    universe_id = (
        str(dataset_manifest.get("universe_id", "")).strip()
        or Path(universe_file).stem
        or Path(str(settings.get("universe_file", ""))).stem
        or "v2_universe"
    )
    start = str(dataset_manifest.get("start", settings.get("start", "")))
    end = str(dataset_manifest.get("end", settings.get("end", "")))
    data_window = f"{start}~{end}" if start or end else ""
    return build_strategy_snapshot(
        strategy_id=strategy_id,
        universe_id=universe_id,
        universe_size=int(dataset_manifest.get("universe_size", dataset_manifest.get("symbol_count", 0)) or 0),
        universe_generation_rule=str(dataset_manifest.get("universe_generation_rule", "")),
        source_universe_manifest_path=str(
            dataset_manifest.get("source_universe_manifest_path", dataset_manifest.get("universe_file", ""))
        ),
        info_manifest_path=str(manifest.get("info_manifest", "")),
        info_hash=str(manifest.get("info_hash", dataset_manifest.get("info_hash", ""))),
        info_shadow_enabled=parse_boolish(
            manifest.get("info_shadow_enabled", dataset_manifest.get("info_shadow_enabled", False)),
            False,
        ),
        external_signal_manifest_path=str(
            manifest.get("external_signal_manifest", dataset_manifest.get("external_signal_manifest", ""))
        ),
        external_signal_version=str(
            manifest.get("external_signal_version", dataset_manifest.get("external_signal_version", "v1"))
        ),
        external_signal_enabled=parse_boolish(
            manifest.get("external_signal_enabled", dataset_manifest.get("external_signal_enabled", False)),
            False,
        ),
        capital_flow_snapshot=dict(
            manifest.get("capital_flow_snapshot", dataset_manifest.get("capital_flow_snapshot", {}))
        ),
        macro_context_snapshot=dict(
            manifest.get("macro_context_snapshot", dataset_manifest.get("macro_context_snapshot", {}))
        ),
        generator_manifest_path=str(
            manifest.get("generator_manifest", dataset_manifest.get("generator_manifest", ""))
        ),
        generator_version=str(manifest.get("generator_version", dataset_manifest.get("generator_version", ""))),
        generator_hash=str(manifest.get("generator_hash", dataset_manifest.get("generator_hash", ""))),
        coarse_pool_size=int(manifest.get("coarse_pool_size", dataset_manifest.get("coarse_pool_size", 0)) or 0),
        refined_pool_size=int(manifest.get("refined_pool_size", dataset_manifest.get("refined_pool_size", 0)) or 0),
        selected_pool_size=int(manifest.get("selected_pool_size", dataset_manifest.get("selected_pool_size", 0)) or 0),
        theme_allocations=[
            dict(item)
            for item in manifest.get("theme_allocations", dataset_manifest.get("theme_allocations", []))
            if isinstance(item, dict)
        ],
        run_id=run_id,
        data_window=data_window,
        model_hashes=model_hashes,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        created_at=str(manifest.get("created_at", "")),
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
        manifest_path=str(manifest_path.resolve()),
        use_us_index_context=parse_boolish(
            dataset_manifest.get("use_us_index_context", manifest.get("use_us_index_context", False)),
            False,
        ),
        us_index_source=str(dataset_manifest.get("us_index_source", manifest.get("us_index_source", ""))),
    )


def hydrate_daily_settings_from_dataset_manifest(
    *,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
    universe_tier: str | None,
    universe_file: str | None,
    path_from_manifest_entry: Callable[..., Path | None],
    load_json_dict: Callable[[object], dict[str, object]],
    parse_boolish: Callable[[object, bool], bool],
) -> dict[str, object]:
    dataset_path = path_from_manifest_entry(manifest.get("dataset_manifest"), run_dir=manifest_path.parent)
    dataset_manifest = load_json_dict(dataset_path) if dataset_path is not None else {}
    if not dataset_manifest:
        return settings
    DatasetManifest.from_payload(dataset_manifest)

    manifest_universe_tier = str(dataset_manifest.get("universe_tier", "")).strip()
    manifest_universe_file = str(dataset_manifest.get("universe_file", "")).strip()
    requested_universe_tier = str(settings.get("universe_tier", "")).strip()
    requested_universe_file = str(settings.get("universe_file", "")).strip()
    if universe_tier is not None and requested_universe_tier and manifest_universe_tier and requested_universe_tier != manifest_universe_tier:
        raise ValueError(
            f"universe tier mismatch: requested={requested_universe_tier}, manifest={manifest_universe_tier}"
        )
    if universe_file is not None and requested_universe_file and manifest_universe_file:
        requested_path = str(Path(requested_universe_file).resolve())
        manifest_path_resolved = str(Path(manifest_universe_file).resolve())
        if requested_path != manifest_path_resolved:
            raise ValueError(
                f"universe file mismatch: requested={requested_path}, manifest={manifest_path_resolved}"
            )

    hydrated = dict(settings)
    hydrated["universe_file"] = str(dataset_manifest.get("universe_file", hydrated.get("universe_file", "")))
    hydrated["universe_limit"] = int(dataset_manifest.get("universe_limit", hydrated.get("universe_limit", 0)))
    hydrated["universe_tier"] = str(dataset_manifest.get("universe_tier", hydrated.get("universe_tier", "")))
    hydrated["universe_id"] = str(dataset_manifest.get("universe_id", hydrated.get("universe_id", "")))
    hydrated["universe_size"] = int(
        dataset_manifest.get("universe_size", dataset_manifest.get("symbol_count", hydrated.get("universe_size", 0)))
    )
    hydrated["universe_generation_rule"] = str(
        dataset_manifest.get("universe_generation_rule", hydrated.get("universe_generation_rule", ""))
    )
    hydrated["source_universe_manifest_path"] = str(
        dataset_manifest.get("source_universe_manifest_path", hydrated.get("source_universe_manifest_path", ""))
    )
    hydrated["symbols"] = [
        str(item)
        for item in dataset_manifest.get("symbols", hydrated.get("symbols", []))
        if str(item).strip()
    ]
    hydrated["symbol_count"] = int(dataset_manifest.get("symbol_count", len(hydrated["symbols"])))
    hydrated["universe_hash"] = str(dataset_manifest.get("universe_hash", hydrated.get("universe_hash", "")))
    hydrated["dynamic_universe_enabled"] = parse_boolish(
        dataset_manifest.get("dynamic_universe_enabled", hydrated.get("dynamic_universe_enabled", False)),
        False,
    )
    hydrated["generator_manifest_path"] = str(
        dataset_manifest.get("generator_manifest", hydrated.get("generator_manifest_path", ""))
    )
    hydrated["generator_version"] = str(dataset_manifest.get("generator_version", hydrated.get("generator_version", "")))
    hydrated["generator_hash"] = str(dataset_manifest.get("generator_hash", hydrated.get("generator_hash", "")))
    hydrated["coarse_pool_size"] = int(dataset_manifest.get("coarse_pool_size", hydrated.get("coarse_pool_size", 0)))
    hydrated["refined_pool_size"] = int(dataset_manifest.get("refined_pool_size", hydrated.get("refined_pool_size", 0)))
    hydrated["selected_pool_size"] = int(dataset_manifest.get("selected_pool_size", hydrated.get("selected_pool_size", 0)))
    hydrated["theme_allocations"] = [
        dict(item)
        for item in dataset_manifest.get("theme_allocations", hydrated.get("theme_allocations", []))
        if isinstance(item, dict)
    ]
    hydrated["info_file"] = str(dataset_manifest.get("info_file", hydrated.get("info_file", "")))
    hydrated["event_file"] = str(dataset_manifest.get("event_file", hydrated.get("event_file", hydrated.get("info_file", ""))))
    hydrated["info_hash"] = str(dataset_manifest.get("info_hash", hydrated.get("info_hash", "")))
    hydrated["info_shadow_enabled"] = parse_boolish(
        dataset_manifest.get("info_shadow_enabled", hydrated.get("info_shadow_enabled", False)),
        False,
    )
    hydrated["capital_flow_file"] = str(dataset_manifest.get("capital_flow_file", hydrated.get("capital_flow_file", "")))
    hydrated["macro_file"] = str(dataset_manifest.get("macro_file", hydrated.get("macro_file", "")))
    hydrated["external_signals"] = parse_boolish(
        dataset_manifest.get("external_signal_enabled", hydrated.get("external_signals", True)),
        True,
    )
    hydrated["external_signal_version"] = str(
        dataset_manifest.get("external_signal_version", hydrated.get("external_signal_version", "v1"))
    )
    hydrated["event_lookback_days"] = int(
        dataset_manifest.get("event_lookback_days", hydrated.get("event_lookback_days", hydrated.get("info_lookback_days", 45)))
    )
    hydrated["capital_flow_lookback_days"] = int(
        dataset_manifest.get("capital_flow_lookback_days", hydrated.get("capital_flow_lookback_days", 20))
    )
    hydrated["macro_lookback_days"] = int(
        dataset_manifest.get("macro_lookback_days", hydrated.get("macro_lookback_days", 60))
    )
    hydrated["event_risk_cutoff"] = float(dataset_manifest.get("event_risk_cutoff", hydrated.get("event_risk_cutoff", 0.55)))
    hydrated["catalyst_boost_cap"] = float(dataset_manifest.get("catalyst_boost_cap", hydrated.get("catalyst_boost_cap", 0.12)))
    hydrated["flow_exposure_cap"] = float(dataset_manifest.get("flow_exposure_cap", hydrated.get("flow_exposure_cap", 0.08)))
    hydrated["info_source_mode"] = str(dataset_manifest.get("info_source_mode", hydrated.get("info_source_mode", "layered")))
    hydrated["use_us_index_context"] = parse_boolish(
        dataset_manifest.get("use_us_index_context", manifest.get("use_us_index_context", False)),
        False,
    )
    hydrated["us_index_source"] = str(dataset_manifest.get("us_index_source", manifest.get("us_index_source", "akshare")))
    hydrated["use_us_sector_etf_context"] = parse_boolish(
        dataset_manifest.get("use_us_sector_etf_context", manifest.get("use_us_sector_etf_context", False)),
        False,
    )
    hydrated["use_cn_etf_context"] = parse_boolish(
        dataset_manifest.get("use_cn_etf_context", manifest.get("use_cn_etf_context", False)),
        False,
    )
    hydrated["cn_etf_source"] = str(dataset_manifest.get("cn_etf_source", manifest.get("cn_etf_source", "akshare")))
    hydrated["info_subsets"] = [
        str(item)
        for item in dataset_manifest.get("info_subsets", hydrated.get("info_subsets", []))
        if str(item).strip()
    ]
    hydrated["announcement_event_tags"] = [
        str(item)
        for item in dataset_manifest.get("announcement_event_tags", hydrated.get("announcement_event_tags", []))
        if str(item).strip()
    ]
    return hydrated


def is_daily_universe_override_mismatch(exc: Exception) -> bool:
    text = str(exc)
    return "universe tier mismatch:" in text or "universe file mismatch:" in text


def build_daily_snapshot_context(
    *,
    strategy_id: str,
    config_path: str,
    source: str | None,
    universe_file: str | None,
    universe_limit: int | None,
    universe_tier: str | None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    info_file: str | None,
    info_lookback_days: int | None,
    info_half_life_days: float | None,
    use_info_fusion: bool | None,
    info_shadow_only: bool | None,
    info_types: str | None,
    info_source_mode: str | None,
    info_subsets: str | None,
    external_signals: bool | None,
    event_file: str | None,
    capital_flow_file: str | None,
    macro_file: str | None,
    use_us_index_context: bool | None,
    us_index_source: str | None,
    artifact_root: str,
    cache_root: str,
    refresh_cache: bool = False,
    run_id: str | None = None,
    snapshot_path: str | None = None,
    allow_retrain: bool,
    load_v2_runtime_settings: Callable[..., dict[str, object]],
    resolve_v2_universe_settings: Callable[..., dict[str, object]],
    load_research_manifest_for_daily_fn: Callable[..., tuple[dict[str, object], Path]],
    hydrate_daily_settings_from_dataset_manifest_fn: Callable[..., dict[str, object]],
    build_snapshot_from_manifest_fn: Callable[..., StrategySnapshot],
    parse_boolish: Callable[[object, bool], bool],
    stable_json_hash: Callable[[object], str],
    sha256_file: Callable[[object], str],
    emit_progress: Callable[[str, str], None],
) -> DailySnapshotContext:
    settings = load_v2_runtime_settings(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        info_file=info_file,
        info_lookback_days=info_lookback_days,
        info_half_life_days=info_half_life_days,
        use_info_fusion=use_info_fusion,
        info_shadow_only=info_shadow_only,
        info_types=info_types,
        info_source_mode=info_source_mode,
        info_subsets=info_subsets,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    settings["refresh_cache"] = bool(refresh_cache)
    settings = resolve_v2_universe_settings(settings=settings, cache_root=cache_root)

    manifest: dict[str, object] = {}
    manifest_path: Path | None = None
    try:
        manifest, manifest_path = load_research_manifest_for_daily_fn(
            strategy_id=strategy_id,
            artifact_root=artifact_root,
            run_id=run_id,
            snapshot_path=snapshot_path,
        )
        if manifest:
            ResearchManifest.from_payload(manifest)
    except Exception:
        if not allow_retrain:
            raise

    resolved_run_id = ""
    if manifest:
        resolved_run_id = str(manifest.get("run_id", "")).strip()
    elif run_id is not None:
        resolved_run_id = str(run_id).strip()

    if manifest and manifest_path is not None:
        try:
            settings = hydrate_daily_settings_from_dataset_manifest_fn(
                settings=settings,
                manifest=manifest,
                manifest_path=manifest_path,
                universe_tier=universe_tier,
                universe_file=universe_file,
            )
            snapshot = build_snapshot_from_manifest_fn(
                strategy_id=strategy_id,
                settings=settings,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except ValueError as exc:
            explicit_universe_override = universe_tier is not None or universe_file is not None
            if not (allow_retrain and explicit_universe_override and is_daily_universe_override_mismatch(exc)):
                raise
            emit_progress(
                "load-strategy-snapshot",
                "detected explicit universe override; bypassing published snapshot and rebuilding daily state.",
            )
            manifest = {}
            manifest_path = None
            resolved_run_id = ""
            data_window = f"{settings.get('start', '')}~{settings.get('end', '')}"
            snapshot = build_strategy_snapshot(
                strategy_id=strategy_id,
                universe_id=str(settings.get("universe_id", "")).strip()
                or Path(str(settings["universe_file"])).stem
                or "v2_universe",
                universe_size=int(settings.get("universe_size", settings.get("symbol_count", 0)) or 0),
                universe_generation_rule=str(settings.get("universe_generation_rule", "")),
                source_universe_manifest_path=str(
                    settings.get("source_universe_manifest_path", settings.get("universe_file", ""))
                ),
                info_manifest_path=str(settings.get("info_manifest_path", "")),
                info_hash=str(settings.get("info_hash", "")),
                info_shadow_enabled=parse_boolish(settings.get("info_shadow_enabled", False), False),
                external_signal_manifest_path=str(settings.get("external_signal_manifest", "")),
                external_signal_version=str(settings.get("external_signal_version", "v1")),
                external_signal_enabled=parse_boolish(settings.get("external_signals", False), False),
                capital_flow_snapshot=dict(settings.get("capital_flow_snapshot", {})),
                macro_context_snapshot=dict(settings.get("macro_context_snapshot", {})),
                generator_manifest_path=str(settings.get("generator_manifest_path", "")),
                generator_version=str(settings.get("generator_version", "")),
                generator_hash=str(settings.get("generator_hash", "")),
                coarse_pool_size=int(settings.get("coarse_pool_size", 0)),
                refined_pool_size=int(settings.get("refined_pool_size", 0)),
                selected_pool_size=int(settings.get("selected_pool_size", 0)),
                theme_allocations=[dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
                run_id="",
                data_window=data_window,
                model_hashes={},
                policy_hash="",
                universe_hash=str(settings.get("universe_hash", "")),
                created_at=str(pd.Timestamp.now().isoformat()),
                snapshot_hash="",
                config_hash=stable_json_hash(settings),
                manifest_path="",
                use_us_index_context=parse_boolish(settings.get("use_us_index_context", False), False),
                us_index_source=str(settings.get("us_index_source", "")),
            )
    else:
        data_window = f"{settings.get('start', '')}~{settings.get('end', '')}"
        snapshot = build_strategy_snapshot(
            strategy_id=strategy_id,
            universe_id=str(settings.get("universe_id", "")).strip() or Path(str(settings["universe_file"])).stem or "v2_universe",
            universe_size=int(settings.get("universe_size", settings.get("symbol_count", 0)) or 0),
            universe_generation_rule=str(settings.get("universe_generation_rule", "")),
            source_universe_manifest_path=str(
                settings.get("source_universe_manifest_path", settings.get("universe_file", ""))
            ),
            info_manifest_path=str(settings.get("info_manifest_path", "")),
            info_hash=str(settings.get("info_hash", "")),
            info_shadow_enabled=parse_boolish(settings.get("info_shadow_enabled", False), False),
            external_signal_manifest_path=str(settings.get("external_signal_manifest", "")),
            external_signal_version=str(settings.get("external_signal_version", "v1")),
            external_signal_enabled=parse_boolish(settings.get("external_signals", True), True),
            capital_flow_snapshot=dict(settings.get("capital_flow_snapshot", {})),
            macro_context_snapshot=dict(settings.get("macro_context_snapshot", {})),
            generator_manifest_path=str(settings.get("generator_manifest_path", "")),
            generator_version=str(settings.get("generator_version", "")),
            generator_hash=str(settings.get("generator_hash", "")),
            coarse_pool_size=int(settings.get("coarse_pool_size", 0)),
            refined_pool_size=int(settings.get("refined_pool_size", 0)),
            selected_pool_size=int(settings.get("selected_pool_size", 0)),
            theme_allocations=[dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
            run_id=resolved_run_id,
            data_window=data_window,
            config_hash=stable_json_hash(settings),
            universe_hash=str(settings.get("universe_hash", "")) or sha256_file(settings.get("universe_file", "")),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
        )

    return DailySnapshotContext(
        settings=settings,
        manifest=manifest,
        manifest_path=manifest_path,
        snapshot=snapshot,
        resolved_run_id=resolved_run_id,
    )
