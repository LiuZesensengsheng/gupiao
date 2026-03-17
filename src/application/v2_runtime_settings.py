from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from src.application.v2_universe_generator import generate_dynamic_universe
from src.infrastructure.discovery import build_predefined_universe, normalize_universe_tier
from src.infrastructure.market_data import set_tushare_token


def coalesce(primary: object, secondary: object, default: object) -> object:
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    return default


def configure_v2_tushare_token(
    *,
    explicit_token: str | None = None,
    daily: dict[str, object] | None = None,
    common: dict[str, object] | None = None,
) -> None:
    candidates: list[object] = [explicit_token]
    if isinstance(daily, dict):
        candidates.append(daily.get("tushare_token"))
    if isinstance(common, dict):
        candidates.append(common.get("tushare_token"))
    for candidate in candidates:
        if candidate is None:
            continue
        token = str(candidate).strip()
        if token:
            set_tushare_token(token)
            os.environ["TUSHARE_TOKEN"] = token
            return


def parse_boolish(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def parse_csv_tokens(value: object, default: Iterable[str]) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or [str(item).strip() for item in default if str(item).strip()]
    text = str(value).strip()
    if not text:
        return [str(item).strip() for item in default if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def stable_json_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path_like: object) -> str:
    path = Path(str(path_like))
    if not path.exists():
        return ""
    h = hashlib.sha256()
    if path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = str(file.relative_to(path)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            with file.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
        return h.hexdigest()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_json_dict(path_like: object) -> dict[str, object]:
    path = Path(str(path_like))
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def load_v2_runtime_settings(
    *,
    config_path: str,
    source: str | None = None,
    tushare_token: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    info_file: str | None = None,
    info_lookback_days: int | None = None,
    info_half_life_days: float | None = None,
    use_info_fusion: bool | None = None,
    use_learned_info_fusion: bool | None = None,
    info_shadow_only: bool | None = None,
    info_types: str | None = None,
    info_source_mode: str | None = None,
    info_subsets: str | None = None,
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    training_window_days: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            payload = raw

    common = payload.get("common", {}) if isinstance(payload.get("common"), dict) else {}
    daily = payload.get("daily", {}) if isinstance(payload.get("daily"), dict) else {}
    configure_v2_tushare_token(explicit_token=tushare_token, daily=daily, common=common)

    def pick(key: str, default: object) -> object:
        return coalesce(daily.get(key), common.get(key), default)

    resolved_universe_limit = int(
        universe_limit
        if universe_limit is not None
        else int(pick("universe_limit", 5))
    )
    default_dynamic_universe = resolved_universe_limit >= 150
    requested_dynamic_universe = (
        bool(dynamic_universe)
        if dynamic_universe is not None
        else parse_boolish(pick("dynamic_universe_enabled", default_dynamic_universe), default_dynamic_universe)
    )
    default_universe_file = (
        str(pick("generated_universe_base_file", "config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json"))
        if requested_dynamic_universe
        else str(pick("universe_file", "config/universe_smoke_5.json"))
    )
    resolved_universe_file = (
        str(universe_file).strip()
        if universe_file is not None and str(universe_file).strip()
        else default_universe_file
    )
    resolved_universe_tier = (
        str(universe_tier).strip()
        if universe_tier is not None and str(universe_tier).strip()
        else ("" if resolved_universe_file else str(pick("universe_tier", "")))
    )
    resolved_generator_target_size = int(
        generator_target_size
        if generator_target_size is not None
        else int(pick("dynamic_universe_target_size", resolved_universe_limit or 300))
    )
    resolved_generator_coarse_size = int(
        generator_coarse_size
        if generator_coarse_size is not None
        else int(pick("dynamic_universe_coarse_size", max(1000, resolved_generator_target_size * 3)))
    )

    return {
        "config_path": str(config_path),
        "watchlist": str(pick("watchlist", "config/watchlist.json")),
        "source": str(source).strip() if source is not None and str(source).strip() else str(pick("source", "auto")),
        "data_dir": str(pick("data_dir", "data")),
        "start": str(pick("start", "2018-01-01")),
        "end": str(pick("end", "2099-12-31")),
        "min_train_days": int(pick("min_train_days", 240)),
        "step_days": int(pick("step_days", 20)),
        "training_window_days": (
            int(training_window_days)
            if training_window_days is not None and int(training_window_days) > 0
            else (
                None
                if training_window_days is not None
                else (
                    int(pick("training_window_days", pick("backtest_training_window_days", 0)))
                    if int(pick("training_window_days", pick("backtest_training_window_days", 0))) > 0
                    else None
                )
            )
        ),
        "l2": float(pick("l2", 0.8)),
        "max_positions": int(pick("max_positions", 5)),
        "use_margin_features": bool(pick("use_margin_features", True)),
        "margin_market_file": str(pick("margin_market_file", "input/margin_market.csv")),
        "margin_stock_file": str(pick("margin_stock_file", "input/margin_stock.csv")),
        "use_us_index_context": (
            bool(use_us_index_context)
            if use_us_index_context is not None
            else parse_boolish(pick("use_us_index_context", False), False)
        ),
        "us_index_source": (
            str(us_index_source).strip()
            if us_index_source is not None and str(us_index_source).strip()
            else str(pick("us_index_source", "akshare")).strip()
        ),
        "use_us_sector_etf_context": parse_boolish(pick("use_us_sector_etf_context", False), False),
        "use_cn_etf_context": parse_boolish(pick("use_cn_etf_context", False), False),
        "cn_etf_source": str(pick("cn_etf_source", "akshare")).strip(),
        "universe_tier": resolved_universe_tier,
        "active_default_universe_tier": str(pick("active_default_universe_tier", "favorites_16")),
        "candidate_default_universe_tier": str(pick("candidate_default_universe_tier", "generated_80")),
        "favorites_universe_file": str(pick("favorites_universe_file", "config/universe_favorites.json")),
        "generated_universe_base_file": str(
            pick("generated_universe_base_file", "config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json")
        ),
        "baseline_reference_run_id": str(pick("baseline_reference_run_id", "20260308_211808")),
        "news_file": str(pick("news_file", "input/news_parts")),
        "info_file": (
            str(info_file).strip()
            if info_file is not None and str(info_file).strip()
            else (
                str(event_file).strip()
                if event_file is not None and str(event_file).strip()
                else str(pick("info_file", pick("news_file", "input/info_parts")))
            )
        ),
        "event_file": (
            str(event_file).strip()
            if event_file is not None and str(event_file).strip()
            else str(pick("event_file", pick("info_file", "input/info_parts")))
        ),
        "info_lookback_days": int(
            info_lookback_days
            if info_lookback_days is not None
            else int(pick("info_lookback_days", pick("news_lookback_days", 45)))
        ),
        "event_lookback_days": int(
            pick(
                "event_lookback_days",
                info_lookback_days if info_lookback_days is not None else pick("info_lookback_days", 45),
            )
        ),
        "learned_info_lookback_days": int(pick("learned_info_lookback_days", pick("learned_news_lookback_days", 720))),
        "info_half_life_days": float(
            info_half_life_days
            if info_half_life_days is not None
            else float(pick("info_half_life_days", pick("news_half_life_days", 10.0)))
        ),
        "capital_flow_file": (
            str(capital_flow_file).strip()
            if capital_flow_file is not None and str(capital_flow_file).strip()
            else str(pick("capital_flow_file", ""))
        ),
        "macro_file": (
            str(macro_file).strip()
            if macro_file is not None and str(macro_file).strip()
            else str(pick("macro_file", ""))
        ),
        "capital_flow_lookback_days": int(pick("capital_flow_lookback_days", 20)),
        "macro_lookback_days": int(pick("macro_lookback_days", 60)),
        "dynamic_universe_enabled": (
            bool(dynamic_universe)
            if dynamic_universe is not None
            else requested_dynamic_universe
        ),
        "generator_target_size": resolved_generator_target_size,
        "generator_coarse_size": resolved_generator_coarse_size,
        "generator_theme_aware": (
            bool(generator_theme_aware)
            if generator_theme_aware is not None
            else parse_boolish(pick("generator_theme_aware", True), True)
        ),
        "generator_use_concepts": (
            bool(generator_use_concepts)
            if generator_use_concepts is not None
            else parse_boolish(pick("generator_use_concepts", True), True)
        ),
        "main_board_only_universe": parse_boolish(
            pick("main_board_only_universe", pick("main_board_only_recommendations", False)),
            False,
        ),
        "main_board_only_recommendations": parse_boolish(
            pick("main_board_only_recommendations", False),
            False,
        ),
        "dynamic_universe_min_history_days": int(pick("dynamic_universe_min_history_days", 480)),
        "dynamic_universe_min_recent_amount": float(pick("dynamic_universe_min_recent_amount", 2.0e7)),
        "dynamic_universe_theme_cap_ratio": float(pick("dynamic_universe_theme_cap_ratio", 0.16)),
        "dynamic_universe_theme_floor_count": int(pick("dynamic_universe_theme_floor_count", 2)),
        "dynamic_universe_turnover_quality_weight": float(pick("dynamic_universe_turnover_quality_weight", 0.25)),
        "dynamic_universe_theme_weight": float(pick("dynamic_universe_theme_weight", 0.18)),
        "external_signals": (
            bool(external_signals)
            if external_signals is not None
            else parse_boolish(pick("external_signals", True), True)
        ),
        "enable_insight_memory": parse_boolish(pick("enable_insight_memory", True), True),
        "insight_notes_dir": str(pick("insight_notes_dir", "input/insight_notes")),
        "execution_overlay_enabled": parse_boolish(pick("execution_overlay_enabled", True), True),
        "external_signal_version": str(pick("external_signal_version", "v1")),
        "event_risk_cutoff": float(pick("event_risk_cutoff", 0.55)),
        "catalyst_boost_cap": float(pick("catalyst_boost_cap", 0.12)),
        "flow_exposure_cap": float(pick("flow_exposure_cap", 0.08)),
        "market_info_strength": float(pick("market_info_strength", pick("market_news_strength", 0.9))),
        "stock_info_strength": float(pick("stock_info_strength", pick("stock_news_strength", 1.1))),
        "use_info_fusion": (
            bool(use_info_fusion)
            if use_info_fusion is not None
            else parse_boolish(pick("use_info_fusion", False), False)
        ),
        "use_learned_info_fusion": (
            bool(use_learned_info_fusion)
            if use_learned_info_fusion is not None
            else parse_boolish(pick("use_learned_info_fusion", pick("use_learned_news_fusion", True)), True)
        ),
        "learned_info_min_samples": int(pick("learned_info_min_samples", pick("learned_news_min_samples", 80))),
        "learned_info_l2": float(pick("learned_info_l2", pick("learned_news_l2", 0.8))),
        "learned_info_holdout_ratio": float(pick("learned_info_holdout_ratio", pick("learned_holdout_ratio", 0.2))),
        "info_source_mode": (
            str(info_source_mode).strip()
            if info_source_mode is not None and str(info_source_mode).strip()
            else str(pick("info_source_mode", "layered")).strip()
        ),
        "info_shadow_only": (
            bool(info_shadow_only)
            if info_shadow_only is not None
            else parse_boolish(pick("info_shadow_only", True), True)
        ),
        "info_types": parse_csv_tokens(
            info_types if info_types is not None else pick("info_types", "news,announcement,research"),
            default=("news", "announcement", "research"),
        ),
        "info_subsets": parse_csv_tokens(
            info_subsets if info_subsets is not None else pick("info_subsets", "market_news,announcements,research"),
            default=("market_news", "announcements", "research"),
        ),
        "announcement_event_tags": parse_csv_tokens(
            pick(
                "announcement_event_tags",
                "earnings_positive,earnings_negative,guidance_positive,guidance_negative,contract_win,contract_loss,regulatory_positive,regulatory_negative,share_reduction,share_increase,trading_halt,delisting_risk",
            ),
            default=(
                "earnings_positive",
                "earnings_negative",
                "guidance_positive",
                "guidance_negative",
                "contract_win",
                "contract_loss",
                "regulatory_positive",
                "regulatory_negative",
                "share_reduction",
                "share_increase",
                "trading_halt",
                "delisting_risk",
            ),
        ),
        "universe_file": resolved_universe_file,
        "universe_limit": resolved_universe_limit,
    }


def extract_universe_rows(payload: object) -> list[dict[str, str]]:
    if isinstance(payload, list):
        return [
            {
                "symbol": str(item),
                "name": str(item),
                "sector": "其他",
            }
            for item in payload
        ]
    if isinstance(payload, dict):
        raw_rows = payload.get("stocks", [])
        if isinstance(raw_rows, list):
            out: list[dict[str, str]] = []
            for item in raw_rows:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "")).strip()
                if not symbol:
                    continue
                out.append(
                    {
                        "symbol": symbol,
                        "name": str(item.get("name", symbol)),
                        "sector": str(item.get("sector", "其他")),
                    }
                )
            return out
    return []


def hydrate_universe_metadata(
    *,
    universe_file: str,
    universe_limit: int,
    universe_tier: str = "",
    universe_generation_rule: str = "",
) -> dict[str, object]:
    path = Path(str(universe_file))
    rows: list[dict[str, str]] = []
    payload = load_json_dict(path)
    if payload:
        rows = extract_universe_rows(payload)
    elif path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = extract_universe_rows(raw)
        except Exception:
            rows = []
    symbols = [str(item["symbol"]) for item in rows]
    universe_id = str(universe_tier).strip() or path.stem or "v2_universe"
    generation_rule = str(universe_generation_rule).strip()
    if not generation_rule and payload:
        generation_rule = str(payload.get("generation_rule", ""))
    if not generation_rule:
        generation_rule = "external_universe_file"
    return {
        "universe_tier": str(universe_tier).strip(),
        "universe_id": universe_id,
        "universe_size": int(len(symbols) if symbols else max(0, int(universe_limit))),
        "universe_generation_rule": generation_rule,
        "source_universe_manifest_path": str(path.resolve()) if path.exists() else str(path),
        "symbols": symbols,
        "symbol_count": int(len(symbols)),
        "universe_hash": sha256_file(path) or stable_json_hash(symbols),
    }


def resolve_v2_universe_settings(
    *,
    settings: dict[str, object],
    cache_root: str,
    generate_dynamic_universe_fn=generate_dynamic_universe,
    build_predefined_universe_fn=build_predefined_universe,
    normalize_universe_tier_fn=normalize_universe_tier,
) -> dict[str, object]:
    resolved = dict(settings)
    requested_tier = str(resolved.get("universe_tier", "")).strip()
    dynamic_universe_enabled = parse_boolish(resolved.get("dynamic_universe_enabled", False), False)
    generator_target_size = int(
        resolved.get("generator_target_size", resolved.get("universe_limit", 300)) or resolved.get("universe_limit", 300)
    )
    generator_source_file = str(resolved.get("universe_file", "")).strip()
    if requested_tier and dynamic_universe_enabled:
        normalized_tier = normalize_universe_tier_fn(requested_tier)
        if normalized_tier.startswith("generated_"):
            if not generator_source_file:
                generator_source_file = str(resolved.get("generated_universe_base_file", generator_source_file)).strip()
            tier_digits = "".join(ch for ch in normalized_tier if ch.isdigit())
            if tier_digits:
                generator_target_size = int(tier_digits)
    if dynamic_universe_enabled and (generator_source_file or requested_tier):
        dynamic_result = generate_dynamic_universe_fn(
            universe_file=generator_source_file,
            data_dir=str(resolved.get("data_dir", "")),
            cache_root=str(cache_root),
            target_size=max(1, int(generator_target_size)),
            coarse_size=max(generator_target_size, int(resolved.get("generator_coarse_size", 1000))),
            theme_aware=parse_boolish(resolved.get("generator_theme_aware", True), True),
            use_concepts=parse_boolish(resolved.get("generator_use_concepts", True), True),
            end_date=str(resolved.get("end", "")),
            min_history_days=int(resolved.get("dynamic_universe_min_history_days", 480)),
            min_recent_amount=float(resolved.get("dynamic_universe_min_recent_amount", 2.0e7)),
            theme_cap_ratio=float(resolved.get("dynamic_universe_theme_cap_ratio", 0.16)),
            theme_floor_count=int(resolved.get("dynamic_universe_theme_floor_count", 2)),
            turnover_quality_weight=float(resolved.get("dynamic_universe_turnover_quality_weight", 0.25)),
            theme_weight=float(resolved.get("dynamic_universe_theme_weight", 0.18)),
            main_board_only=parse_boolish(resolved.get("main_board_only_universe", False), False),
            refresh_cache=parse_boolish(resolved.get("refresh_cache", False), False),
        )
        manifest = dynamic_result.generator_manifest
        selected_symbols = [
            str(item.get("symbol", ""))
            for item in dynamic_result.selected_300
            if str(item.get("symbol", "")).strip()
        ]
        manifest_path_text = str(manifest.manifest_path)
        universe_manifest_path = (
            manifest_path_text.replace(".generator.json", ".json")
            if manifest_path_text.endswith(".generator.json")
            else str(generator_source_file)
        )
        resolved["universe_tier"] = str(requested_tier)
        resolved["universe_file"] = universe_manifest_path
        resolved["universe_limit"] = int(len(selected_symbols))
        resolved["universe_id"] = f"dynamic_{int(generator_target_size)}"
        resolved["universe_size"] = int(len(selected_symbols))
        resolved["universe_generation_rule"] = f"{manifest.generator_version}: coarse->{manifest.coarse_pool_size} select->{manifest.selected_pool_size}"
        resolved["source_universe_manifest_path"] = str(manifest.source_universe_path or generator_source_file)
        resolved["symbols"] = selected_symbols
        resolved["symbol_count"] = int(len(selected_symbols))
        resolved["universe_hash"] = str(manifest.generator_hash)
        resolved["generator_manifest_path"] = str(manifest.manifest_path)
        resolved["generator_version"] = str(manifest.generator_version)
        resolved["generator_hash"] = str(manifest.generator_hash)
        resolved["coarse_pool_size"] = int(manifest.coarse_pool_size)
        resolved["refined_pool_size"] = int(manifest.refined_pool_size)
        resolved["selected_pool_size"] = int(manifest.selected_pool_size)
        resolved["theme_allocations"] = [asdict(item) for item in manifest.theme_allocations]
        return resolved

    if requested_tier:
        normalized_tier = normalize_universe_tier_fn(requested_tier)
        catalog_dir = Path(str(cache_root)) / "universe_catalog"
        manifest_token = stable_json_hash(
            {
                "tier": normalized_tier,
                "data_dir": str(resolved.get("data_dir", "")),
                "favorites_universe_file": str(resolved.get("favorites_universe_file", "")),
                "generated_universe_base_file": str(resolved.get("generated_universe_base_file", "")),
            }
        )[:12]
        manifest_path = catalog_dir / f"{normalized_tier}_{manifest_token}.json"
        built = build_predefined_universe_fn(
            tier_id=normalized_tier,
            data_dir=str(resolved.get("data_dir", "")),
            favorites_file=str(resolved.get("favorites_universe_file", "")),
            generated_base_file=str(resolved.get("generated_universe_base_file", "")),
            output_path=manifest_path,
            exclude_symbols=(),
        )
        resolved["universe_tier"] = normalized_tier
        resolved["universe_file"] = str(manifest_path.resolve())
        resolved["universe_limit"] = int(built.universe_size or len(built.rows))
        resolved["universe_id"] = str(built.universe_id or normalized_tier)
        resolved["universe_size"] = int(built.universe_size or len(built.rows))
        resolved["universe_generation_rule"] = str(built.generation_rule)
        resolved["source_universe_manifest_path"] = str(built.manifest_path or manifest_path.resolve())
        resolved["symbols"] = [str(item.symbol) for item in built.rows]
        resolved["symbol_count"] = int(len(built.rows))
        resolved["universe_hash"] = sha256_file(manifest_path) or stable_json_hash(resolved["symbols"])
        return resolved

    hydrated = hydrate_universe_metadata(
        universe_file=str(resolved.get("universe_file", "")),
        universe_limit=int(resolved.get("universe_limit", 0)),
        universe_generation_rule=str(resolved.get("universe_generation_rule", "")),
    )
    resolved.update(hydrated)
    return resolved
