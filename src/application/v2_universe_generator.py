from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    DynamicUniverseResult,
    ThemeAllocationRecord,
    UniverseGeneratorManifest,
)
from src.domain.entities import Security
from src.infrastructure.discovery import (
    _dedupe_rows,
    _from_data_dir,
    _load_universe_file,
    _resolve_local_symbol_path,
    _safe_read_local_daily,
)
from src.infrastructure.security_metadata import (
    _load_symbol_concepts,
    _load_tushare_stock_basic,
    _theme_from_concepts,
    _theme_from_name_and_industry,
)
from src.infrastructure.universe_feature_cache import (
    build_universe_generator_cache_key,
    load_universe_generator_cache,
    store_universe_generator_cache,
)

_GENERATOR_VERSION = "dynamic_universe_v1"


def _stable_hash(payload: object) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    import hashlib

    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_theme(value: object) -> str:
    text = str(value or "").strip()
    return text or "其他"


def _load_source_rows(*, universe_file: str, data_dir: str) -> list[Security]:
    rows = _load_universe_file(universe_file, enrich_metadata=False) if str(universe_file).strip() else []
    if rows:
        return _dedupe_rows(rows, exclude_symbols=(), enrich_metadata=False)
    return _dedupe_rows(_from_data_dir(data_dir=data_dir, limit=10000), exclude_symbols=(), enrich_metadata=False)


def _slice_daily_frame(frame: pd.DataFrame, end_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    sliced = frame
    end_text = str(end_date or "").strip()
    if end_text and end_text < "2099-01-01":
        cutoff = pd.Timestamp(end_text)
        sliced = sliced[sliced["date"] <= cutoff]
    return sliced.sort_values("date").reset_index(drop=True)


def _ret(frame: pd.DataFrame, lookback: int) -> float:
    if frame.empty or len(frame) <= lookback:
        return 0.0
    latest = float(frame["close"].iloc[-1])
    anchor = float(frame["close"].iloc[-1 - lookback])
    if not np.isfinite(latest) or not np.isfinite(anchor) or abs(anchor) < 1e-9:
        return 0.0
    return float(latest / anchor - 1.0)


def _drawdown(frame: pd.DataFrame, lookback: int) -> float:
    if frame.empty:
        return 0.0
    window = frame.tail(max(lookback, 5))
    closes = pd.to_numeric(window["close"], errors="coerce").dropna()
    if closes.empty:
        return 0.0
    peak = float(closes.max())
    last = float(closes.iloc[-1])
    if not np.isfinite(peak) or peak <= 0.0 or not np.isfinite(last):
        return 0.0
    return float(last / peak - 1.0)


def _volatility(frame: pd.DataFrame, lookback: int) -> float:
    if frame.empty or len(frame) <= 3:
        return 0.0
    returns = pd.to_numeric(frame["close"], errors="coerce").pct_change().dropna().tail(max(lookback, 5))
    if returns.empty:
        return 0.0
    return float(returns.std(ddof=0))


def _safe_rank(frame: pd.DataFrame, column: str, ascending: bool = False) -> pd.Series:
    if column not in frame.columns or frame.empty:
        return pd.Series([0.5] * len(frame), index=frame.index, dtype=float)
    return frame[column].rank(method="average", pct=True, ascending=ascending).fillna(0.5)


def _weighted_theme_strength(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}
    grouped = frame.groupby("theme", dropna=False).agg(
        mean_ret20=("ret20", "mean"),
        mean_ret60=("ret60", "mean"),
        mean_tradeability=("tradeability", "mean"),
        mean_amount=("median_amount60", "mean"),
        count=("symbol", "count"),
    )
    grouped["strength"] = (
        grouped["mean_ret20"] * 0.42
        + grouped["mean_ret60"] * 0.28
        + grouped["mean_tradeability"] * 0.20
        + grouped["mean_amount"].rank(method="average", pct=True) * 0.10
    )
    return {str(idx): float(value) for idx, value in grouped["strength"].items()}


def _compute_theme_allocations(
    *,
    refined: pd.DataFrame,
    target_size: int,
    theme_cap_ratio: float,
    theme_floor_count: int,
) -> dict[str, int]:
    if refined.empty or target_size <= 0:
        return {}
    counts = refined.groupby("theme")["symbol"].count().sort_values(ascending=False)
    mean_scores = refined.groupby("theme")["refined_score"].mean()
    strength = (mean_scores.rank(method="average", pct=True) * 0.7 + counts.rank(method="average", pct=True) * 0.3).to_dict()
    cap = max(3, int(math.ceil(float(target_size) * float(theme_cap_ratio))))
    floors = {
        str(theme): min(int(theme_floor_count), int(count))
        for theme, count in counts.items()
        if float(strength.get(theme, 0.0)) >= 0.6
    }
    allocations = {str(theme): floor for theme, floor in floors.items()}
    allocated = int(sum(allocations.values()))
    remaining = max(0, int(target_size) - allocated)
    ordered_themes = sorted(
        counts.index.tolist(),
        key=lambda theme: (-float(strength.get(theme, 0.0)), -int(counts.get(theme, 0)), str(theme)),
    )
    while remaining > 0 and ordered_themes:
        progressed = False
        for theme in ordered_themes:
            current = int(allocations.get(str(theme), 0))
            theme_supply = int(counts.get(theme, 0))
            if current >= min(cap, theme_supply):
                continue
            allocations[str(theme)] = current + 1
            remaining -= 1
            progressed = True
            if remaining <= 0:
                break
        if not progressed:
            break
    return allocations


def _theme_for_row(
    *,
    symbol: str,
    name: str,
    sector: str,
    metadata_map: dict[str, dict[str, str]],
    concept_map: dict[str, list[str]],
    use_concepts: bool,
) -> str:
    metadata = metadata_map.get(symbol, {})
    industry = str(metadata.get("industry", "")).strip()
    if use_concepts:
        concept_theme = _theme_from_concepts(concept_map.get(symbol, []))
        if concept_theme:
            return _normalize_theme(concept_theme)
    if str(sector).strip() and str(sector).strip() != "其他":
        return _normalize_theme(sector)
    return _normalize_theme(
        _theme_from_name_and_industry(
            symbol=symbol,
            name=name or str(metadata.get("name", symbol)),
            industry=industry,
        )
    )


def _serialize_records(frame: pd.DataFrame, columns: Iterable[str]) -> list[dict[str, object]]:
    if frame.empty:
        return []
    selected_columns = [column for column in columns if column in frame.columns]
    return [
        {
            str(column): (
                float(value)
                if isinstance(value, (np.floating, float))
                else int(value)
                if isinstance(value, (np.integer, int))
                else str(value)
            )
            for column, value in row.items()
        }
        for row in frame[selected_columns].to_dict(orient="records")
    ]


def _result_from_payload(payload: dict[str, object]) -> DynamicUniverseResult:
    theme_allocations = [
        ThemeAllocationRecord(**item)
        for item in payload.get("theme_allocations", [])
        if isinstance(item, dict)
    ]
    manifest_raw = payload.get("generator_manifest", {})
    manifest = UniverseGeneratorManifest(
        generator_version=str(manifest_raw.get("generator_version", "")),
        source_universe_path=str(manifest_raw.get("source_universe_path", "")),
        source_universe_size=int(manifest_raw.get("source_universe_size", 0)),
        eligible_size=int(manifest_raw.get("eligible_size", 0)),
        coarse_pool_size=int(manifest_raw.get("coarse_pool_size", 0)),
        refined_pool_size=int(manifest_raw.get("refined_pool_size", 0)),
        selected_pool_size=int(manifest_raw.get("selected_pool_size", 0)),
        generator_hash=str(manifest_raw.get("generator_hash", "")),
        manifest_path=str(manifest_raw.get("manifest_path", "")),
        theme_allocations=theme_allocations,
        warnings=[str(item) for item in manifest_raw.get("warnings", [])],
        config={
            str(key): value
            for key, value in dict(manifest_raw.get("config", {})).items()
        },
    )
    return DynamicUniverseResult(
        eligible_symbols=[str(item) for item in payload.get("eligible_symbols", [])],
        coarse_pool=[dict(item) for item in payload.get("coarse_pool", []) if isinstance(item, dict)],
        refined_pool=[dict(item) for item in payload.get("refined_pool", []) if isinstance(item, dict)],
        selected_300=[dict(item) for item in payload.get("selected_300", []) if isinstance(item, dict)],
        theme_allocations=theme_allocations,
        generator_manifest=manifest,
    )


def generate_dynamic_universe(
    *,
    universe_file: str,
    data_dir: str,
    cache_root: str,
    target_size: int,
    coarse_size: int,
    theme_aware: bool,
    use_concepts: bool,
    end_date: str,
    min_history_days: int,
    min_recent_amount: float,
    theme_cap_ratio: float,
    theme_floor_count: int,
    turnover_quality_weight: float,
    theme_weight: float,
    refresh_cache: bool = False,
) -> DynamicUniverseResult:
    source_rows = _load_source_rows(universe_file=universe_file, data_dir=data_dir)
    metadata_map = _load_tushare_stock_basic()
    cache_payload = {
        "generator_version": _GENERATOR_VERSION,
        "universe_file": str(universe_file),
        "data_dir": str(data_dir),
        "source_universe_size": len(source_rows),
        "target_size": int(target_size),
        "coarse_size": int(coarse_size),
        "theme_aware": bool(theme_aware),
        "use_concepts": bool(use_concepts),
        "end_date": str(end_date),
        "min_history_days": int(min_history_days),
        "min_recent_amount": float(min_recent_amount),
        "theme_cap_ratio": float(theme_cap_ratio),
        "theme_floor_count": int(theme_floor_count),
        "turnover_quality_weight": float(turnover_quality_weight),
        "theme_weight": float(theme_weight),
    }
    cache_key = build_universe_generator_cache_key(cache_payload)
    if not refresh_cache:
        cached = load_universe_generator_cache(cache_root, cache_key)
        if cached:
            return _result_from_payload(cached)

    records: list[dict[str, object]] = []
    warnings: list[str] = []
    for row in source_rows:
        name = str(row.name or row.symbol)
        if "ST" in name.upper():
            continue
        local_path = _resolve_local_symbol_path(data_dir, row.symbol)
        frame = _slice_daily_frame(_safe_read_local_daily(local_path), end_date)
        if frame.empty:
            continue
        history_days = int(len(frame))
        if history_days < int(min_history_days):
            continue
        recent_60 = frame.tail(min(60, history_days))
        median_amount60 = float(pd.to_numeric(recent_60.get("amount"), errors="coerce").median())
        if not np.isfinite(median_amount60) or median_amount60 < float(min_recent_amount):
            continue
        tradeability = float(min(1.0, median_amount60 / max(float(min_recent_amount) * 6.0, 1.0)))
        record = {
            "symbol": str(row.symbol),
            "name": name,
            "sector": str(row.sector or "其他"),
            "theme": _normalize_theme(str(row.sector or "其他")),
            "history_days": history_days,
            "median_amount60": median_amount60,
            "ret20": _ret(frame, 20),
            "ret60": _ret(frame, 60),
            "drawdown60": _drawdown(frame, 60),
            "volatility20": _volatility(frame, 20),
            "tradeability": tradeability,
        }
        records.append(record)

    eligible = pd.DataFrame(records)
    if eligible.empty:
        manifest = UniverseGeneratorManifest(
            generator_version=_GENERATOR_VERSION,
            source_universe_path=str(universe_file),
            source_universe_size=len(source_rows),
            eligible_size=0,
            coarse_pool_size=0,
            refined_pool_size=0,
            selected_pool_size=0,
            generator_hash=cache_key,
            warnings=["dynamic universe generator returned no eligible symbols"],
            config=cache_payload,
        )
        result = DynamicUniverseResult(generator_manifest=manifest)
        store_universe_generator_cache(
            cache_root,
            cache_key,
            {
                "eligible_symbols": [],
                "coarse_pool": [],
                "refined_pool": [],
                "selected_300": [],
                "theme_allocations": [],
                "generator_manifest": asdict(manifest),
            },
        )
        return result

    eligible["theme"] = eligible["theme"].map(_normalize_theme)
    theme_strength = _weighted_theme_strength(eligible)
    eligible["theme_strength"] = eligible["theme"].map(lambda item: float(theme_strength.get(str(item), 0.0)))
    eligible["amount_rank"] = _safe_rank(eligible, "median_amount60")
    eligible["history_rank"] = _safe_rank(eligible, "history_days")
    eligible["ret20_rank"] = _safe_rank(eligible, "ret20")
    eligible["ret60_rank"] = _safe_rank(eligible, "ret60")
    eligible["drawdown_rank"] = _safe_rank(eligible, "drawdown60")
    eligible["volatility_rank"] = _safe_rank(eligible, "volatility20", ascending=True)
    eligible["tradeability_rank"] = _safe_rank(eligible, "tradeability")
    eligible["theme_strength_rank"] = _safe_rank(eligible, "theme_strength")
    eligible["coarse_score"] = (
        eligible["amount_rank"] * 0.24
        + eligible["history_rank"] * 0.10
        + eligible["ret20_rank"] * 0.18
        + eligible["ret60_rank"] * 0.12
        + eligible["drawdown_rank"] * 0.12
        + eligible["volatility_rank"] * 0.08
        + eligible["tradeability_rank"] * (0.08 + float(turnover_quality_weight) * 0.08)
        + eligible["theme_strength_rank"] * float(theme_weight)
    )
    eligible = eligible.sort_values(["coarse_score", "median_amount60", "history_days"], ascending=[False, False, False])
    coarse = eligible.head(max(int(target_size), int(coarse_size))).copy()

    if bool(use_concepts) and not coarse.empty:
        concept_map = _load_symbol_concepts([str(item) for item in coarse["symbol"].tolist()])
        coarse["theme"] = coarse.apply(
            lambda row: _theme_for_row(
                symbol=str(row["symbol"]),
                name=str(row["name"]),
                sector=str(row["sector"]),
                metadata_map=metadata_map,
                concept_map=concept_map,
                use_concepts=True,
            ),
            axis=1,
        )
    coarse_theme_strength = _weighted_theme_strength(coarse)
    coarse["theme_strength"] = coarse["theme"].map(lambda item: float(coarse_theme_strength.get(str(item), 0.0)))
    coarse["theme_rank"] = _safe_rank(coarse, "theme_strength")
    coarse["quality_score"] = (
        coarse["ret20_rank"] * 0.24
        + coarse["ret60_rank"] * 0.16
        + coarse["amount_rank"] * 0.18
        + coarse["drawdown_rank"] * 0.16
        + coarse["tradeability_rank"] * 0.14
        + coarse["volatility_rank"] * 0.12
    )
    coarse["theme_score"] = coarse["theme_rank"] * 0.65 + coarse["ret20_rank"] * 0.20 + coarse["ret60_rank"] * 0.15
    coarse["refined_score"] = coarse["quality_score"] * (1.0 - float(theme_weight)) + coarse["theme_score"] * float(theme_weight)
    coarse = coarse.sort_values(["refined_score", "quality_score", "median_amount60"], ascending=[False, False, False])
    refined = coarse.head(max(int(target_size), min(len(coarse), max(120, int(target_size) * 2)))).copy()

    selected_frames: list[pd.DataFrame] = []
    allocations_raw: list[ThemeAllocationRecord] = []
    if bool(theme_aware):
        allocations = _compute_theme_allocations(
            refined=refined,
            target_size=int(target_size),
            theme_cap_ratio=float(theme_cap_ratio),
            theme_floor_count=int(theme_floor_count),
        )
        for theme, theme_frame in refined.groupby("theme", sort=False):
            slot_count = int(allocations.get(str(theme), 0))
            if slot_count <= 0:
                continue
            picked = theme_frame.sort_values(["refined_score", "quality_score"], ascending=[False, False]).head(slot_count)
            if not picked.empty:
                selected_frames.append(picked)
            allocations_raw.append(
                ThemeAllocationRecord(
                    theme=str(theme),
                    selected_count=int(len(picked)),
                    refined_count=int(len(theme_frame)),
                    coarse_count=int((coarse["theme"] == theme).sum()),
                    eligible_count=max(
                        int((eligible["theme"] == theme).sum()),
                        int((coarse["theme"] == theme).sum()),
                    ),
                    theme_strength=float(coarse_theme_strength.get(str(theme), 0.0)),
                    cap=max(3, int(math.ceil(float(target_size) * float(theme_cap_ratio)))),
                    floor=min(int(theme_floor_count), int(len(theme_frame))),
                )
            )
    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else pd.DataFrame(columns=refined.columns)
    if len(selected) < int(target_size):
        selected_symbols = set(str(item) for item in selected.get("symbol", pd.Series(dtype=str)).tolist())
        top_up = refined[~refined["symbol"].isin(selected_symbols)].head(int(target_size) - len(selected))
        if not top_up.empty:
            selected = pd.concat([selected, top_up], ignore_index=True)
    selected = selected.sort_values(["refined_score", "quality_score", "median_amount60"], ascending=[False, False, False]).head(int(target_size))

    if not allocations_raw:
        allocations_raw = [
            ThemeAllocationRecord(
                theme=str(theme),
                selected_count=int((selected["theme"] == theme).sum()),
                refined_count=int((refined["theme"] == theme).sum()),
                coarse_count=int((coarse["theme"] == theme).sum()),
                eligible_count=max(
                    int((eligible["theme"] == theme).sum()),
                    int((coarse["theme"] == theme).sum()),
                ),
                theme_strength=float(coarse_theme_strength.get(str(theme), 0.0)),
                cap=max(3, int(math.ceil(float(target_size) * float(theme_cap_ratio)))),
                floor=0,
            )
            for theme in sorted({str(item) for item in selected.get("theme", pd.Series(dtype=str)).tolist()})
        ]

    selected_rows = [
        {
            "symbol": str(row["symbol"]),
            "name": str(row["name"]),
            "sector": str(row["theme"]),
            "coarse_score": float(row["coarse_score"]),
            "refined_score": float(row["refined_score"]),
            "theme_score": float(row["theme_score"]),
            "quality_score": float(row["quality_score"]),
        }
        for _, row in selected.iterrows()
    ]
    selected_manifest_stocks = [
        {
            "symbol": str(row["symbol"]),
            "name": str(row["name"]),
            "sector": str(row["theme"]),
        }
        for _, row in selected.iterrows()
    ]

    manifest_dir = Path(cache_root) / "universe_catalog"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_stem = f"dynamic_{int(target_size)}_{cache_key[:12]}"
    universe_manifest_path = manifest_dir / f"{manifest_stem}.json"
    generator_manifest_path = manifest_dir / f"{manifest_stem}.generator.json"
    universe_payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "universe_id": f"dynamic_{int(target_size)}",
        "universe_size": int(len(selected_manifest_stocks)),
        "source": str(universe_file or data_dir),
        "generation_rule": f"{_GENERATOR_VERSION}: coarse->{int(coarse_size)} refine->{int(target_size)}",
        "generator_manifest": str(generator_manifest_path.resolve()),
        "stocks": selected_manifest_stocks,
        "symbols": [str(item["symbol"]) for item in selected_manifest_stocks],
        "symbol_count": int(len(selected_manifest_stocks)),
        "theme_allocations": [asdict(item) for item in allocations_raw],
    }
    universe_manifest_path.write_text(json.dumps(universe_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    generator_manifest = UniverseGeneratorManifest(
        generator_version=_GENERATOR_VERSION,
        source_universe_path=str(Path(universe_file).resolve()) if str(universe_file).strip() else str(Path(data_dir).resolve()),
        source_universe_size=int(len(source_rows)),
        eligible_size=int(len(eligible)),
        coarse_pool_size=int(len(coarse)),
        refined_pool_size=int(len(refined)),
        selected_pool_size=int(len(selected)),
        generator_hash=cache_key,
        manifest_path=str(generator_manifest_path.resolve()),
        theme_allocations=allocations_raw,
        warnings=warnings,
        config=cache_payload,
    )
    generator_manifest_path.write_text(json.dumps(asdict(generator_manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    result = DynamicUniverseResult(
        eligible_symbols=[str(item) for item in eligible["symbol"].tolist()],
        coarse_pool=_serialize_records(
            coarse,
            ("symbol", "name", "theme", "coarse_score", "median_amount60", "ret20", "ret60"),
        ),
        refined_pool=_serialize_records(
            refined,
            ("symbol", "name", "theme", "refined_score", "theme_score", "quality_score"),
        ),
        selected_300=selected_rows,
        theme_allocations=allocations_raw,
        generator_manifest=generator_manifest,
    )
    store_universe_generator_cache(
        cache_root,
        cache_key,
        {
            "eligible_symbols": result.eligible_symbols,
            "coarse_pool": result.coarse_pool,
            "refined_pool": result.refined_pool,
            "selected_300": result.selected_300,
            "theme_allocations": [asdict(item) for item in allocations_raw],
            "generator_manifest": asdict(generator_manifest),
        },
    )
    return result
