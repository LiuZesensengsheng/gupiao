from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from src.application.v2_contracts import CompositeState, DailyRunResult, InfoItem, Viewpoint
from src.application.v2_stock_role_runtime import build_stock_role_snapshots
from src.application.v2_theme_episode_runtime import build_theme_episodes
from src.domain.info_clock import DEFAULT_INFO_CUTOFF_TIME, as_of_day_cutoff, item_available_as_of, parse_timestamp


_NOTE_FIELDS = {
    "target_type",
    "target",
    "theme",
    "direction",
    "confidence",
    "importance",
    "horizon",
    "reason",
    "invalid_if",
    "event_tag",
    "source",
}


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_direction(value: object) -> str:
    text = _normalize_text(value).lower()
    if text in {"bull", "bullish", "long", "positive", "up"}:
        return "bullish"
    if text in {"bear", "bearish", "short", "negative", "down"}:
        return "bearish"
    return "neutral"


def _reason_hash(value: object) -> str:
    raw = _normalize_text(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16] if raw else ""


def source_weight_for_source(source: object) -> float:
    text = _normalize_text(source).lower()
    if text in {"manual_note", "manual", "note"}:
        return 1.15
    if "announcement" in text:
        return 1.10
    if "research" in text:
        return 1.00
    if "news" in text:
        return 0.90
    return 0.95


def _front_matter_and_body(text: str) -> tuple[dict[str, str], str]:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return {}, text
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    front_lines: list[str] = []
    body_start = 1
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            body_start = idx + 1
            break
        front_lines.append(lines[idx])
    front: dict[str, str] = {}
    for raw in front_lines:
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        front[_normalize_text(key)] = _normalize_text(value)
    body = "\n".join(lines[body_start:])
    return front, body


def _parse_note_blocks(body: str) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("##") or line.startswith("###"):
            if current:
                blocks.append(current)
                current = {}
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = _normalize_text(key)
        if normalized_key not in _NOTE_FIELDS:
            continue
        if normalized_key in current and current:
            blocks.append(current)
            current = {}
        current[normalized_key] = _normalize_text(value)
    if current:
        blocks.append(current)
    return blocks


def build_viewpoints_from_notes(
    *,
    note_dir: str,
    as_of_date: pd.Timestamp,
    lookback_days: int,
    cutoff_time: str = DEFAULT_INFO_CUTOFF_TIME,
) -> list[Viewpoint]:
    base = Path(str(note_dir))
    if not base.exists():
        return []
    cutoff = as_of_date.normalize() - pd.Timedelta(days=max(0, int(lookback_days)))
    out: list[Viewpoint] = []
    cutoff_ts = as_of_day_cutoff(as_of_date, cutoff_time=cutoff_time)
    for path in sorted(base.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
        except Exception:
            continue
        front, body = _front_matter_and_body(raw)
        effective_ts = parse_timestamp(front.get("effective_time")) or parse_timestamp(path.stem)
        if effective_ts is None:
            effective_ts = pd.Timestamp(path.stat().st_mtime, unit="s")
        effective_ts = effective_ts.normalize()
        if effective_ts > as_of_date.normalize() or effective_ts < cutoff:
            continue
        ingest_ts = parse_timestamp(front.get("ingest_time")) or pd.Timestamp(path.stat().st_mtime, unit="s")
        if ingest_ts > cutoff_ts:
            continue
        for block in _parse_note_blocks(body):
            target_type = _normalize_text(block.get("target_type"))
            target = _normalize_text(block.get("target"))
            if not target_type or not target:
                continue
            theme = _normalize_text(block.get("theme")) or (target if target_type == "sector" else "")
            source = _normalize_text(block.get("source")) or _normalize_text(front.get("source")) or "manual_note"
            reason = _normalize_text(block.get("reason"))
            out.append(
                Viewpoint(
                    target_type=target_type,
                    target=target,
                    theme=theme,
                    direction=_normalize_direction(block.get("direction")),
                    confidence=_clip01(float(block.get("confidence", 0.6) or 0.6)),
                    importance=_clip01(float(block.get("importance", 0.6) or 0.6)),
                    horizon=_normalize_text(block.get("horizon")) or "mid",
                    reason=reason,
                    invalid_if=_normalize_text(block.get("invalid_if")),
                    event_tag=_normalize_text(block.get("event_tag")),
                    source=source,
                    source_weight=source_weight_for_source(source),
                    effective_time=str(effective_ts.date()),
                    ingest_time=str(ingest_ts.isoformat()),
                    reason_hash=_reason_hash(reason),
                )
            )
    return out


def build_viewpoints_from_info_items(
    *,
    info_items: list[InfoItem],
    as_of_date: pd.Timestamp,
    cutoff_time: str = DEFAULT_INFO_CUTOFF_TIME,
) -> list[Viewpoint]:
    availability_cutoff = as_of_day_cutoff(as_of_date, cutoff_time=cutoff_time)
    out: list[Viewpoint] = []
    for item in info_items:
        if not item_available_as_of(item, as_of_date, cutoff_time=cutoff_time, availability_cutoff=availability_cutoff):
            continue
        effective_ts = parse_timestamp(item.date) or as_of_date
        if effective_ts.normalize() > as_of_date.normalize():
            continue
        ingest_ts = parse_timestamp(getattr(item, "publish_datetime", "")) or as_of_day_cutoff(effective_ts)
        source = _normalize_text(item.source_subset) or _normalize_text(item.info_type) or "market_news"
        reason = _normalize_text(item.title)
        theme = _normalize_text(item.target) if _normalize_text(item.target_type) == "sector" else ""
        out.append(
            Viewpoint(
                target_type=_normalize_text(item.target_type),
                target=_normalize_text(item.target),
                theme=theme,
                direction=_normalize_direction(item.direction),
                confidence=_clip01(float(getattr(item, "confidence", 0.5))),
                importance=_clip01(min(1.0, float(getattr(item, "strength", 3.0)) / 5.0)),
                horizon=_normalize_text(item.horizon) or "mid",
                reason=reason,
                invalid_if="",
                event_tag=_normalize_text(item.event_tag),
                source=source,
                source_weight=(
                    float(getattr(item, "source_weight", 0.0))
                    if float(getattr(item, "source_weight", 0.0)) > 0.0
                    else source_weight_for_source(source)
                ),
                effective_time=str(effective_ts.normalize().date()),
                ingest_time=str(ingest_ts.isoformat()),
                reason_hash=_reason_hash(reason),
            )
        )
    return out


def deduplicate_viewpoints(
    *,
    viewpoints: list[Viewpoint],
    as_of_date: pd.Timestamp,
    half_life_days: float,
) -> list[Viewpoint]:
    deduped: dict[tuple[str, str, str, str, str], Viewpoint] = {}
    for item in viewpoints:
        effective_ts = parse_timestamp(item.effective_time) or as_of_date
        age_days = max(0.0, float((as_of_date.normalize() - effective_ts.normalize()).days))
        recency = float(0.5 ** (age_days / max(1.0, float(half_life_days))))
        weight = float(recency * float(item.confidence) * float(item.importance) * float(item.source_weight))
        normalized = replace(item, weight=weight)
        key = (
            str(effective_ts.date()),
            _normalize_text(item.target_type).lower(),
            _normalize_text(item.target).lower(),
            _normalize_text(item.theme).lower(),
            _normalize_text(item.direction).lower(),
            _normalize_text(item.reason_hash).lower(),
        )
        current = deduped.get(key)
        if current is None or float(weight) > float(current.weight):
            deduped[key] = normalized
    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            str(item.effective_time),
            float(item.weight),
            float(item.confidence),
            float(item.importance),
        ),
        reverse=True,
    )
    return ordered


def build_viewpoints(
    *,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
) -> list[Viewpoint]:
    note_dir = str(settings.get("insight_notes_dir", "input/insight_notes"))
    lookback_days = int(
        settings.get(
            "insight_lookback_days",
            settings.get("event_lookback_days", settings.get("info_lookback_days", 45)),
        )
    )
    half_life_days = float(settings.get("info_half_life_days", 10.0))
    cutoff_time = str(settings.get("info_cutoff_time", DEFAULT_INFO_CUTOFF_TIME))
    raw = build_viewpoints_from_notes(
        note_dir=note_dir,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
        cutoff_time=cutoff_time,
    )
    raw.extend(
        build_viewpoints_from_info_items(
            info_items=info_items,
            as_of_date=as_of_date,
            cutoff_time=cutoff_time,
        )
    )
    return deduplicate_viewpoints(
        viewpoints=raw,
        as_of_date=as_of_date,
        half_life_days=half_life_days,
    )


def attach_insight_memory_to_state(
    *,
    state: CompositeState,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
) -> CompositeState:
    if not bool(settings.get("enable_insight_memory", True)):
        return replace(state, viewpoints=[], theme_episodes=[], stock_role_states={}, execution_plans=[])
    viewpoints = build_viewpoints(
        settings=settings,
        as_of_date=as_of_date.normalize(),
        info_items=info_items,
    )
    if not viewpoints:
        return replace(state, viewpoints=[], theme_episodes=[], stock_role_states={}, execution_plans=[])
    base_state = replace(state, viewpoints=viewpoints, theme_episodes=[], stock_role_states={}, execution_plans=[])
    theme_episodes = build_theme_episodes(state=base_state)
    if not theme_episodes:
        return replace(base_state, theme_episodes=[], stock_role_states={}, execution_plans=[])
    stock_roles = build_stock_role_snapshots(
        state=replace(base_state, theme_episodes=theme_episodes),
        theme_episodes=theme_episodes,
        previous_roles=getattr(state, "stock_role_states", {}),
    )
    return replace(
        base_state,
        theme_episodes=theme_episodes,
        stock_role_states=stock_roles,
        execution_plans=[],
    )


def build_insight_manifest_payload(
    *,
    state: CompositeState,
    settings: dict[str, object],
    execution_plans: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    viewpoints = list(getattr(state, "viewpoints", []) or [])
    theme_episodes = list(getattr(state, "theme_episodes", []) or [])
    role_states = dict(getattr(state, "stock_role_states", {}) or {})
    execution_plans = list(execution_plans if execution_plans is not None else [asdict(item) for item in getattr(state, "execution_plans", [])])
    source_breakdown: dict[str, int] = {}
    for item in viewpoints:
        key = str(item.source or "unknown")
        source_breakdown[key] = int(source_breakdown.get(key, 0) + 1)
    phase_counts: dict[str, int] = {}
    for item in theme_episodes:
        key = str(item.phase or "unknown")
        phase_counts[key] = int(phase_counts.get(key, 0) + 1)
    role_counts: dict[str, int] = {}
    role_downgrade_count = 0
    for item in role_states.values():
        role = str(item.role or "unknown")
        role_counts[role] = int(role_counts.get(role, 0) + 1)
        if bool(item.role_downgrade):
            role_downgrade_count += 1
    return {
        "as_of_date": str(getattr(getattr(state, "market", None), "as_of_date", "")),
        "enable_insight_memory": bool(settings.get("enable_insight_memory", True)),
        "insight_notes_dir": str(settings.get("insight_notes_dir", "input/insight_notes")),
        "viewpoint_count": int(len(viewpoints)),
        "theme_episode_count": int(len(theme_episodes)),
        "stock_role_count": int(len(role_states)),
        "execution_plan_count": int(len(execution_plans)),
        "source_breakdown": source_breakdown,
        "phase_counts": phase_counts,
        "role_counts": role_counts,
        "role_downgrade_count": int(role_downgrade_count),
    }


def build_insight_artifact_payloads(
    *,
    state: CompositeState,
    settings: dict[str, object],
    execution_plans: list[dict[str, object]] | None = None,
) -> dict[str, dict[str, object] | list[dict[str, object]]]:
    plan_rows = list(execution_plans if execution_plans is not None else [asdict(item) for item in getattr(state, "execution_plans", [])])
    return {
        "insight_manifest": build_insight_manifest_payload(
            state=state,
            settings=settings,
            execution_plans=plan_rows,
        ),
        "viewpoints": [asdict(item) for item in getattr(state, "viewpoints", [])],
        "theme_episodes": [asdict(item) for item in getattr(state, "theme_episodes", [])],
        "stock_roles": [asdict(item) for item in getattr(state, "stock_role_states", {}).values()],
        "execution_plan": plan_rows,
    }


def write_insight_artifacts(
    *,
    base_dir: Path,
    payloads: dict[str, dict[str, object] | list[dict[str, object]]],
) -> dict[str, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "insight_manifest": "insight_manifest.json",
        "viewpoints": "viewpoints.json",
        "theme_episodes": "theme_episodes.json",
        "stock_roles": "stock_roles.json",
        "execution_plan": "execution_plan.json",
    }
    written: dict[str, str] = {}
    for key, filename in mapping.items():
        path = base_dir / filename
        payload = payloads.get(key, {})
        if key != "insight_manifest" and isinstance(payload, list):
            payload = {"items": payload}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        written[key] = str(path)
    return written


def resolve_daily_insight_artifact_dir(
    *,
    artifact_root: str,
    strategy_id: str,
    run_id: str,
    manifest_path: str,
) -> Path:
    manifest = Path(str(manifest_path)) if str(manifest_path).strip() else Path()
    candidate = Path(str(artifact_root)) / str(strategy_id) / str(run_id).strip()
    if str(run_id).strip():
        return candidate
    if manifest and manifest.name == "research_manifest.json":
        return manifest.parent
    if manifest and manifest.name:
        return manifest.parent / "daily_current"
    return Path(str(artifact_root)) / str(strategy_id) / "daily_current"


def persist_daily_insight_artifacts(
    *,
    result: DailyRunResult,
    settings: dict[str, object],
    artifact_root: str,
) -> dict[str, str]:
    base_dir = resolve_daily_insight_artifact_dir(
        artifact_root=artifact_root,
        strategy_id=str(result.snapshot.strategy_id),
        run_id=str(result.snapshot.run_id or result.run_id or ""),
        manifest_path=str(result.snapshot.manifest_path or result.manifest_path or ""),
    )
    payloads = build_insight_artifact_payloads(
        state=result.composite_state,
        settings=settings,
    )
    return write_insight_artifacts(base_dir=base_dir, payloads=payloads)
