from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.application.v2_contracts import InfoItem
from src.domain.news import normalize_direction, normalize_horizon, normalize_target, normalize_target_type

_REQUIRED_COLUMNS = ("date", "target_type", "target", "direction")
_INFO_TYPE_WEIGHTS = {
    "announcement": 1.0,
    "news": 0.85,
    "research": 0.70,
}


def _normalize_info_type(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"announcement", "ann", "gonggao", "公告"}:
        return "announcement"
    if text in {"research", "yanbao", "研报"}:
        return "research"
    return "news"


def _normalize_event_tag(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "earnings_positive": "earnings_positive",
        "earnings_negative": "earnings_negative",
        "guidance_positive": "guidance_positive",
        "guidance_negative": "guidance_negative",
        "contract_win": "contract_win",
        "contract_loss": "contract_loss",
        "regulatory_positive": "regulatory_positive",
        "regulatory_negative": "regulatory_negative",
        "share_reduction": "share_reduction",
        "share_increase": "share_increase",
        "trading_halt": "trading_halt",
        "delisting_risk": "delisting_risk",
    }
    return aliases.get(text, "")


def _validate_columns(raw: pd.DataFrame, *, source_label: str) -> dict[str, str]:
    lower_map = {str(col).lower(): str(col) for col in raw.columns}
    for col in _REQUIRED_COLUMNS:
        if col not in lower_map:
            raise ValueError(f"Info CSV missing required column `{col}`: {source_label}")
    return lower_map


def _load_raw_info(path: Path) -> pd.DataFrame:
    if path.is_file():
        raw = pd.read_csv(path)
        if raw.empty:
            return raw
        _validate_columns(raw, source_label=str(path))
        return raw

    files = sorted(p for p in path.rglob("*.csv") if p.is_file())
    if not files:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for file in files:
        raw = pd.read_csv(file)
        if raw.empty:
            continue
        _validate_columns(raw, source_label=str(file))
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_v2_info_items(
    csv_path: str | Path,
    *,
    as_of_date: pd.Timestamp,
    lookback_days: int = 45,
    info_types: Iterable[str] = ("news", "announcement", "research"),
) -> list[InfoItem]:
    path = Path(csv_path)
    if not path.exists():
        return []

    allowed_info_types = {_normalize_info_type(item) for item in info_types}
    raw = _load_raw_info(path)
    if raw.empty:
        return []

    lower_map = _validate_columns(raw, source_label=str(path))
    out_by_key: dict[tuple[object, ...], InfoItem] = {}
    for _, row in raw.iterrows():
        date = pd.to_datetime(row[lower_map["date"]], errors="coerce")
        if pd.isna(date):
            continue
        date = pd.Timestamp(date).normalize()
        if date > as_of_date:
            continue
        if (as_of_date - date).days > int(lookback_days):
            continue

        target_type = normalize_target_type(str(row[lower_map["target_type"]]))
        if target_type not in {"market", "stock"}:
            continue
        target = normalize_target(target_type, str(row[lower_map["target"]]))
        horizon = "both"
        if "horizon" in lower_map:
            horizon = normalize_horizon(str(row[lower_map["horizon"]]))
        direction_raw = str(row[lower_map["direction"]]).strip()
        if normalize_direction(direction_raw) not in {-1, 0, 1}:
            continue

        info_type = _normalize_info_type(row[lower_map["info_type"]]) if "info_type" in lower_map else "news"
        if info_type not in allowed_info_types:
            continue
        title = str(row[lower_map["title"]]).strip() if "title" in lower_map else ""
        source_url = str(row[lower_map["source_url"]]).strip() if "source_url" in lower_map else ""
        strength = float(row[lower_map["strength"]]) if "strength" in lower_map else 3.0
        confidence = float(row[lower_map["confidence"]]) if "confidence" in lower_map else 0.7
        source_weight = (
            float(row[lower_map["source_weight"]])
            if "source_weight" in lower_map
            else float(_INFO_TYPE_WEIGHTS.get(info_type, 0.85))
        )
        publisher = str(row[lower_map["publisher"]]).strip() if "publisher" in lower_map else ""
        event_tag = _normalize_event_tag(row[lower_map["event_tag"]]) if "event_tag" in lower_map else ""
        event_id = str(row[lower_map["event_id"]]).strip() if "event_id" in lower_map else ""
        item = InfoItem(
            date=str(date.date()),
            target_type=target_type,
            target=target,
            horizon=horizon,
            direction=direction_raw,
            info_type=info_type,
            title=title,
            source_url=source_url,
            strength=float(strength),
            confidence=float(confidence),
            source_weight=float(source_weight),
            publisher=publisher,
            event_tag=event_tag,
            event_id=event_id,
        )
        key = (
            item.date,
            item.target_type,
            item.target,
            item.horizon,
            item.info_type,
            item.direction,
            item.title,
            item.source_url,
        )
        existing = out_by_key.get(key)
        if existing is None or float(item.confidence) > float(existing.confidence):
            out_by_key[key] = item

    items = list(out_by_key.values())
    items.sort(
        key=lambda item: (
            item.date,
            item.target_type,
            item.target,
            item.horizon,
            item.info_type,
            item.direction,
            item.title,
        )
    )
    return items
