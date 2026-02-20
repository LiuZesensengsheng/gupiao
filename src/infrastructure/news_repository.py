from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.domain.entities import NewsItem
from src.domain.news import normalize_horizon, normalize_target, normalize_target_type

_REQUIRED_COLUMNS = ("date", "target_type", "target", "direction")


def _validate_columns(raw: pd.DataFrame, *, source_label: str) -> dict[str, str]:
    lower_map = {c.lower(): c for c in raw.columns}
    for col in _REQUIRED_COLUMNS:
        if col not in lower_map:
            raise ValueError(f"News CSV missing required column `{col}`: {source_label}")
    return lower_map


def _load_raw_news(path: Path) -> pd.DataFrame:
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


def load_news_items(
    csv_path: str | Path,
    as_of_date: pd.Timestamp,
    lookback_days: int = 45,
) -> List[NewsItem]:
    path = Path(csv_path)
    if not path.exists():
        return []

    raw = _load_raw_news(path)
    if raw.empty:
        return []

    lower_map = _validate_columns(raw, source_label=str(path))

    out: List[NewsItem] = []
    for _, row in raw.iterrows():
        date = pd.to_datetime(row[lower_map["date"]], errors="coerce")
        if pd.isna(date):
            continue
        if date > as_of_date:
            continue
        if (as_of_date - date).days > lookback_days:
            continue

        target_type = normalize_target_type(str(row[lower_map["target_type"]]))
        target = normalize_target(target_type, str(row[lower_map["target"]]))
        direction = str(row[lower_map["direction"]]).strip()

        horizon = "both"
        if "horizon" in lower_map:
            horizon = normalize_horizon(str(row[lower_map["horizon"]]))

        strength = 3.0 if "strength" not in lower_map else float(row[lower_map["strength"]])
        confidence = 0.7 if "confidence" not in lower_map else float(row[lower_map["confidence"]])
        source_weight = 0.7 if "source_weight" not in lower_map else float(row[lower_map["source_weight"]])
        title = "" if "title" not in lower_map else str(row[lower_map["title"]])
        source_url = "" if "source_url" not in lower_map else str(row[lower_map["source_url"]])

        out.append(
            NewsItem(
                date=pd.Timestamp(date.normalize()),
                target_type=target_type,
                target=target,
                horizon=horizon,
                direction=direction,
                strength=float(strength),
                confidence=float(confidence),
                source_weight=float(source_weight),
                title=title,
                source_url=source_url,
            )
        )
    deduped: list[NewsItem] = []
    seen: set[tuple[object, ...]] = set()
    for item in out:
        key = (
            item.date,
            item.target_type,
            item.target,
            item.horizon,
            item.direction,
            item.strength,
            item.confidence,
            item.source_weight,
            item.title,
            item.source_url,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    deduped.sort(key=lambda x: (x.date, x.target_type, x.target, x.horizon, x.direction))
    return deduped
