from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


DEFAULT_INFO_CUTOFF_TIME = "23:59:59"
_EASTMONEY_ARTICLE_RE = re.compile(r"/AN(?P<digits>\d{12,18})", re.IGNORECASE)


def parse_timestamp(value: object) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def _normalized_cutoff_time(value: object) -> str:
    text = str(value or "").strip()
    return text or DEFAULT_INFO_CUTOFF_TIME


def _time_delta_for_cutoff(cutoff_time: object) -> pd.Timedelta:
    text = _normalized_cutoff_time(cutoff_time)
    if text.lower() in {"eod", "end_of_day", "day_end"}:
        return pd.Timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        return pd.Timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except Exception:
        return pd.Timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)
    hour = max(0, min(23, hour))
    minute = max(0, min(59, minute))
    second = max(0, min(59, second))
    return pd.Timedelta(hours=hour, minutes=minute, seconds=second)


def as_of_day_cutoff(as_of_date: pd.Timestamp, cutoff_time: object = DEFAULT_INFO_CUTOFF_TIME) -> pd.Timestamp:
    base = pd.Timestamp(as_of_date).normalize()
    return base + _time_delta_for_cutoff(cutoff_time)


def derive_publish_timestamp_from_source_url(source_url: object) -> pd.Timestamp | None:
    text = str(source_url or "").strip()
    if not text:
        return None
    match = _EASTMONEY_ARTICLE_RE.search(text)
    if match is None:
        return None
    digits = match.group("digits")
    candidates = []
    if len(digits) >= 14:
        candidates.append(digits[:14])
    if len(digits) >= 12:
        candidates.append(digits[:12])
    for candidate in candidates:
        fmt = "%Y%m%d%H%M%S" if len(candidate) == 14 else "%Y%m%d%H%M"
        try:
            ts = pd.to_datetime(candidate, format=fmt, errors="coerce")
        except Exception:
            continue
        if pd.isna(ts):
            continue
        return pd.Timestamp(ts)
    return None


def publish_timestamp_for_item(
    item: object,
    *,
    missing_publish_cutoff_time: object = DEFAULT_INFO_CUTOFF_TIME,
) -> pd.Timestamp | None:
    publish_ts = parse_timestamp(getattr(item, "publish_datetime", ""))
    if publish_ts is not None:
        return publish_ts
    source_ts = derive_publish_timestamp_from_source_url(getattr(item, "source_url", ""))
    if source_ts is not None:
        return source_ts
    date_ts = parse_timestamp(getattr(item, "date", ""))
    if date_ts is None:
        return None
    return as_of_day_cutoff(date_ts, cutoff_time=missing_publish_cutoff_time)


def item_available_as_of(
    item: object,
    as_of_date: pd.Timestamp,
    *,
    cutoff_time: object = DEFAULT_INFO_CUTOFF_TIME,
    availability_cutoff: pd.Timestamp | None = None,
) -> bool:
    publish_ts = publish_timestamp_for_item(
        item,
        missing_publish_cutoff_time=cutoff_time,
    )
    if publish_ts is None:
        return True
    cutoff = availability_cutoff or as_of_day_cutoff(as_of_date, cutoff_time=cutoff_time)
    return bool(publish_ts <= cutoff)


def filter_items_as_of(
    items: Iterable[object],
    as_of_date: pd.Timestamp,
    *,
    cutoff_time: object = DEFAULT_INFO_CUTOFF_TIME,
    availability_cutoff: pd.Timestamp | None = None,
) -> list[object]:
    cutoff = availability_cutoff or as_of_day_cutoff(as_of_date, cutoff_time=cutoff_time)
    filtered: list[object] = []
    for item in items:
        publish_ts = publish_timestamp_for_item(
            item,
            missing_publish_cutoff_time=cutoff_time,
        )
        if publish_ts is None or publish_ts <= cutoff:
            filtered.append(item)
    return filtered
