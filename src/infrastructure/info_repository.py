from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.application.v2_contracts import InfoItem
from src.domain.info_clock import (
    DEFAULT_INFO_CUTOFF_TIME,
    as_of_day_cutoff,
    derive_publish_timestamp_from_source_url,
    parse_timestamp,
)
from src.domain.news import normalize_direction, normalize_horizon, normalize_target, normalize_target_type

_REQUIRED_COLUMNS = ("date", "target_type", "target", "direction")
_PUBLISH_DATETIME_COLUMNS = ("publish_datetime", "published_at", "publish_time", "datetime")
_TIME_ONLY_COLUMNS = ("time",)
_INFO_TYPE_WEIGHTS = {
    "announcement": 1.0,
    "news": 0.85,
    "research": 0.70,
}
_SUBSET_DEFAULT_INFO_TYPE = {
    "market_news": "news",
    "announcements": "announcement",
    "research": "research",
}
_STRONG_ANNOUNCEMENT_EVENT_TAGS = (
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
)
_ANNOUNCEMENT_KEYWORDS = (
    "公告",
    "提示性",
    "权益变动",
    "董事会",
    "监事会",
    "股东",
    "减持",
    "增持",
    "回购",
    "质押",
    "停牌",
    "复牌",
    "分红",
    "担保",
    "签署",
    "协议",
    "交易异常波动",
    "项目延期",
    "主要经营数据",
    "业绩预增",
    "业绩预减",
    "业绩预亏",
    "业绩快报",
)
_RESEARCH_KEYWORDS = (
    "研报",
    "点评",
    "首次覆盖",
    "买入",
    "增持评级",
    "推荐",
    "目标价",
    "核查意见",
    "保荐",
    "证券股份有限公司关于",
)


def _normalize_info_type(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"announcement", "ann", "gonggao", "公告"}:
        return "announcement"
    if text in {"research", "yanbao", "研报"}:
        return "research"
    return "news"


def _normalize_info_subset(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"market_news", "market", "macro", "news"}:
        return "market_news"
    if text in {"announcements", "announcement", "ann", "gonggao", "公告"}:
        return "announcements"
    if text in {"research", "report", "yanbao", "研报"}:
        return "research"
    return ""


def _infer_info_type(*, title: object, source_url: object, publisher: object) -> str:
    title_text = str(title).strip()
    source_text = str(source_url).strip().lower()
    publisher_text = str(publisher).strip()
    merged = " ".join([title_text, publisher_text]).lower()
    if any(keyword.lower() in merged for keyword in _RESEARCH_KEYWORDS):
        return "research"
    if any(keyword.lower() in merged for keyword in _ANNOUNCEMENT_KEYWORDS):
        return "announcement"
    if "data.eastmoney.com/notices/detail" in source_text:
        return "announcement"
    return "news"


def _subset_from_info_type(info_type: str) -> str:
    if info_type == "announcement":
        return "announcements"
    if info_type == "research":
        return "research"
    return "market_news"


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


def _infer_event_tag(*, title: object, direction: object, info_type: str) -> str:
    text = str(title).strip().lower()
    direction_text = str(direction).strip().lower()

    if "退市" in text:
        return "delisting_risk"
    if "停牌" in text:
        return "trading_halt"
    if "减持" in text:
        return "share_reduction"
    if "增持" in text or "回购" in text or "注销回购股份" in text:
        return "share_increase"
    if "中标" in text or "签订合同" in text or "订单" in text or "签署" in text:
        if any(token in text for token in ("解除", "终止", "取消", "失败")):
            return "contract_loss"
        if direction_text != "bearish":
            return "contract_win"
    if any(token in text for token in ("监管", "问询", "处罚", "立案", "调查", "风险提示", "行政监管")):
        return "regulatory_negative"
    if any(token in text for token in ("获批", "核准", "通过", "支持")) and info_type != "research":
        return "regulatory_positive"
    if any(token in text for token in ("业绩预增", "扭亏", "同比增长", "净利润增长", "盈利增长", "业绩快报")):
        return "earnings_positive"
    if any(token in text for token in ("业绩预亏", "业绩预减", "亏损", "扣非亏损", "同比下滑", "下滑", "下降", "由盈转亏")):
        return "earnings_negative"
    if "指引" in text or "展望" in text or "预计" in text:
        return "guidance_negative" if direction_text == "bearish" else "guidance_positive"
    return ""


def _validate_columns(raw: pd.DataFrame, *, source_label: str) -> dict[str, str]:
    lower_map = {str(col).lower(): str(col) for col in raw.columns}
    for col in _REQUIRED_COLUMNS:
        if col not in lower_map:
            raise ValueError(f"Info CSV missing required column `{col}`: {source_label}")
    return lower_map


def _default_subset_for_file(*, file: Path, root: Path) -> str:
    try:
        rel = file.relative_to(root)
        for part in rel.parts[:-1]:
            subset = _normalize_info_subset(part)
            if subset:
                return subset
    except Exception:
        pass
    return _normalize_info_subset(file.parent.name)


def _load_raw_info(path: Path) -> pd.DataFrame:
    if path.is_file():
        raw = pd.read_csv(path)
        if raw.empty:
            return raw
        _validate_columns(raw, source_label=str(path))
        raw["__default_subset"] = _default_subset_for_file(file=path, root=path.parent)
        raw["__source_file"] = str(path)
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
        raw["__default_subset"] = _default_subset_for_file(file=file, root=path)
        raw["__source_file"] = str(file)
        frames.append(raw)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _resolve_publish_datetime(
    row: pd.Series,
    *,
    lower_map: dict[str, str],
    base_date: pd.Timestamp,
    source_url: object,
) -> str:
    for col in _PUBLISH_DATETIME_COLUMNS:
        source_col = lower_map.get(col)
        if source_col is None:
            continue
        publish_ts = parse_timestamp(row[source_col])
        if publish_ts is None:
            continue
        return str(publish_ts.isoformat())
    for col in _TIME_ONLY_COLUMNS:
        source_col = lower_map.get(col)
        if source_col is None:
            continue
        time_text = str(row[source_col] or "").strip()
        if not time_text:
            continue
        publish_ts = parse_timestamp(f"{base_date.date()} {time_text}")
        if publish_ts is None:
            continue
        return str(publish_ts.isoformat())
    derived_source_ts = derive_publish_timestamp_from_source_url(source_url)
    if derived_source_ts is not None:
        return str(derived_source_ts.isoformat())
    return ""


def load_v2_info_items(
    csv_path: str | Path,
    *,
    as_of_date: pd.Timestamp,
    lookback_days: int = 45,
    source_mode: str = "layered",
    info_types: Iterable[str] = ("news", "announcement", "research"),
    info_subsets: Iterable[str] = ("market_news", "announcements", "research"),
    announcement_event_tags: Iterable[str] = _STRONG_ANNOUNCEMENT_EVENT_TAGS,
    cutoff_time: str = DEFAULT_INFO_CUTOFF_TIME,
) -> list[InfoItem]:
    path = Path(csv_path)
    if not path.exists():
        return []

    normalized_source_mode = str(source_mode).strip().lower() or "layered"
    allowed_info_types = {_normalize_info_type(item) for item in info_types}
    allowed_info_subsets = {_normalize_info_subset(item) for item in info_subsets if _normalize_info_subset(item)}
    allowed_announcement_tags = {
        _normalize_event_tag(item)
        for item in announcement_event_tags
        if _normalize_event_tag(item)
    }
    raw = _load_raw_info(path)
    if raw.empty:
        return []

    lower_map = _validate_columns(raw, source_label=str(path))
    as_of_cutoff = as_of_day_cutoff(as_of_date, cutoff_time=cutoff_time)
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

        title = str(row[lower_map["title"]]).strip() if "title" in lower_map else ""
        source_url = str(row[lower_map["source_url"]]).strip() if "source_url" in lower_map else ""
        publish_datetime = _resolve_publish_datetime(
            row,
            lower_map=lower_map,
            base_date=date,
            source_url=source_url,
        )
        if publish_datetime:
            publish_ts = parse_timestamp(publish_datetime)
            if publish_ts is not None and publish_ts > as_of_cutoff:
                continue
        publisher = str(row[lower_map["publisher"]]).strip() if "publisher" in lower_map else ""
        subset_from_path = _normalize_info_subset(row.get("__default_subset", ""))
        raw_info_type = row[lower_map["info_type"]] if "info_type" in lower_map else ""
        if str(raw_info_type).strip():
            info_type = _normalize_info_type(raw_info_type)
        elif normalized_source_mode == "layered" and subset_from_path:
            info_type = _SUBSET_DEFAULT_INFO_TYPE.get(subset_from_path, "news")
        else:
            info_type = _infer_info_type(
                title=title,
                source_url=source_url,
                publisher=publisher,
            )
        if info_type not in allowed_info_types:
            continue
        source_subset = subset_from_path or _subset_from_info_type(info_type)
        if allowed_info_subsets and source_subset not in allowed_info_subsets:
            continue
        strength = float(row[lower_map["strength"]]) if "strength" in lower_map else 3.0
        confidence = float(row[lower_map["confidence"]]) if "confidence" in lower_map else 0.7
        source_weight = (
            float(row[lower_map["source_weight"]])
            if "source_weight" in lower_map
            else float(_INFO_TYPE_WEIGHTS.get(info_type, 0.85))
        )
        raw_event_tag = row[lower_map["event_tag"]] if "event_tag" in lower_map else ""
        event_tag = _normalize_event_tag(raw_event_tag) if str(raw_event_tag).strip() else _infer_event_tag(
            title=title,
            direction=direction_raw,
            info_type=info_type,
        )
        if source_subset == "announcements" and event_tag not in allowed_announcement_tags:
            continue
        event_id = str(row[lower_map["event_id"]]).strip() if "event_id" in lower_map else ""
        item = InfoItem(
            date=str(date.date()),
            target_type=target_type,
            target=target,
            horizon=horizon,
            direction=direction_raw,
            info_type=info_type,
            title=title,
            source_subset=source_subset,
            source_url=source_url,
            strength=float(strength),
            confidence=float(confidence),
            source_weight=float(source_weight),
            publisher=publisher,
            event_tag=event_tag,
            event_id=event_id,
            publish_datetime=publish_datetime,
        )
        key = (
            item.date,
            item.target_type,
            item.target,
            item.horizon,
            item.info_type,
            item.source_subset,
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
            item.source_subset,
            item.direction,
            item.title,
        )
    )
    return items
