from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.domain.entities import Security

OTHER_LABELS = {
    "",
    "其他",
    "其它",
    "other",
    "unknown",
    "nan",
    "none",
    "null",
    "未分类",
}
GENERIC_INDUSTRY_LABELS = {
    "制造业",
    "工业",
    "化工",
    "有色金属",
    "基础化工",
    "电气设备",
    "机械设备",
    "电子",
    "通信",
}
THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("光模块", ("光模块", "cpo", "光通信", "光器件", "光纤", "光缆", "光迅", "中际", "新易盛", "天孚", "联特", "仕佳")),
    ("航天军工", ("航天", "航空", "军工", "导弹", "卫星", "舰船", "国防", "兵器", "军贸")),
    ("能源石油", ("石油", "油气", "天然气", "海油", "油服", "炼油", "petro", "lng")),
    ("煤化工", ("煤化工", "煤焦", "焦炭", "尿素", "甲醇", "纯碱", "氨纶", "氮肥", "磷化工", "煤制")),
    ("煤炭", ("煤炭", "焦煤", "动力煤", "煤业")),
    ("资源", ("有色", "黄金", "铜", "铝", "锌", "钼", "稀土", "矿业", "金属", "资源", "锂矿")),
    ("新能源电力", ("电力", "光伏", "风电", "储能", "锂电", "充电桩", "特高压", "绿电", "新能源")),
    ("半导体", ("半导体", "芯片", "集成电路", "晶圆", "封测", "功率器件", "存储")),
    ("通信设备", ("通信设备", "交换机", "算力", "服务器", "数据中心", "运营商", "基站")),
    ("科技软件", ("软件", "互联网", "云计算", "人工智能", "ai", "信息技术", "工业软件")),
    ("汽车", ("汽车", "汽配", "整车", "新能源车", "智能驾驶", "车联网")),
    ("医药", ("医药", "医疗", "生物", "制药", "创新药", "医疗器械")),
    ("消费", ("消费", "食品", "饮料", "白酒", "乳业", "零售", "调味品", "家电")),
    ("金融", ("证券", "银行", "保险", "多元金融", "信托", "期货")),
)
INDUSTRY_FALLBACKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("能源石油", ("石油石化", "油气开采", "油服工程")),
    ("煤炭", ("煤炭开采", "煤炭")),
    ("煤化工", ("化学原料", "化学制品", "煤化工", "化肥", "农化制品")),
    ("资源", ("有色金属", "工业金属", "贵金属", "小金属", "稀有金属", "采掘")),
    ("航天军工", ("国防军工", "航空装备", "航天装备", "地面兵装")),
    ("光模块", ("通信设备", "通信服务", "光学光电子")),
    ("半导体", ("半导体", "电子元件", "消费电子")),
    ("科技软件", ("计算机应用", "计算机设备", "软件开发", "互联网服务")),
    ("新能源电力", ("电力", "电网设备", "光伏设备", "风电设备", "电池")),
    ("汽车", ("汽车整车", "汽车零部件", "汽车服务")),
    ("医药", ("医疗器械", "生物制品", "化学制药", "中药")),
    ("消费", ("食品饮料", "家用电器", "商贸零售", "美容护理")),
    ("金融", ("银行", "证券", "保险", "多元金融")),
)
CACHE_PATH = Path("artifacts/metadata/tushare_stock_basic.json")
SYMBOL_CACHE_PATH = Path("artifacts/metadata/security_symbol_metadata.json")
CONCEPT_CACHE_PATH = Path("artifacts/metadata/tushare_symbol_concepts.json")
CACHE_MAX_AGE_SECONDS = 7 * 24 * 3600
GENERIC_CONCEPT_LABELS = {
    "融资融券",
    "融资标的股",
    "融券标的股",
    "转融券标的",
    "沪股通",
    "深股通",
    "年报预增",
    "msci概念",
    "证金持股",
    "中证500",
    "上证180",
    "沪深300",
}
CONCEPT_THEME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("光模块", ("光模块", "cpo", "光通信", "光纤", "光器件", "硅光")),
    ("航天军工", ("航天军工", "航空航天", "航天系", "卫星导航", "军民融合", "军工信息化", "雷达", "大飞机")),
    ("能源石油", ("石油", "油气", "天然气", "页岩气", "可燃冰", "油服", "海工装备", "lng")),
    ("煤化工", ("煤化工", "甲醇", "焦煤", "焦炭", "尿素", "纯碱", "煤制", "磷化工")),
    ("煤炭", ("煤炭概念", "动力煤", "煤炭")),
    ("资源", ("黄金概念", "有色金属", "稀土永磁", "小金属", "工业金属", "锂矿", "矿产资源")),
    ("新能源电力", ("光伏", "风电", "储能", "特高压", "新能源", "虚拟电厂")),
    ("半导体", ("集成电路概念", "芯片", "存储芯片", "第三代半导体", "半导体")),
    ("通信设备", ("5g", "通信设备", "交换机", "算力租赁", "数据中心", "东数西算")),
    ("科技软件", ("人工智能", "aigc", "云计算", "工业软件", "信创", "互联网")),
)


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_compare_text(value: object) -> str:
    return _normalize_text(value).lower().replace(" ", "")


def _is_placeholder_sector(value: object) -> bool:
    return _normalize_compare_text(value) in OTHER_LABELS


def _should_override_with_theme(current_sector: str, inferred_theme: str) -> bool:
    sector = _normalize_text(current_sector)
    if not sector:
        return True
    if _is_placeholder_sector(sector):
        return True
    return sector in GENERIC_INDUSTRY_LABELS and inferred_theme not in {"其他", sector}


def _theme_from_name_and_industry(*, symbol: str, name: str, industry: str) -> str:
    del symbol
    text = f"{_normalize_text(name)} {_normalize_text(industry)}".lower()
    for theme, keywords in THEME_RULES:
        if any(keyword.lower() in text for keyword in keywords):
            return theme
    normalized_industry = _normalize_text(industry)
    for theme, keywords in INDUSTRY_FALLBACKS:
        if any(keyword in normalized_industry for keyword in keywords):
            return theme
    return normalized_industry or "其他"


def _load_concept_cache() -> dict[str, list[str]]:
    if not CONCEPT_CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(CONCEPT_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[str]] = {}
    for symbol, concepts in payload.items():
        if not isinstance(concepts, list):
            continue
        out[str(symbol)] = [
            _normalize_text(item)
            for item in concepts
            if _normalize_text(item)
        ]
    return out


def _write_concept_cache(payload: dict[str, list[str]]) -> None:
    try:
        CONCEPT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONCEPT_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _fetch_tushare_symbol_concepts(symbol: str) -> list[str]:
    try:
        pro = _get_tushare_pro_api()
    except Exception:
        return []
    try:
        raw = pro.concept_detail(
            ts_code=_normalize_text(symbol),
            fields="concept_name,ts_code",
        )
    except Exception:
        return []
    if raw is None or raw.empty:
        return []
    out: list[str] = []
    for item in raw.get("concept_name", []).tolist():
        label = _normalize_text(item)
        if not label:
            continue
        if label in GENERIC_CONCEPT_LABELS:
            continue
        out.append(label)
    return list(dict.fromkeys(out))


def _load_symbol_concepts(symbols: Sequence[str]) -> dict[str, list[str]]:
    requested = sorted({_normalize_text(symbol) for symbol in symbols if _normalize_text(symbol)})
    if not requested:
        return {}
    cache = _load_concept_cache()
    missing = [symbol for symbol in requested if symbol not in cache]
    if missing:
        dirty = False
        for idx, symbol in enumerate(missing, start=1):
            cache[symbol] = _fetch_tushare_symbol_concepts(symbol)
            dirty = True
            if idx % 10 == 0:
                _write_concept_cache(cache)
                dirty = False
        if dirty:
            _write_concept_cache(cache)
    return {
        symbol: cache.get(symbol, [])
        for symbol in requested
    }


def _theme_from_concepts(concepts: Sequence[str]) -> str:
    normalized = [_normalize_text(item) for item in concepts if _normalize_text(item)]
    if not normalized:
        return ""
    joined = " ".join(normalized).lower()
    for theme, keywords in CONCEPT_THEME_RULES:
        if any(keyword.lower() in joined for keyword in keywords):
            return theme
    return ""


def _load_symbol_cache() -> dict[str, dict[str, str]]:
    if not SYMBOL_CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(SYMBOL_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(symbol): {
            "name": _normalize_text(meta.get("name")),
            "industry": _normalize_text(meta.get("industry")),
        }
        for symbol, meta in payload.items()
        if isinstance(meta, dict)
    }


def _write_symbol_cache(payload: dict[str, dict[str, str]]) -> None:
    try:
        SYMBOL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        SYMBOL_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _fetch_cninfo_symbol_metadata(symbol: str) -> dict[str, str]:
    normalized_symbol = _normalize_text(symbol)
    code = normalized_symbol.split(".")[0]
    if len(code) != 6:
        return {}
    try:
        import akshare as ak
    except Exception:
        return {}
    try:
        frame = ak.stock_industry_change_cninfo(
            symbol=code,
            start_date="20200101",
            end_date=pd.Timestamp.today().strftime("%Y%m%d"),
        )
    except Exception:
        return {}
    if frame is None or frame.empty:
        return {}
    latest = frame.sort_values("变更日期").iloc[-1]
    name = _normalize_text(latest.get("新证券简称"))
    industry = (
        _normalize_text(latest.get("行业大类"))
        or _normalize_text(latest.get("行业中类"))
        or _normalize_text(latest.get("行业次类"))
        or _normalize_text(latest.get("行业门类"))
    )
    if not name and not industry:
        return {}
    return {
        "name": name,
        "industry": industry,
    }


def _load_symbol_metadata_with_fallback(symbols: Sequence[str]) -> dict[str, dict[str, str]]:
    requested = sorted({_normalize_text(symbol) for symbol in symbols if _normalize_text(symbol)})
    if not requested:
        return {}
    cache = _load_symbol_cache()
    missing = [symbol for symbol in requested if symbol not in cache]
    if missing:
        updates: dict[str, dict[str, str]] = {}
        for symbol in missing:
            meta = _fetch_cninfo_symbol_metadata(symbol)
            if meta:
                updates[symbol] = meta
        if updates:
            cache.update(updates)
            _write_symbol_cache(cache)
    return {
        symbol: cache[symbol]
        for symbol in requested
        if symbol in cache
    }


@lru_cache(maxsize=1)
def _load_tushare_stock_basic() -> dict[str, dict[str, str]]:
    if CACHE_PATH.exists():
        age_seconds = max(0.0, float(pd.Timestamp.now().timestamp() - CACHE_PATH.stat().st_mtime))
        if age_seconds <= float(CACHE_MAX_AGE_SECONDS):
            try:
                payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict) and payload:
                return {
                    str(symbol): {
                        "name": _normalize_text(meta.get("name")),
                        "industry": _normalize_text(meta.get("industry")),
                    }
                    for symbol, meta in payload.items()
                    if isinstance(meta, dict)
                }
    try:
        pro = _get_tushare_pro_api()
    except Exception:
        return {}
    try:
        raw = pro.stock_basic(exchange="", list_status="L", fields="ts_code,name,industry")
    except Exception:
        if CACHE_PATH.exists():
            try:
                payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                return {
                    str(symbol): {
                        "name": _normalize_text(meta.get("name")),
                        "industry": _normalize_text(meta.get("industry")),
                    }
                    for symbol, meta in payload.items()
                    if isinstance(meta, dict)
                }
        return {}
    if raw is None or raw.empty:
        return {}
    out: dict[str, dict[str, str]] = {}
    for _, row in raw.iterrows():
        symbol = _normalize_text(row.get("ts_code"))
        if not symbol:
            continue
        out[symbol] = {
            "name": _normalize_text(row.get("name")),
            "industry": _normalize_text(row.get("industry")),
        }
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    return out


@lru_cache(maxsize=1)
def _get_tushare_pro_api():
    token = _normalize_text(os.getenv("TUSHARE_TOKEN"))
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is not configured")
    import tushare as ts

    return ts.pro_api(token)


def enrich_securities_with_metadata(rows: Sequence[Security]) -> list[Security]:
    if not rows:
        return []
    metadata_map = _load_tushare_stock_basic()
    if not metadata_map:
        metadata_map = _load_symbol_metadata_with_fallback([row.symbol for row in rows])
    else:
        missing_symbols = [row.symbol for row in rows if str(row.symbol) not in metadata_map]
        if missing_symbols:
            metadata_map = {
                **_load_symbol_metadata_with_fallback(missing_symbols),
                **metadata_map,
            }
    if not metadata_map:
        return list(rows)
    concept_map = _load_symbol_concepts([row.symbol for row in rows])
    out: list[Security] = []
    for row in rows:
        metadata = metadata_map.get(str(row.symbol), {})
        name = _normalize_text(metadata.get("name")) or _normalize_text(row.name) or str(row.symbol)
        industry = _normalize_text(metadata.get("industry"))
        concepts = concept_map.get(str(row.symbol), [])
        current_sector = _normalize_text(row.sector)
        inferred_theme = _theme_from_concepts(concepts) or _theme_from_name_and_industry(
            symbol=str(row.symbol),
            name=name,
            industry=industry,
        )
        sector = inferred_theme if _should_override_with_theme(current_sector, inferred_theme) else current_sector
        out.append(Security(symbol=str(row.symbol), name=name, sector=sector or "其他"))
    return out


def placeholder_sector_ratio(rows: Iterable[Security]) -> float:
    rows = list(rows)
    if not rows:
        return 0.0
    return float(sum(1 for row in rows if _is_placeholder_sector(row.sector)) / max(1, len(rows)))
