from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from src.application.watchlist import load_watchlist
from src.domain.symbols import SymbolError, normalize_symbol
from src.infrastructure.discovery import _load_universe_file
from src.infrastructure.info_repository import _infer_event_tag

EASTMONEY_NOTICE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"
INFO_COLUMNS = [
    "date",
    "target_type",
    "target",
    "horizon",
    "direction",
    "info_type",
    "title",
    "source_url",
    "publisher",
    "strength",
    "confidence",
    "source_weight",
    "event_tag",
]
POSITIVE_STRONG = (
    "预增",
    "大幅增长",
    "扭亏",
    "回购",
    "分红",
    "中标",
    "签订",
    "签约",
    "获批",
    "增持",
    "上调",
)
POSITIVE_WEAK = (
    "增长",
    "盈利",
    "进展",
    "完成",
    "投产",
    "提升",
    "利好",
    "回暖",
)
NEGATIVE_STRONG = (
    "预亏",
    "亏损",
    "立案",
    "处罚",
    "退市",
    "减持",
    "违约",
    "终止",
    "下修",
    "失败",
)
NEGATIVE_WEAK = (
    "问询",
    "波动",
    "风险提示",
    "冻结",
    "诉讼",
    "质押",
    "延期",
    "异常",
    "不确定",
)
SHORT_HORIZON_HINTS = (
    "交易异常波动",
    "风险提示",
    "问询",
    "监管",
    "停牌",
    "复牌",
)
RESEARCH_POSITIVE = (
    "买入",
    "增持",
    "推荐",
    "强推",
    "跑赢行业",
    "优于大市",
    "outperform",
    "overweight",
    "buy",
)
RESEARCH_NEGATIVE = (
    "卖出",
    "减持",
    "回避",
    "弱于大市",
    "underperform",
    "sell",
)


@dataclass(frozen=True)
class InfoSyncResult:
    out_dir: str
    symbol_count: int
    market_news_rows: int
    announcement_rows: int
    research_rows: int
    notes: list[str]


def _clean_text(value: object) -> str:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _score_text_direction(text: str) -> tuple[str, int, float]:
    score = 0
    for keyword in POSITIVE_STRONG:
        if keyword in text:
            score += 2
    for keyword in POSITIVE_WEAK:
        if keyword in text:
            score += 1
    for keyword in NEGATIVE_STRONG:
        if keyword in text:
            score -= 2
    for keyword in NEGATIVE_WEAK:
        if keyword in text:
            score -= 1
    if score >= 2:
        return "bullish", 4, 0.82
    if score == 1:
        return "bullish", 3, 0.74
    if score <= -2:
        return "bearish", 4, 0.82
    if score == -1:
        return "bearish", 3, 0.74
    return "neutral", 2, 0.62


def _score_research_direction(title: str, rating: str, report_type: str) -> tuple[str, int, float]:
    text = " ".join([title, rating, report_type]).lower()
    if any(token.lower() in text for token in RESEARCH_POSITIVE):
        return "bullish", 3, 0.78
    if any(token.lower() in text for token in RESEARCH_NEGATIVE):
        return "bearish", 3, 0.78
    return _score_text_direction(" ".join([title, rating, report_type]))


def _horizon_for_text(text: str, *, default: str = "mid") -> str:
    return "short" if any(keyword in text for keyword in SHORT_HORIZON_HINTS) else default


def _normalize_symbols(
    *,
    symbols: Sequence[str] | None,
    watchlist_path: str,
    universe_file: str,
    universe_limit: int,
) -> list[str]:
    resolved: list[str] = []
    for token in symbols or ():
        try:
            resolved.append(normalize_symbol(token).symbol)
        except SymbolError:
            continue
    if resolved:
        return sorted(set(resolved))
    if str(universe_file).strip():
        rows = _load_universe_file(universe_file, enrich_metadata=False)
        if int(universe_limit or 0) > 0:
            rows = rows[: int(universe_limit)]
        for item in rows:
            try:
                resolved.append(normalize_symbol(item.symbol).symbol)
            except SymbolError:
                continue
    if resolved:
        return sorted(set(resolved))
    _, stocks, _ = load_watchlist(watchlist_path)
    for item in stocks:
        try:
            resolved.append(normalize_symbol(item.symbol).symbol)
        except SymbolError:
            continue
    return sorted(set(resolved))


def _tushare_client(token: str | None):
    import tushare as ts

    resolved = str(token or os.getenv("TUSHARE_TOKEN", "")).strip()
    if not resolved:
        raise RuntimeError("tushare token missing")
    return ts.pro_api(resolved)


def _pick_first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower_map = {str(col).lower(): str(col) for col in frame.columns}
    for name in candidates:
        key = str(name).lower()
        if key in lower_map:
            return lower_map[key]
    return ""


def _fetch_tushare_market_news(
    *,
    start: str,
    end: str,
    token: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    try:
        pro = _tushare_client(token)
        raw = pro.major_news(start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"market_news fetch skipped: {exc}")
        return pd.DataFrame(columns=INFO_COLUMNS), notes
    if raw is None or raw.empty:
        notes.append("market_news fetch returned 0 rows")
        return pd.DataFrame(columns=INFO_COLUMNS), notes
    date_col = _pick_first_column(raw, ("pub_time", "pubdate", "date", "datetime"))
    title_col = _pick_first_column(raw, ("title", "content", "summary"))
    publisher_col = _pick_first_column(raw, ("src", "source", "media", "publisher"))
    if not date_col or not title_col:
        notes.append("market_news fetch returned unsupported columns")
        return pd.DataFrame(columns=INFO_COLUMNS), notes
    records: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        dt = pd.to_datetime(row.get(date_col), errors="coerce")
        title = _clean_text(row.get(title_col))
        if pd.isna(dt) or not title:
            continue
        direction, strength, confidence = _score_text_direction(title)
        records.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "target_type": "market",
                "target": "MARKET",
                "horizon": _horizon_for_text(title, default="short"),
                "direction": direction,
                "info_type": "news",
                "title": title,
                "source_url": "",
                "publisher": _clean_text(row.get(publisher_col)) if publisher_col else "tushare",
                "strength": int(strength),
                "confidence": float(confidence),
                "source_weight": 0.85,
                "event_tag": _infer_event_tag(title=title, direction=direction, info_type="news"),
            }
        )
    frame = pd.DataFrame(records, columns=INFO_COLUMNS)
    return _dedupe_frame(frame, subset=("date", "target", "title", "publisher")), notes


def _fetch_tushare_research(
    *,
    start: str,
    end: str,
    symbols: Sequence[str],
    token: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    try:
        pro = _tushare_client(token)
        raw = pro.report_rc(start_date=start.replace("-", ""), end_date=end.replace("-", ""))
    except Exception as exc:  # noqa: BLE001
        notes.append(f"research fetch skipped: {exc}")
        return pd.DataFrame(columns=INFO_COLUMNS), notes
    if raw is None or raw.empty:
        notes.append("research fetch returned 0 rows")
        return pd.DataFrame(columns=INFO_COLUMNS), notes
    frame = raw.copy()
    if symbols:
        frame = frame[frame["ts_code"].astype(str).isin(set(symbols))].copy()
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        title = _clean_text(row.get("report_title"))
        symbol = str(row.get("ts_code", "")).strip()
        if not title or not symbol:
            continue
        dt = pd.to_datetime(row.get("report_date"), errors="coerce")
        if pd.isna(dt):
            continue
        rating = _clean_text(row.get("rating"))
        report_type = _clean_text(row.get("report_type"))
        direction, strength, confidence = _score_research_direction(title, rating, report_type)
        records.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "target_type": "stock",
                "target": symbol,
                "horizon": "mid",
                "direction": direction,
                "info_type": "research",
                "title": title,
                "source_url": "",
                "publisher": _clean_text(row.get("org_name")) or "tushare",
                "strength": int(strength),
                "confidence": float(confidence),
                "source_weight": 0.70,
                "event_tag": "",
            }
        )
    return _dedupe_frame(pd.DataFrame(records, columns=INFO_COLUMNS), subset=("date", "target", "title", "publisher")), notes


def _safe_notice_request(params: dict[str, str], *, timeout: float, max_retries: int) -> dict[str, object]:
    last_error: Exception | None = None
    for _ in range(max(1, int(max_retries))):
        try:
            response = requests.get(EASTMONEY_NOTICE_URL, params=params, timeout=float(timeout))
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                return payload
            raise RuntimeError("invalid eastmoney payload")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.8)
    raise RuntimeError(f"eastmoney notice request failed: {params}") from last_error


def _extract_notice_symbol(item: dict[str, object], fallback_symbol: str) -> str:
    codes = item.get("codes", [])
    if isinstance(codes, list):
        for code in codes:
            stock_code = str(getattr(code, "get", lambda *_: "")("stock_code", "")).strip()
            if stock_code.isdigit() and len(stock_code) == 6:
                try:
                    return normalize_symbol(stock_code).symbol
                except SymbolError:
                    continue
    return fallback_symbol


def _fetch_eastmoney_announcements(
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    sleep_ms: int,
    max_retries: int,
    timeout: float,
) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    if not symbols:
        return pd.DataFrame(columns=INFO_COLUMNS), ["announcement fetch skipped: no symbols resolved"]
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    records: list[dict[str, object]] = []
    total = len(symbols)
    for idx, symbol in enumerate(symbols, start=1):
        code = symbol.split(".")[0]
        params = {
            "sr": "-1",
            "page_size": "100",
            "page_index": "1",
            "ann_type": "A",
            "client_source": "web",
            "f_node": "0",
            "s_node": "0",
            "begin_time": str(start_ts.date()),
            "end_time": str(end_ts.date()),
            "stock_list": code,
        }
        try:
            payload = _safe_notice_request(params, timeout=timeout, max_retries=max_retries)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{symbol} announcement fetch failed: {exc}")
            continue
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        total_hits = int(data.get("total_hits", 0) or 0)
        total_pages = max(1, int(math.ceil(total_hits / 100.0))) if total_hits > 0 else 0
        if total_pages <= 0:
            continue
        for page in range(1, total_pages + 1):
            if page > 1:
                page_params = dict(params)
                page_params["page_index"] = str(page)
                try:
                    payload = _safe_notice_request(page_params, timeout=timeout, max_retries=max_retries)
                except Exception as exc:  # noqa: BLE001
                    notes.append(f"{symbol} announcement page {page} failed: {exc}")
                    continue
            rows = payload.get("data", {}).get("list", [])
            for item in rows:
                title = _clean_text(item.get("title"))
                if not title:
                    continue
                notice_date = pd.to_datetime(item.get("notice_date"), errors="coerce")
                if pd.isna(notice_date):
                    continue
                notice_ts = pd.Timestamp(notice_date).normalize()
                if notice_ts < start_ts or notice_ts > end_ts:
                    continue
                direction, strength, confidence = _score_text_direction(title)
                normalized_symbol = _extract_notice_symbol(item, symbol)
                art_code = str(item.get("art_code", "")).strip()
                source_url = f"https://data.eastmoney.com/notices/detail/{code}/{art_code}.html" if art_code else ""
                records.append(
                    {
                        "date": str(notice_ts.date()),
                        "target_type": "stock",
                        "target": normalized_symbol,
                        "horizon": _horizon_for_text(title),
                        "direction": direction,
                        "info_type": "announcement",
                        "title": title,
                        "source_url": source_url,
                        "publisher": "eastmoney_notice",
                        "strength": int(strength),
                        "confidence": float(confidence),
                        "source_weight": 1.0,
                        "event_tag": _infer_event_tag(title=title, direction=direction, info_type="announcement"),
                    }
                )
            if sleep_ms > 0:
                time.sleep(float(sleep_ms) / 1000.0)
        print(f"[INFO] announcements {idx}/{total}: {symbol} rows={len(records)}")
    frame = pd.DataFrame(records, columns=INFO_COLUMNS)
    return _dedupe_frame(frame, subset=("date", "target", "title", "source_url")), notes


def _dedupe_frame(frame: pd.DataFrame, *, subset: Sequence[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=INFO_COLUMNS)
    normalized = frame.copy()
    for column in INFO_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = ""
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized = normalized.dropna(subset=["date"]).copy()
    normalized = normalized.drop_duplicates(subset=list(subset), keep="last")
    normalized = normalized.sort_values(["date", "target_type", "target", "title"]).reset_index(drop=True)
    return normalized[INFO_COLUMNS]


def _merge_into_partition(path: Path, frame: pd.DataFrame, *, subset: Sequence[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = pd.DataFrame(columns=INFO_COLUMNS)
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path)
        except Exception:
            existing = pd.DataFrame(columns=INFO_COLUMNS)
    merged = pd.concat([existing, frame], ignore_index=True)
    merged = _dedupe_frame(merged, subset=subset)
    merged.to_csv(path, index=False, encoding="utf-8")
    return int(len(merged))


def sync_info_data(
    *,
    out_dir: str,
    start: str,
    end: str,
    watchlist: str,
    universe_file: str = "",
    universe_limit: int = 0,
    symbols: Sequence[str] | None = None,
    tushare_token: str | None = None,
    sleep_ms: int = 120,
    max_retries: int = 3,
    timeout: float = 20.0,
) -> InfoSyncResult:
    resolved_symbols = _normalize_symbols(
        symbols=symbols,
        watchlist_path=watchlist,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    out_root = Path(out_dir)
    notes: list[str] = []
    market_news_frame, market_notes = _fetch_tushare_market_news(start=start, end=end, token=tushare_token)
    research_frame, research_notes = _fetch_tushare_research(
        start=start,
        end=end,
        symbols=resolved_symbols,
        token=tushare_token,
    )
    announcement_frame, announcement_notes = _fetch_eastmoney_announcements(
        symbols=resolved_symbols,
        start=start,
        end=end,
        sleep_ms=sleep_ms,
        max_retries=max_retries,
        timeout=timeout,
    )
    notes.extend(market_notes)
    notes.extend(research_notes)
    notes.extend(announcement_notes)

    market_news_rows = _merge_into_partition(
        out_root / "market_news" / "core.csv",
        market_news_frame,
        subset=("date", "target", "title", "publisher"),
    )
    research_rows = _merge_into_partition(
        out_root / "research" / "core.csv",
        research_frame,
        subset=("date", "target", "title", "publisher"),
    )
    announcement_rows = _merge_into_partition(
        out_root / "announcements" / "core.csv",
        announcement_frame,
        subset=("date", "target", "title", "source_url"),
    )
    return InfoSyncResult(
        out_dir=str(out_root.resolve()),
        symbol_count=int(len(resolved_symbols)),
        market_news_rows=market_news_rows,
        announcement_rows=announcement_rows,
        research_rows=research_rows,
        notes=notes,
    )
