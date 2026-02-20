#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.watchlist import load_watchlist
from src.domain.symbols import SymbolError, normalize_symbol

EASTMONEY_NOTICE_URL = "https://np-anotice-stock.eastmoney.com/api/security/ann"

REQUIRED_COLUMNS = [
    "date",
    "target_type",
    "target",
    "horizon",
    "direction",
    "strength",
    "confidence",
    "source_weight",
    "title",
]

POSITIVE_STRONG = [
    "预增",
    "大幅增长",
    "扭亏为盈",
    "回购",
    "分红",
    "中标",
    "签署",
    "签订",
    "获批",
    "摘帽",
    "增持",
]
POSITIVE_WEAK = [
    "增长",
    "盈利",
    "进展",
    "完成",
    "投产",
    "提升",
    "利好",
]
NEGATIVE_STRONG = [
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
]
NEGATIVE_WEAK = [
    "问询",
    "波动",
    "风险提示",
    "冻结",
    "诉讼",
    "质押",
    "延期",
    "异常",
    "不确定",
]
SHORT_HORIZON_HINTS = [
    "交易异常波动",
    "风险提示",
    "问询",
    "监管",
    "停牌",
    "复牌",
]


@dataclass(frozen=True)
class NoticeRow:
    date: str
    symbol: str
    title: str
    source_url: str


def _parse_args() -> argparse.Namespace:
    today = pd.Timestamp.today().date()
    parser = argparse.ArgumentParser(
        description="Collect Eastmoney stock notices and write partitioned news CSV by symbol/year."
    )
    parser.add_argument("--watchlist", default="config/watchlist.json", help="Watchlist JSON path")
    parser.add_argument("--symbols", default="", help="Comma-separated stock symbols, e.g. 600160.SH,000630.SZ")
    parser.add_argument("--start-year", type=int, default=2022, help="Start year (inclusive)")
    parser.add_argument("--end-year", type=int, default=today.year, help="End year (inclusive)")
    parser.add_argument("--out-dir", default="input/news_parts", help="Output partition root directory")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument("--sleep-ms", type=int, default=120, help="Sleep milliseconds between page requests")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per HTTP request")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite yearly files instead of merge-dedup append")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and report stats without writing files")
    return parser.parse_args()


def _safe_request(params: dict[str, str], *, timeout: float, max_retries: int) -> dict:
    last_err: Exception | None = None
    for _ in range(max(1, int(max_retries))):
        try:
            resp = requests.get(EASTMONEY_NOTICE_URL, params=params, timeout=float(timeout))
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, dict):
                raise RuntimeError("Invalid JSON payload")
            return payload
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.8)
    raise RuntimeError(f"request failed after retries: {params}") from last_err


def _extract_code(item: dict, default_code: str) -> str:
    codes = item.get("codes", [])
    for c in codes:
        stock_code = str(c.get("stock_code", "")).strip()
        if stock_code.isdigit() and len(stock_code) == 6:
            return stock_code
    return default_code


def _normalize_symbol_from_code(code: str) -> str:
    return normalize_symbol(str(code)).symbol


def _iter_notices_for_symbol_year(
    code: str,
    year: int,
    *,
    timeout: float,
    sleep_ms: int,
    max_retries: int,
) -> Iterable[NoticeRow]:
    begin_time = f"{year}-01-01"
    end_time = f"{year}-12-31"
    base_params = {
        "sr": "-1",
        "page_size": "100",
        "page_index": "1",
        "ann_type": "A",
        "client_source": "web",
        "f_node": "0",
        "s_node": "0",
        "begin_time": begin_time,
        "end_time": end_time,
        "stock_list": code,
    }

    first_payload = _safe_request(base_params, timeout=timeout, max_retries=max_retries)
    data = first_payload.get("data", {}) if isinstance(first_payload, dict) else {}
    total_hits = int(data.get("total_hits", 0) or 0)
    if total_hits <= 0:
        return []
    total_pages = max(1, int(math.ceil(total_hits / 100.0)))
    out: list[NoticeRow] = []

    def parse_payload(payload: dict) -> None:
        rows = payload.get("data", {}).get("list", [])
        for item in rows:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            raw_code = _extract_code(item, default_code=code)
            try:
                symbol = _normalize_symbol_from_code(raw_code)
            except SymbolError:
                continue
            notice_date = pd.to_datetime(item.get("notice_date"), errors="coerce")
            if pd.isna(notice_date):
                continue
            art_code = str(item.get("art_code", "")).strip()
            source_url = f"https://data.eastmoney.com/notices/detail/{raw_code}/{art_code}.html" if art_code else ""
            out.append(
                NoticeRow(
                    date=str(pd.Timestamp(notice_date).date()),
                    symbol=symbol,
                    title=title,
                    source_url=source_url,
                )
            )

    parse_payload(first_payload)
    if total_pages > 1:
        for page in range(2, total_pages + 1):
            params = dict(base_params)
            params["page_index"] = str(page)
            payload = _safe_request(params, timeout=timeout, max_retries=max_retries)
            parse_payload(payload)
            if sleep_ms > 0:
                time.sleep(float(sleep_ms) / 1000.0)
    return out


def _score_title(title: str) -> tuple[str, str, int, float]:
    text = str(title)
    score = 0
    for kw in POSITIVE_STRONG:
        if kw in text:
            score += 2
    for kw in POSITIVE_WEAK:
        if kw in text:
            score += 1
    for kw in NEGATIVE_STRONG:
        if kw in text:
            score -= 2
    for kw in NEGATIVE_WEAK:
        if kw in text:
            score -= 1

    if score >= 2:
        direction = "bullish"
        strength = 4
        confidence = 0.82
    elif score == 1:
        direction = "bullish"
        strength = 3
        confidence = 0.74
    elif score <= -2:
        direction = "bearish"
        strength = 4
        confidence = 0.82
    elif score == -1:
        direction = "bearish"
        strength = 3
        confidence = 0.74
    else:
        direction = "neutral"
        strength = 2
        confidence = 0.62

    horizon = "short" if any(kw in text for kw in SHORT_HORIZON_HINTS) else "mid"
    return direction, horizon, int(strength), float(confidence)


def _to_news_frame(rows: Iterable[NoticeRow]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in rows:
        direction, horizon, strength, confidence = _score_title(row.title)
        records.append(
            {
                "date": row.date,
                "target_type": "stock",
                "target": row.symbol,
                "horizon": horizon,
                "direction": direction,
                "strength": strength,
                "confidence": confidence,
                "source_weight": 1.0,
                "title": row.title,
                "source_url": row.source_url,
            }
        )
    if not records:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_url"])
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"]).copy()
    df = df.drop_duplicates(subset=["date", "target", "title"]).sort_values(["date", "target", "title"])
    return df.reset_index(drop=True)


def _read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_url"])
    try:
        existing = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=REQUIRED_COLUMNS + ["source_url"])
    for col in REQUIRED_COLUMNS + ["source_url"]:
        if col not in existing.columns:
            existing[col] = ""
    return existing


def _resolve_symbols(args: argparse.Namespace) -> list[str]:
    if str(args.symbols).strip():
        out: list[str] = []
        for token in str(args.symbols).replace(";", ",").replace("\n", ",").split(","):
            code = token.strip()
            if not code:
                continue
            try:
                out.append(normalize_symbol(code).symbol)
            except SymbolError:
                continue
        uniq = sorted(set(out))
        if uniq:
            return uniq
    _, stocks, _ = load_watchlist(args.watchlist)
    return sorted({normalize_symbol(s.symbol).symbol for s in stocks})


def main() -> int:
    args = _parse_args()
    start_year = int(args.start_year)
    end_year = int(args.end_year)
    if end_year < start_year:
        raise ValueError(f"Invalid year range: {start_year}..{end_year}")
    symbols = _resolve_symbols(args)
    if not symbols:
        raise ValueError("No symbols found. Provide --symbols or a valid --watchlist.")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    total_fetched = 0
    total_written = 0
    for symbol in symbols:
        code = normalize_symbol(symbol).code
        for year in range(start_year, end_year + 1):
            notices = list(
                _iter_notices_for_symbol_year(
                    code=code,
                    year=year,
                    timeout=float(args.timeout),
                    sleep_ms=int(args.sleep_ms),
                    max_retries=int(args.max_retries),
                )
            )
            fetched = len(notices)
            total_fetched += fetched
            if fetched <= 0:
                print(f"[SKIP] {symbol} {year}: no notices")
                continue

            new_df = _to_news_frame(notices)
            out_file = out_root / symbol / f"{year}.csv"
            out_file.parent.mkdir(parents=True, exist_ok=True)

            if args.overwrite:
                merged = new_df
            else:
                old_df = _read_existing_csv(out_file)
                if old_df.empty:
                    merged = new_df.copy()
                elif new_df.empty:
                    merged = old_df.copy()
                else:
                    merged = pd.concat([old_df, new_df], ignore_index=True)
                merged = merged.drop_duplicates(subset=["date", "target", "title"]).sort_values(
                    ["date", "target", "title"]
                )

            for col in REQUIRED_COLUMNS:
                if col not in merged.columns:
                    merged[col] = ""
            write_cols = REQUIRED_COLUMNS + (["source_url"] if "source_url" in merged.columns else [])
            merged = merged[write_cols].reset_index(drop=True)

            if args.dry_run:
                print(f"[DRY] {symbol} {year}: fetched={fetched}, merged_rows={len(merged)}")
                continue

            merged.to_csv(out_file, index=False, encoding="utf-8")
            total_written += int(len(merged))
            print(f"[OK] {symbol} {year}: fetched={fetched}, written_rows={len(merged)} -> {out_file}")

    print(f"[DONE] symbols={len(symbols)}, years={end_year - start_year + 1}, fetched_rows={total_fetched}")
    if not args.dry_run:
        print(f"[DONE] total_rows_written={total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
