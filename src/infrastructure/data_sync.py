from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import requests

from src.domain.entities import Security
from src.domain.symbols import SymbolError, normalize_symbol
from src.infrastructure.market_data import DataError, load_symbol_daily


EASTMONEY_CLIST_URL = "https://push2.eastmoney.com/api/qt/clist/get"
INDEX_SECURITIES: tuple[Security, ...] = (
    Security(symbol="000300.SH", name="沪深300", sector="指数"),
    Security(symbol="000001.SH", name="上证指数", sector="指数"),
    Security(symbol="399001.SZ", name="深证成指", sector="指数"),
    Security(symbol="399006.SZ", name="创业板指", sector="指数"),
)
_SYMBOL_FILE_PATTERN = re.compile(r"^(\d{6}\.(SH|SZ))\.csv$", re.IGNORECASE)
_INDEX_SYMBOL_SET = {x.symbol for x in INDEX_SECURITIES} | {"000905.SH", "000852.SH"}


@dataclass(frozen=True)
class DataSyncResult:
    requested_universe_size: int
    universe_size: int
    attempted: int
    downloaded: int
    skipped: int
    failed: int
    failed_symbols: list[str]
    universe_source: str
    universe_file: str


def _safe_symbol(value: str) -> str | None:
    try:
        return normalize_symbol(value).symbol
    except SymbolError:
        return None


def _normalize_exchange_code(code: str, market_id: int | None) -> str | None:
    text = str(code).strip()
    if not text.isdigit() or len(text) != 6:
        return None
    if market_id == 1:
        return f"{text}.SH"
    if market_id == 0:
        return f"{text}.SZ"
    symbol = _safe_symbol(text)
    return symbol


def _is_a_share(symbol: str) -> bool:
    try:
        info = normalize_symbol(symbol)
    except SymbolError:
        return False
    if info.exchange == "SH" and info.code.startswith("900"):
        return False
    if info.exchange == "SZ" and info.code.startswith("200"):
        return False
    return True


def _dedupe_rows(rows: Sequence[Security]) -> list[Security]:
    seen: set[str] = set()
    out: list[Security] = []
    for row in rows:
        symbol = _safe_symbol(row.symbol)
        if symbol is None or symbol in seen:
            continue
        seen.add(symbol)
        out.append(Security(symbol=symbol, name=row.name or symbol, sector=row.sector or "其他"))
    return out


def fetch_eastmoney_universe(limit: int, *, min_amount: float = 5e7, exclude_st: bool = True) -> list[Security]:
    limit = max(1, int(limit))
    out: list[Security] = []
    page = 1
    page_size = 200
    while len(out) < limit:
        params = {
            "pn": str(page),
            "pz": str(page_size),
            "po": "1",
            "np": "1",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
            "fields": "f12,f13,f14,f2,f3,f6",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        }
        resp = requests.get(EASTMONEY_CLIST_URL, params=params, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        rows = data.get("diff") if isinstance(data, dict) else None
        if not rows:
            break
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = _normalize_exchange_code(item.get("f12", ""), item.get("f13", None))
            if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
                continue
            name = str(item.get("f14", "")).strip() or symbol
            if exclude_st and "ST" in name.upper():
                continue
            amount = pd.to_numeric(item.get("f6", 0.0), errors="coerce")
            if pd.notna(amount) and float(amount) < float(min_amount):
                continue
            out.append(Security(symbol=symbol, name=name, sector="其他"))
            if len(out) >= limit:
                break
        page += 1
    return _dedupe_rows(out)[:limit]


def _load_universe_file(path: str | Path) -> list[Security]:
    file_path = Path(path)
    if not str(path).strip() or not file_path.exists():
        return []
    if file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows: list[dict[str, object]]
        if isinstance(payload, list):
            rows = [x for x in payload if isinstance(x, dict)]
        elif isinstance(payload, dict):
            if isinstance(payload.get("stocks"), list):
                rows = [x for x in payload["stocks"] if isinstance(x, dict)]
            else:
                rows = []
                if isinstance(payload.get("market_index"), dict):
                    rows.append(payload["market_index"])
        else:
            rows = []
        out: list[Security] = []
        for row in rows:
            symbol = _safe_symbol(str(row.get("symbol", "")))
            if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
                continue
            out.append(
                Security(
                    symbol=symbol,
                    name=str(row.get("name", symbol)),
                    sector=str(row.get("sector", "其他")),
                )
            )
        return _dedupe_rows(out)

    raw = pd.read_csv(file_path)
    if raw.empty:
        return []
    lower = {c.lower(): c for c in raw.columns}
    symbol_col = lower.get("symbol")
    if symbol_col is None:
        raise ValueError(f"Universe file missing column `symbol`: {file_path}")
    name_col = lower.get("name")
    sector_col = lower.get("sector")
    out: list[Security] = []
    for _, row in raw.iterrows():
        symbol = _safe_symbol(str(row[symbol_col]))
        if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
            continue
        out.append(
            Security(
                symbol=symbol,
                name=str(row[name_col]) if name_col is not None else symbol,
                sector=str(row[sector_col]) if sector_col is not None else "其他",
            )
        )
    return _dedupe_rows(out)


def _prepare_universe(
    *,
    universe_size: int,
    universe_file: str,
    data_dir: str,
) -> tuple[list[Security], str]:
    if str(universe_file).strip():
        rows = _load_universe_file(universe_file)
        if rows:
            return rows[: max(1, int(universe_size))], f"file:{universe_file}"

    last_error = ""
    try:
        rows = fetch_eastmoney_universe(limit=max(1, int(universe_size)))
        if rows:
            return rows, "eastmoney"
    except Exception as exc:
        last_error = str(exc)

    local_rows: list[Security] = []
    for path in sorted(Path(data_dir).glob("*.csv")):
        matched = _SYMBOL_FILE_PATTERN.match(path.name)
        if not matched:
            continue
        symbol = _safe_symbol(matched.group(1))
        if symbol is None or not _is_a_share(symbol) or symbol in _INDEX_SYMBOL_SET:
            continue
        local_rows.append(Security(symbol=symbol, name=symbol, sector="其他"))
        if len(local_rows) >= max(1, int(universe_size)):
            break
    local_rows = _dedupe_rows(local_rows)
    if local_rows:
        return local_rows, "data_dir_fallback"
    if last_error:
        raise RuntimeError(f"failed to fetch universe from eastmoney: {last_error}")
    raise RuntimeError("failed to build universe: empty result")


def _write_universe_file(path: str | Path, rows: Sequence[Security], source: str) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "source": source,
        "stocks": [{"symbol": x.symbol, "name": x.name, "sector": x.sector} for x in rows],
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(target)


def _is_fresh_enough(path: Path, target_end: pd.Timestamp) -> bool:
    if not path.exists():
        return False
    try:
        raw = pd.read_csv(path, usecols=["date"])
    except Exception:
        return False
    if raw.empty:
        return False
    max_date = pd.to_datetime(raw["date"], errors="coerce").dropna()
    if max_date.empty:
        return False
    latest = pd.Timestamp(max_date.max()).normalize()
    return latest >= (target_end - pd.Timedelta(days=5))


def sync_market_data(
    *,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    universe_size: int,
    universe_file: str = "",
    include_indices: bool = True,
    force_refresh: bool = False,
    sleep_ms: int = 80,
    max_failures: int = 100,
    write_universe_file: str = "",
) -> DataSyncResult:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    rows, universe_source = _prepare_universe(
        universe_size=universe_size,
        universe_file=universe_file,
        data_dir=data_dir,
    )
    rows = rows[: max(1, int(universe_size))]
    if include_indices:
        rows = _dedupe_rows(list(rows) + list(INDEX_SECURITIES))
    if not rows:
        raise ValueError("No symbols found for sync.")

    today = pd.Timestamp.today().normalize()
    end_ts = pd.Timestamp(end).normalize()
    target_end = min(today, end_ts)

    downloaded = 0
    skipped = 0
    failed = 0
    attempts = 0
    failed_symbols: list[str] = []
    for sec in rows:
        symbol = normalize_symbol(sec.symbol).symbol
        path = Path(data_dir) / f"{symbol}.csv"
        if not force_refresh and _is_fresh_enough(path, target_end=target_end):
            skipped += 1
            continue
        attempts += 1
        try:
            df = load_symbol_daily(
                symbol=symbol,
                source=source,
                data_dir=data_dir,
                start=start,
                end=end,
            )
            if df.empty:
                raise DataError(f"{symbol}: empty dataframe after load")
            df.to_csv(path, index=False)
            downloaded += 1
        except Exception:
            failed += 1
            failed_symbols.append(symbol)
            if failed >= max(1, int(max_failures)):
                break
        if int(sleep_ms) > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    universe_path = ""
    if str(write_universe_file).strip():
        universe_rows = [x for x in rows if x.sector != "指数"]
        universe_path = _write_universe_file(write_universe_file, universe_rows, source=universe_source)

    return DataSyncResult(
        requested_universe_size=int(universe_size),
        universe_size=len(rows),
        attempted=attempts,
        downloaded=downloaded,
        skipped=skipped,
        failed=failed,
        failed_symbols=failed_symbols[:20],
        universe_source=universe_source,
        universe_file=universe_path,
    )
