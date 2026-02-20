from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.domain.entities import Security
from src.domain.symbols import SymbolError, normalize_symbol


SYMBOL_FILE_PATTERN = re.compile(r"^(\d{6}\.(SH|SZ))\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveryUniverse:
    rows: List[Security]
    source_label: str
    warnings: List[str]


def _safe_symbol(value: str) -> str | None:
    try:
        return normalize_symbol(value).symbol
    except SymbolError:
        return None


def _dedupe_rows(rows: Sequence[Security], exclude_symbols: Iterable[str]) -> List[Security]:
    excluded = {normalize_symbol(x).symbol for x in exclude_symbols if _safe_symbol(x) is not None}
    out: List[Security] = []
    seen: set[str] = set()
    for row in rows:
        symbol = _safe_symbol(row.symbol)
        if symbol is None or symbol in excluded or symbol in seen:
            continue
        seen.add(symbol)
        out.append(Security(symbol=symbol, name=row.name or symbol, sector=row.sector or "其他"))
    return out


def _load_universe_file(path: str | Path) -> List[Security]:
    file_path = Path(path)
    if not str(path).strip() or not file_path.exists():
        return []

    if file_path.suffix.lower() in {".json"}:
        with file_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        rows = payload if isinstance(payload, list) else payload.get("stocks", [])
        out: List[Security] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            symbol = _safe_symbol(str(item.get("symbol", "")))
            if symbol is None:
                continue
            out.append(
                Security(
                    symbol=symbol,
                    name=str(item.get("name", symbol)),
                    sector=str(item.get("sector", "其他")),
                )
            )
        return out

    raw = pd.read_csv(file_path)
    if raw.empty:
        return []
    lower_map = {c.lower(): c for c in raw.columns}
    symbol_col = lower_map.get("symbol")
    if symbol_col is None:
        raise ValueError(f"Universe file missing required column `symbol`: {file_path}")
    name_col = lower_map.get("name")
    sector_col = lower_map.get("sector")

    out: List[Security] = []
    for _, row in raw.iterrows():
        symbol = _safe_symbol(str(row[symbol_col]))
        if symbol is None:
            continue
        name = symbol if name_col is None else str(row[name_col])
        sector = "其他" if sector_col is None else str(row[sector_col])
        out.append(Security(symbol=symbol, name=name, sector=sector))
    return out


def _from_data_dir(data_dir: str | Path, limit: int) -> List[Security]:
    root = Path(data_dir)
    if not root.exists():
        return []
    out: List[Security] = []
    for path in sorted(root.glob("*.csv")):
        matched = SYMBOL_FILE_PATTERN.match(path.name)
        if not matched:
            continue
        symbol = _safe_symbol(matched.group(1))
        if symbol is None:
            continue
        out.append(Security(symbol=symbol, name=symbol, sector="其他"))
        if len(out) >= limit:
            break
    return out


def _pick_col(raw: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in raw.columns:
            return name
    return None


def _from_akshare_spot(limit: int, min_amount: float = 1e8, exclude_st: bool = True) -> List[Security]:
    try:
        import akshare as ak
    except Exception:
        return []

    try:
        raw = ak.stock_zh_a_spot_em()
    except Exception:
        return []
    if raw is None or raw.empty:
        return []

    code_col = _pick_col(raw, ["代码", "symbol", "代码 "])
    name_col = _pick_col(raw, ["名称", "name"])
    amount_col = _pick_col(raw, ["成交额", "amount"])
    if code_col is None:
        return []

    frame = raw.copy()
    if amount_col is not None:
        frame[amount_col] = pd.to_numeric(frame[amount_col], errors="coerce")
        frame = frame[frame[amount_col] >= float(min_amount)]
        frame = frame.sort_values(amount_col, ascending=False)

    out: List[Security] = []
    for _, row in frame.iterrows():
        symbol = _safe_symbol(str(row[code_col]))
        if symbol is None:
            continue
        name = symbol if name_col is None else str(row[name_col])
        if exclude_st and "ST" in name.upper():
            continue
        out.append(Security(symbol=symbol, name=name, sector="其他"))
        if len(out) >= int(limit):
            break
    return out


def build_candidate_universe(
    *,
    source: str,
    data_dir: str,
    universe_file: str,
    candidate_limit: int,
    exclude_symbols: Iterable[str],
) -> DiscoveryUniverse:
    warnings: List[str] = []
    limit = max(5, int(candidate_limit))

    file_rows = _load_universe_file(universe_file)
    if file_rows:
        rows = _dedupe_rows(file_rows, exclude_symbols=exclude_symbols)
        return DiscoveryUniverse(rows=rows[:limit], source_label=f"universe_file:{universe_file}", warnings=warnings)

    source_l = str(source).lower()
    rows: List[Security] = []
    source_label = "data_dir"
    if source_l != "local":
        rows = _from_akshare_spot(limit=limit)
        if rows:
            source_label = "akshare_spot_em"
        else:
            warnings.append("akshare spot list unavailable, fallback to local data_dir symbols.")

    if not rows:
        rows = _from_data_dir(data_dir=data_dir, limit=limit)
        source_label = "data_dir"

    rows = _dedupe_rows(rows, exclude_symbols=exclude_symbols)
    return DiscoveryUniverse(rows=rows[:limit], source_label=source_label, warnings=warnings)


def compute_volume_risk(frame: pd.DataFrame, as_of_date: pd.Timestamp) -> tuple[bool, str]:
    if frame is None or frame.empty:
        return False, "无特征数据"
    history = frame[frame["date"] <= as_of_date].sort_values("date")
    if history.empty:
        return False, "无截至日数据"
    latest = history.iloc[-1]
    hvbd_recent = float(latest.get("hvbd_recent_5", 0.0))
    vol_ratio = float(latest.get("vol_ratio_20", np.nan))
    pos = float(latest.get("price_pos_20", np.nan))
    if hvbd_recent >= 0.5:
        return True, f"高位巨量大阴线(5日内), 量能比={vol_ratio:.2f}, 位置={pos:.2f}"
    return False, f"量能比={vol_ratio:.2f}, 位置={pos:.2f}"
