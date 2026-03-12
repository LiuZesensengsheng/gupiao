from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from src.domain.entities import Security
from src.domain.symbols import SymbolError, normalize_symbol
from src.infrastructure.security_metadata import enrich_securities_with_metadata


SYMBOL_FILE_PATTERN = re.compile(r"^(\d{6}\.(SH|SZ))\.csv$", re.IGNORECASE)


@dataclass(frozen=True)
class DiscoveryUniverse:
    rows: List[Security]
    source_label: str
    warnings: List[str]
    universe_id: str = ""
    universe_size: int = 0
    generation_rule: str = ""
    manifest_path: str = ""
    symbols: List[str] = field(default_factory=list)


PREDEFINED_UNIVERSE_TIERS: dict[str, dict[str, object]] = {
    "favorites_16": {
        "limit": 16,
        "mode": "favorites",
        "legacy_aliases": ["top16", "favorites", "16"],
    },
    "generated_80": {
        "limit": 80,
        "mode": "generated",
        "legacy_aliases": ["top80", "80"],
    },
    "generated_150": {
        "limit": 150,
        "mode": "generated",
        "legacy_aliases": ["top150", "150"],
    },
    "generated_300": {
        "limit": 300,
        "mode": "generated",
        "legacy_aliases": ["top300", "300"],
    },
}

DEFAULT_GENERATED_UNIVERSE_RULE = (
    "local_data_only + exclude_benchmark + exclude_st + min_history + min_liquidity + recent_amount_rank"
)


def _safe_symbol(value: str) -> str | None:
    try:
        return normalize_symbol(value).symbol
    except SymbolError:
        return None


def _dedupe_rows(
    rows: Sequence[Security],
    exclude_symbols: Iterable[str],
    *,
    enrich_metadata: bool = True,
) -> List[Security]:
    excluded = {normalize_symbol(x).symbol for x in exclude_symbols if _safe_symbol(x) is not None}
    out: List[Security] = []
    seen: set[str] = set()
    for row in rows:
        symbol = _safe_symbol(row.symbol)
        if symbol is None or symbol in excluded or symbol in seen:
            continue
        seen.add(symbol)
        out.append(Security(symbol=symbol, name=row.name or symbol, sector=row.sector or "其他"))
    return enrich_securities_with_metadata(out) if enrich_metadata else out


def normalize_universe_tier(tier_id: str | None) -> str:
    raw = str(tier_id or "").strip().lower()
    if not raw:
        return ""
    for canonical, payload in PREDEFINED_UNIVERSE_TIERS.items():
        if raw == canonical:
            return canonical
        aliases = payload.get("legacy_aliases", [])
        if isinstance(aliases, list) and raw in {str(item).lower() for item in aliases}:
            return canonical
    raise ValueError(f"Unsupported universe tier: {tier_id}")


def _load_universe_file(path: str | Path, *, enrich_metadata: bool = True) -> List[Security]:
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
        return enrich_securities_with_metadata(out) if enrich_metadata else out

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
    return enrich_securities_with_metadata(out) if enrich_metadata else out


def _resolve_local_symbol_path(data_dir: str | Path, symbol: str) -> Path:
    return Path(data_dir) / f"{normalize_symbol(symbol).symbol}.csv"


def _safe_read_local_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if raw.empty or "date" not in raw.columns:
        return pd.DataFrame()
    out = raw.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "amount" not in out.columns and {"close", "volume"}.issubset(out.columns):
        out["amount"] = out["close"] * out["volume"]
    return out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _rank_generated_universe_rows(
    *,
    rows: Sequence[Security],
    data_dir: str | Path,
    limit: int,
) -> tuple[List[Security], List[str]]:
    diagnostics: list[tuple[float, int, str, Security]] = []
    warnings: list[str] = []
    min_history_days = 480
    min_recent_amount = 2.0e7
    for row in rows:
        if "ST" in str(row.name or "").upper():
            continue
        local_path = _resolve_local_symbol_path(data_dir, row.symbol)
        daily = _safe_read_local_daily(local_path)
        if daily.empty:
            continue
        history_days = int(len(daily))
        if history_days < min_history_days:
            continue
        recent_window = daily.tail(min(60, history_days))
        median_amount = float(pd.to_numeric(recent_window.get("amount"), errors="coerce").median())
        if not np.isfinite(median_amount) or median_amount < min_recent_amount:
            continue
        diagnostics.append((median_amount, history_days, str(row.symbol), row))
    diagnostics.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [item[3] for item in diagnostics[: max(1, int(limit))]]
    if len(selected) < max(1, int(limit)):
        warnings.append(
            f"generated universe degraded: requested={limit}, selected={len(selected)}, min_history_days={min_history_days}, min_recent_amount={min_recent_amount:.0f}"
        )
    return selected, warnings


def write_predefined_universe_manifest(
    *,
    out_path: str | Path,
    tier_id: str,
    rows: Sequence[Security],
    source: str,
    generation_rule: str,
) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "universe_id": str(tier_id),
        "universe_size": int(len(rows)),
        "source": str(source),
        "generation_rule": str(generation_rule),
        "stocks": [
            {
                "symbol": str(item.symbol),
                "name": str(item.name),
                "sector": str(item.sector),
            }
            for item in rows
        ],
        "symbols": [str(item.symbol) for item in rows],
        "symbol_count": int(len(rows)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_predefined_universe(
    *,
    tier_id: str,
    data_dir: str,
    favorites_file: str,
    generated_base_file: str,
    output_path: str | Path | None = None,
    exclude_symbols: Iterable[str] = (),
) -> DiscoveryUniverse:
    normalized_tier = normalize_universe_tier(tier_id)
    spec = PREDEFINED_UNIVERSE_TIERS[normalized_tier]
    limit = int(spec["limit"])
    mode = str(spec["mode"])
    warnings: List[str] = []
    if mode == "favorites":
        raw_rows = _load_universe_file(favorites_file, enrich_metadata=False)
        rows = _dedupe_rows(raw_rows, exclude_symbols=exclude_symbols, enrich_metadata=False)[:limit]
        rows = enrich_securities_with_metadata(rows)
        source = f"favorites_file:{favorites_file}"
        generation_rule = "manual_favorites_locked"
    else:
        base_rows = _load_universe_file(generated_base_file, enrich_metadata=False)
        if not base_rows:
            base_rows = _from_data_dir(data_dir=data_dir, limit=10000)
            source = f"data_dir:{data_dir}"
        else:
            source = f"generated_base:{generated_base_file}"
        deduped_rows = _dedupe_rows(base_rows, exclude_symbols=exclude_symbols, enrich_metadata=False)
        rows, rank_warnings = _rank_generated_universe_rows(
            rows=deduped_rows,
            data_dir=data_dir,
            limit=limit,
        )
        rows = enrich_securities_with_metadata(rows)
        warnings.extend(rank_warnings)
        generation_rule = DEFAULT_GENERATED_UNIVERSE_RULE
    manifest_path = ""
    if output_path is not None:
        manifest_path = str(
            write_predefined_universe_manifest(
                out_path=output_path,
                tier_id=normalized_tier,
                rows=rows,
                source=source,
                generation_rule=generation_rule,
            ).resolve()
        )
    return DiscoveryUniverse(
        rows=list(rows),
        source_label=source,
        warnings=warnings,
        universe_id=normalized_tier,
        universe_size=int(len(rows)),
        generation_rule=generation_rule,
        manifest_path=manifest_path,
        symbols=[str(item.symbol) for item in rows],
    )


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
    return enrich_securities_with_metadata(out)


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
    return enrich_securities_with_metadata(out)


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

    file_rows = _load_universe_file(universe_file, enrich_metadata=False)
    if file_rows:
        rows = _dedupe_rows(file_rows, exclude_symbols=exclude_symbols, enrich_metadata=False)
        selected = enrich_securities_with_metadata(rows[:limit])
        path = Path(str(universe_file))
        return DiscoveryUniverse(
            rows=selected,
            source_label=f"universe_file:{universe_file}",
            warnings=warnings,
            universe_id=path.stem or "custom_universe",
            universe_size=int(len(selected)),
            generation_rule="external_universe_file",
            manifest_path=str(path.resolve()) if path.exists() else str(path),
            symbols=[str(item.symbol) for item in selected],
        )

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
    selected = rows[:limit]
    return DiscoveryUniverse(
        rows=selected,
        source_label=source_label,
        warnings=warnings,
        universe_id=source_label,
        universe_size=int(len(selected)),
        generation_rule="dynamic_source_selection",
        symbols=[str(item.symbol) for item in selected],
    )


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
