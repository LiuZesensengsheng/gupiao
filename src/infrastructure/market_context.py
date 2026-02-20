from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.domain.symbols import SymbolError, normalize_symbol
from src.infrastructure.features import build_features
from src.infrastructure.margin_features import build_market_margin_features
from src.infrastructure.market_data import DataError, load_local_daily, load_symbol_daily


_SYMBOL_FILE_PATTERN = re.compile(r"^(\d{6}\.(SH|SZ))\.csv$", re.IGNORECASE)
_DEFAULT_INDEX_SPECS: tuple[tuple[str, str], ...] = (
    ("000001.SH", "idx_sh"),
    ("399001.SZ", "idx_sz"),
    ("399006.SZ", "idx_cyb"),
)
_INDEX_FEATURE_BASE = [
    "ret_1",
    "ret_5",
    "ret_20",
    "trend_5_20",
    "trend_20_60",
    "volatility_20",
    "drawdown_20",
    "vol_ratio_20",
]
_BREADTH_FEATURE_COLS = [
    "breadth_up_ratio",
    "breadth_down_ratio",
    "breadth_up_down_diff",
    "breadth_limit_up_ratio",
    "breadth_limit_down_ratio",
    "breadth_limit_spread",
    "breadth_amount_z20",
    "breadth_coverage",
]
_BREADTH_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}


@dataclass(frozen=True)
class MarketContextBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    notes: list[str]


def _safe_symbol(value: str) -> str | None:
    try:
        return normalize_symbol(value).symbol
    except SymbolError:
        return None


def _list_local_symbols(data_dir: str | Path, limit: int, exclude: Iterable[str]) -> list[str]:
    root = Path(data_dir)
    if not root.exists():
        return []
    excluded = {s for s in (_safe_symbol(x) for x in exclude) if s is not None}
    out: list[str] = []
    for path in sorted(root.glob("*.csv")):
        matched = _SYMBOL_FILE_PATTERN.match(path.name)
        if not matched:
            continue
        symbol = _safe_symbol(matched.group(1))
        if symbol is None or symbol in excluded:
            continue
        out.append(symbol)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _build_index_context(
    *,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    index_specs: Sequence[tuple[str, str]],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    notes: list[str] = []
    merged: pd.DataFrame | None = None
    cols: list[str] = []

    for symbol, prefix in index_specs:
        try:
            raw = load_symbol_daily(symbol=symbol, source=source, data_dir=data_dir, start=start, end=end)
        except DataError as exc:
            notes.append(f"index context skipped {symbol}: {exc}")
            continue
        feat = build_features(raw)[["date"] + _INDEX_FEATURE_BASE].copy()
        rename = {name: f"{prefix}_{name}" for name in _INDEX_FEATURE_BASE}
        feat = feat.rename(columns=rename)
        if merged is None:
            merged = feat
        else:
            merged = merged.merge(feat, on="date", how="outer", validate="1:1")
        cols.extend(rename.values())

    if merged is None:
        return pd.DataFrame(columns=["date"]), [], notes
    merged = merged.sort_values("date").drop_duplicates(subset=["date"])
    return merged, cols, notes


def _build_breadth_context(
    *,
    data_dir: str,
    max_symbols: int,
    min_coverage: int,
    exclude_symbols: Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    notes: list[str] = []
    cache_key = (str(Path(data_dir).resolve()), int(max_symbols), int(min_coverage))
    cached = _BREADTH_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy(), list(_BREADTH_FEATURE_COLS), notes

    symbols = _list_local_symbols(
        data_dir=data_dir,
        limit=max_symbols,
        exclude=exclude_symbols,
    )
    if not symbols:
        notes.append("breadth context skipped: no local symbols available")
        return pd.DataFrame(columns=["date"]), [], notes

    parts: list[pd.DataFrame] = []
    for symbol in symbols:
        try:
            frame = load_local_daily(symbol=symbol, data_dir=data_dir)
        except DataError:
            continue
        if frame.empty or len(frame) < 30:
            continue
        ret_1 = frame["close"].pct_change(1)
        amount = pd.to_numeric(frame["amount"], errors="coerce").fillna(frame["close"] * frame["volume"])
        part = pd.DataFrame(
            {
                "date": pd.to_datetime(frame["date"], errors="coerce"),
                "adv": (ret_1 > 0.0).astype(float),
                "dec": (ret_1 < 0.0).astype(float),
                "flat": (ret_1 == 0.0).astype(float),
                "lim_up": (ret_1 >= 0.095).astype(float),
                "lim_dn": (ret_1 <= -0.095).astype(float),
                "amount": amount.astype(float),
                "count": 1.0,
            }
        )
        part = part.dropna(subset=["date", "amount"])
        parts.append(part.iloc[1:].copy())

    if not parts:
        notes.append("breadth context skipped: no usable local symbols")
        return pd.DataFrame(columns=["date"]), [], notes

    agg = pd.concat(parts, ignore_index=True)
    agg = agg.groupby("date", as_index=False)[["adv", "dec", "flat", "lim_up", "lim_dn", "amount", "count"]].sum()
    agg = agg.sort_values("date").reset_index(drop=True)
    agg = agg[agg["count"] >= float(max(1, int(min_coverage)))]
    if agg.empty:
        notes.append("breadth context skipped: coverage below threshold")
        return pd.DataFrame(columns=["date"]), [], notes

    agg["breadth_up_ratio"] = agg["adv"] / np.maximum(agg["count"], 1.0)
    agg["breadth_down_ratio"] = agg["dec"] / np.maximum(agg["count"], 1.0)
    agg["breadth_up_down_diff"] = agg["breadth_up_ratio"] - agg["breadth_down_ratio"]
    agg["breadth_limit_up_ratio"] = agg["lim_up"] / np.maximum(agg["count"], 1.0)
    agg["breadth_limit_down_ratio"] = agg["lim_dn"] / np.maximum(agg["count"], 1.0)
    agg["breadth_limit_spread"] = agg["breadth_limit_up_ratio"] - agg["breadth_limit_down_ratio"]

    amount_log = np.log1p(np.maximum(agg["amount"].astype(float), 0.0))
    amount_std = amount_log.rolling(20).std().replace(0.0, np.nan)
    agg["breadth_amount_z20"] = (amount_log - amount_log.rolling(20).mean()) / (amount_std + 1e-9)
    agg["breadth_coverage"] = agg["count"].astype(float)

    out = agg[["date"] + _BREADTH_FEATURE_COLS].copy()
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    _BREADTH_CACHE[cache_key] = out.copy()
    return out, list(_BREADTH_FEATURE_COLS), notes


def build_market_context_features(
    *,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    market_dates: pd.Series,
    use_margin_features: bool = True,
    margin_market_file: str = "input/margin_market.csv",
    index_specs: Sequence[tuple[str, str]] | None = None,
    breadth_max_symbols: int = 800,
    breadth_min_coverage: int = 30,
    min_valid_ratio: float = 0.55,
    min_valid_points: int = 120,
) -> MarketContextBundle:
    index_specs = tuple(index_specs or _DEFAULT_INDEX_SPECS)
    base = pd.DataFrame({"date": pd.to_datetime(market_dates, errors="coerce")}).dropna(subset=["date"])
    base = base.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if base.empty:
        return MarketContextBundle(frame=pd.DataFrame(columns=["date"]), feature_columns=[], notes=["empty market dates"])

    notes: list[str] = []
    idx_frame, idx_cols, idx_notes = _build_index_context(
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        index_specs=index_specs,
    )
    notes.extend(idx_notes)

    exclude = [x[0] for x in index_specs] + ["000300.SH", "000905.SH", "000852.SH"]
    breadth_frame, breadth_cols, breadth_notes = _build_breadth_context(
        data_dir=data_dir,
        max_symbols=int(max(20, breadth_max_symbols)),
        min_coverage=int(max(1, breadth_min_coverage)),
        exclude_symbols=exclude,
    )
    notes.extend(breadth_notes)
    margin_frame = pd.DataFrame(columns=["date"])
    margin_cols: list[str] = []
    if use_margin_features:
        margin_frame, margin_cols, margin_notes = build_market_margin_features(
            margin_market_file=margin_market_file,
            start=start,
            end=end,
        )
        notes.extend(margin_notes)

    merged = base.copy()
    if not idx_frame.empty:
        merged = merged.merge(idx_frame, on="date", how="left", validate="1:1")
    if not breadth_frame.empty:
        merged = merged.merge(breadth_frame, on="date", how="left", validate="1:1")
    if not margin_frame.empty:
        merged = merged.merge(margin_frame, on="date", how="left", validate="1:1")

    candidate_cols = [col for col in idx_cols + breadth_cols + margin_cols if col in merged.columns]
    selected_cols: list[str] = []
    for col in candidate_cols:
        valid_n = int(merged[col].notna().sum())
        valid_ratio = float(merged[col].notna().mean())
        if valid_n >= int(min_valid_points) and valid_ratio >= float(min_valid_ratio):
            selected_cols.append(col)
        else:
            notes.append(f"context column dropped {col}: valid={valid_n}, ratio={valid_ratio:.2f}")

    if not selected_cols:
        return MarketContextBundle(frame=base, feature_columns=[], notes=notes)

    out = merged[["date"] + selected_cols].copy().sort_values("date")
    for col in selected_cols:
        # Keep alignment stable while avoiding full-row drops from sparse context values.
        out[col] = out[col].ffill()
        if out[col].isna().any():
            out[col] = out[col].fillna(float(out[col].median(skipna=True)))
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return MarketContextBundle(frame=out, feature_columns=selected_cols, notes=notes)
