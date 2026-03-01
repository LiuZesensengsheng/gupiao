from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd

from src.domain.entities import Security
from src.domain.symbols import normalize_symbol
from src.infrastructure.market_data import DataError, load_symbol_daily


def build_sector_daily_frames(
    *,
    stock_securities: Sequence[Security],
    sector_map: Dict[str, str],
    source: str,
    data_dir: str,
    start: str,
    end: str,
) -> dict[str, pd.DataFrame]:
    grouped: dict[str, list[pd.DataFrame]] = {}
    for security in stock_securities:
        symbol = normalize_symbol(security.symbol).symbol
        sector = str(sector_map.get(symbol, security.sector or "其他"))
        try:
            frame = load_symbol_daily(
                symbol=symbol,
                source=source,
                data_dir=data_dir,
                start=start,
                end=end,
            )
        except DataError:
            continue
        if frame.empty:
            continue
        usable = frame[["date", "close", "volume", "amount"]].copy()
        usable["date"] = pd.to_datetime(usable["date"], errors="coerce")
        usable = usable.dropna(subset=["date", "close"]).sort_values("date")
        if len(usable) < 30:
            continue
        usable["ret_1"] = usable["close"].pct_change(1)
        usable["symbol"] = symbol
        grouped.setdefault(sector, []).append(usable)

    out: dict[str, pd.DataFrame] = {}
    for sector, parts in grouped.items():
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        agg = (
            merged.groupby("date", as_index=False)
            .agg(
                eq_ret_1=("ret_1", "mean"),
                volume=("volume", "sum"),
                amount=("amount", "sum"),
                coverage=("symbol", "nunique"),
            )
            .sort_values("date")
            .reset_index(drop=True)
        )
        agg = agg[agg["coverage"] >= 1].copy()
        if agg.empty:
            continue
        nav = (1.0 + agg["eq_ret_1"].fillna(0.0)).cumprod()
        close = 100.0 * nav
        open_ = close.shift(1).fillna(close)
        high = np.maximum(open_, close)
        low = np.minimum(open_, close)
        out[sector] = pd.DataFrame(
            {
                "date": agg["date"],
                "open": open_.astype(float),
                "high": high.astype(float),
                "low": low.astype(float),
                "close": close.astype(float),
                "volume": agg["volume"].astype(float).fillna(0.0),
                "amount": agg["amount"].astype(float).fillna(0.0),
                "coverage": agg["coverage"].astype(float),
            }
        )
    return out
