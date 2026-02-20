from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

from src.domain.symbols import SymbolError, normalize_symbol


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
EASTMONEY_CACHE_DIR = "_eastmoney_cache"
_MEM_CACHE: Dict[tuple[str, str, str, str, str], pd.DataFrame] = {}


class DataError(RuntimeError):
    """Raised when market data loading fails."""


def _normalize_daily_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    out = df.copy()
    if "date" not in out.columns:
        raise DataError(f"{symbol}: missing required column 'date'")

    required_numeric = ["open", "high", "low", "close", "volume"]
    for col in required_numeric:
        if col not in out.columns:
            raise DataError(f"{symbol}: missing required column '{col}'")

    if "amount" not in out.columns:
        out["amount"] = out["close"] * out["volume"]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values("date").drop_duplicates(subset=["date"])
    out["symbol"] = normalize_symbol(symbol).symbol
    return out.reset_index(drop=True)


def _slice_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    return out.sort_values("date").reset_index(drop=True)


def fetch_eastmoney_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
    timeout: int = 10,
) -> pd.DataFrame:
    try:
        info = normalize_symbol(symbol)
    except SymbolError as exc:
        raise DataError(str(exc)) from exc

    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",
        "secid": info.secid,
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(EASTMONEY_KLINE_URL, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        raise DataError(f"{symbol}: eastmoney request failed: {exc}") from exc

    data = payload.get("data") if isinstance(payload, dict) else None
    lines = data.get("klines") if isinstance(data, dict) else None
    if not lines:
        raise DataError(f"{symbol}: eastmoney returned no kline data")

    rows: List[Dict[str, object]] = []
    for line in lines:
        parts = str(line).split(",")
        if len(parts) < 6:
            continue
        rows.append(
            {
                "date": parts[0],
                "open": parts[1],
                "close": parts[2],
                "high": parts[3],
                "low": parts[4],
                "volume": parts[5],
                "amount": parts[6] if len(parts) > 6 else None,
            }
        )

    if not rows:
        raise DataError(f"{symbol}: kline rows are empty after parsing")
    return _normalize_daily_columns(pd.DataFrame(rows), symbol=symbol)


def _eastmoney_cache_path(symbol: str, data_dir: str | Path) -> Path:
    info = normalize_symbol(symbol)
    root = Path(data_dir) / EASTMONEY_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{info.symbol}.csv"


def _load_eastmoney_cache(symbol: str, data_dir: str | Path) -> pd.DataFrame | None:
    path = _eastmoney_cache_path(symbol, data_dir)
    if not path.exists():
        return None
    try:
        raw = pd.read_csv(path)
        return _normalize_daily_columns(raw, symbol=symbol)
    except Exception:
        return None


def _write_eastmoney_cache(symbol: str, data_dir: str | Path, df: pd.DataFrame) -> None:
    path = _eastmoney_cache_path(symbol, data_dir)
    df.to_csv(path, index=False)


def load_local_daily(symbol: str, data_dir: str | Path) -> pd.DataFrame:
    try:
        info = normalize_symbol(symbol)
    except SymbolError as exc:
        raise DataError(str(exc)) from exc

    root = Path(data_dir)
    candidates = [
        root / f"{info.symbol}.csv",
        root / f"{info.code}_{info.exchange}.csv",
        root / f"{info.code}.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        names = ", ".join(p.name for p in candidates)
        raise DataError(f"{info.symbol}: local csv not found, expected one of: {names}")
    raw = pd.read_csv(path)
    return _normalize_daily_columns(raw, symbol=info.symbol)


def load_symbol_daily(
    symbol: str,
    source: str,
    data_dir: str | Path = "data",
    start: str = "2010-01-01",
    end: str = "2099-12-31",
) -> pd.DataFrame:
    try:
        norm_symbol = normalize_symbol(symbol).symbol
    except SymbolError as exc:
        raise DataError(str(exc)) from exc

    cache_key = (norm_symbol, source, str(data_dir), start, end)
    cached_mem = _MEM_CACHE.get(cache_key)
    if cached_mem is not None:
        return cached_mem.copy()

    if source == "eastmoney":
        target_end = min(pd.Timestamp(end), pd.Timestamp.today().normalize())
        cached_disk = _load_eastmoney_cache(norm_symbol, data_dir=data_dir)

        need_refresh = True
        if cached_disk is not None and not cached_disk.empty:
            disk_min = pd.Timestamp(cached_disk["date"].min())
            disk_max = pd.Timestamp(cached_disk["date"].max())
            if disk_min <= pd.Timestamp(start) and disk_max >= target_end - pd.Timedelta(days=3):
                need_refresh = False

        df: pd.DataFrame
        if need_refresh:
            try:
                df = fetch_eastmoney_daily(symbol=norm_symbol, start=start, end=end)
                _write_eastmoney_cache(norm_symbol, data_dir=data_dir, df=df)
            except DataError:
                if cached_disk is None or cached_disk.empty:
                    raise
                df = cached_disk
        else:
            df = cached_disk

        out = _slice_date_range(df, start=start, end=end)
        if out.empty:
            raise DataError(f"{norm_symbol}: no data available after date filtering [{start}, {end}]")
        _MEM_CACHE[cache_key] = out.copy()
        return out.copy()
    if source == "local":
        out = _slice_date_range(load_local_daily(symbol=norm_symbol, data_dir=data_dir), start=start, end=end)
        if out.empty:
            raise DataError(f"{norm_symbol}: no local rows in date range [{start}, {end}]")
        _MEM_CACHE[cache_key] = out.copy()
        return out.copy()
    raise DataError(f"Unknown data source: {source}")
