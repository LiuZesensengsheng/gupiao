from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


class DataError(RuntimeError):
    """Raised when market data loading fails."""


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    code: str
    exchange: str


def normalize_symbol(symbol: str) -> SymbolInfo:
    text = symbol.strip().upper()
    if "." in text:
        code, exchange = text.split(".", 1)
    else:
        code = text
        if not code.isdigit() or len(code) != 6:
            raise DataError(f"Unsupported symbol format: {symbol}")
        exchange = "SH" if code.startswith(("5", "6", "9")) else "SZ"
    if exchange not in {"SH", "SZ"}:
        raise DataError(f"Unsupported exchange in symbol: {symbol}")
    return SymbolInfo(symbol=f"{code}.{exchange}", code=code, exchange=exchange)


def to_secid(symbol: str) -> str:
    info = normalize_symbol(symbol)
    market = 1 if info.exchange == "SH" else 0
    return f"{market}.{info.code}"


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


def fetch_eastmoney_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
    timeout: int = 10,
) -> pd.DataFrame:
    secid = to_secid(symbol)
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "ut": "7eea3edcaed734bea9cbfc24409ed989",
        "klt": "101",
        "fqt": "1",
        "secid": secid,
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
        row = {
            "date": parts[0],
            "open": parts[1],
            "close": parts[2],
            "high": parts[3],
            "low": parts[4],
            "volume": parts[5],
            "amount": parts[6] if len(parts) > 6 else None,
        }
        rows.append(row)

    if not rows:
        raise DataError(f"{symbol}: kline rows are empty after parsing")
    return _normalize_daily_columns(pd.DataFrame(rows), symbol=symbol)


def load_local_daily(symbol: str, data_dir: str | Path) -> pd.DataFrame:
    info = normalize_symbol(symbol)
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
    if source == "eastmoney":
        return fetch_eastmoney_daily(symbol=symbol, start=start, end=end)
    if source == "local":
        return load_local_daily(symbol=symbol, data_dir=data_dir)
    raise DataError(f"Unknown data source: {source}")
