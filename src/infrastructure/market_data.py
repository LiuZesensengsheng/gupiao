from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Dict, List, Sequence

import pandas as pd
import requests

from src.domain.symbols import SymbolError, normalize_symbol


EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
_HTTP_SESSION = requests.Session()
_HTTP_SESSION.trust_env = False
EASTMONEY_CACHE_DIR = "_eastmoney_cache"
AUTO_SOURCE_CHAIN = ("eastmoney", "tushare", "akshare", "baostock")
SUPPORTED_SOURCES = {"eastmoney", "tushare", "akshare", "baostock", "local", "auto"}
_MEM_CACHE: Dict[tuple[str, str, str, str, str], pd.DataFrame] = {}
_US_INDEX_MEM_CACHE: Dict[tuple[str, str], pd.DataFrame] = {}
_US_ETF_MEM_CACHE: Dict[tuple[str, str], pd.DataFrame] = {}
_CN_ETF_MEM_CACHE: Dict[str, pd.DataFrame] = {}
_TUSHARE_TOKEN: str = ""


class DataError(RuntimeError):
    """Raised when market data loading fails."""


def set_tushare_token(token: str | None) -> None:
    global _TUSHARE_TOKEN
    _TUSHARE_TOKEN = "" if token is None else str(token).strip()


def _resolve_tushare_token() -> str:
    if _TUSHARE_TOKEN.strip():
        return _TUSHARE_TOKEN.strip()
    return str(os.getenv("TUSHARE_TOKEN", "")).strip()


def _get_tushare_pro():
    token = _resolve_tushare_token()
    if not token:
        raise DataError("tushare token is missing; set `TUSHARE_TOKEN` or pass `--tushare-token`")
    try:
        import tushare as ts
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataError("tushare is not installed, run: pip install tushare") from exc
    return ts.pro_api(token)


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


def _normalize_external_index_columns(df: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    raw = df.copy()
    lower_map = {str(col).lower(): str(col) for col in raw.columns}

    date_col = lower_map.get("date") or lower_map.get("日期".lower())
    open_col = lower_map.get("open") or lower_map.get("开盘".lower())
    high_col = lower_map.get("high") or lower_map.get("最高".lower())
    low_col = lower_map.get("low") or lower_map.get("最低".lower())
    close_col = lower_map.get("close") or lower_map.get("收盘".lower())
    volume_col = lower_map.get("volume") or lower_map.get("成交量".lower()) or lower_map.get("成交".lower())
    amount_col = lower_map.get("amount") or lower_map.get("成交额".lower())

    required = {
        "date": date_col,
        "open": open_col,
        "high": high_col,
        "low": low_col,
        "close": close_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise DataError(f"{symbol}: missing required columns: {', '.join(missing)}")

    out = pd.DataFrame(
        {
            "date": raw[required["date"]],
            "open": raw[required["open"]],
            "high": raw[required["high"]],
            "low": raw[required["low"]],
            "close": raw[required["close"]],
        }
    )
    if volume_col is not None:
        out["volume"] = raw[volume_col]
    else:
        out["volume"] = 0.0
    if amount_col is not None:
        out["amount"] = raw[amount_col]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "amount" in out.columns:
        out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    else:
        out["amount"] = out["close"] * out["volume"].fillna(0.0)

    out["volume"] = out["volume"].fillna(0.0).clip(lower=0.0)
    out["amount"] = out["amount"].fillna(out["close"] * out["volume"]).clip(lower=0.0)
    out = out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"]).reset_index(drop=True)
    out["symbol"] = str(symbol)
    return out


def _pick_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> pd.Series | None:
    for name in candidates:
        if name in df.columns:
            return df[name]
    if required:
        raise DataError(f"missing required column in source dataframe, candidates={candidates}")
    return None


def fetch_eastmoney_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
    timeout: int = 8,
    retries: int = 2,
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
    payload = None
    last_error: Exception | None = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            resp = _HTTP_SESSION.get(EASTMONEY_KLINE_URL, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:  # pragma: no cover - network jitter dependent
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(0.6 * attempt)
    if payload is None:
        raise DataError(f"{symbol}: eastmoney request failed: {last_error}") from last_error

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


def fetch_tushare_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
) -> pd.DataFrame:
    try:
        info = normalize_symbol(symbol)
    except SymbolError as exc:
        raise DataError(str(exc)) from exc

    pro = _get_tushare_pro()
    ts_code = info.symbol
    start_date = start.replace("-", "")
    end_date = end.replace("-", "")

    def _looks_like_tushare_index() -> bool:
        if info.exchange == "SH" and info.code.startswith("000"):
            return True
        if info.exchange == "SZ" and info.code.startswith("399"):
            return True
        return False

    try:
        if _looks_like_tushare_index():
            raw = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        elif info.code.startswith(("0", "3", "6", "8", "9")):
            raw = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            raw = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as exc:
        if _looks_like_tushare_index() and "没有接口访问权限" in str(exc):
            raise DataError(
                f"{symbol}: tushare index_daily permission missing; see https://tushare.pro/document/1?doc_id=108"
            ) from exc
        raise DataError(f"{symbol}: tushare request failed: {exc}") from exc

    if raw is None or raw.empty:
        # If daily returns empty for index-like symbols, try index_daily fallback.
        try:
            raw = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        except Exception:
            pass

    if (raw is None or raw.empty) and _looks_like_tushare_index():
        raise DataError(
            f"{symbol}: tushare returned no index bars; confirm index_daily permission at https://tushare.pro/document/1?doc_id=108"
        )
    if raw is None or raw.empty:
        raise DataError(f"{symbol}: tushare returned no daily bars")

    frame = pd.DataFrame(
        {
            "date": _pick_column(raw, ["trade_date", "date"]),
            "open": _pick_column(raw, ["open"]),
            "high": _pick_column(raw, ["high"]),
            "low": _pick_column(raw, ["low"]),
            "close": _pick_column(raw, ["close"]),
            "volume": _pick_column(raw, ["vol", "volume"]),
        }
    )
    amount_col = _pick_column(raw, ["amount"], required=False)
    if amount_col is not None:
        frame["amount"] = amount_col
    return _normalize_daily_columns(frame, symbol=symbol)


def _normalize_tushare_batch_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "symbol": _pick_column(raw, ["ts_code", "symbol"]),
            "date": _pick_column(raw, ["trade_date", "date"]),
            "open": _pick_column(raw, ["open"]),
            "high": _pick_column(raw, ["high"]),
            "low": _pick_column(raw, ["low"]),
            "close": _pick_column(raw, ["close"]),
            "volume": _pick_column(raw, ["vol", "volume"]),
        }
    )
    amount_col = _pick_column(raw, ["amount"], required=False)
    if amount_col is not None:
        frame["amount"] = amount_col
    return frame


def _estimate_trading_days(start: str, end: str) -> int:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    days = max(1, int((end_ts - start_ts).days) + 1)
    return max(1, int(days * 245 / 365))


def _merge_tushare_batch_results(
    left: dict[str, pd.DataFrame],
    right: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    merged = dict(left)
    for symbol, frame in right.items():
        existing = merged.get(symbol)
        if existing is None or existing.empty:
            merged[symbol] = frame.copy()
            continue
        combined = pd.concat([existing, frame], ignore_index=True)
        merged[symbol] = _normalize_daily_columns(combined, symbol=symbol)
    return merged


def fetch_tushare_daily_batch(
    symbols: Sequence[str],
    start: str = "2010-01-01",
    end: str = "2099-12-31",
) -> dict[str, pd.DataFrame]:
    normalized: list[str] = []
    for symbol in symbols:
        try:
            info = normalize_symbol(symbol)
        except SymbolError as exc:
            raise DataError(str(exc)) from exc
        if not info.code.startswith(("0", "3", "6", "8", "9")):
            raise DataError(f"{info.symbol}: tushare batch only supports stock daily data")
        normalized.append(info.symbol)

    if not normalized:
        return {}

    estimated_rows = len(normalized) * _estimate_trading_days(start, end)
    if len(normalized) > 1 and estimated_rows > 5500:
        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        if start_ts >= end_ts:
            mid = max(1, len(normalized) // 2)
            merged: dict[str, pd.DataFrame] = {}
            merged.update(fetch_tushare_daily_batch(normalized[:mid], start=start, end=end))
            merged.update(fetch_tushare_daily_batch(normalized[mid:], start=start, end=end))
            return merged
        mid_ts = start_ts + (end_ts - start_ts) / 2
        left_end = mid_ts.normalize().strftime("%Y-%m-%d")
        right_start = (mid_ts.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        merged = fetch_tushare_daily_batch(normalized, start=start, end=left_end)
        tail = fetch_tushare_daily_batch(normalized, start=right_start, end=end)
        return _merge_tushare_batch_results(merged, tail)

    pro = _get_tushare_pro()
    start_date = start.replace("-", "")
    end_date = end.replace("-", "")
    ts_codes = ",".join(normalized)

    try:
        raw = pro.daily(ts_code=ts_codes, start_date=start_date, end_date=end_date)
    except Exception as exc:
        raise DataError(f"tushare batch request failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError("tushare batch returned no daily bars")

    # Tushare daily is documented with a 6000-row cap per request, so split conservatively.
    if len(normalized) > 1 and len(raw) >= 6000:
        mid = len(normalized) // 2
        merged: dict[str, pd.DataFrame] = {}
        merged.update(fetch_tushare_daily_batch(normalized[:mid], start=start, end=end))
        merged.update(fetch_tushare_daily_batch(normalized[mid:], start=start, end=end))
        return merged

    frame = _normalize_tushare_batch_frame(raw)
    grouped: dict[str, pd.DataFrame] = {}
    for symbol, part in frame.groupby("symbol", sort=False):
        grouped[str(symbol)] = _normalize_daily_columns(part.drop(columns=["symbol"]), symbol=str(symbol))

    missing = [symbol for symbol in normalized if symbol not in grouped]
    if missing:
        raise DataError(f"tushare batch missing symbols: {', '.join(missing[:10])}")
    return grouped


def fetch_akshare_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataError("akshare is not installed, run: pip install akshare") from exc

    try:
        info = normalize_symbol(symbol)
    except SymbolError as exc:
        raise DataError(str(exc)) from exc

    start_date = start.replace("-", "")
    end_date = end.replace("-", "")
    try:
        raw = ak.stock_zh_a_hist(
            symbol=info.code,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
    except Exception as exc:
        raise DataError(f"{symbol}: akshare request failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(f"{symbol}: akshare returned no daily bars")

    frame = pd.DataFrame(
        {
            "date": _pick_column(raw, ["日期", "date"]),
            "open": _pick_column(raw, ["开盘", "open"]),
            "high": _pick_column(raw, ["最高", "high"]),
            "low": _pick_column(raw, ["最低", "low"]),
            "close": _pick_column(raw, ["收盘", "close"]),
            "volume": _pick_column(raw, ["成交量", "volume"]),
        }
    )
    amount_col = _pick_column(raw, ["成交额", "amount"], required=False)
    if amount_col is not None:
        frame["amount"] = amount_col
    return _normalize_daily_columns(frame, symbol=symbol)


def fetch_us_index_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
    source: str = "akshare",
) -> pd.DataFrame:
    provider = str(source).strip().lower() or "akshare"
    cache_key = (str(symbol).strip(), provider)
    cached = _US_INDEX_MEM_CACHE.get(cache_key)
    if cached is not None and not cached.empty:
        return _slice_date_range(cached, start=start, end=end)

    if provider != "akshare":
        raise DataError(f"{symbol}: unsupported US index source: {source}")

    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataError("akshare is not installed, run: pip install akshare") from exc

    try:
        raw = ak.index_us_stock_sina(symbol=str(symbol))
    except Exception as exc:  # pragma: no cover - network/provider dependent
        raise DataError(f"{symbol}: akshare US index request failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(f"{symbol}: akshare returned no US index rows")

    normalized = _normalize_external_index_columns(raw, symbol=str(symbol).strip())
    if normalized.empty:
        raise DataError(f"{symbol}: US index rows are empty after normalization")
    _US_INDEX_MEM_CACHE[cache_key] = normalized
    return _slice_date_range(normalized, start=start, end=end)


def fetch_us_etf_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
    source: str = "akshare",
) -> pd.DataFrame:
    provider = str(source).strip().lower() or "akshare"
    cache_key = (str(symbol).strip().upper(), provider)
    cached = _US_ETF_MEM_CACHE.get(cache_key)
    if cached is not None and not cached.empty:
        return _slice_date_range(cached, start=start, end=end)

    if provider != "akshare":
        raise DataError(f"{symbol}: unsupported US ETF source: {source}")

    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataError("akshare is not installed, run: pip install akshare") from exc

    try:
        raw = ak.stock_us_daily(symbol=str(symbol).strip().upper(), adjust="qfq")
    except Exception as exc:  # pragma: no cover - network/provider dependent
        raise DataError(f"{symbol}: akshare US ETF request failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(f"{symbol}: akshare returned no US ETF rows")

    normalized = _normalize_external_index_columns(raw, symbol=str(symbol).strip().upper())
    if normalized.empty:
        raise DataError(f"{symbol}: US ETF rows are empty after normalization")
    _US_ETF_MEM_CACHE[cache_key] = normalized
    return _slice_date_range(normalized, start=start, end=end)


def fetch_cn_etf_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
    source: str = "akshare",
) -> pd.DataFrame:
    provider = str(source).strip().lower() or "akshare"
    cache_key = str(symbol).strip()
    cached = _CN_ETF_MEM_CACHE.get(cache_key)
    if cached is not None and not cached.empty:
        return _slice_date_range(cached, start=start, end=end)

    if provider != "akshare":
        raise DataError(f"{symbol}: unsupported CN ETF source: {source}")

    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataError("akshare is not installed, run: pip install akshare") from exc

    try:
        raw = ak.fund_etf_hist_em(
            symbol=str(symbol).strip(),
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust="qfq",
        )
    except Exception as exc:  # pragma: no cover - network/provider dependent
        raise DataError(f"{symbol}: akshare CN ETF request failed: {exc}") from exc

    if raw is None or raw.empty:
        raise DataError(f"{symbol}: akshare returned no CN ETF rows")

    frame = pd.DataFrame(
        {
            "date": _pick_column(raw, ["日期", "date"]),
            "open": _pick_column(raw, ["开盘", "open"]),
            "high": _pick_column(raw, ["最高", "high"]),
            "low": _pick_column(raw, ["最低", "low"]),
            "close": _pick_column(raw, ["收盘", "close"]),
            "volume": _pick_column(raw, ["成交量", "volume"]),
        }
    )
    amount_col = _pick_column(raw, ["成交额", "amount"], required=False)
    if amount_col is not None:
        frame["amount"] = amount_col

    normalized = _normalize_external_index_columns(frame, symbol=str(symbol).strip())
    if normalized.empty:
        raise DataError(f"{symbol}: CN ETF rows are empty after normalization")
    _CN_ETF_MEM_CACHE[cache_key] = normalized
    return _slice_date_range(normalized, start=start, end=end)


def fetch_baostock_daily(
    symbol: str,
    start: str = "2010-01-01",
    end: str = "2099-12-31",
) -> pd.DataFrame:
    try:
        import baostock as bs
    except Exception as exc:  # pragma: no cover - optional dependency
        raise DataError("baostock is not installed, run: pip install baostock") from exc

    try:
        info = normalize_symbol(symbol)
    except SymbolError as exc:
        raise DataError(str(exc)) from exc

    login = bs.login()
    if str(login.error_code) != "0":
        raise DataError(f"{symbol}: baostock login failed: {login.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            f"{info.exchange.lower()}.{info.code}",
            "date,open,high,low,close,volume,amount",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag="2",
        )
        if str(rs.error_code) != "0":
            raise DataError(f"{symbol}: baostock query failed: {rs.error_msg}")

        rows: List[List[str]] = []
        while rs.next():
            rows.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows:
        raise DataError(f"{symbol}: baostock returned no daily bars")

    raw = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return _normalize_daily_columns(raw, symbol=symbol)


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


def _parse_source_chain(source: str) -> List[str]:
    text = str(source).strip().lower()
    if not text:
        raise DataError("source is empty")
    if text == "auto":
        return list(AUTO_SOURCE_CHAIN)

    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        raise DataError(f"invalid source value: {source}")

    unknown = [p for p in parts if p not in SUPPORTED_SOURCES]
    if unknown:
        valid = ", ".join(sorted(SUPPORTED_SOURCES))
        raise DataError(f"unsupported source(s): {', '.join(unknown)}; valid: {valid}")

    out: List[str] = []
    for part in parts:
        if part == "auto":
            for item in AUTO_SOURCE_CHAIN:
                if item not in out:
                    out.append(item)
            continue
        if part not in out:
            out.append(part)
    return out


def _load_single_source(
    symbol: str,
    source: str,
    data_dir: str | Path,
    start: str,
    end: str,
) -> pd.DataFrame:
    if source == "eastmoney":
        target_end = min(pd.Timestamp(end), pd.Timestamp.today().normalize())
        cached_disk = _load_eastmoney_cache(symbol, data_dir=data_dir)

        need_refresh = True
        if cached_disk is not None and not cached_disk.empty:
            disk_min = pd.Timestamp(cached_disk["date"].min())
            disk_max = pd.Timestamp(cached_disk["date"].max())
            if disk_min <= pd.Timestamp(start) and disk_max >= target_end - pd.Timedelta(days=3):
                need_refresh = False

        df: pd.DataFrame
        if need_refresh:
            try:
                df = fetch_eastmoney_daily(symbol=symbol, start=start, end=end)
                _write_eastmoney_cache(symbol, data_dir=data_dir, df=df)
            except DataError:
                if cached_disk is None or cached_disk.empty:
                    raise
                cached_max = pd.Timestamp(cached_disk["date"].max()).normalize()
                if cached_max < target_end - pd.Timedelta(days=3):
                    raise DataError(
                        f"{symbol}: eastmoney refresh failed and local cache is stale "
                        f"(latest={cached_max.date()}, target_end={target_end.date()})"
                    )
                df = cached_disk
        else:
            df = cached_disk

        out = _slice_date_range(df, start=start, end=end)
        if out.empty:
            raise DataError(f"{symbol}: no data available after date filtering [{start}, {end}]")
        return out

    if source == "akshare":
        out = _slice_date_range(fetch_akshare_daily(symbol=symbol, start=start, end=end), start=start, end=end)
        if out.empty:
            raise DataError(f"{symbol}: no akshare rows in date range [{start}, {end}]")
        return out

    if source == "tushare":
        out = _slice_date_range(fetch_tushare_daily(symbol=symbol, start=start, end=end), start=start, end=end)
        if out.empty:
            raise DataError(f"{symbol}: no tushare rows in date range [{start}, {end}]")
        return out

    if source == "baostock":
        out = _slice_date_range(fetch_baostock_daily(symbol=symbol, start=start, end=end), start=start, end=end)
        if out.empty:
            raise DataError(f"{symbol}: no baostock rows in date range [{start}, {end}]")
        return out

    if source == "local":
        out = _slice_date_range(load_local_daily(symbol=symbol, data_dir=data_dir), start=start, end=end)
        if out.empty:
            raise DataError(f"{symbol}: no local rows in date range [{start}, {end}]")
        return out

    raise DataError(f"unknown data source: {source}")


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

    source_chain = _parse_source_chain(source)
    source_key = ",".join(source_chain)
    cache_key = (norm_symbol, source_key, str(data_dir), start, end)
    cached_mem = _MEM_CACHE.get(cache_key)
    if cached_mem is not None:
        return cached_mem.copy()

    errors: List[str] = []
    for src in source_chain:
        try:
            out = _load_single_source(
                symbol=norm_symbol,
                source=src,
                data_dir=data_dir,
                start=start,
                end=end,
            )
            _MEM_CACHE[cache_key] = out.copy()
            return out.copy()
        except DataError as exc:
            errors.append(f"{src}: {exc}")

    detail = "; ".join(errors) if errors else "no source attempted"
    raise DataError(f"{norm_symbol}: all sources failed ({source_key}). details: {detail}")
