from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.domain.symbols import SymbolError, normalize_symbol


@dataclass(frozen=True)
class MarginSyncResult:
    source_used: str
    market_rows: int
    stock_rows: int
    market_path: str
    stock_path: str
    notes: list[str]


def _safe_symbol(value: str) -> str | None:
    try:
        return normalize_symbol(value).symbol
    except SymbolError:
        text = str(value).strip().upper()
        if text.endswith(".XSHG"):
            return _safe_symbol(text[:6] + ".SH")
        if text.endswith(".XSHE"):
            return _safe_symbol(text[:6] + ".SZ")
        return None


def _pick_col(raw: pd.DataFrame, candidates: list[str]) -> str | None:
    mapper = {str(c).strip().lower(): c for c in raw.columns}
    for key in candidates:
        hit = mapper.get(key.lower())
        if hit is not None:
            return hit
    return None


def _normalize_market_margin(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_col(raw, ["date", "trade_date", "日期", "交易日期", "信用交易日期"])
    fin_bal_col = _pick_col(raw, ["finance_balance", "fin_balance", "融资余额", "rzye"])
    if date_col is None or fin_bal_col is None:
        return pd.DataFrame(columns=["date"])

    sec_bal_col = _pick_col(raw, ["securities_balance", "sec_balance", "融券余额", "rqye"])
    fin_buy_col = _pick_col(raw, ["finance_buy", "fin_buy", "融资买入额", "rzmre"])
    fin_repay_col = _pick_col(raw, ["finance_repay", "fin_repay", "融资偿还额", "rzche"])
    sec_sell_col = _pick_col(raw, ["securities_sell", "sec_sell", "融券卖出量", "融券卖出额", "rqmcl"])
    sec_repay_col = _pick_col(raw, ["securities_repay", "sec_repay", "融券偿还量", "融券偿还额", "rqchl"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "finance_balance": pd.to_numeric(raw[fin_bal_col], errors="coerce"),
            "securities_balance": pd.to_numeric(raw[sec_bal_col], errors="coerce") if sec_bal_col is not None else np.nan,
            "finance_buy": pd.to_numeric(raw[fin_buy_col], errors="coerce") if fin_buy_col is not None else np.nan,
            "finance_repay": pd.to_numeric(raw[fin_repay_col], errors="coerce") if fin_repay_col is not None else np.nan,
            "securities_sell": pd.to_numeric(raw[sec_sell_col], errors="coerce") if sec_sell_col is not None else np.nan,
            "securities_repay": pd.to_numeric(raw[sec_repay_col], errors="coerce") if sec_repay_col is not None else np.nan,
        }
    )
    out = out.dropna(subset=["date", "finance_balance"])
    out = out.sort_values("date").drop_duplicates(subset=["date"])
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out.reset_index(drop=True)


def _normalize_stock_margin(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_col(raw, ["date", "trade_date", "日期", "交易日期", "信用交易日期"])
    symbol_col = _pick_col(raw, ["symbol", "ts_code", "代码", "证券代码", "股票代码"])
    fin_bal_col = _pick_col(raw, ["finance_balance", "fin_balance", "融资余额", "rzye"])
    if date_col is None or symbol_col is None or fin_bal_col is None:
        return pd.DataFrame(columns=["date", "symbol"])

    sec_bal_col = _pick_col(raw, ["securities_balance", "sec_balance", "融券余额", "rqye"])
    fin_buy_col = _pick_col(raw, ["finance_buy", "fin_buy", "融资买入额", "rzmre"])
    fin_repay_col = _pick_col(raw, ["finance_repay", "fin_repay", "融资偿还额", "rzche"])
    sec_sell_col = _pick_col(raw, ["securities_sell", "sec_sell", "融券卖出量", "融券卖出额", "rqmcl"])
    sec_repay_col = _pick_col(raw, ["securities_repay", "sec_repay", "融券偿还量", "融券偿还额", "rqchl"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "symbol": raw[symbol_col].astype(str).map(lambda x: _safe_symbol(x) or ""),
            "finance_balance": pd.to_numeric(raw[fin_bal_col], errors="coerce"),
            "securities_balance": pd.to_numeric(raw[sec_bal_col], errors="coerce") if sec_bal_col is not None else np.nan,
            "finance_buy": pd.to_numeric(raw[fin_buy_col], errors="coerce") if fin_buy_col is not None else np.nan,
            "finance_repay": pd.to_numeric(raw[fin_repay_col], errors="coerce") if fin_repay_col is not None else np.nan,
            "securities_sell": pd.to_numeric(raw[sec_sell_col], errors="coerce") if sec_sell_col is not None else np.nan,
            "securities_repay": pd.to_numeric(raw[sec_repay_col], errors="coerce") if sec_repay_col is not None else np.nan,
        }
    )
    out = out[(out["symbol"] != "") & out["date"].notna() & out["finance_balance"].notna()].copy()
    out = out.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"])
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out.reset_index(drop=True)


def _limit_dates(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    out = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].copy()
    return out.sort_values("date").reset_index(drop=True)


def _ak_call_range(func, start: str, end: str):
    sig = inspect.signature(func)
    params = sig.parameters
    start_x = pd.Timestamp(start).strftime("%Y%m%d")
    end_x = pd.Timestamp(end).strftime("%Y%m%d")
    if "start_date" in params and "end_date" in params:
        return func(start_date=start_x, end_date=end_x)
    if "start" in params and "end" in params:
        return func(start=start_x, end=end_x)
    if "date" in params:
        # Date-by-date API can be very expensive; caller should use a short range only.
        days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
        rows: list[pd.DataFrame] = []
        for d in days:
            try:
                part = func(date=d.strftime("%Y%m%d"))
            except Exception:
                continue
            if isinstance(part, pd.DataFrame) and not part.empty:
                rows.append(part.copy())
            time.sleep(0.01)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)
    return func()


def _fetch_akshare_market(start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    parts: list[pd.DataFrame] = []
    for name in ["stock_margin_sse", "stock_margin_szse"]:
        fn = getattr(ak, name, None)
        if fn is None:
            continue
        try:
            raw = _ak_call_range(fn, start=start, end=end)
        except Exception:
            continue
        frame = _normalize_market_margin(pd.DataFrame(raw))
        if not frame.empty:
            parts.append(frame)

    if not parts:
        return pd.DataFrame(columns=["date"])
    out = pd.concat(parts, ignore_index=True)
    agg_cols = [
        "finance_balance",
        "securities_balance",
        "finance_buy",
        "finance_repay",
        "securities_sell",
        "securities_repay",
    ]
    out = out.groupby("date", as_index=False)[agg_cols].sum(min_count=1)
    return _limit_dates(out, start=start, end=end)


def _fetch_akshare_stock(symbols: Sequence[str], start: str, end: str, max_days: int = 180) -> pd.DataFrame:
    import akshare as ak

    # Most akshare detail APIs are day-by-day; limit range to avoid huge request volume.
    days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
    if len(days) > int(max_days):
        days = days[-int(max_days) :]
    symbols_set = {s for s in (_safe_symbol(x) for x in symbols) if s is not None}

    parts: list[pd.DataFrame] = []
    for name in ["stock_margin_detail_sse", "stock_margin_detail_szse"]:
        fn = getattr(ak, name, None)
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
        except Exception:
            sig = None

        if sig is not None and "date" in sig.parameters and "start_date" not in sig.parameters:
            for d in days:
                try:
                    raw = fn(date=d.strftime("%Y%m%d"))
                except Exception:
                    continue
                frame = _normalize_stock_margin(pd.DataFrame(raw))
                if frame.empty:
                    continue
                frame = frame[frame["symbol"].isin(symbols_set)]
                if not frame.empty:
                    parts.append(frame)
                time.sleep(0.01)
        else:
            try:
                raw = _ak_call_range(fn, start=start, end=end)
            except Exception:
                continue
            frame = _normalize_stock_margin(pd.DataFrame(raw))
            if frame.empty:
                continue
            frame = frame[frame["symbol"].isin(symbols_set)]
            if not frame.empty:
                parts.append(frame)

    if not parts:
        return pd.DataFrame(columns=["date", "symbol"])
    out = pd.concat(parts, ignore_index=True)
    agg_cols = [
        "finance_balance",
        "securities_balance",
        "finance_buy",
        "finance_repay",
        "securities_sell",
        "securities_repay",
    ]
    out = out.groupby(["date", "symbol"], as_index=False)[agg_cols].sum(min_count=1)
    out = out.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"])
    return _limit_dates(out, start=start, end=end)


def _fetch_tushare_market(start: str, end: str, token: str) -> pd.DataFrame:
    import tushare as ts

    if not token.strip():
        token = str(os.getenv("TUSHARE_TOKEN", "")).strip()
    if not token:
        raise RuntimeError("tushare token missing")

    pro = ts.pro_api(token)
    start_x = pd.Timestamp(start).strftime("%Y%m%d")
    end_x = pd.Timestamp(end).strftime("%Y%m%d")
    parts: list[pd.DataFrame] = []
    for ex in ["SSE", "SZSE"]:
        try:
            raw = pro.margin(exchange_id=ex, start_date=start_x, end_date=end_x)
        except Exception:
            continue
        frame = _normalize_market_margin(pd.DataFrame(raw))
        if not frame.empty:
            parts.append(frame)
    if not parts:
        return pd.DataFrame(columns=["date"])
    out = pd.concat(parts, ignore_index=True)
    agg_cols = [
        "finance_balance",
        "securities_balance",
        "finance_buy",
        "finance_repay",
        "securities_sell",
        "securities_repay",
    ]
    out = out.groupby("date", as_index=False)[agg_cols].sum(min_count=1)
    return _limit_dates(out, start=start, end=end)


def _fetch_tushare_stock(symbols: Sequence[str], start: str, end: str, token: str, sleep_ms: int = 80) -> pd.DataFrame:
    import tushare as ts

    if not token.strip():
        token = str(os.getenv("TUSHARE_TOKEN", "")).strip()
    if not token:
        raise RuntimeError("tushare token missing")

    pro = ts.pro_api(token)
    symbols_set = {s for s in (_safe_symbol(x) for x in symbols) if s is not None}
    trade_days = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="B")
    parts: list[pd.DataFrame] = []
    total_days = int(len(trade_days))
    for idx, day in enumerate(trade_days, start=1):
        try:
            raw = pro.margin_detail(trade_date=day.strftime("%Y%m%d"))
        except Exception:
            continue
        frame = _normalize_stock_margin(pd.DataFrame(raw))
        if not frame.empty:
            frame = frame[frame["symbol"].isin(symbols_set)]
            if not frame.empty:
                parts.append(frame)
        if idx == 1 or idx % 50 == 0 or idx == total_days:
            rows = int(sum(len(part) for part in parts)) if parts else 0
            print(f"[SYNC] Tushare margin detail {idx}/{total_days} trade days collected_rows={rows}")
        if int(sleep_ms) > 0:
            time.sleep(float(sleep_ms) / 1000.0)

    if not parts:
        return pd.DataFrame(columns=["date", "symbol"])
    out = pd.concat(parts, ignore_index=True)
    agg_cols = [
        "finance_balance",
        "securities_balance",
        "finance_buy",
        "finance_repay",
        "securities_sell",
        "securities_repay",
    ]
    out = out.groupby(["date", "symbol"], as_index=False)[agg_cols].sum(min_count=1)
    out = out.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"])
    return _limit_dates(out, start=start, end=end)


def _try_source(
    source: str,
    *,
    symbols: Sequence[str],
    start: str,
    end: str,
    tushare_token: str,
    sleep_ms: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    notes: list[str] = []
    s = str(source).strip().lower()
    if s == "akshare":
        market = _fetch_akshare_market(start=start, end=end)
        stock = _fetch_akshare_stock(symbols=symbols, start=start, end=end)
        if stock.empty:
            notes.append("akshare stock detail empty (api/range limits), market margin still written.")
        return market, stock, notes
    if s == "tushare":
        market = _fetch_tushare_market(start=start, end=end, token=tushare_token)
        stock = _fetch_tushare_stock(symbols=symbols, start=start, end=end, token=tushare_token, sleep_ms=sleep_ms)
        return market, stock, notes
    raise RuntimeError(f"unsupported margin source: {source}")


def sync_margin_data(
    *,
    source: str,
    symbols: Sequence[str],
    start: str,
    end: str,
    market_out: str,
    stock_out: str,
    tushare_token: str = "",
    sleep_ms: int = 80,
) -> MarginSyncResult:
    source_order: list[str]
    text = str(source).strip().lower()
    if text in {"", "auto"}:
        source_order = ["akshare", "tushare"]
    elif "," in text:
        source_order = [x.strip() for x in text.split(",") if x.strip()]
    else:
        source_order = [text]

    market_df = pd.DataFrame(columns=["date"])
    stock_df = pd.DataFrame(columns=["date", "symbol"])
    notes: list[str] = []
    market_source = ""
    stock_source = ""
    for src in source_order:
        try:
            m, s, local_notes = _try_source(
                src,
                symbols=symbols,
                start=start,
                end=end,
                tushare_token=tushare_token,
                sleep_ms=sleep_ms,
            )
        except Exception as exc:
            notes.append(f"{src} failed: {exc}")
            continue
        notes.extend(local_notes)
        if market_df.empty and not m.empty:
            market_df = m
            market_source = src
        if stock_df.empty and not s.empty:
            stock_df = s
            stock_source = src
        if not market_df.empty and not stock_df.empty:
            break

    if market_df.empty:
        detail = "; ".join(notes) if notes else "no source returned market rows"
        raise RuntimeError(f"margin sync failed: no market margin rows from all sources ({detail})")

    market_path = Path(market_out)
    stock_path = Path(stock_out)
    market_path.parent.mkdir(parents=True, exist_ok=True)
    stock_path.parent.mkdir(parents=True, exist_ok=True)

    market_df = market_df.sort_values("date").drop_duplicates(subset=["date"])
    market_df.to_csv(market_path, index=False)

    if stock_df.empty:
        # Keep a valid schema even when source cannot provide stock detail.
        notes.append("all margin sources returned empty stock detail; stock file kept as empty schema.")
        stock_df = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "finance_balance",
                "securities_balance",
                "finance_buy",
                "finance_repay",
                "securities_sell",
                "securities_repay",
            ]
        )
    stock_df = stock_df.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"])
    stock_df.to_csv(stock_path, index=False)

    used_parts = [part for part in [market_source, stock_source] if part]
    if market_source and stock_source and market_source != stock_source:
        notes.append(
            f"market margin sourced from {market_source}; stock detail sourced from {stock_source}."
        )

    return MarginSyncResult(
        source_used="+".join(dict.fromkeys(used_parts)) or "unknown",
        market_rows=int(len(market_df)),
        stock_rows=int(len(stock_df)),
        market_path=str(market_path),
        stock_path=str(stock_path),
        notes=notes,
    )
