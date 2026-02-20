from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.domain.symbols import SymbolError, normalize_symbol


_MARKET_CACHE: Dict[tuple[str, str, str], tuple[pd.DataFrame, list[str], list[str]]] = {}
_STOCK_RAW_CACHE: Dict[str, pd.DataFrame] = {}
_STOCK_CACHE: Dict[tuple[str, str, str, str], tuple[pd.DataFrame, list[str], list[str]]] = {}

_MARKET_FEATURE_CANDIDATES = [
    "mrg_mkt_fin_balance_z20",
    "mrg_mkt_fin_balance_chg5",
    "mrg_mkt_sec_balance_z20",
    "mrg_mkt_fin_net_buy_z20",
    "mrg_mkt_sec_net_sell_z20",
    "mrg_mkt_fin_sec_spread_chg5",
]

_STOCK_FEATURE_CANDIDATES = [
    "mrg_stk_fin_balance_z20",
    "mrg_stk_fin_balance_chg5",
    "mrg_stk_sec_balance_z20",
    "mrg_stk_fin_net_buy_z20",
    "mrg_stk_sec_net_sell_z20",
    "mrg_stk_fin_sec_spread_chg5",
]


def _safe_symbol(value: str) -> str | None:
    try:
        return normalize_symbol(value).symbol
    except SymbolError:
        return None


def _pick_col(raw: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in raw.columns}
    for key in candidates:
        hit = lower_map.get(key.lower())
        if hit is not None:
            return hit
    return None


def _normalize_margin_base(raw: pd.DataFrame) -> pd.DataFrame:
    date_col = _pick_col(raw, ["date", "trade_date", "日期", "交易日期", "信用交易日期"])
    fin_balance_col = _pick_col(raw, ["finance_balance", "fin_balance", "融资余额"])
    sec_balance_col = _pick_col(raw, ["securities_balance", "sec_balance", "融券余额"])
    fin_buy_col = _pick_col(raw, ["finance_buy", "fin_buy", "融资买入额"])
    fin_repay_col = _pick_col(raw, ["finance_repay", "fin_repay", "融资偿还额"])
    sec_sell_col = _pick_col(raw, ["securities_sell", "sec_sell", "融券卖出量", "融券卖出额"])
    sec_repay_col = _pick_col(raw, ["securities_repay", "sec_repay", "融券偿还量", "融券偿还额"])

    if date_col is None or fin_balance_col is None:
        return pd.DataFrame(columns=["date"])

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "fin_balance": pd.to_numeric(raw[fin_balance_col], errors="coerce"),
            "sec_balance": pd.to_numeric(raw[sec_balance_col], errors="coerce") if sec_balance_col is not None else np.nan,
            "fin_buy": pd.to_numeric(raw[fin_buy_col], errors="coerce") if fin_buy_col is not None else np.nan,
            "fin_repay": pd.to_numeric(raw[fin_repay_col], errors="coerce") if fin_repay_col is not None else np.nan,
            "sec_sell": pd.to_numeric(raw[sec_sell_col], errors="coerce") if sec_sell_col is not None else np.nan,
            "sec_repay": pd.to_numeric(raw[sec_repay_col], errors="coerce") if sec_repay_col is not None else np.nan,
        }
    )
    out = out.dropna(subset=["date", "fin_balance"]).sort_values("date").drop_duplicates(subset=["date"])
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out.reset_index(drop=True)


def _to_features(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy().sort_values("date").reset_index(drop=True)
    if out.empty:
        return pd.DataFrame(columns=["date"])

    fin_bal = out["fin_balance"].astype(float)
    sec_bal = out["sec_balance"].astype(float)
    fin_net = out["fin_buy"].astype(float) - out["fin_repay"].astype(float)
    sec_net = out["sec_sell"].astype(float) - out["sec_repay"].astype(float)

    fin_std = fin_bal.rolling(20).std().replace(0.0, np.nan)
    sec_std = sec_bal.rolling(20).std().replace(0.0, np.nan)
    fin_net_std = fin_net.rolling(20).std().replace(0.0, np.nan)
    sec_net_std = sec_net.rolling(20).std().replace(0.0, np.nan)

    out[f"{prefix}_fin_balance_z20"] = (fin_bal - fin_bal.rolling(20).mean()) / (fin_std + 1e-9)
    out[f"{prefix}_fin_balance_chg5"] = fin_bal / (fin_bal.shift(5) + 1e-9) - 1.0
    out[f"{prefix}_sec_balance_z20"] = (sec_bal - sec_bal.rolling(20).mean()) / (sec_std + 1e-9)
    out[f"{prefix}_fin_net_buy_z20"] = (fin_net - fin_net.rolling(20).mean()) / (fin_net_std + 1e-9)
    out[f"{prefix}_sec_net_sell_z20"] = (sec_net - sec_net.rolling(20).mean()) / (sec_net_std + 1e-9)
    out[f"{prefix}_fin_sec_spread_chg5"] = (
        fin_bal / (fin_bal.shift(5) + 1e-9) - sec_bal / (sec_bal.shift(5) + 1e-9)
    )

    feat_cols = [c for c in out.columns if c.startswith(prefix)]
    for col in feat_cols:
        out[col] = np.clip(pd.to_numeric(out[col], errors="coerce"), -12.0, 12.0)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out[["date"] + feat_cols].copy()


def _select_columns(
    frame: pd.DataFrame,
    candidates: list[str],
    *,
    min_valid_ratio: float,
    min_valid_points: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    notes: list[str] = []
    if frame.empty:
        return pd.DataFrame(columns=["date"]), [], notes

    selected: list[str] = []
    for col in candidates:
        if col not in frame.columns:
            continue
        valid_n = int(frame[col].notna().sum())
        valid_ratio = float(frame[col].notna().mean())
        if valid_n >= int(min_valid_points) and valid_ratio >= float(min_valid_ratio):
            selected.append(col)
        else:
            notes.append(f"margin column dropped {col}: valid={valid_n}, ratio={valid_ratio:.2f}")

    if not selected:
        return frame[["date"]].copy(), [], notes

    out = frame[["date"] + selected].copy().sort_values("date")
    for col in selected:
        out[col] = out[col].ffill()
        if out[col].isna().any():
            out[col] = out[col].fillna(float(out[col].median(skipna=True)))
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out, selected, notes


def build_market_margin_features(
    *,
    margin_market_file: str,
    start: str,
    end: str,
    min_valid_ratio: float = 0.55,
    min_valid_points: int = 120,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    key = (str(margin_market_file), str(start), str(end))
    cached = _MARKET_CACHE.get(key)
    if cached is not None:
        return cached[0].copy(), list(cached[1]), list(cached[2])

    path = Path(margin_market_file)
    if not str(margin_market_file).strip() or not path.exists():
        result = (pd.DataFrame(columns=["date"]), [], [f"market margin file missing: {margin_market_file}"])
        _MARKET_CACHE[key] = (result[0].copy(), list(result[1]), list(result[2]))
        return result

    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        result = (pd.DataFrame(columns=["date"]), [], [f"market margin read failed: {exc}"])
        _MARKET_CACHE[key] = (result[0].copy(), list(result[1]), list(result[2]))
        return result

    base = _normalize_margin_base(raw)
    if base.empty:
        result = (pd.DataFrame(columns=["date"]), [], ["market margin file has no usable rows"])
        _MARKET_CACHE[key] = (result[0].copy(), list(result[1]), list(result[2]))
        return result

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    base = base[(base["date"] >= start_ts) & (base["date"] <= end_ts)].copy()
    feat = _to_features(base, "mrg_mkt")
    out, cols, notes = _select_columns(
        feat,
        _MARKET_FEATURE_CANDIDATES,
        min_valid_ratio=min_valid_ratio,
        min_valid_points=min_valid_points,
    )
    _MARKET_CACHE[key] = (out.copy(), list(cols), list(notes))
    return out, cols, notes


def _load_stock_raw(path: Path) -> pd.DataFrame:
    cache_key = str(path.resolve())
    cached = _STOCK_RAW_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    raw = pd.read_csv(path)
    symbol_col = _pick_col(raw, ["symbol", "ts_code", "代码", "股票代码"])
    date_col = _pick_col(raw, ["date", "trade_date", "日期", "交易日期", "信用交易日期"])
    fin_balance_col = _pick_col(raw, ["finance_balance", "fin_balance", "融资余额"])
    if symbol_col is None or date_col is None or fin_balance_col is None:
        out = pd.DataFrame(columns=["symbol", "date"])
        _STOCK_RAW_CACHE[cache_key] = out
        return out.copy()

    sec_balance_col = _pick_col(raw, ["securities_balance", "sec_balance", "融券余额"])
    fin_buy_col = _pick_col(raw, ["finance_buy", "fin_buy", "融资买入额"])
    fin_repay_col = _pick_col(raw, ["finance_repay", "fin_repay", "融资偿还额"])
    sec_sell_col = _pick_col(raw, ["securities_sell", "sec_sell", "融券卖出量", "融券卖出额"])
    sec_repay_col = _pick_col(raw, ["securities_repay", "sec_repay", "融券偿还量", "融券偿还额"])

    out = pd.DataFrame(
        {
            "symbol": raw[symbol_col].astype(str).map(lambda x: _safe_symbol(x) or ""),
            "date": pd.to_datetime(raw[date_col], errors="coerce"),
            "fin_balance": pd.to_numeric(raw[fin_balance_col], errors="coerce"),
            "sec_balance": pd.to_numeric(raw[sec_balance_col], errors="coerce") if sec_balance_col is not None else np.nan,
            "fin_buy": pd.to_numeric(raw[fin_buy_col], errors="coerce") if fin_buy_col is not None else np.nan,
            "fin_repay": pd.to_numeric(raw[fin_repay_col], errors="coerce") if fin_repay_col is not None else np.nan,
            "sec_sell": pd.to_numeric(raw[sec_sell_col], errors="coerce") if sec_sell_col is not None else np.nan,
            "sec_repay": pd.to_numeric(raw[sec_repay_col], errors="coerce") if sec_repay_col is not None else np.nan,
        }
    )
    out = out[(out["symbol"] != "") & out["date"].notna() & out["fin_balance"].notna()].copy()
    out = out.sort_values(["symbol", "date"]).drop_duplicates(subset=["symbol", "date"]).reset_index(drop=True)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    _STOCK_RAW_CACHE[cache_key] = out.copy()
    return out


def build_stock_margin_features(
    *,
    margin_stock_file: str,
    symbol: str,
    start: str,
    end: str,
    min_valid_ratio: float = 0.45,
    min_valid_points: int = 80,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    norm_symbol = _safe_symbol(symbol)
    if norm_symbol is None:
        return pd.DataFrame(columns=["date"]), [], [f"invalid symbol for margin: {symbol}"]

    key = (str(margin_stock_file), norm_symbol, str(start), str(end))
    cached = _STOCK_CACHE.get(key)
    if cached is not None:
        return cached[0].copy(), list(cached[1]), list(cached[2])

    path = Path(margin_stock_file)
    if not str(margin_stock_file).strip() or not path.exists():
        result = (pd.DataFrame(columns=["date"]), [], [f"stock margin file missing: {margin_stock_file}"])
        _STOCK_CACHE[key] = (result[0].copy(), list(result[1]), list(result[2]))
        return result

    try:
        raw = _load_stock_raw(path)
    except Exception as exc:
        result = (pd.DataFrame(columns=["date"]), [], [f"stock margin read failed: {exc}"])
        _STOCK_CACHE[key] = (result[0].copy(), list(result[1]), list(result[2]))
        return result

    if raw.empty:
        result = (pd.DataFrame(columns=["date"]), [], ["stock margin file has no usable rows"])
        _STOCK_CACHE[key] = (result[0].copy(), list(result[1]), list(result[2]))
        return result

    frame = raw[raw["symbol"] == norm_symbol].copy()
    if frame.empty:
        result = (pd.DataFrame(columns=["date"]), [], [f"stock margin missing symbol: {norm_symbol}"])
        _STOCK_CACHE[key] = (result[0].copy(), list(result[1]), list(result[2]))
        return result

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    frame = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)].copy()
    feat = _to_features(frame.drop(columns=["symbol"]), "mrg_stk")
    out, cols, notes = _select_columns(
        feat,
        _STOCK_FEATURE_CANDIDATES,
        min_valid_ratio=min_valid_ratio,
        min_valid_points=min_valid_points,
    )
    _STOCK_CACHE[key] = (out.copy(), list(cols), list(notes))
    return out, cols, notes
