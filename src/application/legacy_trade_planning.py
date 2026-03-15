from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.application.config import DailyConfig
from src.domain.entities import BlendedRow, Security, TradeAction
from src.domain.symbols import normalize_symbol
from src.infrastructure.market_data import load_symbol_daily


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        text = str(value).strip()
        if not text or text.lower() in {"na", "nan", "none"}:
            return np.nan
        out = float(text)
        return float(out) if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def safe_weight(value: Any) -> float:
    raw = safe_float(value)
    if pd.isna(raw):
        return np.nan
    weight = float(raw)
    if 1.0 < weight <= 100.0:
        weight = weight / 100.0
    return float(np.clip(weight, 0.0, 1.0))


def to_symbol(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return normalize_symbol(text).symbol
    except Exception:
        return ""


def read_positions_records(path_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path_text = str(path_text).strip()
    if not path_text:
        return [], {}
    path = Path(path_text)
    if not path.exists():
        print(f"[WARN] positions file not found: {path}")
        return [], {}
    if path.suffix.lower() == ".csv":
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] failed to read positions csv: {path} ({exc})")
            return [], {}
        if frame.empty:
            return [], {}
        return frame.to_dict("records"), {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read positions json: {path} ({exc})")
        return [], {}
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, dict)], {}
    if not isinstance(payload, dict):
        return [], {}
    for key in ("positions", "stocks", "rows"):
        part = payload.get(key)
        if isinstance(part, list):
            return [dict(x) for x in part if isinstance(x, dict)], dict(payload)
    return [], dict(payload)


def load_latest_close_map(
    *,
    symbols: list[str],
    config: DailyConfig,
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol in sorted(set(symbols)):
        if not symbol:
            continue
        try:
            raw = load_symbol_daily(
                symbol=symbol,
                source=config.source,
                data_dir=config.data_dir,
                start=config.start,
                end=config.end,
            )
            history = raw[raw["date"] <= as_of_date].sort_values("date")
            if history.empty:
                history = raw.sort_values("date")
            if history.empty:
                continue
            close = safe_float(history.iloc[-1].get("close"))
            if not pd.isna(close) and close > 0:
                out[symbol] = float(close)
        except Exception:
            continue
    return out


def build_trade_actions(
    *,
    config: DailyConfig,
    as_of_date: pd.Timestamp,
    blended_rows: list[BlendedRow],
    current_holdings: list[Security],
) -> tuple[str, float, list[TradeAction]]:
    target_rows = [row for row in blended_rows if float(row.suggested_weight) > 1e-9]
    target_weight_map = {row.symbol: float(row.suggested_weight) for row in target_rows}
    name_map = {row.symbol: row.name for row in blended_rows}
    for security in current_holdings:
        symbol = to_symbol(security.symbol)
        if symbol:
            name_map.setdefault(symbol, security.name)

    records, meta = read_positions_records(config.positions_file)
    parsed_records: list[dict[str, Any]] = []
    for rec in records:
        symbol = to_symbol(rec.get("symbol") or rec.get("ts_code") or rec.get("code"))
        if not symbol:
            continue
        parsed_records.append(
            {
                "symbol": symbol,
                "name": str(rec.get("name") or rec.get("security_name") or name_map.get(symbol, symbol)),
                "weight": safe_weight(rec.get("current_weight") if rec.get("current_weight") is not None else rec.get("weight")),
                "shares": safe_float(rec.get("shares") if rec.get("shares") is not None else rec.get("quantity")),
                "market_value": safe_float(
                    rec.get("market_value") if rec.get("market_value") is not None else rec.get("value")
                ),
            }
        )
        name_map[symbol] = parsed_records[-1]["name"]

    fallback_symbols = [to_symbol(sec.symbol) for sec in current_holdings]
    fallback_symbols = [symbol for symbol in fallback_symbols if symbol]
    symbols_for_price = sorted(
        set(target_weight_map.keys()) | {item["symbol"] for item in parsed_records} | set(fallback_symbols)
    )
    close_map = load_latest_close_map(symbols=symbols_for_price, config=config, as_of_date=as_of_date)

    current_weight_map: dict[str, float] = {}
    current_shares_map: dict[str, float] = {}
    value_map: dict[str, float] = {}
    weight_basis_used = False
    for item in parsed_records:
        symbol = item["symbol"]
        shares = float(item["shares"])
        if not pd.isna(shares) and shares > 0:
            current_shares_map[symbol] = shares
        weight = float(item["weight"])
        if not pd.isna(weight):
            current_weight_map[symbol] = weight
            weight_basis_used = True
        market_value = float(item["market_value"])
        if not pd.isna(market_value) and market_value >= 0:
            value_map[symbol] = market_value
            continue
        price = close_map.get(symbol, np.nan)
        if not pd.isna(shares) and shares > 0 and not pd.isna(price) and price > 0:
            value_map[symbol] = float(shares * price)

    plan_basis = ""
    plan_nav = float(np.nan)
    if weight_basis_used and current_weight_map:
        total_weight = float(sum(max(0.0, weight) for weight in current_weight_map.values()))
        if total_weight > 1e-9:
            if total_weight > 1.0 + 1e-6:
                current_weight_map = {
                    symbol: float(max(0.0, weight) / total_weight)
                    for symbol, weight in current_weight_map.items()
                }
                plan_basis = "positions_file(weight, normalized)"
            else:
                plan_basis = "positions_file(weight)"
        else:
            current_weight_map = {}

    if not current_weight_map and value_map:
        meta_cash = safe_float(meta.get("cash"))
        if pd.isna(meta_cash):
            meta_cash = 0.0
        nav_meta = safe_float(meta.get("portfolio_nav"))
        if pd.isna(nav_meta):
            nav_meta = safe_float(meta.get("total_nav"))
        total_mv = float(sum(max(0.0, value) for value in value_map.values()))
        if pd.isna(nav_meta) or nav_meta <= 0:
            nav_meta = total_mv + float(meta_cash)
        if nav_meta > 0 and total_mv > 0:
            current_weight_map = {symbol: float(value / nav_meta) for symbol, value in value_map.items()}
            plan_nav = float(nav_meta)
            plan_basis = "positions_file(market_value/shares)"

    if not current_weight_map:
        if fallback_symbols:
            equal_weight = 1.0 / float(len(fallback_symbols))
            current_weight_map = {symbol: equal_weight for symbol in fallback_symbols}
            plan_basis = "watchlist_equal_weight(fallback)"
        else:
            plan_basis = "target_only(no_current_positions)"

    if config.portfolio_nav > 0:
        plan_nav = float(config.portfolio_nav)

    lot_size = max(1, int(config.trade_lot_size))
    trade_threshold = max(0.0, float(config.min_weight_change_to_trade))
    symbols_union = sorted(set(current_weight_map.keys()) | set(target_weight_map.keys()))
    actions: list[TradeAction] = []
    for symbol in symbols_union:
        current_weight = float(current_weight_map.get(symbol, 0.0))
        target_weight = float(target_weight_map.get(symbol, 0.0))
        delta_weight = float(target_weight - current_weight)
        if delta_weight > trade_threshold:
            action = "BUY"
        elif delta_weight < -trade_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        est_price = float(close_map.get(symbol, np.nan))
        est_delta_value = float(np.nan)
        est_delta_shares = float(np.nan)
        est_delta_lots = float(np.nan)
        note_parts: list[str] = []
        if not pd.isna(plan_nav) and plan_nav > 0:
            est_delta_value = float(delta_weight * plan_nav)
        if not pd.isna(est_delta_value) and not pd.isna(est_price) and est_price > 0:
            raw_shares = float(est_delta_value / est_price)
            sign = 1.0 if raw_shares >= 0 else -1.0
            lots = float(np.floor(abs(raw_shares) / float(lot_size)))
            est_delta_lots = sign * lots
            est_delta_shares = sign * lots * float(lot_size)
            if action == "HOLD":
                est_delta_lots = 0.0
                est_delta_shares = 0.0
            current_shares = float(current_shares_map.get(symbol, np.nan))
            if action == "SELL" and not pd.isna(current_shares) and current_shares > 0 and est_delta_shares < 0:
                max_sell_lots = float(np.floor(current_shares / float(lot_size)))
                capped = -min(abs(est_delta_lots), max_sell_lots)
                est_delta_lots = float(capped)
                est_delta_shares = float(capped * lot_size)
                if abs(capped) < abs(lots):
                    note_parts.append("sell_capped_by_current_shares")
        else:
            if pd.isna(plan_nav) or plan_nav <= 0:
                note_parts.append("missing_portfolio_nav")
            if pd.isna(est_price):
                note_parts.append("missing_price")
        if action == "HOLD" and abs(delta_weight) > 1e-9:
            note_parts.append("below_min_weight_change")
        actions.append(
            TradeAction(
                symbol=symbol,
                name=name_map.get(symbol, symbol),
                action=action,
                current_weight=current_weight,
                target_weight=target_weight,
                delta_weight=delta_weight,
                est_price=est_price,
                est_delta_value=est_delta_value,
                est_delta_shares=est_delta_shares,
                est_delta_lots=est_delta_lots,
                current_shares=float(current_shares_map.get(symbol, np.nan)),
                note=";".join(note_parts),
            )
        )

    actions.sort(key=lambda item: (abs(float(item.delta_weight)), float(item.target_weight)), reverse=True)
    return plan_basis, plan_nav, actions
