from __future__ import annotations

import numpy as np
import pandas as pd

from src.application.v2_contracts import MarketFactsState


def clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return out


def signal_unit(value: object, scale: float) -> float:
    denom = max(1e-9, float(scale))
    return clip(safe_float(value, 0.0) / denom, -1.0, 1.0)


def is_main_board_symbol(symbol: object) -> bool:
    text = str(symbol or "").strip().upper()
    if not text or "." not in text:
        return False
    code, market = text.split(".", 1)
    if market == "SH":
        return code.startswith(("600", "601", "603", "605"))
    if market == "SZ":
        return code.startswith(("000", "001", "002"))
    return False


def next_session_label(as_of_date: object) -> str:
    ts = pd.Timestamp(as_of_date)
    if pd.isna(ts):
        return ""
    return str((ts + pd.offsets.BDay(1)).date())


def market_facts_from_row(row: pd.Series) -> MarketFactsState:
    coverage = int(round(float(safe_float(row.get("breadth_coverage", 0.0), 0.0))))
    return MarketFactsState(
        sample_coverage=max(0, coverage),
        advancers=int(round(float(safe_float(row.get("breadth_advancers", 0.0), 0.0)))),
        decliners=int(round(float(safe_float(row.get("breadth_decliners", 0.0), 0.0)))),
        flats=int(round(float(safe_float(row.get("breadth_flats", 0.0), 0.0)))),
        limit_up_count=int(round(float(safe_float(row.get("breadth_limit_up_count", 0.0), 0.0)))),
        limit_down_count=int(round(float(safe_float(row.get("breadth_limit_down_count", 0.0), 0.0)))),
        new_high_count=int(round(float(safe_float(row.get("breadth_new_high_count", 0.0), 0.0)))),
        new_low_count=int(round(float(safe_float(row.get("breadth_new_low_count", 0.0), 0.0)))),
        median_return=float(safe_float(row.get("breadth_median_return", 0.0), 0.0)),
        sample_amount=float(safe_float(row.get("breadth_sample_amount", 0.0), 0.0)),
        amount_z20=float(safe_float(row.get("breadth_amount_z20", 0.0), 0.0)),
    )


def is_actionable_status(status: str) -> bool:
    return str(status) not in {"halted", "delisted"}


def status_tradeability_limit(status: str) -> float:
    status = str(status)
    if status in {"halted", "delisted"}:
        return 0.0
    if status == "data_insufficient":
        return 0.35
    return 1.0
