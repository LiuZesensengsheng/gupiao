from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.application.v2_contracts import CapitalFlowState, InfoItem, MacroContextState
from src.domain.news import normalize_direction


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return out


def _hashable_path(path_like: object) -> str:
    text = str(path_like or "").strip()
    if not text:
        return ""
    return str(Path(text))


def _load_optional_csv_payload(path_like: object) -> pd.DataFrame:
    text = str(path_like or "").strip()
    if not text:
        return pd.DataFrame()
    path = Path(text)
    if not path.exists():
        return pd.DataFrame()
    if path.is_file():
        frame = pd.read_csv(path)
        if frame.empty:
            return frame
        frame["__source_file"] = str(path)
        return frame
    files = sorted(item for item in path.rglob("*.csv") if item.is_file())
    frames: list[pd.DataFrame] = []
    for file in files:
        frame = pd.read_csv(file)
        if frame.empty:
            continue
        frame["__source_file"] = str(file)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_window(frame: pd.DataFrame, *, as_of_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return pd.DataFrame()
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.dropna(subset=["date"])
    if dated.empty:
        return dated
    dated["date"] = dated["date"].dt.normalize()
    mask = (dated["date"] <= as_of_date.normalize()) & (
        (as_of_date.normalize() - dated["date"]).dt.days <= max(0, int(lookback_days))
    )
    return dated.loc[mask].sort_values("date")


def _latest_row(frame: pd.DataFrame) -> pd.Series | None:
    if frame.empty:
        return None
    return frame.sort_values("date").iloc[-1]


def build_capital_flow_state(
    *,
    as_of_date: pd.Timestamp,
    capital_flow_file: object,
    lookback_days: int,
    info_items: Iterable[InfoItem] = (),
) -> CapitalFlowState:
    frame = _filter_window(
        _load_optional_csv_payload(capital_flow_file),
        as_of_date=as_of_date,
        lookback_days=lookback_days,
    )
    row = _latest_row(frame)
    northbound_net_flow = _safe_float(None if row is None else row.get("northbound_net_flow"), 0.0)
    margin_balance_change = _safe_float(None if row is None else row.get("margin_balance_change"), 0.0)
    turnover_heat = _clip(_safe_float(None if row is None else row.get("turnover_heat"), 0.5), 0.0, 1.0)
    large_order_bias = _clip(_safe_float(None if row is None else row.get("large_order_bias"), 0.0), -1.0, 1.0)

    flow_signal = (
        0.35 * _clip(northbound_net_flow, -1.0, 1.0)
        + 0.25 * _clip(margin_balance_change, -1.0, 1.0)
        + 0.20 * (2.0 * turnover_heat - 1.0)
        + 0.20 * large_order_bias
    )
    info_boost = 0.0
    for item in info_items:
        if item.target_type != "market":
            continue
        tag = str(item.event_tag).strip()
        if tag in {"share_increase", "contract_win", "regulatory_positive", "earnings_positive"}:
            info_boost += 0.03
        elif tag in {"share_reduction", "regulatory_negative", "earnings_negative", "delisting_risk"}:
            info_boost -= 0.03
    flow_signal += info_boost

    if flow_signal >= 0.18:
        flow_regime = "strong_inflow"
    elif flow_signal >= 0.05:
        flow_regime = "inflow"
    elif flow_signal <= -0.18:
        flow_regime = "strong_outflow"
    elif flow_signal <= -0.05:
        flow_regime = "outflow"
    else:
        flow_regime = "neutral"
    return CapitalFlowState(
        northbound_net_flow=float(northbound_net_flow),
        margin_balance_change=float(margin_balance_change),
        turnover_heat=float(turnover_heat),
        large_order_bias=float(large_order_bias),
        flow_regime=str(flow_regime),
    )


def build_macro_context_state(
    *,
    as_of_date: pd.Timestamp,
    macro_file: object,
    lookback_days: int,
    info_items: Iterable[InfoItem] = (),
) -> MacroContextState:
    frame = _filter_window(
        _load_optional_csv_payload(macro_file),
        as_of_date=as_of_date,
        lookback_days=lookback_days,
    )
    row = _latest_row(frame)
    style_regime = str("" if row is None else row.get("style_regime", "")).strip() or "balanced"
    commodity_pressure = _clip(_safe_float(None if row is None else row.get("commodity_pressure"), 0.0), 0.0, 1.0)
    fx_pressure = _clip(_safe_float(None if row is None else row.get("fx_pressure"), 0.0), 0.0, 1.0)
    index_breadth_proxy = _clip(_safe_float(None if row is None else row.get("index_breadth_proxy"), 0.5), 0.0, 1.0)

    market_info_score = 0.0
    market_items = 0
    for item in info_items:
        if item.target_type != "market":
            continue
        market_items += 1
        market_info_score += float(normalize_direction(item.direction))
    if market_items:
        index_breadth_proxy = _clip(0.75 * index_breadth_proxy + 0.25 * ((market_info_score / market_items + 1.0) / 2.0), 0.0, 1.0)

    macro_pressure = 0.45 * commodity_pressure + 0.35 * fx_pressure + 0.20 * (1.0 - index_breadth_proxy)
    if macro_pressure >= 0.65:
        macro_risk_level = "high"
    elif macro_pressure >= 0.40:
        macro_risk_level = "elevated"
    elif macro_pressure <= 0.18:
        macro_risk_level = "low"
    else:
        macro_risk_level = "neutral"
    return MacroContextState(
        style_regime=str(style_regime),
        commodity_pressure=float(commodity_pressure),
        fx_pressure=float(fx_pressure),
        index_breadth_proxy=float(index_breadth_proxy),
        macro_risk_level=str(macro_risk_level),
    )


def build_external_signal_manifest(
    *,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: Iterable[InfoItem],
    capital_flow_state: CapitalFlowState,
    macro_context_state: MacroContextState,
) -> dict[str, object]:
    items = list(info_items)
    event_tags = [str(item.event_tag) for item in items if str(item.event_tag).strip()]
    source_files = {
        "event_file": _hashable_path(settings.get("event_file") or settings.get("info_file")),
        "capital_flow_file": _hashable_path(settings.get("capital_flow_file")),
        "macro_file": _hashable_path(settings.get("macro_file")),
    }
    unique_publishers = {str(item.publisher).strip() for item in items if str(item.publisher).strip()}
    unique_subsets = {str(item.source_subset).strip() for item in items if str(item.source_subset).strip()}
    return {
        "as_of_date": str(as_of_date.date()),
        "external_signal_version": str(settings.get("external_signal_version", "v1")),
        "external_signal_enabled": bool(settings.get("external_signals", True)),
        "sources": source_files,
        "windows": {
            "event_lookback_days": int(settings.get("event_lookback_days", settings.get("info_lookback_days", 45))),
            "capital_flow_lookback_days": int(settings.get("capital_flow_lookback_days", 20)),
            "macro_lookback_days": int(settings.get("macro_lookback_days", 60)),
        },
        "coverage": {
            "event_item_count": int(len(items)),
            "publisher_count": int(len(unique_publishers)),
            "source_subset_count": int(len(unique_subsets)),
            "event_tag_count": int(len(set(event_tags))),
        },
        "event_summary": {
            "top_event_tags": sorted(set(event_tags))[:10],
        },
        "capital_flow_snapshot": asdict(capital_flow_state),
        "macro_context_snapshot": asdict(macro_context_state),
    }
