from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _safe_latest_value(frame: object, key: str, default: float = 0.0) -> float:
    if frame is None:
        return float(default)
    try:
        value = float(frame.get(key, default))  # type: ignore[union-attr]
    except Exception:
        return float(default)
    if value != value:
        return float(default)
    return float(value)


@dataclass(frozen=True)
class CrossSectionForecastRecord:
    as_of_date: pd.Timestamp
    large_vs_small_bias: float
    growth_vs_value_bias: float
    fund_flow_strength: float
    margin_risk_on_score: float
    breadth_strength: float
    leader_participation: float
    weak_stock_ratio: float
    breadth_regime_score: float
    flow_regime_score: float


def forecast_cross_section_state(
    market_frame: pd.DataFrame,
) -> CrossSectionForecastRecord:
    if market_frame is None or market_frame.empty:
        raise ValueError("market_frame is empty")

    latest = market_frame.sort_values("date").iloc[-1]

    large_vs_small = _clip(
        _safe_latest_value(latest, "idx_cyb_trend_20_60") - _safe_latest_value(latest, "idx_sh_trend_20_60"),
        -1.0,
        1.0,
    )
    growth_vs_value = _clip(
        _safe_latest_value(latest, "idx_cyb_ret_20") - _safe_latest_value(latest, "idx_sh_ret_20"),
        -1.0,
        1.0,
    )

    fin_net = _safe_latest_value(latest, "mrg_mkt_fin_net_buy_z20")
    fin_spread = _safe_latest_value(latest, "mrg_mkt_fin_sec_spread_chg5")
    breadth_up_down = _safe_latest_value(latest, "breadth_up_down_diff")
    breadth_up = _safe_latest_value(latest, "breadth_up_ratio")
    breadth_down = _safe_latest_value(latest, "breadth_down_ratio")
    limit_spread = _safe_latest_value(latest, "breadth_limit_spread")
    limit_up_ratio = _safe_latest_value(latest, "breadth_limit_up_ratio")
    amount_z20 = _safe_latest_value(latest, "breadth_amount_z20")

    flow_regime_score = _clip(0.55 * fin_net + 0.45 * fin_spread, -1.0, 1.0)
    breadth_regime_score = _clip(0.55 * breadth_up_down + 0.30 * limit_spread + 0.15 * amount_z20, -1.0, 1.0)

    fund_flow_strength = _clip(flow_regime_score, -1.0, 1.0)
    margin_risk_on_score = _clip(0.5 * fin_spread + 0.5 * fin_net, -1.0, 1.0)
    breadth_strength = _clip(breadth_regime_score, -1.0, 1.0)
    leader_participation = _clip(breadth_up + 2.0 * limit_up_ratio, 0.0, 1.0)
    weak_stock_ratio = _clip(breadth_down, 0.0, 1.0)

    return CrossSectionForecastRecord(
        as_of_date=pd.Timestamp(latest["date"]),
        large_vs_small_bias=float(large_vs_small),
        growth_vs_value_bias=float(growth_vs_value),
        fund_flow_strength=float(fund_flow_strength),
        margin_risk_on_score=float(margin_risk_on_score),
        breadth_strength=float(breadth_strength),
        leader_participation=float(leader_participation),
        weak_stock_ratio=float(weak_stock_ratio),
        breadth_regime_score=float(breadth_regime_score),
        flow_regime_score=float(flow_regime_score),
    )
