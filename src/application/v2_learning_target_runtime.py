from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.application.v2_contracts import CompositeState


@dataclass(frozen=True)
class LearningTargetDependencies:
    stock_policy_score: Callable[[object], float]
    safe_float: Callable[[object, float], float]
    alpha_opportunity_metrics: Callable[[object], dict[str, float]]
    signal_unit: Callable[[object, float], float]
    normalize_universe_tier: Callable[[str | None], str]
    clip: Callable[[float, float, float], float]


def derive_learning_targets(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    horizon_metrics: dict[str, dict[str, float]] | None = None,
    universe_tier: str | None = None,
    deps: LearningTargetDependencies,
) -> tuple[float, float, float, float]:
    ranked = sorted(state.stocks, key=deps.stock_policy_score, reverse=True)
    realized: list[float] = []
    realized_excess_1d: list[float] = []
    realized_excess_5d: list[float] = []
    realized_excess_20d_sector: list[float] = []
    for stock in ranked[:4]:
        frame = stock_frames.get(stock.symbol)
        if frame is None:
            continue
        row = frame[frame["date"] == date]
        if row.empty:
            continue
        realized.append(deps.safe_float(row.iloc[0].get("fwd_ret_1"), 0.0))
        realized_excess_1d.append(deps.safe_float(row.iloc[0].get("excess_ret_1_vs_mkt"), 0.0))
        realized_excess_5d.append(deps.safe_float(row.iloc[0].get("excess_ret_5_vs_mkt"), 0.0))
        realized_excess_20d_sector.append(deps.safe_float(row.iloc[0].get("excess_ret_20_vs_sector"), 0.0))
    lead_ret = float(np.mean(realized)) if realized else 0.0
    lead_excess_1d = float(np.mean(realized_excess_1d)) if realized_excess_1d else 0.0
    lead_excess_5d = float(np.mean(realized_excess_5d)) if realized_excess_5d else 0.0
    lead_excess_20d_sector = float(np.mean(realized_excess_20d_sector)) if realized_excess_20d_sector else 0.0
    alpha_metrics = deps.alpha_opportunity_metrics(state.stocks)
    opportunity_signal = float(
        0.45 * deps.signal_unit(alpha_metrics["alpha_headroom"], 0.04)
        + 0.30 * deps.signal_unit(alpha_metrics["breadth_ratio"] - 0.08, 0.12)
        + 0.25 * deps.signal_unit(alpha_metrics["top_score"] - 0.56, 0.10)
    )
    quality_20d = {} if horizon_metrics is None else dict(horizon_metrics.get("20d", {}))
    ranking_signal = float(
        0.55 * deps.signal_unit(quality_20d.get("rank_ic", 0.0), 0.12)
        + 0.30 * deps.signal_unit(quality_20d.get("top_bottom_spread", 0.0), 0.08)
        + 0.15 * deps.signal_unit(deps.safe_float(quality_20d.get("top_k_hit_rate", 0.5), 0.5) - 0.5, 0.20)
    )
    realized_alpha_signal = float(
        0.50 * deps.signal_unit(lead_excess_1d, 0.02)
        + 0.30 * deps.signal_unit(lead_excess_5d, 0.04)
        + 0.20 * deps.signal_unit(lead_excess_20d_sector, 0.08)
    )
    composite_signal = float(
        0.45 * realized_alpha_signal
        + 0.30 * ranking_signal
        + 0.25 * opportunity_signal
    )

    if deps.normalize_universe_tier(universe_tier) == "generated_80":
        regime_floor = 0.45 if state.risk_regime == "risk_on" else (0.35 if state.risk_regime == "cautious" else 0.25)
        exposure = float(deps.clip(0.58 + 0.22 * composite_signal, regime_floor, 0.92))
        if state.market.drawdown_risk > 0.45:
            exposure *= 0.90
        if state.market.volatility_regime == "high":
            exposure *= 0.92
        if state.cross_section.weak_stock_ratio > 0.55:
            exposure *= 0.92
        if state.cross_section.breadth_strength < 0.05:
            exposure *= 0.95
        exposure = float(deps.clip(exposure, regime_floor, 0.92))

        positions = 3
        if alpha_metrics["breadth_ratio"] >= 0.10:
            positions += 1
        if alpha_metrics["alpha_headroom"] >= 0.02 and state.cross_section.breadth_strength >= 0.10:
            positions += 1
        if composite_signal <= -0.20 or state.cross_section.weak_stock_ratio >= 0.60:
            positions -= 1
        if state.market.volatility_regime == "high":
            positions = min(positions, 4)
        positions = int(np.clip(positions, 1, 5))

        turnover = float(
            0.18
            + 0.07 * max(0.0, realized_alpha_signal)
            + 0.04 * max(0.0, ranking_signal)
            + 0.05 * max(0.0, opportunity_signal)
            + 0.03 * abs(composite_signal)
        )
        if composite_signal < -0.10:
            turnover = min(turnover, 0.18)
        if state.market.drawdown_risk > 0.45:
            turnover = min(turnover, 0.18)
        if state.market.volatility_regime == "high":
            turnover = min(turnover, 0.24)
        turnover = float(deps.clip(turnover, 0.12, 0.32))
        sample_weight = float(
            1.0
            + 1.2 * abs(composite_signal)
            + 0.8 * max(0.0, realized_alpha_signal)
            + 0.5 * max(0.0, ranking_signal)
            + 0.6 * max(0.0, opportunity_signal)
        )
        return float(exposure), float(positions), float(turnover), sample_weight

    if lead_ret >= 0.008:
        exposure = 0.85
    elif lead_ret >= 0.0:
        exposure = 0.60
    else:
        exposure = 0.35

    breadth_bonus = 1 if float(state.cross_section.breadth_strength) > 0.05 else 0
    weakness_penalty = 1 if float(state.cross_section.weak_stock_ratio) > 0.55 else 0
    positions = int(np.clip(3 + breadth_bonus - weakness_penalty + (1 if lead_ret >= 0.012 else 0), 1, 5))

    turnover = 0.18
    if abs(lead_ret) >= 0.01:
        turnover = 0.32
    elif abs(lead_ret) >= 0.004:
        turnover = 0.25
    if float(state.market.drawdown_risk) > 0.45:
        turnover = min(turnover, 0.18)

    sample_weight = float(1.0 + 1.5 * abs(lead_ret))
    return float(exposure), float(positions), float(turnover), sample_weight
