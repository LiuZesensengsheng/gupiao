from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def blend_horizon_score(short_prob: float, mid_prob: float, short_weight: float = 0.55) -> float:
    short_weight = float(np.clip(short_weight, 0.0, 1.0))
    return short_weight * float(short_prob) + (1.0 - short_weight) * float(mid_prob)


@dataclass(frozen=True)
class MarketStateDecision:
    state_code: str
    state_label: str
    strategy_template: str
    intraday_t_level: str
    exposure_cap: float
    weight_threshold: float
    max_positions: int
    max_trades_per_stock_per_day: int
    max_trades_per_stock_per_week: int


def decide_market_state(
    short_prob: float,
    mid_prob: float,
    *,
    base_weight_threshold: float = 0.50,
    base_max_positions: int = 5,
    base_max_trades_per_stock_per_day: int = 1,
    base_max_trades_per_stock_per_week: int = 3,
) -> MarketStateDecision:
    score = blend_horizon_score(short_prob, mid_prob, short_weight=0.6)
    dispersion = abs(float(short_prob) - float(mid_prob))

    base_max_positions = max(1, int(base_max_positions))
    base_max_trades_per_stock_per_day = max(1, int(base_max_trades_per_stock_per_day))
    base_max_trades_per_stock_per_week = max(1, int(base_max_trades_per_stock_per_week))
    base_weight_threshold = float(base_weight_threshold)

    is_risk_off = (score <= 0.45) or (mid_prob <= 0.46 and short_prob <= 0.50)
    is_trend = (score >= 0.58 and mid_prob >= 0.53) and dispersion <= 0.22

    if is_risk_off:
        return MarketStateDecision(
            state_code="risk_off",
            state_label="风险收缩 (Risk-Off)",
            strategy_template="defensive",
            intraday_t_level="low",
            exposure_cap=0.25,
            weight_threshold=min(0.62, base_weight_threshold + 0.06),
            max_positions=max(1, base_max_positions - 2),
            max_trades_per_stock_per_day=1,
            max_trades_per_stock_per_week=max(1, min(base_max_trades_per_stock_per_week, 2)),
        )
    if is_trend:
        return MarketStateDecision(
            state_code="trend",
            state_label="趋势延续 (Trend)",
            strategy_template="trend_push",
            intraday_t_level="medium",
            exposure_cap=0.72,
            weight_threshold=max(0.40, base_weight_threshold - 0.02),
            max_positions=base_max_positions,
            max_trades_per_stock_per_day=min(1, base_max_trades_per_stock_per_day),
            max_trades_per_stock_per_week=max(1, min(base_max_trades_per_stock_per_week, 3)),
        )
    return MarketStateDecision(
        state_code="range",
        state_label="震荡 (Range)",
        strategy_template="buy_low_sell_high",
        intraday_t_level="high",
        exposure_cap=0.55,
        weight_threshold=min(0.60, base_weight_threshold + 0.02),
        max_positions=max(1, base_max_positions - 1),
        max_trades_per_stock_per_day=min(1, base_max_trades_per_stock_per_day),
        max_trades_per_stock_per_week=max(1, min(5, base_max_trades_per_stock_per_week + 1)),
    )


def market_regime(short_prob: float, mid_prob: float) -> str:
    decision = decide_market_state(short_prob, mid_prob)
    if decision.state_code == "trend":
        return "偏强 (Risk-On)"
    if decision.state_code == "risk_off":
        return "偏弱 (Risk-Off)"
    return "中性 (Neutral)"


def target_exposure(short_prob: float, mid_prob: float) -> float:
    return float(decide_market_state(short_prob, mid_prob).exposure_cap)


def allocate_weights(
    scores: list[float],
    total_exposure: float,
    threshold: float = 0.50,
    max_positions: int | None = None,
) -> list[float]:
    if not scores:
        return []
    n = len(scores)
    max_pos = None if max_positions is None else max(1, min(int(max_positions), n))
    raw = np.array([max(0.0, float(s) - float(threshold)) for s in scores], dtype=float)

    if max_pos is not None and max_pos < n:
        ranked = np.argsort(-raw)
        keep = ranked[:max_pos]
        mask = np.zeros(n, dtype=bool)
        mask[keep] = True
        raw = np.where(mask, raw, 0.0)

    if np.all(raw <= 1e-12):
        if max_pos is not None and max_pos < n:
            ranked_scores = np.argsort(-np.asarray(scores, dtype=float))
            keep = ranked_scores[:max_pos]
            out = np.zeros(n, dtype=float)
            out[keep] = float(total_exposure) / float(len(keep))
            return out.tolist()
        return [float(total_exposure) / len(scores)] * len(scores)
    raw = raw / raw.sum()
    alloc = raw * float(total_exposure)
    return alloc.tolist()
