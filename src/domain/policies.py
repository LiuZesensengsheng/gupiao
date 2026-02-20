from __future__ import annotations

import numpy as np


def blend_horizon_score(short_prob: float, mid_prob: float, short_weight: float = 0.55) -> float:
    short_weight = float(np.clip(short_weight, 0.0, 1.0))
    return short_weight * float(short_prob) + (1.0 - short_weight) * float(mid_prob)


def market_regime(short_prob: float, mid_prob: float) -> str:
    score = blend_horizon_score(short_prob, mid_prob, short_weight=0.6)
    if score >= 0.60:
        return "偏强 (Risk-On)"
    if score <= 0.40:
        return "偏弱 (Risk-Off)"
    return "中性 (Neutral)"


def target_exposure(short_prob: float, mid_prob: float) -> float:
    score = blend_horizon_score(short_prob, mid_prob, short_weight=0.6)
    if score <= 0.40:
        return 0.25
    if score <= 0.50:
        return 0.40
    if score <= 0.60:
        return 0.55
    return 0.72


def allocate_weights(scores: list[float], total_exposure: float, threshold: float = 0.50) -> list[float]:
    if not scores:
        return []
    raw = np.array([max(0.0, float(s) - float(threshold)) for s in scores], dtype=float)
    if np.all(raw <= 1e-12):
        return [float(total_exposure) / len(scores)] * len(scores)
    raw = raw / raw.sum()
    alloc = raw * float(total_exposure)
    return alloc.tolist()

