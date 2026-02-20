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
