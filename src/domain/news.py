from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .entities import NewsItem, SentimentAggregate
from .symbols import normalize_symbol


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def normalize_direction(direction: str) -> int:
    d = direction.strip().lower()
    if d in {"bullish", "positive", "利好", "多", "up"}:
        return 1
    if d in {"bearish", "negative", "利空", "空", "down"}:
        return -1
    return 0


def normalize_horizon(horizon: str) -> str:
    h = horizon.strip().lower()
    if h in {"short", "1d", "短期"}:
        return "short"
    if h in {"mid", "middle", "20d", "中期"}:
        return "mid"
    return "both"


def normalize_target_type(target_type: str) -> str:
    t = target_type.strip().lower()
    if t in {"market", "index", "大盘"}:
        return "market"
    return "stock"


def normalize_target(target_type: str, target: str) -> str:
    if target_type == "market":
        return "MARKET"
    return normalize_symbol(target).symbol


def freshness_decay(news_date: pd.Timestamp, as_of_date: pd.Timestamp, half_life_days: float) -> float:
    if pd.isna(news_date) or pd.isna(as_of_date):
        return 1.0
    days = max(0.0, float((as_of_date - news_date).days))
    return float(0.5 ** (days / max(half_life_days, 1e-6)))


def news_item_weight(item: NewsItem, as_of_date: pd.Timestamp, half_life_days: float) -> float:
    strength = _clip01(item.strength / 5.0)
    confidence = _clip01(item.confidence)
    source_weight = _clip01(item.source_weight)
    freshness = freshness_decay(item.date, as_of_date, half_life_days=half_life_days)
    return float(strength * confidence * source_weight * freshness)


def horizon_multiplier(item_horizon: str, horizon: str) -> float:
    if item_horizon == "both":
        return 1.0
    if item_horizon == horizon:
        return 1.0
    return 0.25


def aggregate_sentiment(
    news_items: Sequence[NewsItem],
    as_of_date: pd.Timestamp,
    target: str,
    horizon: str,
    half_life_days: float = 10.0,
    market_to_stock_carry: float = 0.35,
) -> SentimentAggregate:
    target_norm = normalize_symbol(target).symbol if target != "MARKET" else "MARKET"
    weights: list[float] = []
    signs: list[int] = []

    for item in news_items:
        if item.date > as_of_date:
            continue
        if item.target not in {target_norm, "MARKET"} and target_norm != "MARKET":
            continue
        if target_norm == "MARKET" and item.target != "MARKET":
            continue

        sign = normalize_direction(item.direction)
        if sign not in {-1, 0, 1}:
            continue

        w = news_item_weight(item, as_of_date=as_of_date, half_life_days=half_life_days)
        w = w * horizon_multiplier(item.horizon, horizon=horizon)
        if target_norm != "MARKET" and item.target == "MARKET":
            w = w * float(np.clip(market_to_stock_carry, 0.0, 1.0))
        if w <= 1e-12:
            continue

        weights.append(w)
        signs.append(sign)

    if not weights:
        return SentimentAggregate(bullish=0.0, bearish=0.0, neutral=1.0, score=0.0, items=0)

    w = np.asarray(weights, dtype=float)
    s = np.asarray(signs, dtype=float)
    bull = float(np.sum(w * np.clip(s, 0, 1)))
    bear = float(np.sum(w * np.clip(-s, 0, 1)))
    tot = float(np.sum(w))
    score = float((bull - bear) / (tot + 1e-9))
    neutral = float(np.clip(1.0 - (bull + bear) / (tot + 1e-9), 0.0, 1.0))
    return SentimentAggregate(
        bullish=float(np.clip(bull / (tot + 1e-9), 0.0, 1.0)),
        bearish=float(np.clip(bear / (tot + 1e-9), 0.0, 1.0)),
        neutral=neutral,
        score=float(np.clip(score, -1.0, 1.0)),
        items=len(weights),
    )


def blend_probability(base_prob: float, sentiment_score: float, sentiment_strength: float) -> float:
    p = float(np.clip(base_prob, 1e-6, 1 - 1e-6))
    z = np.log(p / (1 - p))
    z_adj = z + float(sentiment_strength) * float(np.clip(sentiment_score, -1.0, 1.0))
    p_adj = 1.0 / (1.0 + np.exp(-z_adj))
    return float(np.clip(p_adj, 1e-6, 1 - 1e-6))
