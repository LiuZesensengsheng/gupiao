from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .data import normalize_symbol


@dataclass(frozen=True)
class NewsItem:
    date: pd.Timestamp
    target_type: str
    target: str
    horizon: str
    direction: str
    strength: float
    confidence: float
    source_weight: float
    title: str


@dataclass
class SentimentAggregate:
    bullish: float
    bearish: float
    neutral: float
    score: float
    items: int


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _norm_direction(direction: str) -> int:
    d = direction.strip().lower()
    if d in {"bullish", "positive", "利好", "多", "up"}:
        return 1
    if d in {"bearish", "negative", "利空", "空", "down"}:
        return -1
    return 0


def _norm_horizon(horizon: str) -> str:
    h = horizon.strip().lower()
    if h in {"short", "1d", "短期"}:
        return "short"
    if h in {"mid", "middle", "20d", "中期"}:
        return "mid"
    return "both"


def _norm_target_type(target_type: str) -> str:
    t = target_type.strip().lower()
    if t in {"market", "index", "大盘"}:
        return "market"
    return "stock"


def _normalize_target(target_type: str, target: str) -> str:
    if target_type == "market":
        return "MARKET"
    return normalize_symbol(target).symbol


def _freshness_decay(news_date: pd.Timestamp, as_of_date: pd.Timestamp, half_life_days: float) -> float:
    if pd.isna(news_date) or pd.isna(as_of_date):
        return 1.0
    days = max(0.0, float((as_of_date - news_date).days))
    return float(0.5 ** (days / max(half_life_days, 1e-6)))


def load_news_items(
    csv_path: str | Path,
    as_of_date: pd.Timestamp,
    lookback_days: int = 7,
) -> List[NewsItem]:
    path = Path(csv_path)
    if not path.exists():
        return []

    raw = pd.read_csv(path)
    if raw.empty:
        return []

    lower_map = {c.lower(): c for c in raw.columns}
    required = ["date", "target_type", "target", "direction"]
    for col in required:
        if col not in lower_map:
            raise ValueError(f"News CSV missing required column: {col}")

    out: List[NewsItem] = []
    for _, row in raw.iterrows():
        date = pd.to_datetime(row[lower_map["date"]], errors="coerce")
        if pd.isna(date):
            continue
        if date > as_of_date:
            continue
        if (as_of_date - date).days > lookback_days:
            continue

        target_type = _norm_target_type(str(row[lower_map["target_type"]]))
        target = _normalize_target(target_type, str(row[lower_map["target"]]))
        direction = str(row[lower_map["direction"]]).strip()

        horizon = "both"
        if "horizon" in lower_map:
            horizon = _norm_horizon(str(row[lower_map["horizon"]]))

        strength = 3.0
        if "strength" in lower_map:
            strength = float(row[lower_map["strength"]])
        confidence = 0.7
        if "confidence" in lower_map:
            confidence = float(row[lower_map["confidence"]])
        source_weight = 0.7
        if "source_weight" in lower_map:
            source_weight = float(row[lower_map["source_weight"]])
        title = ""
        if "title" in lower_map:
            title = str(row[lower_map["title"]])

        out.append(
            NewsItem(
                date=pd.Timestamp(date.normalize()),
                target_type=target_type,
                target=target,
                horizon=horizon,
                direction=direction,
                strength=float(strength),
                confidence=float(confidence),
                source_weight=float(source_weight),
                title=title,
            )
        )
    return out


def _item_weight(item: NewsItem, as_of_date: pd.Timestamp, half_life_days: float) -> float:
    strength = _clip01(item.strength / 5.0)
    confidence = _clip01(item.confidence)
    source_weight = _clip01(item.source_weight)
    freshness = _freshness_decay(item.date, as_of_date, half_life_days=half_life_days)
    return float(strength * confidence * source_weight * freshness)


def _horizon_multiplier(item_horizon: str, horizon: str) -> float:
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
    half_life_days: float = 2.0,
    market_to_stock_carry: float = 0.35,
) -> SentimentAggregate:
    target_norm = normalize_symbol(target).symbol if target != "MARKET" else "MARKET"
    weights: List[float] = []
    signs: List[int] = []

    for item in news_items:
        if item.target not in {target_norm, "MARKET"} and target_norm != "MARKET":
            continue
        if target_norm == "MARKET" and item.target != "MARKET":
            continue

        sign = _norm_direction(item.direction)
        if sign not in {-1, 0, 1}:
            continue

        w = _item_weight(item, as_of_date=as_of_date, half_life_days=half_life_days)
        w = w * _horizon_multiplier(item.horizon, horizon=horizon)
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
    score = float(np.clip(score, -1.0, 1.0))
    neutral = float(np.clip(1.0 - (bull + bear) / (tot + 1e-9), 0.0, 1.0))
    return SentimentAggregate(
        bullish=float(np.clip(bull / (tot + 1e-9), 0.0, 1.0)),
        bearish=float(np.clip(bear / (tot + 1e-9), 0.0, 1.0)),
        neutral=neutral,
        score=score,
        items=len(weights),
    )


def blend_probability(base_prob: float, sentiment_score: float, sentiment_strength: float) -> float:
    p = float(np.clip(base_prob, 1e-6, 1 - 1e-6))
    z = np.log(p / (1 - p))
    z_adj = z + float(sentiment_strength) * float(np.clip(sentiment_score, -1.0, 1.0))
    p_adj = 1.0 / (1.0 + np.exp(-z_adj))
    return float(np.clip(p_adj, 1e-6, 1 - 1e-6))
