"""Compatibility wrapper for news-matrix APIs.

Use `src.domain.news` and `src.infrastructure.news_repository` for new code.
"""

from src.domain.entities import NewsItem, SentimentAggregate
from src.domain.news import (
    aggregate_sentiment,
    blend_probability,
    freshness_decay as _freshness_decay,
    normalize_direction as _norm_direction,
    normalize_horizon as _norm_horizon,
    normalize_target as _normalize_target,
    normalize_target_type as _norm_target_type,
)
from src.infrastructure.news_repository import load_news_items

__all__ = [
    "NewsItem",
    "SentimentAggregate",
    "_freshness_decay",
    "_norm_direction",
    "_norm_horizon",
    "_norm_target_type",
    "_normalize_target",
    "aggregate_sentiment",
    "blend_probability",
    "load_news_items",
]

