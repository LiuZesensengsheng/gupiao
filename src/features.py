"""Compatibility wrapper for feature-engineering APIs.

Use `src.infrastructure.features` for new code.
"""

from src.infrastructure.features import (
    BASE_FEATURE_COLUMNS,
    MARKET_FEATURE_COLUMNS,
    build_features,
    make_market_feature_frame,
    make_stock_feature_frame,
    stock_feature_columns,
)

__all__ = [
    "BASE_FEATURE_COLUMNS",
    "MARKET_FEATURE_COLUMNS",
    "build_features",
    "make_market_feature_frame",
    "make_stock_feature_frame",
    "stock_feature_columns",
]

