"""Compatibility wrapper for data-loading APIs.

Use `src.infrastructure.market_data` for new code.
"""

from src.domain.symbols import SymbolInfo, normalize_symbol
from src.infrastructure.market_data import DataError, fetch_eastmoney_daily, load_local_daily, load_symbol_daily

__all__ = [
    "DataError",
    "SymbolInfo",
    "fetch_eastmoney_daily",
    "load_local_daily",
    "load_symbol_daily",
    "normalize_symbol",
]

