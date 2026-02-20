"""Compatibility wrapper for effect analysis APIs.

Use `src.infrastructure.effect_analysis` for new code.
"""

from src.infrastructure.effect_analysis import (
    build_latest_snapshot,
    compute_effect_summary,
    compute_sector_table,
)

__all__ = ["build_latest_snapshot", "compute_effect_summary", "compute_sector_table"]

