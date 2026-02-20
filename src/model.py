"""Compatibility wrapper for modeling APIs.

Use `src.infrastructure.modeling` and `src.domain.entities` for new code.
"""

from src.domain.entities import BinaryMetrics
from src.infrastructure.modeling import LogisticBinaryModel, binary_metrics

__all__ = ["BinaryMetrics", "LogisticBinaryModel", "binary_metrics"]

