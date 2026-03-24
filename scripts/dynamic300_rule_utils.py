from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RuleCandidateRow:
    symbol: str
    name: str
    theme: str
    refined_score: float
    fresh_pool_score: float
    fresh_pool_pass: bool
    recent_high_gap20: float
    amount_ratio20: float
    theme_selected_count: int
    theme_strength: float
    close: float
    ma20: float
    ma60: float
    ret20: float = 0.0
    ret60: float = 0.0
    breakout_pos_120: float = 0.0
    volatility20: float = 0.0
    tradeability: float = 0.0
    bucket: str = "reserve"
    portfolio_score: float = 0.0
    weight: float = 0.0


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def coalesce(*values: object, default: object) -> object:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def triangle_score(value: float, low: float, ideal: float, high: float) -> float:
    value = _safe_float(value)
    if value <= low or value >= high:
        return 0.0
    if abs(value - ideal) <= 1e-12:
        return 1.0
    if value < ideal:
        return max(0.0, min(1.0, (value - low) / max(ideal - low, 1e-9)))
    return max(0.0, min(1.0, (high - value) / max(high - ideal, 1e-9)))


def descending_score(value: float, good: float, bad: float) -> float:
    value = _safe_float(value)
    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return max(0.0, min(1.0, 1.0 - ((value - good) / max(bad - good, 1e-9))))


def _price_distance_score(close: float, anchor: float) -> float:
    close = _safe_float(close)
    anchor = _safe_float(anchor)
    if close <= 0.0 or anchor <= 0.0:
        return 0.0
    return descending_score(abs(close / anchor - 1.0), 0.0, 0.06)


def candidate_bucket(row: RuleCandidateRow | dict[str, object]) -> str:
    getter = row.get if isinstance(row, dict) else lambda key, default=None: getattr(row, key, default)
    close = _safe_float(getter("close", 0.0))
    ma20 = _safe_float(getter("ma20", 0.0))
    ret20 = _safe_float(getter("ret20", 0.0))
    recent_high_gap20 = _safe_float(getter("recent_high_gap20", -1.0))
    amount_ratio20 = _safe_float(getter("amount_ratio20", 0.0))
    breakout_pos_120 = _safe_float(getter("breakout_pos_120", 0.0))
    fresh_pool_pass = bool(getter("fresh_pool_pass", False))

    if (
        fresh_pool_pass
        and ret20 >= 0.03
        and close >= ma20 > 0.0
        and -0.12 <= recent_high_gap20 <= -0.045
        and amount_ratio20 >= 0.85
    ):
        return "pullback"
    if (
        fresh_pool_pass
        and ret20 >= 0.06
        and recent_high_gap20 >= -0.055
        and amount_ratio20 >= 0.90
    ):
        return "trend"
    if (
        close >= ma20 * 0.99 > 0.0
        and breakout_pos_120 >= 0.86
        and amount_ratio20 >= 1.00
        and ret20 >= 0.015
    ):
        return "breakout"
    return "reserve"


def portfolio_score(row: RuleCandidateRow | dict[str, object], theme_count: int, theme_strength: float) -> float:
    getter = row.get if isinstance(row, dict) else lambda key, default=None: getattr(row, key, default)
    bucket_name = str(getter("bucket", "")).strip().lower() or candidate_bucket(row)
    refined = _safe_float(getter("refined_score", 0.0))
    fresh = _safe_float(getter("fresh_pool_score", 0.0))
    amount_ratio20 = _safe_float(getter("amount_ratio20", 0.0))
    recent_high_gap20 = _safe_float(getter("recent_high_gap20", -0.5))
    close = _safe_float(getter("close", 0.0))
    ma20 = _safe_float(getter("ma20", 0.0))
    ret20 = _safe_float(getter("ret20", 0.0))
    ret60 = _safe_float(getter("ret60", 0.0))
    breakout_pos_120 = _safe_float(getter("breakout_pos_120", 0.0))
    volatility20 = _safe_float(getter("volatility20", 0.0))

    theme_count_score = min(1.0, max(0.0, float(theme_count) / 12.0))
    amount_balance = triangle_score(amount_ratio20, 0.70, 1.15, 2.20)
    near_high_score = triangle_score(recent_high_gap20, -0.12, -0.03, 0.01)
    base = (
        0.34 * refined
        + 0.10 * fresh
        + 0.12 * theme_count_score
        + 0.10 * _safe_float(theme_strength)
        + 0.08 * amount_balance
        + 0.06 * near_high_score
    )
    if bucket_name == "trend":
        score = (
            base
            + 0.10 * triangle_score(ret20, 0.02, 0.10, 0.24)
            + 0.08 * triangle_score(ret60, 0.05, 0.22, 0.60)
            + 0.06 * descending_score(volatility20, 0.018, 0.050)
            + 0.04 * _price_distance_score(close, ma20)
        )
    elif bucket_name == "pullback":
        score = (
            base
            + 0.12 * triangle_score(recent_high_gap20, -0.14, -0.07, -0.01)
            + 0.10 * _price_distance_score(close, ma20)
            + 0.08 * descending_score(volatility20, 0.018, 0.048)
            + 0.06 * triangle_score(ret20, 0.02, 0.06, 0.18)
        )
    elif bucket_name == "breakout":
        score = (
            base
            + 0.13 * triangle_score(breakout_pos_120, 0.72, 0.97, 1.01)
            + 0.10 * triangle_score(amount_ratio20, 0.90, 1.25, 2.40)
            + 0.07 * triangle_score(ret20, 0.01, 0.07, 0.20)
            + 0.04 * descending_score(volatility20, 0.018, 0.055)
        )
    else:
        score = base - 0.05
    return max(0.0, min(2.0, float(score)))


def is_primary_candidate(row: RuleCandidateRow | dict[str, object], theme_count: int) -> bool:
    getter = row.get if isinstance(row, dict) else lambda key, default=None: getattr(row, key, default)
    bucket_name = str(getter("bucket", "")).strip().lower() or candidate_bucket(row)
    close = _safe_float(getter("close", 0.0))
    ma20 = _safe_float(getter("ma20", 0.0))
    fresh = _safe_float(getter("fresh_pool_score", 0.0))
    fresh_pass = bool(getter("fresh_pool_pass", False))
    ret20 = _safe_float(getter("ret20", 0.0))
    recent_gap = _safe_float(getter("recent_high_gap20", -1.0))
    amount_ratio20 = _safe_float(getter("amount_ratio20", 0.0))
    breakout_pos_120 = _safe_float(getter("breakout_pos_120", 0.0))
    if bucket_name == "trend":
        return bool(fresh_pass and fresh >= 0.58 and ret20 >= 0.05 and theme_count >= 4)
    if bucket_name == "pullback":
        return bool(
            fresh_pass
            and close >= ma20 > 0.0
            and -0.13 <= recent_gap <= -0.03
            and ret20 >= 0.03
            and theme_count >= 3
        )
    if bucket_name == "breakout":
        return bool(
            close >= ma20 * 0.99 > 0.0
            and breakout_pos_120 >= 0.90
            and amount_ratio20 >= 1.0
            and ret20 >= 0.015
        )
    return bool(fresh_pass and fresh >= 0.62 and theme_count >= 6)


def _bucket_soft_cap(bucket_name: str, top_n: int) -> int:
    bucket_name = str(bucket_name).strip().lower()
    if bucket_name == "trend":
        return max(1, min(2, top_n))
    if bucket_name in {"pullback", "breakout"}:
        return max(1, min(2, top_n))
    return 1


def select_portfolio(rows: Iterable[RuleCandidateRow], *, top_n: int, max_per_theme: int) -> list[RuleCandidateRow]:
    ordered = list(rows)
    selected: list[RuleCandidateRow] = []
    selected_symbols: set[str] = set()
    theme_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}

    def _can_pick(row: RuleCandidateRow, *, ignore_bucket_cap: bool = False) -> bool:
        if row.symbol in selected_symbols:
            return False
        if theme_counts.get(row.theme, 0) >= int(max_per_theme):
            return False
        if not ignore_bucket_cap and bucket_counts.get(row.bucket, 0) >= _bucket_soft_cap(row.bucket, top_n):
            return False
        return True

    for bucket_name in ("trend", "pullback", "breakout"):
        if len(selected) >= int(top_n):
            break
        for row in ordered:
            if row.bucket != bucket_name:
                continue
            if not _can_pick(row):
                continue
            selected.append(row)
            selected_symbols.add(row.symbol)
            theme_counts[row.theme] = theme_counts.get(row.theme, 0) + 1
            bucket_counts[row.bucket] = bucket_counts.get(row.bucket, 0) + 1
            break

    for row in ordered:
        if len(selected) >= int(top_n):
            break
        if not _can_pick(row):
            continue
        selected.append(row)
        selected_symbols.add(row.symbol)
        theme_counts[row.theme] = theme_counts.get(row.theme, 0) + 1
        bucket_counts[row.bucket] = bucket_counts.get(row.bucket, 0) + 1

    for row in ordered:
        if len(selected) >= int(top_n):
            break
        if not _can_pick(row, ignore_bucket_cap=True):
            continue
        selected.append(row)
        selected_symbols.add(row.symbol)
        theme_counts[row.theme] = theme_counts.get(row.theme, 0) + 1
        bucket_counts[row.bucket] = bucket_counts.get(row.bucket, 0) + 1

    total_score = sum(max(float(row.portfolio_score), 1e-9) for row in selected)
    if total_score <= 1e-9:
        equal_weight = 1.0 / max(1, len(selected))
        return [replace(row, weight=float(equal_weight)) for row in selected]
    return [
        replace(row, weight=float(max(float(row.portfolio_score), 1e-9) / total_score))
        for row in selected
    ]


def buy_zone_bounds(row: RuleCandidateRow) -> tuple[float, float]:
    lower = max(float(row.ma20), float(row.close) * 0.985)
    upper = float(row.close) * 1.010
    if lower > upper:
        lower = min(float(row.close), float(row.ma20))
    return float(lower), float(upper)


def avoid_upper_bound(row: RuleCandidateRow) -> float:
    return float(row.close) * 1.035


def monitor_reason(row: RuleCandidateRow) -> str:
    if row.fresh_pool_pass:
        return f"bucket={row.bucket}; passed fresh gate, but stronger theme/candidate occupied the core slots."
    return f"bucket={row.bucket}; failed the primary fresh gate, keep on monitor instead of forcing a buy."
