from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CompositeState,
    InfoAggregateState,
    InfoDivergenceRecord,
    InfoItem,
    InfoSignalRecord,
)
from src.domain.news import blend_probability, normalize_direction

_NEGATIVE_EVENT_TAGS = {
    "earnings_negative",
    "guidance_negative",
    "contract_loss",
    "regulatory_negative",
    "share_reduction",
    "trading_halt",
    "delisting_risk",
}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return out


def _freshness_decay(item_date: str, as_of_date: pd.Timestamp, half_life_days: float) -> float:
    try:
        dt = pd.Timestamp(item_date)
    except Exception:
        return 1.0
    days = max(0.0, float((as_of_date.normalize() - dt.normalize()).days))
    return float(0.5 ** (days / max(float(half_life_days), 1e-6)))


def _item_weight(item: InfoItem, as_of_date: pd.Timestamp, half_life_days: float, carry: float = 1.0) -> float:
    strength = _clip(_safe_float(item.strength, 3.0) / 5.0, 0.0, 1.0)
    confidence = _clip(_safe_float(item.confidence, 0.7), 0.0, 1.0)
    source_weight = _clip(_safe_float(item.source_weight, 0.7), 0.0, 1.0)
    freshness = _freshness_decay(item.date, as_of_date, half_life_days)
    return float(strength * confidence * source_weight * freshness * max(0.0, float(carry)))


def _is_short_horizon(horizon: str) -> bool:
    return str(horizon).strip().lower() in {"short", "both"}


def _is_mid_horizon(horizon: str) -> bool:
    return str(horizon).strip().lower() in {"mid", "both"}


def _negative_event_severity(item: InfoItem) -> float:
    sign = normalize_direction(item.direction)
    if sign >= 0 and item.event_tag not in _NEGATIVE_EVENT_TAGS:
        return 0.0
    if item.event_tag in {"trading_halt", "delisting_risk"}:
        base = 1.0
    elif item.event_tag in {"earnings_negative", "guidance_negative", "regulatory_negative"}:
        base = 0.8
    elif item.event_tag in {"contract_loss", "share_reduction"}:
        base = 0.6
    else:
        base = 0.35 if sign < 0 else 0.0
    return float(base)


def _aggregate_items(
    items: Iterable[tuple[InfoItem, float]],
    *,
    as_of_date: pd.Timestamp,
    half_life_days: float,
    info_strength: float,
) -> InfoAggregateState:
    short_total = 0.0
    short_score = 0.0
    mid_total = 0.0
    mid_score = 0.0
    item_count = 0
    announcement_count = 0
    research_count = 0
    negative_risk_total = 0.0
    negative_risk_weight = 0.0

    for item, carry in items:
        weight = _item_weight(item, as_of_date=as_of_date, half_life_days=half_life_days, carry=carry)
        if weight <= 1e-12:
            continue
        sign = float(normalize_direction(item.direction))
        if _is_short_horizon(item.horizon):
            short_total += weight
            short_score += weight * sign
        if _is_mid_horizon(item.horizon):
            mid_total += weight
            mid_score += weight * sign
        item_count += 1
        if item.info_type == "announcement":
            announcement_count += 1
        if item.info_type == "research":
            research_count += 1
        neg_severity = _negative_event_severity(item)
        if neg_severity > 0.0:
            negative_risk_total += weight * neg_severity
            negative_risk_weight += weight

    short_value = 0.0 if short_total <= 1e-12 else float(np.clip(short_score / short_total, -1.0, 1.0))
    mid_value = 0.0 if mid_total <= 1e-12 else float(np.clip(mid_score / mid_total, -1.0, 1.0))
    negative_event_risk = 0.0 if negative_risk_weight <= 1e-12 else float(
        np.clip(negative_risk_total / negative_risk_weight, 0.0, 1.0)
    )
    return InfoAggregateState(
        short_score=short_value,
        mid_score=mid_value,
        item_count=int(item_count),
        announcement_count=int(announcement_count),
        research_count=int(research_count),
        negative_event_risk=negative_event_risk,
        coverage_ratio=1.0 if item_count > 0 else 0.0,
        info_prob_1d=float(blend_probability(0.5, short_value, sentiment_strength=info_strength)),
        info_prob_5d=float(blend_probability(0.5, short_value, sentiment_strength=0.9 * float(info_strength))),
        info_prob_20d=float(blend_probability(0.5, mid_value, sentiment_strength=info_strength)),
        shadow_prob_1d=0.5,
        shadow_prob_5d=0.5,
        shadow_prob_20d=0.5,
    )


def build_info_state_maps(
    *,
    info_items: Iterable[InfoItem],
    as_of_date: pd.Timestamp,
    stock_symbols: Iterable[str],
    sector_map: dict[str, str],
    market_to_stock_carry: float,
    info_half_life_days: float,
    market_info_strength: float,
    stock_info_strength: float,
) -> tuple[InfoAggregateState, dict[str, InfoAggregateState], dict[str, InfoAggregateState]]:
    items = list(info_items)
    market_items = [(item, 1.0) for item in items if item.target_type == "market" and item.target == "MARKET"]
    market_state = _aggregate_items(
        market_items,
        as_of_date=as_of_date,
        half_life_days=info_half_life_days,
        info_strength=market_info_strength,
    )

    stock_states: dict[str, InfoAggregateState] = {}
    sector_buckets: dict[str, list[InfoAggregateState]] = {}
    symbols = [str(symbol) for symbol in stock_symbols]
    stock_targeted_total = sum(1 for item in items if item.target_type == "stock")
    for symbol in symbols:
        direct_items = [(item, 1.0) for item in items if item.target_type == "stock" and item.target == symbol]
        carried_market = [(item, market_to_stock_carry * carry) for item, carry in market_items]
        stock_state = _aggregate_items(
            [*direct_items, *carried_market],
            as_of_date=as_of_date,
            half_life_days=info_half_life_days,
            info_strength=stock_info_strength,
        )
        coverage_ratio = 0.0
        if stock_targeted_total > 0:
            coverage_ratio = float(np.clip(stock_state.item_count / stock_targeted_total, 0.0, 1.0))
        stock_state = InfoAggregateState(
            **{
                **stock_state.__dict__,
                "coverage_ratio": coverage_ratio,
            }
        )
        stock_states[symbol] = stock_state
        sector_buckets.setdefault(str(sector_map.get(symbol, "其他")), []).append(stock_state)

    sector_states: dict[str, InfoAggregateState] = {}
    for sector, bucket in sector_buckets.items():
        if not bucket:
            continue
        item_count = int(sum(item.item_count for item in bucket))
        covered = int(sum(1 for item in bucket if item.item_count > 0))
        announcement_count = int(sum(item.announcement_count for item in bucket))
        research_count = int(sum(item.research_count for item in bucket))
        weight = np.asarray([max(1.0, float(item.item_count)) for item in bucket], dtype=float)
        short_score = float(np.average([item.short_score for item in bucket], weights=weight))
        mid_score = float(np.average([item.mid_score for item in bucket], weights=weight))
        negative_event_risk = float(np.average([item.negative_event_risk for item in bucket], weights=weight))
        info_prob_1d = float(np.average([item.info_prob_1d for item in bucket], weights=weight))
        info_prob_5d = float(np.average([item.info_prob_5d for item in bucket], weights=weight))
        info_prob_20d = float(np.average([item.info_prob_20d for item in bucket], weights=weight))
        sector_states[sector] = InfoAggregateState(
            short_score=short_score,
            mid_score=mid_score,
            item_count=item_count,
            announcement_count=announcement_count,
            research_count=research_count,
            negative_event_risk=negative_event_risk,
            coverage_ratio=float(covered / max(1, len(bucket))),
            info_prob_1d=info_prob_1d,
            info_prob_5d=info_prob_5d,
            info_prob_20d=info_prob_20d,
            shadow_prob_1d=0.5,
            shadow_prob_5d=0.5,
            shadow_prob_20d=0.5,
        )
    return market_state, sector_states, stock_states


def event_tag_counts(items: Iterable[InfoItem]) -> dict[str, int]:
    counter = Counter(item.event_tag for item in items if str(item.event_tag).strip())
    return {str(key): int(value) for key, value in sorted(counter.items())}


def top_negative_events(
    items: Iterable[InfoItem],
    *,
    as_of_date: pd.Timestamp,
    half_life_days: float,
    top_n: int = 5,
) -> list[InfoSignalRecord]:
    rows: list[tuple[float, InfoSignalRecord]] = []
    for item in items:
        severity = _negative_event_severity(item)
        if severity <= 0.0:
            continue
        score = _item_weight(item, as_of_date=as_of_date, half_life_days=half_life_days) * severity
        rows.append(
            (
                float(score),
                InfoSignalRecord(
                    target=item.target,
                    target_name=item.target,
                    title=item.title,
                    info_type=item.info_type,
                    direction=item.direction,
                    horizon=item.horizon,
                    event_tag=item.event_tag,
                    score=-float(score),
                    negative_event_risk=float(severity),
                    source_url=item.source_url,
                ),
            )
        )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in rows[: max(0, int(top_n))]]


def top_positive_stock_signals(
    state: CompositeState,
    *,
    symbol_names: dict[str, str],
    top_n: int = 5,
) -> list[InfoSignalRecord]:
    rows: list[InfoSignalRecord] = []
    for stock in state.stocks:
        info_state = state.stock_info_states.get(stock.symbol, InfoAggregateState())
        score = 0.45 * float(info_state.mid_score) + 0.35 * float(info_state.short_score) - 0.20 * float(info_state.negative_event_risk)
        if score <= 0.0 or info_state.item_count <= 0:
            continue
        rows.append(
            InfoSignalRecord(
                target=stock.symbol,
                target_name=str(symbol_names.get(stock.symbol, stock.symbol)),
                title=f"{str(symbol_names.get(stock.symbol, stock.symbol))} info shadow",
                info_type="mixed",
                direction="bullish",
                horizon="both",
                score=float(score),
                negative_event_risk=float(info_state.negative_event_risk),
                source_url="",
            )
        )
    rows.sort(key=lambda item: (item.score, -item.negative_event_risk), reverse=True)
    return rows[: max(0, int(top_n))]


def quant_info_divergence_rows(
    state: CompositeState,
    *,
    symbol_names: dict[str, str],
    top_n: int = 5,
) -> list[InfoDivergenceRecord]:
    rows: list[InfoDivergenceRecord] = []
    for stock in state.stocks:
        info_state = state.stock_info_states.get(stock.symbol, InfoAggregateState())
        gap = abs(float(stock.up_20d_prob) - float(info_state.info_prob_20d))
        if gap <= 1e-6:
            continue
        rows.append(
            InfoDivergenceRecord(
                symbol=stock.symbol,
                name=str(symbol_names.get(stock.symbol, stock.symbol)),
                quant_prob_20d=float(stock.up_20d_prob),
                info_prob_20d=float(info_state.info_prob_20d),
                shadow_prob_20d=float(info_state.shadow_prob_20d),
                gap=float(gap),
            )
        )
    rows.sort(key=lambda item: item.gap, reverse=True)
    return rows[: max(0, int(top_n))]
