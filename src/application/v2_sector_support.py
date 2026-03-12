from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from src.application.v2_contracts import SectorForecastState, StockForecastState


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _is_actionable_status(status: str) -> bool:
    return str(status) not in {"halted", "delisted"}


def build_sector_states(
    stocks: Iterable[StockForecastState],
    *,
    stock_score_fn: Callable[[StockForecastState], float],
) -> list[SectorForecastState]:
    grouped: dict[str, list[StockForecastState]] = {}
    for stock in stocks:
        grouped.setdefault(str(stock.sector), []).append(stock)

    sector_states: list[SectorForecastState] = []
    for sector, items in grouped.items():
        actionable = [item for item in items if _is_actionable_status(getattr(item, "tradability_status", "normal"))]
        rows = actionable or list(items)
        if not rows:
            continue
        n = max(1, len(rows))
        up5 = float(sum(float(item.up_5d_prob) for item in rows) / n)
        up20 = float(sum(float(item.up_20d_prob) for item in rows) / n)
        base_rel = float(sum(float(item.up_20d_prob) - 0.5 for item in rows) / n)
        alpha_scores = sorted((float(stock_score_fn(item)) for item in rows), reverse=True)
        top_slice = alpha_scores[: min(2, len(alpha_scores))]
        top_alpha = float(sum(top_slice) / max(1, len(top_slice)))
        median_alpha = float(np.median(alpha_scores))
        alpha_breadth = float(sum(1 for score in alpha_scores if score >= max(0.56, median_alpha + 0.02)) / max(1, len(alpha_scores)))
        leadership = max(0.0, top_alpha - 0.50)
        dispersion = float(np.mean([abs(score - median_alpha) for score in alpha_scores])) if alpha_scores else 0.0
        rotation = float(
            np.mean([abs(float(item.up_5d_prob) - float(item.up_20d_prob)) for item in rows])
            + 0.30 * alpha_breadth
            + 0.18 * min(1.0, len(rows) / 4.0)
        )
        crowding = float(
            0.50 * max(0.0, up20 - 0.50)
            + 0.25 * leadership
            + 0.15 * alpha_breadth
            + 0.10 * max(0.0, 0.12 - dispersion) / 0.12
        )
        relative_strength = float(base_rel + 0.35 * leadership + 0.12 * alpha_breadth - 0.08 * crowding)
        sector_states.append(
            SectorForecastState(
                sector=sector,
                up_5d_prob=up5,
                up_20d_prob=up20,
                relative_strength=float(_clip(relative_strength, -0.35, 0.65)),
                rotation_speed=float(_clip(rotation, 0.0, 1.0)),
                crowding_score=float(_clip(crowding, 0.0, 1.0)),
            )
        )
    return sector_states


def ranked_sector_budgets_with_alpha(
    *,
    sectors: Iterable[SectorForecastState],
    stocks: Iterable[StockForecastState],
    target_exposure: float,
    stock_score_fn: Callable[[StockForecastState], float],
) -> dict[str, float]:
    sector_rows = list(sectors)
    if not sector_rows:
        return {}

    stock_rows = [stock for stock in stocks if _is_actionable_status(getattr(stock, "tradability_status", "normal"))]
    sector_alpha: dict[str, list[float]] = {}
    for stock in stock_rows:
        sector_alpha.setdefault(str(stock.sector), []).append(float(stock_score_fn(stock)))

    raw_scores: list[float] = []
    for item in sector_rows:
        base_trend = max(0.0, float(item.up_20d_prob) - 0.50)
        base_score = (
            0.70 * base_trend
            + 0.65 * max(0.0, float(item.relative_strength))
            + 0.18 * float(item.rotation_speed)
        )
        stock_scores = sorted(sector_alpha.get(item.sector, []), reverse=True)
        alpha_top = 0.0
        alpha_breadth = 0.0
        if stock_scores:
            top_slice = stock_scores[: min(2, len(stock_scores))]
            alpha_top = float(sum(top_slice) / max(1, len(top_slice)))
            alpha_breadth = float(sum(1 for score in stock_scores if score >= 0.56) / max(1, len(stock_scores)))
        alpha_score = max(0.0, alpha_top - 0.50) + 0.22 * alpha_breadth
        if float(item.up_20d_prob) < 0.50 and float(item.relative_strength) < 0.0 and alpha_top < 0.60:
            alpha_score *= 0.55
        crowding_penalty = 0.12 * max(0.0, float(item.crowding_score) - 0.55)
        raw = max(0.0, base_score + 0.55 * alpha_score - crowding_penalty)
        raw_scores.append(raw)

    total = float(sum(raw_scores))
    if total <= 1e-9:
        equal = float(target_exposure) / float(len(sector_rows))
        return {item.sector: equal for item in sector_rows}
    return {
        item.sector: float(target_exposure) * float(score) / total
        for item, score in zip(sector_rows, raw_scores)
    }


def cap_sector_budgets(
    *,
    sector_budgets: dict[str, float],
    target_exposure: float,
    risk_regime: str,
    breadth_strength: float,
) -> tuple[dict[str, float], list[str]]:
    if not sector_budgets:
        return {}, []
    notes: list[str] = []
    active = {sector: max(0.0, float(weight)) for sector, weight in sector_budgets.items() if float(weight) > 1e-9}
    if not active:
        return {}, notes
    if len(active) == 1 and "其他" in active:
        return active, notes
    if risk_regime == "risk_on":
        cap_ratio = 0.68
    elif risk_regime == "cautious":
        cap_ratio = 0.58
    else:
        cap_ratio = 0.50
    if float(breadth_strength) < 0.10:
        cap_ratio = min(cap_ratio, 0.50)
    max_sector_weight = float(target_exposure) * float(cap_ratio)
    if max_sector_weight <= 1e-9:
        return active, notes
    clipped = dict(active)
    overflow = 0.0
    for sector, weight in list(clipped.items()):
        if weight > max_sector_weight + 1e-9:
            overflow += float(weight - max_sector_weight)
            clipped[sector] = float(max_sector_weight)
            notes.append(f"{sector}: sector budget capped for concentration control.")
    if len(clipped) == 1:
        return clipped, notes
    if overflow <= 1e-9:
        return clipped, notes
    receivers = [sector for sector, weight in clipped.items() if weight < max_sector_weight - 1e-9]
    while overflow > 1e-9 and receivers:
        headroom_total = float(sum(max(0.0, max_sector_weight - clipped[sector]) for sector in receivers))
        if headroom_total <= 1e-9:
            break
        for sector in list(receivers):
            headroom = max(0.0, max_sector_weight - clipped[sector])
            if headroom <= 1e-9:
                continue
            take = min(headroom, overflow * headroom / headroom_total)
            clipped[sector] = float(clipped[sector] + take)
            overflow -= float(take)
            if overflow <= 1e-9:
                break
        receivers = [sector for sector, weight in clipped.items() if weight < max_sector_weight - 1e-9]
    total = float(sum(clipped.values()))
    if total > float(target_exposure) + 1e-9 and total > 1e-9:
        scale = float(target_exposure) / total
        clipped = {sector: float(weight) * scale for sector, weight in clipped.items()}
    return clipped, notes


def allocate_sector_slots(
    *,
    sector_budgets: dict[str, float],
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]],
    total_slots: int,
    sector_strengths: dict[str, float] | None = None,
) -> dict[str, int]:
    active_sectors = [
        sector for sector, budget in sector_budgets.items()
        if float(budget) > 1e-9 and available_by_sector.get(sector)
    ]
    if not active_sectors or total_slots <= 0:
        return {}

    strengths = dict(sector_strengths or {})
    slots = {sector: 0 for sector in active_sectors}
    ordered = sorted(
        active_sectors,
        key=lambda sector: (
            float(strengths.get(sector, 0.0)),
            float(sector_budgets.get(sector, 0.0)),
            len(available_by_sector.get(sector, [])),
        ),
        reverse=True,
    )

    for sector in ordered:
        if total_slots <= 0:
            break
        slots[sector] = 1
        total_slots -= 1

    while total_slots > 0:
        expandable = [
            sector for sector in ordered
            if slots[sector] < len(available_by_sector.get(sector, []))
        ]
        if not expandable:
            break
        best = max(
            expandable,
            key=lambda sector: (
                float(strengths.get(sector, 0.0)) + float(sector_budgets.get(sector, 0.0)) / float(max(1, slots[sector] + 1)),
                len(available_by_sector.get(sector, [])),
            ),
        )
        slots[best] += 1
        total_slots -= 1
    return slots


def allocate_with_sector_budgets(
    *,
    stocks: list[StockForecastState],
    sector_budgets: dict[str, float],
    target_position_count: int,
    stock_score_fn: Callable[[StockForecastState], float],
    sector_strengths: dict[str, float] | None = None,
    max_single_position: float = 0.35,
) -> dict[str, float]:
    sector_candidates: dict[str, list[tuple[StockForecastState, float]]] = {}
    active_budget_count = max(1, sum(1 for weight in sector_budgets.values() if float(weight) > 1e-9))
    avg_sector_budget = float(sum(max(0.0, float(weight)) for weight in sector_budgets.values()) / active_budget_count)
    strengths = dict(sector_strengths or {})
    for stock in stocks:
        if not _is_actionable_status(getattr(stock, "tradability_status", "normal")):
            continue
        score = float(stock_score_fn(stock))
        sector = str(stock.sector)
        sector_candidates.setdefault(sector, []).append((stock, score))
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]] = {}
    for sector, candidates in sector_candidates.items():
        ordered = sorted(candidates, key=lambda item: item[1], reverse=True)
        gate = 0.50
        sector_strength = float(strengths.get(sector, 0.0))
        sector_budget = float(sector_budgets.get(sector, 0.0))
        if sector_strength < 0.10 and sector_budget <= avg_sector_budget:
            gate = 0.57
        filtered = [pair for pair in ordered if pair[1] + 1e-9 >= gate]
        if not filtered and ordered and (sector_strength >= 0.45 or sector_budget > avg_sector_budget):
            filtered = [ordered[0]]
        if filtered:
            available_by_sector[sector] = filtered

    slots_by_sector = allocate_sector_slots(
        sector_budgets=sector_budgets,
        available_by_sector=available_by_sector,
        total_slots=max(1, int(target_position_count)),
        sector_strengths=strengths,
    )
    symbol_target_weights: dict[str, float] = {}
    for sector, slots in slots_by_sector.items():
        sector_budget = float(sector_budgets.get(sector, 0.0))
        picks = available_by_sector.get(sector, [])[: max(0, int(slots))]
        if not picks or sector_budget <= 1e-9:
            continue
        cap = min(float(max_single_position), float(sector_budget))
        remaining = float(sector_budget)
        uncapped = list(picks)
        while uncapped and remaining > 1e-9:
            sector_scores = [max(0.0, score - 0.50) for _, score in uncapped]
            sector_total = float(sum(sector_scores))
            if sector_total <= 1e-9:
                provisional = [remaining / float(len(uncapped))] * len(uncapped)
            else:
                provisional = [remaining * float(score) / sector_total for score in sector_scores]
            over_limit = [
                idx for idx, weight in enumerate(provisional)
                if float(weight) > cap + 1e-9
            ]
            if not over_limit:
                for (stock, _), weight in zip(uncapped, provisional):
                    symbol_target_weights[stock.symbol] = float(weight)
                remaining = 0.0
                break
            next_uncapped: list[tuple[StockForecastState, float]] = []
            for idx, pair in enumerate(uncapped):
                stock, score = pair
                if idx in over_limit:
                    symbol_target_weights[stock.symbol] = float(cap)
                    remaining -= float(cap)
                else:
                    next_uncapped.append((stock, score))
            uncapped = next_uncapped
    return symbol_target_weights
