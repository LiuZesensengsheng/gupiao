from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from src.application.v2_contracts import (
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    MainlineState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _is_actionable_status(status: str) -> bool:
    return str(status) not in {"halted", "delisted"}


def _sector_priority(
    sector: SectorForecastState,
    *,
    strategy_mode: str,
    risk_regime: str,
) -> float:
    trend_term = 0.58 * max(0.0, float(sector.up_20d_prob) - 0.50)
    strength_term = 0.62 * max(0.0, float(sector.relative_strength))
    rotation_term = 0.18 * float(sector.rotation_speed)
    crowding_penalty = 0.12 * max(0.0, float(sector.crowding_score) - 0.55)
    if strategy_mode == "range_rotation":
        rotation_term *= 1.30
    elif strategy_mode == "defensive":
        crowding_penalty *= 1.15
    if risk_regime == "risk_off":
        crowding_penalty *= 1.20
        rotation_term *= 0.85
    return float(max(0.0, trend_term + strength_term + rotation_term - crowding_penalty))


def _regime_sector_cap(*, strategy_mode: str, risk_regime: str) -> int:
    if risk_regime == "risk_off":
        return 2
    if risk_regime == "cautious":
        return 3 if strategy_mode != "defensive" else 2
    return 4 if strategy_mode != "defensive" else 3


def _shortlist_cap(*, total_scored: int, risk_regime: str, strategy_mode: str) -> int:
    if total_scored <= 24:
        return total_scored
    if risk_regime == "risk_off":
        cap = min(max(12, total_scored // 10), 24)
    elif risk_regime == "cautious":
        cap = min(max(16, total_scored // 8), 36)
    else:
        cap = min(max(20, total_scored // 6), 48)
    if strategy_mode == "range_rotation":
        cap = min(56, cap + 4)
    if total_scored <= 80:
        cap = min(total_scored, max(cap, total_scored // 2))
    return int(max(1, cap))


def _stock_fragility(stock: StockForecastState) -> float:
    up_1d = float(getattr(stock, "up_1d_prob", 0.5))
    up_5d = float(getattr(stock, "up_5d_prob", 0.5))
    up_20d = float(getattr(stock, "up_20d_prob", 0.5))
    excess_vs_sector = float(getattr(stock, "excess_vs_sector_prob", 0.5))
    tradeability = float(getattr(stock, "tradeability_score", 0.5))
    event_impact = float(getattr(stock, "event_impact_score", 0.5))
    reversal_penalty = max(0.0, up_1d - max(up_5d, up_20d))
    weak_mid_penalty = max(0.0, 0.53 - up_20d)
    low_excess_penalty = max(0.0, 0.52 - excess_vs_sector)
    tradeability_penalty = max(0.0, 0.82 - tradeability)
    event_penalty = max(0.0, 0.10 - event_impact)
    return float(
        _clip(
            0.34 * reversal_penalty / 0.12
            + 0.24 * weak_mid_penalty / 0.10
            + 0.16 * low_excess_penalty / 0.08
            + 0.14 * tradeability_penalty / 0.12
            + 0.12 * event_penalty / 0.10,
            0.0,
            1.0,
        )
    )


def _stock_selection_priority(stock: StockForecastState, *, stock_score: float, risk_penalty: float) -> float:
    up_5d = float(getattr(stock, "up_5d_prob", 0.5))
    up_20d = float(getattr(stock, "up_20d_prob", 0.5))
    excess_vs_sector = float(getattr(stock, "excess_vs_sector_prob", 0.5))
    tradeability = float(getattr(stock, "tradeability_score", 0.5))
    stability_bonus = (
        0.40 * max(0.0, up_20d - 0.50)
        + 0.28 * max(0.0, up_5d - 0.50)
        + 0.18 * max(0.0, excess_vs_sector - 0.50)
        + 0.14 * max(0.0, tradeability - 0.78)
    )
    return float(stock_score + stability_bonus - risk_penalty)


def _mainline_priority_maps(mainlines: Iterable[MainlineState]) -> tuple[dict[str, float], dict[str, float]]:
    sector_boosts: dict[str, float] = {}
    symbol_boosts: dict[str, float] = {}
    rows = list(mainlines or [])[:3]
    top_conviction = max((float(getattr(item, "conviction", 0.0)) for item in rows), default=0.0)
    cutoff = max(0.28, top_conviction - 0.08)
    for rank, mainline in enumerate(rows):
        conviction = float(getattr(mainline, "conviction", 0.0))
        event_risk = float(getattr(mainline, "event_risk_level", 0.0))
        if conviction < cutoff or event_risk >= 0.60:
            continue
        boost = float(
            _clip(
                0.08
                + 0.24 * max(0.0, conviction - cutoff)
                + 0.06 * float(getattr(mainline, "leadership", 0.0))
                + 0.05 * float(getattr(mainline, "catalyst_strength", 0.0))
                - 0.03 * rank,
                0.04,
                0.22,
            )
        )
        for sector in getattr(mainline, "sectors", []):
            sector_key = str(sector)
            sector_boosts[sector_key] = max(sector_boosts.get(sector_key, 0.0), boost)
        for symbol in getattr(mainline, "representative_symbols", []):
            symbol_key = str(symbol)
            symbol_boosts[symbol_key] = max(symbol_boosts.get(symbol_key, 0.0), boost + 0.03)
    return sector_boosts, symbol_boosts


def _sector_stock_support(
    grouped: dict[str, list[tuple[StockForecastState, float, float]]],
    *,
    median_score: float,
) -> dict[str, float]:
    support: dict[str, float] = {}
    strong_cut = max(0.56, float(median_score) + 0.02)
    for sector, items in grouped.items():
        if not items:
            support[sector] = 0.0
            continue
        raw_scores = sorted((float(score) for _, _, score in items), reverse=True)
        top_scores = raw_scores[: min(3, len(raw_scores))]
        top_mean = float(np.mean(top_scores)) if top_scores else 0.0
        strong_window = raw_scores[: min(5, len(raw_scores))]
        strong_ratio = float(
            sum(1 for score in strong_window if score >= strong_cut) / max(1, len(strong_window))
        )
        resilience = float(
            np.mean(
                [
                    1.0 - 0.65 * _stock_fragility(stock)
                    for stock, _, _ in items[: min(3, len(items))]
                ]
            )
        )
        support[sector] = float(
            _clip(
                0.55 * max(0.0, top_mean - max(0.54, float(median_score))) / 0.10
                + 0.25 * strong_ratio
                + 0.20 * resilience,
                0.0,
                1.0,
            )
        )
    return support


def _minimum_shortlist_size(
    *,
    total_scored: int,
    shortlist_cap: int,
    risk_regime: str,
    quant_breadth: float,
    breadth_strength: float,
) -> int:
    if total_scored <= 24:
        return total_scored
    if risk_regime == "risk_off":
        floor = 4 if total_scored >= 120 else 3
        if quant_breadth >= 0.08 or breadth_strength >= 0.08:
            floor += 1
    elif risk_regime == "cautious":
        floor = 6 if total_scored >= 120 else 4
    else:
        floor = 8 if total_scored >= 120 else 5
    return int(min(shortlist_cap, max(1, floor)))


def candidate_risk_snapshot(stocks: Iterable[StockForecastState]) -> dict[str, float]:
    rows = list(stocks)
    if not rows:
        return {"fragile_ratio": 0.0, "reversal_ratio": 0.0, "durability_score": 0.0}
    fragilities = [_stock_fragility(stock) for stock in rows]
    reversal_ratio = float(
        sum(
            1
            for stock in rows
            if float(getattr(stock, "up_1d_prob", 0.5))
            > max(float(getattr(stock, "up_5d_prob", 0.5)), float(getattr(stock, "up_20d_prob", 0.5))) + 0.02
        )
        / max(1, len(rows))
    )
    durability = float(
        np.mean(
            [
                0.44 * float(getattr(stock, "up_20d_prob", 0.5))
                + 0.26 * float(getattr(stock, "up_5d_prob", 0.5))
                + 0.15 * float(getattr(stock, "excess_vs_sector_prob", 0.5))
                + 0.15 * float(getattr(stock, "tradeability_score", 0.5))
                - 0.10 * _stock_fragility(stock)
                for stock in rows
            ]
        )
    )
    return {
        "fragile_ratio": float(sum(1 for value in fragilities if value >= 0.45) / max(1, len(fragilities))),
        "reversal_ratio": reversal_ratio,
        "durability_score": durability,
    }


def build_candidate_selection_state(
    *,
    market: MarketForecastState,
    cross_section: CrossSectionForecastState,
    sectors: Iterable[SectorForecastState],
    stocks: Iterable[StockForecastState],
    mainlines: Iterable[MainlineState] | None = None,
    strategy_mode: str,
    risk_regime: str,
    stock_score_fn: Callable[[StockForecastState], float],
) -> CandidateSelectionState:
    stock_rows = list(stocks)
    actionable = [
        stock for stock in stock_rows
        if _is_actionable_status(getattr(stock, "tradability_status", "normal"))
    ]
    rows = actionable or stock_rows
    total_scored = int(len(stock_rows))
    if not rows:
        return CandidateSelectionState(total_scored=total_scored)
    compact_universe = len(rows) <= 24
    stressed_tape = (
        str(getattr(market, "volatility_regime", "")) == "high"
        or float(getattr(market, "drawdown_risk", 0.0)) >= 0.35
        or float(getattr(cross_section, "weak_stock_ratio", 0.0)) >= 0.48
    )
    if compact_universe and not stressed_tape:
        symbols = [str(stock.symbol) for stock in rows]
        return CandidateSelectionState(
            shortlisted_symbols=symbols,
            shortlisted_sectors=list(dict.fromkeys(str(stock.sector) for stock in rows)),
            total_scored=total_scored,
            shortlist_size=len(symbols),
            shortlist_ratio=1.0 if total_scored else 0.0,
            selection_mode="full_universe",
            selection_notes=["Universe already compact; full stock list kept actionable."],
        )

    sector_rows = list(sectors)
    grouped: dict[str, list[tuple[StockForecastState, float, float]]] = {}
    all_scores: list[float] = []
    risk_penalty_multiplier = 1.0
    if str(getattr(market, "volatility_regime", "")) == "high":
        risk_penalty_multiplier += 0.35
    if float(getattr(market, "drawdown_risk", 0.0)) >= 0.35:
        risk_penalty_multiplier += 0.20
    if float(getattr(cross_section, "weak_stock_ratio", 0.0)) >= 0.48:
        risk_penalty_multiplier += 0.25
    if float(getattr(cross_section, "breadth_strength", 0.0)) <= 0.08:
        risk_penalty_multiplier += 0.15
    for stock in rows:
        score = float(stock_score_fn(stock))
        fragility = _stock_fragility(stock)
        priority = _stock_selection_priority(
            stock,
            stock_score=score,
            risk_penalty=float(risk_penalty_multiplier) * fragility,
        )
        grouped.setdefault(str(stock.sector), []).append((stock, priority, score))
        all_scores.append(score)
    for items in grouped.values():
        items.sort(key=lambda item: item[1], reverse=True)

    median_score = float(np.median(all_scores)) if all_scores else 0.0
    strong_cut = max(0.56, median_score + (0.025 if total_scored >= 80 else 0.015))
    quant_breadth = float(sum(1 for score in all_scores if score >= strong_cut) / max(1, len(all_scores)))
    sector_stock_support = _sector_stock_support(grouped, median_score=median_score)

    sector_priority = {
        str(sector.sector): _sector_priority(sector, strategy_mode=strategy_mode, risk_regime=risk_regime)
        for sector in sector_rows
    }
    ranked_sectors = [
        sector for sector in sector_rows
        if grouped.get(str(sector.sector))
    ]
    ranked_sectors.sort(
        key=lambda sector: (
            float(sector_stock_support.get(str(sector.sector), 0.0)),
            float(sector_priority.get(str(sector.sector), 0.0)),
            float(sector.up_20d_prob),
            float(sector.relative_strength),
        ),
        reverse=True,
    )
    if not ranked_sectors:
        symbols = [str(stock.symbol) for stock, _ in sorted(grouped.get("other", []), key=lambda item: item[1], reverse=True)]
        return CandidateSelectionState(
            shortlisted_symbols=symbols,
            total_scored=total_scored,
            shortlist_size=len(symbols),
            shortlist_ratio=float(len(symbols) / max(1, total_scored)),
            selection_mode="fallback_shortlist",
            selection_notes=["Sector metadata missing; fallback to ranked stock list."],
        )

    mainline_sector_boosts, mainline_symbol_boosts = _mainline_priority_maps(mainlines or [])
    strong_mainline_sectors = {sector for sector, boost in mainline_sector_boosts.items() if boost >= 0.08}
    representative_symbols = {symbol for symbol, boost in mainline_symbol_boosts.items() if boost >= 0.10}
    if strong_mainline_sectors:
        ranked_sectors.sort(
            key=lambda sector: (
                float(mainline_sector_boosts.get(str(sector.sector), 0.0)),
                float(sector_stock_support.get(str(sector.sector), 0.0)),
                float(sector_priority.get(str(sector.sector), 0.0)),
                float(sector.up_20d_prob),
                float(sector.relative_strength),
            ),
            reverse=True,
        )

    max_sectors = min(len(ranked_sectors), _regime_sector_cap(strategy_mode=strategy_mode, risk_regime=risk_regime))
    if str(getattr(market, "volatility_regime", "")) == "high" or float(getattr(market, "drawdown_risk", 0.0)) >= 0.35:
        max_sectors = max(1, max_sectors - 1)
    shortlist_cap = _shortlist_cap(total_scored=total_scored, risk_regime=risk_regime, strategy_mode=strategy_mode)
    if str(getattr(market, "volatility_regime", "")) == "high":
        shortlist_cap = max(10, int(shortlist_cap * 0.8))
    if float(getattr(cross_section, "weak_stock_ratio", 0.0)) >= 0.48:
        shortlist_cap = max(10, int(shortlist_cap * 0.85))
    supported_sector_count = int(sum(1 for value in sector_stock_support.values() if value >= 0.58))
    if (
        risk_regime == "risk_off"
        and total_scored >= 120
        and supported_sector_count >= 3
        and float(getattr(cross_section, "breadth_strength", 0.0)) >= 0.05
    ):
        max_sectors = min(len(ranked_sectors), max_sectors + 1)
    if (
        risk_regime == "risk_off"
        and total_scored >= 200
        and supported_sector_count >= 5
        and float(getattr(cross_section, "weak_stock_ratio", 0.0)) <= 0.52
    ):
        max_sectors = min(len(ranked_sectors), max_sectors + 1)
    base_gate = max(0.53, median_score + (0.01 if total_scored >= 80 else 0.0))
    if str(getattr(market, "volatility_regime", "")) == "high":
        base_gate += 0.015
    if float(getattr(market, "drawdown_risk", 0.0)) >= 0.35:
        base_gate += 0.010
    if float(getattr(cross_section, "weak_stock_ratio", 0.0)) >= 0.48:
        base_gate += 0.010

    eligible_sectors: list[str] = []
    for idx, sector in enumerate(ranked_sectors):
        priority = float(sector_priority.get(str(sector.sector), 0.0))
        stock_support = float(sector_stock_support.get(str(sector.sector), 0.0))
        if (
            idx == 0
            or str(sector.sector) in strong_mainline_sectors
            or priority >= 0.03
            or stock_support >= 0.60
        ):
            eligible_sectors.append(str(sector.sector))
        if len(eligible_sectors) >= max_sectors:
            break
    if not eligible_sectors:
        eligible_sectors.append(str(ranked_sectors[0].sector))

    slots = {sector: 1 for sector in eligible_sectors}
    remaining_slots = max(0, shortlist_cap - len(eligible_sectors))
    per_sector_cap = max(2, min(6, shortlist_cap // max(1, len(eligible_sectors))))
    if str(getattr(market, "volatility_regime", "")) == "high":
        per_sector_cap = max(2, per_sector_cap - 1)
    if strong_mainline_sectors:
        per_sector_cap = min(max(per_sector_cap, 3), max(3, per_sector_cap))
    while remaining_slots > 0:
        expandable = [
            sector for sector in eligible_sectors
            if slots[sector] < min(per_sector_cap, len(grouped.get(sector, [])))
        ]
        if not expandable:
            break
        best_sector = max(
            expandable,
            key=lambda sector: (
                float(mainline_sector_boosts.get(sector, 0.0)),
                float(sector_stock_support.get(sector, 0.0)),
                float(sector_priority.get(sector, 0.0)) / float(slots[sector] + 0.35),
                len(grouped.get(sector, [])),
            ),
        )
        slots[best_sector] += 1
        remaining_slots -= 1

    selected_symbols: list[str] = []
    notes = [f"Macro shortlist active: {len(eligible_sectors)} sectors prioritized before fine ranking."]
    if strong_mainline_sectors:
        notes.append("Mainline focus: " + ", ".join(sorted(str(sector) for sector in strong_mainline_sectors)[:3]))
    for sector in eligible_sectors:
        priority = float(sector_priority.get(sector, 0.0))
        ordered = grouped.get(sector, [])
        if not ordered:
            continue
        sector_gate = max(0.51, base_gate - 0.03 * min(1.0, priority / 0.20))
        sector_gate -= min(0.035, 0.05 * float(sector_stock_support.get(sector, 0.0)))
        if sector in strong_mainline_sectors:
            sector_gate -= min(0.03, float(mainline_sector_boosts.get(sector, 0.0)) * 0.18)
        elif strong_mainline_sectors:
            sector_gate += 0.01
        sector_selected = 0
        for idx, (stock, score, raw_score) in enumerate(ordered):
            if sector_selected >= slots[sector]:
                break
            score += float(mainline_symbol_boosts.get(str(stock.symbol), 0.0))
            if idx > 0 and score + 1e-9 < sector_gate:
                continue
            if idx > 0 and _stock_fragility(stock) >= (0.60 if raw_score >= strong_cut else 0.55):
                continue
            selected_symbols.append(str(stock.symbol))
            sector_selected += 1
    if not selected_symbols:
        selected_symbols = [str(stock.symbol) for stock, _, _ in sorted(
            (item for items in grouped.values() for item in items),
            key=lambda item: item[1],
            reverse=True,
        )[:shortlist_cap]]
        notes.append("Shortlist fell back to raw stock ranking because sector gates were too strict.")

    minimum_shortlist = _minimum_shortlist_size(
        total_scored=total_scored,
        shortlist_cap=shortlist_cap,
        risk_regime=risk_regime,
        quant_breadth=quant_breadth,
        breadth_strength=float(getattr(cross_section, "breadth_strength", 0.0)),
    )
    if len(selected_symbols) < minimum_shortlist:
        selected_set = set(selected_symbols)
        global_ranked = sorted(
            (item for items in grouped.values() for item in items),
            key=lambda item: (
                float(mainline_symbol_boosts.get(str(item[0].symbol), 0.0)),
                float(item[1]),
                float(item[2]),
            ),
            reverse=True,
        )
        added = 0
        for stock, _, raw_score in global_ranked:
            symbol = str(stock.symbol)
            sector = str(stock.sector)
            if symbol in selected_set:
                continue
            fragility = _stock_fragility(stock)
            support = float(sector_stock_support.get(sector, 0.0))
            if fragility >= 0.70:
                continue
            if sector not in eligible_sectors and support < 0.55 and raw_score < max(base_gate - 0.02, strong_cut - 0.01):
                continue
            selected_symbols.append(symbol)
            selected_set.add(symbol)
            slots[sector] = int(slots.get(sector, 0)) + 1
            added += 1
            if len(selected_symbols) >= minimum_shortlist or len(selected_symbols) >= shortlist_cap:
                break
        if added > 0:
            notes.append(
                f"Quant breadth fill added {added} names beyond macro-sector core to avoid an over-compressed shortlist."
            )

    macro_core_symbols = list(dict.fromkeys(selected_symbols))[:shortlist_cap]
    global_ranked_symbols = [
        str(stock.symbol)
        for stock, _, _ in sorted(
            (item for items in grouped.values() for item in items),
            key=lambda item: (
                float(mainline_symbol_boosts.get(str(item[0].symbol), 0.0)),
                float(sector_stock_support.get(str(item[0].sector), 0.0)),
                float(item[1]),
                float(item[2]),
            ),
            reverse=True,
        )
    ]
    selected_set = set(macro_core_symbols)
    selected_symbols = list(macro_core_symbols)
    selected_symbols.extend(symbol for symbol in global_ranked_symbols if symbol not in selected_set)
    notes.append(
        f"Macro ranking keeps all actionable names ordered; top {len(macro_core_symbols)} names receive sector-prioritized placement."
    )
    symbol_sector_map = {str(stock.symbol): str(stock.sector) for stock in stock_rows}
    shortlisted_sectors = list(
        dict.fromkeys(
            symbol_sector_map.get(symbol, "")
            for symbol in macro_core_symbols
            if symbol_sector_map.get(symbol, "")
        )
    )
    shortlist_size = int(len(macro_core_symbols))
    return CandidateSelectionState(
        shortlisted_symbols=selected_symbols,
        shortlisted_sectors=shortlisted_sectors,
        sector_slots={sector: int(slots.get(sector, 0)) for sector in shortlisted_sectors},
        total_scored=total_scored,
        shortlist_size=shortlist_size,
        shortlist_ratio=float(shortlist_size / max(1, total_scored)),
        selection_mode="macro_sector_ranking",
        selection_notes=notes,
    )


def candidate_stocks_from_state(state: CompositeState) -> list[StockForecastState]:
    selection = getattr(state, "candidate_selection", CandidateSelectionState())
    symbols = [str(symbol) for symbol in getattr(selection, "shortlisted_symbols", []) if str(symbol).strip()]
    if not symbols:
        return list(getattr(state, "stocks", []))
    shortlist_size = int(getattr(selection, "shortlist_size", 0) or 0)
    if shortlist_size > 0:
        symbols = symbols[:shortlist_size]
    order = {symbol: idx for idx, symbol in enumerate(symbols)}
    shortlist = [stock for stock in getattr(state, "stocks", []) if stock.symbol in order]
    shortlist.sort(key=lambda stock: order.get(stock.symbol, len(order)))
    return shortlist or list(getattr(state, "stocks", []))
