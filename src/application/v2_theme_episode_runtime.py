from __future__ import annotations

import math

from src.application.v2_contracts import CompositeState, ThemeEpisode, Viewpoint


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _normalize_key(value: object) -> str:
    return str(value or "").strip().lower()


def _theme_viewpoints(
    *,
    theme: str,
    sectors: list[str],
    representative_symbols: list[str],
    viewpoints: list[Viewpoint],
) -> list[Viewpoint]:
    theme_key = _normalize_key(theme)
    sector_keys = {_normalize_key(item) for item in sectors}
    symbol_keys = {_normalize_key(item) for item in representative_symbols}
    matched: list[Viewpoint] = []
    for item in viewpoints:
        if _normalize_key(item.theme) == theme_key:
            matched.append(item)
            continue
        if item.target_type == "sector" and _normalize_key(item.target) in sector_keys:
            matched.append(item)
            continue
        if item.target_type == "stock" and _normalize_key(item.target) in symbol_keys:
            matched.append(item)
    return matched


def _avg_or_default(values: list[float], default: float = 0.0) -> float:
    if not values:
        return float(default)
    return float(sum(values) / max(1, len(values)))


def _phase_for_episode(
    *,
    conviction: float,
    breadth: float,
    leadership: float,
    catalyst_strength: float,
    event_risk: float,
    crowding: float,
    viewpoint_conflict: float,
) -> tuple[str, str]:
    if conviction < 0.48 or event_risk >= 0.55:
        return "fading", "conviction slipped or event risk crossed the defensive threshold."
    if conviction >= 0.70 and max(crowding, event_risk) >= 0.55:
        return "crowded", "high conviction is now paired with crowding or event risk."
    if breadth < 0.28 or leadership < 0.22 or viewpoint_conflict >= 0.30:
        return "diverging", "breadth or leadership is weakening and viewpoints are no longer aligned."
    if conviction >= 0.55 and breadth >= 0.28 and leadership >= 0.22:
        return "strengthening", "conviction is confirmed with breadth and leadership still intact."
    if conviction < 0.55 and catalyst_strength >= 0.18:
        return "emerging", "catalyst strength is rising before conviction is fully confirmed."
    return "emerging", "theme is building but has not yet fully confirmed."


def build_theme_episodes(
    *,
    state: CompositeState,
) -> list[ThemeEpisode]:
    episodes: list[ThemeEpisode] = []
    sector_map = {str(item.sector): item for item in getattr(state, "sectors", [])}
    market_info = getattr(state, "market_info_state", None)
    cross = getattr(state, "cross_section", None)
    capital_flow = getattr(state, "capital_flow_state", None)
    macro_context = getattr(state, "macro_context_state", None)
    viewpoints = list(getattr(state, "viewpoints", []) or [])

    for mainline in list(getattr(state, "mainlines", []) or []):
        sectors = [str(item) for item in getattr(mainline, "sectors", [])]
        representatives = [str(item) for item in getattr(mainline, "representative_symbols", [])]
        scoped_viewpoints = _theme_viewpoints(
            theme=str(mainline.name),
            sectors=sectors,
            representative_symbols=representatives,
            viewpoints=viewpoints,
        )
        bullish_score = float(
            sum(item.weight for item in scoped_viewpoints if str(item.direction).strip().lower() == "bullish")
        )
        bearish_score = float(
            sum(item.weight for item in scoped_viewpoints if str(item.direction).strip().lower() == "bearish")
        )
        total_score = float(bullish_score + bearish_score)
        viewpoint_score = float(bullish_score - bearish_score)
        viewpoint_conflict = float(min(bullish_score, bearish_score) / max(total_score, 1e-9))
        sector_rows = [sector_map[item] for item in sectors if item in sector_map]
        sector_crowding = _avg_or_default([float(item.crowding_score) for item in sector_rows], 0.0)
        sector_event_risk = _avg_or_default(
            [float(getattr(state.sector_info_states.get(item.sector, None), "event_risk_level", 0.0)) for item in sector_rows],
            0.0,
        )
        sector_catalyst = _avg_or_default(
            [float(getattr(state.sector_info_states.get(item.sector, None), "catalyst_strength", 0.0)) for item in sector_rows],
            0.0,
        )
        conviction = _clip01(float(mainline.conviction) + 0.18 * viewpoint_score)
        breadth = _clip01(
            0.62 * float(mainline.breadth)
            + 0.26 * float(getattr(cross, "breadth_strength", 0.0))
            + 0.12 * float(getattr(market_info, "coverage_ratio", 0.0))
        )
        leadership = _clip01(
            0.62 * float(mainline.leadership)
            + 0.28 * float(getattr(cross, "leader_participation", 0.0))
            - 0.12 * float(getattr(cross, "weak_stock_ratio", 0.0))
        )
        catalyst_strength = _clip01(
            max(
                float(mainline.catalyst_strength),
                float(getattr(market_info, "catalyst_strength", 0.0)),
                float(sector_catalyst),
                min(0.35, bullish_score),
            )
        )
        event_risk = _clip01(
            max(
                float(mainline.event_risk_level),
                float(getattr(market_info, "event_risk_level", 0.0)),
                float(sector_event_risk),
                min(0.65, bearish_score * 1.25),
            )
        )
        crowding = _clip01(sector_crowding + max(0.0, conviction - 0.65) + 0.18 * viewpoint_conflict)
        capital_support = _clip01(
            0.55 * float(mainline.capital_support)
            + 0.25 * max(0.0, float(getattr(capital_flow, "turnover_heat", 0.0)) - 0.50)
            + 0.20 * max(0.0, float(getattr(capital_flow, "large_order_bias", 0.0)))
        )
        macro_alignment = _clip01(
            0.60 * float(mainline.macro_alignment)
            + 0.25 * float(getattr(macro_context, "index_breadth_proxy", 0.0))
            + 0.15 * (1.0 - float(getattr(macro_context, "commodity_pressure", 0.0)))
        )
        phase, phase_reason = _phase_for_episode(
            conviction=conviction,
            breadth=breadth,
            leadership=leadership,
            catalyst_strength=catalyst_strength,
            event_risk=event_risk,
            crowding=crowding,
            viewpoint_conflict=viewpoint_conflict,
        )
        episodes.append(
            ThemeEpisode(
                theme=str(mainline.name),
                phase=phase,
                conviction=conviction,
                breadth=breadth,
                leadership=leadership,
                catalyst_strength=catalyst_strength,
                event_risk=event_risk,
                crowding=crowding,
                capital_support=capital_support,
                macro_alignment=macro_alignment,
                viewpoint_score=viewpoint_score,
                viewpoint_conflict=viewpoint_conflict,
                viewpoint_count=int(len(scoped_viewpoints)),
                sectors=sectors,
                representative_symbols=representatives,
                effective_time=str(getattr(state.market, "as_of_date", "")),
                phase_reason=phase_reason,
            )
        )

    episodes.sort(
        key=lambda item: (
            float(item.conviction),
            float(item.breadth),
            float(item.leadership),
            -float(item.event_risk),
        ),
        reverse=True,
    )
    return episodes
