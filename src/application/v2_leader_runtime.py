from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import pandas as pd

from src.application.v2_contracts import CandidateSelectionState, CompositeState, MainlineState, StockRoleSnapshot, ThemeEpisode


@dataclass(frozen=True)
class LeaderScoreSnapshot:
    symbol: str
    sector: str
    theme: str
    theme_phase: str
    role: str
    role_downgrade: bool = False
    negative_score: float = 0.0
    candidate_score: float = 0.0
    conviction_score: float = 0.0
    theme_rank: int = 0
    theme_size: int = 0
    hard_negative: bool = False
    alpha_score: float = 0.0
    excess_prob: float = 0.5
    up_1d_prob: float = 0.5
    up_5d_prob: float = 0.5
    up_20d_prob: float = 0.5
    tradeability_score: float = 0.5
    breakdown_risk: float = 0.0
    short_term_spike: float = 0.0
    theme_event_risk: float = 0.0
    breakout_quality_score: float = 0.0
    exhaustion_reversal_risk: float = 0.0
    pullback_reclaim_score: float = 0.0
    distance_to_20d_high: float = 0.0
    distance_to_20d_low: float = 0.0
    volume_breakout_ratio: float = 0.0
    upper_shadow_ratio_1: float = 0.0
    body_ratio_1: float = 0.0
    narrow_range_rank_20: float = 0.0
    breakdown_below_20_low: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LeaderTrainingLabel:
    date: str
    symbol: str
    sector: str
    theme: str
    theme_phase: str
    role: str
    role_downgrade: bool = False
    theme_rank: int = 0
    theme_size: int = 0
    theme_percentile: float = 0.0
    negative_score: float = 0.0
    candidate_score: float = 0.0
    conviction_score: float = 0.0
    hard_negative: bool = False
    future_excess_5d_vs_sector: float = 0.0
    future_excess_20d_vs_sector: float = 0.0
    future_theme_score: float = 0.0
    future_theme_rank: int = 0
    future_theme_percentile: float = 0.0
    is_true_leader: bool = False
    leader_bucket: str = "neutral"
    leader_tri_label: str = "not_leader"
    is_possible_leader: bool = False
    is_confirmed_leader: bool = False


@dataclass(frozen=True)
class ExitTrainingLabel:
    date: str
    symbol: str
    sector: str
    theme: str
    theme_phase: str
    role: str
    role_downgrade: bool = False
    theme_rank: int = 0
    theme_size: int = 0
    theme_percentile: float = 0.0
    sample_source: str = ""
    negative_score: float = 0.0
    candidate_score: float = 0.0
    conviction_score: float = 0.0
    hard_negative: bool = False
    alpha_score: float = 0.0
    excess_prob: float = 0.5
    up_1d_prob: float = 0.5
    up_5d_prob: float = 0.5
    up_20d_prob: float = 0.5
    tradeability_score: float = 0.5
    breakdown_risk: float = 0.0
    short_term_spike: float = 0.0
    theme_event_risk: float = 0.0
    breakout_quality_score: float = 0.0
    exhaustion_reversal_risk: float = 0.0
    pullback_reclaim_score: float = 0.0
    distance_to_20d_high: float = 0.0
    distance_to_20d_low: float = 0.0
    volume_breakout_ratio: float = 0.0
    upper_shadow_ratio_1: float = 0.0
    body_ratio_1: float = 0.0
    narrow_range_rank_20: float = 0.0
    breakdown_below_20_low: float = 0.0
    future_ret_1d: float = 0.0
    future_excess_1d_vs_mkt: float = 0.0
    future_ret_2d: float = 0.0
    future_ret_3d: float = 0.0
    future_ret_5d: float = 0.0
    future_excess_5d_vs_sector: float = 0.0
    future_excess_20d_vs_sector: float = 0.0
    future_excess_2d_vs_sector: float = 0.0
    future_excess_3d_vs_sector: float = 0.0
    path_failure_score: float = 0.0
    rebound_failure_score: float = 0.0
    breakdown_path_score: float = 0.0
    future_drag_score: float = 0.0
    hold_score: float = 0.0
    exit_pressure_score: float = 0.0
    exit_severity_label: str = "keep"
    exit_label: str = "keep"
    should_watch: bool = False
    should_reduce: bool = False
    should_exit_fast: bool = False
    should_exit_early: bool = False
    sample_weight: float = 1.0


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number):
        return float(default)
    return number


def _rank_percentile(rank: int, size: int) -> float:
    if size <= 1 or rank <= 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - ((rank - 1) / max(1, size - 1)))))


def _signed_prob_edge(value: object, scale: float = 0.20) -> float:
    centered = (_safe_float(value, 0.5) - 0.5) / max(1e-9, float(scale))
    return float(max(-1.0, min(1.0, centered)))


def _theme_episode_map(state: CompositeState) -> dict[str, ThemeEpisode]:
    return {
        _normalize_text(item.theme): item
        for item in getattr(state, "theme_episodes", []) or []
        if _normalize_text(item.theme)
    }


def _mainline_for_stock(
    *,
    stock: object,
    mainlines: list[MainlineState],
) -> MainlineState | None:
    symbol = _normalize_text(getattr(stock, "symbol", ""))
    sector = _normalize_text(getattr(stock, "sector", ""))
    matched = [
        item
        for item in mainlines
        if symbol in {_normalize_text(value) for value in getattr(item, "representative_symbols", [])}
        or sector in {_normalize_text(value) for value in getattr(item, "sectors", [])}
    ]
    if not matched:
        return None
    matched.sort(
        key=lambda item: (
            _safe_float(getattr(item, "conviction", 0.0)),
            _safe_float(getattr(item, "leadership", 0.0)),
            _safe_float(getattr(item, "catalyst_strength", 0.0)),
        ),
        reverse=True,
    )
    return matched[0]


def _theme_context_for_stock(
    *,
    stock: object,
    role_state: StockRoleSnapshot | None,
    theme_episode_map: dict[str, ThemeEpisode],
    mainlines: list[MainlineState],
) -> tuple[str, str, float, float, float]:
    if role_state is not None and _normalize_text(getattr(role_state, "theme", "")):
        theme = _normalize_text(getattr(role_state, "theme", ""))
        episode = theme_episode_map.get(theme)
        if episode is not None:
            return (
                theme,
                _normalize_text(getattr(episode, "phase", "")),
                _safe_float(getattr(episode, "conviction", 0.0)),
                _safe_float(getattr(episode, "leadership", 0.0)),
                _safe_float(getattr(episode, "event_risk", 0.0)),
            )
        return theme, "", 0.0, 0.0, 0.0

    mainline = _mainline_for_stock(stock=stock, mainlines=mainlines)
    if mainline is not None:
        theme = _normalize_text(getattr(mainline, "name", ""))
        episode = theme_episode_map.get(theme)
        if episode is not None:
            return (
                theme,
                _normalize_text(getattr(episode, "phase", "")),
                _safe_float(getattr(episode, "conviction", 0.0)),
                _safe_float(getattr(episode, "leadership", 0.0)),
                _safe_float(getattr(episode, "event_risk", 0.0)),
            )
        return (
            theme,
            "",
            _safe_float(getattr(mainline, "conviction", 0.0)),
            _safe_float(getattr(mainline, "leadership", 0.0)),
            _safe_float(getattr(mainline, "event_risk_level", 0.0)),
        )
    return _normalize_text(getattr(stock, "sector", "")) or "other", "", 0.0, 0.0, 0.0


def _base_negative_score(
    *,
    stock: object,
    theme_event_risk: float,
) -> float:
    excess = _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5)
    up_5d = _safe_float(getattr(stock, "up_5d_prob", 0.5), 0.5)
    up_20d = _safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5)
    tradeability = _safe_float(getattr(stock, "tradeability_score", 0.5), 0.5)
    event_impact = _safe_float(getattr(stock, "event_impact_score", 0.5), 0.5)
    return _clip01(
        0.28 * max(0.0, 0.54 - excess) / 0.12
        + 0.22 * max(0.0, 0.55 - up_5d) / 0.12
        + 0.14 * max(0.0, 0.55 - up_20d) / 0.12
        + 0.18 * max(0.0, 0.82 - tradeability) / 0.16
        + 0.08 * max(0.0, 0.10 - event_impact) / 0.10
        + 0.10 * max(0.0, theme_event_risk - 0.55) / 0.20
    )


def _durability_score(stock: object) -> float:
    up_1d = _safe_float(getattr(stock, "up_1d_prob", 0.5), 0.5)
    up_5d = _safe_float(getattr(stock, "up_5d_prob", 0.5), 0.5)
    up_20d = _safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5)
    excess = _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5)
    tradeability = _safe_float(getattr(stock, "tradeability_score", 0.5), 0.5)
    short_spike = max(0.0, up_1d - max(up_5d, up_20d))
    return _clip01(
        0.34 * max(0.0, up_20d - 0.50) / 0.20
        + 0.24 * max(0.0, up_5d - 0.50) / 0.16
        + 0.20 * max(0.0, excess - 0.50) / 0.14
        + 0.14 * max(0.0, tradeability - 0.80) / 0.16
        - 0.18 * short_spike / 0.12
    )


def _info_support_components(
    *,
    stock_info: object | None,
    sector_info: object | None,
    market_info: object | None,
) -> tuple[float, float, float, float]:
    stock_positive_edge = max(
        _signed_prob_edge(getattr(stock_info, "info_prob_5d", 0.5) if stock_info is not None else 0.5),
        _signed_prob_edge(getattr(stock_info, "info_prob_20d", 0.5) if stock_info is not None else 0.5),
    )
    sector_positive_edge = max(
        _signed_prob_edge(getattr(sector_info, "info_prob_5d", 0.5) if sector_info is not None else 0.5),
        _signed_prob_edge(getattr(sector_info, "info_prob_20d", 0.5) if sector_info is not None else 0.5),
    )
    market_positive_edge = max(
        _signed_prob_edge(getattr(market_info, "info_prob_5d", 0.5) if market_info is not None else 0.5),
        _signed_prob_edge(getattr(market_info, "info_prob_20d", 0.5) if market_info is not None else 0.5),
    )
    stock_catalyst = _safe_float(getattr(stock_info, "catalyst_strength", 0.0) if stock_info is not None else 0.0, 0.0)
    sector_catalyst = _safe_float(
        getattr(sector_info, "catalyst_strength", 0.0) if sector_info is not None else 0.0,
        0.0,
    )
    market_catalyst = _safe_float(
        getattr(market_info, "catalyst_strength", 0.0) if market_info is not None else 0.0,
        0.0,
    )
    coverage_ratio = max(
        _safe_float(getattr(stock_info, "coverage_ratio", 0.0) if stock_info is not None else 0.0, 0.0),
        _safe_float(getattr(sector_info, "coverage_ratio", 0.0) if sector_info is not None else 0.0, 0.0),
        _safe_float(getattr(stock_info, "coverage_confidence", 0.0) if stock_info is not None else 0.0, 0.0),
    )
    source_diversity = max(
        _safe_float(getattr(stock_info, "source_diversity", 0.0) if stock_info is not None else 0.0, 0.0),
        _safe_float(getattr(sector_info, "source_diversity", 0.0) if sector_info is not None else 0.0, 0.0),
        _safe_float(getattr(market_info, "source_diversity", 0.0) if market_info is not None else 0.0, 0.0),
    )
    info_support = _clip01(
        0.32 * max(stock_catalyst, sector_catalyst)
        + 0.18 * max(0.0, stock_positive_edge)
        + 0.12 * max(0.0, sector_positive_edge)
        + 0.08 * max(0.0, market_positive_edge)
        + 0.18 * coverage_ratio
        + 0.12 * source_diversity
    )
    info_risk = _clip01(
        0.42
        * max(
            _safe_float(getattr(stock_info, "event_risk_level", 0.0) if stock_info is not None else 0.0, 0.0),
            _safe_float(getattr(sector_info, "event_risk_level", 0.0) if sector_info is not None else 0.0, 0.0),
            _safe_float(getattr(market_info, "event_risk_level", 0.0) if market_info is not None else 0.0, 0.0),
        )
        + 0.20 * max(0.0, -stock_positive_edge)
        + 0.12 * max(0.0, -sector_positive_edge)
        + 0.08 * max(0.0, -market_positive_edge)
        + 0.18
        * max(
            _safe_float(getattr(stock_info, "negative_event_risk", 0.0) if stock_info is not None else 0.0, 0.0),
            _safe_float(getattr(sector_info, "negative_event_risk", 0.0) if sector_info is not None else 0.0, 0.0),
        )
    )
    info_confidence = _clip01(max(coverage_ratio, source_diversity))
    catalyst_strength = _clip01(max(stock_catalyst, sector_catalyst, market_catalyst))
    return info_support, info_risk, info_confidence, catalyst_strength


def _theme_support_components(
    *,
    episode: ThemeEpisode | None,
    theme_conviction: float,
    theme_leadership: float,
    theme_event_risk: float,
) -> dict[str, float]:
    breadth = _safe_float(getattr(episode, "breadth", 0.0) if episode is not None else 0.0, 0.0)
    crowding = _safe_float(getattr(episode, "crowding", 0.0) if episode is not None else 0.0, 0.0)
    capital_support = _safe_float(getattr(episode, "capital_support", 0.0) if episode is not None else 0.0, 0.0)
    macro_alignment = _safe_float(getattr(episode, "macro_alignment", 0.0) if episode is not None else 0.0, 0.0)
    viewpoint_support = _clip01(
        max(0.0, _safe_float(getattr(episode, "viewpoint_score", 0.0) if episode is not None else 0.0, 0.0))
    )
    viewpoint_conflict = _clip01(
        _safe_float(getattr(episode, "viewpoint_conflict", 0.0) if episode is not None else 0.0, 0.0)
    )
    quality = _clip01(
        0.30 * theme_conviction
        + 0.18 * breadth
        + 0.16 * theme_leadership
        + 0.12 * capital_support
        + 0.10 * macro_alignment
        + 0.08 * (1.0 - theme_event_risk)
        + 0.08 * viewpoint_support
        - 0.10 * crowding
        - 0.08 * viewpoint_conflict
    )
    return {
        "quality": quality,
        "breadth": breadth,
        "crowding": crowding,
        "capital_support": capital_support,
        "macro_alignment": macro_alignment,
        "viewpoint_support": viewpoint_support,
        "viewpoint_conflict": viewpoint_conflict,
    }


def _shortlist_rank_maps(
    selection: CandidateSelectionState,
) -> tuple[dict[str, int], int, int]:
    symbols = [str(item) for item in getattr(selection, "shortlisted_symbols", []) if str(item).strip()]
    shortlist_size = int(getattr(selection, "shortlist_size", 0) or 0)
    if shortlist_size <= 0:
        shortlist_size = len(symbols)
    total_scored = int(getattr(selection, "total_scored", 0) or 0)
    if total_scored <= 0:
        total_scored = len(symbols)
    return {symbol: idx for idx, symbol in enumerate(symbols)}, shortlist_size, total_scored


def _shortlist_support(
    *,
    symbol: str,
    shortlist_rank_map: dict[str, int],
    shortlist_size: int,
    total_scored: int,
) -> float:
    rank = shortlist_rank_map.get(symbol)
    if rank is None or shortlist_size <= 0:
        return 0.0
    if rank < shortlist_size:
        return float(max(0.20, 1.0 - (rank / max(1, shortlist_size))))
    tail_window = max(1, total_scored - shortlist_size)
    return float(max(0.0, 0.25 - 0.25 * ((rank - shortlist_size) / tail_window)))


def _fallback_role(
    *,
    stock: object,
    rank: int,
    theme_size: int,
    base_negative_score: float,
) -> str:
    leader_cut = max(2, int(math.ceil(theme_size * 0.10)))
    core_cut = max(leader_cut, int(math.ceil(theme_size * 0.30)))
    excess = _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5)
    rebound_strength = (
        _safe_float(getattr(stock, "up_1d_prob", 0.5), 0.5)
        + _safe_float(getattr(stock, "up_2d_prob", 0.5), 0.5)
        + _safe_float(getattr(stock, "up_3d_prob", 0.5), 0.5)
    ) / 3.0
    if rank <= leader_cut and excess >= 0.56:
        return "leader"
    if rank <= core_cut:
        return "core"
    if rebound_strength >= 0.57 and excess < 0.54:
        return "rebound"
    if base_negative_score >= 0.45 or rank > max(core_cut, int(math.ceil(theme_size * 0.70))):
        return "laggard"
    return "follower"


def _role_bonus(role: str) -> float:
    return {
        "leader": 0.12,
        "core": 0.08,
        "follower": 0.02,
        "rebound": 0.00,
        "laggard": -0.10,
    }.get(_normalize_text(role).lower(), 0.0)


def _phase_bonus(phase: str) -> float:
    return {
        "strengthening": 0.08,
        "emerging": 0.04,
        "crowded": -0.01,
        "diverging": -0.03,
        "fading": -0.04,
    }.get(_normalize_text(phase).lower(), 0.0)


def _negative_adjustment_for_role_and_phase(
    *,
    role: str,
    phase: str,
    role_downgrade: bool,
) -> float:
    role_term = {
        "leader": -0.04,
        "core": 0.00,
        "follower": 0.04,
        "rebound": 0.06,
        "laggard": 0.16,
    }.get(_normalize_text(role).lower(), 0.0)
    phase_term = {
        "strengthening": -0.03,
        "emerging": 0.00,
        "crowded": 0.04,
        "diverging": 0.07,
        "fading": 0.14,
    }.get(_normalize_text(phase).lower(), 0.0)
    downgrade_term = 0.10 if role_downgrade else 0.0
    return role_term + phase_term + downgrade_term


def _theme_relative_score(stock: object) -> float:
    return float(
        0.45 * _safe_float(getattr(stock, "alpha_score", 0.0), 0.0)
        + 0.32 * _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5)
        + 0.23 * _safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5)
    )


def _reason_list(
    *,
    stock: object,
    role: str,
    phase: str,
    role_downgrade: bool,
    negative_score: float,
    candidate_score: float,
    conviction_score: float,
    theme_rank: int,
    theme_size: int,
    info_support: float,
    info_risk: float,
    crowding: float,
    viewpoint_conflict: float,
    shortlist_support: float,
    durability: float,
) -> list[str]:
    reasons: list[str] = []
    excess = _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5)
    alpha = _safe_float(getattr(stock, "alpha_score", 0.0), 0.0)
    tradeability = _safe_float(getattr(stock, "tradeability_score", 0.5), 0.5)
    if phase == "strengthening":
        reasons.append("theme strengthening")
    elif phase == "emerging":
        reasons.append("theme emerging")
    elif phase == "fading":
        reasons.append("theme fading")
    elif phase == "crowded":
        reasons.append("theme crowded")
    if role_downgrade:
        reasons.append("role downgrade active")
    if role in {"leader", "core"}:
        reasons.append(f"theme role {role}")
    elif role == "laggard":
        reasons.append("laggard inside theme")
    if theme_size > 0 and theme_rank > 0:
        reasons.append(f"theme rank {theme_rank}/{theme_size}")
    if excess >= 0.58:
        reasons.append("strong excess vs sector")
    elif excess <= 0.52:
        reasons.append("weak excess vs sector")
    if alpha >= 0.62:
        reasons.append("alpha score in upper band")
    if tradeability >= 0.88:
        reasons.append("tradeability supportive")
    elif tradeability <= 0.75:
        reasons.append("tradeability below comfort zone")
    if info_support >= 0.42:
        reasons.append("info catalysts aligned")
    elif info_risk >= 0.45:
        reasons.append("info risk elevated")
    if shortlist_support >= 0.65:
        reasons.append("macro shortlist confirmed")
    if crowding >= 0.60:
        reasons.append("theme crowding elevated")
    if viewpoint_conflict >= 0.28:
        reasons.append("viewpoints conflicted")
    if durability >= 0.58:
        reasons.append("multi-day trend durable")
    if negative_score >= 0.55:
        reasons.append("hard negative risk elevated")
    elif candidate_score >= 0.65 and conviction_score >= 0.65:
        reasons.append("candidate and conviction both confirmed")
    deduped: list[str] = []
    seen: set[str] = set()
    for item in reasons:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= 6:
            break
    return deduped


def build_leader_score_snapshots(
    *,
    state: CompositeState,
) -> list[LeaderScoreSnapshot]:
    stocks = list(getattr(state, "stocks", []) or [])
    if not stocks:
        return []
    theme_episode_map = _theme_episode_map(state)
    role_states = dict(getattr(state, "stock_role_states", {}) or {})
    mainlines = list(getattr(state, "mainlines", []) or [])
    market_info = getattr(state, "market_info_state", None)
    sector_info_states = dict(getattr(state, "sector_info_states", {}) or {})
    stock_info_states = dict(getattr(state, "stock_info_states", {}) or {})
    shortlist_rank_map, shortlist_size, total_scored = _shortlist_rank_maps(
        getattr(state, "candidate_selection", CandidateSelectionState())
    )
    themed_rows: dict[str, list[dict[str, object]]] = {}

    for stock in stocks:
        role_state = role_states.get(_normalize_text(getattr(stock, "symbol", "")))
        theme, phase, theme_conviction, theme_leadership, theme_event_risk = _theme_context_for_stock(
            stock=stock,
            role_state=role_state,
            theme_episode_map=theme_episode_map,
            mainlines=mainlines,
        )
        base_negative = _base_negative_score(
            stock=stock,
            theme_event_risk=theme_event_risk,
        )
        themed_rows.setdefault(theme, []).append(
            {
                "stock": stock,
                "role_state": role_state,
                "theme": theme,
                "phase": phase,
                "theme_conviction": theme_conviction,
                "theme_leadership": theme_leadership,
                "theme_event_risk": theme_event_risk,
                "base_negative": base_negative,
                "theme_relative": _theme_relative_score(stock),
            }
        )

    out: list[LeaderScoreSnapshot] = []
    for theme, rows in themed_rows.items():
        ranked = sorted(
            rows,
            key=lambda item: (
                _safe_float(item.get("theme_relative", 0.0), 0.0),
                _safe_float(getattr(item["stock"], "excess_vs_sector_prob", 0.5), 0.5),
                _safe_float(getattr(item["stock"], "up_5d_prob", 0.5), 0.5),
            ),
            reverse=True,
        )
        theme_size = int(len(ranked))
        for idx, row in enumerate(ranked, start=1):
            stock = row["stock"]
            role_state = row.get("role_state")
            symbol = _normalize_text(getattr(stock, "symbol", ""))
            sector = _normalize_text(getattr(stock, "sector", ""))
            role = _normalize_text(getattr(role_state, "role", "")) or _fallback_role(
                stock=stock,
                rank=idx,
                theme_size=theme_size,
                base_negative_score=_safe_float(row.get("base_negative", 0.0), 0.0),
            )
            role_downgrade = bool(getattr(role_state, "role_downgrade", False))
            phase = _normalize_text(row.get("phase", ""))
            theme_conviction = _safe_float(row.get("theme_conviction", 0.0), 0.0)
            theme_leadership = _safe_float(row.get("theme_leadership", 0.0), 0.0)
            theme_event_risk = _safe_float(row.get("theme_event_risk", 0.0), 0.0)
            episode = theme_episode_map.get(theme)
            theme_metrics = _theme_support_components(
                episode=episode,
                theme_conviction=theme_conviction,
                theme_leadership=theme_leadership,
                theme_event_risk=theme_event_risk,
            )
            info_support, info_risk, info_confidence, info_catalyst = _info_support_components(
                stock_info=stock_info_states.get(symbol),
                sector_info=sector_info_states.get(sector),
                market_info=market_info,
            )
            durability = _durability_score(stock)
            shortlist_support = _shortlist_support(
                symbol=symbol,
                shortlist_rank_map=shortlist_rank_map,
                shortlist_size=shortlist_size,
                total_scored=max(total_scored, len(stocks)),
            )
            short_spike = max(
                0.0,
                _safe_float(getattr(stock, "up_1d_prob", 0.5), 0.5)
                - max(
                    _safe_float(getattr(stock, "up_5d_prob", 0.5), 0.5),
                    _safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5),
                ),
            )
            negative_score = _clip01(
                _safe_float(row.get("base_negative", 0.0), 0.0)
                + _negative_adjustment_for_role_and_phase(
                    role=role,
                    phase=phase,
                    role_downgrade=role_downgrade,
                )
                + 0.12 * info_risk
                + 0.08 * max(0.0, float(theme_metrics["crowding"]) - 0.55) / 0.25
                + 0.08 * float(theme_metrics["viewpoint_conflict"])
                + 0.06 * short_spike / 0.12
                - 0.05 * info_support
                - 0.04 * durability
            )
            theme_support = (
                0.40 * theme_conviction
                + 0.16 * float(theme_metrics["quality"])
                + 0.14 * float(theme_metrics["breadth"])
                + 0.12 * theme_leadership
                + 0.10 * float(theme_metrics["capital_support"])
                + 0.08 * float(theme_metrics["macro_alignment"])
                + 0.10 * (1.0 - theme_event_risk)
            )
            candidate_score = _clip01(
                0.18 * _safe_float(getattr(stock, "alpha_score", 0.0), 0.0)
                + 0.18 * _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5)
                + 0.10 * _safe_float(getattr(stock, "up_5d_prob", 0.5), 0.5)
                + 0.09 * _safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5)
                + 0.08 * _safe_float(getattr(stock, "tradeability_score", 0.5), 0.5)
                + 0.12 * theme_support
                + 0.07 * info_support
                + 0.05 * durability
                + 0.04 * info_confidence
                + 0.04 * shortlist_support
                + 0.03 * info_catalyst
                + 0.03 * float(theme_metrics["viewpoint_support"])
                + 0.05 * (1.0 - float(idx / max(1, theme_size)))
                + _role_bonus(role)
                + _phase_bonus(phase)
                - 0.22 * negative_score
            )
            conviction_score = _clip01(
                0.20 * _safe_float(getattr(stock, "alpha_score", 0.0), 0.0)
                + 0.19 * _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5)
                + 0.12 * _safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5)
                + 0.08 * _safe_float(getattr(stock, "up_5d_prob", 0.5), 0.5)
                + 0.12 * theme_conviction
                + 0.06 * float(theme_metrics["breadth"])
                + 0.05 * theme_leadership
                + 0.05 * float(theme_metrics["capital_support"])
                + 0.03 * float(theme_metrics["macro_alignment"])
                + 0.05 * info_support
                + 0.05 * durability
                + 0.04 * info_catalyst
                + 0.03 * shortlist_support
                + 0.60 * _role_bonus(role)
                + 0.50 * _phase_bonus(phase)
                - 0.28 * negative_score
                - (0.06 if role_downgrade else 0.0)
            )
            hard_negative = bool(
                negative_score >= 0.58
                or (role == "laggard" and candidate_score < 0.56)
                or (
                    phase in {"fading", "diverging"}
                    and max(theme_event_risk, float(theme_metrics["crowding"])) >= 0.55
                    and info_support < 0.18
                    and _safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5) < 0.54
                )
                or (info_risk >= 0.62 and durability <= 0.36)
                or (float(theme_metrics["viewpoint_conflict"]) >= 0.35 and candidate_score < 0.60)
            )
            out.append(
                LeaderScoreSnapshot(
                    symbol=symbol,
                    sector=sector,
                    theme=theme,
                    theme_phase=phase,
                    role=role,
                    role_downgrade=role_downgrade,
                    negative_score=negative_score,
                    candidate_score=candidate_score,
                    conviction_score=conviction_score,
                    theme_rank=idx,
                    theme_size=theme_size,
                    hard_negative=hard_negative,
                    alpha_score=_safe_float(getattr(stock, "alpha_score", 0.0), 0.0),
                    excess_prob=_safe_float(getattr(stock, "excess_vs_sector_prob", 0.5), 0.5),
                    up_1d_prob=_safe_float(getattr(stock, "up_1d_prob", 0.5), 0.5),
                    up_5d_prob=_safe_float(getattr(stock, "up_5d_prob", 0.5), 0.5),
                    up_20d_prob=_safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5),
                    tradeability_score=_safe_float(getattr(stock, "tradeability_score", 0.5), 0.5),
                    breakdown_risk=_safe_float(getattr(role_state, "breakdown_risk", 0.0), 0.0),
                    short_term_spike=short_spike,
                    theme_event_risk=theme_event_risk,
                    breakout_quality_score=_safe_float(getattr(stock, "breakout_quality_score", 0.0), 0.0),
                    exhaustion_reversal_risk=_safe_float(getattr(stock, "exhaustion_reversal_risk", 0.0), 0.0),
                    pullback_reclaim_score=_safe_float(getattr(stock, "pullback_reclaim_score", 0.0), 0.0),
                    distance_to_20d_high=_safe_float(getattr(stock, "distance_to_20d_high", 0.0), 0.0),
                    distance_to_20d_low=_safe_float(getattr(stock, "distance_to_20d_low", 0.0), 0.0),
                    volume_breakout_ratio=_safe_float(getattr(stock, "volume_breakout_ratio", 0.0), 0.0),
                    upper_shadow_ratio_1=_safe_float(getattr(stock, "upper_shadow_ratio_1", 0.0), 0.0),
                    body_ratio_1=_safe_float(getattr(stock, "body_ratio_1", 0.0), 0.0),
                    narrow_range_rank_20=_safe_float(getattr(stock, "narrow_range_rank_20", 0.0), 0.0),
                    breakdown_below_20_low=_safe_float(getattr(stock, "breakdown_below_20_low", 0.0), 0.0),
                    reasons=_reason_list(
                        stock=stock,
                        role=role,
                        phase=phase,
                        role_downgrade=role_downgrade,
                        negative_score=negative_score,
                        candidate_score=candidate_score,
                        conviction_score=conviction_score,
                        theme_rank=idx,
                        theme_size=theme_size,
                        info_support=info_support,
                        info_risk=info_risk,
                        crowding=float(theme_metrics["crowding"]),
                        viewpoint_conflict=float(theme_metrics["viewpoint_conflict"]),
                        shortlist_support=shortlist_support,
                        durability=durability,
                    ),
                )
            )
    out.sort(
        key=lambda item: (
            not item.hard_negative,
            item.conviction_score,
            item.candidate_score,
            -item.negative_score,
        ),
        reverse=True,
    )
    return out


def top_leader_candidates(
    *,
    state: CompositeState,
    limit: int = 12,
    leader_rank_model: dict[str, object] | None = None,
) -> list[LeaderScoreSnapshot]:
    snapshots = build_leader_score_snapshots(state=state)
    if not snapshots:
        return []

    def _sort_key(item: LeaderScoreSnapshot) -> tuple[float, float, float, float]:
        filter_score, rank_score, threshold = _leader_runtime_scores(
            snapshot=item,
            shortlist_support=0.0,
            leader_rank_model=leader_rank_model,
        )
        passes = 1.0 if (not bool(item.hard_negative) and filter_score >= float(threshold)) else 0.0
        return (
            passes,
            float(rank_score),
            float(filter_score),
            -float(item.negative_score),
        )

    ranked = sorted(snapshots, key=_sort_key, reverse=True)
    return ranked[: max(1, int(limit))]


def apply_leader_candidate_overlay(
    *,
    state: CompositeState,
    leader_rank_model: dict[str, object] | None = None,
) -> CompositeState:
    selection = getattr(state, "candidate_selection", CandidateSelectionState())
    symbols = [str(item) for item in getattr(selection, "shortlisted_symbols", []) if str(item).strip()]
    if len(symbols) <= 1:
        return state
    snapshots = build_leader_score_snapshots(state=state)
    if not snapshots:
        return state

    rank_map = {symbol: idx for idx, symbol in enumerate(symbols)}
    score_map = {item.symbol: item for item in snapshots}
    shortlist_size = int(getattr(selection, "shortlist_size", 0) or 0)
    if shortlist_size <= 0:
        shortlist_size = len(symbols)
    shortlist_size = min(shortlist_size, len(symbols))
    original_core = list(symbols[:shortlist_size])
    stock_map = {
        str(getattr(stock, "symbol", "")): stock
        for stock in getattr(state, "stocks", []) or []
        if str(getattr(stock, "symbol", "")).strip()
    }
    symbol_sector_map = {
        str(getattr(stock, "symbol", "")): str(getattr(stock, "sector", ""))
        for stock in getattr(state, "stocks", []) or []
        if str(getattr(stock, "symbol", "")).strip()
    }
    stressed_tape = bool(
        str(getattr(getattr(state, "market", None), "volatility_regime", "") or "").strip().lower() == "high"
        or float(getattr(getattr(state, "market", None), "drawdown_risk", 0.0) or 0.0) >= 0.35
        or float(getattr(getattr(state, "cross_section", None), "weak_stock_ratio", 0.0) or 0.0) >= 0.48
    )

    def _stress_fragility(symbol: str) -> float:
        stock = stock_map.get(symbol)
        if stock is None:
            return 0.0
        up_1d = _safe_float(getattr(stock, "up_1d_prob", 0.5), 0.5)
        up_5d = _safe_float(getattr(stock, "up_5d_prob", 0.5), 0.5)
        up_20d = _safe_float(getattr(stock, "up_20d_prob", 0.5), 0.5)
        tradeability = _safe_float(getattr(stock, "tradeability_score", 0.5), 0.5)
        event_impact = _safe_float(getattr(stock, "event_impact_score", 0.5), 0.5)
        short_spike = max(0.0, up_1d - max(up_5d, up_20d))
        weak_mid = max(0.0, 0.55 - up_20d)
        return _clip01(
            0.42 * short_spike / 0.14
            + 0.20 * weak_mid / 0.12
            + 0.18 * max(0.0, 0.84 - tradeability) / 0.14
            + 0.12 * max(0.0, 0.10 - event_impact) / 0.10
        )

    def _stress_promotion_allowed(symbol: str) -> bool:
        payload = score_map.get(symbol)
        if payload is None or bool(payload.hard_negative):
            return False
        role = str(getattr(payload, "role", "") or "").strip().lower()
        if role not in {"leader", "core"}:
            return False
        if float(getattr(payload, "negative_score", 0.0)) > 0.22:
            return False
        if float(getattr(payload, "candidate_score", 0.0)) < 0.64:
            return False
        if _stress_fragility(symbol) > 0.24:
            return False
        return True

    def _sort_key(symbol: str) -> tuple[float, float, float, float, int]:
        payload = score_map.get(symbol)
        if payload is None:
            return (2.0, 0.0, 0.0, 2.0, rank_map.get(symbol, len(rank_map)))
        shortlist_support = _shortlist_support(
            symbol=symbol,
            shortlist_rank_map=rank_map,
            shortlist_size=shortlist_size,
            total_scored=max(len(symbols), int(getattr(selection, "total_scored", len(symbols)) or len(symbols))),
        )
        filter_score, overlay_rank_score, threshold = _leader_runtime_scores(
            snapshot=payload,
            shortlist_support=float(shortlist_support),
            leader_rank_model=leader_rank_model,
        )
        passes_filter = 1.0 if (not payload.hard_negative and filter_score >= float(threshold)) else 0.0
        return (
            passes_filter,
            float(overlay_rank_score),
            float(filter_score),
            -float(payload.negative_score),
            rank_map.get(symbol, len(rank_map)),
        )

    reordered = sorted(symbols, key=_sort_key, reverse=True)
    if stressed_tape:
        protected_core: list[str] = []
        for symbol in reordered:
            if len(protected_core) >= shortlist_size:
                break
            if symbol in original_core:
                protected_core.append(symbol)
                continue
            if _stress_promotion_allowed(symbol):
                protected_core.append(symbol)
        for symbol in original_core:
            if len(protected_core) >= shortlist_size:
                break
            if symbol not in protected_core:
                protected_core.append(symbol)
        reordered = protected_core + [symbol for symbol in reordered if symbol not in set(protected_core)]
    if reordered == symbols:
        return state

    refreshed_core = list(reordered[:shortlist_size])
    top_promoted = [symbol for symbol in refreshed_core if symbol not in original_core]
    demoted_from_core = [symbol for symbol in original_core if symbol not in refreshed_core]
    shortlisted_sectors = list(
        dict.fromkeys(
            symbol_sector_map.get(symbol, "")
            for symbol in refreshed_core
            if symbol_sector_map.get(symbol, "")
        )
    )
    sector_slots: dict[str, int] = {}
    for symbol in refreshed_core:
        sector = symbol_sector_map.get(symbol, "")
        if not sector:
            continue
        sector_slots[sector] = int(sector_slots.get(sector, 0) + 1)
    notes = list(getattr(selection, "selection_notes", []) or [])
    note = "Leader overlay reprioritized shortlist order and refreshed core membership."
    if _leader_model_active(leader_rank_model):
        note = "Learned leader overlay reprioritized shortlist order and refreshed core membership."
    if top_promoted:
        note += " Promoted into core: " + ", ".join(top_promoted[:3])
    if demoted_from_core:
        note += " Demoted from core: " + ", ".join(demoted_from_core[:3])
    notes.append(note)
    return replace(
        state,
        candidate_selection=replace(
            selection,
            shortlisted_symbols=reordered,
            shortlisted_sectors=shortlisted_sectors,
            sector_slots=sector_slots,
            selection_notes=notes,
        ),
    )


def _dcg(relevances: list[float]) -> float:
    total = 0.0
    for idx, value in enumerate(relevances, start=1):
        total += float(value) / math.log2(idx + 1.0)
    return float(total)


_LEADER_MODEL_ROLES = ["leader", "core", "follower", "rebound", "laggard"]
_LEADER_MODEL_PHASES = ["emerging", "strengthening", "crowded", "diverging", "fading"]


def _leader_snapshot_row(snapshot: LeaderScoreSnapshot) -> dict[str, object]:
    return {
        "symbol": str(snapshot.symbol),
        "theme": str(snapshot.theme),
        "role": str(snapshot.role).strip().lower(),
        "theme_phase": str(snapshot.theme_phase).strip().lower(),
        "negative_score": float(snapshot.negative_score),
        "candidate_score": float(snapshot.candidate_score),
        "conviction_score": float(snapshot.conviction_score),
        "theme_percentile": _rank_percentile(int(snapshot.theme_rank), int(snapshot.theme_size)),
        "theme_size": int(snapshot.theme_size),
        "role_downgrade": bool(snapshot.role_downgrade),
        "hard_negative": bool(snapshot.hard_negative),
    }


def _leader_model_feature_vector(row: dict[str, object]) -> list[float]:
    role = str(row.get("role", "")).strip().lower()
    phase = str(row.get("theme_phase", "")).strip().lower()
    return [
        float(row.get("negative_score", 0.0) or 0.0),
        float(row.get("candidate_score", 0.0) or 0.0),
        float(row.get("conviction_score", 0.0) or 0.0),
        float(row.get("theme_percentile", 0.0) or 0.0),
        _clip01(float(row.get("theme_size", 0.0) or 0.0) / 20.0),
        1.0 if bool(row.get("role_downgrade", False)) else 0.0,
        1.0 if bool(row.get("hard_negative", False)) else 0.0,
        *[1.0 if role == item else 0.0 for item in _LEADER_MODEL_ROLES],
        *[1.0 if phase == item else 0.0 for item in _LEADER_MODEL_PHASES],
    ]


def _leader_model_active(model_payload: dict[str, object] | None) -> bool:
    if not isinstance(model_payload, dict):
        return False
    evaluation_metrics = model_payload.get("evaluation_metrics", {})
    if isinstance(evaluation_metrics, dict) and evaluation_metrics:
        top1_hit_rate = float(evaluation_metrics.get("top1_hit_rate", 0.0) or 0.0)
        ndcg_at_3 = float(evaluation_metrics.get("ndcg_at_3", 0.0) or 0.0)
        filter_metrics = model_payload.get("leader_filter_model", {})
        filter_eval = filter_metrics.get("evaluation_metrics", {}) if isinstance(filter_metrics, dict) else {}
        filter_precision = float(filter_eval.get("precision", 0.0) or 0.0) if isinstance(filter_eval, dict) else 0.0
        if top1_hit_rate < 0.40 or ndcg_at_3 < 0.92 or filter_precision < 0.58:
            return False
    coef = model_payload.get("coef", [])
    if not isinstance(coef, list) or not coef:
        return False
    if any(abs(float(value or 0.0)) > 1e-9 for value in coef):
        return True
    return int(model_payload.get("train_rows", 0) or 0) > 0


def _score_linear_model(
    *,
    model_payload: dict[str, object] | None,
    row: dict[str, object],
) -> float | None:
    if not _leader_model_active(model_payload):
        return None
    feature_names = model_payload.get("feature_names", [])
    vector = _leader_model_feature_vector(row)
    if not isinstance(feature_names, list) or len(feature_names) != len(vector):
        return None
    coef = model_payload.get("coef", [])
    if not isinstance(coef, list):
        return None
    intercept = float(model_payload.get("intercept", 0.0) or 0.0)
    total = intercept
    for weight, value in zip(coef, vector):
        total += float(weight or 0.0) * float(value)
    return _clip01(total)


def _leader_model_threshold(model_payload: dict[str, object] | None) -> float:
    if not isinstance(model_payload, dict):
        return 0.60
    filter_model = model_payload.get("leader_filter_model", {})
    if isinstance(filter_model, dict) and "threshold" in filter_model:
        return _clip01(float(filter_model.get("threshold", 0.60) or 0.60))
    manifest = model_payload.get("leader_two_stage_manifest", {})
    if isinstance(manifest, dict) and "filter_threshold" in manifest:
        return _clip01(float(manifest.get("filter_threshold", 0.60) or 0.60))
    return 0.60


def _leader_runtime_scores(
    *,
    snapshot: LeaderScoreSnapshot,
    shortlist_support: float = 0.0,
    leader_rank_model: dict[str, object] | None = None,
) -> tuple[float, float, float]:
    theme_percentile = _rank_percentile(int(snapshot.theme_rank), int(snapshot.theme_size))
    role_term = _role_bonus(str(snapshot.role))
    phase_term = _phase_bonus(str(snapshot.theme_phase))
    heuristic_filter = _clip01(
        0.44
        + 0.34 * float(snapshot.candidate_score)
        + 0.18 * float(snapshot.conviction_score)
        + 0.10 * float(theme_percentile)
        + 0.18 * float(role_term)
        + 0.10 * float(phase_term)
        - 0.42 * float(snapshot.negative_score)
        - (0.22 if bool(snapshot.role_downgrade) else 0.0)
        - (0.35 if bool(snapshot.hard_negative) else 0.0)
    )
    heuristic_rank = (
        0.58 * float(snapshot.candidate_score)
        + 0.24 * float(snapshot.conviction_score)
        + 0.12 * float(theme_percentile)
        + 0.10 * float(role_term)
        + 0.06 * float(phase_term)
        + 0.08 * float(shortlist_support)
        - 0.28 * float(snapshot.negative_score)
        - (0.12 if bool(snapshot.role_downgrade) else 0.0)
        - (0.25 if bool(snapshot.hard_negative) else 0.0)
    )
    row = _leader_snapshot_row(snapshot)
    learned_filter = _score_linear_model(
        model_payload=leader_rank_model.get("leader_filter_model", {}) if isinstance(leader_rank_model, dict) else None,
        row=row,
    )
    learned_rank = _score_linear_model(
        model_payload=leader_rank_model,
        row=row,
    )
    if learned_filter is None or learned_rank is None:
        return heuristic_filter, heuristic_rank, 0.60
    combined_filter = _clip01(0.56 * heuristic_filter + 0.44 * learned_filter)
    combined_rank = (
        0.52 * float(heuristic_rank)
        + 0.38 * float(learned_rank)
        + 0.06 * float(learned_filter)
        + 0.04 * float(shortlist_support)
    )
    return combined_filter, combined_rank, _leader_model_threshold(leader_rank_model)


def _future_lookup_for_trajectory(trajectory: object) -> dict[str, pd.DataFrame]:
    prepared = getattr(trajectory, "prepared", None)
    stock_frames = getattr(prepared, "stock_frames", {}) if prepared is not None else {}
    lookup: dict[str, pd.DataFrame] = {}
    for symbol, frame in stock_frames.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty or "date" not in frame.columns:
            continue
        work = frame.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values("date")
        if work.empty:
            continue
        lookup[str(symbol)] = work.set_index("date", drop=False)
    return lookup


def _future_theme_scores(
    *,
    group: list[LeaderScoreSnapshot],
    step_date: pd.Timestamp,
    future_lookup: dict[str, pd.DataFrame],
) -> tuple[dict[str, float], set[str], int]:
    frame = _future_theme_frame(
        group=group,
        step_date=step_date,
        future_lookup=future_lookup,
    )
    if frame.empty:
        return {}, set(), 0
    top_score = float(frame["future_score"].max())
    leader_cut = max(0.80, top_score - 0.05)
    leaders = {
        str(row.symbol)
        for row in frame.itertuples(index=False)
        if float(row.future_score) >= leader_cut and float(row.excess_5) > 0.0
    }
    if not leaders and top_score >= 0.70:
        best = frame.sort_values(["future_score", "excess_5", "excess_20"], ascending=False).iloc[0]
        if float(best["excess_5"]) > 0.0 or float(best["excess_20"]) > 0.0:
            leaders.add(str(best["symbol"]))
    return {
        str(row.symbol): float(row.future_score)
        for row in frame.itertuples(index=False)
    }, leaders, int(len(frame))


def _future_theme_frame(
    *,
    group: list[LeaderScoreSnapshot],
    step_date: pd.Timestamp,
    future_lookup: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in group:
        frame = future_lookup.get(str(item.symbol))
        if frame is None or step_date not in frame.index:
            continue
        row = frame.loc[step_date]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        excess_5 = _safe_float(row.get("excess_ret_5_vs_sector", float("nan")), float("nan"))
        excess_20 = _safe_float(row.get("excess_ret_20_vs_sector", float("nan")), float("nan"))
        if math.isnan(excess_5) or math.isnan(excess_20):
            continue
        rows.append(
            {
                "symbol": str(item.symbol),
                "future_ret_1": _safe_float(row.get("fwd_ret_1", 0.0), 0.0),
                "future_ret_2": _safe_float(row.get("fwd_ret_2", float("nan")), float("nan")),
                "future_ret_3": _safe_float(row.get("fwd_ret_3", float("nan")), float("nan")),
                "future_ret_5": _safe_float(row.get("fwd_ret_5", float("nan")), float("nan")),
                "future_excess_1": _safe_float(row.get("excess_ret_1_vs_mkt", 0.0), 0.0),
                "excess_2": _safe_float(row.get("excess_ret_2_vs_sector", float("nan")), float("nan")),
                "excess_3": _safe_float(row.get("excess_ret_3_vs_sector", float("nan")), float("nan")),
                "excess_5": excess_5,
                "excess_20": excess_20,
            }
        )
    if len(rows) < 2:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["score_5_rank"] = frame["excess_5"].rank(method="average", pct=True)
    frame["score_20_rank"] = frame["excess_20"].rank(method="average", pct=True)
    frame["future_score"] = (
        0.60 * frame["score_5_rank"]
        + 0.40 * frame["score_20_rank"]
        + 0.05 * (frame["excess_5"] > 0.0).astype(float)
        + 0.05 * (frame["excess_20"] > 0.0).astype(float)
    )
    frame = frame.sort_values(["future_score", "excess_5", "excess_20"], ascending=False).reset_index(drop=True)
    frame["future_rank"] = range(1, len(frame) + 1)
    frame["future_percentile"] = [
        _rank_percentile(int(rank), int(len(frame)))
        for rank in frame["future_rank"].tolist()
    ]
    return frame


def _leader_bucket(
    *,
    is_true_leader: bool,
    future_score: float,
    hard_negative: bool,
    future_excess_5: float,
) -> str:
    if is_true_leader:
        return "true_leader"
    if hard_negative and future_score <= 0.50 and future_excess_5 <= 0.0:
        return "hard_negative"
    if future_score >= 0.70 or future_excess_5 > 0.0:
        return "contender"
    return "neutral"


def _leader_tri_label(
    *,
    is_true_leader: bool,
    future_score: float,
    future_percentile: float,
    hard_negative: bool,
    future_excess_5: float,
    future_excess_20: float,
) -> str:
    if is_true_leader:
        return "confirmed_leader"
    if (
        hard_negative
        and future_score <= 0.50
        and future_excess_5 <= 0.0
        and future_excess_20 <= 0.0
    ):
        return "not_leader"
    if (
        future_score >= 0.65
        or future_percentile >= 0.55
        or future_excess_5 > 0.0
        or future_excess_20 > 0.0
    ):
        return "possible_leader"
    return "not_leader"


def _hold_score(
    *,
    role: str,
    phase: str,
    negative_score: float,
    candidate_score: float,
    conviction_score: float,
) -> float:
    role_score = {
        "leader": 1.00,
        "core": 0.85,
        "follower": 0.55,
        "rebound": 0.45,
        "laggard": 0.20,
    }.get(_normalize_text(role).lower(), 0.40)
    phase_score = {
        "strengthening": 1.00,
        "emerging": 0.80,
        "crowded": 0.55,
        "diverging": 0.35,
        "fading": 0.10,
    }.get(_normalize_text(phase).lower(), 0.45)
    return _clip01(
        0.34 * conviction_score
        + 0.26 * candidate_score
        + 0.20 * role_score
        + 0.12 * phase_score
        + 0.08 * (1.0 - negative_score)
    )


def _future_drag_score(
    *,
    future_ret_1: float,
    future_excess_1: float,
    future_excess_5: float,
    future_excess_20: float,
) -> float:
    return _clip01(
        0.10 * max(0.0, -future_ret_1 / 0.03)
        + 0.18 * max(0.0, -future_excess_1 / 0.02)
        + 0.36 * max(0.0, -future_excess_5 / 0.05)
        + 0.28 * max(0.0, -future_excess_20 / 0.08)
        + 0.08 * (1.0 if future_excess_5 < 0.0 and future_excess_20 < 0.0 else 0.0)
    )


def _finite_metric_values(*values: float) -> list[float]:
    out: list[float] = []
    for value in values:
        number = _safe_float(value, float("nan"))
        if math.isnan(number):
            continue
        out.append(float(number))
    return out


def _fallback_metric(primary: float, *fallbacks: float) -> float:
    for value in (primary, *fallbacks):
        number = _safe_float(value, float("nan"))
        if not math.isnan(number):
            return float(number)
    return 0.0


def _future_path_failure_components(
    *,
    future_ret_1: float,
    future_ret_2: float,
    future_ret_3: float,
    future_ret_5: float,
    future_excess_1: float,
    future_excess_2: float,
    future_excess_3: float,
    future_excess_5: float,
) -> dict[str, float]:
    ret_values = _finite_metric_values(future_ret_1, future_ret_2, future_ret_3, future_ret_5)
    excess_values = _finite_metric_values(future_excess_1, future_excess_2, future_excess_3, future_excess_5)
    if not ret_values and not excess_values:
        return {
            "path_failure_score": 0.0,
            "rebound_failure_score": 0.0,
            "breakdown_path_score": 0.0,
        }

    min_ret = min(ret_values) if ret_values else 0.0
    min_excess = min(excess_values) if excess_values else 0.0
    ret_negative_ratio = float(sum(1 for value in ret_values if value < 0.0) / max(1, len(ret_values)))
    excess_negative_ratio = float(sum(1 for value in excess_values if value < 0.0) / max(1, len(excess_values)))

    ret_3 = _fallback_metric(future_ret_3, future_ret_5, future_ret_2, future_ret_1)
    ret_5 = _fallback_metric(future_ret_5, future_ret_3, future_ret_2, future_ret_1)
    excess_3 = _fallback_metric(future_excess_3, future_excess_5, future_excess_2, future_excess_1)
    excess_5 = _fallback_metric(future_excess_5, future_excess_3, future_excess_2, future_excess_1)

    breakdown_path_score = _clip01(
        0.52 * max(0.0, -min_ret / 0.06)
        + 0.38 * max(0.0, -min_excess / 0.05)
        + 0.10 * max(ret_negative_ratio, excess_negative_ratio)
    )
    rebound_failure_score = _clip01(
        0.45 * max(0.0, future_ret_1 - max(ret_3, ret_5)) / 0.06
        + 0.35 * max(0.0, future_excess_1 - max(excess_3, excess_5)) / 0.05
        + 0.20 * (1.0 if future_ret_1 > 0.0 and min(ret_3, ret_5) < 0.0 else 0.0)
    )
    persistence_score = _clip01(
        0.55 * excess_negative_ratio
        + 0.25 * ret_negative_ratio
        + 0.20 * (1.0 if min_excess < 0.0 and excess_5 < 0.0 else 0.0)
    )
    return {
        "path_failure_score": _clip01(
            0.46 * breakdown_path_score + 0.32 * rebound_failure_score + 0.22 * persistence_score
        ),
        "rebound_failure_score": rebound_failure_score,
        "breakdown_path_score": breakdown_path_score,
    }


def _exit_severity_label(
    *,
    phase: str,
    role_downgrade: bool,
    hard_negative: bool,
    future_drag_score: float,
    exit_pressure_score: float,
    path_failure_score: float,
    rebound_failure_score: float,
    breakdown_path_score: float,
) -> str:
    phase_value = _normalize_text(phase).lower()
    if (
        exit_pressure_score >= 0.78
        or breakdown_path_score >= 0.72
        or (hard_negative and exit_pressure_score >= 0.46)
        or (phase_value == "fading" and exit_pressure_score >= 0.40)
    ):
        return "exit_fast"
    if (
        exit_pressure_score >= 0.58
        or rebound_failure_score >= 0.56
        or (role_downgrade and path_failure_score >= 0.28)
        or (phase_value in {"crowded", "diverging", "fading"} and exit_pressure_score >= 0.34)
    ):
        return "reduce"
    if (
        exit_pressure_score >= 0.32
        or path_failure_score >= 0.36
        or role_downgrade
        or hard_negative
        or phase_value in {"crowded", "diverging", "fading"}
    ):
        return "watch"
    return "keep"


def _exit_sample_source(
    *,
    symbol: str,
    shortlisted: set[str],
    top_actionable: set[str],
    role: str,
    phase: str,
    candidate_score: float,
    role_downgrade: bool,
    hard_negative: bool,
) -> str:
    if symbol in shortlisted:
        return "shortlist"
    if symbol in top_actionable and (role in {"leader", "core", "follower"} or float(candidate_score) >= 0.62):
        return "top_conviction"
    if role in {"leader", "core"}:
        return "theme_role"
    if role_downgrade or hard_negative or phase in {"diverging", "fading"} or float(candidate_score) >= 0.60:
        return "candidate_score"
    return ""


def _exit_sample_weight(
    *,
    sample_source: str,
    exit_pressure_score: float,
    path_failure_score: float,
    role_downgrade: bool,
    hard_negative: bool,
) -> float:
    base = {
        "shortlist": 1.20,
        "top_conviction": 1.10,
        "theme_role": 1.00,
        "candidate_score": 0.90,
    }.get(sample_source, 0.80)
    return float(
        1.0
        * base
        * (1.0 + 0.70 * float(exit_pressure_score) + 0.25 * float(path_failure_score))
        * (1.10 if role_downgrade else 1.0)
        * (1.15 if hard_negative else 1.0)
    )


def build_leader_training_labels(
    *,
    trajectory: object | None,
    min_theme_size: int = 3,
) -> list[LeaderTrainingLabel]:
    if trajectory is None or not getattr(trajectory, "steps", None):
        return []

    future_lookup = _future_lookup_for_trajectory(trajectory)
    rows: list[LeaderTrainingLabel] = []
    for step in getattr(trajectory, "steps", []) or []:
        step_date = pd.Timestamp(getattr(step, "date"))
        step_state = getattr(step, "composite_state", None)
        snapshots = [] if step_state is None else build_leader_score_snapshots(state=step_state)
        grouped: dict[str, list[LeaderScoreSnapshot]] = {}
        for item in snapshots:
            grouped.setdefault(str(item.theme), []).append(item)
        for group in grouped.values():
            if len(group) < max(2, int(min_theme_size)):
                continue
            future_frame = _future_theme_frame(
                group=group,
                step_date=step_date,
                future_lookup=future_lookup,
            )
            if future_frame.empty:
                continue
            top_score = float(future_frame["future_score"].max())
            leader_cut = max(0.80, top_score - 0.05)
            true_leaders = {
                str(row.symbol)
                for row in future_frame.itertuples(index=False)
                if float(row.future_score) >= leader_cut and float(row.excess_5) > 0.0
            }
            if not true_leaders and top_score >= 0.70:
                best = future_frame.iloc[0]
                if float(best["excess_5"]) > 0.0 or float(best["excess_20"]) > 0.0:
                    true_leaders.add(str(best["symbol"]))
            future_map = {
                str(row["symbol"]): row
                for row in future_frame.to_dict(orient="records")
            }
            for item in group:
                metrics = future_map.get(str(item.symbol))
                if metrics is None:
                    continue
                future_score = _safe_float(metrics.get("future_score", 0.0), 0.0)
                future_excess_5 = _safe_float(metrics.get("excess_5", 0.0), 0.0)
                future_excess_20 = _safe_float(metrics.get("excess_20", 0.0), 0.0)
                future_percentile = _safe_float(metrics.get("future_percentile", 0.0), 0.0)
                is_true_leader = str(item.symbol) in true_leaders
                tri_label = _leader_tri_label(
                    is_true_leader=is_true_leader,
                    future_score=future_score,
                    future_percentile=future_percentile,
                    hard_negative=bool(item.hard_negative),
                    future_excess_5=future_excess_5,
                    future_excess_20=future_excess_20,
                )
                rows.append(
                    LeaderTrainingLabel(
                        date=str(step_date.date()),
                        symbol=str(item.symbol),
                        sector=str(item.sector),
                        theme=str(item.theme),
                        theme_phase=str(item.theme_phase),
                        role=str(item.role),
                        role_downgrade=bool(item.role_downgrade),
                        theme_rank=int(item.theme_rank),
                        theme_size=int(item.theme_size),
                        theme_percentile=_rank_percentile(int(item.theme_rank), int(item.theme_size)),
                        negative_score=float(item.negative_score),
                        candidate_score=float(item.candidate_score),
                        conviction_score=float(item.conviction_score),
                        hard_negative=bool(item.hard_negative),
                        future_excess_5d_vs_sector=future_excess_5,
                        future_excess_20d_vs_sector=future_excess_20,
                        future_theme_score=future_score,
                        future_theme_rank=int(metrics.get("future_rank", 0) or 0),
                        future_theme_percentile=future_percentile,
                        is_true_leader=is_true_leader,
                        leader_bucket=_leader_bucket(
                            is_true_leader=is_true_leader,
                            future_score=future_score,
                            hard_negative=bool(item.hard_negative),
                            future_excess_5=future_excess_5,
                        ),
                        leader_tri_label=tri_label,
                        is_possible_leader=tri_label == "possible_leader",
                        is_confirmed_leader=tri_label == "confirmed_leader",
                    )
                )
    return rows


def build_exit_training_labels(
    *,
    trajectory: object | None,
    min_theme_size: int = 2,
    candidate_limit: int = 8,
) -> list[ExitTrainingLabel]:
    if trajectory is None or not getattr(trajectory, "steps", None):
        return []

    future_lookup = _future_lookup_for_trajectory(trajectory)
    rows: list[ExitTrainingLabel] = []
    for step in getattr(trajectory, "steps", []) or []:
        step_date = pd.Timestamp(getattr(step, "date"))
        step_state = getattr(step, "composite_state", None)
        if step_state is None:
            continue
        snapshots = build_leader_score_snapshots(state=step_state)
        grouped: dict[str, list[LeaderScoreSnapshot]] = {}
        for item in snapshots:
            grouped.setdefault(str(item.theme), []).append(item)
        shortlisted = {
            str(item)
            for item in getattr(getattr(step_state, "candidate_selection", None), "shortlisted_symbols", []) or []
            if str(item).strip()
        }
        for group in grouped.values():
            if len(group) < max(2, int(min_theme_size)):
                continue
            future_frame = _future_theme_frame(
                group=group,
                step_date=step_date,
                future_lookup=future_lookup,
            )
            if future_frame.empty:
                continue
            future_map = {
                str(row["symbol"]): row
                for row in future_frame.to_dict(orient="records")
            }
            ranked_group = sorted(
                group,
                key=lambda item: (item.conviction_score, item.candidate_score, -item.negative_score),
                reverse=True,
            )
            top_actionable = {
                str(item.symbol)
                for item in ranked_group[: max(2, min(int(candidate_limit), len(ranked_group)))]
            }
            for item in group:
                symbol = str(item.symbol)
                metrics = future_map.get(symbol)
                if metrics is None:
                    continue
                sample_source = _exit_sample_source(
                    symbol=symbol,
                    shortlisted=shortlisted,
                    top_actionable=top_actionable,
                    role=str(item.role),
                    phase=str(item.theme_phase),
                    candidate_score=float(item.candidate_score),
                    role_downgrade=bool(item.role_downgrade),
                    hard_negative=bool(item.hard_negative),
                )
                if not sample_source:
                    continue
                future_ret_1 = _safe_float(metrics.get("future_ret_1", 0.0), 0.0)
                future_ret_2 = _safe_float(metrics.get("future_ret_2", float("nan")), float("nan"))
                future_ret_3 = _safe_float(metrics.get("future_ret_3", float("nan")), float("nan"))
                future_ret_5 = _safe_float(metrics.get("future_ret_5", float("nan")), float("nan"))
                future_excess_1 = _safe_float(metrics.get("future_excess_1", 0.0), 0.0)
                future_excess_2 = _safe_float(metrics.get("excess_2", float("nan")), float("nan"))
                future_excess_3 = _safe_float(metrics.get("excess_3", float("nan")), float("nan"))
                future_excess_5 = _safe_float(metrics.get("excess_5", 0.0), 0.0)
                future_excess_20 = _safe_float(metrics.get("excess_20", 0.0), 0.0)
                hold_score = _hold_score(
                    role=str(item.role),
                    phase=str(item.theme_phase),
                    negative_score=float(item.negative_score),
                    candidate_score=float(item.candidate_score),
                    conviction_score=float(item.conviction_score),
                )
                future_drag_score = _future_drag_score(
                    future_ret_1=future_ret_1,
                    future_excess_1=future_excess_1,
                    future_excess_5=future_excess_5,
                    future_excess_20=future_excess_20,
                )
                path_components = _future_path_failure_components(
                    future_ret_1=future_ret_1,
                    future_ret_2=future_ret_2,
                    future_ret_3=future_ret_3,
                    future_ret_5=future_ret_5,
                    future_excess_1=future_excess_1,
                    future_excess_2=future_excess_2,
                    future_excess_3=future_excess_3,
                    future_excess_5=future_excess_5,
                )
                path_failure_score = float(path_components["path_failure_score"])
                rebound_failure_score = float(path_components["rebound_failure_score"])
                breakdown_path_score = float(path_components["breakdown_path_score"])
                exit_pressure_score = _clip01(0.58 * future_drag_score + 0.42 * path_failure_score)
                severity_label = _exit_severity_label(
                    phase=str(item.theme_phase),
                    role_downgrade=bool(item.role_downgrade),
                    hard_negative=bool(item.hard_negative),
                    future_drag_score=future_drag_score,
                    exit_pressure_score=exit_pressure_score,
                    path_failure_score=path_failure_score,
                    rebound_failure_score=rebound_failure_score,
                    breakdown_path_score=breakdown_path_score,
                )
                sample_weight = _exit_sample_weight(
                    sample_source=sample_source,
                    exit_pressure_score=exit_pressure_score,
                    path_failure_score=path_failure_score,
                    role_downgrade=bool(item.role_downgrade),
                    hard_negative=bool(item.hard_negative),
                )
                rows.append(
                    ExitTrainingLabel(
                        date=str(step_date.date()),
                        symbol=symbol,
                        sector=str(item.sector),
                        theme=str(item.theme),
                        theme_phase=str(item.theme_phase),
                        role=str(item.role),
                        role_downgrade=bool(item.role_downgrade),
                        theme_rank=int(item.theme_rank),
                        theme_size=int(item.theme_size),
                        theme_percentile=_rank_percentile(int(item.theme_rank), int(item.theme_size)),
                        sample_source=sample_source,
                        negative_score=float(item.negative_score),
                        candidate_score=float(item.candidate_score),
                        conviction_score=float(item.conviction_score),
                        hard_negative=bool(item.hard_negative),
                        alpha_score=float(item.alpha_score),
                        excess_prob=float(item.excess_prob),
                        up_1d_prob=float(item.up_1d_prob),
                        up_5d_prob=float(item.up_5d_prob),
                        up_20d_prob=float(item.up_20d_prob),
                        tradeability_score=float(item.tradeability_score),
                        breakdown_risk=float(item.breakdown_risk),
                        short_term_spike=float(item.short_term_spike),
                        theme_event_risk=float(item.theme_event_risk),
                        breakout_quality_score=float(item.breakout_quality_score),
                        exhaustion_reversal_risk=float(item.exhaustion_reversal_risk),
                        pullback_reclaim_score=float(item.pullback_reclaim_score),
                        distance_to_20d_high=float(item.distance_to_20d_high),
                        distance_to_20d_low=float(item.distance_to_20d_low),
                        volume_breakout_ratio=float(item.volume_breakout_ratio),
                        upper_shadow_ratio_1=float(item.upper_shadow_ratio_1),
                        body_ratio_1=float(item.body_ratio_1),
                        narrow_range_rank_20=float(item.narrow_range_rank_20),
                        breakdown_below_20_low=float(item.breakdown_below_20_low),
                        future_ret_1d=future_ret_1,
                        future_excess_1d_vs_mkt=future_excess_1,
                        future_ret_2d=0.0 if math.isnan(future_ret_2) else future_ret_2,
                        future_ret_3d=0.0 if math.isnan(future_ret_3) else future_ret_3,
                        future_ret_5d=0.0 if math.isnan(future_ret_5) else future_ret_5,
                        future_excess_5d_vs_sector=future_excess_5,
                        future_excess_20d_vs_sector=future_excess_20,
                        future_excess_2d_vs_sector=0.0 if math.isnan(future_excess_2) else future_excess_2,
                        future_excess_3d_vs_sector=0.0 if math.isnan(future_excess_3) else future_excess_3,
                        path_failure_score=path_failure_score,
                        rebound_failure_score=rebound_failure_score,
                        breakdown_path_score=breakdown_path_score,
                        future_drag_score=future_drag_score,
                        hold_score=hold_score,
                        exit_pressure_score=exit_pressure_score,
                        exit_severity_label=severity_label,
                        exit_label=severity_label,
                        should_watch=severity_label in {"watch", "reduce", "exit_fast"},
                        should_reduce=severity_label in {"reduce", "exit_fast"},
                        should_exit_fast=severity_label == "exit_fast",
                        should_exit_early=severity_label in {"reduce", "exit_fast"},
                        sample_weight=sample_weight,
                    )
                )
    return rows


def build_research_label_artifact_payloads(
    *,
    trajectory: object | None,
    min_leader_theme_size: int = 3,
    min_exit_theme_size: int = 2,
    exit_candidate_limit: int = 8,
) -> dict[str, object]:
    leader_rows = build_leader_training_labels(
        trajectory=trajectory,
        min_theme_size=min_leader_theme_size,
    )
    exit_rows = build_exit_training_labels(
        trajectory=trajectory,
        min_theme_size=min_exit_theme_size,
        candidate_limit=exit_candidate_limit,
    )
    leader_bucket_counts: dict[str, int] = {}
    leader_tri_label_counts: dict[str, int] = {}
    for item in leader_rows:
        leader_bucket_counts[item.leader_bucket] = int(leader_bucket_counts.get(item.leader_bucket, 0) + 1)
        leader_tri_label_counts[item.leader_tri_label] = int(
            leader_tri_label_counts.get(item.leader_tri_label, 0) + 1
        )
    exit_label_counts: dict[str, int] = {}
    exit_source_counts: dict[str, int] = {}
    exit_severity_counts: dict[str, int] = {}
    exit_source_avg_drag: dict[str, list[float]] = {}
    for item in exit_rows:
        exit_label_counts[item.exit_label] = int(exit_label_counts.get(item.exit_label, 0) + 1)
        exit_source_counts[item.sample_source] = int(exit_source_counts.get(item.sample_source, 0) + 1)
        exit_severity_counts[item.exit_severity_label] = int(exit_severity_counts.get(item.exit_severity_label, 0) + 1)
        exit_source_avg_drag.setdefault(item.sample_source, []).append(float(item.future_drag_score))
    all_dates = [item.date for item in leader_rows] + [item.date for item in exit_rows]
    label_manifest = {
        "leader_row_count": int(len(leader_rows)),
        "leader_true_count": int(sum(1 for item in leader_rows if item.is_true_leader)),
        "leader_hard_negative_count": int(sum(1 for item in leader_rows if item.hard_negative)),
        "leader_bucket_counts": leader_bucket_counts,
        "leader_tri_label_counts": leader_tri_label_counts,
        "leader_confirmed_count": int(sum(1 for item in leader_rows if item.leader_tri_label == "confirmed_leader")),
        "leader_possible_count": int(sum(1 for item in leader_rows if item.leader_tri_label == "possible_leader")),
        "leader_not_count": int(sum(1 for item in leader_rows if item.leader_tri_label == "not_leader")),
        "exit_row_count": int(len(exit_rows)),
        "exit_watch_count": int(sum(1 for item in exit_rows if item.exit_severity_label == "watch")),
        "exit_reduce_count": int(sum(1 for item in exit_rows if item.exit_label == "reduce")),
        "exit_fast_count": int(sum(1 for item in exit_rows if item.exit_label == "exit_fast")),
        "exit_label_counts": exit_label_counts,
        "exit_severity_counts": exit_severity_counts,
        "exit_sample_sources": exit_source_counts,
        "exit_avg_future_drag_score": float(
            sum(float(item.future_drag_score) for item in exit_rows) / max(1, len(exit_rows))
        ),
        "exit_avg_exit_pressure_score": float(
            sum(float(item.exit_pressure_score) for item in exit_rows) / max(1, len(exit_rows))
        ),
        "exit_avg_path_failure_score": float(
            sum(float(item.path_failure_score) for item in exit_rows) / max(1, len(exit_rows))
        ),
        "exit_avg_future_drag_by_source": {
            key: float(sum(values) / max(1, len(values)))
            for key, values in exit_source_avg_drag.items()
        },
        "window": {
            "start": "" if not all_dates else min(all_dates),
            "end": "" if not all_dates else max(all_dates),
        },
    }
    return {
        "training_label_manifest": label_manifest,
        "leader_training_labels": [asdict(item) for item in leader_rows],
        "exit_training_labels": [asdict(item) for item in exit_rows],
    }


def evaluate_leader_candidates(
    *,
    trajectory: object | None,
    top_k: int = 3,
) -> dict[str, object]:
    if trajectory is None or not getattr(trajectory, "steps", None):
        return {
            "theme_group_count": 0,
            "evaluated_stock_count": 0,
            "true_leader_count": 0,
            "candidate_recall_at_k": 0.0,
            "conviction_precision_at_1": 0.0,
            "confirmed_precision_at_1": 0.0,
            "possible_recall_at_k": 0.0,
            "not_leader_avoid_rate": 0.0,
            "ndcg_at_k": 0.0,
            "hard_negative_survival_recall": 0.0,
            "hard_negative_filter_rate": 0.0,
        }

    future_lookup = _future_lookup_for_trajectory(trajectory)
    theme_group_count = 0
    evaluated_stock_count = 0
    true_leader_count = 0
    candidate_hits = 0.0
    conviction_hits = 0.0
    possible_hits = 0.0
    ndcg_values: list[float] = []
    surviving_true_leaders = 0
    total_true_leaders = 0
    filtered_stock_count = 0
    not_leader_avoid_values: list[float] = []

    for step in getattr(trajectory, "steps", []) or []:
        step_date = pd.Timestamp(getattr(step, "date"))
        snapshots = build_leader_score_snapshots(state=getattr(step, "composite_state"))
        grouped: dict[str, list[LeaderScoreSnapshot]] = {}
        for item in snapshots:
            grouped.setdefault(str(item.theme), []).append(item)
        for group in grouped.values():
            if len(group) < 3:
                continue
            future_frame = _future_theme_frame(
                group=group,
                step_date=step_date,
                future_lookup=future_lookup,
            )
            if future_frame.empty:
                continue
            valid_stock_count = int(len(future_frame))
            future_scores = {
                str(row["symbol"]): float(row["future_score"])
                for row in future_frame.to_dict(orient="records")
            }
            top_score = float(future_frame["future_score"].max())
            leader_cut = max(0.80, top_score - 0.05)
            true_leaders = {
                str(row.symbol)
                for row in future_frame.itertuples(index=False)
                if float(row.future_score) >= leader_cut and float(row.excess_5) > 0.0
            }
            if not true_leaders and top_score >= 0.70:
                best = future_frame.sort_values(["future_score", "excess_5", "excess_20"], ascending=False).iloc[0]
                if float(best["excess_5"]) > 0.0 or float(best["excess_20"]) > 0.0:
                    true_leaders.add(str(best["symbol"]))
            if valid_stock_count < 2 or not true_leaders:
                continue
            theme_group_count += 1
            evaluated_stock_count += int(valid_stock_count)
            true_leader_count += int(len(true_leaders))
            total_true_leaders += int(len(true_leaders))
            filtered_stock_count += int(sum(1 for item in group if item.hard_negative))

            candidate_ranked = sorted(group, key=lambda item: item.candidate_score, reverse=True)
            conviction_ranked = sorted(group, key=lambda item: item.conviction_score, reverse=True)
            top_candidate_symbols = {item.symbol for item in candidate_ranked[: max(1, int(top_k))]}
            if top_candidate_symbols & true_leaders:
                candidate_hits += 1.0
            if conviction_ranked and conviction_ranked[0].symbol in true_leaders:
                conviction_hits += 1.0

            future_map = {
                str(row["symbol"]): row
                for row in future_frame.to_dict(orient="records")
            }
            tri_label_map: dict[str, str] = {}
            for item in group:
                metrics = future_map.get(str(item.symbol))
                if metrics is None:
                    continue
                future_score = _safe_float(metrics.get("future_score", 0.0), 0.0)
                future_percentile = _safe_float(metrics.get("future_percentile", 0.0), 0.0)
                tri_label_map[str(item.symbol)] = _leader_tri_label(
                    is_true_leader=str(item.symbol) in true_leaders,
                    future_score=future_score,
                    future_percentile=future_percentile,
                    hard_negative=bool(item.hard_negative),
                    future_excess_5=_safe_float(metrics.get("excess_5", 0.0), 0.0),
                    future_excess_20=_safe_float(metrics.get("excess_20", 0.0), 0.0),
                )
            possible_leaders = {
                symbol for symbol, tri_label in tri_label_map.items()
                if tri_label in {"possible_leader", "confirmed_leader"}
            }
            not_leaders = {
                symbol for symbol, tri_label in tri_label_map.items()
                if tri_label == "not_leader"
            }
            if top_candidate_symbols & possible_leaders:
                possible_hits += 1.0
            if not_leaders:
                not_leader_avoid_values.append(
                    float(sum(1 for symbol in not_leaders if symbol not in top_candidate_symbols) / max(1, len(not_leaders)))
                )

            predicted = [float(future_scores.get(item.symbol, 0.0)) for item in conviction_ranked[: max(1, int(top_k))]]
            ideal = sorted(future_scores.values(), reverse=True)[: max(1, int(top_k))]
            idcg = _dcg(ideal)
            ndcg_values.append(0.0 if idcg <= 1e-9 else _dcg(predicted) / idcg)

            for item in group:
                if item.symbol in true_leaders and not item.hard_negative:
                    surviving_true_leaders += 1

    filter_denominator = max(1, evaluated_stock_count)
    return {
        "theme_group_count": int(theme_group_count),
        "evaluated_stock_count": int(evaluated_stock_count),
        "true_leader_count": int(true_leader_count),
        "candidate_recall_at_k": float(candidate_hits / max(1, theme_group_count)),
        "conviction_precision_at_1": float(conviction_hits / max(1, theme_group_count)),
        "confirmed_precision_at_1": float(conviction_hits / max(1, theme_group_count)),
        "possible_recall_at_k": float(possible_hits / max(1, theme_group_count)),
        "not_leader_avoid_rate": float(sum(not_leader_avoid_values) / max(1, len(not_leader_avoid_values))),
        "ndcg_at_k": float(sum(ndcg_values) / max(1, len(ndcg_values))),
        "hard_negative_survival_recall": float(surviving_true_leaders / max(1, total_true_leaders)),
        "hard_negative_filter_rate": float(filtered_stock_count / filter_denominator),
    }


def build_leader_artifact_payloads(
    *,
    state: CompositeState | None,
    trajectory: object | None = None,
    top_k: int = 3,
    limit: int = 16,
) -> dict[str, object]:
    snapshots = [] if state is None else build_leader_score_snapshots(state=state)
    candidates = [] if state is None else top_leader_candidates(state=state, limit=limit)
    evaluation = evaluate_leader_candidates(
        trajectory=trajectory,
        top_k=top_k,
    )
    theme_breakdown: dict[str, int] = {}
    for item in candidates:
        theme_breakdown[item.theme] = int(theme_breakdown.get(item.theme, 0) + 1)
    return {
        "leader_manifest": {
            "as_of_date": "" if state is None else str(getattr(getattr(state, "market", None), "as_of_date", "")),
            "snapshot_count": int(len(snapshots)),
            "candidate_count": int(len(candidates)),
            "hard_negative_count": int(sum(1 for item in snapshots if item.hard_negative)),
            "theme_breakdown": theme_breakdown,
            "evaluation": evaluation,
        },
        "leader_candidates": [asdict(item) for item in candidates],
    }
