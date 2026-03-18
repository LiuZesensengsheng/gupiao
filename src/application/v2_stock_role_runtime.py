from __future__ import annotations

import math

from src.application.v2_contracts import CompositeState, StockRoleSnapshot, ThemeEpisode


_ROLE_STRENGTH = {
    "leader": 0,
    "core": 1,
    "follower": 2,
    "rebound": 3,
    "laggard": 4,
}


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _theme_score(stock: object) -> float:
    return float(
        0.44 * float(getattr(stock, "alpha_score", 0.0))
        + 0.34 * float(getattr(stock, "excess_vs_sector_prob", 0.5))
        + 0.22 * float(getattr(stock, "up_20d_prob", 0.5))
    )


def _breakdown_risk(stock: object) -> float:
    return _clip01(
        0.50 * max(0.0, 0.55 - float(getattr(stock, "up_5d_prob", 0.5)))
        + 0.70 * max(0.0, 0.52 - float(getattr(stock, "excess_vs_sector_prob", 0.5)))
        + 0.30 * max(0.0, 0.60 - float(getattr(stock, "tradeability_score", 0.5)))
    )


def _role_for_stock(
    *,
    stock: object,
    rank: int,
    theme_size: int,
    breakdown_risk: float,
) -> str:
    leader_cut = max(2, int(math.ceil(theme_size * 0.10)))
    core_cut = max(leader_cut, int(math.ceil(theme_size * 0.30)))
    if rank <= leader_cut and float(getattr(stock, "excess_vs_sector_prob", 0.0)) >= 0.56:
        return "leader"
    if rank <= core_cut:
        return "core"
    rebound_strength = float(
        (
            float(getattr(stock, "up_1d_prob", 0.5))
            + float(getattr(stock, "up_2d_prob", 0.5))
            + float(getattr(stock, "up_3d_prob", 0.5))
        ) / 3.0
    )
    if rebound_strength >= 0.57 and float(getattr(stock, "excess_vs_sector_prob", 0.5)) < 0.54:
        return "rebound"
    if breakdown_risk >= 0.42 or rank > max(core_cut, int(math.ceil(theme_size * 0.70))):
        return "laggard"
    return "follower"


def build_stock_role_snapshots(
    *,
    state: CompositeState,
    theme_episodes: list[ThemeEpisode] | None = None,
    previous_roles: dict[str, StockRoleSnapshot] | None = None,
) -> dict[str, StockRoleSnapshot]:
    episodes = list(theme_episodes if theme_episodes is not None else getattr(state, "theme_episodes", []) or [])
    previous = dict(previous_roles or getattr(state, "stock_role_states", {}) or {})
    out: dict[str, StockRoleSnapshot] = {}

    for episode in episodes:
        sector_scope = {str(item) for item in episode.sectors}
        symbol_scope = {str(item) for item in episode.representative_symbols}
        theme_stocks = [
            stock
            for stock in getattr(state, "stocks", []) or []
            if (sector_scope and str(getattr(stock, "sector", "")) in sector_scope)
            or str(getattr(stock, "symbol", "")) in symbol_scope
        ]
        if not theme_stocks and sector_scope:
            continue
        ranked = sorted(
            theme_stocks,
            key=lambda stock: (
                _theme_score(stock),
                float(getattr(stock, "excess_vs_sector_prob", 0.5)),
                float(getattr(stock, "up_5d_prob", 0.5)),
            ),
            reverse=True,
        )
        theme_size = int(len(ranked))
        for index, stock in enumerate(ranked, start=1):
            symbol = str(getattr(stock, "symbol", ""))
            breakdown_risk = _breakdown_risk(stock)
            role = _role_for_stock(
                stock=stock,
                rank=index,
                theme_size=theme_size,
                breakdown_risk=breakdown_risk,
            )
            previous_role = ""
            role_downgrade = False
            if symbol in previous:
                previous_role = str(previous[symbol].role)
                role_downgrade = _ROLE_STRENGTH.get(role, 99) > _ROLE_STRENGTH.get(previous_role, 99)
            out[symbol] = StockRoleSnapshot(
                symbol=symbol,
                theme=str(episode.theme),
                role=role,
                previous_role=previous_role,
                role_downgrade=bool(role_downgrade),
                theme_rank=index,
                theme_size=theme_size,
                theme_percentile=float(index / max(1, theme_size)),
                alpha_score=float(getattr(stock, "alpha_score", 0.0)),
                excess_vs_sector=float(getattr(stock, "excess_vs_sector_prob", 0.5)),
                breakdown_risk=breakdown_risk,
                note=(
                    "role downgraded inside the same theme."
                    if role_downgrade
                    else f"{role} inside {episode.theme}."
                ),
            )
    return out
