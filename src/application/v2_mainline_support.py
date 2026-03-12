from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from src.application.v2_contracts import (
    CapitalFlowState,
    CrossSectionForecastState,
    InfoAggregateState,
    MacroContextState,
    MainlineState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
)


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _theme_name_for_sector(sector: str) -> str:
    text = str(sector).strip()
    if not text or text in {"其他", "其它", "未分类"}:
        return "其他"
    preferred_themes = {
        "光模块",
        "航天军工",
        "能源石油",
        "煤化工",
        "煤炭",
        "资源",
        "新能源电力",
        "半导体",
        "通信设备",
        "科技软件",
        "汽车",
        "医药",
        "消费",
        "金融",
    }
    if text in preferred_themes:
        return text
    lowered = text.lower()
    theme_keywords = {
        "资源": ("有色", "黄金", "煤炭", "钢铁", "铜", "铝", "锌", "稀土", "资源", "mining", "metal", "resource"),
        "科技": ("科技", "半导体", "算力", "软件", "通信", "电子", "ai", "芯片", "computer", "tech"),
        "高端制造": ("机械", "设备", "工业", "制造", "机器人", "自动化"),
        "消费": ("消费", "食品", "饮料", "家电", "零售", "consumer"),
        "金融": ("金融", "证券", "银行", "保险", "broker", "bank"),
        "能源": ("电力", "新能源", "光伏", "风电", "石油", "天然气", "energy", "solar"),
        "化工": ("化工", "材料", "煤化工", "chemical", "material"),
    }
    for theme, keywords in theme_keywords.items():
        if text == theme:
            return theme
        if any(keyword in text or keyword in lowered for keyword in keywords):
            return theme
    return text


def _capital_support_score(capital_flow_state: CapitalFlowState | None) -> float:
    if capital_flow_state is None:
        return 0.5
    flow_bonus = 0.0
    regime = str(getattr(capital_flow_state, "flow_regime", "neutral"))
    if regime == "strong_inflow":
        flow_bonus = 0.18
    elif regime == "inflow":
        flow_bonus = 0.10
    elif regime == "outflow":
        flow_bonus = -0.08
    elif regime == "strong_outflow":
        flow_bonus = -0.15
    score = (
        0.50
        + flow_bonus
        + 0.16 * float(getattr(capital_flow_state, "turnover_heat", 0.5) - 0.5)
        + 0.12 * float(getattr(capital_flow_state, "large_order_bias", 0.0))
    )
    return float(_clip(score, 0.0, 1.0))


def _macro_alignment_score(
    *,
    market: MarketForecastState,
    cross_section: CrossSectionForecastState,
    macro_context_state: MacroContextState | None,
) -> float:
    macro_penalty = 0.0
    if macro_context_state is not None:
        macro_risk = str(getattr(macro_context_state, "macro_risk_level", "neutral"))
        if macro_risk == "high":
            macro_penalty = 0.18
        elif macro_risk == "elevated":
            macro_penalty = 0.10
    score = (
        0.45
        + 0.18 * float(market.up_20d_prob - 0.5)
        + 0.14 * float(cross_section.breadth_strength)
        + 0.10 * float(cross_section.leader_participation - 0.5)
        - 0.12 * float(cross_section.weak_stock_ratio)
        - macro_penalty
    )
    return float(_clip(score, 0.0, 1.0))


def _driver_label(
    *,
    catalyst_strength: float,
    event_risk_level: float,
    rotation_speed: float,
    capital_support: float,
    relative_strength: float,
) -> str:
    if event_risk_level >= 0.55:
        return "risk_watched"
    if catalyst_strength >= 0.40:
        return "catalyst_confirmed"
    if capital_support >= 0.62 and relative_strength >= 0.08:
        return "flow_supported"
    if rotation_speed >= 0.28:
        return "rotation_lead"
    return "sector_leadership"


def build_mainline_states(
    *,
    market: MarketForecastState,
    cross_section: CrossSectionForecastState,
    sectors: Iterable[SectorForecastState],
    stocks: Iterable[StockForecastState],
    stock_score_fn: Callable[[StockForecastState], float],
    sector_info_states: dict[str, InfoAggregateState] | None = None,
    stock_info_states: dict[str, InfoAggregateState] | None = None,
    capital_flow_state: CapitalFlowState | None = None,
    macro_context_state: MacroContextState | None = None,
    limit: int = 3,
) -> list[MainlineState]:
    sector_info_states = dict(sector_info_states or {})
    stock_info_states = dict(stock_info_states or {})
    sector_rows = list(sectors)
    stock_rows = list(stocks)
    if not sector_rows or not stock_rows:
        return []

    sectors_by_theme: dict[str, list[SectorForecastState]] = defaultdict(list)
    for sector in sector_rows:
        sectors_by_theme[_theme_name_for_sector(sector.sector)].append(sector)

    stocks_by_sector: dict[str, list[StockForecastState]] = defaultdict(list)
    for stock in stock_rows:
        stocks_by_sector[str(stock.sector)].append(stock)

    capital_support = _capital_support_score(capital_flow_state)
    macro_alignment = _macro_alignment_score(
        market=market,
        cross_section=cross_section,
        macro_context_state=macro_context_state,
    )
    out: list[MainlineState] = []
    for theme_name, theme_sectors in sectors_by_theme.items():
        theme_stocks = [
            stock
            for sector in theme_sectors
            for stock in stocks_by_sector.get(str(sector.sector), [])
        ]
        if not theme_stocks:
            continue
        stock_scores = sorted(
            ((stock, float(stock_score_fn(stock))) for stock in theme_stocks),
            key=lambda item: item[1],
            reverse=True,
        )
        raw_scores = [score for _, score in stock_scores]
        median_score = float(np.median(raw_scores)) if raw_scores else 0.0
        breadth = float(
            sum(1 for score in raw_scores if score >= max(0.56, median_score + 0.02))
            / max(1, len(raw_scores))
        )
        leadership = float(
            np.mean(raw_scores[: min(2, len(raw_scores))]) - 0.50
            if raw_scores
            else 0.0
        )
        sector_trend = float(np.mean([float(sector.up_20d_prob) for sector in theme_sectors]))
        sector_strength = float(np.mean([max(0.0, float(sector.relative_strength)) for sector in theme_sectors]))
        rotation_speed = float(np.mean([float(sector.rotation_speed) for sector in theme_sectors]))
        crowding = float(np.mean([float(sector.crowding_score) for sector in theme_sectors]))
        sector_catalysts = [
            float(sector_info_states.get(str(sector.sector), InfoAggregateState()).catalyst_strength)
            for sector in theme_sectors
        ]
        sector_risks = [
            float(sector_info_states.get(str(sector.sector), InfoAggregateState()).event_risk_level)
            for sector in theme_sectors
        ]
        top_stock_info = [
            stock_info_states.get(stock.symbol, InfoAggregateState())
            for stock, _ in stock_scores[: min(3, len(stock_scores))]
        ]
        catalyst_strength = float(
            _clip(
                0.60 * (float(np.mean(sector_catalysts)) if sector_catalysts else 0.0)
                + 0.40
                * (
                    float(np.mean([float(item.catalyst_strength) for item in top_stock_info]))
                    if top_stock_info
                    else 0.0
                ),
                0.0,
                1.0,
            )
        )
        event_risk_level = float(
            _clip(
                0.65 * (float(np.mean(sector_risks)) if sector_risks else 0.0)
                + 0.35
                * (
                    float(np.mean([float(item.event_risk_level) for item in top_stock_info]))
                    if top_stock_info
                    else 0.0
                ),
                0.0,
                1.0,
            )
        )
        conviction = float(
            _clip(
                0.24 * _clip((sector_trend - 0.50) / 0.16, 0.0, 1.0)
                + 0.22 * _clip(sector_strength / 0.20, 0.0, 1.0)
                + 0.16 * breadth
                + 0.14 * _clip(leadership / 0.18, 0.0, 1.0)
                + 0.10 * catalyst_strength
                + 0.08 * capital_support
                + 0.06 * macro_alignment
                - 0.12 * event_risk_level
                - 0.06 * max(0.0, crowding - 0.60),
                0.0,
                1.0,
            )
        )
        driver = _driver_label(
            catalyst_strength=catalyst_strength,
            event_risk_level=event_risk_level,
            rotation_speed=rotation_speed,
            capital_support=capital_support,
            relative_strength=sector_strength,
        )
        out.append(
            MainlineState(
                name=str(theme_name),
                driver=driver,
                conviction=conviction,
                breadth=float(_clip(breadth, 0.0, 1.0)),
                leadership=float(_clip(leadership, 0.0, 1.0)),
                catalyst_strength=catalyst_strength,
                event_risk_level=event_risk_level,
                capital_support=capital_support,
                macro_alignment=macro_alignment,
                sectors=[str(sector.sector) for sector in theme_sectors],
                representative_symbols=[stock.symbol for stock, _ in stock_scores[:3]],
            )
        )
    out.sort(
        key=lambda item: (
            float(item.conviction),
            float(item.catalyst_strength) - float(item.event_risk_level),
            float(item.leadership),
        ),
        reverse=True,
    )
    return out[: max(1, int(limit))]


def dominant_mainline_sectors(
    mainlines: Iterable[MainlineState],
    *,
    conviction_cutoff: float = 0.55,
    risk_cutoff: float = 0.55,
) -> set[str]:
    out: set[str] = set()
    for mainline in mainlines:
        if float(mainline.conviction) < float(conviction_cutoff):
            continue
        if float(mainline.event_risk_level) >= float(risk_cutoff):
            continue
        out.update(str(sector) for sector in mainline.sectors)
    return out
