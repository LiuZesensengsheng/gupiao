from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from src.application.v2_contracts import CompositeState, CrossSectionForecastState, InfoAggregateState, MarketForecastState, PolicyDecision, StockForecastState


@dataclass(frozen=True)
class ReportStateRuntimeDependencies:
    build_horizon_forecasts: Callable[..., object]
    profile_from_horizon_map: Callable[[dict[str, object], str], object | None]
    build_market_sentiment_state: Callable[..., object]
    candidate_stocks_from_state: Callable[[CompositeState], list[StockForecastState]]
    stock_reason_bundle: Callable[..., tuple[list[str], list[str], list[str], str, str, str, str]]
    is_main_board_symbol: Callable[[object], bool]
    build_candidate_selection_state: Callable[..., object]
    stock_policy_score: Callable[[StockForecastState], float]
    build_market_and_cross_section_states: Callable[..., tuple[MarketForecastState, CrossSectionForecastState]]


def profile_from_horizon_map(
    forecasts: dict[str, object],
    key: str,
    *,
    return_quantile_profile_cls: type,
) -> object | None:
    item = forecasts.get(key)
    if item is None:
        return None
    q10 = float(item.q10)
    q50 = float(item.q50)
    q90 = float(item.q90)
    q30 = float((2.0 * q10 + q50) / 3.0)
    q70 = float((q50 + 2.0 * q90) / 3.0)
    return return_quantile_profile_cls(
        expected_return=float(item.expected_return),
        q10=q10,
        q30=q30,
        q20=float(0.5 * (q10 + q30)),
        q50=q50,
        q70=q70,
        q80=float(0.5 * (q70 + q90)),
        q90=q90,
    )


def decorate_composite_state_for_reporting(
    *,
    state: CompositeState,
    policy: PolicyDecision,
    calibration_priors: dict[str, dict[str, float]],
    reporting_market: MarketForecastState | None = None,
    reporting_cross_section: CrossSectionForecastState | None = None,
    deps: ReportStateRuntimeDependencies,
) -> CompositeState:
    base_market = reporting_market or state.market
    base_cross_section = reporting_cross_section or state.cross_section
    updated_market = replace(
        base_market,
        horizon_forecasts=deps.build_horizon_forecasts(
            latest_close=float(getattr(base_market, "latest_close", np.nan)),
            horizon_probs={
                "1d": float(base_market.up_1d_prob),
                "2d": float(base_market.up_2d_prob),
                "3d": float(base_market.up_3d_prob),
                "5d": float(base_market.up_5d_prob),
                "10d": float(getattr(base_market, "up_10d_prob", 0.45 * base_market.up_5d_prob + 0.55 * base_market.up_20d_prob)),
                "20d": float(base_market.up_20d_prob),
            },
            short_profile=deps.profile_from_horizon_map(dict(getattr(base_market, "horizon_forecasts", {})), "1d"),
            mid_profile=deps.profile_from_horizon_map(dict(getattr(base_market, "horizon_forecasts", {})), "20d"),
            calibration_priors=calibration_priors,
        ),
        sentiment=deps.build_market_sentiment_state(
            market=base_market,
            cross_section=base_cross_section,
            capital_flow=state.capital_flow_state,
            macro=state.macro_context_state,
        ),
    )
    reporting_state = replace(state, market=updated_market, cross_section=base_cross_section)
    ordered_candidates = deps.candidate_stocks_from_state(reporting_state)
    rank_map = {stock.symbol: idx for idx, stock in enumerate(ordered_candidates, start=1)}
    updated_stocks: list[StockForecastState] = []
    for stock in reporting_state.stocks:
        info_state = reporting_state.stock_info_states.get(stock.symbol, InfoAggregateState())
        refreshed_forecasts = deps.build_horizon_forecasts(
            latest_close=float(getattr(stock, "latest_close", np.nan)),
            horizon_probs={
                "1d": float(stock.up_1d_prob),
                "2d": float(stock.up_2d_prob),
                "3d": float(stock.up_3d_prob),
                "5d": float(stock.up_5d_prob),
                "10d": float(getattr(stock, "up_10d_prob", 0.45 * stock.up_5d_prob + 0.55 * stock.up_20d_prob)),
                "20d": float(stock.up_20d_prob),
            },
            short_profile=deps.profile_from_horizon_map(dict(getattr(stock, "horizon_forecasts", {})), "1d"),
            mid_profile=deps.profile_from_horizon_map(dict(getattr(stock, "horizon_forecasts", {})), "20d"),
            info_state=info_state,
            calibration_priors=calibration_priors,
            tradability_status=str(getattr(stock, "tradability_status", "normal")),
        )
        reasons, ranking, risks, invalidation, action_reason, weight_reason, blocked_reason = deps.stock_reason_bundle(
            stock=stock,
            info_state=info_state,
            state=reporting_state,
            rank=int(rank_map.get(stock.symbol, len(rank_map) + 1)),
            policy=policy,
        )
        updated_stocks.append(
            replace(
                stock,
                horizon_forecasts=refreshed_forecasts,
                selection_reasons=reasons,
                ranking_reasons=ranking,
                risk_flags=risks,
                invalidation_rule=invalidation,
                action_reason=action_reason,
                weight_reason=weight_reason,
                blocked_reason=blocked_reason,
            )
        )
    updated_stocks.sort(key=lambda item: rank_map.get(item.symbol, len(rank_map) + 999))
    return replace(reporting_state, market=updated_market, cross_section=base_cross_section, stocks=updated_stocks)


def filter_state_for_recommendation_scope(
    *,
    state: CompositeState,
    main_board_only: bool,
    deps: ReportStateRuntimeDependencies,
) -> CompositeState:
    if not main_board_only:
        return state
    filtered_stocks = [stock for stock in state.stocks if deps.is_main_board_symbol(stock.symbol)]
    if not filtered_stocks:
        return state
    filtered_selection = deps.build_candidate_selection_state(
        market=state.market,
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=filtered_stocks,
        mainlines=state.mainlines,
        strategy_mode=state.strategy_mode,
        risk_regime=state.risk_regime,
        stock_score_fn=deps.stock_policy_score,
    )
    selection_notes = list(getattr(filtered_selection, "selection_notes", []) or [])
    selection_notes.append("Recommendation scope limited to main-board listings only.")
    filtered_selection = replace(filtered_selection, selection_notes=selection_notes)
    filtered_symbols = {stock.symbol for stock in filtered_stocks}
    filtered_info_states = {
        symbol: payload
        for symbol, payload in state.stock_info_states.items()
        if symbol in filtered_symbols
    }
    return replace(
        state,
        stocks=filtered_stocks,
        candidate_selection=filtered_selection,
        stock_info_states=filtered_info_states,
    )


def build_live_market_reporting_overlay(
    *,
    settings: dict[str, object],
    universe_ctx: object,
    state: CompositeState,
    deps: ReportStateRuntimeDependencies,
) -> tuple[MarketForecastState | None, CrossSectionForecastState | None]:
    try:
        market_state, cross_section = deps.build_market_and_cross_section_states(
            market_symbol=str(getattr(getattr(universe_ctx, "market_security", None), "symbol", "")),
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
            use_margin_features=bool(settings.get("use_margin_features", False)),
            margin_market_file=str(settings.get("margin_market_file", "")),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
            use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
            use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
            cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
            market_short_prob=float(state.market.up_1d_prob),
            market_two_prob=float(getattr(state.market, "up_2d_prob", 0.5)),
            market_three_prob=float(getattr(state.market, "up_3d_prob", 0.5)),
            market_five_prob=float(state.market.up_5d_prob),
            market_mid_prob=float(state.market.up_20d_prob),
            market_short_profile=deps.profile_from_horizon_map(dict(getattr(state.market, "horizon_forecasts", {})), "1d"),
            market_mid_profile=deps.profile_from_horizon_map(dict(getattr(state.market, "horizon_forecasts", {})), "20d"),
        )
        return market_state, cross_section
    except Exception:
        return None, None
