from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    DailyRunResult,
    MarketForecastState,
    PolicyDecision,
    PolicyInput,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
    StrategySnapshot,
)
from src.application.watchlist import load_watchlist
from src.domain.entities import TradeAction
from src.domain.policies import decide_market_state
from src.infrastructure.discovery import build_candidate_universe
from src.infrastructure.cross_section_forecast import forecast_cross_section_state
from src.infrastructure.features import (
    MARKET_FEATURE_COLUMNS,
    make_market_feature_frame,
    make_stock_feature_frame,
    stock_feature_columns,
)
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import load_symbol_daily
from src.infrastructure.modeling import LogisticBinaryModel
from src.infrastructure.sector_data import build_sector_daily_frames
from src.infrastructure.sector_forecast import run_sector_forecast


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _coalesce(primary: object, secondary: object, default: object) -> object:
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    return default


def build_strategy_snapshot(
    *,
    strategy_id: str,
    universe_id: str = "top_liquid_200",
) -> StrategySnapshot:
    return StrategySnapshot(
        strategy_id=str(strategy_id).strip() or "swing_v2",
        universe_id=str(universe_id).strip() or "top_liquid_200",
        feature_set_version="fset_v2_core",
        market_model_id="mkt_lr_v2",
        sector_model_id="sector_lr_v2",
        stock_model_id="stock_lr_v2",
        cross_section_model_id="cross_section_v2",
        policy_version="policy_v2_rules",
        execution_version="exec_t1_v2",
    )


def _load_v2_runtime_settings(
    *,
    config_path: str,
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            payload = raw

    common = payload.get("common", {}) if isinstance(payload.get("common"), dict) else {}
    daily = payload.get("daily", {}) if isinstance(payload.get("daily"), dict) else {}

    def pick(key: str, default: object) -> object:
        return _coalesce(daily.get(key), common.get(key), default)

    return {
        "config_path": str(config_path),
        "watchlist": str(pick("watchlist", "config/watchlist.json")),
        "source": str(source).strip() if source is not None and str(source).strip() else str(pick("source", "auto")),
        "data_dir": str(pick("data_dir", "data")),
        "start": str(pick("start", "2018-01-01")),
        "end": str(pick("end", "2099-12-31")),
        "min_train_days": int(pick("min_train_days", 240)),
        "step_days": int(pick("step_days", 20)),
        "l2": float(pick("l2", 0.8)),
        "max_positions": int(pick("max_positions", 5)),
        "use_margin_features": bool(pick("use_margin_features", True)),
        "margin_market_file": str(pick("margin_market_file", "input/margin_market.csv")),
        "margin_stock_file": str(pick("margin_stock_file", "input/margin_stock.csv")),
        "universe_file": (
            str(universe_file).strip()
            if universe_file is not None and str(universe_file).strip()
            else str(pick("universe_file", "config/universe_smoke_5.json"))
        ),
        "universe_limit": int(
            universe_limit
            if universe_limit is not None
            else int(pick("universe_limit", 5))
        ),
    }


def _build_market_and_cross_section_states(
    *,
    market_symbol: str,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    use_margin_features: bool,
    margin_market_file: str,
    market_short_prob: float,
    market_mid_prob: float,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    market_raw = load_symbol_daily(
        symbol=market_symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        market_dates=market_feat_base["date"],
        use_margin_features=use_margin_features,
        margin_market_file=margin_market_file,
    )
    market_frame = market_feat_base.merge(market_context.frame, on="date", how="left")
    latest = market_frame.sort_values("date").iloc[-1]
    state = decide_market_state(float(market_short_prob), float(market_mid_prob))
    cross_section_record = forecast_cross_section_state(market_frame)

    mkt_vol_20 = float(latest.get("mkt_volatility_20", 0.0))
    if mkt_vol_20 != mkt_vol_20:
        mkt_vol_20 = 0.0
    mkt_vol_60 = float(latest.get("mkt_volatility_60", mkt_vol_20))
    if mkt_vol_60 != mkt_vol_60:
        mkt_vol_60 = mkt_vol_20
    mkt_vol_60 = max(1e-6, mkt_vol_60)
    if mkt_vol_20 >= mkt_vol_60 * 1.15:
        volatility_regime = "high"
    elif mkt_vol_20 <= mkt_vol_60 * 0.85:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    market = MarketForecastState(
        as_of_date=str(latest["date"].date()),
        up_1d_prob=float(market_short_prob),
        up_5d_prob=float(0.6 * market_short_prob + 0.4 * market_mid_prob),
        up_20d_prob=float(market_mid_prob),
        trend_state=str(state.state_code),
        drawdown_risk=_clip(abs(float(latest.get("mkt_drawdown_20", 0.0) or 0.0)), 0.0, 1.0),
        volatility_regime=volatility_regime,
        liquidity_stress=_clip(0.5 - float(cross_section_record.breadth_strength), 0.0, 1.0),
    )
    cross_section = CrossSectionForecastState(
        as_of_date=str(cross_section_record.as_of_date.date()),
        large_vs_small_bias=float(cross_section_record.large_vs_small_bias),
        growth_vs_value_bias=float(cross_section_record.growth_vs_value_bias),
        fund_flow_strength=float(cross_section_record.fund_flow_strength),
        margin_risk_on_score=float(cross_section_record.margin_risk_on_score),
        breadth_strength=float(cross_section_record.breadth_strength),
        leader_participation=float(cross_section_record.leader_participation),
        weak_stock_ratio=float(cross_section_record.weak_stock_ratio),
    )
    return market, cross_section


def _build_stock_states_from_rows(
    rows: list[object],
    sector_map: dict[str, str],
    *,
    sector_strength_map: dict[str, float] | None = None,
) -> list[StockForecastState]:
    out: list[StockForecastState] = []
    sector_mid_map: dict[str, float] = {}
    sector_counts: dict[str, int] = {}
    for row in rows:
        sector = sector_map.get(row.symbol, "其他")
        sector_mid_map[sector] = sector_mid_map.get(sector, 0.0) + float(row.mid_prob)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    sector_avg_mid = {
        sector: sector_mid_map[sector] / max(1, sector_counts[sector])
        for sector in sector_mid_map
    }

    for row in rows:
        sector = sector_map.get(row.symbol, "其他")
        tradeability = _clip(1.0 - abs(float(row.short_prob) - float(row.mid_prob)), 0.0, 1.0)
        sector_excess_anchor = 0.0 if sector_strength_map is None else float(sector_strength_map.get(sector, 0.0))
        out.append(
            StockForecastState(
                symbol=row.symbol,
                sector=sector,
                up_1d_prob=float(row.short_prob),
                up_5d_prob=float(0.6 * float(row.short_prob) + 0.4 * float(row.mid_prob)),
                up_20d_prob=float(row.mid_prob),
                excess_vs_sector_prob=float(
                    _clip(
                        float(row.mid_prob)
                        - sector_avg_mid.get(sector, float(row.mid_prob))
                        + 0.5
                        + 0.2 * sector_excess_anchor,
                        0.0,
                        1.0,
                    )
                ),
                event_impact_score=0.0,
                tradeability_score=float(tradeability),
            )
        )
    out.sort(
        key=lambda item: (item.up_20d_prob, item.excess_vs_sector_prob, item.tradeability_score),
        reverse=True,
    )
    return out


def compose_state(
    *,
    market: MarketForecastState,
    sectors: list[SectorForecastState],
    stocks: list[StockForecastState],
    cross_section: CrossSectionForecastState,
) -> CompositeState:
    risk_score = max(
        float(market.drawdown_risk),
        float(market.liquidity_stress),
        float(cross_section.weak_stock_ratio),
    )
    if risk_score >= 0.60:
        risk_regime = "risk_off"
    elif risk_score >= 0.40:
        risk_regime = "cautious"
    else:
        risk_regime = "risk_on"

    if market.trend_state == "trend" and cross_section.breadth_strength >= 0.10:
        strategy_mode = "trend_follow"
    elif market.trend_state == "range":
        strategy_mode = "range_rotation"
    else:
        strategy_mode = "defensive"

    ordered_sectors = sorted(sectors, key=lambda item: (item.up_20d_prob, item.relative_strength), reverse=True)
    ordered_stocks = sorted(
        stocks,
        key=lambda item: (item.up_20d_prob, item.excess_vs_sector_prob, item.tradeability_score),
        reverse=True,
    )
    return CompositeState(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        strategy_mode=strategy_mode,
        risk_regime=risk_regime,
    )


def _ranked_sector_budgets(sectors: Iterable[SectorForecastState], *, target_exposure: float) -> dict[str, float]:
    rows = list(sectors)
    if not rows:
        return {}
    raw = [max(0.0, float(item.up_20d_prob) - 0.50) + max(0.0, float(item.relative_strength)) for item in rows]
    total = sum(raw)
    if total <= 1e-9:
        equal = float(target_exposure) / float(len(rows))
        return {item.sector: equal for item in rows}
    return {item.sector: float(target_exposure) * score / total for item, score in zip(rows, raw)}


def _stock_policy_score(stock: StockForecastState) -> float:
    return float(
        0.45 * float(stock.up_20d_prob)
        + 0.25 * float(stock.up_5d_prob)
        + 0.15 * float(stock.excess_vs_sector_prob)
        + 0.10 * float(stock.tradeability_score)
        + 0.05 * float(stock.event_impact_score)
    )


def _allocate_sector_slots(
    *,
    sector_budgets: dict[str, float],
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]],
    total_slots: int,
) -> dict[str, int]:
    active_sectors = [
        sector for sector, budget in sector_budgets.items()
        if float(budget) > 1e-9 and available_by_sector.get(sector)
    ]
    if not active_sectors or total_slots <= 0:
        return {}

    slots = {sector: 0 for sector in active_sectors}
    ordered = sorted(active_sectors, key=lambda sector: float(sector_budgets.get(sector, 0.0)), reverse=True)

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
                float(sector_budgets.get(sector, 0.0)) / float(max(1, slots[sector] + 1)),
                len(available_by_sector.get(sector, [])),
            ),
        )
        slots[best] += 1
        total_slots -= 1
    return slots


def _allocate_with_sector_budgets(
    *,
    stocks: list[StockForecastState],
    sector_budgets: dict[str, float],
    target_position_count: int,
) -> dict[str, float]:
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]] = {}
    for stock in stocks:
        score = _stock_policy_score(stock)
        available_by_sector.setdefault(stock.sector, []).append((stock, score))
    for sector in list(available_by_sector):
        available_by_sector[sector].sort(key=lambda item: item[1], reverse=True)

    slots_by_sector = _allocate_sector_slots(
        sector_budgets=sector_budgets,
        available_by_sector=available_by_sector,
        total_slots=max(1, int(target_position_count)),
    )
    symbol_target_weights: dict[str, float] = {}
    for sector, slots in slots_by_sector.items():
        sector_budget = float(sector_budgets.get(sector, 0.0))
        picks = available_by_sector.get(sector, [])[: max(0, int(slots))]
        if not picks or sector_budget <= 1e-9:
            continue
        sector_scores = [max(0.0, score - 0.50) for _, score in picks]
        sector_total = float(sum(sector_scores))
        if sector_total <= 1e-9:
            equal_weight = sector_budget / float(len(picks))
            for stock, _ in picks:
                symbol_target_weights[stock.symbol] = float(equal_weight)
            continue
        for (stock, _), score in zip(picks, sector_scores):
            symbol_target_weights[stock.symbol] = float(sector_budget) * float(score) / sector_total
    return symbol_target_weights


def apply_policy(
    policy_input: PolicyInput,
    *,
    policy_spec: PolicySpec | None = None,
) -> PolicyDecision:
    policy_spec = policy_spec or PolicySpec()
    state = policy_input.composite_state
    market = state.market
    cross = state.cross_section

    target_exposure = float(policy_spec.risk_off_exposure)
    target_position_count = int(policy_spec.risk_off_positions)
    turnover_cap = float(policy_spec.risk_off_turnover_cap)
    intraday_t_allowed = False
    risk_notes: list[str] = []

    if state.risk_regime == "risk_on":
        target_exposure = float(policy_spec.risk_on_exposure)
        target_position_count = int(policy_spec.risk_on_positions)
        turnover_cap = float(policy_spec.risk_on_turnover_cap)
        intraday_t_allowed = state.strategy_mode == "range_rotation"
    elif state.risk_regime == "cautious":
        target_exposure = float(policy_spec.cautious_exposure)
        target_position_count = int(policy_spec.cautious_positions)
        turnover_cap = float(policy_spec.cautious_turnover_cap)
    else:
        risk_notes.append("Risk-off regime: exposure capped aggressively.")

    if market.up_20d_prob < 0.50:
        target_exposure = min(target_exposure, 0.45)
        risk_notes.append("Market mid-term probability below 0.50.")
    if market.drawdown_risk >= 0.50:
        target_exposure = min(target_exposure, 0.35)
        turnover_cap = min(turnover_cap, 0.18)
        risk_notes.append("Drawdown risk elevated.")
    if cross.fund_flow_strength < 0.0:
        target_exposure *= 0.85
        risk_notes.append("Fund flow weak.")

    target_exposure = _clip(target_exposure, 0.0, 1.0)
    sector_budgets = _ranked_sector_budgets(state.sectors[: max(1, target_position_count)], target_exposure=target_exposure)
    symbol_target_weights = _allocate_with_sector_budgets(
        stocks=state.stocks,
        sector_budgets=sector_budgets,
        target_position_count=int(target_position_count),
    )

    current_total = sum(max(0.0, float(v)) for v in policy_input.current_weights.values())
    rebalance_gap = abs(float(target_exposure) - float(current_total))
    rebalance_now = rebalance_gap >= 0.05
    rebalance_intensity = _clip(rebalance_gap / max(0.05, turnover_cap), 0.0, 1.0)

    return PolicyDecision(
        target_exposure=float(target_exposure),
        target_position_count=int(target_position_count),
        rebalance_now=bool(rebalance_now),
        rebalance_intensity=float(rebalance_intensity),
        intraday_t_allowed=bool(intraday_t_allowed),
        turnover_cap=float(turnover_cap),
        sector_budgets=sector_budgets,
        symbol_target_weights=symbol_target_weights,
        risk_notes=risk_notes,
    )


def build_trade_actions(
    *,
    decision: PolicyDecision,
    current_weights: dict[str, float],
) -> list[TradeAction]:
    all_symbols = sorted(set(current_weights) | set(decision.symbol_target_weights))
    actions: list[TradeAction] = []
    for symbol in all_symbols:
        current_weight = max(0.0, float(current_weights.get(symbol, 0.0)))
        target_weight = max(0.0, float(decision.symbol_target_weights.get(symbol, 0.0)))
        delta_weight = float(target_weight - current_weight)
        if delta_weight > 0.02:
            action = "BUY"
        elif delta_weight < -0.02:
            action = "SELL"
        else:
            action = "HOLD"

        note = ""
        if action == "HOLD" and abs(delta_weight) > 1e-9:
            note = "below_rebalance_threshold"

        actions.append(
            TradeAction(
                symbol=symbol,
                name=symbol,
                action=action,
                current_weight=float(current_weight),
                target_weight=float(target_weight),
                delta_weight=float(delta_weight),
                note=note,
            )
        )
    actions.sort(key=lambda item: (abs(float(item.delta_weight)), float(item.target_weight)), reverse=True)
    return actions


def _to_v2_backtest_summary(
    *,
    returns: list[float],
    turnovers: list[float],
    costs: list[float],
    dates: list[pd.Timestamp],
) -> V2BacktestSummary:
    if not returns or not dates:
        return V2BacktestSummary(
            start_date="",
            end_date="",
            n_days=0,
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            avg_turnover=0.0,
            total_cost=0.0,
        )
    nav = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.maximum(peak, 1e-12) - 1.0
    total_return = float(nav[-1] - 1.0)
    n_days = len(returns)
    annual_return = float((1.0 + total_return) ** (252.0 / max(1, n_days)) - 1.0)
    return V2BacktestSummary(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        n_days=int(n_days),
        total_return=float(total_return),
        annual_return=float(annual_return),
        max_drawdown=float(np.min(drawdown)),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        total_cost=float(np.sum(costs)) if costs else 0.0,
    )


def _build_market_and_cross_section_from_prebuilt_frame(
    *,
    market_frame: pd.DataFrame,
    market_short_prob: float,
    market_mid_prob: float,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    latest = market_frame.sort_values("date").iloc[-1]
    state = decide_market_state(float(market_short_prob), float(market_mid_prob))
    cross_section_record = forecast_cross_section_state(market_frame)

    mkt_vol_20 = float(latest.get("mkt_volatility_20", 0.0))
    if mkt_vol_20 != mkt_vol_20:
        mkt_vol_20 = 0.0
    mkt_vol_60 = float(latest.get("mkt_volatility_60", mkt_vol_20))
    if mkt_vol_60 != mkt_vol_60:
        mkt_vol_60 = mkt_vol_20
    mkt_vol_60 = max(1e-6, mkt_vol_60)
    if mkt_vol_20 >= mkt_vol_60 * 1.15:
        volatility_regime = "high"
    elif mkt_vol_20 <= mkt_vol_60 * 0.85:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    drawdown_raw = float(latest.get("mkt_drawdown_20", 0.0))
    if drawdown_raw != drawdown_raw:
        drawdown_raw = 0.0
    market = MarketForecastState(
        as_of_date=str(latest["date"].date()),
        up_1d_prob=float(market_short_prob),
        up_5d_prob=float(0.6 * market_short_prob + 0.4 * market_mid_prob),
        up_20d_prob=float(market_mid_prob),
        trend_state=str(state.state_code),
        drawdown_risk=_clip(abs(drawdown_raw), 0.0, 1.0),
        volatility_regime=volatility_regime,
        liquidity_stress=_clip(0.5 - float(cross_section_record.breadth_strength), 0.0, 1.0),
    )
    cross_section = CrossSectionForecastState(
        as_of_date=str(cross_section_record.as_of_date.date()),
        large_vs_small_bias=float(cross_section_record.large_vs_small_bias),
        growth_vs_value_bias=float(cross_section_record.growth_vs_value_bias),
        fund_flow_strength=float(cross_section_record.fund_flow_strength),
        margin_risk_on_score=float(cross_section_record.margin_risk_on_score),
        breadth_strength=float(cross_section_record.breadth_strength),
        leader_participation=float(cross_section_record.leader_participation),
        weak_stock_ratio=float(cross_section_record.weak_stock_ratio),
    )
    return market, cross_section


def run_v2_backtest_live(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    policy_spec: PolicySpec | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> V2BacktestSummary:
    settings = _load_v2_runtime_settings(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    market_security, _, _ = load_watchlist(str(settings["watchlist"]))
    universe = build_candidate_universe(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    stocks = universe.rows
    if not stocks:
        return _to_v2_backtest_summary(returns=[], turnovers=[], costs=[], dates=[])

    sector_map = {stock.symbol: (stock.sector or "其他") for stock in stocks}
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_dates=market_feat_base["date"],
        use_margin_features=bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
    )
    market_frame = market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")
    market_feature_cols = list(MARKET_FEATURE_COLUMNS) + list(market_context.feature_columns)
    market_valid = market_frame.dropna(subset=market_feature_cols + ["mkt_target_1d_up", "mkt_target_20d_up"]).sort_values("date").copy()
    if market_valid.empty:
        return _to_v2_backtest_summary(returns=[], turnovers=[], costs=[], dates=[])

    stock_frames: dict[str, pd.DataFrame] = {}
    stock_cols_map: dict[str, list[str]] = {}
    for stock in stocks:
        raw = load_symbol_daily(
            symbol=stock.symbol,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
        )
        feat = make_stock_feature_frame(raw, market_frame)
        extra_stock_cols: list[str] = []
        if bool(settings["use_margin_features"]):
            margin_frame, margin_cols, _ = build_stock_margin_features(
                margin_stock_file=str(settings["margin_stock_file"]),
                symbol=stock.symbol,
                start=str(settings["start"]),
                end=str(settings["end"]),
            )
            if margin_cols:
                feat = feat.merge(margin_frame, on="date", how="left", validate="1:1")
                extra_stock_cols = list(margin_cols)
        cols = stock_feature_columns(
            extra_market_cols=market_context.feature_columns,
            extra_stock_cols=extra_stock_cols,
        )
        valid = feat.dropna(subset=cols + ["target_1d_up", "target_20d_up", "fwd_ret_1"]).sort_values("date").copy()
        if valid.empty:
            continue
        stock_frames[stock.symbol] = valid
        stock_cols_map[stock.symbol] = cols

    if not stock_frames:
        return _to_v2_backtest_summary(returns=[], turnovers=[], costs=[], dates=[])

    common_dates = set(pd.to_datetime(market_valid["date"]))
    for frame in stock_frames.values():
        common_dates &= set(pd.to_datetime(frame["date"]))
    dates = sorted(pd.Timestamp(d) for d in common_dates)
    min_train_days = int(settings["min_train_days"])
    if len(dates) <= min_train_days + 1:
        return _to_v2_backtest_summary(returns=[], turnovers=[], costs=[], dates=[])

    total_cost_rate = max(0.0, float(commission_bps) + float(slippage_bps)) / 10000.0
    returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    out_dates: list[pd.Timestamp] = []
    prev_weights: dict[str, float] = {}

    for block_start in range(min_train_days, len(dates) - 1, max(1, int(retrain_days))):
        train_dates = set(dates[:block_start])
        market_train = market_valid[market_valid["date"].isin(train_dates)].copy()
        if market_train.empty:
            continue
        market_short_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
            market_train,
            market_feature_cols,
            "mkt_target_1d_up",
        )
        market_mid_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
            market_train,
            market_feature_cols,
            "mkt_target_20d_up",
        )
        stock_models: dict[str, tuple[LogisticBinaryModel, LogisticBinaryModel]] = {}
        for symbol, frame in stock_frames.items():
            train = frame[frame["date"].isin(train_dates)].copy()
            cols = stock_cols_map[symbol]
            if train.empty:
                continue
            short_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(train, cols, "target_1d_up")
            mid_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(train, cols, "target_20d_up")
            stock_models[symbol] = (short_model, mid_model)

        block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
        for idx in range(block_start, block_end):
            date = dates[idx]
            next_date = dates[idx + 1]
            market_row = market_valid[market_valid["date"] == date].copy()
            if market_row.empty:
                continue
            mkt_short = float(market_short_model.predict_proba(market_row, market_feature_cols)[0])
            mkt_mid = float(market_mid_model.predict_proba(market_row, market_feature_cols)[0])
            market_state, cross_section = _build_market_and_cross_section_from_prebuilt_frame(
                market_frame=market_valid[market_valid["date"] <= date].copy(),
                market_short_prob=mkt_short,
                market_mid_prob=mkt_mid,
            )

            stock_states: list[StockForecastState] = []
            for symbol, models in stock_models.items():
                frame = stock_frames[symbol]
                row = frame[frame["date"] == date].copy()
                if row.empty:
                    continue
                cols = stock_cols_map[symbol]
                short_prob = float(models[0].predict_proba(row, cols)[0])
                mid_prob = float(models[1].predict_proba(row, cols)[0])
                sector = sector_map.get(symbol, "其他")
                stock_states.append(
                    StockForecastState(
                        symbol=symbol,
                        sector=sector,
                        up_1d_prob=float(short_prob),
                        up_5d_prob=float(0.6 * short_prob + 0.4 * mid_prob),
                        up_20d_prob=float(mid_prob),
                        excess_vs_sector_prob=0.5,
                        event_impact_score=0.0,
                        tradeability_score=float(_clip(1.0 - abs(short_prob - mid_prob), 0.0, 1.0)),
                    )
                )
            if not stock_states:
                continue

            grouped: dict[str, list[StockForecastState]] = {}
            for stock_state in stock_states:
                grouped.setdefault(stock_state.sector, []).append(stock_state)
            sector_states: list[SectorForecastState] = []
            for sector, items in grouped.items():
                n = max(1, len(items))
                up5 = sum(item.up_5d_prob for item in items) / n
                up20 = sum(item.up_20d_prob for item in items) / n
                rel = sum(item.up_20d_prob - 0.5 for item in items) / n
                rotation = sum(abs(item.up_5d_prob - item.up_20d_prob) for item in items) / n
                crowding = sum(max(0.0, item.up_20d_prob - 0.5) for item in items) / n
                sector_states.append(
                    SectorForecastState(
                        sector=sector,
                        up_5d_prob=float(up5),
                        up_20d_prob=float(up20),
                        relative_strength=float(rel),
                        rotation_speed=float(_clip(rotation, 0.0, 1.0)),
                        crowding_score=float(_clip(crowding, 0.0, 1.0)),
                    )
                )
            composite_state = compose_state(
                market=market_state,
                sectors=sector_states,
                stocks=stock_states,
                cross_section=cross_section,
            )
            decision = apply_policy(
                PolicyInput(
                    composite_state=composite_state,
                    current_weights=prev_weights,
                    current_cash=max(0.0, 1.0 - sum(prev_weights.values())),
                    total_equity=1.0,
                ),
                policy_spec=policy_spec,
            )
            turnover = float(
                sum(
                    abs(float(decision.symbol_target_weights.get(symbol, 0.0)) - float(prev_weights.get(symbol, 0.0)))
                    for symbol in set(prev_weights) | set(decision.symbol_target_weights)
                )
            )
            cost = float(turnover * total_cost_rate)
            gross_ret = 0.0
            for symbol, weight in decision.symbol_target_weights.items():
                frame = stock_frames.get(symbol)
                if frame is None:
                    continue
                row = frame[frame["date"] == date]
                if row.empty:
                    continue
                gross_ret += float(weight) * float(row.iloc[0]["fwd_ret_1"])
            returns.append(float(gross_ret - cost))
            turnovers.append(turnover)
            costs.append(cost)
            out_dates.append(next_date)
            prev_weights = {
                symbol: float(weight)
                for symbol, weight in decision.symbol_target_weights.items()
                if float(weight) > 1e-9
            }

    return _to_v2_backtest_summary(
        returns=returns,
        turnovers=turnovers,
        costs=costs,
        dates=out_dates,
    )


def calibrate_v2_policy(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
) -> V2CalibrationResult:
    baseline_spec = PolicySpec()
    baseline = run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        policy_spec=baseline_spec,
    )
    candidates = [
        baseline_spec,
        PolicySpec(risk_on_exposure=0.75, cautious_exposure=0.50, risk_off_exposure=0.25),
        PolicySpec(risk_on_exposure=0.90, cautious_exposure=0.65, risk_off_exposure=0.35),
        PolicySpec(risk_on_positions=5, cautious_positions=3, risk_off_positions=1),
    ]
    best_spec = baseline_spec
    best_summary = baseline
    best_score = float(baseline.annual_return) - 0.5 * abs(float(baseline.max_drawdown))
    for spec in candidates[1:]:
        summary = run_v2_backtest_live(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            policy_spec=spec,
        )
        score = float(summary.annual_return) - 0.5 * abs(float(summary.max_drawdown))
        if score > best_score:
            best_score = float(score)
            best_spec = spec
            best_summary = summary
    return V2CalibrationResult(
        best_policy=best_spec,
        best_score=float(best_score),
        baseline=baseline,
        calibrated=best_summary,
    )


def run_daily_v2_live(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
) -> DailyRunResult:
    settings = _load_v2_runtime_settings(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    snapshot = build_strategy_snapshot(
        strategy_id=strategy_id,
        universe_id=Path(str(settings["universe_file"])).stem or "v2_universe",
    )

    market_security, current_holdings, base_sector_map = load_watchlist(str(settings["watchlist"]))
    universe = build_candidate_universe(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    stocks = universe.rows or current_holdings
    sector_map = {
        stock.symbol: (stock.sector or base_sector_map.get(stock.symbol, "其他"))
        for stock in stocks
    }

    market_forecast, stock_rows = run_quant_pipeline(
        market_security=market_security,
        stock_securities=stocks,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        min_train_days=int(settings["min_train_days"]),
        step_days=int(settings["step_days"]),
        l2=float(settings["l2"]),
        max_positions=int(settings["max_positions"]),
        use_margin_features=bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
        margin_stock_file=str(settings["margin_stock_file"]),
        enable_walk_forward_eval=False,
    )

    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_state, cross_section = _build_market_and_cross_section_states(
        market_symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        use_margin_features=bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
        market_short_prob=float(market_forecast.short_prob),
        market_mid_prob=float(market_forecast.mid_prob),
    )
    sector_frames = build_sector_daily_frames(
        stock_securities=stocks,
        sector_map=sector_map,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    sector_records = run_sector_forecast(
        sector_frames=sector_frames,
        market_raw=market_raw,
        l2=float(settings["l2"]),
    )
    sectors = [
        SectorForecastState(
            sector=item.sector,
            up_5d_prob=float(item.up_5d_prob),
            up_20d_prob=float(item.up_20d_prob),
            relative_strength=float(item.excess_vs_market_prob - 0.5),
            rotation_speed=float(item.rotation_speed),
            crowding_score=float(item.crowding_score),
        )
        for item in sector_records
    ]
    sector_strength_map = {item.sector: float(item.excess_vs_market_prob - 0.5) for item in sector_records}
    stocks_state = _build_stock_states_from_rows(stock_rows, sector_map, sector_strength_map=sector_strength_map)
    composite_state = compose_state(
        market=market_state,
        sectors=sectors,
        stocks=stocks_state,
        cross_section=cross_section,
    )

    current_weights = {}
    if current_holdings:
        equal_weight = 1.0 / float(len(current_holdings))
        current_weights = {item.symbol: float(equal_weight) for item in current_holdings}

    policy_decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights=current_weights,
            current_cash=max(0.0, 1.0 - sum(current_weights.values())),
            total_equity=1.0,
        )
    )
    trade_actions = build_trade_actions(
        decision=policy_decision,
        current_weights=current_weights,
    )
    return DailyRunResult(
        snapshot=snapshot,
        composite_state=composite_state,
        policy_decision=policy_decision,
        trade_actions=trade_actions,
    )


def summarize_daily_run(result: DailyRunResult) -> dict[str, object]:
    return {
        "strategy_id": result.snapshot.strategy_id,
        "strategy_mode": result.composite_state.strategy_mode,
        "risk_regime": result.composite_state.risk_regime,
        "market": asdict(result.composite_state.market),
        "policy": asdict(result.policy_decision),
        "trade_plan": [
            {
                "symbol": action.symbol,
                "action": action.action,
                "current_weight": action.current_weight,
                "target_weight": action.target_weight,
                "delta_weight": action.delta_weight,
                "note": action.note,
            }
            for action in result.trade_actions
        ],
    }


def summarize_v2_backtest(result: V2BacktestSummary) -> dict[str, object]:
    return asdict(result)


def summarize_v2_calibration(result: V2CalibrationResult) -> dict[str, object]:
    return {
        "best_score": float(result.best_score),
        "best_policy": asdict(result.best_policy),
        "baseline": asdict(result.baseline),
        "calibrated": asdict(result.calibrated),
    }
