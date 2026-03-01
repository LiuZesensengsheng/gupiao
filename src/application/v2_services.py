from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
    DailyRunResult,
    LearnedPolicyModel,
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


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return out


def _policy_feature_names() -> list[str]:
    return [
        "mkt_up_1d",
        "mkt_up_20d",
        "mkt_drawdown_risk",
        "mkt_liquidity_stress",
        "cross_fund_flow",
        "cross_margin_risk_on",
        "cross_breadth",
        "cross_leader_participation",
        "cross_weak_ratio",
        "top_sector_up_20d",
        "top_sector_relative_strength",
        "top_stock_up_20d",
        "top_stock_tradeability",
        "top_stock_excess_vs_sector",
    ]


def _policy_feature_vector(state: CompositeState) -> np.ndarray:
    top_sector = state.sectors[0] if state.sectors else None
    top_stock = state.stocks[0] if state.stocks else None
    return np.asarray(
        [
            float(state.market.up_1d_prob),
            float(state.market.up_20d_prob),
            float(state.market.drawdown_risk),
            float(state.market.liquidity_stress),
            float(state.cross_section.fund_flow_strength),
            float(state.cross_section.margin_risk_on_score),
            float(state.cross_section.breadth_strength),
            float(state.cross_section.leader_participation),
            float(state.cross_section.weak_stock_ratio),
            0.0 if top_sector is None else float(top_sector.up_20d_prob),
            0.0 if top_sector is None else float(top_sector.relative_strength),
            0.0 if top_stock is None else float(top_stock.up_20d_prob),
            0.0 if top_stock is None else float(top_stock.tradeability_score),
            0.0 if top_stock is None else float(top_stock.excess_vs_sector_prob),
        ],
        dtype=float,
    )


def _fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
) -> tuple[float, np.ndarray]:
    if X.size == 0 or y.size == 0:
        return 0.0, np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    X_aug = np.hstack([ones, X])
    reg = np.eye(X_aug.shape[1], dtype=float) * float(max(0.0, l2))
    reg[0, 0] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    return float(coef[0]), np.asarray(coef[1:], dtype=float)


def _predict_ridge(features: np.ndarray, intercept: float, coef: np.ndarray) -> float:
    return float(intercept + np.dot(np.asarray(features, dtype=float), np.asarray(coef, dtype=float)))


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


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


def _policy_spec_from_model(
    *,
    state: CompositeState,
    model: LearnedPolicyModel,
) -> PolicySpec:
    features = _policy_feature_vector(state)
    exposure = _clip(
        _predict_ridge(features, model.exposure_intercept, np.asarray(model.exposure_coef, dtype=float)),
        0.20,
        0.95,
    )
    positions = int(
        round(
            _clip(
                _predict_ridge(features, model.position_intercept, np.asarray(model.position_coef, dtype=float)),
                1.0,
                6.0,
            )
        )
    )
    turnover_cap = _clip(
        _predict_ridge(features, model.turnover_intercept, np.asarray(model.turnover_coef, dtype=float)),
        0.10,
        0.45,
    )
    cautious_exposure = _clip(0.5 * (exposure + 0.35), 0.30, exposure)
    risk_off_exposure = _clip(0.5 * cautious_exposure, 0.20, 0.40)
    cautious_positions = min(positions, max(1, positions - 1))
    risk_off_positions = max(1, positions - 2)
    cautious_turnover = _clip(min(turnover_cap, 0.85 * turnover_cap), 0.10, turnover_cap)
    risk_off_turnover = _clip(min(cautious_turnover, 0.70 * turnover_cap), 0.08, cautious_turnover)
    return PolicySpec(
        risk_on_exposure=float(exposure),
        cautious_exposure=float(cautious_exposure),
        risk_off_exposure=float(risk_off_exposure),
        risk_on_positions=int(positions),
        cautious_positions=int(cautious_positions),
        risk_off_positions=int(risk_off_positions),
        risk_on_turnover_cap=float(turnover_cap),
        cautious_turnover_cap=float(cautious_turnover),
        risk_off_turnover_cap=float(risk_off_turnover),
    )


def _simulate_execution_day(
    *,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    decision: PolicyDecision,
    current_weights: dict[str, float],
    current_cash: float,
    stock_states: list[StockForecastState],
    stock_frames: dict[str, pd.DataFrame],
    total_commission_rate: float,
    base_slippage_rate: float,
) -> tuple[float, float, float, float, float, dict[str, float], float]:
    state_map = {item.symbol: item for item in stock_states}
    symbols = sorted(set(current_weights) | set(decision.symbol_target_weights))
    raw_deltas = {
        symbol: float(decision.symbol_target_weights.get(symbol, 0.0)) - float(current_weights.get(symbol, 0.0))
        for symbol in symbols
    }

    executed_deltas: dict[str, float] = {}
    fill_ratios: list[float] = []
    slippage_rates: list[float] = []
    slippage_amounts: list[float] = []
    total_turnover_budget = float(max(0.0, decision.turnover_cap))
    used_turnover = 0.0

    ordered_symbols = sorted(symbols, key=lambda sym: abs(raw_deltas.get(sym, 0.0)), reverse=True)
    for symbol in ordered_symbols:
        delta = float(raw_deltas.get(symbol, 0.0))
        if abs(delta) <= 1e-4:
            continue
        state = state_map.get(symbol)
        tradeability = 0.45 if state is None else _clip(float(state.tradeability_score), 0.10, 1.0)
        liquidity_cap = 0.03 + 0.12 * tradeability
        remaining_turnover = max(0.0, total_turnover_budget - used_turnover)
        max_abs_trade = min(abs(delta), liquidity_cap, remaining_turnover)
        if max_abs_trade <= 1e-6:
            continue
        fill_ratio = max_abs_trade / max(abs(delta), 1e-9)
        executed = float(np.sign(delta) * max_abs_trade)
        executed_deltas[symbol] = executed
        fill_ratios.append(float(fill_ratio))
        impact = max_abs_trade / max(liquidity_cap, 1e-6)
        slippage_rate = float(base_slippage_rate * (0.65 + 0.7 * impact + 0.35 * (1.0 - tradeability)))
        slippage_rates.append(slippage_rate)
        slippage_amounts.append(float(abs(executed) * slippage_rate))
        used_turnover += abs(executed)

    executed_weights = {symbol: max(0.0, float(weight)) for symbol, weight in current_weights.items()}
    for symbol, delta in executed_deltas.items():
        executed_weights[symbol] = max(0.0, float(executed_weights.get(symbol, 0.0)) + float(delta))
    executed_weights = {
        symbol: float(weight)
        for symbol, weight in executed_weights.items()
        if float(weight) > 1e-6
    }

    invested_after_trade = float(sum(executed_weights.values()))
    if invested_after_trade > 1.0:
        scale = 1.0 / invested_after_trade
        executed_weights = {symbol: float(weight) * scale for symbol, weight in executed_weights.items()}
        invested_after_trade = 1.0
    cash_after_trade = max(0.0, 1.0 - invested_after_trade)

    gross_end_value = float(cash_after_trade)
    position_values: dict[str, float] = {}
    for symbol, weight in executed_weights.items():
        frame = stock_frames.get(symbol)
        if frame is None:
            realized_ret = 0.0
        else:
            row = frame[frame["date"] == date]
            realized_ret = 0.0 if row.empty else float(row.iloc[0]["fwd_ret_1"])
        value = float(weight) * (1.0 + realized_ret)
        position_values[symbol] = value
        gross_end_value += value

    commission_cost = float(used_turnover * total_commission_rate)
    slippage_cost = float(sum(slippage_amounts)) if slippage_amounts else 0.0
    total_cost = float(commission_cost + slippage_cost)
    net_end_value = max(1e-9, gross_end_value - total_cost)

    next_weights = {
        symbol: float(value) / net_end_value
        for symbol, value in position_values.items()
        if float(value) > 1e-9
    }
    next_cash = max(0.0, float(cash_after_trade - total_cost) / net_end_value)
    daily_return = float(net_end_value - 1.0)
    avg_fill_ratio = float(np.mean(fill_ratios)) if fill_ratios else 0.0
    avg_slippage_bps = float(np.mean(slippage_rates) * 10000.0) if slippage_rates else 0.0
    return (
        daily_return,
        float(used_turnover),
        float(total_cost),
        avg_fill_ratio,
        avg_slippage_bps,
        next_weights,
        next_cash,
    )


def _to_v2_backtest_summary(
    *,
    returns: list[float],
    turnovers: list[float],
    costs: list[float],
    gross_returns: list[float],
    fill_ratios: list[float],
    slippage_bps: list[float],
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
    ret_arr = np.asarray(returns, dtype=float)
    nav = np.cumprod(1.0 + ret_arr)
    gross_nav = np.cumprod(1.0 + np.asarray(gross_returns, dtype=float)) if gross_returns else nav
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.maximum(peak, 1e-12) - 1.0
    total_return = float(nav[-1] - 1.0)
    gross_total_return = float(gross_nav[-1] - 1.0)
    n_days = len(returns)
    annual_return = float((1.0 + total_return) ** (252.0 / max(1, n_days)) - 1.0)
    annual_vol = float(np.std(ret_arr, ddof=0) * np.sqrt(252.0))
    win_rate = float(np.mean(ret_arr > 0.0))
    return V2BacktestSummary(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        n_days=int(n_days),
        total_return=float(total_return),
        annual_return=float(annual_return),
        max_drawdown=float(np.min(drawdown)),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        total_cost=float(np.sum(costs)) if costs else 0.0,
        gross_total_return=float(gross_total_return),
        annual_vol=float(annual_vol),
        win_rate=float(win_rate),
        trade_days=int(sum(1 for item in turnovers if float(item) > 1e-9)),
        avg_fill_ratio=float(np.mean(fill_ratios)) if fill_ratios else 0.0,
        avg_slippage_bps=float(np.mean(slippage_bps)) if slippage_bps else 0.0,
        nav_curve=[float(x) for x in nav.tolist()],
        curve_dates=[str(item.date()) for item in dates],
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


def _derive_learning_targets(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> tuple[float, float, float]:
    ranked = sorted(state.stocks, key=_stock_policy_score, reverse=True)
    realized: list[float] = []
    for stock in ranked[:4]:
        frame = stock_frames.get(stock.symbol)
        if frame is None:
            continue
        row = frame[frame["date"] == date]
        if row.empty:
            continue
        realized.append(float(row.iloc[0]["fwd_ret_1"]))
    lead_ret = float(np.mean(realized)) if realized else 0.0

    if lead_ret >= 0.008:
        exposure = 0.85
    elif lead_ret >= 0.0:
        exposure = 0.60
    else:
        exposure = 0.35

    breadth_bonus = 1 if float(state.cross_section.breadth_strength) > 0.05 else 0
    weakness_penalty = 1 if float(state.cross_section.weak_stock_ratio) > 0.55 else 0
    positions = int(np.clip(3 + breadth_bonus - weakness_penalty + (1 if lead_ret >= 0.012 else 0), 1, 5))

    turnover = 0.18
    if abs(lead_ret) >= 0.01:
        turnover = 0.32
    elif abs(lead_ret) >= 0.004:
        turnover = 0.25
    if float(state.market.drawdown_risk) > 0.45:
        turnover = min(turnover, 0.18)

    return float(exposure), float(positions), float(turnover)


def _run_v2_backtest_core(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    capture_learning_rows: bool = False,
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
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
        return _to_v2_backtest_summary(
            returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        ), []

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
        return _to_v2_backtest_summary(
            returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        ), []

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
        return _to_v2_backtest_summary(
            returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        ), []

    common_dates = set(pd.to_datetime(market_valid["date"]))
    for frame in stock_frames.values():
        common_dates &= set(pd.to_datetime(frame["date"]))
    dates = sorted(pd.Timestamp(d) for d in common_dates)
    min_train_days = int(settings["min_train_days"])
    if len(dates) <= min_train_days + 1:
        return _to_v2_backtest_summary(
            returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        ), []

    commission_rate = max(0.0, float(commission_bps)) / 10000.0
    slippage_rate = max(0.0, float(slippage_bps)) / 10000.0
    returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    fill_ratios: list[float] = []
    slippage_cost_bps: list[float] = []
    out_dates: list[pd.Timestamp] = []
    prev_weights: dict[str, float] = {}
    prev_cash = 1.0
    learning_rows: list[dict[str, float]] = []

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
            active_policy_spec = policy_spec
            if learned_policy is not None:
                active_policy_spec = _policy_spec_from_model(
                    state=composite_state,
                    model=learned_policy,
                )
            decision = apply_policy(
                PolicyInput(
                    composite_state=composite_state,
                    current_weights=prev_weights,
                    current_cash=max(0.0, prev_cash),
                    total_equity=1.0,
                ),
                policy_spec=active_policy_spec,
            )
            gross_ret = float(
                sum(
                    float(weight) * _safe_float(
                        stock_frames[symbol][stock_frames[symbol]["date"] == date].iloc[0]["fwd_ret_1"],
                        0.0,
                    )
                    for symbol, weight in decision.symbol_target_weights.items()
                    if symbol in stock_frames and not stock_frames[symbol][stock_frames[symbol]["date"] == date].empty
                )
            )
            daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
                date=date,
                next_date=next_date,
                decision=decision,
                current_weights=prev_weights,
                current_cash=prev_cash,
                stock_states=stock_states,
                stock_frames=stock_frames,
                total_commission_rate=commission_rate,
                base_slippage_rate=slippage_rate,
            )
            returns.append(float(daily_ret))
            gross_returns.append(float(gross_ret))
            turnovers.append(turnover)
            costs.append(cost)
            fill_ratios.append(fill_ratio)
            slippage_cost_bps.append(slip_bps)
            out_dates.append(next_date)
            prev_weights = next_weights
            prev_cash = next_cash

            if capture_learning_rows:
                target_exposure, target_positions, target_turnover = _derive_learning_targets(
                    state=composite_state,
                    stock_frames=stock_frames,
                    date=date,
                )
                row = {
                    name: float(value)
                    for name, value in zip(_policy_feature_names(), _policy_feature_vector(composite_state))
                }
                row.update(
                    {
                        "target_exposure": float(target_exposure),
                        "target_positions": float(target_positions),
                        "target_turnover": float(target_turnover),
                    }
                )
                learning_rows.append(row)

    return _to_v2_backtest_summary(
        returns=returns,
        turnovers=turnovers,
        costs=costs,
        gross_returns=gross_returns,
        fill_ratios=fill_ratios,
        slippage_bps=slippage_cost_bps,
        dates=out_dates,
    ), learning_rows


def run_v2_backtest_live(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
) -> V2BacktestSummary:
    summary, _ = _run_v2_backtest_core(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        policy_spec=policy_spec,
        learned_policy=learned_policy,
        retrain_days=retrain_days,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        capture_learning_rows=False,
    )
    return summary


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
    trials: list[dict[str, object]] = [
        {
            "policy": asdict(baseline_spec),
            "summary": asdict(baseline),
            "score": float(best_score),
        }
    ]
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
        trials.append(
            {
                "policy": asdict(spec),
                "summary": asdict(summary),
                "score": float(score),
            }
        )
        if score > best_score:
            best_score = float(score)
            best_spec = spec
            best_summary = summary
    return V2CalibrationResult(
        best_policy=best_spec,
        best_score=float(best_score),
        baseline=baseline,
        calibrated=best_summary,
        trials=trials,
    )


def learn_v2_policy_model(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    l2: float = 1.0,
    baseline: V2BacktestSummary | None = None,
) -> V2PolicyLearningResult:
    baseline = baseline or run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    _, rows = _run_v2_backtest_core(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        capture_learning_rows=True,
    )
    feature_names = _policy_feature_names()
    if not rows:
        model = LearnedPolicyModel(
            feature_names=feature_names,
            exposure_intercept=0.60,
            exposure_coef=[0.0] * len(feature_names),
            position_intercept=3.0,
            position_coef=[0.0] * len(feature_names),
            turnover_intercept=0.22,
            turnover_coef=[0.0] * len(feature_names),
            train_rows=0,
            train_r2_exposure=0.0,
            train_r2_positions=0.0,
            train_r2_turnover=0.0,
        )
        learned_summary = run_v2_backtest_live(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            learned_policy=model,
        )
        return V2PolicyLearningResult(model=model, baseline=baseline, learned=learned_summary)

    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y_exposure = np.asarray([float(row["target_exposure"]) for row in rows], dtype=float)
    y_positions = np.asarray([float(row["target_positions"]) for row in rows], dtype=float)
    y_turnover = np.asarray([float(row["target_turnover"]) for row in rows], dtype=float)

    exp_b, exp_w = _fit_ridge_regression(X, y_exposure, l2=l2)
    pos_b, pos_w = _fit_ridge_regression(X, y_positions, l2=l2)
    turn_b, turn_w = _fit_ridge_regression(X, y_turnover, l2=l2)

    pred_exp = np.asarray([_predict_ridge(row, exp_b, exp_w) for row in X], dtype=float)
    pred_pos = np.asarray([_predict_ridge(row, pos_b, pos_w) for row in X], dtype=float)
    pred_turn = np.asarray([_predict_ridge(row, turn_b, turn_w) for row in X], dtype=float)

    model = LearnedPolicyModel(
        feature_names=feature_names,
        exposure_intercept=float(exp_b),
        exposure_coef=[float(x) for x in exp_w.tolist()],
        position_intercept=float(pos_b),
        position_coef=[float(x) for x in pos_w.tolist()],
        turnover_intercept=float(turn_b),
        turnover_coef=[float(x) for x in turn_w.tolist()],
        train_rows=int(len(rows)),
        train_r2_exposure=float(_r2_score(y_exposure, pred_exp)),
        train_r2_positions=float(_r2_score(y_positions, pred_pos)),
        train_r2_turnover=float(_r2_score(y_turnover, pred_turn)),
    )
    learned_summary = run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        learned_policy=model,
    )
    return V2PolicyLearningResult(
        model=model,
        baseline=baseline,
        learned=learned_summary,
    )


def load_published_v2_policy_model(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
) -> LearnedPolicyModel | None:
    model_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    if not model_path.exists():
        return None
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return LearnedPolicyModel(
        feature_names=[str(x) for x in payload.get("feature_names", [])],
        exposure_intercept=float(payload.get("exposure_intercept", 0.60)),
        exposure_coef=[float(x) for x in payload.get("exposure_coef", [])],
        position_intercept=float(payload.get("position_intercept", 3.0)),
        position_coef=[float(x) for x in payload.get("position_coef", [])],
        turnover_intercept=float(payload.get("turnover_intercept", 0.22)),
        turnover_coef=[float(x) for x in payload.get("turnover_coef", [])],
        train_rows=int(payload.get("train_rows", 0)),
        train_r2_exposure=float(payload.get("train_r2_exposure", 0.0)),
        train_r2_positions=float(payload.get("train_r2_positions", 0.0)),
        train_r2_turnover=float(payload.get("train_r2_turnover", 0.0)),
    )


def publish_v2_research_artifacts(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    settings: dict[str, object] | None = None,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
) -> dict[str, str]:
    settings = settings or _load_v2_runtime_settings(
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
    )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(str(artifact_root)) / str(strategy_id) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    symbols = []
    universe_path = Path(str(settings.get("universe_file", "")))
    if universe_path.exists():
        try:
            raw = json.loads(universe_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                symbols = [str(item) for item in raw]
        except Exception:
            symbols = []

    dataset_manifest = {
        "strategy_id": str(strategy_id),
        "config_path": str(settings.get("config_path", "")),
        "source": str(settings.get("source", "")),
        "watchlist": str(settings.get("watchlist", "")),
        "universe_file": str(settings.get("universe_file", "")),
        "universe_limit": int(settings.get("universe_limit", 0)),
        "start": str(settings.get("start", "")),
        "end": str(settings.get("end", "")),
        "symbols": symbols,
        "symbol_count": len(symbols),
    }
    calibration_manifest = {
        "best_score": float(calibration.best_score),
        "best_policy": asdict(calibration.best_policy),
        "trials": calibration.trials,
    }
    learning_manifest = asdict(learning.model)
    backtest_manifest = {
        "baseline": asdict(baseline),
        "calibrated": asdict(calibration.calibrated),
        "learned": asdict(learning.learned),
    }

    dataset_path = base_dir / "dataset_manifest.json"
    calibration_path = base_dir / "policy_calibration.json"
    learning_path = base_dir / "learned_policy_model.json"
    backtest_path = base_dir / "backtest_summary.json"
    manifest_path = base_dir / "research_manifest.json"
    latest_policy_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    latest_manifest_path = Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json"
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    learning_path.write_text(json.dumps(learning_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    backtest_path.write_text(json.dumps(backtest_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "strategy_id": str(strategy_id),
        "dataset_manifest": str(dataset_path),
        "policy_calibration": str(calibration_path),
        "learned_policy_model": str(learning_path),
        "backtest_summary": str(backtest_path),
        "published_policy_model": str(latest_policy_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_policy_path.write_text(json.dumps(learning_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    latest_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_dir": str(base_dir),
        "dataset_manifest": str(dataset_path),
        "policy_calibration": str(calibration_path),
        "learned_policy_model": str(learning_path),
        "backtest_summary": str(backtest_path),
        "research_manifest": str(manifest_path),
        "published_policy_model": str(latest_policy_path),
    }


def run_daily_v2_live(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    artifact_root: str = "artifacts/v2",
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

    learned_policy = load_published_v2_policy_model(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
    )
    active_policy_spec = None
    if learned_policy is not None:
        active_policy_spec = _policy_spec_from_model(
            state=composite_state,
            model=learned_policy,
        )

    policy_decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights=current_weights,
            current_cash=max(0.0, 1.0 - sum(current_weights.values())),
            total_equity=1.0,
        ),
        policy_spec=active_policy_spec,
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
    payload = asdict(result)
    payload.pop("nav_curve", None)
    payload.pop("curve_dates", None)
    return payload


def summarize_v2_calibration(result: V2CalibrationResult) -> dict[str, object]:
    return {
        "best_score": float(result.best_score),
        "best_policy": asdict(result.best_policy),
        "baseline": summarize_v2_backtest(result.baseline),
        "calibrated": summarize_v2_backtest(result.calibrated),
        "trial_count": len(result.trials),
    }


def summarize_v2_policy_learning(result: V2PolicyLearningResult) -> dict[str, object]:
    return {
        "model": asdict(result.model),
        "baseline": summarize_v2_backtest(result.baseline),
        "learned": summarize_v2_backtest(result.learned),
    }
