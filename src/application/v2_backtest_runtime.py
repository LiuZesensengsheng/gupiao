from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.application.v2_contracts import LearnedPolicyModel, PolicyInput, PolicySpec, StockForecastState, V2BacktestSummary


FrameRow = dict[str, object]


@dataclass(frozen=True)
class BacktestExecutionDependencies:
    safe_float: Callable[[object, float], float]
    clip: Callable[[float, float, float], float]
    status_tradeability_limit: Callable[[str], float]
    is_actionable_status: Callable[[str], bool]
    policy_spec_from_model: Callable[..., PolicySpec]
    apply_policy: Callable[..., Any]
    advance_holding_days: Callable[..., dict[str, int]]
    derive_learning_targets: Callable[..., tuple[float, float, float, float]]
    policy_feature_names: Callable[[], list[str]]
    policy_feature_vector: Callable[[Any], np.ndarray]
    to_v2_backtest_summary: Callable[..., V2BacktestSummary]


@dataclass(frozen=True)
class BacktestCoreDependencies:
    load_or_build_v2_backtest_trajectory: Callable[..., Any]
    empty_v2_backtest_result: Callable[[], tuple[V2BacktestSummary, list[dict[str, float]]]]
    execute_v2_backtest_trajectory: Callable[..., tuple[V2BacktestSummary, list[dict[str, float]]]]


@dataclass(frozen=True)
class BacktestFrameLookups:
    stock_rows_by_symbol_date: dict[str, dict[pd.Timestamp, FrameRow]]
    market_valid_by_date: dict[pd.Timestamp, FrameRow]


def _date_lookup_key(value: object) -> pd.Timestamp:
    return pd.Timestamp(value)


def _build_first_row_lookup(frame: pd.DataFrame | None) -> dict[pd.Timestamp, FrameRow]:
    if frame is None or frame.empty or "date" not in frame.columns:
        return {}
    records = frame.drop_duplicates(subset=["date"], keep="first").to_dict(orient="records")
    return {
        _date_lookup_key(record["date"]): record
        for record in records
        if record.get("date") is not None
    }


def _build_backtest_frame_lookups(
    *,
    stock_frames: dict[str, pd.DataFrame],
    market_valid: pd.DataFrame | None,
) -> BacktestFrameLookups:
    return BacktestFrameLookups(
        stock_rows_by_symbol_date={
            symbol: _build_first_row_lookup(frame)
            for symbol, frame in stock_frames.items()
        },
        market_valid_by_date=_build_first_row_lookup(market_valid),
    )


def _lookup_stock_row(
    *,
    symbol: str,
    date: pd.Timestamp,
    stock_frames: dict[str, pd.DataFrame],
    stock_rows_by_symbol_date: dict[str, dict[pd.Timestamp, FrameRow]] | None = None,
) -> FrameRow | None:
    lookup_key = _date_lookup_key(date)
    if stock_rows_by_symbol_date is not None:
        symbol_rows = stock_rows_by_symbol_date.get(symbol)
        if symbol_rows is not None:
            return symbol_rows.get(lookup_key)
    frame = stock_frames.get(symbol)
    if frame is None:
        return None
    row = frame[frame["date"] == date]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def to_v2_backtest_summary(
    *,
    returns: list[float],
    benchmark_returns: list[float] | None = None,
    turnovers: list[float],
    costs: list[float],
    gross_returns: list[float],
    fill_ratios: list[float],
    slippage_bps: list[float],
    rank_ics: list[float] | None = None,
    top_decile_returns: list[float] | None = None,
    top_bottom_spreads: list[float] | None = None,
    top_k_hit_rates: list[float] | None = None,
    horizon_metrics: dict[str, dict[str, list[float]]] | None = None,
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
            avg_rank_ic=0.0,
            avg_top_decile_return=0.0,
            avg_top_bottom_spread=0.0,
            avg_top_k_hit_rate=0.0,
            horizon_metrics={},
        )
    ret_arr = np.asarray(returns, dtype=float)
    nav = np.cumprod(1.0 + ret_arr)
    bench_arr = np.asarray(benchmark_returns if benchmark_returns is not None else np.zeros_like(ret_arr), dtype=float)
    if bench_arr.shape != ret_arr.shape:
        bench_arr = np.resize(bench_arr, ret_arr.shape)
    benchmark_nav = np.cumprod(1.0 + bench_arr)
    excess_ret_arr = (1.0 + ret_arr) / np.maximum(1.0 + bench_arr, 1e-9) - 1.0
    excess_nav = np.cumprod(1.0 + excess_ret_arr)
    gross_nav = np.cumprod(1.0 + np.asarray(gross_returns, dtype=float)) if gross_returns else nav
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.maximum(peak, 1e-12) - 1.0
    excess_peak = np.maximum.accumulate(excess_nav)
    total_return = float(nav[-1] - 1.0)
    benchmark_total_return = float(benchmark_nav[-1] - 1.0)
    excess_total_return = float(excess_nav[-1] - 1.0)
    gross_total_return = float(gross_nav[-1] - 1.0)
    n_days = len(returns)
    annual_return = float((1.0 + total_return) ** (252.0 / max(1, n_days)) - 1.0)
    benchmark_annual_return = float((1.0 + benchmark_total_return) ** (252.0 / max(1, n_days)) - 1.0)
    excess_annual_return = float((1.0 + excess_total_return) ** (252.0 / max(1, n_days)) - 1.0)
    annual_vol = float(np.std(ret_arr, ddof=0) * np.sqrt(252.0))
    win_rate = float(np.mean(ret_arr > 0.0))
    excess_drawdown = excess_nav / np.maximum(excess_peak, 1e-12) - 1.0
    excess_vol = float(np.std(excess_ret_arr, ddof=0))
    information_ratio = 0.0 if excess_vol <= 1e-12 else float(np.mean(excess_ret_arr) / excess_vol * np.sqrt(252.0))
    horizon_summary: dict[str, dict[str, float]] = {}
    if horizon_metrics:
        for horizon, metric_map in horizon_metrics.items():
            horizon_summary[horizon] = {
                key: float(np.mean(values)) if values else 0.0
                for key, values in metric_map.items()
            }
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
        avg_rank_ic=float(np.mean(rank_ics)) if rank_ics else 0.0,
        avg_top_decile_return=float(np.mean(top_decile_returns)) if top_decile_returns else 0.0,
        avg_top_bottom_spread=float(np.mean(top_bottom_spreads)) if top_bottom_spreads else 0.0,
        avg_top_k_hit_rate=float(np.mean(top_k_hit_rates)) if top_k_hit_rates else 0.0,
        horizon_metrics=horizon_summary,
        benchmark_total_return=float(benchmark_total_return),
        benchmark_annual_return=float(benchmark_annual_return),
        excess_total_return=float(excess_total_return),
        excess_annual_return=float(excess_annual_return),
        excess_max_drawdown=float(np.min(excess_drawdown)),
        information_ratio=float(information_ratio),
        nav_curve=[float(x) for x in nav.tolist()],
        benchmark_nav_curve=[float(x) for x in benchmark_nav.tolist()],
        excess_nav_curve=[float(x) for x in excess_nav.tolist()],
        curve_dates=[str(item.date()) for item in dates],
    )


def simulate_execution_day(
    *,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    decision: Any,
    current_weights: dict[str, float],
    current_cash: float,
    stock_states: list[StockForecastState],
    stock_frames: dict[str, pd.DataFrame],
    total_commission_rate: float,
    base_slippage_rate: float,
    stock_rows_by_symbol_date: dict[str, dict[pd.Timestamp, FrameRow]] | None = None,
    deps: BacktestExecutionDependencies,
) -> tuple[float, float, float, float, float, dict[str, float], float]:
    _ = next_date, current_cash
    state_map = {item.symbol: item for item in stock_states}
    day_rows: dict[str, FrameRow | None] = {}

    def _get_day_row(symbol: str) -> FrameRow | None:
        if symbol not in day_rows:
            day_rows[symbol] = _lookup_stock_row(
                symbol=symbol,
                date=date,
                stock_frames=stock_frames,
                stock_rows_by_symbol_date=stock_rows_by_symbol_date,
            )
        return day_rows[symbol]

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
        frame = stock_frames.get(symbol)
        day_row = _get_day_row(symbol)
        status = "normal"
        if state is not None:
            status = str(getattr(state, "tradability_status", "normal") or "normal")
        elif frame is None or day_row is None:
            status = "halted"
        if not deps.is_actionable_status(status):
            continue
        if status == "data_insufficient" and delta > 0.0:
            continue
        if day_row is not None:
            latest = day_row
            close_px = deps.safe_float(latest.get("close"), np.nan)
            open_px = deps.safe_float(latest.get("open"), np.nan)
            low_px = deps.safe_float(latest.get("low"), np.nan)
            high_px = deps.safe_float(latest.get("high"), np.nan)
            ret_1 = deps.safe_float(latest.get("ret_1"), np.nan)
            if close_px == close_px and ret_1 == ret_1:
                prev_close = close_px / max(1e-9, 1.0 + ret_1)
                limit_up_px = prev_close * 1.098
                limit_down_px = prev_close * 0.902
                if delta > 0.0 and low_px == low_px and low_px >= limit_up_px:
                    continue
                if delta < 0.0 and high_px == high_px and high_px <= limit_down_px:
                    continue
        tradeability = 0.45 if state is None else deps.clip(float(state.tradeability_score), 0.10, 1.0)
        tradeability = min(tradeability, deps.status_tradeability_limit(status))
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
        open_gap_penalty = 0.0
        intraday_range_penalty = 0.0
        if day_row is not None:
            latest = day_row
            close_px = deps.safe_float(latest.get("close"), np.nan)
            open_px = deps.safe_float(latest.get("open"), np.nan)
            low_px = deps.safe_float(latest.get("low"), np.nan)
            high_px = deps.safe_float(latest.get("high"), np.nan)
            ret_1 = deps.safe_float(latest.get("ret_1"), np.nan)
            if close_px == close_px and ret_1 == ret_1:
                prev_close = close_px / max(1e-9, 1.0 + ret_1)
                if open_px == open_px:
                    open_gap = float(open_px / max(prev_close, 1e-9) - 1.0)
                    open_gap_penalty = max(0.0, open_gap) if delta > 0.0 else max(0.0, -open_gap)
                if low_px == low_px and high_px == high_px:
                    intraday_range_penalty = max(0.0, float(high_px - low_px) / max(prev_close, 1e-9))
        slippage_rate = float(
            base_slippage_rate
            * (
                0.65
                + 0.7 * impact
                + 0.35 * (1.0 - tradeability)
                + 0.40 * open_gap_penalty
                + 0.10 * intraday_range_penalty
            )
        )
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
        state = state_map.get(symbol)
        status = "normal" if state is None else str(getattr(state, "tradability_status", "normal") or "normal")
        frame = stock_frames.get(symbol)
        if frame is None:
            realized_ret = -0.30 if status == "delisted" else 0.0
        else:
            row = _get_day_row(symbol)
            if row is None:
                realized_ret = -0.30 if status == "delisted" else 0.0
            else:
                realized_ret = float(row["fwd_ret_1"])
                if status == "delisted":
                    realized_ret = min(realized_ret, -0.20)
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
    return daily_return, float(used_turnover), float(total_cost), avg_fill_ratio, avg_slippage_bps, next_weights, next_cash


def execute_v2_backtest_trajectory(
    trajectory: Any,
    *,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    capture_learning_rows: bool = False,
    deps: BacktestExecutionDependencies,
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    _ = retrain_days
    commission_rate = max(0.0, float(commission_bps)) / 10000.0
    slippage_rate = max(0.0, float(slippage_bps)) / 10000.0
    returns: list[float] = []
    benchmark_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    fill_ratios: list[float] = []
    slippage_cost_bps: list[float] = []
    rank_ics: list[float] = []
    top_decile_returns: list[float] = []
    top_bottom_spreads: list[float] = []
    top_k_hit_rates: list[float] = []
    horizon_metric_series: dict[str, dict[str, list[float]]] = {}
    out_dates: list[pd.Timestamp] = []
    prev_weights: dict[str, float] = {}
    prev_holding_days: dict[str, int] = {}
    prev_cash = 1.0
    learning_rows: list[dict[str, float]] = []
    frame_lookups = _build_backtest_frame_lookups(
        stock_frames=trajectory.prepared.stock_frames,
        market_valid=trajectory.prepared.market_valid,
    )

    for step in trajectory.steps:
        step_stock_rows = frame_lookups.stock_rows_by_symbol_date
        step_date_key = _date_lookup_key(step.date)
        rank_ics.append(float(step.horizon_metrics["20d"]["rank_ic"]))
        top_decile_returns.append(float(step.horizon_metrics["20d"]["top_decile_return"]))
        top_bottom_spreads.append(float(step.horizon_metrics["20d"]["top_bottom_spread"]))
        top_k_hit_rates.append(float(step.horizon_metrics["20d"]["top_k_hit_rate"]))
        for horizon, metric_map in step.horizon_metrics.items():
            horizon_bucket = horizon_metric_series.setdefault(
                horizon,
                {"rank_ic": [], "top_decile_return": [], "top_bottom_spread": [], "top_k_hit_rate": []},
            )
            for name, value in metric_map.items():
                horizon_bucket.setdefault(name, []).append(float(value))

        active_policy_spec = policy_spec
        if learned_policy is not None:
            active_policy_spec = deps.policy_spec_from_model(
                state=step.composite_state,
                model=learned_policy,
            )
        decision = deps.apply_policy(
            PolicyInput(
                composite_state=step.composite_state,
                current_weights=prev_weights,
                current_cash=max(0.0, prev_cash),
                total_equity=1.0,
                current_holding_days=prev_holding_days,
            ),
            policy_spec=active_policy_spec,
        )
        gross_ret = 0.0
        for symbol, weight in decision.symbol_target_weights.items():
            row = step_stock_rows.get(symbol, {}).get(step_date_key)
            if row is None:
                continue
            gross_ret += float(weight) * deps.safe_float(row.get("fwd_ret_1"), 0.0)
        gross_ret = float(gross_ret)
        daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = simulate_execution_day(
            date=step.date,
            next_date=step.next_date,
            decision=decision,
            current_weights=prev_weights,
            current_cash=prev_cash,
            stock_states=step.stock_states,
            stock_frames=trajectory.prepared.stock_frames,
            total_commission_rate=commission_rate,
            base_slippage_rate=slippage_rate,
            stock_rows_by_symbol_date=step_stock_rows,
            deps=deps,
        )
        benchmark_row = frame_lookups.market_valid_by_date.get(step_date_key)
        benchmark_ret = 0.0
        if benchmark_row is not None:
            benchmark_ret = deps.safe_float(benchmark_row.get("mkt_fwd_ret_1", 0.0), 0.0)
        returns.append(float(daily_ret))
        benchmark_returns.append(float(benchmark_ret))
        gross_returns.append(float(gross_ret))
        turnovers.append(turnover)
        costs.append(cost)
        fill_ratios.append(fill_ratio)
        slippage_cost_bps.append(slip_bps)
        out_dates.append(step.next_date)
        old_weights = dict(prev_weights)
        prev_weights = next_weights
        prev_holding_days = deps.advance_holding_days(
            prev_holding_days=prev_holding_days,
            prev_weights=old_weights,
            next_weights=next_weights,
        )
        prev_cash = next_cash

        if capture_learning_rows:
            target_exposure, target_positions, target_turnover, sample_weight = deps.derive_learning_targets(
                state=step.composite_state,
                stock_frames=trajectory.prepared.stock_frames,
                date=step.date,
                horizon_metrics=step.horizon_metrics,
                universe_tier=trajectory.prepared.settings.get("universe_tier"),
            )
            row = {
                name: float(value)
                for name, value in zip(deps.policy_feature_names(), deps.policy_feature_vector(step.composite_state))
            }
            row.update(
                {
                    "target_exposure": float(target_exposure),
                    "target_positions": float(target_positions),
                    "target_turnover": float(target_turnover),
                    "sample_weight": float(sample_weight),
                }
            )
            learning_rows.append(row)

    return (
        deps.to_v2_backtest_summary(
            returns=returns,
            benchmark_returns=benchmark_returns,
            turnovers=turnovers,
            costs=costs,
            gross_returns=gross_returns,
            fill_ratios=fill_ratios,
            slippage_bps=slippage_cost_bps,
            rank_ics=rank_ics,
            top_decile_returns=top_decile_returns,
            top_bottom_spreads=top_bottom_spreads,
            top_k_hit_rates=top_k_hit_rates,
            horizon_metrics=horizon_metric_series,
            dates=out_dates,
        ),
        learning_rows,
    )


def run_v2_backtest_core(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    capture_learning_rows: bool = False,
    trajectory: Any | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    training_window_days: int | None = None,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    deps: BacktestCoreDependencies,
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    _ = strategy_id
    if trajectory is None:
        trajectory = deps.load_or_build_v2_backtest_trajectory(
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            dynamic_universe=dynamic_universe,
            generator_target_size=generator_target_size,
            generator_coarse_size=generator_coarse_size,
            generator_theme_aware=generator_theme_aware,
            generator_use_concepts=generator_use_concepts,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            training_window_days=training_window_days,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        if trajectory is None:
            return deps.empty_v2_backtest_result()
    return deps.execute_v2_backtest_trajectory(
        trajectory,
        policy_spec=policy_spec,
        learned_policy=learned_policy,
        retrain_days=retrain_days,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        capture_learning_rows=capture_learning_rows,
    )
