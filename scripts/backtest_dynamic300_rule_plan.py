from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dynamic300_rule_utils import (
    RuleCandidateRow,
    candidate_bucket,
    coalesce,
    is_primary_candidate,
    load_config,
    portfolio_score,
    select_portfolio,
)
from src.application.v2_backtest_runtime import to_v2_backtest_summary
from src.application.v2_universe_generator import generate_dynamic_universe


@dataclass(frozen=True)
class RulePlanSnapshot:
    signal_date: str
    execution_date: str
    holdings: list[RuleCandidateRow]
    selected_count: int
    fresh_pool_pass_count: int


@dataclass(frozen=True)
class DailyBar:
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float
    pre_close: float
    prev_close: float
    ret_cc: float
    ma20: float
    ma60: float
    prev_ma20: float
    prev_ma60: float


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _load_settings(config_path: str) -> dict[str, Any]:
    payload = load_config(config_path)
    daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
    common = payload.get("common", {}) if isinstance(payload, dict) else {}
    return {
        "data_dir": str(coalesce(daily.get("data_dir"), common.get("data_dir"), default="data")),
        "cache_root": str(coalesce(daily.get("cache_root"), default="cache")),
        "base_universe_file": str(
            coalesce(
                daily.get("generated_universe_base_file"),
                daily.get("universe_file"),
                default="config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json",
            )
        ),
        "target_size": int(coalesce(daily.get("dynamic_universe_target_size"), default=300)),
        "coarse_size": int(coalesce(daily.get("dynamic_universe_coarse_size"), default=1000)),
        "min_history_days": int(coalesce(daily.get("dynamic_universe_min_history_days"), default=480)),
        "min_recent_amount": float(coalesce(daily.get("dynamic_universe_min_recent_amount"), default=100000.0)),
        "theme_cap_ratio": float(coalesce(daily.get("dynamic_universe_theme_cap_ratio"), default=0.16)),
        "theme_floor_count": int(coalesce(daily.get("dynamic_universe_theme_floor_count"), default=2)),
        "turnover_quality_weight": float(
            coalesce(daily.get("dynamic_universe_turnover_quality_weight"), default=0.25)
        ),
        "theme_weight": float(coalesce(daily.get("dynamic_universe_theme_weight"), default=0.18)),
        "main_board_only": bool(coalesce(daily.get("main_board_only_universe"), default=True)),
        "use_concepts": bool(coalesce(daily.get("generator_use_concepts"), default=True)),
    }


def _build_candidate_row(raw: dict[str, object], theme_info: dict[str, float]) -> RuleCandidateRow:
    bucket = candidate_bucket(raw)
    score = portfolio_score(
        {**raw, "bucket": bucket},
        theme_count=int(theme_info["selected_count"]),
        theme_strength=float(theme_info["theme_strength"]),
    )
    if not is_primary_candidate({**raw, "bucket": bucket}, int(theme_info["selected_count"])):
        score *= 0.92
    return RuleCandidateRow(
        symbol=str(raw["symbol"]),
        name=str(raw["name"]),
        theme=str(raw.get("sector", "其他")),
        refined_score=float(raw.get("refined_score", 0.0)),
        fresh_pool_score=float(raw.get("fresh_pool_score", 0.0)),
        fresh_pool_pass=bool(raw.get("fresh_pool_pass", False)),
        recent_high_gap20=float(raw.get("recent_high_gap20", 0.0)),
        amount_ratio20=float(raw.get("amount_ratio20", 0.0)),
        theme_selected_count=int(theme_info["selected_count"]),
        theme_strength=float(theme_info["theme_strength"]),
        close=float(raw.get("close", 0.0)),
        ma20=float(raw.get("ma20", 0.0)),
        ma60=float(raw.get("ma60", 0.0)),
        ret20=float(raw.get("ret20", 0.0)),
        ret60=float(raw.get("ret60", 0.0)),
        breakout_pos_120=float(raw.get("breakout_pos_120", 0.0)),
        volatility20=float(raw.get("volatility20", 0.0)),
        tradeability=float(raw.get("tradeability", 0.0)),
        bucket=bucket,
        portfolio_score=float(score),
    )


def _build_snapshot(
    *,
    settings: dict[str, Any],
    signal_date: str,
    execution_date: str,
    top_n: int,
    max_per_theme: int,
) -> RulePlanSnapshot:
    result = generate_dynamic_universe(
        universe_file=settings["base_universe_file"],
        data_dir=settings["data_dir"],
        cache_root=settings["cache_root"],
        target_size=settings["target_size"],
        coarse_size=settings["coarse_size"],
        theme_aware=True,
        use_concepts=settings["use_concepts"],
        end_date=signal_date,
        min_history_days=settings["min_history_days"],
        min_recent_amount=settings["min_recent_amount"],
        theme_cap_ratio=settings["theme_cap_ratio"],
        theme_floor_count=settings["theme_floor_count"],
        turnover_quality_weight=settings["turnover_quality_weight"],
        theme_weight=settings["theme_weight"],
        main_board_only=settings["main_board_only"],
        refresh_cache=False,
    )
    theme_map = {
        str(item.theme): {
            "selected_count": int(item.selected_count),
            "theme_strength": float(item.theme_strength),
        }
        for item in result.theme_allocations
    }
    rows = [
        _build_candidate_row(
            dict(raw),
            theme_map.get(str(raw.get("sector", "其他")), {"selected_count": 0, "theme_strength": 0.0}),
        )
        for raw in result.selected_300
    ]
    ranked = sorted(
        rows,
        key=lambda item: (
            -item.portfolio_score,
            -item.refined_score,
            -item.fresh_pool_score,
            -item.theme_selected_count,
            item.symbol,
        ),
    )
    holdings = select_portfolio(ranked, top_n=top_n, max_per_theme=max_per_theme)
    fresh_pool_pass_count = int(result.generator_manifest.config.get("fresh_pool_pass_count", 0))
    return RulePlanSnapshot(
        signal_date=signal_date,
        execution_date=execution_date,
        holdings=holdings,
        selected_count=len(result.selected_300),
        fresh_pool_pass_count=fresh_pool_pass_count,
    )


@lru_cache(maxsize=None)
def _load_bar_frame(data_dir: str, symbol: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}.csv"
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for column in numeric_cols:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    frame["pre_close"] = frame["close"].shift(1)
    frame["prev_close"] = frame["close"].shift(1)
    frame["ret_cc"] = frame["close"].pct_change().fillna(0.0)
    frame["ma20"] = frame["close"].rolling(20).mean()
    frame["ma60"] = frame["close"].rolling(60).mean()
    frame["prev_ma20"] = frame["ma20"].shift(1)
    frame["prev_ma60"] = frame["ma60"].shift(1)
    return frame


@lru_cache(maxsize=None)
def _bar_lookup(data_dir: str, symbol: str) -> dict[pd.Timestamp, DailyBar]:
    frame = _load_bar_frame(data_dir, symbol)
    lookup: dict[pd.Timestamp, DailyBar] = {}
    for row in frame.itertuples(index=False):
        timestamp = pd.Timestamp(row.date)
        lookup[timestamp] = DailyBar(
            date=timestamp,
            open=float(row.open) if pd.notna(row.open) else 0.0,
            high=float(row.high) if pd.notna(row.high) else 0.0,
            low=float(row.low) if pd.notna(row.low) else 0.0,
            close=float(row.close) if pd.notna(row.close) else 0.0,
            volume=float(row.volume) if pd.notna(row.volume) else 0.0,
            amount=float(row.amount) if pd.notna(row.amount) else 0.0,
            pre_close=float(row.pre_close) if pd.notna(row.pre_close) else 0.0,
            prev_close=float(row.prev_close) if pd.notna(row.prev_close) else 0.0,
            ret_cc=float(row.ret_cc) if pd.notna(row.ret_cc) else 0.0,
            ma20=float(row.ma20) if pd.notna(row.ma20) else 0.0,
            ma60=float(row.ma60) if pd.notna(row.ma60) else 0.0,
            prev_ma20=float(row.prev_ma20) if pd.notna(row.prev_ma20) else 0.0,
            prev_ma60=float(row.prev_ma60) if pd.notna(row.prev_ma60) else 0.0,
        )
    return lookup


def _bar_for_date(data_dir: str, symbol: str, date: pd.Timestamp) -> DailyBar | None:
    return _bar_lookup(data_dir, symbol).get(pd.Timestamp(date))


def _has_price(data_dir: str, symbol: str, date: pd.Timestamp) -> bool:
    return _bar_for_date(data_dir, symbol, date) is not None


def _normalize_target_weights(
    holdings: list[RuleCandidateRow],
    *,
    data_dir: str,
    date: pd.Timestamp,
) -> dict[str, float]:
    available = [row for row in holdings if _has_price(data_dir, row.symbol, date)]
    if not available:
        return {}
    total = sum(max(float(row.weight), 0.0) for row in available)
    if total <= 1e-9:
        equal_weight = 1.0 / max(1, len(available))
        return {row.symbol: equal_weight for row in available}
    return {row.symbol: float(row.weight) / float(total) for row in available}


def _limit_ratio(symbol: str) -> float:
    _ = symbol
    return 0.10


def _limit_up_price(pre_close: float, symbol: str) -> float:
    return float(round(float(pre_close) * (1.0 + _limit_ratio(symbol)) + 1e-8, 2))


def _limit_down_price(pre_close: float, symbol: str) -> float:
    return float(round(float(pre_close) * (1.0 - _limit_ratio(symbol)) + 1e-8, 2))


def _is_buy_locked(symbol: str, bar: DailyBar | None) -> bool:
    if bar is None or bar.pre_close <= 0.0:
        return False
    limit_up = _limit_up_price(bar.pre_close, symbol)
    return bool(bar.open >= limit_up * 0.997 and bar.low >= limit_up * 0.995)


def _is_sell_locked(symbol: str, bar: DailyBar | None) -> bool:
    if bar is None or bar.pre_close <= 0.0:
        return False
    limit_down = _limit_down_price(bar.pre_close, symbol)
    return bool(bar.open <= limit_down * 1.003 and bar.high <= limit_down * 1.006)


def _estimate_buy_fill_ratio(row: RuleCandidateRow, bar: DailyBar | None) -> float:
    from scripts.dynamic300_rule_utils import avoid_upper_bound, buy_zone_bounds

    if bar is None or bar.pre_close <= 0.0:
        return 0.0
    if bar.open > avoid_upper_bound(row):
        return 0.0
    if _is_buy_locked(row.symbol, bar):
        return 0.0
    lower, upper = buy_zone_bounds(row)
    if bar.high < lower or bar.low > upper:
        return 0.0
    liquidity = 0.55 + 0.45 * _clip(row.tradeability, 0.0, 1.0)
    if lower <= bar.open <= upper:
        base = 1.0
    elif bar.open < lower <= bar.high:
        base = 0.82
    elif bar.open > upper and bar.low <= upper:
        base = 0.58
    else:
        base = 0.72
    if row.ma20 > 0.0 and bar.low < row.ma20 * 0.995:
        base *= 0.85
    return _clip(base * liquidity, 0.0, 1.0)


def _estimate_sell_fill_ratio(*, symbol: str, bar: DailyBar | None, tradeability: float, aggressive: bool) -> float:
    if bar is None:
        return 0.0
    if _is_sell_locked(symbol, bar):
        return 0.0
    liquidity = 0.60 + 0.40 * _clip(tradeability, 0.0, 1.0)
    spread = (bar.high - bar.low) / max(bar.close, 1e-9)
    base = 0.88 if aggressive else 0.78
    if spread < 0.012:
        base -= 0.10
    if bar.close < bar.open:
        base += 0.06
    return _clip(base * liquidity, 0.0, 1.0)


def _exit_target_weight(
    current_weight: float,
    bar: DailyBar | None,
    *,
    allow_soft_reduce: bool = True,
    allow_intraday_trim: bool = True,
) -> float:
    if bar is None:
        return float(current_weight)
    if bar.ma60 > 0.0:
        cross_below_ma60 = bar.prev_close >= bar.prev_ma60 > 0.0 and bar.close < bar.ma60
        if cross_below_ma60 or bar.close < bar.ma60 * 0.985:
            return 0.0
    target = float(current_weight)
    if allow_soft_reduce and bar.ma20 > 0.0:
        cross_below_ma20 = bar.prev_close >= bar.prev_ma20 > 0.0 and bar.close < bar.ma20
        if cross_below_ma20:
            target = min(target, float(current_weight) * 0.50)
    if allow_intraday_trim:
        upper_shadow = (bar.high - max(bar.open, bar.close)) / max(bar.close, 1e-9)
        weak_close = bar.close <= bar.open * 0.99
        if upper_shadow >= 0.04 and weak_close:
            target = min(target, float(current_weight) * 0.70)
    return max(0.0, float(target))


def _execute_weight_transition(
    *,
    data_dir: str,
    date: pd.Timestamp,
    pre_trade_weights: dict[str, float],
    pre_trade_cash: float,
    desired_weights: dict[str, float],
    current_meta: dict[str, RuleCandidateRow],
    planned_rows_by_symbol: dict[str, RuleCandidateRow],
) -> tuple[dict[str, float], float, float, float]:
    requested_turnover = float(
        sum(
            abs(float(desired_weights.get(symbol, 0.0)) - float(pre_trade_weights.get(symbol, 0.0)))
            for symbol in sorted(set(pre_trade_weights) | set(desired_weights))
        )
    )
    if requested_turnover <= 1e-9:
        return dict(pre_trade_weights), float(pre_trade_cash), 0.0, 0.0

    working_weights = dict(pre_trade_weights)
    cash = float(pre_trade_cash)
    executed_turnover = 0.0

    for symbol in sorted(set(pre_trade_weights) | set(desired_weights)):
        current_weight = float(working_weights.get(symbol, 0.0))
        target_weight = float(desired_weights.get(symbol, 0.0))
        if target_weight >= current_weight - 1e-9:
            continue
        row = current_meta.get(symbol) or planned_rows_by_symbol.get(symbol)
        bar = _bar_for_date(data_dir, symbol, date)
        tradeability = float(row.tradeability) if row is not None else 0.55
        aggressive = target_weight <= 1e-9
        fill_ratio = _estimate_sell_fill_ratio(
            symbol=symbol,
            bar=bar,
            tradeability=tradeability,
            aggressive=aggressive,
        )
        new_weight = current_weight + (target_weight - current_weight) * fill_ratio
        executed = max(0.0, current_weight - new_weight)
        if executed > 1e-9:
            cash += executed
            executed_turnover += executed
        if new_weight > 1e-9:
            working_weights[symbol] = float(new_weight)
        else:
            working_weights.pop(symbol, None)

    buy_orders: list[tuple[str, float, float]] = []
    for symbol, target_weight in desired_weights.items():
        current_weight = float(working_weights.get(symbol, 0.0))
        target_weight = float(target_weight)
        if target_weight <= current_weight + 1e-9:
            continue
        row = planned_rows_by_symbol.get(symbol) or current_meta.get(symbol)
        if row is None:
            continue
        bar = _bar_for_date(data_dir, symbol, date)
        fill_ratio = _estimate_buy_fill_ratio(row, bar)
        request = max(0.0, target_weight - current_weight)
        buy_orders.append((symbol, request, fill_ratio))

    raw_buy = sum(request * fill_ratio for _, request, fill_ratio in buy_orders)
    buy_scale = min(1.0, cash / max(raw_buy, 1e-12)) if raw_buy > 1e-12 else 0.0
    for symbol, request, fill_ratio in buy_orders:
        executed = request * fill_ratio * buy_scale
        if executed <= 1e-9:
            continue
        working_weights[symbol] = float(working_weights.get(symbol, 0.0) + executed)
        cash -= executed
        executed_turnover += executed

    total_weight = sum(max(weight, 0.0) for weight in working_weights.values())
    cash = max(0.0, 1.0 - total_weight)
    fill_ratio = executed_turnover / max(requested_turnover, 1e-9)
    return (
        {symbol: float(weight) for symbol, weight in working_weights.items() if float(weight) > 1e-9},
        float(cash),
        float(executed_turnover),
        float(_clip(fill_ratio, 0.0, 1.0)),
    )


def _apply_transaction_cost(
    weights: dict[str, float],
    cash: float,
    cost: float,
) -> tuple[dict[str, float], float]:
    weight_values = {symbol: max(0.0, float(value)) for symbol, value in weights.items() if float(value) > 1e-9}
    cash_value = max(0.0, float(cash))
    remaining_cost = max(0.0, float(cost))

    if remaining_cost <= cash_value:
        cash_value -= remaining_cost
    else:
        remaining_cost -= cash_value
        cash_value = 0.0
        total_weight = sum(weight_values.values())
        if total_weight > 1e-12:
            scale = max(0.0, (total_weight - remaining_cost) / total_weight)
            weight_values = {symbol: value * scale for symbol, value in weight_values.items() if value * scale > 1e-9}

    nav_after_cost = cash_value + sum(weight_values.values())
    if nav_after_cost <= 1e-12:
        return {}, 1.0
    normalized_weights = {symbol: value / nav_after_cost for symbol, value in weight_values.items() if value > 1e-9}
    normalized_cash = max(0.0, cash_value / nav_after_cost)
    return normalized_weights, normalized_cash


def _build_signal_schedule(
    *,
    benchmark_frame: pd.DataFrame,
    end_date: str,
    lookback_trading_days: int,
    rebalance_interval: int,
) -> list[tuple[str, str]]:
    trade_dates = benchmark_frame.loc[
        benchmark_frame["date"] <= pd.Timestamp(end_date),
        "date",
    ].sort_values().reset_index(drop=True)
    if len(trade_dates) < rebalance_interval + 3:
        return []
    start_idx = max(0, len(trade_dates) - int(lookback_trading_days) - 1)
    signal_dates = list(trade_dates.iloc[start_idx:-1:int(rebalance_interval)])
    schedule: list[tuple[str, str]] = []
    for signal_date in signal_dates:
        signal_idx = int(trade_dates[trade_dates == signal_date].index[0])
        if signal_idx + 1 >= len(trade_dates):
            continue
        execution_date = pd.Timestamp(trade_dates.iloc[signal_idx + 1]).strftime("%Y-%m-%d")
        schedule.append((pd.Timestamp(signal_date).strftime("%Y-%m-%d"), execution_date))
    return schedule


def _build_snapshots(
    *,
    settings: dict[str, Any],
    schedule: list[tuple[str, str]],
    top_n: int,
    max_per_theme: int,
    workers: int,
) -> list[RulePlanSnapshot]:
    if not schedule:
        return []
    if int(workers) <= 1:
        return [
            _build_snapshot(
                settings=settings,
                signal_date=signal_date,
                execution_date=execution_date,
                top_n=top_n,
                max_per_theme=max_per_theme,
            )
            for signal_date, execution_date in schedule
        ]

    ordered: dict[str, RulePlanSnapshot] = {}
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        future_map = {
            executor.submit(
                _build_snapshot,
                settings=settings,
                signal_date=signal_date,
                execution_date=execution_date,
                top_n=top_n,
                max_per_theme=max_per_theme,
            ): signal_date
            for signal_date, execution_date in schedule
        }
        for future in as_completed(future_map):
            snapshot = future.result()
            ordered[snapshot.signal_date] = snapshot
    return [ordered[signal_date] for signal_date, _ in schedule if signal_date in ordered]


def _simulate_fast_backtest(
    *,
    settings: dict[str, Any],
    benchmark_symbol: str,
    end_date: str,
    snapshots: list[RulePlanSnapshot],
    commission_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    benchmark_frame = _load_bar_frame(settings["data_dir"], benchmark_symbol)
    benchmark_frame = benchmark_frame.loc[benchmark_frame["date"] <= pd.Timestamp(end_date)].copy()
    if benchmark_frame.empty or not snapshots:
        summary = to_v2_backtest_summary(
            returns=[],
            benchmark_returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        )
        return {"summary": asdict(summary), "snapshots": [asdict(item) for item in snapshots]}

    trade_dates = benchmark_frame["date"].sort_values().reset_index(drop=True)
    first_execution = pd.Timestamp(snapshots[0].execution_date)
    trade_dates = trade_dates.loc[trade_dates >= first_execution].reset_index(drop=True)
    if trade_dates.empty:
        summary = to_v2_backtest_summary(
            returns=[],
            benchmark_returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        )
        return {"summary": asdict(summary), "snapshots": [asdict(item) for item in snapshots]}

    snapshot_by_execution = {pd.Timestamp(item.execution_date): item for item in snapshots}
    benchmark_lookup = _bar_lookup(settings["data_dir"], benchmark_symbol)
    cost_rate = (max(0.0, float(commission_bps)) + max(0.0, float(slippage_bps))) / 10000.0

    returns: list[float] = []
    benchmark_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    fill_ratios: list[float] = []
    slippage_series: list[float] = []
    dates: list[pd.Timestamp] = []

    current_weights: dict[str, float] = {}
    current_cash = 1.0
    current_meta: dict[str, RuleCandidateRow] = {}
    first_entry_done = False

    for date in trade_dates:
        date = pd.Timestamp(date)
        benchmark_bar = benchmark_lookup.get(date)
        benchmark_ret = float(benchmark_bar.ret_cc) if benchmark_bar is not None else 0.0

        gross_ret = 0.0
        position_values: dict[str, float] = {}
        for symbol, weight in current_weights.items():
            bar = _bar_for_date(settings["data_dir"], symbol, date)
            ret = float(bar.ret_cc) if bar is not None else 0.0
            gross_ret += float(weight) * ret
            position_values[symbol] = float(weight) * (1.0 + ret)

        nav_mid = current_cash + sum(position_values.values())
        nav_mid = max(nav_mid, 1e-12)
        pre_trade_weights = {
            symbol: float(value) / nav_mid
            for symbol, value in position_values.items()
            if float(value) > 1e-12
        }
        pre_trade_cash = max(0.0, float(current_cash) / nav_mid)

        desired_weights = dict(pre_trade_weights)
        planned_rows_by_symbol: dict[str, RuleCandidateRow] = {}

        for symbol, weight in list(pre_trade_weights.items()):
            bar = _bar_for_date(settings["data_dir"], symbol, date)
            overlay_weight = _exit_target_weight(
                weight,
                bar,
                allow_soft_reduce=False,
                allow_intraday_trim=False,
            )
            if overlay_weight < weight - 1e-9:
                desired_weights[symbol] = overlay_weight

        if date in snapshot_by_execution:
            snapshot = snapshot_by_execution[date]
            snapshot_targets = _normalize_target_weights(
                snapshot.holdings,
                data_dir=settings["data_dir"],
                date=date,
            )
            desired_weights = dict(snapshot_targets)
            planned_rows_by_symbol = {
                row.symbol: row for row in snapshot.holdings if row.symbol in snapshot_targets
            }
            for symbol, weight in pre_trade_weights.items():
                bar = _bar_for_date(settings["data_dir"], symbol, date)
                overlay_weight = _exit_target_weight(
                    weight,
                    bar,
                    allow_soft_reduce=True,
                    allow_intraday_trim=True,
                )
                if overlay_weight < weight - 1e-9:
                    desired_weights[symbol] = min(float(desired_weights.get(symbol, 0.0)), overlay_weight)

        trade_requested = any(
            abs(float(desired_weights.get(symbol, 0.0)) - float(pre_trade_weights.get(symbol, 0.0))) > 1e-9
            for symbol in set(pre_trade_weights) | set(desired_weights)
        )

        turnover = 0.0
        cost = 0.0
        fill_ratio = 0.0
        if trade_requested:
            post_trade_weights, post_trade_cash, turnover, fill_ratio = _execute_weight_transition(
                data_dir=settings["data_dir"],
                date=date,
                pre_trade_weights=pre_trade_weights,
                pre_trade_cash=pre_trade_cash,
                desired_weights=desired_weights,
                current_meta=current_meta,
                planned_rows_by_symbol=planned_rows_by_symbol,
            )
            cost = float(turnover * cost_rate)
            current_weights, current_cash = _apply_transaction_cost(post_trade_weights, post_trade_cash, cost)
            next_meta: dict[str, RuleCandidateRow] = {}
            for symbol in current_weights:
                row = planned_rows_by_symbol.get(symbol) or current_meta.get(symbol)
                if row is not None:
                    next_meta[symbol] = row
            current_meta = next_meta
            if not first_entry_done and date in snapshot_by_execution and turnover > 1e-9:
                benchmark_ret = 0.0
                first_entry_done = True
        else:
            current_weights = pre_trade_weights
            current_cash = pre_trade_cash

        nav_after = nav_mid * (1.0 - cost)
        daily_return = float(nav_after - 1.0)
        returns.append(daily_return)
        benchmark_returns.append(float(benchmark_ret))
        gross_returns.append(float(gross_ret))
        turnovers.append(float(turnover))
        costs.append(float(cost))
        fill_ratios.append(float(fill_ratio))
        slippage_series.append(float(slippage_bps) if turnover > 1e-9 else 0.0)
        dates.append(date)

    summary = to_v2_backtest_summary(
        returns=returns,
        benchmark_returns=benchmark_returns,
        turnovers=turnovers,
        costs=costs,
        gross_returns=gross_returns,
        fill_ratios=fill_ratios,
        slippage_bps=slippage_series,
        dates=dates,
    )
    return {"summary": asdict(summary), "snapshots": [asdict(item) for item in snapshots]}


def _markdown_report(
    *,
    end_date: str,
    benchmark_symbol: str,
    lookback_trading_days: int,
    rebalance_interval: int,
    top_n: int,
    max_per_theme: int,
    workers: int,
    result: dict[str, Any],
) -> str:
    summary = dict(result.get("summary", {}))
    snapshots = list(result.get("snapshots", []))
    lines = [
        "# Dynamic 300 快速规则回测",
        "",
        f"- 截止日期: {end_date}",
        f"- 基准: {benchmark_symbol}",
        f"- 回看交易日: {lookback_trading_days}",
        f"- 调仓步长: {rebalance_interval} 个交易日",
        f"- 组合持仓数: {top_n}",
        f"- 同主题上限: {max_per_theme}",
        f"- 并发快照数: {workers}",
        "",
        "## 结果",
        "",
        f"- 区间: {summary.get('start_date', '')} -> {summary.get('end_date', '')}",
        f"- 总收益: {float(summary.get('total_return', 0.0)):.2%}",
        f"- 年化收益: {float(summary.get('annual_return', 0.0)):.2%}",
        f"- 基准总收益: {float(summary.get('benchmark_total_return', 0.0)):.2%}",
        f"- 基准年化: {float(summary.get('benchmark_annual_return', 0.0)):.2%}",
        f"- 超额总收益: {float(summary.get('excess_total_return', 0.0)):.2%}",
        f"- 年化超额: {float(summary.get('excess_annual_return', 0.0)):.2%}",
        f"- 最大回撤: {float(summary.get('max_drawdown', 0.0)):.2%}",
        f"- 信息比率: {float(summary.get('information_ratio', 0.0)):.3f}",
        f"- 平均换手: {float(summary.get('avg_turnover', 0.0)):.2%}",
        f"- 总成本: {float(summary.get('total_cost', 0.0)):.2%}",
        f"- 平均成交率: {float(summary.get('avg_fill_ratio', 0.0)):.2%}",
        "",
        "## 调仓快照",
        "",
    ]
    for item in snapshots:
        holdings = item.get("holdings", [])
        names = ", ".join(f"{row['name']}({row['symbol']},{row.get('bucket', 'na')})" for row in holdings)
        lines.append(
            f"- {item.get('signal_date')} -> {item.get('execution_date')}: {names or '无有效持仓'}"
        )
    return "\n".join(lines)


def run_fast_backtest(
    *,
    config_path: str,
    end_date: str,
    lookback_trading_days: int,
    rebalance_interval: int,
    top_n: int,
    max_per_theme: int,
    benchmark_symbol: str,
    commission_bps: float,
    slippage_bps: float,
    workers: int,
) -> dict[str, str]:
    settings = _load_settings(config_path)
    benchmark_frame = _load_bar_frame(settings["data_dir"], benchmark_symbol)
    schedule = _build_signal_schedule(
        benchmark_frame=benchmark_frame,
        end_date=end_date,
        lookback_trading_days=lookback_trading_days,
        rebalance_interval=rebalance_interval,
    )
    snapshots = _build_snapshots(
        settings=settings,
        schedule=schedule,
        top_n=top_n,
        max_per_theme=max_per_theme,
        workers=workers,
    )
    result = _simulate_fast_backtest(
        settings=settings,
        benchmark_symbol=benchmark_symbol,
        end_date=end_date,
        snapshots=snapshots,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    result["params"] = {
        "config_path": config_path,
        "end_date": end_date,
        "lookback_trading_days": int(lookback_trading_days),
        "rebalance_interval": int(rebalance_interval),
        "top_n": int(top_n),
        "max_per_theme": int(max_per_theme),
        "benchmark_symbol": benchmark_symbol,
        "commission_bps": float(commission_bps),
        "slippage_bps": float(slippage_bps),
        "workers": int(workers),
    }

    date_token = str(end_date).replace("-", "")
    report_root = Path("reports")
    report_root.mkdir(parents=True, exist_ok=True)
    json_path = report_root / f"dynamic300_rule_fast_backtest_{date_token}.json"
    md_path = report_root / f"dynamic300_rule_fast_backtest_{date_token}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        _markdown_report(
            end_date=end_date,
            benchmark_symbol=benchmark_symbol,
            lookback_trading_days=lookback_trading_days,
            rebalance_interval=rebalance_interval,
            top_n=top_n,
            max_per_theme=max_per_theme,
            workers=workers,
            result=result,
        ),
        encoding="utf-8",
    )
    return {
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fast rule-based backtest on the dynamic 300 portfolio.")
    parser.add_argument("--config", default="config/api.json")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--lookback-trading-days", type=int, default=120)
    parser.add_argument("--rebalance-interval", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--max-per-theme", type=int, default=1)
    parser.add_argument("--benchmark-symbol", default="000300.SH")
    parser.add_argument("--commission-bps", type=float, default=1.5)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    outputs = run_fast_backtest(
        config_path=str(args.config),
        end_date=str(args.end_date),
        lookback_trading_days=max(40, int(args.lookback_trading_days)),
        rebalance_interval=max(2, int(args.rebalance_interval)),
        top_n=max(1, int(args.top_n)),
        max_per_theme=max(1, int(args.max_per_theme)),
        benchmark_symbol=str(args.benchmark_symbol),
        commission_bps=float(args.commission_bps),
        slippage_bps=float(args.slippage_bps),
        workers=max(1, int(args.workers)),
    )
    print(f"[fast-backtest] markdown: {outputs['markdown']}")
    print(f"[fast-backtest] json: {outputs['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
