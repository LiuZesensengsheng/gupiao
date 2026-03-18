from __future__ import annotations

import argparse
import json
import math
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

from scripts.build_dynamic300_rule_plan import _coalesce, _load_config
from src.application.v2_backtest_runtime import to_v2_backtest_summary
from src.application.v2_universe_generator import generate_dynamic_universe


@dataclass(frozen=True)
class BacktestSelectionRow:
    symbol: str
    name: str
    theme: str
    refined_score: float
    fresh_pool_score: float
    fresh_pool_pass: bool
    recent_high_gap20: float
    amount_ratio20: float
    theme_selected_count: int
    theme_strength: float
    portfolio_score: float
    weight: float = 0.0


@dataclass(frozen=True)
class RulePlanSnapshot:
    signal_date: str
    execution_date: str
    holdings: list[BacktestSelectionRow]
    selected_count: int
    fresh_pool_pass_count: int


def _load_settings(config_path: str) -> dict[str, Any]:
    payload = _load_config(config_path)
    daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
    common = payload.get("common", {}) if isinstance(payload, dict) else {}
    return {
        "data_dir": str(_coalesce(daily.get("data_dir"), common.get("data_dir"), default="data")),
        "cache_root": str(_coalesce(daily.get("cache_root"), default="cache")),
        "base_universe_file": str(
            _coalesce(
                daily.get("generated_universe_base_file"),
                daily.get("universe_file"),
                default="config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json",
            )
        ),
        "target_size": int(_coalesce(daily.get("dynamic_universe_target_size"), default=300)),
        "coarse_size": int(_coalesce(daily.get("dynamic_universe_coarse_size"), default=1000)),
        "min_history_days": int(_coalesce(daily.get("dynamic_universe_min_history_days"), default=480)),
        "min_recent_amount": float(_coalesce(daily.get("dynamic_universe_min_recent_amount"), default=100000.0)),
        "theme_cap_ratio": float(_coalesce(daily.get("dynamic_universe_theme_cap_ratio"), default=0.16)),
        "theme_floor_count": int(_coalesce(daily.get("dynamic_universe_theme_floor_count"), default=2)),
        "turnover_quality_weight": float(_coalesce(daily.get("dynamic_universe_turnover_quality_weight"), default=0.25)),
        "theme_weight": float(_coalesce(daily.get("dynamic_universe_theme_weight"), default=0.18)),
        "main_board_only": bool(_coalesce(daily.get("main_board_only_universe"), default=True)),
        "use_concepts": bool(_coalesce(daily.get("generator_use_concepts"), default=True)),
    }


def _portfolio_score(row: dict[str, object], theme_count: int, theme_strength: float) -> float:
    refined = float(row.get("refined_score", 0.0))
    fresh = float(row.get("fresh_pool_score", 0.0))
    amount_ratio20 = float(row.get("amount_ratio20", 0.0))
    near_high = float(row.get("recent_high_gap20", -0.5))
    theme_count_score = min(1.0, max(0.0, float(theme_count) / 12.0))
    amount_balance = 1.0 - min(abs(amount_ratio20 - 1.15) / 1.05, 1.0)
    near_high_score = 1.0 - min(abs(near_high + 0.03) / 0.12, 1.0)
    return float(
        0.50 * refined
        + 0.20 * fresh
        + 0.12 * theme_count_score
        + 0.10 * float(theme_strength)
        + 0.04 * amount_balance
        + 0.04 * near_high_score
    )


def _is_primary_candidate(row: dict[str, object], theme_count: int) -> bool:
    return bool(
        bool(row.get("fresh_pool_pass", False))
        and float(row.get("fresh_pool_score", 0.0)) >= 0.55
        and theme_count >= 4
    )


def _select_portfolio(
    rows: list[BacktestSelectionRow],
    *,
    top_n: int,
    max_per_theme: int,
) -> list[BacktestSelectionRow]:
    selected: list[BacktestSelectionRow] = []
    counts: dict[str, int] = {}
    for row in rows:
        if len(selected) >= top_n:
            break
        if counts.get(row.theme, 0) >= int(max_per_theme):
            continue
        selected.append(row)
        counts[row.theme] = counts.get(row.theme, 0) + 1
    if len(selected) < top_n:
        selected_symbols = {row.symbol for row in selected}
        for row in rows:
            if len(selected) >= top_n:
                break
            if row.symbol in selected_symbols:
                continue
            selected.append(row)
            selected_symbols.add(row.symbol)
    total_score = sum(max(row.portfolio_score, 1e-9) for row in selected)
    weighted: list[BacktestSelectionRow] = []
    for row in selected:
        weight = max(row.portfolio_score, 1e-9) / max(total_score, 1e-9)
        weighted.append(
            BacktestSelectionRow(
                symbol=row.symbol,
                name=row.name,
                theme=row.theme,
                refined_score=row.refined_score,
                fresh_pool_score=row.fresh_pool_score,
                fresh_pool_pass=row.fresh_pool_pass,
                recent_high_gap20=row.recent_high_gap20,
                amount_ratio20=row.amount_ratio20,
                theme_selected_count=row.theme_selected_count,
                theme_strength=row.theme_strength,
                portfolio_score=row.portfolio_score,
                weight=float(weight),
            )
        )
    return weighted


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
    rows: list[BacktestSelectionRow] = []
    for row in result.selected_300:
        theme = str(row.get("sector", "其他"))
        theme_info = theme_map.get(theme, {"selected_count": 0, "theme_strength": 0.0})
        score = _portfolio_score(
            dict(row),
            theme_count=int(theme_info["selected_count"]),
            theme_strength=float(theme_info["theme_strength"]),
        )
        if not _is_primary_candidate(dict(row), int(theme_info["selected_count"])):
            score *= 0.92
        rows.append(
            BacktestSelectionRow(
                symbol=str(row["symbol"]),
                name=str(row["name"]),
                theme=theme,
                refined_score=float(row.get("refined_score", 0.0)),
                fresh_pool_score=float(row.get("fresh_pool_score", 0.0)),
                fresh_pool_pass=bool(row.get("fresh_pool_pass", False)),
                recent_high_gap20=float(row.get("recent_high_gap20", 0.0)),
                amount_ratio20=float(row.get("amount_ratio20", 0.0)),
                theme_selected_count=int(theme_info["selected_count"]),
                theme_strength=float(theme_info["theme_strength"]),
                portfolio_score=float(score),
            )
        )
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
    holdings = _select_portfolio(ranked, top_n=top_n, max_per_theme=max_per_theme)
    fresh_pool_pass_count = int(result.generator_manifest.config.get("fresh_pool_pass_count", 0))
    return RulePlanSnapshot(
        signal_date=signal_date,
        execution_date=execution_date,
        holdings=holdings,
        selected_count=len(result.selected_300),
        fresh_pool_pass_count=fresh_pool_pass_count,
    )


@lru_cache(maxsize=None)
def _load_close_frame(data_dir: str, symbol: str) -> pd.DataFrame:
    path = Path(data_dir) / f"{symbol}.csv"
    frame = pd.read_csv(path, usecols=["date", "close"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    frame["ret_cc"] = frame["close"].pct_change().fillna(0.0)
    return frame


@lru_cache(maxsize=None)
def _close_return_lookup(data_dir: str, symbol: str) -> dict[pd.Timestamp, float]:
    frame = _load_close_frame(data_dir, symbol)
    return {
        pd.Timestamp(row.date): float(row.ret_cc)
        for row in frame.itertuples(index=False)
    }


def _has_price(data_dir: str, symbol: str, date: pd.Timestamp) -> bool:
    return pd.Timestamp(date) in _close_return_lookup(data_dir, symbol)


def _normalize_target_weights(
    holdings: list[BacktestSelectionRow],
    *,
    data_dir: str,
    date: pd.Timestamp,
) -> dict[str, float]:
    available = [row for row in holdings if _has_price(data_dir, row.symbol, date)]
    if not available:
        return {}
    total = sum(max(float(row.weight), 0.0) for row in available)
    if total <= 1e-9:
        eq_weight = 1.0 / float(len(available))
        return {row.symbol: eq_weight for row in available}
    return {
        row.symbol: float(row.weight) / float(total)
        for row in available
    }


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
    snapshots: list[RulePlanSnapshot] = []
    if int(workers) <= 1:
        for signal_date, execution_date in schedule:
            snapshots.append(
                _build_snapshot(
                    settings=settings,
                    signal_date=signal_date,
                    execution_date=execution_date,
                    top_n=top_n,
                    max_per_theme=max_per_theme,
                )
            )
        return snapshots
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        future_map = {
            executor.submit(
                _build_snapshot,
                settings=settings,
                signal_date=signal_date,
                execution_date=execution_date,
                top_n=top_n,
                max_per_theme=max_per_theme,
            ): (signal_date, execution_date)
            for signal_date, execution_date in schedule
        }
        ordered: dict[str, RulePlanSnapshot] = {}
        for future in as_completed(future_map):
            snapshot = future.result()
            ordered[snapshot.signal_date] = snapshot
        snapshots = [ordered[signal_date] for signal_date, _ in schedule if signal_date in ordered]
    return snapshots


def _rebalance_weights_after_cost(
    target_weights: dict[str, float],
    cost: float,
) -> tuple[dict[str, float], float]:
    investable = max(0.0, 1.0 - float(cost))
    scaled = {
        symbol: float(weight) * investable
        for symbol, weight in target_weights.items()
        if float(weight) > 1e-9
    }
    cash = max(0.0, 1.0 - sum(scaled.values()))
    return scaled, cash


def _simulate_fast_backtest(
    *,
    settings: dict[str, Any],
    benchmark_symbol: str,
    end_date: str,
    snapshots: list[RulePlanSnapshot],
    commission_bps: float,
    slippage_bps: float,
) -> dict[str, Any]:
    benchmark_frame = _load_close_frame(settings["data_dir"], benchmark_symbol)
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

    snapshot_by_execution = {
        pd.Timestamp(item.execution_date): item
        for item in snapshots
    }
    benchmark_return_lookup = _close_return_lookup(settings["data_dir"], benchmark_symbol)
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
    first_entry_done = False

    for date in trade_dates:
        date = pd.Timestamp(date)
        nav_before = 1.0
        benchmark_ret = float(benchmark_return_lookup.get(date, 0.0))
        position_values: dict[str, float] = {}
        gross_ret = 0.0
        for symbol, weight in current_weights.items():
            ret = float(_close_return_lookup(settings["data_dir"], symbol).get(date, 0.0))
            gross_ret += float(weight) * ret
            position_values[symbol] = float(weight) * (1.0 + ret)
        nav_mid = current_cash + sum(position_values.values())
        if nav_mid <= 1e-12:
            nav_mid = 1e-12
        pre_trade_weights = {
            symbol: float(value) / nav_mid
            for symbol, value in position_values.items()
            if float(value) > 1e-12
        }
        pre_trade_cash = max(0.0, float(current_cash) / nav_mid)

        turnover = 0.0
        cost = 0.0
        if date in snapshot_by_execution:
            snapshot = snapshot_by_execution[date]
            target_weights = _normalize_target_weights(
                snapshot.holdings,
                data_dir=settings["data_dir"],
                date=date,
            )
            all_symbols = sorted(set(pre_trade_weights) | set(target_weights))
            turnover = float(
                sum(
                    abs(float(target_weights.get(symbol, 0.0)) - float(pre_trade_weights.get(symbol, 0.0)))
                    for symbol in all_symbols
                )
            )
            cost = float(turnover * cost_rate)
            current_weights, current_cash = _rebalance_weights_after_cost(target_weights, cost)
            if not first_entry_done:
                benchmark_ret = 0.0
                first_entry_done = True
        else:
            current_weights = pre_trade_weights
            current_cash = pre_trade_cash

        nav_after = nav_mid * (1.0 - cost)
        daily_return = float(nav_after / nav_before - 1.0)
        returns.append(daily_return)
        benchmark_returns.append(float(benchmark_ret))
        gross_returns.append(float(gross_ret))
        turnovers.append(float(turnover))
        costs.append(float(cost))
        fill_ratios.append(1.0 if turnover > 1e-9 else 0.0)
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
        "# Dynamic 300 轻量规则版快速回测",
        "",
        f"- 截止日期: {end_date}",
        f"- 基准: {benchmark_symbol}",
        f"- 回看交易日: {lookback_trading_days}",
        f"- 调仓步长: {rebalance_interval} 个交易日",
        f"- 组合持仓数: {top_n}",
        f"- 同主题上限: {max_per_theme}",
        f"- 并发快照数: {workers}",
        "",
        "## 回测结果",
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
        "",
        "## 调仓快照",
        "",
    ]
    for item in snapshots:
        holdings = item.get("holdings", [])
        names = ", ".join(f"{row['name']}({row['symbol']})" for row in holdings)
        lines.append(
            f"- {item.get('signal_date')} -> {item.get('execution_date')}: "
            f"{names or '无有效持仓'}"
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
    benchmark_frame = _load_close_frame(settings["data_dir"], benchmark_symbol)
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
