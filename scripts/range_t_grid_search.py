from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.application.config import DailyConfig
from src.application.use_cases import DailyFusionResult, generate_daily_fusion
from src.application.watchlist import load_watchlist
from src.domain.symbols import normalize_symbol
from src.infrastructure.discovery import build_candidate_universe
from src.interfaces.cli.run_api_cli import DEFAULT_COMMON, DEFAULT_TASK


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    text = str(v).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _parse_years(value: Any) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        out = sorted({int(v) for v in value if int(v) > 0})
        return tuple(out or [1])
    text = str(value).strip()
    if not text:
        return (1,)
    out = sorted({int(x.strip()) for x in text.split(",") if x.strip() and int(x.strip()) > 0})
    return tuple(out or [1])


def _parse_float_list(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Grid list cannot be empty")
    return vals


def _to_percent(v: float) -> str:
    if np.isnan(v):
        return "NA"
    return f"{v * 100:.2f}%"


def _build_daily_config(settings: dict[str, Any]) -> DailyConfig:
    return DailyConfig(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        min_train_days=int(settings["min_train_days"]),
        step_days=int(settings["step_days"]),
        l2=float(settings["l2"]),
        max_positions=int(settings["max_positions"]),
        use_margin_features=_parse_bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
        margin_stock_file=str(settings["margin_stock_file"]),
        positions_file=str(settings.get("positions_file", "")).strip(),
        portfolio_nav=max(0.0, float(settings.get("portfolio_nav", 0.0))),
        trade_lot_size=max(1, int(settings.get("trade_lot_size", 100))),
        news_file=str(settings["news_file"]),
        news_lookback_days=int(settings["news_lookback_days"]),
        learned_news_lookback_days=int(settings["learned_news_lookback_days"]),
        news_half_life_days=float(settings["news_half_life_days"]),
        market_news_strength=float(settings["market_news_strength"]),
        stock_news_strength=float(settings["stock_news_strength"]),
        use_learned_news_fusion=_parse_bool(settings["use_learned_news_fusion"]),
        learned_news_min_samples=int(settings["learned_news_min_samples"]),
        learned_holdout_ratio=float(settings["learned_holdout_ratio"]),
        learned_news_l2=float(settings["learned_news_l2"]),
        learned_fusion_l2=float(settings["learned_fusion_l2"]),
        backtest_years=_parse_years(settings["backtest_years"]),
        backtest_retrain_days=int(settings["backtest_retrain_days"]),
        backtest_weight_threshold=float(settings["backtest_weight_threshold"]),
        backtest_time_budget_minutes=float(settings["backtest_time_budget_minutes"]),
        commission_bps=float(settings["commission_bps"]),
        slippage_bps=float(settings["slippage_bps"]),
        use_turnover_control=_parse_bool(settings["use_turnover_control"]),
        max_trades_per_stock_per_day=max(1, int(settings["max_trades_per_stock_per_day"])),
        max_trades_per_stock_per_week=max(1, int(settings["max_trades_per_stock_per_week"])),
        min_weight_change_to_trade=max(0.0, float(settings["min_weight_change_to_trade"])),
        range_t_sell_ret_1_min=float(settings["range_t_sell_ret_1_min"]),
        range_t_sell_price_pos_20_min=float(settings["range_t_sell_price_pos_20_min"]),
        range_t_buy_ret_1_max=float(settings["range_t_buy_ret_1_max"]),
        range_t_buy_price_pos_20_max=float(settings["range_t_buy_price_pos_20_max"]),
        use_tradeability_guard=_parse_bool(settings["use_tradeability_guard"]),
        tradeability_limit_tolerance=max(0.0, float(settings["tradeability_limit_tolerance"])),
        tradeability_min_volume=max(0.0, float(settings["tradeability_min_volume"])),
        limit_rule_file=str(settings["limit_rule_file"]).strip(),
        use_index_constituent_guard=_parse_bool(settings["use_index_constituent_guard"]),
        index_constituent_file=str(settings["index_constituent_file"]).strip(),
        index_constituent_symbol=str(settings["index_constituent_symbol"]).strip() or "000300.SH",
        enable_acceptance_checks=_parse_bool(settings["enable_acceptance_checks"]),
        acceptance_target_years=max(1, int(settings["acceptance_target_years"])),
        use_strategy_optimizer=_parse_bool(settings["use_strategy_optimizer"]),
        optimizer_retrain_days=tuple(int(v) for v in settings["optimizer_retrain_days"]),
        optimizer_weight_thresholds=tuple(float(v) for v in settings["optimizer_weight_thresholds"]),
        optimizer_max_positions=tuple(int(v) for v in settings["optimizer_max_positions"]),
        optimizer_market_news_strengths=tuple(float(v) for v in settings["optimizer_market_news_strengths"]),
        optimizer_stock_news_strengths=tuple(float(v) for v in settings["optimizer_stock_news_strengths"]),
        optimizer_turnover_penalty=float(settings["optimizer_turnover_penalty"]),
        optimizer_drawdown_penalty=float(settings["optimizer_drawdown_penalty"]),
        optimizer_target_years=max(1, int(settings["optimizer_target_years"])),
        optimizer_top_trials=max(1, int(settings["optimizer_top_trials"])),
        optimizer_time_budget_minutes=float(settings["optimizer_time_budget_minutes"]),
        report_date=str(settings.get("report_date", "")),
    )


def _prepare_settings(config_path: str) -> dict[str, Any]:
    payload = _read_json(config_path)
    common = payload.get("common", {}) if isinstance(payload.get("common", {}), dict) else {}
    daily = payload.get("daily", {}) if isinstance(payload.get("daily", {}), dict) else {}

    settings: dict[str, Any] = {}
    for k, v in DEFAULT_COMMON.items():
        settings[k] = daily.get(k, common.get(k, v))
    for k, v in DEFAULT_TASK["daily"].items():
        settings[k] = daily.get(k, common.get(k, v))
    return settings


def _score_result(res: DailyFusionResult) -> tuple[int, int, float, float, float]:
    return (
        1 if res.acceptance_constraints_pass else 0,
        1 if res.acceptance_ab_pass else 0,
        float(-res.acceptance_delta_max_drawdown) if not np.isnan(res.acceptance_delta_max_drawdown) else -1e9,
        float(-res.acceptance_delta_annual_turnover) if not np.isnan(res.acceptance_delta_annual_turnover) else -1e9,
        float(res.acceptance_delta_excess_annual_return) if not np.isnan(res.acceptance_delta_excess_annual_return) else -1e9,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search range-state T whitelist thresholds")
    parser.add_argument("--config", default="config/api.json")
    parser.add_argument("--sell-ret-grid", default="0.015,0.02,0.025")
    parser.add_argument("--sell-pos-grid", default="0.75,0.8,0.85")
    parser.add_argument("--buy-ret-grid", default="-0.015,-0.02,-0.025")
    parser.add_argument("--buy-pos-grid", default="0.30,0.35,0.40")
    parser.add_argument("--start", default="", help="Optional override for config start date")
    parser.add_argument("--backtest-years", default="", help="Optional override, e.g. 1 or 3,5")
    parser.add_argument("--acceptance-target-years", type=int, default=0, help="Optional override (>0)")
    parser.add_argument(
        "--use-strategy-optimizer",
        default="",
        help="Optional override true/false; empty means use config value",
    )
    parser.add_argument("--max-combos", type=int, default=0, help="0 means run all combos")
    parser.add_argument("--out", default="reports/range_t_grid_search.md")
    args = parser.parse_args()

    settings = _prepare_settings(args.config)
    if str(args.start).strip():
        settings["start"] = str(args.start).strip()
    if str(args.backtest_years).strip():
        settings["backtest_years"] = str(args.backtest_years).strip()
    if int(args.acceptance_target_years) > 0:
        settings["acceptance_target_years"] = int(args.acceptance_target_years)
    if str(args.use_strategy_optimizer).strip():
        settings["use_strategy_optimizer"] = _parse_bool(args.use_strategy_optimizer)
    settings["enable_acceptance_checks"] = True

    base_cfg = _build_daily_config(settings)
    print(
        "[GRID] effective settings: "
        f"start={base_cfg.start}, years={','.join(str(x) for x in base_cfg.backtest_years)}, "
        f"acceptance_target_years={base_cfg.acceptance_target_years}, "
        f"use_strategy_optimizer={base_cfg.use_strategy_optimizer}"
    )

    market_security, stocks, _ = load_watchlist(settings["watchlist"])
    universe = build_candidate_universe(
        source=settings["source"],
        data_dir=settings["data_dir"],
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    if not universe.rows:
        raise ValueError("Daily universe is empty.")
    stocks = sorted(universe.rows, key=lambda x: normalize_symbol(x.symbol).symbol)
    sector_map = {normalize_symbol(s.symbol).symbol: (s.sector or "其他") for s in stocks}

    sell_ret_grid = _parse_float_list(args.sell_ret_grid)
    sell_pos_grid = _parse_float_list(args.sell_pos_grid)
    buy_ret_grid = _parse_float_list(args.buy_ret_grid)
    buy_pos_grid = _parse_float_list(args.buy_pos_grid)

    combos = list(itertools.product(sell_ret_grid, sell_pos_grid, buy_ret_grid, buy_pos_grid))
    if args.max_combos > 0:
        combos = combos[: int(args.max_combos)]

    rows: list[dict[str, Any]] = []
    t0 = time.monotonic()
    total = len(combos)
    print(f"[GRID] start: {total} combos")

    for idx, (sell_ret, sell_pos, buy_ret, buy_pos) in enumerate(combos, start=1):
        cfg = replace(
            base_cfg,
            range_t_sell_ret_1_min=float(sell_ret),
            range_t_sell_price_pos_20_min=float(sell_pos),
            range_t_buy_ret_1_max=float(buy_ret),
            range_t_buy_price_pos_20_max=float(buy_pos),
        )
        result = generate_daily_fusion(
            config=cfg,
            market_security=market_security,
            stocks=stocks,
            sector_map=sector_map,
        )
        row = {
            "sell_ret_1_min": float(sell_ret),
            "sell_price_pos_20_min": float(sell_pos),
            "buy_ret_1_max": float(buy_ret),
            "buy_price_pos_20_max": float(buy_pos),
            "constraints_pass": bool(result.acceptance_constraints_pass),
            "ab_pass": bool(result.acceptance_ab_pass),
            "delta_excess_annual_return": float(result.acceptance_delta_excess_annual_return),
            "delta_max_drawdown": float(result.acceptance_delta_max_drawdown),
            "delta_annual_turnover": float(result.acceptance_delta_annual_turnover),
            "weekly_overflow": int(result.acceptance_limit_violations),
            "oversell_violations": int(result.acceptance_oversell_violations),
            "summary": str(result.acceptance_summary),
        }
        rows.append(row)
        elapsed = time.monotonic() - t0
        print(
            f"[GRID] {idx}/{total} "
            f"sell_ret={sell_ret:.3f}, sell_pos={sell_pos:.2f}, buy_ret={buy_ret:.3f}, buy_pos={buy_pos:.2f} | "
            f"ab={row['ab_pass']}, constraints={row['constraints_pass']} | "
            f"d_excess={_to_percent(row['delta_excess_annual_return'])}, "
            f"d_dd={_to_percent(row['delta_max_drawdown'])}, "
            f"d_turnover={_to_percent(row['delta_annual_turnover'])} | "
            f"elapsed={elapsed:.1f}s"
        )

    rows_sorted = sorted(rows, key=lambda x: _score_result_obj(x), reverse=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Range-T Grid Search")
    lines.append("")
    lines.append(f"- combos: {len(rows)}")
    lines.append(f"- elapsed_sec: {time.monotonic() - t0:.1f}")
    lines.append("")
    lines.append("| rank | sell_ret_1_min | sell_price_pos_20_min | buy_ret_1_max | buy_price_pos_20_max | constraints_pass | ab_pass | delta_excess | delta_max_dd | delta_turnover | weekly_overflow | oversell |")
    lines.append("|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows_sorted, start=1):
        lines.append(
            f"| {i} | {r['sell_ret_1_min']:.3f} | {r['sell_price_pos_20_min']:.2f} | "
            f"{r['buy_ret_1_max']:.3f} | {r['buy_price_pos_20_max']:.2f} | "
            f"{'Y' if r['constraints_pass'] else 'N'} | {'Y' if r['ab_pass'] else 'N'} | "
            f"{_to_percent(float(r['delta_excess_annual_return']))} | {_to_percent(float(r['delta_max_drawdown']))} | "
            f"{_to_percent(float(r['delta_annual_turnover']))} | {int(r['weekly_overflow'])} | {int(r['oversell_violations'])} |"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[GRID] report: {out_path.resolve()}")
    if rows_sorted:
        best = rows_sorted[0]
        print(
            "[GRID] best: "
            f"sell_ret={best['sell_ret_1_min']:.3f}, sell_pos={best['sell_price_pos_20_min']:.2f}, "
            f"buy_ret={best['buy_ret_1_max']:.3f}, buy_pos={best['buy_price_pos_20_max']:.2f}, "
            f"ab={best['ab_pass']}, constraints={best['constraints_pass']}"
        )
    return 0


def _score_result_obj(r: dict[str, Any]) -> tuple[int, int, float, float, float]:
    dd = float(r["delta_max_drawdown"])
    turn = float(r["delta_annual_turnover"])
    excess = float(r["delta_excess_annual_return"])
    dd_score = -dd if not np.isnan(dd) else -1e9
    turn_score = -turn if not np.isnan(turn) else -1e9
    excess_score = excess if not np.isnan(excess) else -1e9
    return (
        1 if bool(r["constraints_pass"]) else 0,
        1 if bool(r["ab_pass"]) else 0,
        dd_score,
        turn_score,
        excess_score,
    )


if __name__ == "__main__":
    raise SystemExit(main())
