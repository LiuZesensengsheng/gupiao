from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.domain.symbols import SymbolError, normalize_symbol

DEFAULT_COMMON: dict[str, Any] = {
    "source": "auto",
    "tushare_token": "",
    "watchlist": "config/watchlist.json",
    "data_dir": "data",
    "start": "2018-01-01",
    "end": "2099-12-31",
    "min_train_days": 240,
    "step_days": 20,
    "l2": 0.8,
    "max_positions": 5,
    "use_margin_features": True,
    "margin_market_file": "input/margin_market.csv",
    "margin_stock_file": "input/margin_stock.csv",
    "use_us_index_context": False,
    "us_index_source": "akshare",
}

DEFAULT_TASK: dict[str, dict[str, Any]] = {
    "forecast": {
        "report": "reports/latest_report.md",
    },
    "discover": {
        "universe_file": "",
        "candidate_limit": 120,
        "top_k": 20,
        "exclude_watchlist": False,
        "report": "reports/discovery_report.md",
    },
    "sync-data": {
        "universe_size": 500,
        "universe_file": "",
        "universe_min_amount": 50000000.0,
        "universe_exclude_st": True,
        "include_indices": True,
        "force_refresh": False,
        "sleep_ms": 80,
        "parallel_workers": 1,
        "max_failures": 100,
        "write_universe_file": "config/universe_auto.json",
    },
    "sync-margin": {
        "symbols": "",
        "universe_file": "",
        "universe_limit": 0,
        "sleep_ms": 80,
    },
    "daily": {
        "universe_file": "config/universe_auto_longtrain.json",
        "universe_limit": 500,
        "positions_file": "",
        "portfolio_nav": 0.0,
        "trade_lot_size": 100,
        "news_file": "input/news_parts",
        "news_lookback_days": 45,
        "learned_news_lookback_days": 720,
        "news_half_life_days": 10.0,
        "market_news_strength": 0.9,
        "stock_news_strength": 1.1,
        "use_learned_news_fusion": True,
        "learned_news_min_samples": 80,
        "learned_holdout_ratio": 0.2,
        "learned_news_l2": 0.8,
        "learned_fusion_l2": 0.6,
        "report_date": "",
        "report": "reports/daily_report.md",
        "dashboard": "reports/daily_dashboard.html",
        "backtest_years": [3, 5],
        "backtest_retrain_days": 20,
        "backtest_weight_threshold": 0.50,
        "backtest_time_budget_minutes": 0.0,
        "commission_bps": 1.5,
        "slippage_bps": 2.0,
        "use_turnover_control": True,
        "max_trades_per_stock_per_day": 1,
        "max_trades_per_stock_per_week": 3,
        "min_weight_change_to_trade": 0.03,
        "range_t_sell_ret_1_min": 0.015,
        "range_t_sell_price_pos_20_min": 0.75,
        "range_t_buy_ret_1_max": -0.015,
        "range_t_buy_price_pos_20_max": 0.30,
        "use_tradeability_guard": True,
        "tradeability_limit_tolerance": 0.002,
        "tradeability_min_volume": 0.0,
        "limit_rule_file": "",
        "use_index_constituent_guard": False,
        "index_constituent_file": "",
        "index_constituent_symbol": "000300.SH",
        "enable_acceptance_checks": True,
        "acceptance_target_years": 3,
        "use_strategy_optimizer": True,
        "optimizer_retrain_days": [20, 40],
        "optimizer_weight_thresholds": [0.50, 0.60],
        "optimizer_max_positions": [3, 5],
        "optimizer_market_news_strengths": [0.8, 1.0],
        "optimizer_stock_news_strengths": [1.0, 1.2],
        "optimizer_force_full_news_strength_grid": False,
        "optimizer_turnover_penalty": 0.0015,
        "optimizer_drawdown_penalty": 0.20,
        "optimizer_target_years": 3,
        "optimizer_top_trials": 12,
        "optimizer_time_budget_minutes": 0.0,
    },
}


def _read_json_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config root must be an object: {config_path}")
    return payload


def _read_config_section(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = payload.get(section_name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section `{section_name}` must be an object")
    return section


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _parse_years(value: Any) -> tuple[int, ...]:
    if value is None:
        return (3, 5)
    if isinstance(value, (list, tuple)):
        parsed = [int(v) for v in value if int(v) > 0]
        return tuple(sorted(set(parsed))) if parsed else (3, 5)
    text = str(value).strip()
    if not text:
        return (3, 5)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    parsed = [int(p) for p in parts if int(p) > 0]
    return tuple(sorted(set(parsed))) if parsed else (3, 5)


def _parse_int_list(value: Any, *, min_value: int = 1) -> tuple[int, ...]:
    if value is None:
        return ()
    raw: list[Any]
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw = [p.strip() for p in text.split(",") if str(p).strip()]

    out: list[int] = []
    for item in raw:
        try:
            v = int(item)
        except (TypeError, ValueError):
            continue
        if v >= int(min_value):
            out.append(v)
    return tuple(sorted(set(out)))


def _parse_float_list(value: Any, *, min_value: float | None = None) -> tuple[float, ...]:
    if value is None:
        return ()
    raw: list[Any]
    if isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw = [p.strip() for p in text.split(",") if str(p).strip()]

    out: list[float] = []
    for item in raw:
        try:
            v = float(item)
        except (TypeError, ValueError):
            continue
        if min_value is None or v >= float(min_value):
            out.append(v)
    uniq = sorted({round(x, 6) for x in out})
    return tuple(float(x) for x in uniq)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def _parse_symbol_list(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    out: list[str] = []
    for token in text.replace("\n", ",").replace(";", ",").split(","):
        code = token.strip()
        if not code:
            continue
        try:
            out.append(normalize_symbol(code).symbol)
        except SymbolError:
            continue
    return sorted(set(out))


def _resolve_settings(args: argparse.Namespace, payload: dict[str, Any]) -> dict[str, Any]:
    common_cfg = _read_config_section(payload, "common")
    task_cfg = _read_config_section(payload, args.task)
    defaults = DEFAULT_TASK[args.task]
    resolved: dict[str, Any] = {}

    for key, default in DEFAULT_COMMON.items():
        resolved[key] = _coalesce(
            getattr(args, key, None),
            task_cfg.get(key),
            common_cfg.get(key),
            default,
        )

    for key, default in defaults.items():
        resolved[key] = _coalesce(
            getattr(args, key, None),
            task_cfg.get(key),
            common_cfg.get(key),
            default,
        )

    return resolved


def _masked_settings(settings: dict[str, Any]) -> dict[str, Any]:
    out = dict(settings)
    if "tushare_token" in out and str(out["tushare_token"]).strip():
        out["tushare_token"] = "***"
    return out
