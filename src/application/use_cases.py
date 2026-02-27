from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.application.config import DailyConfig, DiscoverConfig, ForecastConfig
from src.domain.entities import (
    BacktestMetrics,
    BlendedRow,
    DiscoveryRow,
    EffectSummary,
    ForecastRow,
    FusionDiagnostics,
    MarketForecast,
    NewsItem,
    Security,
    SentimentAggregate,
    StrategyTrial,
    TradeAction,
)
from src.domain.symbols import normalize_symbol
from src.domain.policies import allocate_weights, blend_horizon_score, decide_market_state, target_exposure
from src.infrastructure.backtesting import BacktestResult, run_portfolio_backtest
from src.infrastructure.discovery import build_candidate_universe, compute_volume_risk
from src.infrastructure.effect_analysis import build_latest_snapshot, compute_effect_summary, compute_sector_table
from src.infrastructure.features import MARKET_FEATURE_COLUMNS, make_market_feature_frame, make_stock_feature_frame, stock_feature_columns
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.market_data import load_symbol_daily
from src.infrastructure.news_fusion import predict_with_learned_fusion
from src.infrastructure.news_repository import load_news_items


@dataclass(frozen=True)
class ForecastResult:
    market_forecast: MarketForecast
    stock_rows: List[ForecastRow]


@dataclass(frozen=True)
class DailyFusionResult:
    as_of_date: pd.Timestamp
    source: str
    market_forecast: MarketForecast
    market_news_short_prob: float
    market_news_mid_prob: float
    market_final_short: float
    market_final_mid: float
    market_state_code: str
    market_state_label: str
    strategy_template: str
    intraday_t_level: str
    effective_total_exposure: float
    effective_weight_threshold: float
    effective_max_positions: int
    effective_max_trades_per_stock_per_day: int
    effective_max_trades_per_stock_per_week: int
    market_short_sent: SentimentAggregate
    market_mid_sent: SentimentAggregate
    market_fusion_mode_short: str
    market_fusion_mode_mid: str
    blended_rows: List[BlendedRow]
    learning_diagnostics: List[FusionDiagnostics]
    effect_summary: EffectSummary
    sector_table: pd.DataFrame
    backtest_metrics: List[BacktestMetrics]
    backtest_curve: pd.DataFrame
    strategy_objective_text: str
    strategy_target_metric_label: str
    strategy_selected: StrategyTrial | None
    strategy_trials: List[StrategyTrial]
    acceptance_enabled: bool
    acceptance_ab_pass: bool
    acceptance_constraints_pass: bool
    acceptance_summary: str
    acceptance_delta_excess_annual_return: float
    acceptance_delta_max_drawdown: float
    acceptance_delta_annual_turnover: float
    acceptance_limit_violations: int
    acceptance_oversell_violations: int
    trade_plan_basis: str
    trade_plan_nav: float
    trade_plan_lot_size: int
    trade_actions: List[TradeAction]
    news_items_count: int
    news_items: List[NewsItem]


@dataclass(frozen=True)
class DiscoveryResult:
    as_of_date: pd.Timestamp
    source: str
    market_forecast: MarketForecast
    universe_size: int
    universe_source: str
    warnings: List[str]
    rows: List[DiscoveryRow]


@dataclass(frozen=True)
class _StrategySelection:
    retrain_days: int
    weight_threshold: float
    max_positions: int
    market_news_strength: float
    stock_news_strength: float
    objective_text: str
    target_metric_label: str
    selected_trial: StrategyTrial | None
    trials: List[StrategyTrial]
    best_backtest: BacktestResult | None


def generate_forecast(
    config: ForecastConfig,
    market_security: Security,
    stocks: List[Security],
    *,
    enable_walk_forward_eval: bool = True,
) -> ForecastResult:
    market_forecast, stock_rows = run_quant_pipeline(
        market_security=market_security,
        stock_securities=stocks,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        min_train_days=config.min_train_days,
        step_days=config.step_days,
        l2=config.l2,
        max_positions=config.max_positions,
        use_margin_features=config.use_margin_features,
        margin_market_file=config.margin_market_file,
        margin_stock_file=config.margin_stock_file,
        enable_walk_forward_eval=bool(enable_walk_forward_eval),
    )
    return ForecastResult(market_forecast=market_forecast, stock_rows=stock_rows)


def generate_discovery(
    config: DiscoverConfig,
    market_security: Security,
    watchlist_stocks: List[Security],
) -> DiscoveryResult:
    exclude_symbols = [market_security.symbol]
    if config.exclude_watchlist:
        exclude_symbols.extend(s.symbol for s in watchlist_stocks)

    universe = build_candidate_universe(
        source=config.source,
        data_dir=config.data_dir,
        universe_file=config.universe_file,
        candidate_limit=config.candidate_limit,
        exclude_symbols=exclude_symbols,
    )
    if not universe.rows:
        raise ValueError("Discovery universe is empty. Provide `--universe-file` or prepare more symbols in data directory.")

    forecast = generate_forecast(
        config=ForecastConfig(
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            l2=config.l2,
            max_positions=config.max_positions,
            use_margin_features=config.use_margin_features,
            margin_market_file=config.margin_market_file,
            margin_stock_file=config.margin_stock_file,
        ),
        market_security=market_security,
        stocks=universe.rows,
        enable_walk_forward_eval=False,
    )
    as_of_date = pd.Timestamp(forecast.market_forecast.latest_date).normalize()
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
    )
    market_feat = make_market_feature_frame(market_raw)

    top_rows = sorted(forecast.stock_rows, key=lambda x: x.score, reverse=True)[: max(1, int(config.top_k))]
    out: List[DiscoveryRow] = []
    for row in top_rows:
        stock_raw = load_symbol_daily(
            symbol=row.symbol,
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
        )
        stock_feat = make_stock_feature_frame(stock_raw, market_feat)
        risk_flag, risk_note = compute_volume_risk(stock_feat, as_of_date=as_of_date)
        out.append(
            DiscoveryRow(
                symbol=row.symbol,
                name=row.name,
                short_prob=row.short_prob,
                mid_prob=row.mid_prob,
                score=row.score,
                suggested_weight=row.suggested_weight,
                volume_risk_flag=risk_flag,
                volume_risk_note=risk_note,
                short_drivers=row.short_drivers,
                mid_drivers=row.mid_drivers,
            )
        )

    return DiscoveryResult(
        as_of_date=as_of_date,
        source=config.source,
        market_forecast=forecast.market_forecast,
        universe_size=len(universe.rows),
        universe_source=universe.source_label,
        warnings=universe.warnings,
        rows=out,
    )


def _prepare_learning_frames(
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
) -> tuple[pd.DataFrame, List[str], Dict[str, List[str]], Dict[str, pd.DataFrame]]:
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        market_dates=market_feat_base["date"],
        use_margin_features=config.use_margin_features,
        margin_market_file=config.margin_market_file,
    )
    market_feat = market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")
    market_feature_cols = MARKET_FEATURE_COLUMNS + market_context.feature_columns
    stock_feature_cols_map: Dict[str, List[str]] = {}
    stock_frames: Dict[str, pd.DataFrame] = {}
    for security in stocks:
        symbol = normalize_symbol(security.symbol).symbol
        stock_raw = load_symbol_daily(
            symbol=symbol,
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
        )
        stock_feat = make_stock_feature_frame(stock_raw, market_feat)
        stock_margin_cols: list[str] = []
        if config.use_margin_features:
            margin_frame, margin_cols, _ = build_stock_margin_features(
                margin_stock_file=config.margin_stock_file,
                symbol=symbol,
                start=config.start,
                end=config.end,
            )
            if margin_cols:
                stock_feat = stock_feat.merge(margin_frame, on="date", how="left", validate="1:1")
                stock_margin_cols = list(margin_cols)
        stock_frames[symbol] = stock_feat
        stock_feature_cols_map[symbol] = stock_feature_columns(
            extra_market_cols=market_context.feature_columns,
            extra_stock_cols=stock_margin_cols,
        )
    return market_feat, market_feature_cols, stock_feature_cols_map, stock_frames


def _blend_stock_rows(
    rows: List[ForecastRow],
    news_items_train: List[NewsItem],
    news_items_live: List[NewsItem],
    as_of_date: pd.Timestamp,
    config: DailyConfig,
    stock_news_strength: float,
    stock_feature_cols_map: Dict[str, List[str]],
    stock_feature_frames: Dict[str, pd.DataFrame],
) -> tuple[List[BlendedRow], List[FusionDiagnostics]]:
    out: List[BlendedRow] = []
    diagnostics: List[FusionDiagnostics] = []
    for row in rows:
        feature_frame = stock_feature_frames.get(row.symbol)
        volume_risk_flag = False
        volume_risk_note = ""
        if feature_frame is not None and not feature_frame.empty:
            history = feature_frame[feature_frame["date"] <= as_of_date].sort_values("date")
            if not history.empty:
                latest = history.iloc[-1]
                hvbd_recent = float(latest.get("hvbd_recent_5", 0.0))
                vol_ratio = float(latest.get("vol_ratio_20", np.nan))
                price_pos = float(latest.get("price_pos_20", np.nan))
                if hvbd_recent >= 0.5:
                    volume_risk_flag = True
                    volume_risk_note = f"高位巨量大阴线(5日内), 量能比={vol_ratio:.2f}, 位置={price_pos:.2f}"
                else:
                    volume_risk_note = f"量能比={vol_ratio:.2f}, 位置={price_pos:.2f}"
        feature_cols = stock_feature_cols_map.get(row.symbol, stock_feature_columns())
        short_pred = predict_with_learned_fusion(
            enabled=config.use_learned_news_fusion,
            base_prob=row.short_prob,
            target=row.symbol,
            horizon="short",
            feature_frame=feature_frame,
            feature_cols=feature_cols,
            target_col="target_1d_up",
            news_items_train=news_items_train,
            news_items_live=news_items_live,
            as_of_date=as_of_date,
            half_life_days=config.news_half_life_days,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            quant_l2=config.l2,
            news_l2=config.learned_news_l2,
            fusion_l2=config.learned_fusion_l2,
            min_samples=config.learned_news_min_samples,
            holdout_ratio=config.learned_holdout_ratio,
            fallback_strength=stock_news_strength,
        )
        mid_pred = predict_with_learned_fusion(
            enabled=config.use_learned_news_fusion,
            base_prob=row.mid_prob,
            target=row.symbol,
            horizon="mid",
            feature_frame=feature_frame,
            feature_cols=feature_cols,
            target_col="target_20d_up",
            news_items_train=news_items_train,
            news_items_live=news_items_live,
            as_of_date=as_of_date,
            half_life_days=config.news_half_life_days,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            quant_l2=config.l2,
            news_l2=config.learned_news_l2,
            fusion_l2=config.learned_fusion_l2,
            min_samples=config.learned_news_min_samples,
            holdout_ratio=config.learned_holdout_ratio,
            fallback_strength=stock_news_strength,
        )
        diagnostics.extend([short_pred.diagnostics, mid_pred.diagnostics])
        final_short = short_pred.final_prob
        final_mid = mid_pred.final_prob
        final_score = blend_horizon_score(final_short, final_mid, short_weight=0.55)
        out.append(
            BlendedRow(
                symbol=row.symbol,
                name=row.name,
                base_short=row.short_prob,
                base_mid=row.mid_prob,
                news_short_prob=short_pred.news_prob,
                news_mid_prob=mid_pred.news_prob,
                final_short=final_short,
                final_mid=final_mid,
                final_score=final_score,
                short_sent=short_pred.sentiment,
                mid_sent=mid_pred.sentiment,
                fusion_mode_short=short_pred.mode,
                fusion_mode_mid=mid_pred.mode,
                volume_risk_flag=volume_risk_flag,
                volume_risk_note=volume_risk_note,
                short_drivers=list(row.short_drivers),
                mid_drivers=list(row.mid_drivers),
            )
        )
    return out, diagnostics


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return np.nan
        text = str(value).strip()
        if not text or text.lower() in {"na", "nan", "none"}:
            return np.nan
        out = float(text)
        return float(out) if np.isfinite(out) else np.nan
    except Exception:
        return np.nan


def _safe_weight(value: Any) -> float:
    raw = _safe_float(value)
    if pd.isna(raw):
        return np.nan
    w = float(raw)
    if 1.0 < w <= 100.0:
        w = w / 100.0
    return float(np.clip(w, 0.0, 1.0))


def _to_symbol(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    try:
        return normalize_symbol(text).symbol
    except Exception:
        return ""


def _read_positions_records(path_text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path_text = str(path_text).strip()
    if not path_text:
        return [], {}
    path = Path(path_text)
    if not path.exists():
        print(f"[WARN] positions file not found: {path}")
        return [], {}
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] failed to read positions csv: {path} ({exc})")
            return [], {}
        if frame.empty:
            return [], {}
        return frame.to_dict("records"), {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] failed to read positions json: {path} ({exc})")
        return [], {}
    if isinstance(payload, list):
        return [dict(x) for x in payload if isinstance(x, dict)], {}
    if not isinstance(payload, dict):
        return [], {}
    for key in ("positions", "stocks", "rows"):
        part = payload.get(key)
        if isinstance(part, list):
            return [dict(x) for x in part if isinstance(x, dict)], dict(payload)
    return [], dict(payload)


def _load_latest_close_map(
    *,
    symbols: list[str],
    config: DailyConfig,
    as_of_date: pd.Timestamp,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for symbol in sorted(set(symbols)):
        if not symbol:
            continue
        try:
            raw = load_symbol_daily(
                symbol=symbol,
                source=config.source,
                data_dir=config.data_dir,
                start=config.start,
                end=config.end,
            )
            history = raw[raw["date"] <= as_of_date].sort_values("date")
            if history.empty:
                history = raw.sort_values("date")
            if history.empty:
                continue
            close = _safe_float(history.iloc[-1].get("close"))
            if not pd.isna(close) and close > 0:
                out[symbol] = float(close)
        except Exception:
            continue
    return out


def _build_trade_actions(
    *,
    config: DailyConfig,
    as_of_date: pd.Timestamp,
    blended_rows: list[BlendedRow],
    current_holdings: list[Security],
) -> tuple[str, float, list[TradeAction]]:
    target_rows = [row for row in blended_rows if float(row.suggested_weight) > 1e-9]
    target_weight_map = {row.symbol: float(row.suggested_weight) for row in target_rows}
    name_map = {row.symbol: row.name for row in blended_rows}
    for sec in current_holdings:
        symbol = _to_symbol(sec.symbol)
        if symbol:
            name_map.setdefault(symbol, sec.name)

    records, meta = _read_positions_records(config.positions_file)
    parsed_records: list[dict[str, Any]] = []
    for rec in records:
        symbol = _to_symbol(rec.get("symbol") or rec.get("ts_code") or rec.get("code"))
        if not symbol:
            continue
        parsed_records.append(
            {
                "symbol": symbol,
                "name": str(rec.get("name") or rec.get("security_name") or name_map.get(symbol, symbol)),
                "weight": _safe_weight(
                    rec.get("current_weight")
                    if rec.get("current_weight") is not None
                    else rec.get("weight")
                ),
                "shares": _safe_float(rec.get("shares") if rec.get("shares") is not None else rec.get("quantity")),
                "market_value": _safe_float(
                    rec.get("market_value") if rec.get("market_value") is not None else rec.get("value")
                ),
            }
        )
        name_map[symbol] = parsed_records[-1]["name"]

    fallback_symbols = [_to_symbol(sec.symbol) for sec in current_holdings]
    fallback_symbols = [s for s in fallback_symbols if s]
    symbols_for_price = sorted(set(target_weight_map.keys()) | {item["symbol"] for item in parsed_records} | set(fallback_symbols))
    close_map = _load_latest_close_map(symbols=symbols_for_price, config=config, as_of_date=as_of_date)

    current_weight_map: dict[str, float] = {}
    current_shares_map: dict[str, float] = {}
    value_map: dict[str, float] = {}
    weight_basis_used = False
    for item in parsed_records:
        symbol = item["symbol"]
        shares = float(item["shares"])
        if not pd.isna(shares) and shares > 0:
            current_shares_map[symbol] = shares
        weight = float(item["weight"])
        if not pd.isna(weight):
            current_weight_map[symbol] = weight
            weight_basis_used = True
        market_value = float(item["market_value"])
        if not pd.isna(market_value) and market_value >= 0:
            value_map[symbol] = market_value
            continue
        price = close_map.get(symbol, np.nan)
        if not pd.isna(shares) and shares > 0 and not pd.isna(price) and price > 0:
            value_map[symbol] = float(shares * price)

    plan_basis = ""
    plan_nav = float(np.nan)
    if weight_basis_used and current_weight_map:
        total_weight = float(sum(max(0.0, w) for w in current_weight_map.values()))
        if total_weight > 1e-9:
            if total_weight > 1.0 + 1e-6:
                current_weight_map = {s: float(max(0.0, w) / total_weight) for s, w in current_weight_map.items()}
                plan_basis = "positions_file(weight, normalized)"
            else:
                plan_basis = "positions_file(weight)"
        else:
            current_weight_map = {}

    if not current_weight_map and value_map:
        meta_cash = _safe_float(meta.get("cash"))
        if pd.isna(meta_cash):
            meta_cash = 0.0
        nav_meta = _safe_float(meta.get("portfolio_nav"))
        if pd.isna(nav_meta):
            nav_meta = _safe_float(meta.get("total_nav"))
        total_mv = float(sum(max(0.0, v) for v in value_map.values()))
        if pd.isna(nav_meta) or nav_meta <= 0:
            nav_meta = total_mv + float(meta_cash)
        if nav_meta > 0 and total_mv > 0:
            current_weight_map = {s: float(v / nav_meta) for s, v in value_map.items()}
            plan_nav = float(nav_meta)
            plan_basis = "positions_file(market_value/shares)"

    if not current_weight_map:
        if fallback_symbols:
            eq = 1.0 / float(len(fallback_symbols))
            current_weight_map = {s: eq for s in fallback_symbols}
            plan_basis = "watchlist_equal_weight(fallback)"
        else:
            plan_basis = "target_only(no_current_positions)"

    if config.portfolio_nav > 0:
        plan_nav = float(config.portfolio_nav)
    if pd.isna(plan_nav) and config.portfolio_nav > 0:
        plan_nav = float(config.portfolio_nav)

    lot_size = max(1, int(config.trade_lot_size))
    trade_threshold = max(0.0, float(config.min_weight_change_to_trade))
    symbols_union = sorted(set(current_weight_map.keys()) | set(target_weight_map.keys()))
    actions: list[TradeAction] = []
    for symbol in symbols_union:
        name = name_map.get(symbol, symbol)
        current_weight = float(current_weight_map.get(symbol, 0.0))
        target_weight = float(target_weight_map.get(symbol, 0.0))
        delta_weight = float(target_weight - current_weight)
        if delta_weight > trade_threshold:
            action = "BUY"
        elif delta_weight < -trade_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        est_price = float(close_map.get(symbol, np.nan))
        est_delta_value = float(np.nan)
        est_delta_shares = float(np.nan)
        est_delta_lots = float(np.nan)
        note_parts: list[str] = []
        if not pd.isna(plan_nav) and plan_nav > 0:
            est_delta_value = float(delta_weight * plan_nav)
        if not pd.isna(est_delta_value) and not pd.isna(est_price) and est_price > 0:
            raw_shares = float(est_delta_value / est_price)
            sign = 1.0 if raw_shares >= 0 else -1.0
            lots = float(np.floor(abs(raw_shares) / float(lot_size)))
            est_delta_lots = sign * lots
            est_delta_shares = sign * lots * float(lot_size)
            if action == "HOLD":
                est_delta_lots = 0.0
                est_delta_shares = 0.0
            current_shares = float(current_shares_map.get(symbol, np.nan))
            if action == "SELL" and not pd.isna(current_shares) and current_shares > 0 and est_delta_shares < 0:
                max_sell_lots = float(np.floor(current_shares / float(lot_size)))
                capped = -min(abs(est_delta_lots), max_sell_lots)
                est_delta_lots = float(capped)
                est_delta_shares = float(capped * lot_size)
                if abs(capped) < abs(lots):
                    note_parts.append("sell_capped_by_current_shares")
        else:
            if pd.isna(plan_nav) or plan_nav <= 0:
                note_parts.append("missing_portfolio_nav")
            if pd.isna(est_price):
                note_parts.append("missing_price")

        if action == "HOLD" and abs(delta_weight) > 1e-9:
            note_parts.append("below_min_weight_change")
        actions.append(
            TradeAction(
                symbol=symbol,
                name=name,
                action=action,
                current_weight=current_weight,
                target_weight=target_weight,
                delta_weight=delta_weight,
                est_price=est_price,
                est_delta_value=est_delta_value,
                est_delta_shares=est_delta_shares,
                est_delta_lots=est_delta_lots,
                current_shares=float(current_shares_map.get(symbol, np.nan)),
                note=";".join(note_parts),
            )
        )

    actions.sort(key=lambda x: (abs(float(x.delta_weight)), float(x.target_weight)), reverse=True)
    return plan_basis, plan_nav, actions


def _sanitize_int_grid(values: tuple[int, ...], *, fallback: int, min_value: int = 1, max_value: int | None = None) -> list[int]:
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value < int(min_value):
            continue
        if max_value is not None and value > int(max_value):
            continue
        out.append(value)
    uniq = sorted(set(out))
    return uniq if uniq else [int(fallback)]


def _sanitize_float_grid(values: tuple[float, ...], *, fallback: float, min_value: float = 0.0) -> list[float]:
    out: list[float] = []
    for raw in values:
        value = float(raw)
        if value < float(min_value):
            continue
        out.append(value)
    uniq = sorted({round(v, 6) for v in out})
    return [float(v) for v in uniq] if uniq else [float(fallback)]


def _run_daily_backtest(
    *,
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
    news_items_train: List[NewsItem],
    retrain_days: int,
    weight_threshold: float,
    max_positions: int,
    market_news_strength: float,
    stock_news_strength: float,
    max_trades_per_stock_per_day: int,
    max_trades_per_stock_per_week: int,
    use_state_engine: bool = True,
) -> BacktestResult:
    return run_portfolio_backtest(
        market_security=market_security,
        stock_securities=stocks,
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        min_train_days=config.min_train_days,
        l2=config.l2,
        retrain_days=int(retrain_days),
        weight_threshold=float(weight_threshold),
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps,
        window_years=config.backtest_years,
        news_items=news_items_train,
        apply_news_fusion=True,
        max_runtime_seconds=max(0.0, float(config.backtest_time_budget_minutes) * 60.0),
        news_half_life_days=config.news_half_life_days,
        market_news_strength=float(market_news_strength),
        stock_news_strength=float(stock_news_strength),
        use_learned_news_fusion=config.use_learned_news_fusion,
        learned_news_min_samples=config.learned_news_min_samples,
        learned_news_l2=config.learned_news_l2,
        learned_fusion_l2=config.learned_fusion_l2,
        max_positions=int(max_positions),
        use_turnover_control=config.use_turnover_control,
        max_trades_per_stock_per_day=int(max_trades_per_stock_per_day),
        max_trades_per_stock_per_week=int(max_trades_per_stock_per_week),
        min_weight_change_to_trade=float(config.min_weight_change_to_trade),
        range_t_sell_ret_1_min=float(config.range_t_sell_ret_1_min),
        range_t_sell_price_pos_20_min=float(config.range_t_sell_price_pos_20_min),
        range_t_buy_ret_1_max=float(config.range_t_buy_ret_1_max),
        range_t_buy_price_pos_20_max=float(config.range_t_buy_price_pos_20_max),
        use_tradeability_guard=bool(config.use_tradeability_guard),
        tradeability_limit_tolerance=float(config.tradeability_limit_tolerance),
        tradeability_min_volume=float(config.tradeability_min_volume),
        limit_rule_file=str(config.limit_rule_file),
        use_index_constituent_guard=bool(config.use_index_constituent_guard),
        index_constituent_file=str(config.index_constituent_file),
        index_constituent_symbol=str(config.index_constituent_symbol),
        use_margin_features=config.use_margin_features,
        margin_market_file=config.margin_market_file,
        margin_stock_file=config.margin_stock_file,
        use_state_engine=bool(use_state_engine),
    )


def _metric_delta(new: float, old: float) -> float:
    if pd.isna(new) or pd.isna(old):
        return np.nan
    return float(new - old)


def _pick_target_metric(metrics: List[BacktestMetrics], target_years: int) -> BacktestMetrics | None:
    labels = [
        f"融合策略-近{int(target_years)}年",
        "融合策略-全样本",
        f"近{int(target_years)}年",
        "全样本",
    ]
    for label in labels:
        for item in metrics:
            if item.label == label:
                return item
    return metrics[0] if metrics else None


def _strategy_objective(
    metrics: BacktestMetrics | None,
    *,
    turnover_penalty: float,
    drawdown_penalty: float,
) -> float:
    if metrics is None:
        return float("-inf")
    if pd.isna(metrics.excess_annual_return):
        return float("-inf")
    annual_turnover = float(max(0.0, metrics.annual_turnover)) if not pd.isna(metrics.annual_turnover) else 0.0
    max_dd = float(abs(metrics.max_drawdown)) if not pd.isna(metrics.max_drawdown) else 0.0
    return float(
        float(metrics.excess_annual_return)
        - float(turnover_penalty) * annual_turnover
        - float(drawdown_penalty) * max_dd
    )


def _optimize_strategy_selection(
    *,
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
    news_items_train: List[NewsItem],
) -> _StrategySelection:
    objective_text = (
        f"score = excess_annual_return - {float(config.optimizer_turnover_penalty):.4f}*annual_turnover "
        f"- {float(config.optimizer_drawdown_penalty):.3f}*abs(max_drawdown)"
    )
    target_metric_label = f"融合策略-近{int(config.optimizer_target_years)}年"
    baseline = _StrategySelection(
        retrain_days=int(config.backtest_retrain_days),
        weight_threshold=float(config.backtest_weight_threshold),
        max_positions=int(config.max_positions),
        market_news_strength=float(config.market_news_strength),
        stock_news_strength=float(config.stock_news_strength),
        objective_text=objective_text,
        target_metric_label=target_metric_label,
        selected_trial=None,
        trials=[],
        best_backtest=None,
    )
    if not config.use_strategy_optimizer:
        return baseline

    retrain_grid = _sanitize_int_grid(
        config.optimizer_retrain_days,
        fallback=int(config.backtest_retrain_days),
        min_value=1,
    )
    threshold_grid = _sanitize_float_grid(
        config.optimizer_weight_thresholds,
        fallback=float(config.backtest_weight_threshold),
        min_value=0.0,
    )
    max_pos_grid = _sanitize_int_grid(
        config.optimizer_max_positions,
        fallback=int(config.max_positions),
        min_value=1,
        max_value=int(config.max_positions),
    )
    market_strength_grid = _sanitize_float_grid(
        config.optimizer_market_news_strengths,
        fallback=float(config.market_news_strength),
        min_value=0.0,
    )
    stock_strength_grid = _sanitize_float_grid(
        config.optimizer_stock_news_strengths,
        fallback=float(config.stock_news_strength),
        min_value=0.0,
    )

    trials: list[StrategyTrial] = []
    best_trial: StrategyTrial | None = None
    best_backtest: BacktestResult | None = None
    best_score = float("-inf")
    total_trials = (
        len(retrain_grid)
        * len(threshold_grid)
        * len(max_pos_grid)
        * len(market_strength_grid)
        * len(stock_strength_grid)
    )
    time_budget_sec = max(0.0, float(config.optimizer_time_budget_minutes) * 60.0)
    start_ts = time.monotonic()
    budget_text = "unlimited" if time_budget_sec <= 0 else f"{time_budget_sec:.0f}s"
    print(f"[OPT] strategy search started: trials={total_trials}, budget={budget_text}")

    for trial_idx, (retrain_days, threshold, max_pos, market_strength, stock_strength) in enumerate(
        product(
            retrain_grid,
            threshold_grid,
            max_pos_grid,
            market_strength_grid,
            stock_strength_grid,
        ),
        start=1,
    ):
        elapsed = time.monotonic() - start_ts
        if time_budget_sec > 0 and elapsed >= time_budget_sec:
            print(
                f"[OPT] time budget reached at {elapsed:.1f}s, stop search ({trial_idx - 1}/{total_trials} trials)."
            )
            break
        print(
            f"[OPT] trial {trial_idx}/{total_trials} "
            f"(retrain={int(retrain_days)}, threshold={float(threshold):.2f}, "
            f"max_pos={int(max_pos)}, m_news={float(market_strength):.2f}, s_news={float(stock_strength):.2f})"
        )
        try:
            backtest = _run_daily_backtest(
                config=config,
                market_security=market_security,
                stocks=stocks,
                news_items_train=news_items_train,
                retrain_days=int(retrain_days),
                weight_threshold=float(threshold),
                max_positions=int(max_pos),
                market_news_strength=float(market_strength),
                stock_news_strength=float(stock_strength),
                max_trades_per_stock_per_day=int(config.max_trades_per_stock_per_day),
                max_trades_per_stock_per_week=int(config.max_trades_per_stock_per_week),
            )
        except Exception as exc:
            print(f"[OPT] trial {trial_idx}/{total_trials} failed: {exc}")
            continue

        metric = _pick_target_metric(backtest.metrics, target_years=int(config.optimizer_target_years))
        score = _strategy_objective(
            metric,
            turnover_penalty=float(config.optimizer_turnover_penalty),
            drawdown_penalty=float(config.optimizer_drawdown_penalty),
        )
        if metric is None:
            print(f"[OPT] trial {trial_idx}/{total_trials} skipped: missing target metric")
            continue
        print(
            f"[OPT] trial {trial_idx}/{total_trials} result: "
            f"objective={float(score):.4f}, annual={float(metric.annual_return):.2%}, "
            f"excess={float(metric.excess_annual_return):.2%}, max_dd={float(metric.max_drawdown):.2%}"
        )
        trial = StrategyTrial(
            rank=0,
            metric_label=metric.label,
            retrain_days=int(retrain_days),
            weight_threshold=float(threshold),
            max_positions=int(max_pos),
            market_news_strength=float(market_strength),
            stock_news_strength=float(stock_strength),
            objective_score=float(score),
            annual_return=float(metric.annual_return),
            excess_annual_return=float(metric.excess_annual_return),
            max_drawdown=float(metric.max_drawdown),
            annual_turnover=float(metric.annual_turnover),
            total_cost=float(metric.total_cost),
            sharpe=float(metric.sharpe),
            avg_trades_per_stock_per_week=float(metric.avg_trades_per_stock_per_week),
        )
        trials.append(trial)
        if score > best_score:
            best_score = float(score)
            best_trial = trial
            best_backtest = backtest
            print(
                f"[OPT] new best at trial {trial_idx}/{total_trials}: "
                f"score={best_score:.4f}, label={metric.label}"
            )

    if not trials:
        return baseline

    trials.sort(key=lambda x: x.objective_score, reverse=True)
    top_n = max(1, int(config.optimizer_top_trials))
    trials = trials[:top_n]
    for idx, trial in enumerate(trials, start=1):
        trial.rank = int(idx)

    if best_trial is None:
        return baseline

    return _StrategySelection(
        retrain_days=int(best_trial.retrain_days),
        weight_threshold=float(best_trial.weight_threshold),
        max_positions=int(best_trial.max_positions),
        market_news_strength=float(best_trial.market_news_strength),
        stock_news_strength=float(best_trial.stock_news_strength),
        objective_text=objective_text,
        target_metric_label=target_metric_label,
        selected_trial=best_trial,
        trials=trials,
        best_backtest=best_backtest,
    )


def generate_daily_fusion(
    config: DailyConfig,
    market_security: Security,
    stocks: List[Security],
    sector_map: Dict[str, str],
    current_holdings: List[Security] | None = None,
) -> DailyFusionResult:
    forecast = generate_forecast(
        config=ForecastConfig(
            source=config.source,
            data_dir=config.data_dir,
            start=config.start,
            end=config.end,
            min_train_days=config.min_train_days,
            step_days=config.step_days,
            l2=config.l2,
            max_positions=config.max_positions,
            use_margin_features=config.use_margin_features,
            margin_market_file=config.margin_market_file,
            margin_stock_file=config.margin_stock_file,
        ),
        market_security=market_security,
        stocks=stocks,
    )
    market_forecast = forecast.market_forecast
    stock_rows = forecast.stock_rows

    as_of_date = pd.Timestamp(market_forecast.latest_date).normalize()
    if config.report_date:
        as_of_date = pd.Timestamp(config.report_date).normalize()

    news_items_live = load_news_items(
        config.news_file,
        as_of_date=as_of_date,
        lookback_days=config.news_lookback_days,
    )
    history_lookback_days = max(
        config.news_lookback_days,
        config.learned_news_lookback_days,
        int(max(30, (as_of_date - pd.Timestamp(config.start)).days + 7)),
    )
    news_items_train = load_news_items(
        config.news_file,
        as_of_date=as_of_date,
        lookback_days=history_lookback_days,
    )

    market_feat = pd.DataFrame()
    market_feature_cols = MARKET_FEATURE_COLUMNS
    stock_feature_cols_map: Dict[str, List[str]] = {}
    stock_feature_frames: Dict[str, pd.DataFrame] = {}
    if config.use_learned_news_fusion:
        market_feat, market_feature_cols, stock_feature_cols_map, stock_feature_frames = _prepare_learning_frames(
            config=config,
            market_security=market_security,
            stocks=stocks,
        )

    strategy_selection = _optimize_strategy_selection(
        config=config,
        market_security=market_security,
        stocks=stocks,
        news_items_train=news_items_train,
    )
    market_news_strength_live = float(strategy_selection.market_news_strength)
    stock_news_strength_live = float(strategy_selection.stock_news_strength)
    weight_threshold_live = float(strategy_selection.weight_threshold)
    max_positions_live = int(strategy_selection.max_positions)

    market_short_pred = predict_with_learned_fusion(
        enabled=config.use_learned_news_fusion,
        base_prob=market_forecast.short_prob,
        target="MARKET",
        horizon="short",
        feature_frame=market_feat,
        feature_cols=market_feature_cols,
        target_col="mkt_target_1d_up",
        news_items_train=news_items_train,
        news_items_live=news_items_live,
        as_of_date=as_of_date,
        half_life_days=config.news_half_life_days,
        min_train_days=config.min_train_days,
        step_days=config.step_days,
        quant_l2=config.l2,
        news_l2=config.learned_news_l2,
        fusion_l2=config.learned_fusion_l2,
        min_samples=config.learned_news_min_samples,
        holdout_ratio=config.learned_holdout_ratio,
        fallback_strength=market_news_strength_live,
    )
    market_mid_pred = predict_with_learned_fusion(
        enabled=config.use_learned_news_fusion,
        base_prob=market_forecast.mid_prob,
        target="MARKET",
        horizon="mid",
        feature_frame=market_feat,
        feature_cols=market_feature_cols,
        target_col="mkt_target_20d_up",
        news_items_train=news_items_train,
        news_items_live=news_items_live,
        as_of_date=as_of_date,
        half_life_days=config.news_half_life_days,
        min_train_days=config.min_train_days,
        step_days=config.step_days,
        quant_l2=config.l2,
        news_l2=config.learned_news_l2,
        fusion_l2=config.learned_fusion_l2,
        min_samples=config.learned_news_min_samples,
        holdout_ratio=config.learned_holdout_ratio,
        fallback_strength=market_news_strength_live,
    )

    market_final_short = market_short_pred.final_prob
    market_final_mid = market_mid_pred.final_prob

    blended_rows, stock_diagnostics = _blend_stock_rows(
        rows=stock_rows,
        news_items_train=news_items_train,
        news_items_live=news_items_live,
        as_of_date=as_of_date,
        config=config,
        stock_news_strength=stock_news_strength_live,
        stock_feature_cols_map=stock_feature_cols_map,
        stock_feature_frames=stock_feature_frames,
    )
    state = decide_market_state(
        market_final_short,
        market_final_mid,
        base_weight_threshold=weight_threshold_live,
        base_max_positions=max_positions_live,
        base_max_trades_per_stock_per_day=int(config.max_trades_per_stock_per_day),
        base_max_trades_per_stock_per_week=int(config.max_trades_per_stock_per_week),
    )
    total = float(state.exposure_cap)
    weights = allocate_weights(
        [row.final_score for row in blended_rows],
        total_exposure=total,
        threshold=float(state.weight_threshold),
        max_positions=int(state.max_positions),
    )
    for row, weight in zip(blended_rows, weights):
        row.suggested_weight = float(weight)
    blended_rows.sort(key=lambda x: x.final_score, reverse=True)
    learning_diagnostics = [market_short_pred.diagnostics, market_mid_pred.diagnostics] + stock_diagnostics
    holdings_for_plan = current_holdings or []
    trade_plan_basis, trade_plan_nav, trade_actions = _build_trade_actions(
        config=config,
        as_of_date=as_of_date,
        blended_rows=blended_rows,
        current_holdings=holdings_for_plan,
    )

    snapshot = build_latest_snapshot(
        source=config.source,
        data_dir=config.data_dir,
        start=config.start,
        end=config.end,
        stocks=stocks,
        sector_map=sector_map,
    )
    effect_summary = compute_effect_summary(snapshot)
    sector_table = compute_sector_table(snapshot)
    backtest = strategy_selection.best_backtest
    if backtest is None:
        backtest = _run_daily_backtest(
            config=config,
            market_security=market_security,
            stocks=stocks,
            news_items_train=news_items_train,
            retrain_days=strategy_selection.retrain_days,
            weight_threshold=strategy_selection.weight_threshold,
            max_positions=strategy_selection.max_positions,
            market_news_strength=strategy_selection.market_news_strength,
            stock_news_strength=strategy_selection.stock_news_strength,
            max_trades_per_stock_per_day=int(state.max_trades_per_stock_per_day),
            max_trades_per_stock_per_week=int(state.max_trades_per_stock_per_week),
            use_state_engine=True,
        )

    acceptance_enabled = bool(config.enable_acceptance_checks)
    acceptance_ab_pass = False
    acceptance_constraints_pass = False
    acceptance_summary = "Acceptance checks disabled."
    acceptance_delta_excess_annual_return = np.nan
    acceptance_delta_max_drawdown = np.nan
    acceptance_delta_annual_turnover = np.nan
    acceptance_limit_violations = 0
    acceptance_oversell_violations = 0

    if acceptance_enabled:
        baseline = _run_daily_backtest(
            config=config,
            market_security=market_security,
            stocks=stocks,
            news_items_train=news_items_train,
            retrain_days=strategy_selection.retrain_days,
            weight_threshold=strategy_selection.weight_threshold,
            max_positions=strategy_selection.max_positions,
            market_news_strength=strategy_selection.market_news_strength,
            stock_news_strength=strategy_selection.stock_news_strength,
            max_trades_per_stock_per_day=int(state.max_trades_per_stock_per_day),
            max_trades_per_stock_per_week=int(state.max_trades_per_stock_per_week),
            use_state_engine=False,
        )
        metric_new = _pick_target_metric(backtest.metrics, target_years=int(config.acceptance_target_years))
        metric_old = _pick_target_metric(baseline.metrics, target_years=int(config.acceptance_target_years))
        if metric_new is not None and metric_old is not None:
            acceptance_delta_excess_annual_return = _metric_delta(
                float(metric_new.excess_annual_return), float(metric_old.excess_annual_return)
            )
            acceptance_delta_max_drawdown = _metric_delta(float(metric_new.max_drawdown), float(metric_old.max_drawdown))
            acceptance_delta_annual_turnover = _metric_delta(
                float(metric_new.annual_turnover), float(metric_old.annual_turnover)
            )
            dd_not_worse = (
                not pd.isna(acceptance_delta_max_drawdown)
                and float(acceptance_delta_max_drawdown) >= -1e-9
            )
            turnover_better = (
                not pd.isna(acceptance_delta_annual_turnover)
                and float(acceptance_delta_annual_turnover) <= 1e-9
            )
            acceptance_ab_pass = bool(dd_not_worse and turnover_better)

        audit = backtest.audit
        acceptance_limit_violations = int(audit.get("limit_violations_fused", 0))
        acceptance_oversell_violations = int(audit.get("oversell_violations_fused", 0))
        blocked_total = int(audit.get("blocked_total_fused", 0))
        suspended_symbol_days = int(audit.get("suspended_symbol_days", 0))
        non_member_symbol_days = int(audit.get("non_member_symbol_days", 0))
        no_member_snapshot_days = int(audit.get("no_member_snapshot_days", 0))
        acceptance_constraints_pass = (
            int(acceptance_oversell_violations) == 0
        )
        acceptance_summary = (
            f"A/B pass={acceptance_ab_pass} "
            f"(delta_excess={acceptance_delta_excess_annual_return:.2%}, "
            f"delta_max_dd={acceptance_delta_max_drawdown:.2%}, "
            f"delta_turnover={acceptance_delta_annual_turnover:.2%}); "
            f"constraints pass={acceptance_constraints_pass} "
            f"(weekly_overflow={acceptance_limit_violations}, oversell_violations={acceptance_oversell_violations}, "
            f"blocked_by_tradability={blocked_total}, suspended_symbol_days={suspended_symbol_days}, "
            f"non_member_symbol_days={non_member_symbol_days}, no_member_snapshot_days={no_member_snapshot_days})"
        )

    return DailyFusionResult(
        as_of_date=as_of_date,
        source=config.source,
        market_forecast=market_forecast,
        market_news_short_prob=market_short_pred.news_prob,
        market_news_mid_prob=market_mid_pred.news_prob,
        market_final_short=market_final_short,
        market_final_mid=market_final_mid,
        market_state_code=state.state_code,
        market_state_label=state.state_label,
        strategy_template=state.strategy_template,
        intraday_t_level=state.intraday_t_level,
        effective_total_exposure=float(state.exposure_cap),
        effective_weight_threshold=float(state.weight_threshold),
        effective_max_positions=int(state.max_positions),
        effective_max_trades_per_stock_per_day=int(state.max_trades_per_stock_per_day),
        effective_max_trades_per_stock_per_week=int(state.max_trades_per_stock_per_week),
        market_short_sent=market_short_pred.sentiment,
        market_mid_sent=market_mid_pred.sentiment,
        market_fusion_mode_short=market_short_pred.mode,
        market_fusion_mode_mid=market_mid_pred.mode,
        blended_rows=blended_rows,
        learning_diagnostics=learning_diagnostics,
        effect_summary=effect_summary,
        sector_table=sector_table,
        backtest_metrics=backtest.metrics,
        backtest_curve=backtest.curve_frame,
        strategy_objective_text=strategy_selection.objective_text,
        strategy_target_metric_label=strategy_selection.target_metric_label,
        strategy_selected=strategy_selection.selected_trial,
        strategy_trials=strategy_selection.trials,
        acceptance_enabled=acceptance_enabled,
        acceptance_ab_pass=acceptance_ab_pass,
        acceptance_constraints_pass=acceptance_constraints_pass,
        acceptance_summary=acceptance_summary,
        acceptance_delta_excess_annual_return=float(acceptance_delta_excess_annual_return),
        acceptance_delta_max_drawdown=float(acceptance_delta_max_drawdown),
        acceptance_delta_annual_turnover=float(acceptance_delta_annual_turnover),
        acceptance_limit_violations=int(acceptance_limit_violations),
        acceptance_oversell_violations=int(acceptance_oversell_violations),
        trade_plan_basis=str(trade_plan_basis),
        trade_plan_nav=float(trade_plan_nav),
        trade_plan_lot_size=int(config.trade_lot_size),
        trade_actions=trade_actions,
        news_items_count=len(news_items_live),
        news_items=news_items_live,
    )
