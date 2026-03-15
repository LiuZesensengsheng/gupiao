from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.application.config import DailyConfig, DiscoverConfig, ForecastConfig
from src.application.legacy_strategy_selection import (
    metric_delta as _metric_delta,
    optimize_strategy_selection as _optimize_strategy_selection,
    pick_target_metric as _pick_target_metric,
    run_daily_backtest as _run_daily_backtest,
)
from src.application.legacy_trade_planning import build_trade_actions as _build_trade_actions
from src.application.legacy_use_case_models import DailyFusionResult, DiscoveryResult, ForecastResult
from src.domain.entities import BlendedRow, DiscoveryRow, ForecastRow, FusionDiagnostics, NewsItem, Security
from src.domain.policies import allocate_weights, blend_horizon_score, decide_market_state
from src.domain.symbols import normalize_symbol
from src.infrastructure.discovery import build_candidate_universe, compute_volume_risk
from src.infrastructure.effect_analysis import build_latest_snapshot, compute_effect_summary, compute_sector_table
from src.infrastructure.features import MARKET_FEATURE_COLUMNS, make_market_feature_frame, make_stock_feature_frame, stock_feature_columns
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import load_symbol_daily
from src.infrastructure.news_fusion import predict_with_learned_fusion
from src.infrastructure.news_repository import load_news_items


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
        use_us_index_context=config.use_us_index_context,
        us_index_source=config.us_index_source,
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
        exclude_symbols.extend(security.symbol for security in watchlist_stocks)

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
            use_us_index_context=config.use_us_index_context,
            us_index_source=config.us_index_source,
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

    rows: List[DiscoveryRow] = []
    top_rows = sorted(forecast.stock_rows, key=lambda item: item.score, reverse=True)[: max(1, int(config.top_k))]
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
        rows.append(
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
        rows=rows,
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
        use_us_index_context=config.use_us_index_context,
        us_index_source=config.us_index_source,
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
                    volume_risk_note = f"楂樹綅宸ㄩ噺澶ч槾绾?5鏃ュ唴), 閲忚兘姣?{vol_ratio:.2f}, 浣嶇疆={price_pos:.2f}"
                else:
                    volume_risk_note = f"閲忚兘姣?{vol_ratio:.2f}, 浣嶇疆={price_pos:.2f}"
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
            use_us_index_context=config.use_us_index_context,
            us_index_source=config.us_index_source,
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
    weights = allocate_weights(
        [row.final_score for row in blended_rows],
        total_exposure=float(state.exposure_cap),
        threshold=float(state.weight_threshold),
        max_positions=int(state.max_positions),
    )
    for row, weight in zip(blended_rows, weights):
        row.suggested_weight = float(weight)
    blended_rows.sort(key=lambda item: item.final_score, reverse=True)

    learning_diagnostics = [market_short_pred.diagnostics, market_mid_pred.diagnostics] + stock_diagnostics
    trade_plan_basis, trade_plan_nav, trade_actions = _build_trade_actions(
        config=config,
        as_of_date=as_of_date,
        blended_rows=blended_rows,
        current_holdings=current_holdings or [],
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
                float(metric_new.excess_annual_return),
                float(metric_old.excess_annual_return),
            )
            acceptance_delta_max_drawdown = _metric_delta(
                float(metric_new.max_drawdown),
                float(metric_old.max_drawdown),
            )
            acceptance_delta_annual_turnover = _metric_delta(
                float(metric_new.annual_turnover),
                float(metric_old.annual_turnover),
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
        acceptance_constraints_pass = int(acceptance_oversell_violations) == 0
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
