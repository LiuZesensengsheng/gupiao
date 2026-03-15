from __future__ import annotations

from pathlib import Path
from typing import Any

from src.application.config import DailyConfig, DiscoverConfig, ForecastConfig


def run_daily(settings: dict[str, Any], *, dependencies: Any) -> int:
    deps = dependencies
    deps.set_tushare_token(settings.get("tushare_token", ""))
    market_security, stocks, sector_map = deps.load_watchlist(settings["watchlist"])
    current_holdings = list(stocks)
    universe_file = str(settings.get("universe_file", "")).strip()
    if not universe_file:
        raise ValueError(
            "Daily requires `--universe-file` (large pool). "
            "The 5-stock watchlist fallback has been disabled."
        )
    universe = deps.build_candidate_universe(
        source=settings["source"],
        data_dir=settings["data_dir"],
        universe_file=universe_file,
        candidate_limit=max(5, int(settings.get("universe_limit", 500))),
        exclude_symbols=[market_security.symbol],
    )
    if not universe.rows:
        raise ValueError(f"Daily universe is empty: {universe_file}")
    stocks = universe.rows
    sector_map = {deps.normalize_symbol(s.symbol).symbol: (s.sector or "鍏朵粬") for s in stocks}
    print(f"[OK] Daily universe source: {universe.source_label}")
    print(f"[OK] Daily universe size: {len(stocks)}")
    for warning in universe.warnings:
        print(f"[WARN] {warning}")

    config = DailyConfig(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        min_train_days=settings["min_train_days"],
        step_days=settings["step_days"],
        l2=settings["l2"],
        max_positions=int(settings["max_positions"]),
        use_margin_features=deps._parse_bool(settings["use_margin_features"]),
        margin_market_file=settings["margin_market_file"],
        margin_stock_file=settings["margin_stock_file"],
        use_us_index_context=deps._parse_bool(settings["use_us_index_context"]),
        us_index_source=str(settings["us_index_source"]),
        positions_file=str(settings["positions_file"]).strip(),
        portfolio_nav=max(0.0, float(settings["portfolio_nav"])),
        trade_lot_size=max(1, int(settings["trade_lot_size"])),
        news_file=settings["news_file"],
        news_lookback_days=settings["news_lookback_days"],
        learned_news_lookback_days=int(settings["learned_news_lookback_days"]),
        news_half_life_days=settings["news_half_life_days"],
        market_news_strength=settings["market_news_strength"],
        stock_news_strength=settings["stock_news_strength"],
        use_learned_news_fusion=deps._parse_bool(settings["use_learned_news_fusion"]),
        learned_news_min_samples=int(settings["learned_news_min_samples"]),
        learned_holdout_ratio=float(settings["learned_holdout_ratio"]),
        learned_news_l2=float(settings["learned_news_l2"]),
        learned_fusion_l2=float(settings["learned_fusion_l2"]),
        backtest_years=deps._parse_years(settings["backtest_years"]),
        backtest_retrain_days=int(settings["backtest_retrain_days"]),
        backtest_weight_threshold=float(settings["backtest_weight_threshold"]),
        backtest_time_budget_minutes=float(settings["backtest_time_budget_minutes"]),
        commission_bps=float(settings["commission_bps"]),
        slippage_bps=float(settings["slippage_bps"]),
        use_turnover_control=deps._parse_bool(settings["use_turnover_control"]),
        max_trades_per_stock_per_day=max(1, int(settings["max_trades_per_stock_per_day"])),
        max_trades_per_stock_per_week=max(1, int(settings["max_trades_per_stock_per_week"])),
        min_weight_change_to_trade=max(0.0, float(settings["min_weight_change_to_trade"])),
        range_t_sell_ret_1_min=float(settings["range_t_sell_ret_1_min"]),
        range_t_sell_price_pos_20_min=float(settings["range_t_sell_price_pos_20_min"]),
        range_t_buy_ret_1_max=float(settings["range_t_buy_ret_1_max"]),
        range_t_buy_price_pos_20_max=float(settings["range_t_buy_price_pos_20_max"]),
        use_tradeability_guard=deps._parse_bool(settings["use_tradeability_guard"]),
        tradeability_limit_tolerance=max(0.0, float(settings["tradeability_limit_tolerance"])),
        tradeability_min_volume=max(0.0, float(settings["tradeability_min_volume"])),
        limit_rule_file=str(settings["limit_rule_file"]).strip(),
        use_index_constituent_guard=deps._parse_bool(settings["use_index_constituent_guard"]),
        index_constituent_file=str(settings["index_constituent_file"]).strip(),
        index_constituent_symbol=str(settings["index_constituent_symbol"]).strip() or "000300.SH",
        enable_acceptance_checks=deps._parse_bool(settings["enable_acceptance_checks"]),
        acceptance_target_years=max(1, int(settings["acceptance_target_years"])),
        use_strategy_optimizer=deps._parse_bool(settings["use_strategy_optimizer"]),
        optimizer_retrain_days=deps._parse_int_list(settings["optimizer_retrain_days"], min_value=1) or (20, 40),
        optimizer_weight_thresholds=deps._parse_float_list(settings["optimizer_weight_thresholds"], min_value=0.0) or (0.50, 0.60),
        optimizer_max_positions=deps._parse_int_list(settings["optimizer_max_positions"], min_value=1) or (3, 5),
        optimizer_market_news_strengths=deps._parse_float_list(settings["optimizer_market_news_strengths"], min_value=0.0) or (0.8, 1.0),
        optimizer_stock_news_strengths=deps._parse_float_list(settings["optimizer_stock_news_strengths"], min_value=0.0) or (1.0, 1.2),
        optimizer_turnover_penalty=float(settings["optimizer_turnover_penalty"]),
        optimizer_drawdown_penalty=float(settings["optimizer_drawdown_penalty"]),
        optimizer_target_years=max(1, int(settings["optimizer_target_years"])),
        optimizer_top_trials=max(1, int(settings["optimizer_top_trials"])),
        optimizer_time_budget_minutes=float(settings["optimizer_time_budget_minutes"]),
        report_date=settings["report_date"],
    )

    result = deps.generate_daily_fusion(
        config=config,
        market_security=market_security,
        stocks=stocks,
        sector_map=sector_map,
        current_holdings=current_holdings,
    )
    report_path = deps.write_daily_report(settings["report"], result)
    print(f"[OK] Daily report generated: {report_path.resolve()}")

    if settings["dashboard"].strip():
        dashboard_path = deps.write_daily_dashboard(settings["dashboard"], result)
        print(f"[OK] Daily dashboard generated: {dashboard_path.resolve()}")
    return 0


def run_forecast(settings: dict[str, Any], *, dependencies: Any) -> int:
    deps = dependencies
    deps.set_tushare_token(settings.get("tushare_token", ""))
    market_security, stocks, _ = deps.load_watchlist(settings["watchlist"])
    config = ForecastConfig(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        min_train_days=settings["min_train_days"],
        step_days=settings["step_days"],
        l2=settings["l2"],
        max_positions=int(settings["max_positions"]),
        use_margin_features=deps._parse_bool(settings["use_margin_features"]),
        margin_market_file=settings["margin_market_file"],
        margin_stock_file=settings["margin_stock_file"],
        use_us_index_context=deps._parse_bool(settings["use_us_index_context"]),
        us_index_source=str(settings["us_index_source"]),
    )
    result = deps.generate_forecast(config=config, market_security=market_security, stocks=stocks)
    path = deps.write_forecast_report(settings["report"], result.market_forecast, result.stock_rows)
    print(f"[OK] Report generated: {path.resolve()}")
    return 0


def run_discover(settings: dict[str, Any], *, dependencies: Any) -> int:
    deps = dependencies
    deps.set_tushare_token(settings.get("tushare_token", ""))
    market_security, stocks, _ = deps.load_watchlist(settings["watchlist"])
    config = DiscoverConfig(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        min_train_days=settings["min_train_days"],
        step_days=settings["step_days"],
        l2=settings["l2"],
        max_positions=int(settings["max_positions"]),
        use_margin_features=deps._parse_bool(settings["use_margin_features"]),
        margin_market_file=settings["margin_market_file"],
        margin_stock_file=settings["margin_stock_file"],
        use_us_index_context=deps._parse_bool(settings["use_us_index_context"]),
        us_index_source=str(settings["us_index_source"]),
        universe_file=settings["universe_file"],
        candidate_limit=int(settings["candidate_limit"]),
        top_k=int(settings["top_k"]),
        exclude_watchlist=deps._parse_bool(settings["exclude_watchlist"]),
    )
    result = deps.generate_discovery(config=config, market_security=market_security, watchlist_stocks=stocks)
    path = deps.write_discovery_report(settings["report"], result)
    print(f"[OK] Discovery report generated: {path.resolve()}")
    return 0


def run_sync_data(settings: dict[str, Any], *, dependencies: Any) -> int:
    deps = dependencies
    deps.set_tushare_token(settings.get("tushare_token", ""))
    result = deps.sync_market_data(
        source=settings["source"],
        data_dir=settings["data_dir"],
        start=settings["start"],
        end=settings["end"],
        universe_size=int(settings["universe_size"]),
        universe_file=settings["universe_file"],
        include_indices=deps._parse_bool(settings["include_indices"]),
        force_refresh=deps._parse_bool(settings["force_refresh"]),
        sleep_ms=int(settings["sleep_ms"]),
        parallel_workers=max(1, int(settings.get("parallel_workers", 1) or 1)),
        max_failures=int(settings["max_failures"]),
        write_universe_file=settings["write_universe_file"],
        universe_min_amount=max(0.0, float(settings["universe_min_amount"])),
        universe_exclude_st=deps._parse_bool(settings["universe_exclude_st"]),
    )
    if result.resumed:
        print(f"[OK] Resumed from checkpoint: {result.resume_completed} symbols already completed")
    print(f"[OK] Universe source: {result.universe_source}")
    print(f"[OK] Universe size: {result.universe_size} (requested {result.requested_universe_size})")
    print(f"[OK] Downloaded: {result.downloaded}, skipped: {result.skipped}, failed: {result.failed}, attempted: {result.attempted}")
    if result.checkpoint_file:
        print(f"[OK] Resume checkpoint saved: {Path(result.checkpoint_file).resolve()}")
    if result.universe_file:
        print(f"[OK] Universe file written: {Path(result.universe_file).resolve()}")
    if result.failed_symbols:
        print(f"[WARN] Failed symbols (first 20): {', '.join(result.failed_symbols)}")
    if result.downloaded <= 0 and result.skipped <= 0:
        print("[ERROR] No symbols were synced.")
        return 2
    return 0


def run_sync_margin(settings: dict[str, Any], *, dependencies: Any) -> int:
    deps = dependencies
    deps.set_tushare_token(settings.get("tushare_token", ""))
    symbols = deps._parse_symbol_list(settings.get("symbols", ""))
    if not symbols:
        universe_file = str(settings.get("universe_file", "")).strip()
        if universe_file:
            rows = deps._load_discovery_universe_file(universe_file, enrich_metadata=False)
            limit = int(settings.get("universe_limit", 0) or 0)
            if limit > 0:
                rows = rows[:limit]
            symbols = [deps.normalize_symbol(sec.symbol).symbol for sec in rows]
    if not symbols:
        _, stocks, _ = deps.load_watchlist(settings["watchlist"])
        symbols = [deps.normalize_symbol(sec.symbol).symbol for sec in stocks]
    if not symbols:
        print("[ERROR] No valid symbols for margin sync. Set --symbols, --universe-file, or check watchlist.")
        return 2

    result = deps.sync_margin_data(
        source=settings["source"],
        symbols=symbols,
        start=settings["start"],
        end=settings["end"],
        market_out=settings["margin_market_file"],
        stock_out=settings["margin_stock_file"],
        tushare_token=str(settings.get("tushare_token", "")),
        sleep_ms=int(settings["sleep_ms"]),
    )
    print(f"[OK] Margin source used: {result.source_used}")
    print(f"[OK] Symbols: {len(symbols)}")
    print(f"[OK] Market rows: {result.market_rows} -> {Path(result.market_path).resolve()}")
    print(f"[OK] Stock rows: {result.stock_rows} -> {Path(result.stock_path).resolve()}")
    if result.notes:
        for note in result.notes:
            print(f"[WARN] {note}")
    return 0
