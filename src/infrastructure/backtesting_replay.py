from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src.domain.entities import NewsItem, Security, SentimentAggregate
from src.domain.news import aggregate_sentiment
from src.domain.policies import allocate_weights, blend_horizon_score, decide_market_state, target_exposure
from src.domain.symbols import normalize_symbol
from src.infrastructure.backtesting import (
    BacktestResult,
    PreparedBacktestDay,
    PreparedBacktestTape,
    _BlockFuser,
    _apply_block_fuser,
    _apply_range_t_whitelist,
    _apply_tradeability_guard,
    _apply_turnover_control,
    _build_tradeability_flags_from_arrays,
    _empty_backtest_result,
    _finalize_backtest_result,
    _fit_block_fuser,
    _load_index_constituent_book,
    _load_limit_rule_book,
    _members_asof,
    _normalize_window_years,
    _with_market_forward_return,
)
from src.infrastructure.features import MARKET_FEATURE_COLUMNS, make_stock_feature_frame, stock_feature_columns
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import DataError, load_symbol_daily
from src.infrastructure.modeling import LogisticBinaryModel


def _prepare_day_snapshot(
    *,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    symbols: list[str],
    stock_frames: Dict[str, pd.DataFrame],
    stock_cols_map: Dict[str, list[str]],
    stock_models: Dict[str, tuple[LogisticBinaryModel, LogisticBinaryModel]],
    stock_fusers: Dict[str, Dict[str, _BlockFuser]],
    stock_news_strength: float,
    sentiment_getter,
    benchmark_ret: float,
    market_short_quant: float,
    market_mid_quant: float,
    market_short_fused: float,
    market_mid_fused: float,
    news_enabled: bool,
    use_index_constituent_guard: bool,
    index_book,
) -> PreparedBacktestDay:
    quant_scores = np.zeros(len(symbols), dtype=float)
    fused_scores = np.zeros(len(symbols), dtype=float)
    fwd_ret_1 = np.zeros(len(symbols), dtype=float)
    ret_1_arr = np.full(len(symbols), np.nan, dtype=float)
    price_pos_20_arr = np.full(len(symbols), np.nan, dtype=float)
    open_arr = np.full(len(symbols), np.nan, dtype=float)
    high_arr = np.full(len(symbols), np.nan, dtype=float)
    low_arr = np.full(len(symbols), np.nan, dtype=float)
    close_arr = np.full(len(symbols), np.nan, dtype=float)
    volume_arr = np.full(len(symbols), np.nan, dtype=float)

    for idx, symbol in enumerate(symbols):
        models = stock_models.get(symbol)
        stock_cols = stock_cols_map.get(symbol)
        if models is None or stock_cols is None:
            continue
        row = stock_frames[symbol].loc[[date]]
        short_prob = float(models[0].predict_proba(row, stock_cols)[0])
        mid_prob = float(models[1].predict_proba(row, stock_cols)[0])
        quant_scores[idx] = float(blend_horizon_score(short_prob, mid_prob, short_weight=0.55))

        fused_short = short_prob
        fused_mid = mid_prob
        if news_enabled:
            fuser_short = stock_fusers.get(symbol, {}).get("short")
            fuser_mid = stock_fusers.get(symbol, {}).get("mid")
            if fuser_short is None:
                fuser_short = _BlockFuser(mode="rule", reason="missing", fallback_strength=float(stock_news_strength))
            if fuser_mid is None:
                fuser_mid = _BlockFuser(mode="rule", reason="missing", fallback_strength=float(stock_news_strength))
            fused_short = _apply_block_fuser(fuser_short, short_prob, date=date, target=symbol, horizon="short", sentiment_getter=sentiment_getter)
            fused_mid = _apply_block_fuser(fuser_mid, mid_prob, date=date, target=symbol, horizon="mid", sentiment_getter=sentiment_getter)
        fused_scores[idx] = float(blend_horizon_score(fused_short, fused_mid, short_weight=0.55))
        fwd_ret_1[idx] = float(row["fwd_ret_1"].iloc[0])
        ret_1_arr[idx] = float(row["ret_1"].iloc[0])
        price_pos_20_arr[idx] = float(row["price_pos_20"].iloc[0])
        open_arr[idx] = float(row["open"].iloc[0])
        high_arr[idx] = float(row["high"].iloc[0])
        low_arr[idx] = float(row["low"].iloc[0])
        close_arr[idx] = float(row["close"].iloc[0])
        volume_arr[idx] = float(row["volume"].iloc[0])

    member_mask = np.ones(len(symbols), dtype=bool)
    missing_member_snapshot = False
    if use_index_constituent_guard and index_book is not None:
        members = _members_asof(index_book, date)
        if members is None:
            member_mask = np.zeros(len(symbols), dtype=bool)
            missing_member_snapshot = True
        else:
            member_mask = np.array([symbol in members for symbol in symbols], dtype=bool)

    return PreparedBacktestDay(
        date=pd.Timestamp(date),
        next_date=pd.Timestamp(next_date),
        benchmark_ret=float(benchmark_ret),
        market_short_prob_quant=float(market_short_quant),
        market_mid_prob_quant=float(market_mid_quant),
        market_short_prob_fused=float(market_short_fused),
        market_mid_prob_fused=float(market_mid_fused),
        quant_scores=quant_scores,
        fused_scores=fused_scores,
        fwd_ret_1=fwd_ret_1,
        ret_1=ret_1_arr,
        price_pos_20=price_pos_20_arr,
        open_prices=open_arr,
        high_prices=high_arr,
        low_prices=low_arr,
        close_prices=close_arr,
        volumes=volume_arr,
        member_mask=member_mask,
        missing_member_snapshot=missing_member_snapshot,
    )


def prepare_portfolio_backtest_tape(
    market_security: Security,
    stock_securities: Sequence[Security],
    source: str,
    data_dir: str,
    start: str,
    end: str,
    min_train_days: int,
    l2: float,
    retrain_days: int,
    window_years: Sequence[int],
    news_items: Sequence[NewsItem] | None = None,
    apply_news_fusion: bool = False,
    news_half_life_days: float = 10.0,
    market_news_strength: float = 0.9,
    stock_news_strength: float = 1.1,
    use_learned_news_fusion: bool = False,
    learned_news_min_samples: int = 80,
    learned_news_l2: float = 0.8,
    learned_fusion_l2: float = 0.6,
    max_runtime_seconds: float = 0.0,
    use_margin_features: bool = True,
    margin_market_file: str = "input/margin_market.csv",
    margin_stock_file: str = "input/margin_stock.csv",
    use_us_index_context: bool = False,
    us_index_source: str = "akshare",
    use_state_engine: bool = True,
    limit_rule_file: str = "",
    use_index_constituent_guard: bool = False,
    index_constituent_file: str = "",
    index_constituent_symbol: str = "000300.SH",
) -> PreparedBacktestTape:
    retrain_days = max(1, int(retrain_days))
    news_enabled = bool(apply_news_fusion)
    news_list = list(news_items or [])
    limit_rule_book = _load_limit_rule_book(limit_rule_file)
    index_book = _load_index_constituent_book(index_constituent_file, index_constituent_symbol)
    runtime_budget = max(0.0, float(max_runtime_seconds))
    start_ts = time.monotonic()
    deadline_ts = start_ts + runtime_budget if runtime_budget > 0.0 else None

    sent_cache: Dict[tuple[pd.Timestamp, str, str], SentimentAggregate] = {}

    def _sentiment_getter(date: pd.Timestamp, target: str, horizon: str) -> SentimentAggregate:
        key = (pd.Timestamp(date).normalize(), str(target), str(horizon))
        cached = sent_cache.get(key)
        if cached is not None:
            return cached
        sent = aggregate_sentiment(
            news_items=news_list,
            as_of_date=key[0],
            target=target,
            horizon=horizon,
            half_life_days=float(news_half_life_days),
        )
        sent_cache[key] = sent
        return sent

    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_frame_base = _with_market_forward_return(market_raw)
    market_context = build_market_context_features(
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        market_dates=market_frame_base["date"],
        use_margin_features=use_margin_features,
        margin_market_file=margin_market_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    market_frame = market_frame_base.merge(market_context.frame, on="date", how="left", validate="1:1")
    market_feature_cols = MARKET_FEATURE_COLUMNS + market_context.feature_columns
    market_need = market_feature_cols + ["mkt_target_1d_up", "mkt_target_20d_up", "mkt_fwd_ret_1"]
    market_valid = market_frame.dropna(subset=market_need).sort_values("date").copy().set_index("date", drop=False)
    if market_valid.empty:
        raise DataError("Backtest failed: no valid market rows.")

    stock_frames: Dict[str, pd.DataFrame] = {}
    stock_cols_map: Dict[str, list[str]] = {}
    for sec in stock_securities:
        symbol = normalize_symbol(sec.symbol).symbol
        stock_raw = load_symbol_daily(
            symbol=symbol,
            source=source,
            data_dir=data_dir,
            start=start,
            end=end,
        )
        stock_feat = make_stock_feature_frame(stock_raw, market_frame)
        stock_margin_cols: list[str] = []
        if use_margin_features:
            margin_frame, margin_cols, _ = build_stock_margin_features(
                margin_stock_file=margin_stock_file,
                symbol=symbol,
                start=start,
                end=end,
            )
            if margin_cols:
                stock_feat = stock_feat.merge(margin_frame, on="date", how="left", validate="1:1")
                stock_margin_cols = list(margin_cols)

        stock_cols = stock_feature_columns(
            extra_market_cols=market_context.feature_columns,
            extra_stock_cols=stock_margin_cols,
        )
        stock_need = stock_cols + ["target_1d_up", "target_20d_up", "fwd_ret_1"]
        valid = stock_feat.dropna(subset=stock_need).sort_values("date").copy()
        if valid.empty:
            continue
        stock_frames[symbol] = valid.set_index("date", drop=False)
        stock_cols_map[symbol] = list(stock_cols)

    if not stock_frames:
        raise DataError("Backtest failed: no valid stock rows.")

    common_dates = set(market_valid.index)
    for frame in stock_frames.values():
        common_dates &= set(frame.index)
    aligned_dates = sorted(pd.Timestamp(d) for d in common_dates)
    if len(aligned_dates) <= int(min_train_days) + 1:
        raise DataError("Backtest failed: insufficient aligned rows for training/testing.")

    total_test_days = max(0, len(aligned_dates) - int(min_train_days) - 1)
    block_starts = list(range(int(min_train_days), len(aligned_dates) - 1, retrain_days))
    total_blocks = len(block_starts)
    budget_text = "unlimited" if deadline_ts is None else f"{runtime_budget:.1f}s"
    print(f"[BT-PREP] prepare start: symbols={len(stock_securities)}, test_days={total_test_days}, blocks={total_blocks}, budget={budget_text}")

    symbols = [normalize_symbol(s.symbol).symbol for s in stock_securities if normalize_symbol(s.symbol).symbol in stock_frames]
    days: list[PreparedBacktestDay] = []
    timeout_hit = False
    prepared_days = 0

    for block_idx, block_start in enumerate(block_starts, start=1):
        if deadline_ts is not None and time.monotonic() >= deadline_ts:
            timeout_hit = True
            print(f"[BT-PREP] time budget reached before block {block_idx}/{total_blocks}, stop.")
            break
        elapsed = time.monotonic() - start_ts
        print(f"[BT-PREP] block {block_idx}/{total_blocks} start (elapsed={elapsed:.1f}s, prepared_days={prepared_days}/{total_test_days})")
        train_dates = aligned_dates[:block_start]
        train_index = pd.Index(train_dates)

        market_train = market_valid.loc[train_index]
        market_short_model = LogisticBinaryModel(l2=l2).fit(
            market_train,
            feature_cols=market_feature_cols,
            target_col="mkt_target_1d_up",
        )
        market_mid_model = LogisticBinaryModel(l2=l2).fit(
            market_train,
            feature_cols=market_feature_cols,
            target_col="mkt_target_20d_up",
        )

        stock_models: Dict[str, tuple[LogisticBinaryModel, LogisticBinaryModel]] = {}
        for symbol in symbols:
            train = stock_frames[symbol].loc[train_index]
            stock_cols = stock_cols_map.get(symbol)
            if stock_cols is None or train.empty:
                continue
            stock_models[symbol] = (
                LogisticBinaryModel(l2=l2).fit(train, feature_cols=stock_cols, target_col="target_1d_up"),
                LogisticBinaryModel(l2=l2).fit(train, feature_cols=stock_cols, target_col="target_20d_up"),
            )

        market_fusers = {
            "short": _BlockFuser(mode="rule", reason="disabled", fallback_strength=float(market_news_strength)),
            "mid": _BlockFuser(mode="rule", reason="disabled", fallback_strength=float(market_news_strength)),
        }
        stock_fusers: Dict[str, Dict[str, _BlockFuser]] = {
            symbol: {
                "short": _BlockFuser(mode="rule", reason="disabled", fallback_strength=float(stock_news_strength)),
                "mid": _BlockFuser(mode="rule", reason="disabled", fallback_strength=float(stock_news_strength)),
            }
            for symbol in symbols
        }

        if news_enabled and use_learned_news_fusion:
            train_end = pd.Timestamp(train_dates[-1]) if train_dates else pd.Timestamp(aligned_dates[0])
            news_train = [item for item in news_list if item.date <= train_end]
            if news_train:
                market_fusers["short"] = _fit_block_fuser(
                    train_frame=market_train,
                    quant_model=market_short_model,
                    feature_cols=market_feature_cols,
                    target_col="mkt_target_1d_up",
                    target="MARKET",
                    horizon="short",
                    fallback_strength=float(market_news_strength),
                    min_samples=int(learned_news_min_samples),
                    news_l2=float(learned_news_l2),
                    fusion_l2=float(learned_fusion_l2),
                    sentiment_getter=_sentiment_getter,
                )
                market_fusers["mid"] = _fit_block_fuser(
                    train_frame=market_train,
                    quant_model=market_mid_model,
                    feature_cols=market_feature_cols,
                    target_col="mkt_target_20d_up",
                    target="MARKET",
                    horizon="mid",
                    fallback_strength=float(market_news_strength),
                    min_samples=int(learned_news_min_samples),
                    news_l2=float(learned_news_l2),
                    fusion_l2=float(learned_fusion_l2),
                    sentiment_getter=_sentiment_getter,
                )
                for symbol in symbols:
                    train = stock_frames[symbol].loc[train_index]
                    models = stock_models.get(symbol)
                    stock_cols = stock_cols_map.get(symbol)
                    if stock_cols is None or train.empty or models is None:
                        continue
                    stock_fusers[symbol]["short"] = _fit_block_fuser(
                        train_frame=train,
                        quant_model=models[0],
                        feature_cols=stock_cols,
                        target_col="target_1d_up",
                        target=symbol,
                        horizon="short",
                        fallback_strength=float(stock_news_strength),
                        min_samples=int(learned_news_min_samples),
                        news_l2=float(learned_news_l2),
                        fusion_l2=float(learned_fusion_l2),
                        sentiment_getter=_sentiment_getter,
                    )
                    stock_fusers[symbol]["mid"] = _fit_block_fuser(
                        train_frame=train,
                        quant_model=models[1],
                        feature_cols=stock_cols,
                        target_col="target_20d_up",
                        target=symbol,
                        horizon="mid",
                        fallback_strength=float(stock_news_strength),
                        min_samples=int(learned_news_min_samples),
                        news_l2=float(learned_news_l2),
                        fusion_l2=float(learned_fusion_l2),
                        sentiment_getter=_sentiment_getter,
                    )

        block_end = min(block_start + retrain_days, len(aligned_dates) - 1)
        for offset in range(block_start, block_end):
            if deadline_ts is not None and time.monotonic() >= deadline_ts:
                timeout_hit = True
                print(f"[BT-PREP] time budget reached inside block {block_idx}/{total_blocks}, stop.")
                break
            date = aligned_dates[offset]
            next_date = aligned_dates[offset + 1]
            market_row = market_valid.loc[[date]]
            market_short_quant = float(market_short_model.predict_proba(market_row, market_feature_cols)[0])
            market_mid_quant = float(market_mid_model.predict_proba(market_row, market_feature_cols)[0])
            market_short_fused = market_short_quant
            market_mid_fused = market_mid_quant
            if news_enabled:
                market_short_fused = _apply_block_fuser(market_fusers["short"], market_short_quant, date=date, target="MARKET", horizon="short", sentiment_getter=_sentiment_getter)
                market_mid_fused = _apply_block_fuser(market_fusers["mid"], market_mid_quant, date=date, target="MARKET", horizon="mid", sentiment_getter=_sentiment_getter)

            days.append(
                _prepare_day_snapshot(
                    date=date,
                    next_date=next_date,
                    symbols=symbols,
                    stock_frames=stock_frames,
                    stock_cols_map=stock_cols_map,
                    stock_models=stock_models,
                    stock_fusers=stock_fusers,
                    stock_news_strength=stock_news_strength,
                    sentiment_getter=_sentiment_getter,
                    benchmark_ret=float(market_valid.loc[date, "mkt_fwd_ret_1"]),
                    market_short_quant=market_short_quant,
                    market_mid_quant=market_mid_quant,
                    market_short_fused=market_short_fused,
                    market_mid_fused=market_mid_fused,
                    news_enabled=news_enabled,
                    use_index_constituent_guard=use_index_constituent_guard,
                    index_book=index_book,
                )
            )
            prepared_days += 1
        if timeout_hit:
            break
        elapsed = time.monotonic() - start_ts
        print(f"[BT-PREP] block {block_idx}/{total_blocks} done (elapsed={elapsed:.1f}s, prepared_days={prepared_days}/{total_test_days})")

    if timeout_hit:
        elapsed = time.monotonic() - start_ts
        print(f"[BT-PREP] prepare stopped by time budget at {elapsed:.1f}s; partial tape returned.")

    return PreparedBacktestTape(
        symbols=list(symbols),
        days=[day for day in days if day is not None],
        news_enabled=news_enabled,
        use_state_engine=bool(use_state_engine),
        window_years=tuple(_normalize_window_years(window_years)),
        limit_rule_book=limit_rule_book,
        use_index_constituent_guard=bool(use_index_constituent_guard),
        index_constituent_symbol=str(index_constituent_symbol).strip().upper(),
    )


def replay_portfolio_backtest_tape(
    tape: PreparedBacktestTape,
    *,
    weight_threshold: float,
    max_positions: int,
    commission_bps: float,
    slippage_bps: float,
    max_trades_per_stock_per_day: int = 1,
    max_trades_per_stock_per_week: int = 3,
    min_weight_change_to_trade: float = 0.03,
    range_t_sell_ret_1_min: float = 0.02,
    range_t_sell_price_pos_20_min: float = 0.80,
    range_t_buy_ret_1_max: float = -0.02,
    range_t_buy_price_pos_20_max: float = 0.35,
    use_turnover_control: bool = True,
    use_tradeability_guard: bool = True,
    tradeability_limit_tolerance: float = 0.002,
    tradeability_min_volume: float = 0.0,
) -> BacktestResult:
    if not tape.days:
        return _empty_backtest_result(news_enabled=tape.news_enabled, use_state_engine=tape.use_state_engine)

    total_cost_rate = max(0.0, float(commission_bps) + float(slippage_bps)) / 10000.0
    symbols = list(tape.symbols)
    prev_weights_quant = np.zeros(len(symbols), dtype=float)
    prev_weights_fused = np.zeros(len(symbols), dtype=float)
    trade_history_quant: list[list[int]] = [[] for _ in symbols]
    trade_history_fused: list[list[int]] = [[] for _ in symbols]
    records: list[dict[str, object]] = []
    max_observed_week_trades_quant = 0
    max_observed_week_trades_fused = 0
    oversell_violations_quant = 0
    oversell_violations_fused = 0
    blocked_buy_quant = 0
    blocked_sell_quant = 0
    blocked_buy_fused = 0
    blocked_sell_fused = 0
    suspended_symbol_days = 0
    non_member_symbol_days = 0
    no_member_snapshot_days = 0

    for day in tape.days:
        if tape.use_state_engine:
            state_quant = decide_market_state(
                day.market_short_prob_quant,
                day.market_mid_prob_quant,
                base_weight_threshold=weight_threshold,
                base_max_positions=max_positions,
                base_max_trades_per_stock_per_day=max_trades_per_stock_per_day,
                base_max_trades_per_stock_per_week=max_trades_per_stock_per_week,
            )
            total_exposure_quant = float(state_quant.exposure_cap)
            threshold_quant = float(state_quant.weight_threshold)
            max_pos_quant = int(state_quant.max_positions)
            max_day_quant = int(state_quant.max_trades_per_stock_per_day)
            max_week_quant = int(state_quant.max_trades_per_stock_per_week)
            state_code_quant = str(state_quant.state_code)
        else:
            total_exposure_quant = float(target_exposure(day.market_short_prob_quant, day.market_mid_prob_quant))
            threshold_quant = float(weight_threshold)
            max_pos_quant = int(max_positions)
            max_day_quant = int(max_trades_per_stock_per_day)
            max_week_quant = int(max_trades_per_stock_per_week)
            state_code_quant = "legacy"

        if tape.use_state_engine:
            state_fused = decide_market_state(
                day.market_short_prob_fused,
                day.market_mid_prob_fused,
                base_weight_threshold=weight_threshold,
                base_max_positions=max_positions,
                base_max_trades_per_stock_per_day=max_trades_per_stock_per_day,
                base_max_trades_per_stock_per_week=max_trades_per_stock_per_week,
            )
            total_exposure_fused = float(state_fused.exposure_cap)
            threshold_fused = float(state_fused.weight_threshold)
            max_pos_fused = int(state_fused.max_positions)
            max_day_fused = int(state_fused.max_trades_per_stock_per_day)
            max_week_fused = int(state_fused.max_trades_per_stock_per_week)
            state_code_fused = str(state_fused.state_code)
        else:
            total_exposure_fused = float(target_exposure(day.market_short_prob_fused, day.market_mid_prob_fused))
            threshold_fused = float(weight_threshold)
            max_pos_fused = int(max_positions)
            max_day_fused = int(max_trades_per_stock_per_day)
            max_week_fused = int(max_trades_per_stock_per_week)
            state_code_fused = "legacy"

        member_mask = day.member_mask if tape.use_index_constituent_guard else np.ones(len(symbols), dtype=bool)
        if tape.use_index_constituent_guard:
            non_member_symbol_days += int(np.sum(~member_mask))
            if day.missing_member_snapshot:
                no_member_snapshot_days += 1

        target_weights_quant = np.zeros(len(symbols), dtype=float)
        target_weights_fused = np.zeros(len(symbols), dtype=float)
        target_weights_quant[member_mask] = np.asarray(
            allocate_weights(
                day.quant_scores[member_mask].tolist(),
                total_exposure=total_exposure_quant,
                threshold=threshold_quant,
                max_positions=max_pos_quant,
            ),
            dtype=float,
        )
        target_weights_fused[member_mask] = np.asarray(
            allocate_weights(
                day.fused_scores[member_mask].tolist(),
                total_exposure=total_exposure_fused,
                threshold=threshold_fused,
                max_positions=max_pos_fused,
            ),
            dtype=float,
        )

        if state_code_quant == "range":
            target_weights_quant = _apply_range_t_whitelist(
                prev_weights=prev_weights_quant,
                target_weights=target_weights_quant,
                ret_1=day.ret_1,
                price_pos_20=day.price_pos_20,
                min_weight_change_to_trade=min_weight_change_to_trade,
                sell_ret_1_min=range_t_sell_ret_1_min,
                sell_price_pos_20_min=range_t_sell_price_pos_20_min,
                buy_ret_1_max=range_t_buy_ret_1_max,
                buy_price_pos_20_max=range_t_buy_price_pos_20_max,
            )
        if state_code_fused == "range":
            target_weights_fused = _apply_range_t_whitelist(
                prev_weights=prev_weights_fused,
                target_weights=target_weights_fused,
                ret_1=day.ret_1,
                price_pos_20=day.price_pos_20,
                min_weight_change_to_trade=min_weight_change_to_trade,
                sell_ret_1_min=range_t_sell_ret_1_min,
                sell_price_pos_20_min=range_t_sell_price_pos_20_min,
                buy_ret_1_max=range_t_buy_ret_1_max,
                buy_price_pos_20_max=range_t_buy_price_pos_20_max,
            )

        if use_tradeability_guard:
            can_buy, can_sell, suspended_count = _build_tradeability_flags_from_arrays(
                date=day.date,
                symbols=symbols,
                open_arr=day.open_prices,
                high_arr=day.high_prices,
                low_arr=day.low_prices,
                close_arr=day.close_prices,
                volume_arr=day.volumes,
                ret_1_arr=day.ret_1,
                min_volume=float(tradeability_min_volume),
                limit_tolerance=float(tradeability_limit_tolerance),
                limit_rule_book=tape.limit_rule_book,
            )
            if tape.use_index_constituent_guard:
                can_buy = can_buy & member_mask
            suspended_symbol_days += int(suspended_count)
            target_weights_quant, block_buy_q, block_sell_q = _apply_tradeability_guard(
                prev_weights=prev_weights_quant,
                target_weights=target_weights_quant,
                can_buy=can_buy,
                can_sell=can_sell,
                min_weight_change_to_trade=min_weight_change_to_trade,
            )
            target_weights_fused, block_buy_f, block_sell_f = _apply_tradeability_guard(
                prev_weights=prev_weights_fused,
                target_weights=target_weights_fused,
                can_buy=can_buy,
                can_sell=can_sell,
                min_weight_change_to_trade=min_weight_change_to_trade,
            )
            blocked_buy_quant += int(block_buy_q)
            blocked_sell_quant += int(block_sell_q)
            blocked_buy_fused += int(block_buy_f)
            blocked_sell_fused += int(block_sell_f)

        if use_turnover_control:
            curr_weights_quant, turnover_quant, trade_count_quant, max_week_observed_quant, oversell_viol_quant = _apply_turnover_control(
                prev_weights=prev_weights_quant,
                target_weights=target_weights_quant,
                total_exposure=total_exposure_quant,
                trade_history=trade_history_quant,
                max_trades_per_stock_per_day=min(int(max_trades_per_stock_per_day), max_day_quant),
                max_trades_per_stock_per_week=min(int(max_trades_per_stock_per_week), max_week_quant),
                min_weight_change_to_trade=min_weight_change_to_trade,
            )
            curr_weights_fused, turnover_fused, trade_count_fused, max_week_observed_fused, oversell_viol_fused = _apply_turnover_control(
                prev_weights=prev_weights_fused,
                target_weights=target_weights_fused,
                total_exposure=total_exposure_fused,
                trade_history=trade_history_fused,
                max_trades_per_stock_per_day=min(int(max_trades_per_stock_per_day), max_day_fused),
                max_trades_per_stock_per_week=min(int(max_trades_per_stock_per_week), max_week_fused),
                min_weight_change_to_trade=min_weight_change_to_trade,
            )
            max_observed_week_trades_quant = max(max_observed_week_trades_quant, int(max_week_observed_quant))
            max_observed_week_trades_fused = max(max_observed_week_trades_fused, int(max_week_observed_fused))
            oversell_violations_quant += int(oversell_viol_quant)
            oversell_violations_fused += int(oversell_viol_fused)
        else:
            curr_weights_quant = target_weights_quant
            curr_weights_fused = target_weights_fused
            turnover_quant = float(np.sum(np.abs(curr_weights_quant - prev_weights_quant)))
            turnover_fused = float(np.sum(np.abs(curr_weights_fused - prev_weights_fused)))
            trade_count_quant = int(np.sum(np.abs(curr_weights_quant - prev_weights_quant) > 1e-9))
            trade_count_fused = int(np.sum(np.abs(curr_weights_fused - prev_weights_fused) > 1e-9))

        cost_quant = turnover_quant * total_cost_rate
        cost_fused = turnover_fused * total_cost_rate
        quant_gross = float(np.dot(curr_weights_quant, np.nan_to_num(day.fwd_ret_1, nan=0.0)))
        fused_gross = float(np.dot(curr_weights_fused, np.nan_to_num(day.fwd_ret_1, nan=0.0)))
        records.append(
            {
                "date": day.next_date,
                "quant_gross_ret": quant_gross,
                "quant_ret": quant_gross - cost_quant,
                "fused_gross_ret": fused_gross,
                "fused_ret": fused_gross - cost_fused,
                "benchmark_ret": float(day.benchmark_ret),
                "turnover_quant": turnover_quant,
                "turnover_fused": turnover_fused,
                "cost_quant": cost_quant,
                "cost_fused": cost_fused,
                "trade_count_quant": trade_count_quant,
                "trade_count_fused": trade_count_fused,
                "n_symbols": len(symbols),
                "market_short_prob_quant": float(day.market_short_prob_quant),
                "market_mid_prob_quant": float(day.market_mid_prob_quant),
                "market_short_prob_fused": float(day.market_short_prob_fused),
                "market_mid_prob_fused": float(day.market_mid_prob_fused),
                "total_exposure_quant": total_exposure_quant,
                "total_exposure_fused": total_exposure_fused,
            }
        )
        prev_weights_quant = curr_weights_quant
        prev_weights_fused = curr_weights_fused

    audit: dict[str, float | int | bool] = {
        "use_state_engine": bool(tape.use_state_engine),
        "use_tradeability_guard": bool(use_tradeability_guard),
        "tradeability_limit_tolerance": float(tradeability_limit_tolerance),
        "tradeability_min_volume": float(tradeability_min_volume),
        "limit_rule_file_enabled": bool(tape.limit_rule_book is not None),
        "use_index_constituent_guard": bool(tape.use_index_constituent_guard),
        "index_constituent_file_enabled": bool(tape.use_index_constituent_guard),
        "index_constituent_symbol": str(tape.index_constituent_symbol).strip().upper(),
        "non_member_symbol_days": int(non_member_symbol_days),
        "no_member_snapshot_days": int(no_member_snapshot_days),
        "blocked_buy_quant": int(blocked_buy_quant),
        "blocked_sell_quant": int(blocked_sell_quant),
        "blocked_buy_fused": int(blocked_buy_fused),
        "blocked_sell_fused": int(blocked_sell_fused),
        "blocked_total_quant": int(blocked_buy_quant + blocked_sell_quant),
        "blocked_total_fused": int(blocked_buy_fused + blocked_sell_fused),
        "suspended_symbol_days": int(suspended_symbol_days),
        "max_trades_per_stock_per_day_limit": int(max_trades_per_stock_per_day),
        "max_trades_per_stock_per_week_limit": int(max_trades_per_stock_per_week),
        "max_observed_week_trades_quant": int(max_observed_week_trades_quant),
        "max_observed_week_trades_fused": int(max_observed_week_trades_fused),
        "limit_violations_quant": int(max(0, max_observed_week_trades_quant - int(max_trades_per_stock_per_week))),
        "limit_violations_fused": int(max(0, max_observed_week_trades_fused - int(max_trades_per_stock_per_week))),
        "oversell_violations_quant": int(oversell_violations_quant),
        "oversell_violations_fused": int(oversell_violations_fused),
    }
    return _finalize_backtest_result(
        records=records,
        news_enabled=tape.news_enabled,
        window_years=tape.window_years,
        use_state_engine=tape.use_state_engine,
        audit=audit,
    )
