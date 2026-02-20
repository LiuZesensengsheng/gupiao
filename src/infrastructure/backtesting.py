from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import pandas as pd

from src.domain.entities import BacktestMetrics, Security
from src.domain.policies import allocate_weights, blend_horizon_score, target_exposure
from src.domain.symbols import normalize_symbol
from src.infrastructure.features import (
    MARKET_FEATURE_COLUMNS,
    make_market_feature_frame,
    make_stock_feature_frame,
    stock_feature_columns,
)
from src.infrastructure.market_data import DataError, load_symbol_daily
from src.infrastructure.modeling import LogisticBinaryModel


TRADING_DAYS_PER_YEAR = 252.0


@dataclass(frozen=True)
class BacktestResult:
    daily_frame: pd.DataFrame
    curve_frame: pd.DataFrame
    metrics: list[BacktestMetrics]


def _annualized(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return np.nan
    return float((1.0 + total_return) ** (TRADING_DAYS_PER_YEAR / float(n_days)) - 1.0)


def _max_drawdown(nav: np.ndarray) -> float:
    if nav.size == 0:
        return np.nan
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(drawdown))


def _to_metrics(frame: pd.DataFrame, label: str) -> BacktestMetrics:
    if frame.empty:
        return BacktestMetrics.empty(label)

    daily = frame["strategy_ret"].astype(float).to_numpy()
    benchmark = frame["benchmark_ret"].astype(float).to_numpy()
    excess = frame["excess_ret"].astype(float).to_numpy()
    turnover = frame["turnover"].astype(float).to_numpy()
    costs = frame["cost"].astype(float).to_numpy()
    n_days = int(len(frame))

    strategy_nav = np.cumprod(1.0 + daily)
    benchmark_nav = np.cumprod(1.0 + benchmark)
    excess_nav = np.cumprod(1.0 + excess)

    total_return = float(strategy_nav[-1] - 1.0)
    benchmark_total_return = float(benchmark_nav[-1] - 1.0)
    excess_total_return = float(excess_nav[-1] - 1.0)

    annual_return = _annualized(total_return, n_days)
    benchmark_annual_return = _annualized(benchmark_total_return, n_days)
    excess_annual_return = _annualized(excess_total_return, n_days)

    annual_vol = float(np.std(daily, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    tracking_error = float(np.std(excess, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    downside = np.minimum(daily, 0.0)
    downside_vol = float(np.std(downside, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))
    max_drawdown = _max_drawdown(strategy_nav)

    sharpe = float(annual_return / annual_vol) if annual_vol > 1e-12 else np.nan
    sortino = float(annual_return / downside_vol) if downside_vol > 1e-12 else np.nan
    calmar = float(annual_return / abs(max_drawdown)) if max_drawdown < -1e-12 else np.nan

    info_ratio = float(np.mean(excess) / np.std(excess, ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)) if tracking_error > 1e-12 else np.nan

    return BacktestMetrics(
        label=label,
        start_date=pd.Timestamp(frame["date"].iloc[0]),
        end_date=pd.Timestamp(frame["date"].iloc[-1]),
        n_days=n_days,
        total_return=total_return,
        annual_return=annual_return,
        annual_vol=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        benchmark_total_return=benchmark_total_return,
        benchmark_annual_return=benchmark_annual_return,
        excess_total_return=excess_total_return,
        excess_annual_return=excess_annual_return,
        information_ratio=info_ratio,
        tracking_error=tracking_error,
        win_rate=float(np.mean(daily > 0.0)),
        avg_turnover=float(np.mean(turnover)),
        annual_turnover=float(np.mean(turnover) * TRADING_DAYS_PER_YEAR),
        total_cost=float(np.sum(costs)),
    )


def _with_market_forward_return(market_raw: pd.DataFrame) -> pd.DataFrame:
    feat = make_market_feature_frame(market_raw)
    close = market_raw[["date", "close"]].copy()
    close["date"] = pd.to_datetime(close["date"], errors="coerce")
    close = close.sort_values("date").dropna(subset=["date", "close"])
    close["mkt_fwd_ret_1"] = close["close"].shift(-1) / close["close"] - 1.0
    out = feat.merge(close[["date", "mkt_fwd_ret_1"]], on="date", how="left", validate="1:1")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def _normalize_window_years(values: Sequence[int]) -> list[int]:
    out = sorted({int(v) for v in values if int(v) > 0})
    return out if out else [3, 5]


def run_portfolio_backtest(
    market_security: Security,
    stock_securities: Sequence[Security],
    source: str,
    data_dir: str,
    start: str,
    end: str,
    min_train_days: int,
    l2: float,
    retrain_days: int,
    weight_threshold: float,
    commission_bps: float,
    slippage_bps: float,
    window_years: Sequence[int],
) -> BacktestResult:
    retrain_days = max(1, int(retrain_days))
    total_cost_rate = max(0.0, float(commission_bps) + float(slippage_bps)) / 10000.0

    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_frame = _with_market_forward_return(market_raw)
    market_need = MARKET_FEATURE_COLUMNS + ["mkt_target_1d_up", "mkt_target_20d_up", "mkt_fwd_ret_1"]
    market_valid = market_frame.dropna(subset=market_need).sort_values("date").copy()
    market_valid = market_valid.set_index("date", drop=False)
    if market_valid.empty:
        raise DataError("Backtest failed: no valid market rows.")

    stock_cols = stock_feature_columns()
    stock_frames: Dict[str, pd.DataFrame] = {}
    stock_need = stock_cols + ["target_1d_up", "target_20d_up", "fwd_ret_1"]
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
        valid = stock_feat.dropna(subset=stock_need).sort_values("date").copy()
        if valid.empty:
            continue
        stock_frames[symbol] = valid.set_index("date", drop=False)

    if not stock_frames:
        raise DataError("Backtest failed: no valid stock rows.")

    common_dates = set(market_valid.index)
    for frame in stock_frames.values():
        common_dates &= set(frame.index)
    aligned_dates = sorted(pd.Timestamp(d) for d in common_dates)
    if len(aligned_dates) <= int(min_train_days) + 1:
        raise DataError("Backtest failed: insufficient aligned rows for training/testing.")

    symbols = [normalize_symbol(s.symbol).symbol for s in stock_securities if normalize_symbol(s.symbol).symbol in stock_frames]
    prev_weights = np.zeros(len(symbols), dtype=float)
    records: list[dict[str, object]] = []

    for block_start in range(int(min_train_days), len(aligned_dates) - 1, retrain_days):
        train_dates = aligned_dates[:block_start]
        train_index = pd.Index(train_dates)

        market_train = market_valid.loc[train_index]
        market_short_model = LogisticBinaryModel(l2=l2).fit(
            market_train,
            feature_cols=MARKET_FEATURE_COLUMNS,
            target_col="mkt_target_1d_up",
        )
        market_mid_model = LogisticBinaryModel(l2=l2).fit(
            market_train,
            feature_cols=MARKET_FEATURE_COLUMNS,
            target_col="mkt_target_20d_up",
        )

        stock_models: Dict[str, tuple[LogisticBinaryModel, LogisticBinaryModel]] = {}
        for symbol in symbols:
            train = stock_frames[symbol].loc[train_index]
            if train.empty:
                continue
            short_model = LogisticBinaryModel(l2=l2).fit(train, feature_cols=stock_cols, target_col="target_1d_up")
            mid_model = LogisticBinaryModel(l2=l2).fit(train, feature_cols=stock_cols, target_col="target_20d_up")
            stock_models[symbol] = (short_model, mid_model)

        block_end = min(block_start + retrain_days, len(aligned_dates) - 1)
        for i in range(block_start, block_end):
            date = aligned_dates[i]
            next_date = aligned_dates[i + 1]

            market_row = market_valid.loc[[date]]
            market_short = float(market_short_model.predict_proba(market_row, MARKET_FEATURE_COLUMNS)[0])
            market_mid = float(market_mid_model.predict_proba(market_row, MARKET_FEATURE_COLUMNS)[0])
            total_exposure = target_exposure(market_short, market_mid)

            score_items: list[tuple[str, float, float]] = []
            for symbol in symbols:
                models = stock_models.get(symbol)
                if models is None:
                    continue
                row = stock_frames[symbol].loc[[date]]
                short_prob = float(models[0].predict_proba(row, stock_cols)[0])
                mid_prob = float(models[1].predict_proba(row, stock_cols)[0])
                score_items.append((symbol, blend_horizon_score(short_prob, mid_prob, short_weight=0.55), float(row["fwd_ret_1"].iloc[0])))

            if not score_items:
                continue

            weights_raw = allocate_weights(
                [item[1] for item in score_items],
                total_exposure=total_exposure,
                threshold=weight_threshold,
            )
            weight_map = {item[0]: float(w) for item, w in zip(score_items, weights_raw)}
            curr_weights = np.array([weight_map.get(symbol, 0.0) for symbol in symbols], dtype=float)

            turnover = float(np.sum(np.abs(curr_weights - prev_weights)))
            cost = turnover * total_cost_rate
            gross = float(sum(weight_map[item[0]] * item[2] for item in score_items))
            net = gross - cost
            benchmark_ret = float(market_valid.loc[date, "mkt_fwd_ret_1"])

            records.append(
                {
                    "date": next_date,
                    "gross_ret": gross,
                    "strategy_ret": net,
                    "benchmark_ret": benchmark_ret,
                    "excess_ret": net - benchmark_ret,
                    "turnover": turnover,
                    "cost": cost,
                    "market_short_prob": market_short,
                    "market_mid_prob": market_mid,
                    "total_exposure": total_exposure,
                }
            )
            prev_weights = curr_weights

    daily_frame = pd.DataFrame(records).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    if daily_frame.empty:
        return BacktestResult(
            daily_frame=daily_frame,
            curve_frame=pd.DataFrame(columns=["date", "strategy_nav", "benchmark_nav", "excess_nav"]),
            metrics=[BacktestMetrics.empty("全样本")],
        )

    curve = daily_frame[["date"]].copy()
    curve["strategy_nav"] = (1.0 + daily_frame["strategy_ret"].astype(float)).cumprod()
    curve["benchmark_nav"] = (1.0 + daily_frame["benchmark_ret"].astype(float)).cumprod()
    curve["excess_nav"] = (1.0 + daily_frame["excess_ret"].astype(float)).cumprod()

    metrics: list[BacktestMetrics] = [_to_metrics(daily_frame, "全样本")]
    end_date = pd.Timestamp(daily_frame["date"].iloc[-1])
    for years in _normalize_window_years(window_years):
        window_start = end_date - pd.DateOffset(years=years)
        window_frame = daily_frame[daily_frame["date"] >= window_start].copy()
        metrics.append(_to_metrics(window_frame, f"近{years}年"))

    return BacktestResult(daily_frame=daily_frame, curve_frame=curve, metrics=metrics)
