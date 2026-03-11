from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd

from src.domain.entities import BacktestMetrics, NewsItem, Security, SentimentAggregate
from src.domain.news import aggregate_sentiment, blend_probability
from src.domain.policies import allocate_weights, blend_horizon_score, decide_market_state, target_exposure
from src.domain.symbols import normalize_symbol
from src.infrastructure.features import (
    MARKET_FEATURE_COLUMNS,
    make_market_feature_frame,
    make_stock_feature_frame,
    stock_feature_columns,
)
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import DataError, load_symbol_daily
from src.infrastructure.modeling import LogisticBinaryModel


TRADING_DAYS_PER_YEAR = 252.0
TRADING_DAYS_PER_WEEK = 5.0
NEWS_BACKTEST_FEATURE_COLUMNS = [
    "sent_score",
    "sent_bullish",
    "sent_bearish",
    "sent_neutral",
    "sent_items_log",
    "sent_abs_score",
    "sent_signed_items",
]
FUSION_BACKTEST_FEATURE_COLUMNS = ["q_logit", "n_logit", "q_minus_n"]


@dataclass(frozen=True)
class BacktestResult:
    daily_frame: pd.DataFrame
    curve_frame: pd.DataFrame
    metrics: list[BacktestMetrics]
    audit: dict[str, float | int | bool]


@dataclass
class _BlockFuser:
    mode: str
    reason: str
    fallback_strength: float
    news_model: LogisticBinaryModel | None = None
    fusion_model: LogisticBinaryModel | None = None


@dataclass(frozen=True)
class _LimitRule:
    symbol: str | None
    exchange: str | None
    code_prefix: str | None
    start_date: pd.Timestamp | None
    end_date: pd.Timestamp | None
    limit_rate: float


@dataclass(frozen=True)
class _LimitRuleBook:
    default_limit_rate: float
    rules: list[_LimitRule]


@dataclass(frozen=True)
class _IndexConstituentBook:
    index_symbol: str
    snapshot_dates: list[pd.Timestamp]
    snapshot_members: list[set[str]]


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
    trade_count = frame["trade_count"].astype(float).to_numpy() if "trade_count" in frame.columns else np.zeros(len(frame), dtype=float)
    n_symbols = int(frame["n_symbols"].iloc[0]) if "n_symbols" in frame.columns and len(frame) > 0 else 1
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
        avg_trade_count_per_day=float(np.mean(trade_count)),
        avg_trades_per_stock_per_week=float(np.mean(trade_count) * TRADING_DAYS_PER_WEEK / float(max(1, n_symbols))),
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


def _clip_prob(v: float) -> float:
    return float(np.clip(v, 1e-6, 1.0 - 1e-6))


def _logit(arr: np.ndarray) -> np.ndarray:
    x = np.clip(arr, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _sentiment_features(sent: SentimentAggregate) -> dict[str, float]:
    score = float(np.clip(sent.score, -1.0, 1.0))
    items_log = float(np.log1p(max(0, int(sent.items))))
    return {
        "sent_score": score,
        "sent_bullish": float(np.clip(sent.bullish, 0.0, 1.0)),
        "sent_bearish": float(np.clip(sent.bearish, 0.0, 1.0)),
        "sent_neutral": float(np.clip(sent.neutral, 0.0, 1.0)),
        "sent_items_log": items_log,
        "sent_abs_score": float(abs(score)),
        "sent_signed_items": float(score * items_log),
    }


def _fusion_feature_frame(p_quant: Sequence[float], p_news: Sequence[float]) -> pd.DataFrame:
    q = np.asarray([_clip_prob(float(v)) for v in p_quant], dtype=float)
    n = np.asarray([_clip_prob(float(v)) for v in p_news], dtype=float)
    return pd.DataFrame(
        {
            "q_logit": _logit(q),
            "n_logit": _logit(n),
            "q_minus_n": q - n,
        }
    )


def _fit_block_fuser(
    train_frame: pd.DataFrame,
    quant_model: LogisticBinaryModel,
    feature_cols: list[str],
    target_col: str,
    *,
    target: str,
    horizon: str,
    fallback_strength: float,
    min_samples: int,
    news_l2: float,
    fusion_l2: float,
    sentiment_getter: Callable[[pd.Timestamp, str, str], SentimentAggregate],
) -> _BlockFuser:
    valid = train_frame.dropna(subset=["date"] + feature_cols + [target_col]).sort_values("date").copy()
    if len(valid) < int(min_samples):
        return _BlockFuser(mode="rule", reason="insufficient_samples", fallback_strength=fallback_strength)

    try:
        quant_prob = quant_model.predict_proba(valid, feature_cols=feature_cols)
        rows: list[dict[str, float | pd.Timestamp]] = []
        for d, y, p in zip(valid["date"].tolist(), valid[target_col].astype(float).tolist(), quant_prob.tolist()):
            sent = sentiment_getter(pd.Timestamp(d), target, horizon)
            feat = _sentiment_features(sent)
            feat["date"] = pd.Timestamp(d)
            feat["y"] = float(y)
            feat["p_quant"] = _clip_prob(float(p))
            rows.append(feat)
        dataset = pd.DataFrame(rows).dropna()
        if len(dataset) < int(min_samples):
            return _BlockFuser(mode="rule", reason="insufficient_after_merge", fallback_strength=fallback_strength)

        news_model = LogisticBinaryModel(l2=float(news_l2)).fit(dataset, NEWS_BACKTEST_FEATURE_COLUMNS, "y")
        p_news_hist = news_model.predict_proba(dataset, NEWS_BACKTEST_FEATURE_COLUMNS)
        fusion_train = _fusion_feature_frame(dataset["p_quant"].to_numpy(dtype=float), p_news_hist)
        fusion_train["y"] = dataset["y"].astype(float).to_numpy()
        fusion_model = LogisticBinaryModel(l2=float(fusion_l2)).fit(fusion_train, FUSION_BACKTEST_FEATURE_COLUMNS, "y")
        return _BlockFuser(
            mode="learned",
            reason="ok",
            fallback_strength=fallback_strength,
            news_model=news_model,
            fusion_model=fusion_model,
        )
    except Exception:
        return _BlockFuser(mode="rule", reason="learning_error", fallback_strength=fallback_strength)


def _apply_block_fuser(
    fuser: _BlockFuser,
    base_prob: float,
    *,
    date: pd.Timestamp,
    target: str,
    horizon: str,
    sentiment_getter: Callable[[pd.Timestamp, str, str], SentimentAggregate],
) -> float:
    sent = sentiment_getter(date, target, horizon)
    if fuser.mode == "learned" and fuser.news_model is not None and fuser.fusion_model is not None:
        try:
            news_feat = pd.DataFrame([_sentiment_features(sent)])
            p_news = float(fuser.news_model.predict_proba(news_feat, NEWS_BACKTEST_FEATURE_COLUMNS)[0])
            fusion_feat = _fusion_feature_frame([base_prob], [p_news])
            p_final = float(fuser.fusion_model.predict_proba(fusion_feat, FUSION_BACKTEST_FEATURE_COLUMNS)[0])
            return _clip_prob(p_final)
        except Exception:
            pass
    return blend_probability(base_prob, sent.score, sentiment_strength=fuser.fallback_strength)


def _to_strategy_frame(
    daily_frame: pd.DataFrame,
    *,
    ret_col: str,
    turnover_col: str,
    cost_col: str,
    trade_count_col: str,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(daily_frame["date"], errors="coerce"),
            "strategy_ret": daily_frame[ret_col].astype(float),
            "benchmark_ret": daily_frame["benchmark_ret"].astype(float),
            "turnover": daily_frame[turnover_col].astype(float),
            "cost": daily_frame[cost_col].astype(float),
            "trade_count": daily_frame[trade_count_col].astype(float),
            "n_symbols": daily_frame["n_symbols"].astype(float),
        }
    )
    out["excess_ret"] = out["strategy_ret"] - out["benchmark_ret"]
    return out.sort_values("date").dropna(subset=["date"]).reset_index(drop=True)


def _apply_turnover_control(
    *,
    prev_weights: np.ndarray,
    target_weights: np.ndarray,
    total_exposure: float,
    trade_history: list[list[int]],
    max_trades_per_stock_per_day: int,
    max_trades_per_stock_per_week: int,
    min_weight_change_to_trade: float,
) -> tuple[np.ndarray, float, int, int, int]:
    prev = np.asarray(prev_weights, dtype=float)
    out = np.asarray(target_weights, dtype=float).copy()
    n = len(prev)
    if out.shape[0] != n:
        out = np.zeros(n, dtype=float)

    max_trades_day = max(1, int(max_trades_per_stock_per_day))
    max_trades_week = max(1, int(max_trades_per_stock_per_week))
    max_trades = min(max_trades_week, max_trades_day * int(TRADING_DAYS_PER_WEEK))
    min_delta = max(0.0, float(min_weight_change_to_trade))
    fixed = np.zeros(n, dtype=bool)
    oversell_violations = 0

    for i in range(n):
        delta = float(out[i] - prev[i])
        if abs(delta) < min_delta:
            out[i] = prev[i]
            fixed[i] = True
            continue
        if delta < -float(prev[i]) - 1e-12:
            oversell_violations += 1
            out[i] = max(0.0, float(prev[i]) - float(prev[i]))
        lookback = max(0, int(TRADING_DAYS_PER_WEEK) - 1)
        projected_window = list(trade_history[i][-lookback:]) + [1]
        projected_trades = int(sum(projected_window))
        if projected_trades > max_trades:
            out[i] = prev[i]
            fixed[i] = True

    # Enforce no short/oversell: today's sell amount cannot exceed yesterday's holdings.
    out = np.clip(out, 0.0, None)
    target_total = float(np.clip(total_exposure, 0.0, 1.0))
    fixed_sum = float(np.sum(out[fixed])) if n > 0 else 0.0
    free_mask = ~fixed

    if np.any(free_mask):
        remaining = max(0.0, target_total - fixed_sum)
        free_sum = float(np.sum(out[free_mask]))
        if free_sum > 1e-12:
            out[free_mask] = out[free_mask] / free_sum * remaining
        else:
            prev_free = np.clip(prev[free_mask], 0.0, None)
            prev_free_sum = float(np.sum(prev_free))
            if prev_free_sum > 1e-12:
                out[free_mask] = prev_free / prev_free_sum * remaining
            else:
                out[free_mask] = 0.0
    else:
        full_sum = float(np.sum(out))
        if full_sum > target_total + 1e-12 and full_sum > 1e-12:
            out = out / full_sum * target_total

    delta_abs = np.abs(out - prev)
    traded = delta_abs >= min_delta
    turnover = float(np.sum(delta_abs))
    trade_count = int(np.sum(traded))

    for i in range(n):
        trade_history[i].append(1 if traded[i] else 0)
        if len(trade_history[i]) > int(TRADING_DAYS_PER_WEEK):
            trade_history[i].pop(0)
    max_recent_trades_week = 0
    for history in trade_history:
        max_recent_trades_week = max(max_recent_trades_week, int(sum(history[-int(TRADING_DAYS_PER_WEEK) :])))

    return out, turnover, trade_count, max_recent_trades_week, oversell_violations


def _apply_range_t_whitelist(
    *,
    prev_weights: np.ndarray,
    target_weights: np.ndarray,
    ret_1: np.ndarray,
    price_pos_20: np.ndarray,
    min_weight_change_to_trade: float,
    sell_ret_1_min: float,
    sell_price_pos_20_min: float,
    buy_ret_1_max: float,
    buy_price_pos_20_max: float,
) -> np.ndarray:
    prev = np.asarray(prev_weights, dtype=float)
    out = np.asarray(target_weights, dtype=float).copy()
    n = len(prev)
    if out.shape[0] != n:
        return prev.copy()

    min_delta = max(0.0, float(min_weight_change_to_trade))
    ret_1 = np.asarray(ret_1, dtype=float)
    price_pos_20 = np.asarray(price_pos_20, dtype=float)
    if ret_1.shape[0] != n or price_pos_20.shape[0] != n:
        return out

    for i in range(n):
        delta = float(out[i] - prev[i])
        if abs(delta) < min_delta:
            continue
        sell_trigger = (ret_1[i] >= float(sell_ret_1_min)) and (price_pos_20[i] >= float(sell_price_pos_20_min))
        buy_trigger = (ret_1[i] <= float(buy_ret_1_max)) and (price_pos_20[i] <= float(buy_price_pos_20_max))
        if delta < 0.0 and not bool(sell_trigger):
            out[i] = prev[i]
        elif delta > 0.0 and not bool(buy_trigger):
            out[i] = prev[i]
    return out


def _default_price_limit_rate(symbol: str) -> float:
    text = str(symbol).strip().upper()
    code = text.split(".", 1)[0]
    if text.endswith(".BJ"):
        return 0.30
    if text.endswith(".SH") and code.startswith("688"):
        return 0.20
    if text.endswith(".SZ") and code.startswith("300"):
        return 0.20
    return 0.10


def _to_rule_date(value: object) -> pd.Timestamp | None:
    text = str(value).strip()
    if not text:
        return None
    ts = pd.to_datetime(text, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _load_limit_rule_book(path: str) -> _LimitRuleBook | None:
    p = Path(str(path).strip())
    if not str(path).strip() or not p.exists():
        return None
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    default_limit = float(payload.get("default_limit_rate", 0.10))
    raw_rules = payload.get("rules", [])
    if not isinstance(raw_rules, list):
        raw_rules = []
    rules: list[_LimitRule] = []
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol", "")).strip().upper() or None
        exchange = str(raw.get("exchange", "")).strip().upper() or None
        code_prefix = str(raw.get("code_prefix", "")).strip() or None
        rate = pd.to_numeric(raw.get("limit_rate", np.nan), errors="coerce")
        if pd.isna(rate):
            continue
        start_date = _to_rule_date(raw.get("start_date", ""))
        end_date = _to_rule_date(raw.get("end_date", ""))
        rules.append(
            _LimitRule(
                symbol=symbol,
                exchange=exchange,
                code_prefix=code_prefix,
                start_date=start_date,
                end_date=end_date,
                limit_rate=float(rate),
            )
        )
    return _LimitRuleBook(default_limit_rate=float(default_limit), rules=rules)


def _resolve_price_limit_rate(
    symbol: str,
    *,
    date: pd.Timestamp,
    rule_book: _LimitRuleBook | None,
) -> float:
    if rule_book is None:
        return _default_price_limit_rate(symbol)
    info = normalize_symbol(symbol)
    day = pd.Timestamp(date).normalize()
    matched: _LimitRule | None = None
    for rule in rule_book.rules:
        if rule.symbol is not None and rule.symbol != info.symbol:
            continue
        if rule.exchange is not None and rule.exchange != info.exchange:
            continue
        if rule.code_prefix is not None and not info.code.startswith(rule.code_prefix):
            continue
        if rule.start_date is not None and day < rule.start_date:
            continue
        if rule.end_date is not None and day > rule.end_date:
            continue
        matched = rule
        break
    if matched is not None:
        return float(matched.limit_rate)
    return float(rule_book.default_limit_rate) if rule_book.default_limit_rate > 0 else _default_price_limit_rate(symbol)


def _load_index_constituent_book(path: str, index_symbol: str) -> _IndexConstituentBook | None:
    p = Path(str(path).strip())
    if not str(path).strip() or not p.exists():
        return None
    raw = pd.read_csv(p)
    if raw.empty:
        return None
    cols = {c.lower(): c for c in raw.columns}
    date_col = cols.get("date")
    symbol_col = cols.get("symbol")
    if date_col is None or symbol_col is None:
        raise ValueError(f"index constituent file missing required columns: {p}")
    idx_col = cols.get("index_symbol")
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    frame["symbol"] = frame[symbol_col].astype(str).str.strip().str.upper()
    if idx_col is not None:
        idx = frame[idx_col].astype(str).str.strip().str.upper()
        frame = frame[idx == str(index_symbol).strip().upper()]
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return None
    frame = frame[frame["symbol"].str.len() > 0]
    frame = frame.sort_values(["date", "symbol"])
    snapshot_dates: list[pd.Timestamp] = []
    snapshot_members: list[set[str]] = []
    for d, grp in frame.groupby("date", sort=True):
        members: set[str] = set()
        for sym in grp["symbol"].tolist():
            try:
                members.add(normalize_symbol(sym).symbol)
            except Exception:
                continue
        if members:
            snapshot_dates.append(pd.Timestamp(d).normalize())
            snapshot_members.append(members)
    if not snapshot_dates:
        return None
    return _IndexConstituentBook(
        index_symbol=str(index_symbol).strip().upper(),
        snapshot_dates=snapshot_dates,
        snapshot_members=snapshot_members,
    )


def _members_asof(book: _IndexConstituentBook, date: pd.Timestamp) -> set[str] | None:
    day = pd.Timestamp(date).normalize()
    dates = book.snapshot_dates
    lo = 0
    hi = len(dates) - 1
    idx = -1
    while lo <= hi:
        mid = (lo + hi) // 2
        if dates[mid] <= day:
            idx = mid
            lo = mid + 1
        else:
            hi = mid - 1
    if idx < 0:
        return None
    return book.snapshot_members[idx]


def _is_one_price_bar(open_: float, high: float, low: float, close: float) -> bool:
    if not (np.isfinite(open_) and np.isfinite(high) and np.isfinite(low) and np.isfinite(close)):
        return False
    base = max(1.0, abs(close))
    tol = 1e-4 * base
    return (abs(high - low) <= tol) and (abs(open_ - close) <= tol)


def _build_tradeability_flags(
    *,
    date: pd.Timestamp,
    symbols: list[str],
    item_map: dict[str, dict[str, float | str]],
    min_volume: float,
    limit_tolerance: float,
    limit_rule_book: _LimitRuleBook | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    can_buy = np.ones(len(symbols), dtype=bool)
    can_sell = np.ones(len(symbols), dtype=bool)
    suspended_count = 0
    tol = max(0.0, float(limit_tolerance))
    min_vol = max(0.0, float(min_volume))

    for i, symbol in enumerate(symbols):
        item = item_map.get(symbol)
        if item is None:
            can_buy[i] = False
            can_sell[i] = False
            suspended_count += 1
            continue
        open_ = float(item.get("open", np.nan))
        high = float(item.get("high", np.nan))
        low = float(item.get("low", np.nan))
        close = float(item.get("close", np.nan))
        volume = float(item.get("volume", np.nan))
        ret_1 = float(item.get("ret_1", np.nan))

        is_suspended = (
            (not np.isfinite(open_))
            or (not np.isfinite(high))
            or (not np.isfinite(low))
            or (not np.isfinite(close))
            or (not np.isfinite(volume))
            or (volume <= min_vol)
        )
        if is_suspended:
            can_buy[i] = False
            can_sell[i] = False
            suspended_count += 1
            continue

        limit_rate = _resolve_price_limit_rate(symbol, date=date, rule_book=limit_rule_book)
        one_price = _is_one_price_bar(open_, high, low, close)
        hit_up = np.isfinite(ret_1) and (ret_1 >= (limit_rate - tol))
        hit_down = np.isfinite(ret_1) and (ret_1 <= (-limit_rate + tol))

        if one_price and hit_up:
            can_buy[i] = False
        if one_price and hit_down:
            can_sell[i] = False

    return can_buy, can_sell, suspended_count


def _apply_tradeability_guard(
    *,
    prev_weights: np.ndarray,
    target_weights: np.ndarray,
    can_buy: np.ndarray,
    can_sell: np.ndarray,
    min_weight_change_to_trade: float,
) -> tuple[np.ndarray, int, int]:
    prev = np.asarray(prev_weights, dtype=float)
    out = np.asarray(target_weights, dtype=float).copy()
    n = len(prev)
    if out.shape[0] != n:
        out = np.zeros(n, dtype=float)
    if can_buy.shape[0] != n or can_sell.shape[0] != n:
        return prev.copy(), 0, 0

    min_delta = max(0.0, float(min_weight_change_to_trade))
    blocked_buy = 0
    blocked_sell = 0
    for i in range(n):
        delta = float(out[i] - prev[i])
        if abs(delta) < min_delta:
            continue
        if delta > 0.0 and not bool(can_buy[i]):
            out[i] = prev[i]
            blocked_buy += 1
        elif delta < 0.0 and not bool(can_sell[i]):
            out[i] = prev[i]
            blocked_sell += 1
    return out, blocked_buy, blocked_sell


def _build_window_metrics(frame: pd.DataFrame, window_years: Sequence[int], prefix: str = "") -> list[BacktestMetrics]:
    def _label(base: str) -> str:
        return f"{prefix}{base}" if prefix else base

    metrics: list[BacktestMetrics] = [_to_metrics(frame, _label("全样本"))]
    if frame.empty:
        return metrics

    end_date = pd.Timestamp(frame["date"].iloc[-1])
    for years in _normalize_window_years(window_years):
        window_start = end_date - pd.DateOffset(years=years)
        window_frame = frame[frame["date"] >= window_start].copy()
        metrics.append(_to_metrics(window_frame, _label(f"近{years}年")))
    return metrics


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
    news_items: Sequence[NewsItem] | None = None,
    apply_news_fusion: bool = False,
    news_half_life_days: float = 10.0,
    market_news_strength: float = 0.9,
    stock_news_strength: float = 1.1,
    use_learned_news_fusion: bool = False,
    learned_news_min_samples: int = 80,
    learned_news_l2: float = 0.8,
    learned_fusion_l2: float = 0.6,
    max_positions: int = 5,
    use_turnover_control: bool = True,
    max_trades_per_stock_per_day: int = 1,
    max_trades_per_stock_per_week: int = 3,
    min_weight_change_to_trade: float = 0.03,
    range_t_sell_ret_1_min: float = 0.02,
    range_t_sell_price_pos_20_min: float = 0.80,
    range_t_buy_ret_1_max: float = -0.02,
    range_t_buy_price_pos_20_max: float = 0.35,
    max_runtime_seconds: float = 0.0,
    use_margin_features: bool = True,
    margin_market_file: str = "input/margin_market.csv",
    margin_stock_file: str = "input/margin_stock.csv",
    use_us_index_context: bool = False,
    us_index_source: str = "akshare",
    use_state_engine: bool = True,
    use_tradeability_guard: bool = True,
    tradeability_limit_tolerance: float = 0.002,
    tradeability_min_volume: float = 0.0,
    limit_rule_file: str = "",
    use_index_constituent_guard: bool = False,
    index_constituent_file: str = "",
    index_constituent_symbol: str = "000300.SH",
) -> BacktestResult:
    retrain_days = max(1, int(retrain_days))
    total_cost_rate = max(0.0, float(commission_bps) + float(slippage_bps)) / 10000.0
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
    market_valid = market_frame.dropna(subset=market_need).sort_values("date").copy()
    market_valid = market_valid.set_index("date", drop=False)
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
    print(f"[BT] backtest start: symbols={len(stock_securities)}, test_days={total_test_days}, blocks={total_blocks}, budget={budget_text}")

    symbols = [normalize_symbol(s.symbol).symbol for s in stock_securities if normalize_symbol(s.symbol).symbol in stock_frames]
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

    timeout_hit = False
    processed_days = 0
    for block_idx, block_start in enumerate(block_starts, start=1):
        if deadline_ts is not None and time.monotonic() >= deadline_ts:
            timeout_hit = True
            print(f"[BT] time budget reached before block {block_idx}/{total_blocks}, stop.")
            break
        elapsed = time.monotonic() - start_ts
        print(f"[BT] block {block_idx}/{total_blocks} start (elapsed={elapsed:.1f}s, processed_days={processed_days}/{total_test_days})")
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
            if stock_cols is None:
                continue
            if train.empty:
                continue
            short_model = LogisticBinaryModel(l2=l2).fit(train, feature_cols=stock_cols, target_col="target_1d_up")
            mid_model = LogisticBinaryModel(l2=l2).fit(train, feature_cols=stock_cols, target_col="target_20d_up")
            stock_models[symbol] = (short_model, mid_model)

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
                    if stock_cols is None:
                        continue
                    if train.empty or models is None:
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
        for i in range(block_start, block_end):
            if deadline_ts is not None and time.monotonic() >= deadline_ts:
                timeout_hit = True
                print(f"[BT] time budget reached inside block {block_idx}/{total_blocks}, stop.")
                break
            date = aligned_dates[i]
            next_date = aligned_dates[i + 1]

            market_row = market_valid.loc[[date]]
            market_short_quant = float(market_short_model.predict_proba(market_row, market_feature_cols)[0])
            market_mid_quant = float(market_mid_model.predict_proba(market_row, market_feature_cols)[0])
            if use_state_engine:
                state_quant = decide_market_state(
                    market_short_quant,
                    market_mid_quant,
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
                total_exposure_quant = float(target_exposure(market_short_quant, market_mid_quant))
                threshold_quant = float(weight_threshold)
                max_pos_quant = int(max_positions)
                max_day_quant = int(max_trades_per_stock_per_day)
                max_week_quant = int(max_trades_per_stock_per_week)
                state_code_quant = "legacy"

            market_short_fused = market_short_quant
            market_mid_fused = market_mid_quant
            if news_enabled:
                market_short_fused = _apply_block_fuser(
                    market_fusers["short"],
                    market_short_quant,
                    date=date,
                    target="MARKET",
                    horizon="short",
                    sentiment_getter=_sentiment_getter,
                )
                market_mid_fused = _apply_block_fuser(
                    market_fusers["mid"],
                    market_mid_quant,
                    date=date,
                    target="MARKET",
                    horizon="mid",
                    sentiment_getter=_sentiment_getter,
                )
            if use_state_engine:
                state_fused = decide_market_state(
                    market_short_fused,
                    market_mid_fused,
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
                total_exposure_fused = float(target_exposure(market_short_fused, market_mid_fused))
                threshold_fused = float(weight_threshold)
                max_pos_fused = int(max_positions)
                max_day_fused = int(max_trades_per_stock_per_day)
                max_week_fused = int(max_trades_per_stock_per_week)
                state_code_fused = "legacy"

            score_items: list[dict[str, float | str]] = []
            for symbol in symbols:
                models = stock_models.get(symbol)
                stock_cols = stock_cols_map.get(symbol)
                if models is None or stock_cols is None:
                    continue
                row = stock_frames[symbol].loc[[date]]
                short_prob = float(models[0].predict_proba(row, stock_cols)[0])
                mid_prob = float(models[1].predict_proba(row, stock_cols)[0])
                quant_score = blend_horizon_score(short_prob, mid_prob, short_weight=0.55)

                fused_short = short_prob
                fused_mid = mid_prob
                if news_enabled:
                    fuser_short = stock_fusers.get(symbol, {}).get("short")
                    fuser_mid = stock_fusers.get(symbol, {}).get("mid")
                    if fuser_short is None:
                        fuser_short = _BlockFuser(mode="rule", reason="missing", fallback_strength=float(stock_news_strength))
                    if fuser_mid is None:
                        fuser_mid = _BlockFuser(mode="rule", reason="missing", fallback_strength=float(stock_news_strength))
                    fused_short = _apply_block_fuser(
                        fuser_short,
                        short_prob,
                        date=date,
                        target=symbol,
                        horizon="short",
                        sentiment_getter=_sentiment_getter,
                    )
                    fused_mid = _apply_block_fuser(
                        fuser_mid,
                        mid_prob,
                        date=date,
                        target=symbol,
                        horizon="mid",
                        sentiment_getter=_sentiment_getter,
                    )
                fused_score = blend_horizon_score(fused_short, fused_mid, short_weight=0.55)

                score_items.append(
                    {
                        "symbol": symbol,
                        "quant_score": float(quant_score),
                        "fused_score": float(fused_score),
                        "fwd_ret_1": float(row["fwd_ret_1"].iloc[0]),
                        "ret_1": float(row["ret_1"].iloc[0]),
                        "price_pos_20": float(row["price_pos_20"].iloc[0]),
                        "open": float(row["open"].iloc[0]),
                        "high": float(row["high"].iloc[0]),
                        "low": float(row["low"].iloc[0]),
                        "close": float(row["close"].iloc[0]),
                        "volume": float(row["volume"].iloc[0]),
                    }
                )

            if not score_items:
                continue

            member_mask = np.ones(len(symbols), dtype=bool)
            if use_index_constituent_guard and index_book is not None:
                members = _members_asof(index_book, date)
                if members is None:
                    member_mask = np.zeros(len(symbols), dtype=bool)
                    no_member_snapshot_days += 1
                else:
                    member_mask = np.array([symbol in members for symbol in symbols], dtype=bool)
                non_member_symbol_days += int(np.sum(~member_mask))

            eligible_symbols = [s for s, keep in zip(symbols, member_mask) if bool(keep)]
            quant_score_map = {str(item["symbol"]): float(item["quant_score"]) for item in score_items}
            fused_score_map = {str(item["symbol"]): float(item["fused_score"]) for item in score_items}
            eligible_scores_quant = [quant_score_map.get(sym, 0.0) for sym in eligible_symbols]
            eligible_scores_fused = [fused_score_map.get(sym, 0.0) for sym in eligible_symbols]

            quant_weights_eligible = allocate_weights(
                eligible_scores_quant,
                total_exposure=total_exposure_quant,
                threshold=threshold_quant,
                max_positions=max_pos_quant,
            )
            fused_weights_eligible = allocate_weights(
                eligible_scores_fused,
                total_exposure=total_exposure_fused,
                threshold=threshold_fused,
                max_positions=max_pos_fused,
            )
            quant_weight_map = {str(sym): float(w) for sym, w in zip(eligible_symbols, quant_weights_eligible)}
            fused_weight_map = {str(sym): float(w) for sym, w in zip(eligible_symbols, fused_weights_eligible)}
            item_map = {str(item["symbol"]): item for item in score_items}
            target_weights_quant = np.array([quant_weight_map.get(symbol, 0.0) for symbol in symbols], dtype=float)
            target_weights_fused = np.array([fused_weight_map.get(symbol, 0.0) for symbol in symbols], dtype=float)
            ret_1_arr = np.array([float(item_map.get(symbol, {}).get("ret_1", np.nan)) for symbol in symbols], dtype=float)
            price_pos_20_arr = np.array(
                [float(item_map.get(symbol, {}).get("price_pos_20", np.nan)) for symbol in symbols],
                dtype=float,
            )

            if state_code_quant == "range":
                target_weights_quant = _apply_range_t_whitelist(
                    prev_weights=prev_weights_quant,
                    target_weights=target_weights_quant,
                    ret_1=ret_1_arr,
                    price_pos_20=price_pos_20_arr,
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
                    ret_1=ret_1_arr,
                    price_pos_20=price_pos_20_arr,
                    min_weight_change_to_trade=min_weight_change_to_trade,
                    sell_ret_1_min=range_t_sell_ret_1_min,
                    sell_price_pos_20_min=range_t_sell_price_pos_20_min,
                    buy_ret_1_max=range_t_buy_ret_1_max,
                    buy_price_pos_20_max=range_t_buy_price_pos_20_max,
                )

            if use_tradeability_guard:
                can_buy, can_sell, suspended_count = _build_tradeability_flags(
                    date=date,
                    symbols=symbols,
                    item_map=item_map,
                    min_volume=float(tradeability_min_volume),
                    limit_tolerance=float(tradeability_limit_tolerance),
                    limit_rule_book=limit_rule_book,
                )
                if use_index_constituent_guard and index_book is not None:
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
                (
                    curr_weights_quant,
                    turnover_quant,
                    trade_count_quant,
                    max_week_observed_quant,
                    oversell_viol_quant,
                ) = _apply_turnover_control(
                    prev_weights=prev_weights_quant,
                    target_weights=target_weights_quant,
                    total_exposure=total_exposure_quant,
                    trade_history=trade_history_quant,
                    max_trades_per_stock_per_day=min(int(max_trades_per_stock_per_day), max_day_quant),
                    max_trades_per_stock_per_week=min(int(max_trades_per_stock_per_week), max_week_quant),
                    min_weight_change_to_trade=min_weight_change_to_trade,
                )
                (
                    curr_weights_fused,
                    turnover_fused,
                    trade_count_fused,
                    max_week_observed_fused,
                    oversell_viol_fused,
                ) = _apply_turnover_control(
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
            quant_curr_map = {symbol: float(w) for symbol, w in zip(symbols, curr_weights_quant)}
            fused_curr_map = {symbol: float(w) for symbol, w in zip(symbols, curr_weights_fused)}
            quant_gross = float(sum(quant_curr_map[str(item["symbol"])] * float(item["fwd_ret_1"]) for item in score_items))
            fused_gross = float(sum(fused_curr_map[str(item["symbol"])] * float(item["fwd_ret_1"]) for item in score_items))
            quant_net = quant_gross - cost_quant
            fused_net = fused_gross - cost_fused
            benchmark_ret = float(market_valid.loc[date, "mkt_fwd_ret_1"])

            records.append(
                {
                    "date": next_date,
                    "quant_gross_ret": quant_gross,
                    "quant_ret": quant_net,
                    "fused_gross_ret": fused_gross,
                    "fused_ret": fused_net,
                    "benchmark_ret": benchmark_ret,
                    "turnover_quant": turnover_quant,
                    "turnover_fused": turnover_fused,
                    "cost_quant": cost_quant,
                    "cost_fused": cost_fused,
                    "trade_count_quant": trade_count_quant,
                    "trade_count_fused": trade_count_fused,
                    "n_symbols": len(symbols),
                    "market_short_prob_quant": market_short_quant,
                    "market_mid_prob_quant": market_mid_quant,
                    "market_short_prob_fused": market_short_fused,
                    "market_mid_prob_fused": market_mid_fused,
                    "total_exposure_quant": total_exposure_quant,
                    "total_exposure_fused": total_exposure_fused,
                }
            )
            processed_days += 1
            prev_weights_quant = curr_weights_quant
            prev_weights_fused = curr_weights_fused
        if timeout_hit:
            break
        elapsed = time.monotonic() - start_ts
        print(f"[BT] block {block_idx}/{total_blocks} done (elapsed={elapsed:.1f}s, processed_days={processed_days}/{total_test_days})")

    if timeout_hit:
        elapsed = time.monotonic() - start_ts
        print(f"[BT] backtest stopped by time budget at {elapsed:.1f}s; partial results returned.")

    daily_frame = pd.DataFrame(records).sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    if daily_frame.empty:
        empty_curve = pd.DataFrame(columns=["date", "strategy_nav", "benchmark_nav", "excess_nav"])
        if news_enabled:
            return BacktestResult(
                daily_frame=daily_frame,
                curve_frame=empty_curve,
                metrics=[BacktestMetrics.empty("融合策略-全样本"), BacktestMetrics.empty("量化基线-全样本")],
                audit={"use_state_engine": bool(use_state_engine)},
            )
        return BacktestResult(
            daily_frame=daily_frame,
            curve_frame=empty_curve,
            metrics=[BacktestMetrics.empty("全样本")],
            audit={"use_state_engine": bool(use_state_engine)},
        )

    quant_frame = _to_strategy_frame(
        daily_frame,
        ret_col="quant_ret",
        turnover_col="turnover_quant",
        cost_col="cost_quant",
        trade_count_col="trade_count_quant",
    )
    fused_frame = _to_strategy_frame(
        daily_frame,
        ret_col="fused_ret",
        turnover_col="turnover_fused",
        cost_col="cost_fused",
        trade_count_col="trade_count_fused",
    )
    active_frame = fused_frame if news_enabled else quant_frame

    curve = active_frame[["date"]].copy()
    curve["strategy_nav"] = (1.0 + active_frame["strategy_ret"].astype(float)).cumprod()
    curve["benchmark_nav"] = (1.0 + active_frame["benchmark_ret"].astype(float)).cumprod()
    curve["excess_nav"] = (1.0 + active_frame["excess_ret"].astype(float)).cumprod()
    if news_enabled:
        curve["quant_nav"] = (1.0 + quant_frame["strategy_ret"].astype(float)).cumprod()
        curve["fused_nav"] = (1.0 + fused_frame["strategy_ret"].astype(float)).cumprod()

    if news_enabled:
        metrics = _build_window_metrics(fused_frame, window_years, prefix="融合策略-")
        metrics.extend(_build_window_metrics(quant_frame, window_years, prefix="量化基线-"))
    else:
        metrics = _build_window_metrics(quant_frame, window_years, prefix="")

    audit: dict[str, float | int | bool] = {
        "use_state_engine": bool(use_state_engine),
        "use_tradeability_guard": bool(use_tradeability_guard),
        "tradeability_limit_tolerance": float(tradeability_limit_tolerance),
        "tradeability_min_volume": float(tradeability_min_volume),
        "limit_rule_file_enabled": bool(str(limit_rule_file).strip()),
        "use_index_constituent_guard": bool(use_index_constituent_guard),
        "index_constituent_file_enabled": bool(str(index_constituent_file).strip()),
        "index_constituent_symbol": str(index_constituent_symbol).strip().upper(),
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
    return BacktestResult(daily_frame=daily_frame, curve_frame=curve, metrics=metrics, audit=audit)
