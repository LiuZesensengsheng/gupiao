from __future__ import annotations

import pandas as pd

from scripts.backtest_dynamic300_rule_plan import DailyBar, _estimate_buy_fill_ratio, _exit_target_weight
from scripts.dynamic300_rule_utils import RuleCandidateRow, candidate_bucket, select_portfolio


def _row(
    *,
    symbol: str,
    theme: str,
    bucket: str | None = None,
    score: float = 1.0,
    fresh_pool_pass: bool = True,
    recent_high_gap20: float = -0.04,
    amount_ratio20: float = 1.10,
    close: float = 10.0,
    ma20: float = 9.8,
    ma60: float = 9.2,
    ret20: float = 0.08,
    ret60: float = 0.20,
    breakout_pos_120: float = 0.95,
    volatility20: float = 0.02,
    tradeability: float = 0.9,
) -> RuleCandidateRow:
    row = RuleCandidateRow(
        symbol=symbol,
        name=symbol,
        theme=theme,
        refined_score=score,
        fresh_pool_score=score,
        fresh_pool_pass=fresh_pool_pass,
        recent_high_gap20=recent_high_gap20,
        amount_ratio20=amount_ratio20,
        theme_selected_count=5,
        theme_strength=0.8,
        close=close,
        ma20=ma20,
        ma60=ma60,
        ret20=ret20,
        ret60=ret60,
        breakout_pos_120=breakout_pos_120,
        volatility20=volatility20,
        tradeability=tradeability,
        bucket="reserve",
        portfolio_score=score,
    )
    return row if bucket is None else RuleCandidateRow(**{**row.__dict__, "bucket": bucket})


def test_candidate_bucket_classifies_pullback() -> None:
    row = _row(
        symbol="000001.SZ",
        theme="主题A",
        fresh_pool_pass=True,
        recent_high_gap20=-0.07,
        amount_ratio20=1.00,
        close=10.2,
        ma20=10.0,
        ret20=0.06,
        breakout_pos_120=0.82,
    )
    assert candidate_bucket(row) == "pullback"


def test_select_portfolio_spreads_across_buckets() -> None:
    rows = [
        _row(symbol="000001.SZ", theme="主题A", bucket="trend", score=1.20),
        _row(symbol="000002.SZ", theme="主题B", bucket="trend", score=1.18),
        _row(symbol="000003.SZ", theme="主题C", bucket="pullback", score=1.05),
        _row(symbol="000004.SZ", theme="主题D", bucket="breakout", score=1.01),
    ]
    selected = select_portfolio(rows, top_n=3, max_per_theme=1)
    assert len(selected) == 3
    assert {row.bucket for row in selected} == {"trend", "pullback", "breakout"}


def test_buy_fill_ratio_blocks_gap_above_avoid_zone() -> None:
    row = _row(symbol="000001.SZ", theme="主题A", bucket="trend", close=10.0, ma20=9.8)
    bar = DailyBar(
        date=pd.Timestamp("2026-03-20"),
        open=10.50,
        high=10.60,
        low=10.30,
        close=10.42,
        volume=1000000.0,
        amount=10000000.0,
        pre_close=10.0,
        prev_close=10.0,
        ret_cc=0.042,
        ma20=9.9,
        ma60=9.4,
        prev_ma20=9.85,
        prev_ma60=9.35,
    )
    assert _estimate_buy_fill_ratio(row, bar) == 0.0


def test_exit_target_weight_exits_below_ma60() -> None:
    bar = DailyBar(
        date=pd.Timestamp("2026-03-20"),
        open=9.15,
        high=9.20,
        low=8.88,
        close=8.95,
        volume=1000000.0,
        amount=10000000.0,
        pre_close=9.35,
        prev_close=9.35,
        ret_cc=-0.0428,
        ma20=9.18,
        ma60=9.12,
        prev_ma20=9.25,
        prev_ma60=9.10,
    )
    assert _exit_target_weight(0.25, bar) == 0.0
