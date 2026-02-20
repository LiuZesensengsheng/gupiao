#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.data import DataError
from src.effect_analysis import EffectSummary, build_latest_snapshot, compute_effect_summary, compute_sector_table
from src.news_matrix import SentimentAggregate, aggregate_sentiment, blend_probability, load_news_items
from src.pipeline import ForecastRow, Security, run_pipeline


@dataclass
class BlendedRow:
    symbol: str
    name: str
    base_short: float
    base_mid: float
    final_short: float
    final_mid: float
    final_score: float
    short_sent: SentimentAggregate
    mid_sent: SentimentAggregate
    suggested_weight: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily A-share report with quant baseline + fuzzy news matrix")
    parser.add_argument("--source", default="eastmoney", choices=["eastmoney", "local"], help="Data source")
    parser.add_argument("--watchlist", default="config/watchlist.json", help="Watchlist JSON path")
    parser.add_argument("--data-dir", default="data", help="Directory for local CSV when source=local")
    parser.add_argument("--start", default="2018-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2099-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--min-train-days", type=int, default=240, help="Min train days for walk-forward")
    parser.add_argument("--step-days", type=int, default=20, help="Walk-forward test block size")
    parser.add_argument("--l2", type=float, default=0.8, help="L2 regularization strength")
    parser.add_argument("--news-file", default="input/news.csv", help="CSV file for news events")
    parser.add_argument("--news-lookback-days", type=int, default=45, help="News lookback window in days")
    parser.add_argument("--news-half-life-days", type=float, default=10.0, help="Decay half-life for news")
    parser.add_argument("--market-news-strength", type=float, default=0.9, help="Market news blend strength")
    parser.add_argument("--stock-news-strength", type=float, default=1.1, help="Stock news blend strength")
    parser.add_argument("--report-date", default="", help="Override report date YYYY-MM-DD")
    parser.add_argument("--report", default="reports/daily_report.md", help="Output report path")
    return parser.parse_args()


def load_watchlist(path: str | Path) -> tuple[Security, List[Security], Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    market = payload["market_index"]
    market_security = Security(symbol=market["symbol"], name=market["name"])
    stocks = [Security(symbol=item["symbol"], name=item["name"]) for item in payload["stocks"]]
    sector_map: Dict[str, str] = {}
    for item in payload["stocks"]:
        symbol = item["symbol"].strip().upper()
        sector_map[symbol] = item.get("sector", "其他")
    return market_security, stocks, sector_map


def _to_percent(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 100:.1f}%"


def _to_bp(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 10000:.0f}bp"


def _market_regime(short_prob: float, mid_prob: float) -> str:
    score = 0.6 * short_prob + 0.4 * mid_prob
    if score >= 0.60:
        return "偏强 (Risk-On)"
    if score <= 0.40:
        return "偏弱 (Risk-Off)"
    return "中性 (Neutral)"


def _target_exposure(short_prob: float, mid_prob: float) -> float:
    score = 0.6 * short_prob + 0.4 * mid_prob
    if score <= 0.40:
        return 0.25
    if score <= 0.50:
        return 0.40
    if score <= 0.60:
        return 0.55
    return 0.72


def _apply_stock_weights(rows: List[BlendedRow], total_exposure: float) -> None:
    raw = np.array([max(0.0, row.final_score - 0.50) for row in rows], dtype=float)
    if np.all(raw <= 1e-12):
        equal = total_exposure / len(rows) if rows else 0.0
        for row in rows:
            row.suggested_weight = equal
        return
    raw = raw / raw.sum()
    alloc = raw * total_exposure
    for row, w in zip(rows, alloc.tolist()):
        row.suggested_weight = float(w)


def _blend_stock_rows(
    rows: List[ForecastRow],
    news_items: List,
    as_of_date: pd.Timestamp,
    stock_news_strength: float,
    news_half_life_days: float,
) -> List[BlendedRow]:
    out: List[BlendedRow] = []
    for row in rows:
        short_sent = aggregate_sentiment(
            news_items=news_items,
            as_of_date=as_of_date,
            target=row.symbol,
            horizon="short",
            half_life_days=news_half_life_days,
        )
        mid_sent = aggregate_sentiment(
            news_items=news_items,
            as_of_date=as_of_date,
            target=row.symbol,
            horizon="mid",
            half_life_days=news_half_life_days,
        )
        final_short = blend_probability(row.short_prob, short_sent.score, sentiment_strength=stock_news_strength)
        final_mid = blend_probability(row.mid_prob, mid_sent.score, sentiment_strength=stock_news_strength)
        final_score = 0.55 * final_short + 0.45 * final_mid

        out.append(
            BlendedRow(
                symbol=row.symbol,
                name=row.name,
                base_short=row.short_prob,
                base_mid=row.mid_prob,
                final_short=final_short,
                final_mid=final_mid,
                final_score=final_score,
                short_sent=short_sent,
                mid_sent=mid_sent,
            )
        )
    return out


def _write_daily_report(
    out_path: str | Path,
    as_of_date: pd.Timestamp,
    source: str,
    market_name: str,
    market_symbol: str,
    market_base_short: float,
    market_base_mid: float,
    market_final_short: float,
    market_final_mid: float,
    market_short_sent: SentimentAggregate,
    market_mid_sent: SentimentAggregate,
    stock_rows: List[BlendedRow],
    effect_summary: EffectSummary,
    sector_table: pd.DataFrame,
    news_items_count: int,
) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    regime = _market_regime(market_final_short, market_final_mid)
    exposure = _target_exposure(market_final_short, market_final_mid)

    lines: List[str] = []
    lines.append("# A股每日融合报告 (量化 + 新闻模糊矩阵)")
    lines.append("")
    lines.append(f"- 报告日期: {as_of_date.date()}")
    lines.append(f"- 数据源: {source}")
    lines.append(f"- 市场基准: {market_name} ({market_symbol})")
    lines.append(f"- 近窗新闻条数: {news_items_count}")
    lines.append(f"- 市场状态: {regime}")
    lines.append(f"- 建议总仓位: {_to_percent(exposure)}")
    lines.append("")
    lines.append("## 大盘融合结果")
    lines.append("")
    lines.append("| 维度 | 模型概率 | 新闻净分 | 融合后概率 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| 短期(1日) | {_to_percent(market_base_short)} | {market_short_sent.score:+.3f} | {_to_percent(market_final_short)} |"
    )
    lines.append(
        f"| 中期(20日) | {_to_percent(market_base_mid)} | {market_mid_sent.score:+.3f} | {_to_percent(market_final_mid)} |"
    )
    lines.append("")
    lines.append("## 个股融合结果")
    lines.append("")
    lines.append("| 个股 | 模型短期 | 模型中期 | 新闻短期净分 | 新闻中期净分 | 融合短期 | 融合中期 | 综合分数 | 建议权重 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(stock_rows, key=lambda x: x.final_score, reverse=True):
        lines.append(
            f"| {row.name} ({row.symbol}) | {_to_percent(row.base_short)} | {_to_percent(row.base_mid)} | {row.short_sent.score:+.3f} | {row.mid_sent.score:+.3f} | {_to_percent(row.final_short)} | {_to_percent(row.final_mid)} | {row.final_score:.3f} | {_to_percent(row.suggested_weight)} |"
        )
    lines.append("")
    lines.append("## 新闻模糊矩阵")
    lines.append("")
    lines.append("| 标的 | 维度 | 利好隶属 | 利空隶属 | 中性隶属 | 事件条数 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    lines.append(
        f"| {market_name} ({market_symbol}) | 短期 | {market_short_sent.bullish:.3f} | {market_short_sent.bearish:.3f} | {market_short_sent.neutral:.3f} | {market_short_sent.items} |"
    )
    lines.append(
        f"| {market_name} ({market_symbol}) | 中期 | {market_mid_sent.bullish:.3f} | {market_mid_sent.bearish:.3f} | {market_mid_sent.neutral:.3f} | {market_mid_sent.items} |"
    )
    for row in sorted(stock_rows, key=lambda x: x.final_score, reverse=True):
        lines.append(
            f"| {row.name} ({row.symbol}) | 短期 | {row.short_sent.bullish:.3f} | {row.short_sent.bearish:.3f} | {row.short_sent.neutral:.3f} | {row.short_sent.items} |"
        )
        lines.append(
            f"| {row.name} ({row.symbol}) | 中期 | {row.mid_sent.bullish:.3f} | {row.mid_sent.bearish:.3f} | {row.mid_sent.neutral:.3f} | {row.mid_sent.items} |"
        )
    lines.append("")
    lines.append("## 市场效应分析")
    lines.append("")
    lines.append(f"- 样本覆盖: {effect_summary.sample_size} 只股票")
    lines.append(
        f"- 赚钱效应: {effect_summary.pnl_label} | 1日胜率 {_to_percent(effect_summary.win_rate_1d)} | 5日胜率 {_to_percent(effect_summary.win_rate_5d)} | 强势率(5日>3%) {_to_percent(effect_summary.strong_rate_5d)}"
    )
    lines.append(
        f"- 亏钱效应: 风险{effect_summary.risk_label} | 1日亏损率 {_to_percent(effect_summary.loss_rate_1d)} | 深亏率 {_to_percent(effect_summary.deep_loss_rate)} | 20日中位回撤 {_to_percent(effect_summary.median_drawdown_20)}"
    )
    lines.append(
        f"- 资金状态: {effect_summary.money_label} | 量能比(20日) {effect_summary.avg_vol_ratio_20:.2f} | OBV Z值 {effect_summary.avg_obv_z_20:.2f} | 资金分数 {effect_summary.money_score:+.3f}"
    )
    lines.append(
        f"- 筹码结构: {effect_summary.chip_label} | 20日位置 {effect_summary.avg_price_pos_20:.2f} | 量能集中度(5/20) {effect_summary.avg_vol_conc_5_20:.2f} | 筹码分数 {effect_summary.chip_score:+.3f}"
    )
    lines.append("")
    lines.append("## 板块热度")
    lines.append("")
    if sector_table.empty:
        lines.append("- 无可用板块数据")
    else:
        lines.append("| 板块 | 热度分数 | 1日胜率 | 5日中位收益 | 资金分数 | 筹码分数 | 样本数 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, row in sector_table.iterrows():
            lines.append(
                f"| {row['sector']} | {row['heat_score']:+.3f} | {_to_percent(float(row['win_rate_1d']))} | {_to_bp(float(row['median_ret_5d']))} | {float(row['money_score']):+.3f} | {float(row['chip_score']):+.3f} | {int(row['count'])} |"
            )
    lines.append("")
    lines.append("## 使用说明")
    lines.append("")
    lines.append("- 新闻净分范围为 [-1, +1]，正值偏利好，负值偏利空。")
    lines.append("- 融合后概率 = 模型概率经过新闻净分的非线性修正。")
    lines.append("- 若你希望更保守，可降低 `--stock-news-strength` 与 `--market-news-strength`。")
    lines.append("- 本结果为研究支持，不构成投资建议。")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    market_security, stocks, sector_map = load_watchlist(args.watchlist)

    try:
        market_forecast, stock_rows = run_pipeline(
            market_security=market_security,
            stock_securities=stocks,
            source=args.source,
            data_dir=args.data_dir,
            start=args.start,
            end=args.end,
            min_train_days=args.min_train_days,
            step_days=args.step_days,
            l2=args.l2,
        )
    except DataError as exc:
        print(f"[ERROR] {exc}")
        print("Hint: if online source fails, use `--source local --data-dir data`.")
        return 2
    except Exception as exc:
        print(f"[ERROR] Unexpected failure: {exc}")
        return 3

    as_of_date = pd.Timestamp(market_forecast.latest_date).normalize()
    if args.report_date:
        as_of_date = pd.Timestamp(args.report_date).normalize()

    try:
        news_items = load_news_items(args.news_file, as_of_date=as_of_date, lookback_days=args.news_lookback_days)
    except Exception as exc:
        print(f"[ERROR] News file parse failure: {exc}")
        return 4

    market_short_sent = aggregate_sentiment(
        news_items=news_items,
        as_of_date=as_of_date,
        target="MARKET",
        horizon="short",
        half_life_days=args.news_half_life_days,
    )
    market_mid_sent = aggregate_sentiment(
        news_items=news_items,
        as_of_date=as_of_date,
        target="MARKET",
        horizon="mid",
        half_life_days=args.news_half_life_days,
    )

    market_final_short = blend_probability(
        market_forecast.short_prob,
        market_short_sent.score,
        sentiment_strength=args.market_news_strength,
    )
    market_final_mid = blend_probability(
        market_forecast.mid_prob,
        market_mid_sent.score,
        sentiment_strength=args.market_news_strength,
    )

    blended_rows = _blend_stock_rows(
        rows=stock_rows,
        news_items=news_items,
        as_of_date=as_of_date,
        stock_news_strength=args.stock_news_strength,
        news_half_life_days=args.news_half_life_days,
    )
    total_exposure = _target_exposure(market_final_short, market_final_mid)
    _apply_stock_weights(blended_rows, total_exposure=total_exposure)

    snapshot = build_latest_snapshot(
        source=args.source,
        data_dir=args.data_dir,
        start=args.start,
        end=args.end,
        stocks=stocks,
        sector_map=sector_map,
    )
    effect_summary = compute_effect_summary(snapshot)
    sector_table = compute_sector_table(snapshot)

    path = _write_daily_report(
        out_path=args.report,
        as_of_date=as_of_date,
        source=args.source,
        market_name=market_forecast.name,
        market_symbol=market_forecast.symbol,
        market_base_short=market_forecast.short_prob,
        market_base_mid=market_forecast.mid_prob,
        market_final_short=market_final_short,
        market_final_mid=market_final_mid,
        market_short_sent=market_short_sent,
        market_mid_sent=market_mid_sent,
        stock_rows=blended_rows,
        effect_summary=effect_summary,
        sector_table=sector_table,
        news_items_count=len(news_items),
    )
    print(f"[OK] Daily report generated: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
