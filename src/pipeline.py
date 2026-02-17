from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .data import DataError, load_symbol_daily, normalize_symbol
from .features import (
    MARKET_FEATURE_COLUMNS,
    make_market_feature_frame,
    make_stock_feature_frame,
    stock_feature_columns,
)
from .model import BinaryMetrics, LogisticBinaryModel, binary_metrics


@dataclass(frozen=True)
class Security:
    symbol: str
    name: str


@dataclass
class ForecastRow:
    symbol: str
    name: str
    latest_date: pd.Timestamp
    short_prob: float
    mid_prob: float
    score: float
    short_drivers: List[str]
    mid_drivers: List[str]
    short_eval: BinaryMetrics
    mid_eval: BinaryMetrics
    suggested_weight: float = 0.0


@dataclass
class MarketForecast:
    symbol: str
    name: str
    latest_date: pd.Timestamp
    short_prob: float
    mid_prob: float
    short_eval: BinaryMetrics
    mid_eval: BinaryMetrics


def _empty_metrics() -> BinaryMetrics:
    return BinaryMetrics(n=0, accuracy=np.nan, brier=np.nan, auc=np.nan, base_rate=np.nan)


def _to_percent(v: float) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v * 100:.1f}%"


def _fit_latest_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    l2: float,
) -> LogisticBinaryModel:
    model = LogisticBinaryModel(l2=l2)
    model.fit(df, feature_cols=feature_cols, target_col=target_col)
    return model


def _walk_forward_eval(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    l2: float,
    min_train_days: int,
    step_days: int,
) -> BinaryMetrics:
    frame = df.dropna(subset=feature_cols + [target_col]).sort_values("date").copy()
    if frame.empty:
        return _empty_metrics()

    dates = frame["date"].drop_duplicates().sort_values().tolist()
    if len(dates) <= min_train_days:
        return _empty_metrics()

    all_prob: List[float] = []
    all_true: List[float] = []

    for i in range(min_train_days, len(dates), step_days):
        train_dates = dates[:i]
        test_dates = dates[i : i + step_days]
        if not test_dates:
            break

        train = frame[frame["date"].isin(train_dates)]
        test = frame[frame["date"].isin(test_dates)]
        if train.empty or test.empty:
            continue

        model = LogisticBinaryModel(l2=l2)
        model.fit(train, feature_cols=feature_cols, target_col=target_col)
        prob = model.predict_proba(test, feature_cols=feature_cols)
        all_prob.extend(prob.tolist())
        all_true.extend(test[target_col].astype(float).tolist())

    if not all_true:
        return _empty_metrics()
    return binary_metrics(np.asarray(all_true), np.asarray(all_prob))


def _latest_row_with_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    valid = df.dropna(subset=feature_cols).sort_values("date")
    if valid.empty:
        raise DataError("No valid rows with complete features.")
    return valid.iloc[-1]


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


def _apply_stock_weights(rows: List[ForecastRow], total_exposure: float) -> None:
    raw = np.array([max(0.0, row.score - 0.50) for row in rows], dtype=float)
    if np.all(raw <= 1e-12):
        equal = total_exposure / len(rows) if rows else 0.0
        for row in rows:
            row.suggested_weight = equal
        return
    raw = raw / raw.sum()
    alloc = raw * total_exposure
    for row, w in zip(rows, alloc.tolist()):
        row.suggested_weight = float(w)


def _metrics_line(metrics: BinaryMetrics) -> str:
    auc_text = f"{metrics.auc:.3f}" if not pd.isna(metrics.auc) else "NA"
    brier_text = f"{metrics.brier:.3f}" if not pd.isna(metrics.brier) else "NA"
    return f"n={metrics.n}, acc={_to_percent(metrics.accuracy)}, auc={auc_text}, brier={brier_text}"


def run_pipeline(
    market_security: Security,
    stock_securities: Sequence[Security],
    source: str,
    data_dir: str,
    start: str,
    end: str,
    min_train_days: int,
    step_days: int,
    l2: float,
) -> tuple[MarketForecast, List[ForecastRow]]:
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_feat = make_market_feature_frame(market_raw)

    market_short_model = _fit_latest_model(
        market_feat, feature_cols=MARKET_FEATURE_COLUMNS, target_col="mkt_target_1d_up", l2=l2
    )
    market_mid_model = _fit_latest_model(
        market_feat, feature_cols=MARKET_FEATURE_COLUMNS, target_col="mkt_target_20d_up", l2=l2
    )
    mkt_latest = _latest_row_with_features(market_feat, MARKET_FEATURE_COLUMNS)
    mkt_latest_df = pd.DataFrame([mkt_latest])

    market_forecast = MarketForecast(
        symbol=normalize_symbol(market_security.symbol).symbol,
        name=market_security.name,
        latest_date=pd.Timestamp(mkt_latest["date"]),
        short_prob=float(market_short_model.predict_proba(mkt_latest_df, MARKET_FEATURE_COLUMNS)[0]),
        mid_prob=float(market_mid_model.predict_proba(mkt_latest_df, MARKET_FEATURE_COLUMNS)[0]),
        short_eval=_walk_forward_eval(
            market_feat,
            feature_cols=MARKET_FEATURE_COLUMNS,
            target_col="mkt_target_1d_up",
            l2=l2,
            min_train_days=min_train_days,
            step_days=step_days,
        ),
        mid_eval=_walk_forward_eval(
            market_feat,
            feature_cols=MARKET_FEATURE_COLUMNS,
            target_col="mkt_target_20d_up",
            l2=l2,
            min_train_days=min_train_days,
            step_days=step_days,
        ),
    )

    feature_cols = stock_feature_columns()
    stock_rows: List[ForecastRow] = []
    for security in stock_securities:
        stock_raw = load_symbol_daily(
            symbol=security.symbol,
            source=source,
            data_dir=data_dir,
            start=start,
            end=end,
        )
        stock_feat = make_stock_feature_frame(stock_raw, market_feat)

        short_model = _fit_latest_model(stock_feat, feature_cols=feature_cols, target_col="target_1d_up", l2=l2)
        mid_model = _fit_latest_model(stock_feat, feature_cols=feature_cols, target_col="target_20d_up", l2=l2)
        latest_row = _latest_row_with_features(stock_feat, feature_cols)
        latest_df = pd.DataFrame([latest_row])

        short_prob = float(short_model.predict_proba(latest_df, feature_cols=feature_cols)[0])
        mid_prob = float(mid_model.predict_proba(latest_df, feature_cols=feature_cols)[0])
        score = 0.55 * short_prob + 0.45 * mid_prob

        stock_rows.append(
            ForecastRow(
                symbol=normalize_symbol(security.symbol).symbol,
                name=security.name,
                latest_date=pd.Timestamp(latest_row["date"]),
                short_prob=short_prob,
                mid_prob=mid_prob,
                score=score,
                short_drivers=short_model.top_drivers(latest_row, top_n=3),
                mid_drivers=mid_model.top_drivers(latest_row, top_n=3),
                short_eval=_walk_forward_eval(
                    stock_feat,
                    feature_cols=feature_cols,
                    target_col="target_1d_up",
                    l2=l2,
                    min_train_days=min_train_days,
                    step_days=step_days,
                ),
                mid_eval=_walk_forward_eval(
                    stock_feat,
                    feature_cols=feature_cols,
                    target_col="target_20d_up",
                    l2=l2,
                    min_train_days=min_train_days,
                    step_days=step_days,
                ),
            )
        )

    exposure = _target_exposure(market_forecast.short_prob, market_forecast.mid_prob)
    _apply_stock_weights(stock_rows, total_exposure=exposure)
    stock_rows.sort(key=lambda x: x.score, reverse=True)
    return market_forecast, stock_rows


def write_report(
    out_path: str | Path,
    market_forecast: MarketForecast,
    stock_rows: Sequence[ForecastRow],
) -> Path:
    report_path = Path(out_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    regime = _market_regime(market_forecast.short_prob, market_forecast.mid_prob)
    target_exposure = _target_exposure(market_forecast.short_prob, market_forecast.mid_prob)

    lines: List[str] = []
    lines.append("# A股多周期预测报告 (V1)")
    lines.append("")
    lines.append(f"- 数据截至: {market_forecast.latest_date.date()}")
    lines.append(f"- 市场基准: {market_forecast.name} ({market_forecast.symbol})")
    lines.append(f"- 市场状态: {regime}")
    lines.append(f"- 建议总仓位: {_to_percent(target_exposure)}")
    lines.append("")
    lines.append("## 大盘预测")
    lines.append("")
    lines.append("| 维度 | 上涨概率 | 说明 |")
    lines.append("|---|---:|---|")
    lines.append(f"| 短期(1日) | {_to_percent(market_forecast.short_prob)} | 次日方向概率 |")
    lines.append(f"| 中期(20日) | {_to_percent(market_forecast.mid_prob)} | 未来20日方向概率 |")
    lines.append("")
    lines.append(f"- 短期样本外: {_metrics_line(market_forecast.short_eval)}")
    lines.append(f"- 中期样本外: {_metrics_line(market_forecast.mid_eval)}")
    lines.append("")
    lines.append("## 个股预测")
    lines.append("")
    lines.append("| 个股 | 短期概率 | 中期概率 | 综合分数 | 建议权重 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in stock_rows:
        lines.append(
            f"| {row.name} ({row.symbol}) | {_to_percent(row.short_prob)} | {_to_percent(row.mid_prob)} | {row.score:.3f} | {_to_percent(row.suggested_weight)} |"
        )
    lines.append("")
    lines.append("## 因子解释 (最新截面)")
    lines.append("")
    for row in stock_rows:
        short_d = ", ".join(row.short_drivers) if row.short_drivers else "NA"
        mid_d = ", ".join(row.mid_drivers) if row.mid_drivers else "NA"
        lines.append(f"### {row.name} ({row.symbol})")
        lines.append(f"- 短期驱动: {short_d}")
        lines.append(f"- 中期驱动: {mid_d}")
        lines.append(f"- 短期样本外: {_metrics_line(row.short_eval)}")
        lines.append(f"- 中期样本外: {_metrics_line(row.mid_eval)}")
        lines.append("")

    lines.append("## 风险提示")
    lines.append("")
    lines.append("- 本报告为统计预测结果，不构成投资建议。")
    lines.append("- 若市场处于 Risk-Off，建议优先执行仓位约束。")
    lines.append("- 建议叠加基本面与事件风控，不建议仅凭单模型交易。")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
