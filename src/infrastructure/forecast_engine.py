from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from typing import Sequence

import numpy as np
import pandas as pd

from src.domain.entities import BinaryMetrics, ForecastRow, MarketForecast, Security
from src.domain.policies import allocate_weights, blend_horizon_score, target_exposure
from src.domain.symbols import normalize_symbol
from src.infrastructure.features import (
    MARKET_FEATURE_COLUMNS,
    MID_RETURN_BUCKET_EDGES,
    SHORT_RETURN_BUCKET_EDGES,
    make_market_feature_frame,
)
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import DataError, load_symbol_daily
from src.infrastructure.modeling import LogisticBinaryModel, QuantileLinearModel, binary_metrics
from src.infrastructure.panel_dataset import StockPanelDataset, build_stock_panel_dataset


SHORT_BUCKET_LABELS = ["大跌", "小跌", "震荡", "小涨", "大涨"]
MID_BUCKET_LABELS = ["大跌", "小跌", "震荡", "小涨", "大涨"]
SHORT_BUCKET_REPRESENTATIVES = [-0.04, -0.0125, 0.0, 0.0125, 0.04]
MID_BUCKET_REPRESENTATIVES = [-0.16, -0.06, 0.0, 0.06, 0.16]
QUANTILE_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.90)


def _is_actionable_status(status: str) -> bool:
    return str(status) not in {"halted", "delisted"}


def _adjust_score_for_status(score: float, status: str) -> float:
    status = str(status)
    if status == "delisted":
        return -1e6
    if status == "halted":
        return min(float(score), 0.05)
    if status == "data_insufficient":
        return float(score) * 0.65
    return float(score)


@dataclass(frozen=True)
class ReturnBucketProfile:
    bucket_probs: list[float]
    expected_return: float
    q10: float
    q30: float
    q20: float
    q50: float
    q70: float
    q80: float
    q90: float


def _fit_latest_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> LogisticBinaryModel:
    model = LogisticBinaryModel(l2=l2)
    model.fit(df, feature_cols=feature_cols, target_col=target_col)
    return model


def _walk_forward_eval(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    l2: float,
    min_train_days: int,
    step_days: int,
) -> BinaryMetrics:
    frame = df.dropna(subset=feature_cols + [target_col]).sort_values("date").copy()
    if frame.empty:
        return BinaryMetrics.empty()

    dates = frame["date"].drop_duplicates().sort_values().tolist()
    if len(dates) <= min_train_days:
        return BinaryMetrics.empty()

    all_prob: list[float] = []
    all_true: list[float] = []

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
        return BinaryMetrics.empty()
    return binary_metrics(np.asarray(all_true), np.asarray(all_prob))


def _latest_row_with_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    valid = df.dropna(subset=feature_cols).sort_values("date")
    if valid.empty:
        raise DataError("No valid rows with complete features.")
    return valid.iloc[-1]


def _bucket_probs_from_quantiles(
    *,
    thresholds: Sequence[float],
    q10: float,
    q30: float,
    q50: float,
    q70: float,
    q90: float,
) -> list[float]:
    step_lo = max(1e-6, abs(float(q30) - float(q10)))
    step_hi = max(1e-6, abs(float(q90) - float(q70)))
    x_points = np.asarray(
        [
            float(q10 - 1.5 * step_lo),
            float(q10),
            float(q30),
            float(q50),
            float(q70),
            float(q90),
            float(q90 + 1.5 * step_hi),
        ],
        dtype=float,
    )
    y_points = np.asarray([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=float)
    finite_edges = [float(edge) for edge in thresholds]
    cdf_values = [0.0]
    for edge in finite_edges:
        cdf_values.append(float(np.clip(np.interp(edge, x_points, y_points), 0.0, 1.0)))
    cdf_values.append(1.0)

    bucket_probs: list[float] = []
    for left, right in zip(cdf_values[:-1], cdf_values[1:]):
        bucket_probs.append(float(np.clip(right - left, 0.0, 1.0)))
    total = float(sum(bucket_probs))
    if total <= 1e-9:
        return [1.0 / max(1, len(bucket_probs))] * max(1, len(bucket_probs))
    return [float(x) / total for x in bucket_probs]


def estimate_return_quantiles(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    return_col: str,
    thresholds: Sequence[float],
    representatives: Sequence[float],
    l2: float,
) -> ReturnBucketProfile:
    frame = df.dropna(subset=feature_cols + [return_col]).sort_values("date").copy()
    if frame.empty:
        fallback = [1.0 / max(1, len(representatives))] * max(1, len(representatives))
        mid = float(np.median(np.asarray(representatives, dtype=float))) if representatives else 0.0
        return ReturnBucketProfile(
            bucket_probs=[float(x) for x in fallback],
            expected_return=mid,
            q10=mid,
            q30=mid,
            q20=mid,
            q50=mid,
            q70=mid,
            q80=mid,
            q90=mid,
        )

    latest = frame.dropna(subset=feature_cols).sort_values("date").iloc[[-1]]
    q_models = [
        QuantileLinearModel(quantile=quantile, l2=l2).fit(frame, feature_cols, return_col)
        for quantile in QUANTILE_LEVELS
    ]
    quantiles = [float(model.predict(latest, feature_cols)[0]) for model in q_models]
    q10, q30, q50, q70, q90 = [float(x) for x in np.maximum.accumulate(np.asarray(quantiles, dtype=float))]
    q20 = float(0.5 * (q10 + q30))
    q80 = float(0.5 * (q70 + q90))

    bucket_probs = _bucket_probs_from_quantiles(
        thresholds=thresholds,
        q10=q10,
        q30=q30,
        q50=q50,
        q70=q70,
        q90=q90,
    )
    expected_return = float(0.10 * q10 + 0.20 * q30 + 0.40 * q50 + 0.20 * q70 + 0.10 * q90)
    return ReturnBucketProfile(
        bucket_probs=[float(x) for x in bucket_probs],
        expected_return=float(expected_return),
        q10=float(q10),
        q30=float(q30),
        q20=float(q20),
        q50=float(q50),
        q70=float(q70),
        q80=float(q80),
        q90=float(q90),
    )


def estimate_return_bucket_profile(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    return_col: str,
    thresholds: Sequence[float],
    representatives: Sequence[float],
    l2: float,
) -> ReturnBucketProfile:
    return estimate_return_quantiles(
        df,
        feature_cols=feature_cols,
        return_col=return_col,
        thresholds=thresholds,
        representatives=representatives,
        l2=l2,
    )


def _distributional_score(
    *,
    short_prob: float,
    two_prob: float,
    three_prob: float,
    five_prob: float,
    mid_prob: float,
    short_expected_ret: float,
    mid_expected_ret: float,
) -> float:
    base_score = float(
        0.14 * float(short_prob)
        + 0.18 * float(two_prob)
        + 0.22 * float(three_prob)
        + 0.28 * float(five_prob)
        + 0.18 * float(mid_prob)
    )
    short_ret_score = float(np.clip(0.5 + float(short_expected_ret) / 0.06, 0.0, 1.0))
    two_ret_score = float(
        np.clip(0.5 + (0.75 * float(short_expected_ret) + 0.25 * float(mid_expected_ret)) / 0.08, 0.0, 1.0)
    )
    three_ret_score = float(
        np.clip(0.5 + (0.60 * float(short_expected_ret) + 0.40 * float(mid_expected_ret)) / 0.10, 0.0, 1.0)
    )
    five_ret_score = float(
        np.clip(0.5 + (0.35 * float(short_expected_ret) + 0.65 * float(mid_expected_ret)) / 0.12, 0.0, 1.0)
    )
    mid_ret_score = float(np.clip(0.5 + float(mid_expected_ret) / 0.20, 0.0, 1.0))
    dist_score = float(
        0.14 * short_ret_score
        + 0.18 * two_ret_score
        + 0.22 * three_ret_score
        + 0.24 * five_ret_score
        + 0.22 * mid_ret_score
    )
    return float(0.4 * base_score + 0.6 * dist_score)


def _fit_quantile_quintet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel]:
    return tuple(
        QuantileLinearModel(quantile=quantile, l2=l2).fit(df, feature_cols, target_col)
        for quantile in QUANTILE_LEVELS
    )


def _predict_quantile_profile_from_models(
    latest_row: pd.DataFrame,
    *,
    feature_cols: list[str],
    thresholds: Sequence[float],
    q_models: tuple[QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel],
) -> ReturnBucketProfile:
    quantiles = [float(model.predict(latest_row, feature_cols)[0]) for model in q_models]
    q10, q30, q50, q70, q90 = [float(x) for x in np.maximum.accumulate(np.asarray(quantiles, dtype=float))]
    q20 = float(0.5 * (q10 + q30))
    q80 = float(0.5 * (q70 + q90))
    bucket_probs = _bucket_probs_from_quantiles(
        thresholds=thresholds,
        q10=q10,
        q30=q30,
        q50=q50,
        q70=q70,
        q90=q90,
    )
    expected_return = float(0.10 * q10 + 0.20 * q30 + 0.40 * q50 + 0.20 * q70 + 0.10 * q90)
    return ReturnBucketProfile(
        bucket_probs=bucket_probs,
        expected_return=expected_return,
        q10=q10,
        q30=q30,
        q20=q20,
        q50=q50,
        q70=q70,
        q80=q80,
        q90=q90,
    )


def run_quant_pipeline(
    market_security: Security,
    stock_securities: Sequence[Security],
    source: str,
    data_dir: str,
    start: str,
    end: str,
    min_train_days: int,
    step_days: int,
    l2: float,
    max_positions: int = 5,
    weight_threshold: float = 0.50,
    use_margin_features: bool = True,
    margin_market_file: str = "input/margin_market.csv",
    margin_stock_file: str = "input/margin_stock.csv",
    enable_walk_forward_eval: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[MarketForecast, list[ForecastRow]]:
    def _notify(message: str) -> None:
        if progress_callback is not None:
            progress_callback(str(message))

    _notify("开始拟合市场模型")
    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        market_dates=market_feat_base["date"],
        use_margin_features=use_margin_features,
        margin_market_file=margin_market_file,
    )
    market_feat = market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")
    market_feature_cols = MARKET_FEATURE_COLUMNS + market_context.feature_columns

    market_short_model = _fit_latest_model(
        market_feat, feature_cols=market_feature_cols, target_col="mkt_target_1d_up", l2=l2
    )
    market_two_model = _fit_latest_model(
        market_feat, feature_cols=market_feature_cols, target_col="mkt_target_2d_up", l2=l2
    )
    market_three_model = _fit_latest_model(
        market_feat, feature_cols=market_feature_cols, target_col="mkt_target_3d_up", l2=l2
    )
    market_five_model = _fit_latest_model(
        market_feat, feature_cols=market_feature_cols, target_col="mkt_target_5d_up", l2=l2
    )
    market_mid_model = _fit_latest_model(
        market_feat, feature_cols=market_feature_cols, target_col="mkt_target_20d_up", l2=l2
    )
    mkt_latest = _latest_row_with_features(market_feat, market_feature_cols)
    mkt_latest_df = pd.DataFrame([mkt_latest])

    market_forecast = MarketForecast(
        symbol=normalize_symbol(market_security.symbol).symbol,
        name=market_security.name,
        latest_date=pd.Timestamp(mkt_latest["date"]),
        short_prob=float(market_short_model.predict_proba(mkt_latest_df, market_feature_cols)[0]),
        two_prob=float(market_two_model.predict_proba(mkt_latest_df, market_feature_cols)[0]),
        three_prob=float(market_three_model.predict_proba(mkt_latest_df, market_feature_cols)[0]),
        five_prob=float(market_five_model.predict_proba(mkt_latest_df, market_feature_cols)[0]),
        mid_prob=float(market_mid_model.predict_proba(mkt_latest_df, market_feature_cols)[0]),
        short_eval=(
            _walk_forward_eval(
                market_feat,
                feature_cols=market_feature_cols,
                target_col="mkt_target_1d_up",
                l2=l2,
                min_train_days=min_train_days,
                step_days=step_days,
            )
            if enable_walk_forward_eval
            else BinaryMetrics.empty()
        ),
        mid_eval=(
            _walk_forward_eval(
                market_feat,
                feature_cols=market_feature_cols,
                target_col="mkt_target_20d_up",
                l2=l2,
                min_train_days=min_train_days,
                step_days=step_days,
            )
            if enable_walk_forward_eval
            else BinaryMetrics.empty()
        ),
    )
    market_short_bucket = estimate_return_bucket_profile(
        market_feat,
        feature_cols=market_feature_cols,
        return_col="mkt_fwd_ret_1",
        thresholds=SHORT_RETURN_BUCKET_EDGES[1:-1],
        representatives=SHORT_BUCKET_REPRESENTATIVES,
        l2=l2,
    )
    market_mid_bucket = estimate_return_bucket_profile(
        market_feat,
        feature_cols=market_feature_cols,
        return_col="mkt_fwd_ret_20",
        thresholds=MID_RETURN_BUCKET_EDGES[1:-1],
        representatives=MID_BUCKET_REPRESENTATIVES,
        l2=l2,
    )
    market_forecast.short_expected_ret = float(market_short_bucket.expected_return)
    market_forecast.mid_expected_ret = float(market_mid_bucket.expected_return)
    market_forecast.short_q10 = float(market_short_bucket.q10)
    market_forecast.short_q30 = float(market_short_bucket.q30)
    market_forecast.short_q20 = float(market_short_bucket.q20)
    market_forecast.short_q50 = float(market_short_bucket.q50)
    market_forecast.short_q70 = float(market_short_bucket.q70)
    market_forecast.short_q80 = float(market_short_bucket.q80)
    market_forecast.short_q90 = float(market_short_bucket.q90)
    market_forecast.mid_q10 = float(market_mid_bucket.q10)
    market_forecast.mid_q30 = float(market_mid_bucket.q30)
    market_forecast.mid_q20 = float(market_mid_bucket.q20)
    market_forecast.mid_q50 = float(market_mid_bucket.q50)
    market_forecast.mid_q70 = float(market_mid_bucket.q70)
    market_forecast.mid_q80 = float(market_mid_bucket.q80)
    market_forecast.mid_q90 = float(market_mid_bucket.q90)
    market_forecast.short_bucket_probs = list(market_short_bucket.bucket_probs)
    market_forecast.mid_bucket_probs = list(market_mid_bucket.bucket_probs)

    _notify(f"开始构建股票面板: universe={len(stock_securities)}")
    panel_bundle: StockPanelDataset = build_stock_panel_dataset(
        stock_securities=stock_securities,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        market_frame=market_feat,
        extra_market_cols=market_context.feature_columns,
        use_margin_features=use_margin_features,
        margin_stock_file=margin_stock_file,
    )
    panel = panel_bundle.frame
    feature_cols = panel_bundle.feature_columns

    if panel.empty or not feature_cols:
        raise DataError("No valid stock rows with complete panel features for selected universe.")

    _notify(f"开始拟合股票面板模型: rows={len(panel)}, features={len(feature_cols)}")
    panel_short_model = _fit_latest_model(panel, feature_cols=feature_cols, target_col="target_1d_excess_mkt_up", l2=l2)
    panel_two_model = _fit_latest_model(panel, feature_cols=feature_cols, target_col="target_2d_excess_mkt_up", l2=l2)
    panel_three_model = _fit_latest_model(panel, feature_cols=feature_cols, target_col="target_3d_excess_mkt_up", l2=l2)
    panel_five_model = _fit_latest_model(panel, feature_cols=feature_cols, target_col="target_5d_excess_mkt_up", l2=l2)
    panel_mid_model = _fit_latest_model(panel, feature_cols=feature_cols, target_col="target_20d_excess_sector_up", l2=l2)
    panel_short_q_models = _fit_quantile_quintet(
        panel,
        feature_cols=feature_cols,
        target_col="excess_ret_1_vs_mkt",
        l2=l2,
    )
    panel_mid_q_models = _fit_quantile_quintet(
        panel,
        feature_cols=feature_cols,
        target_col="excess_ret_20_vs_sector",
        l2=l2,
    )

    panel_short_eval = (
        _walk_forward_eval(
            panel,
            feature_cols=feature_cols,
            target_col="target_1d_excess_mkt_up",
            l2=l2,
            min_train_days=min_train_days,
            step_days=step_days,
        )
        if enable_walk_forward_eval
        else BinaryMetrics.empty()
    )
    panel_mid_eval = (
        _walk_forward_eval(
            panel,
            feature_cols=feature_cols,
            target_col="target_20d_excess_sector_up",
            l2=l2,
            min_train_days=min_train_days,
            step_days=step_days,
        )
        if enable_walk_forward_eval
        else BinaryMetrics.empty()
    )

    latest_dates = panel.groupby("symbol", observed=True)["date"].max().rename("latest_date")
    latest_panel = panel.merge(latest_dates, on="symbol", how="inner")
    latest_panel = latest_panel[latest_panel["date"] == latest_panel["latest_date"]].copy()
    latest_panel = latest_panel.sort_values(["date", "symbol"]).drop_duplicates(subset=["symbol"], keep="last")
    latest_panel = latest_panel.drop(columns=["latest_date"])
    if latest_panel.empty:
        raise DataError("No latest panel rows available for stock prediction.")

    _notify(f"开始批量股票打分: symbols={len(latest_panel)}")
    short_probs = panel_short_model.predict_proba(latest_panel, feature_cols=feature_cols)
    two_probs = panel_two_model.predict_proba(latest_panel, feature_cols=feature_cols)
    three_probs = panel_three_model.predict_proba(latest_panel, feature_cols=feature_cols)
    five_probs = panel_five_model.predict_proba(latest_panel, feature_cols=feature_cols)
    mid_probs = panel_mid_model.predict_proba(latest_panel, feature_cols=feature_cols)

    stock_rows: list[ForecastRow] = []
    for idx, (_, latest_row) in enumerate(latest_panel.iterrows()):
        latest_df = pd.DataFrame([latest_row])
        short_bucket = _predict_quantile_profile_from_models(
            latest_df,
            feature_cols=feature_cols,
            thresholds=SHORT_RETURN_BUCKET_EDGES[1:-1],
            q_models=panel_short_q_models,
        )
        mid_bucket = _predict_quantile_profile_from_models(
            latest_df,
            feature_cols=feature_cols,
            thresholds=MID_RETURN_BUCKET_EDGES[1:-1],
            q_models=panel_mid_q_models,
        )
        short_prob = float(short_probs[idx])
        two_prob = float(two_probs[idx])
        three_prob = float(three_probs[idx])
        five_prob = float(five_probs[idx])
        mid_prob = float(mid_probs[idx])
        status = str(latest_row.get("tradability_status", "normal") or "normal")
        score = _adjust_score_for_status(
            _distributional_score(
                short_prob=short_prob,
                two_prob=two_prob,
                three_prob=three_prob,
                five_prob=five_prob,
                mid_prob=mid_prob,
                short_expected_ret=float(short_bucket.expected_return),
                mid_expected_ret=float(mid_bucket.expected_return),
            ),
            status,
        )
        symbol = normalize_symbol(str(latest_row["symbol"])).symbol
        stock_rows.append(
            ForecastRow(
                symbol=symbol,
                name=str(latest_row.get("name", symbol)),
                latest_date=pd.Timestamp(latest_row["date"]),
                short_prob=short_prob,
                two_prob=two_prob,
                three_prob=three_prob,
                five_prob=five_prob,
                mid_prob=mid_prob,
                score=score,
                short_drivers=panel_short_model.top_drivers(latest_row, top_n=3),
                mid_drivers=panel_mid_model.top_drivers(latest_row, top_n=3),
                short_eval=panel_short_eval,
                mid_eval=panel_mid_eval,
                short_expected_ret=float(short_bucket.expected_return),
                mid_expected_ret=float(mid_bucket.expected_return),
                short_q10=float(short_bucket.q10),
                short_q30=float(short_bucket.q30),
                short_q20=float(short_bucket.q20),
                short_q50=float(short_bucket.q50),
                short_q70=float(short_bucket.q70),
                short_q80=float(short_bucket.q80),
                short_q90=float(short_bucket.q90),
                mid_q10=float(mid_bucket.q10),
                mid_q30=float(mid_bucket.q30),
                mid_q20=float(mid_bucket.q20),
                mid_q50=float(mid_bucket.q50),
                mid_q70=float(mid_bucket.q70),
                mid_q80=float(mid_bucket.q80),
                mid_q90=float(mid_bucket.q90),
                short_bucket_probs=list(short_bucket.bucket_probs),
                mid_bucket_probs=list(mid_bucket.bucket_probs),
                tradability_status=status,
            )
        )

    if not stock_rows:
        raise DataError("No valid stock rows with complete features for selected universe.")

    total_exposure = target_exposure(market_forecast.short_prob, market_forecast.mid_prob)
    tradable_indices = [
        idx for idx, row in enumerate(stock_rows)
        if _is_actionable_status(row.tradability_status)
    ]
    tradable_rows = [stock_rows[idx] for idx in tradable_indices]
    tradable_weights = allocate_weights(
        [row.score for row in tradable_rows],
        total_exposure=total_exposure,
        threshold=float(weight_threshold),
        max_positions=int(max_positions),
    )
    for row in stock_rows:
        row.suggested_weight = 0.0
    for idx, weight in zip(tradable_indices, tradable_weights):
        stock_rows[idx].suggested_weight = float(weight)
    stock_rows.sort(key=lambda x: x.score, reverse=True)
    _notify(f"量化预测完成: actionable={len(tradable_rows)}, total={len(stock_rows)}")
    return market_forecast, stock_rows
