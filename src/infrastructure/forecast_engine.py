from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ReturnBucketProfile:
    bucket_probs: list[float]
    expected_return: float
    q20: float
    q50: float
    q80: float


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
    q20: float,
    q50: float,
    q80: float,
) -> list[float]:
    step_lo = max(1e-6, abs(float(q50) - float(q20)))
    step_hi = max(1e-6, abs(float(q80) - float(q50)))
    x_points = np.asarray(
        [
            float(q20 - 2.0 * step_lo),
            float(q20),
            float(q50),
            float(q80),
            float(q80 + 2.0 * step_hi),
        ],
        dtype=float,
    )
    y_points = np.asarray([0.0, 0.2, 0.5, 0.8, 1.0], dtype=float)
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
            q20=mid,
            q50=mid,
            q80=mid,
        )

    latest = frame.dropna(subset=feature_cols).sort_values("date").iloc[[-1]]
    q20_model = QuantileLinearModel(quantile=0.20, l2=l2).fit(frame, feature_cols, return_col)
    q50_model = QuantileLinearModel(quantile=0.50, l2=l2).fit(frame, feature_cols, return_col)
    q80_model = QuantileLinearModel(quantile=0.80, l2=l2).fit(frame, feature_cols, return_col)

    q20 = float(q20_model.predict(latest, feature_cols)[0])
    q50 = float(q50_model.predict(latest, feature_cols)[0])
    q80 = float(q80_model.predict(latest, feature_cols)[0])
    ordered = sorted([q20, q50, q80])
    q20, q50, q80 = float(ordered[0]), float(ordered[1]), float(ordered[2])

    bucket_probs = _bucket_probs_from_quantiles(
        thresholds=thresholds,
        q20=q20,
        q50=q50,
        q80=q80,
    )
    expected_return = float(0.25 * q20 + 0.5 * q50 + 0.25 * q80)
    return ReturnBucketProfile(
        bucket_probs=[float(x) for x in bucket_probs],
        expected_return=float(expected_return),
        q20=float(q20),
        q50=float(q50),
        q80=float(q80),
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
    mid_prob: float,
    short_expected_ret: float,
    mid_expected_ret: float,
) -> float:
    base_score = blend_horizon_score(float(short_prob), float(mid_prob), short_weight=0.55)
    short_ret_score = float(np.clip(0.5 + float(short_expected_ret) / 0.06, 0.0, 1.0))
    mid_ret_score = float(np.clip(0.5 + float(mid_expected_ret) / 0.20, 0.0, 1.0))
    dist_score = blend_horizon_score(short_ret_score, mid_ret_score, short_weight=0.35)
    return float(0.4 * base_score + 0.6 * dist_score)


def _fit_quantile_triplet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[QuantileLinearModel, QuantileLinearModel, QuantileLinearModel]:
    return (
        QuantileLinearModel(quantile=0.20, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.50, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.80, l2=l2).fit(df, feature_cols, target_col),
    )


def _predict_quantile_profile_from_models(
    latest_row: pd.DataFrame,
    *,
    feature_cols: list[str],
    thresholds: Sequence[float],
    q_models: tuple[QuantileLinearModel, QuantileLinearModel, QuantileLinearModel],
) -> ReturnBucketProfile:
    q20 = float(q_models[0].predict(latest_row, feature_cols)[0])
    q50 = float(q_models[1].predict(latest_row, feature_cols)[0])
    q80 = float(q_models[2].predict(latest_row, feature_cols)[0])
    ordered = sorted([q20, q50, q80])
    q20, q50, q80 = float(ordered[0]), float(ordered[1]), float(ordered[2])
    bucket_probs = _bucket_probs_from_quantiles(
        thresholds=thresholds,
        q20=q20,
        q50=q50,
        q80=q80,
    )
    expected_return = float(0.25 * q20 + 0.5 * q50 + 0.25 * q80)
    return ReturnBucketProfile(
        bucket_probs=bucket_probs,
        expected_return=expected_return,
        q20=q20,
        q50=q50,
        q80=q80,
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
) -> tuple[MarketForecast, list[ForecastRow]]:
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
    market_forecast.short_q20 = float(market_short_bucket.q20)
    market_forecast.short_q50 = float(market_short_bucket.q50)
    market_forecast.short_q80 = float(market_short_bucket.q80)
    market_forecast.mid_q20 = float(market_mid_bucket.q20)
    market_forecast.mid_q50 = float(market_mid_bucket.q50)
    market_forecast.mid_q80 = float(market_mid_bucket.q80)
    market_forecast.short_bucket_probs = list(market_short_bucket.bucket_probs)
    market_forecast.mid_bucket_probs = list(market_mid_bucket.bucket_probs)

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

    panel_short_model = _fit_latest_model(panel, feature_cols=feature_cols, target_col="target_1d_up", l2=l2)
    panel_mid_model = _fit_latest_model(panel, feature_cols=feature_cols, target_col="target_20d_up", l2=l2)
    panel_short_q_models = _fit_quantile_triplet(
        panel,
        feature_cols=feature_cols,
        target_col="fwd_ret_1",
        l2=l2,
    )
    panel_mid_q_models = _fit_quantile_triplet(
        panel,
        feature_cols=feature_cols,
        target_col="fwd_ret_20",
        l2=l2,
    )

    panel_short_eval = (
        _walk_forward_eval(
            panel,
            feature_cols=feature_cols,
            target_col="target_1d_up",
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
            target_col="target_20d_up",
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

    short_probs = panel_short_model.predict_proba(latest_panel, feature_cols=feature_cols)
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
        mid_prob = float(mid_probs[idx])
        score = _distributional_score(
            short_prob=short_prob,
            mid_prob=mid_prob,
            short_expected_ret=float(short_bucket.expected_return),
            mid_expected_ret=float(mid_bucket.expected_return),
        )
        symbol = normalize_symbol(str(latest_row["symbol"])).symbol
        stock_rows.append(
            ForecastRow(
                symbol=symbol,
                name=str(latest_row.get("name", symbol)),
                latest_date=pd.Timestamp(latest_row["date"]),
                short_prob=short_prob,
                mid_prob=mid_prob,
                score=score,
                short_drivers=panel_short_model.top_drivers(latest_row, top_n=3),
                mid_drivers=panel_mid_model.top_drivers(latest_row, top_n=3),
                short_eval=panel_short_eval,
                mid_eval=panel_mid_eval,
                short_expected_ret=float(short_bucket.expected_return),
                mid_expected_ret=float(mid_bucket.expected_return),
                short_q20=float(short_bucket.q20),
                short_q50=float(short_bucket.q50),
                short_q80=float(short_bucket.q80),
                mid_q20=float(mid_bucket.q20),
                mid_q50=float(mid_bucket.q50),
                mid_q80=float(mid_bucket.q80),
                short_bucket_probs=list(short_bucket.bucket_probs),
                mid_bucket_probs=list(mid_bucket.bucket_probs),
            )
        )

    if not stock_rows:
        raise DataError("No valid stock rows with complete features for selected universe.")

    total_exposure = target_exposure(market_forecast.short_prob, market_forecast.mid_prob)
    weights = allocate_weights(
        [row.score for row in stock_rows],
        total_exposure=total_exposure,
        threshold=float(weight_threshold),
        max_positions=int(max_positions),
    )
    for row, weight in zip(stock_rows, weights):
        row.suggested_weight = float(weight)
    stock_rows.sort(key=lambda x: x.score, reverse=True)
    return market_forecast, stock_rows
