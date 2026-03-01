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
    make_stock_feature_frame,
    stock_feature_columns,
)
from src.infrastructure.margin_features import build_stock_margin_features
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import DataError, load_symbol_daily
from src.infrastructure.modeling import LogisticBinaryModel, binary_metrics


SHORT_BUCKET_LABELS = ["大跌", "小跌", "震荡", "小涨", "大涨"]
MID_BUCKET_LABELS = ["大跌", "小跌", "震荡", "小涨", "大涨"]
SHORT_BUCKET_REPRESENTATIVES = [-0.04, -0.0125, 0.0, 0.0125, 0.04]
MID_BUCKET_REPRESENTATIVES = [-0.16, -0.06, 0.0, 0.06, 0.16]


@dataclass(frozen=True)
class ReturnBucketProfile:
    bucket_probs: list[float]
    expected_return: float


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


def _normalize_exceedance_probs(raw: Sequence[float]) -> list[float]:
    if not raw:
        return []
    out = [float(np.clip(raw[0], 0.0, 1.0))]
    for value in raw[1:]:
        out.append(float(np.clip(min(out[-1], float(value)), 0.0, 1.0)))
    return out


def estimate_return_bucket_profile(
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
        return ReturnBucketProfile(
            bucket_probs=[float(x) for x in fallback],
            expected_return=float(np.mean(np.asarray(representatives, dtype=float))) if representatives else 0.0,
        )

    latest = frame.dropna(subset=feature_cols).sort_values("date").iloc[[-1]]
    exceedance_probs: list[float] = []
    for threshold in thresholds:
        target_col = "__bucket_target__"
        train = frame.copy()
        train[target_col] = (train[return_col].astype(float) > float(threshold)).astype(float)
        positives = int((train[target_col] == 1).sum())
        negatives = int((train[target_col] == 0).sum())
        if positives == 0 and negatives == 0:
            prob = 0.5
        elif positives == 0:
            prob = 0.0
        elif negatives == 0:
            prob = 1.0
        else:
            model = _fit_latest_model(train, feature_cols=feature_cols, target_col=target_col, l2=l2)
            prob = float(model.predict_proba(latest, feature_cols=feature_cols)[0])
        exceedance_probs.append(prob)

    exceedance_probs = _normalize_exceedance_probs(exceedance_probs)
    bucket_probs: list[float] = []
    prev = 1.0
    for prob in exceedance_probs:
        bucket_probs.append(float(np.clip(prev - prob, 0.0, 1.0)))
        prev = prob
    bucket_probs.append(float(np.clip(prev, 0.0, 1.0)))

    total = float(sum(bucket_probs))
    if total <= 1e-9:
        bucket_probs = [1.0 / max(1, len(representatives))] * max(1, len(representatives))
    else:
        bucket_probs = [float(x) / total for x in bucket_probs]

    reps = np.asarray(list(representatives), dtype=float)
    expected_return = float(np.dot(np.asarray(bucket_probs, dtype=float), reps)) if reps.size else 0.0
    return ReturnBucketProfile(
        bucket_probs=[float(x) for x in bucket_probs],
        expected_return=float(expected_return),
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
    market_forecast.short_bucket_probs = list(market_short_bucket.bucket_probs)
    market_forecast.mid_bucket_probs = list(market_mid_bucket.bucket_probs)

    stock_rows: list[ForecastRow] = []
    for security in stock_securities:
        try:
            stock_raw = load_symbol_daily(
                symbol=security.symbol,
                source=source,
                data_dir=data_dir,
                start=start,
                end=end,
            )
            stock_feat = make_stock_feature_frame(stock_raw, market_feat)
            stock_margin_cols: list[str] = []
            if use_margin_features:
                margin_frame, margin_cols, _ = build_stock_margin_features(
                    margin_stock_file=margin_stock_file,
                    symbol=security.symbol,
                    start=start,
                    end=end,
                )
                if margin_cols:
                    stock_feat = stock_feat.merge(margin_frame, on="date", how="left", validate="1:1")
                    stock_margin_cols = list(margin_cols)

            feature_cols = stock_feature_columns(
                extra_market_cols=market_context.feature_columns,
                extra_stock_cols=stock_margin_cols,
            )

            short_model = _fit_latest_model(stock_feat, feature_cols=feature_cols, target_col="target_1d_up", l2=l2)
            mid_model = _fit_latest_model(stock_feat, feature_cols=feature_cols, target_col="target_20d_up", l2=l2)
            latest_row = _latest_row_with_features(stock_feat, feature_cols)
            latest_df = pd.DataFrame([latest_row])

            short_prob = float(short_model.predict_proba(latest_df, feature_cols=feature_cols)[0])
            mid_prob = float(mid_model.predict_proba(latest_df, feature_cols=feature_cols)[0])
            short_bucket = estimate_return_bucket_profile(
                stock_feat,
                feature_cols=feature_cols,
                return_col="fwd_ret_1",
                thresholds=SHORT_RETURN_BUCKET_EDGES[1:-1],
                representatives=SHORT_BUCKET_REPRESENTATIVES,
                l2=l2,
            )
            mid_bucket = estimate_return_bucket_profile(
                stock_feat,
                feature_cols=feature_cols,
                return_col="fwd_ret_20",
                thresholds=MID_RETURN_BUCKET_EDGES[1:-1],
                representatives=MID_BUCKET_REPRESENTATIVES,
                l2=l2,
            )
            score = _distributional_score(
                short_prob=short_prob,
                mid_prob=mid_prob,
                short_expected_ret=float(short_bucket.expected_return),
                mid_expected_ret=float(mid_bucket.expected_return),
            )

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
                    short_eval=(
                        _walk_forward_eval(
                            stock_feat,
                            feature_cols=feature_cols,
                            target_col="target_1d_up",
                            l2=l2,
                            min_train_days=min_train_days,
                            step_days=step_days,
                        )
                        if enable_walk_forward_eval
                        else BinaryMetrics.empty()
                    ),
                    mid_eval=(
                        _walk_forward_eval(
                            stock_feat,
                            feature_cols=feature_cols,
                            target_col="target_20d_up",
                            l2=l2,
                            min_train_days=min_train_days,
                            step_days=step_days,
                        )
                        if enable_walk_forward_eval
                        else BinaryMetrics.empty()
                    ),
                    short_expected_ret=float(short_bucket.expected_return),
                    mid_expected_ret=float(mid_bucket.expected_return),
                    short_bucket_probs=list(short_bucket.bucket_probs),
                    mid_bucket_probs=list(mid_bucket.bucket_probs),
                )
            )
        except Exception:
            # Skip symbols with broken/insufficient data so large-universe scans can continue.
            continue

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
