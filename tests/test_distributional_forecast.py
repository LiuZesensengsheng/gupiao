from __future__ import annotations

import numpy as np
import pandas as pd

from src.infrastructure.features import BASE_FEATURE_COLUMNS, build_features
from src.infrastructure.forecast_engine import (
    MID_BUCKET_REPRESENTATIVES,
    MID_RETURN_BUCKET_EDGES,
    SHORT_BUCKET_REPRESENTATIVES,
    SHORT_RETURN_BUCKET_EDGES,
    _calibrate_latest_probabilities,
    _distributional_score,
    estimate_return_bucket_profile,
)


def _make_price_frame(n: int = 220) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    base = 100.0
    close = []
    for idx in range(n):
        drift = 0.0025 if idx % 11 < 6 else -0.0015
        swing = 0.003 * np.sin(idx / 7.0)
        base *= 1.0 + drift + swing
        close.append(base)
    close_arr = np.asarray(close, dtype=float)
    open_arr = close_arr * (1.0 + 0.001 * np.cos(np.arange(n) / 5.0))
    high_arr = np.maximum(open_arr, close_arr) * 1.01
    low_arr = np.minimum(open_arr, close_arr) * 0.99
    volume_arr = 1_000_000 + 50_000 * np.sin(np.arange(n) / 9.0)
    amount_arr = close_arr * volume_arr
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_arr,
            "high": high_arr,
            "low": low_arr,
            "close": close_arr,
            "volume": volume_arr,
            "amount": amount_arr,
        }
    )


def test_build_features_adds_bucket_targets() -> None:
    feat = build_features(_make_price_frame())

    assert "target_1d_bucket" in feat.columns
    assert "target_2d_up" in feat.columns
    assert "target_3d_up" in feat.columns
    assert "target_5d_up" in feat.columns
    assert "target_20d_bucket" in feat.columns
    valid_2d = feat["target_2d_up"].dropna()
    valid_3d = feat["target_3d_up"].dropna()
    valid_5d = feat["target_5d_up"].dropna()
    valid_1d = feat["target_1d_bucket"].dropna()
    valid_20d = feat["target_20d_bucket"].dropna()
    assert valid_2d.between(0, 1).all()
    assert valid_3d.between(0, 1).all()
    assert valid_5d.between(0, 1).all()
    assert valid_1d.between(0, 4).all()
    assert valid_20d.between(0, 4).all()


def test_estimate_return_bucket_profile_produces_valid_distribution() -> None:
    feat = build_features(_make_price_frame())
    profile_1d = estimate_return_bucket_profile(
        feat,
        feature_cols=list(BASE_FEATURE_COLUMNS),
        return_col="fwd_ret_1",
        thresholds=SHORT_RETURN_BUCKET_EDGES[1:-1],
        representatives=SHORT_BUCKET_REPRESENTATIVES,
        l2=0.8,
    )
    profile_20d = estimate_return_bucket_profile(
        feat,
        feature_cols=list(BASE_FEATURE_COLUMNS),
        return_col="fwd_ret_20",
        thresholds=MID_RETURN_BUCKET_EDGES[1:-1],
        representatives=MID_BUCKET_REPRESENTATIVES,
        l2=0.8,
    )

    assert len(profile_1d.bucket_probs) == 5
    assert len(profile_20d.bucket_probs) == 5
    assert abs(sum(profile_1d.bucket_probs) - 1.0) < 1e-6
    assert abs(sum(profile_20d.bucket_probs) - 1.0) < 1e-6
    assert profile_1d.q10 <= profile_1d.q30 <= profile_1d.q50 <= profile_1d.q70 <= profile_1d.q90
    assert profile_20d.q10 <= profile_20d.q30 <= profile_20d.q50 <= profile_20d.q70 <= profile_20d.q90
    assert profile_1d.q20 <= profile_1d.q50 <= profile_1d.q80
    assert profile_20d.q20 <= profile_20d.q50 <= profile_20d.q80
    assert all(prob >= 0.0 for prob in profile_1d.bucket_probs)
    assert all(prob >= 0.0 for prob in profile_20d.bucket_probs)
    assert -0.05 <= profile_1d.expected_return <= 0.05
    assert -0.20 <= profile_20d.expected_return <= 0.20


def test_distributional_score_improves_when_five_day_signal_improves() -> None:
    weak_mid = _distributional_score(
        short_prob=0.52,
        two_prob=0.53,
        three_prob=0.54,
        five_prob=0.48,
        mid_prob=0.56,
        short_expected_ret=0.002,
        mid_expected_ret=0.018,
    )
    strong_mid = _distributional_score(
        short_prob=0.52,
        two_prob=0.53,
        three_prob=0.54,
        five_prob=0.66,
        mid_prob=0.56,
        short_expected_ret=0.002,
        mid_expected_ret=0.018,
    )

    assert strong_mid > weak_mid


def test_build_features_adds_pattern_and_breakout_features() -> None:
    dates = pd.date_range("2024-01-01", periods=35, freq="B")
    close = np.concatenate(
        [
            np.linspace(10.0, 11.0, 30),
            np.array([11.1, 11.15, 11.2, 11.25, 12.2]),
        ]
    )
    open_ = close.copy()
    open_[-1] = 11.6
    high = close + 0.15
    low = close - 0.12
    high[-1] = 12.4
    low[-1] = 11.55
    volume = np.full(len(close), 800_000.0, dtype=float)
    volume[-1] = 2_400_000.0
    amount = close * volume
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "amount": amount,
        }
    )

    feat = build_features(frame)
    latest = feat.iloc[-1]

    for col in [
        "upper_shadow_ratio_1",
        "lower_shadow_ratio_1",
        "body_ratio_1",
        "up_streak_3",
        "down_streak_3",
        "narrow_range_rank_20",
        "range_contraction_5",
        "breakout_above_20_high",
        "breakdown_below_20_low",
        "distance_to_20d_high",
        "distance_to_20d_low",
        "volume_breakout_ratio",
        "squeeze_breakout_score",
        "breakout_quality_score",
        "exhaustion_reversal_risk",
        "pullback_reclaim_score",
    ]:
        assert col in feat.columns

    assert 0.0 <= float(latest["upper_shadow_ratio_1"]) <= 1.0
    assert 0.0 <= float(latest["lower_shadow_ratio_1"]) <= 1.0
    assert 0.0 <= float(latest["body_ratio_1"]) <= 1.0
    assert 0.0 <= float(latest["up_streak_3"]) <= 1.0
    assert 0.0 <= float(latest["down_streak_3"]) <= 1.0
    assert 0.0 <= float(latest["narrow_range_rank_20"]) <= 1.0
    assert float(latest["range_contraction_5"]) > 0.0
    assert float(latest["breakout_above_20_high"]) == 1.0
    assert float(latest["breakdown_below_20_low"]) == 0.0
    assert float(latest["distance_to_20d_high"]) > 0.0
    assert float(latest["distance_to_20d_low"]) > 0.0
    assert float(latest["volume_breakout_ratio"]) > 1.0
    assert float(latest["squeeze_breakout_score"]) >= 0.0
    assert float(latest["breakout_quality_score"]) > 0.0
    assert float(latest["exhaustion_reversal_risk"]) >= 0.0
    assert float(latest["pullback_reclaim_score"]) >= 0.0


def test_calibrate_latest_probabilities_improves_brier_on_shifted_probs() -> None:
    dates = pd.date_range("2024-01-01", periods=180, freq="B")
    y_true = np.asarray(([0.0] * 90) + ([1.0] * 90), dtype=float)
    raw_prob = np.asarray(([0.35] * 90) + ([0.65] * 90), dtype=float)
    pred_frame = pd.DataFrame(
        {
            "date": dates,
            "y_true": y_true,
            "raw_prob": raw_prob,
        }
    )

    result = _calibrate_latest_probabilities(
        pred_frame=pred_frame,
        latest_prob=np.asarray([0.35, 0.65], dtype=float),
    )

    assert result.method in {"platt", "isotonic"}
    assert result.metrics.calibrated_brier < result.metrics.brier
    assert np.all(result.latest_probs > 0.0)
    assert np.all(result.latest_probs < 1.0)
