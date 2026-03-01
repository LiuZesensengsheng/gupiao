from __future__ import annotations

import numpy as np
import pandas as pd

from src.infrastructure.features import BASE_FEATURE_COLUMNS, build_features
from src.infrastructure.forecast_engine import (
    MID_BUCKET_REPRESENTATIVES,
    MID_RETURN_BUCKET_EDGES,
    SHORT_BUCKET_REPRESENTATIVES,
    SHORT_RETURN_BUCKET_EDGES,
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
    assert "target_20d_bucket" in feat.columns
    valid_1d = feat["target_1d_bucket"].dropna()
    valid_20d = feat["target_20d_bucket"].dropna()
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
    assert all(prob >= 0.0 for prob in profile_1d.bucket_probs)
    assert all(prob >= 0.0 for prob in profile_20d.bucket_probs)
    assert -0.05 <= profile_1d.expected_return <= 0.05
    assert -0.20 <= profile_20d.expected_return <= 0.20
