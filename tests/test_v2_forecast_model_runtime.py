from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.application.v2_forecast_model_runtime import (
    fit_mlp_quantile_quintet,
    fit_quantile_quintet,
    predict_quantile_profiles,
)
from src.infrastructure.modeling import MLPQuantileModel, QuantileLinearModel


def _make_quantile_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7, 0.9, 1.0, 1.1],
            "f2": [1.0, 0.9, 0.7, 0.8, 0.3, 0.4, 0.2, 0.1, 0.05, 0.02],
            "target": [-0.02, -0.01, 0.0, 0.01, 0.03, 0.025, 0.04, 0.05, 0.055, 0.06],
        }
    )


def test_fit_quantile_quintet_matches_manual_models() -> None:
    frame = _make_quantile_frame()
    feature_cols = ["f1", "f2"]

    runtime_models = fit_quantile_quintet(
        frame,
        feature_cols=feature_cols,
        target_col="target",
        l2=0.2,
    )
    manual_models = tuple(
        QuantileLinearModel(quantile=q, l2=0.2).fit(frame, feature_cols, "target")
        for q in (0.10, 0.30, 0.50, 0.70, 0.90)
    )

    runtime_profiles = predict_quantile_profiles(frame, feature_cols=feature_cols, q_models=runtime_models)
    manual_profiles = predict_quantile_profiles(frame, feature_cols=feature_cols, q_models=manual_models)

    assert runtime_profiles.to_numpy(dtype=float) == pytest.approx(manual_profiles.to_numpy(dtype=float))


def test_fit_mlp_quantile_quintet_matches_manual_models() -> None:
    frame = _make_quantile_frame()
    feature_cols = ["f1", "f2"]

    runtime_models = fit_mlp_quantile_quintet(
        frame,
        feature_cols=feature_cols,
        target_col="target",
        l2=0.2,
    )
    manual_models = tuple(
        MLPQuantileModel(quantile=q, l2=0.2).fit(frame, feature_cols, "target")
        for q in (0.10, 0.30, 0.50, 0.70, 0.90)
    )

    runtime_profiles = predict_quantile_profiles(frame, feature_cols=feature_cols, q_models=runtime_models)
    manual_profiles = predict_quantile_profiles(frame, feature_cols=feature_cols, q_models=manual_models)

    assert np.asarray(runtime_profiles.to_numpy(dtype=float)) == pytest.approx(
        np.asarray(manual_profiles.to_numpy(dtype=float))
    )
