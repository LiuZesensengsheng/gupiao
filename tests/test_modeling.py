from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.modeling import (
    LogisticBinaryModel,
    MLPBinaryModel,
    MLPQuantileModel,
    QuantileLinearModel,
)


def _make_binary_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7, 0.9],
            "f2": [1.0, 0.9, 0.7, 0.8, 0.3, 0.4, 0.2, 0.1],
            "target": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


def _make_quantile_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7, 0.9, 1.0, 1.1],
            "f2": [1.0, 0.9, 0.7, 0.8, 0.3, 0.4, 0.2, 0.1, 0.05, 0.02],
            "target": [-0.02, -0.01, 0.0, 0.01, 0.03, 0.025, 0.04, 0.05, 0.055, 0.06],
        }
    )


def test_logistic_binary_fit_prepared_matches_fit() -> None:
    frame = _make_binary_training_frame()
    feature_cols = ["f1", "f2"]
    fitted = LogisticBinaryModel(l2=0.2).fit(frame, feature_cols, "target")

    prepared = frame.dropna(subset=feature_cols + ["target"]).copy()
    prepared_model = LogisticBinaryModel(l2=0.2).fit_prepared(
        x=prepared[feature_cols].astype(float).to_numpy(copy=True),
        y=prepared["target"].astype(float).to_numpy(copy=True),
        feature_cols=feature_cols,
    )

    expected = fitted.predict_proba(frame, feature_cols)
    actual = prepared_model.predict_proba(frame, feature_cols)

    assert actual == pytest.approx(expected)


def test_mlp_binary_fit_prepared_matches_fit() -> None:
    frame = _make_binary_training_frame()
    feature_cols = ["f1", "f2"]
    fitted = MLPBinaryModel(l2=0.2, epochs=40, random_state=11).fit(frame, feature_cols, "target")

    prepared = frame.dropna(subset=feature_cols + ["target"]).copy()
    prepared_model = MLPBinaryModel(l2=0.2, epochs=40, random_state=11).fit_prepared(
        x=prepared[feature_cols].astype(float).to_numpy(copy=True),
        y=prepared["target"].astype(float).to_numpy(copy=True),
        feature_cols=feature_cols,
    )

    expected = fitted.predict_proba(frame, feature_cols)
    actual = prepared_model.predict_proba(frame, feature_cols)

    assert np.asarray(actual) == pytest.approx(np.asarray(expected))


def test_quantile_linear_fit_prepared_matches_fit() -> None:
    frame = _make_quantile_training_frame()
    feature_cols = ["f1", "f2"]
    fitted = QuantileLinearModel(quantile=0.7, l2=0.2).fit(frame, feature_cols, "target")

    prepared = frame.dropna(subset=feature_cols + ["target"]).copy()
    prepared_model = QuantileLinearModel(quantile=0.7, l2=0.2).fit_prepared(
        x=prepared[feature_cols].astype(float).to_numpy(copy=True),
        y=prepared["target"].astype(float).to_numpy(copy=True),
        feature_cols=feature_cols,
    )

    expected = fitted.predict(frame, feature_cols)
    actual = prepared_model.predict(frame, feature_cols)

    assert np.asarray(actual) == pytest.approx(np.asarray(expected))


def test_mlp_quantile_fit_prepared_matches_fit() -> None:
    frame = _make_quantile_training_frame()
    feature_cols = ["f1", "f2"]
    fitted = MLPQuantileModel(quantile=0.7, l2=0.2, epochs=40, random_state=11).fit(frame, feature_cols, "target")

    prepared = frame.dropna(subset=feature_cols + ["target"]).copy()
    prepared_model = MLPQuantileModel(quantile=0.7, l2=0.2, epochs=40, random_state=11).fit_prepared(
        x=prepared[feature_cols].astype(float).to_numpy(copy=True),
        y=prepared["target"].astype(float).to_numpy(copy=True),
        feature_cols=feature_cols,
    )

    expected = fitted.predict(frame, feature_cols)
    actual = prepared_model.predict(frame, feature_cols)

    assert np.asarray(actual) == pytest.approx(np.asarray(expected))


def test_mlp_binary_device_numpy_forces_numpy_backend() -> None:
    frame = _make_binary_training_frame()
    model = MLPBinaryModel(l2=0.2, epochs=20, random_state=11, device="numpy").fit(frame, ["f1", "f2"], "target")

    assert model.training_backend_ == "numpy"
    assert model.training_device_ == "cpu"


def test_mlp_binary_uses_torch_backend_when_available() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    frame = _make_binary_training_frame()
    model = MLPBinaryModel(l2=0.2, epochs=20, random_state=11, device="auto").fit(frame, ["f1", "f2"], "target")

    assert model.training_backend_ == "torch"
    assert model.training_device_ in {"cpu", "cuda"}
