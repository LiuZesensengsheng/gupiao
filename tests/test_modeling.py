from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.infrastructure.modeling import LogisticBinaryModel, MLPBinaryModel


def _make_binary_training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "f1": [0.1, 0.2, 0.4, 0.3, 0.8, 0.6, 0.7, 0.9],
            "f2": [1.0, 0.9, 0.7, 0.8, 0.3, 0.4, 0.2, 0.1],
            "target": [0, 0, 0, 1, 1, 1, 1, 1],
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
