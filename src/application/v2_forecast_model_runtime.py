from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.infrastructure.modeling import (
    LogisticBinaryModel,
    MLPBinaryModel,
    MLPQuantileModel,
    QuantileLinearModel,
)


@dataclass(frozen=True)
class ReturnQuantileProfile:
    expected_return: float
    q10: float
    q30: float
    q20: float
    q50: float
    q70: float
    q80: float
    q90: float


def fit_quantile_quintet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel]:
    return (
        QuantileLinearModel(quantile=0.10, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.30, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.50, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.70, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.90, l2=l2).fit(df, feature_cols, target_col),
    )


def fit_mlp_quantile_quintet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[MLPQuantileModel, MLPQuantileModel, MLPQuantileModel, MLPQuantileModel, MLPQuantileModel]:
    return (
        MLPQuantileModel(quantile=0.10, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.30, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.50, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.70, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.90, l2=l2).fit(df, feature_cols, target_col),
    )


def predict_quantile_profile(
    row: pd.DataFrame,
    *,
    feature_cols: list[str],
    q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> ReturnQuantileProfile:
    raw = [float(model.predict(row, feature_cols)[0]) for model in q_models]
    q10, q30, q50, q70, q90 = [float(x) for x in np.maximum.accumulate(np.asarray(raw, dtype=float))]
    q20 = float(0.5 * (q10 + q30))
    q80 = float(0.5 * (q70 + q90))
    return ReturnQuantileProfile(
        expected_return=float(0.10 * q10 + 0.20 * q30 + 0.40 * q50 + 0.20 * q70 + 0.10 * q90),
        q10=q10,
        q30=q30,
        q20=q20,
        q50=q50,
        q70=q70,
        q80=q80,
        q90=q90,
    )


def predict_quantile_profiles(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["expected_return", "q10", "q30", "q20", "q50", "q70", "q80", "q90"],
        )
    raw = np.column_stack([model.predict(frame, feature_cols) for model in q_models]).astype(float)
    ordered = np.maximum.accumulate(raw, axis=1)
    q10 = ordered[:, 0]
    q30 = ordered[:, 1]
    q50 = ordered[:, 2]
    q70 = ordered[:, 3]
    q90 = ordered[:, 4]
    q20 = 0.5 * (q10 + q30)
    q80 = 0.5 * (q70 + q90)
    expected = 0.10 * q10 + 0.20 * q30 + 0.40 * q50 + 0.20 * q70 + 0.10 * q90
    return pd.DataFrame(
        {
            "expected_return": expected.astype(float),
            "q10": q10.astype(float),
            "q30": q30.astype(float),
            "q20": q20.astype(float),
            "q50": q50.astype(float),
            "q70": q70.astype(float),
            "q80": q80.astype(float),
            "q90": q90.astype(float),
        },
        index=frame.index,
    )


def serialize_binary_model(model: LogisticBinaryModel | MLPBinaryModel) -> dict[str, object]:
    if isinstance(model, LogisticBinaryModel):
        return {
            "model_type": "logistic_linear",
            "l2": float(model.l2),
            "max_iter": int(model.max_iter),
            "feature_names": list(model.feature_names),
            "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
            "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
            "coef": [] if model.coef_ is None else np.asarray(model.coef_, dtype=float).tolist(),
            "intercept": float(model.intercept_),
            "fallback_prob": None if model.fallback_prob_ is None else float(model.fallback_prob_),
        }
    return {
        "model_type": "mlp_binary",
        "l2": float(model.l2),
        "hidden_dim": int(model.hidden_dim),
        "epochs": int(model.epochs),
        "learning_rate": float(model.learning_rate),
        "random_state": int(model.random_state),
        "feature_names": list(model.feature_names),
        "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
        "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
        "w1": [] if model.w1_ is None else np.asarray(model.w1_, dtype=float).tolist(),
        "b1": [] if model.b1_ is None else np.asarray(model.b1_, dtype=float).tolist(),
        "w2": [] if model.w2_ is None else np.asarray(model.w2_, dtype=float).tolist(),
        "b2": float(model.b2_),
        "fallback_prob": None if model.fallback_prob_ is None else float(model.fallback_prob_),
    }


def deserialize_binary_model(payload: dict[str, object]) -> LogisticBinaryModel | MLPBinaryModel:
    model_type = str(payload.get("model_type", "logistic_linear"))
    if model_type == "mlp_binary":
        model = MLPBinaryModel(
            l2=float(payload.get("l2", 1.0)),
            hidden_dim=int(payload.get("hidden_dim", 24)),
            epochs=int(payload.get("epochs", 120)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            random_state=int(payload.get("random_state", 7)),
        )
        model.feature_names = [str(item) for item in payload.get("feature_names", [])]
        model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
        model.std_ = np.asarray(payload.get("std", []), dtype=float)
        model.w1_ = np.asarray(payload.get("w1", []), dtype=float)
        model.b1_ = np.asarray(payload.get("b1", []), dtype=float)
        model.w2_ = np.asarray(payload.get("w2", []), dtype=float)
        model.b2_ = float(payload.get("b2", 0.0))
        fallback_prob = payload.get("fallback_prob")
        model.fallback_prob_ = None if fallback_prob is None else float(fallback_prob)
        return model

    model = LogisticBinaryModel(
        l2=float(payload.get("l2", 1.0)),
        max_iter=int(payload.get("max_iter", 400)),
    )
    model.feature_names = [str(item) for item in payload.get("feature_names", [])]
    model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
    model.std_ = np.asarray(payload.get("std", []), dtype=float)
    model.coef_ = np.asarray(payload.get("coef", []), dtype=float)
    model.intercept_ = float(payload.get("intercept", 0.0))
    fallback_prob = payload.get("fallback_prob")
    model.fallback_prob_ = None if fallback_prob is None else float(fallback_prob)
    return model


def serialize_quantile_model(model: QuantileLinearModel | MLPQuantileModel) -> dict[str, object]:
    if isinstance(model, QuantileLinearModel):
        return {
            "model_type": "quantile_linear",
            "quantile": float(model.quantile),
            "l2": float(model.l2),
            "max_iter": int(model.max_iter),
            "feature_names": list(model.feature_names),
            "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
            "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
            "coef": [] if model.coef_ is None else np.asarray(model.coef_, dtype=float).tolist(),
            "intercept": float(model.intercept_),
            "fallback_value": None if model.fallback_value_ is None else float(model.fallback_value_),
        }
    return {
        "model_type": "mlp_quantile",
        "quantile": float(model.quantile),
        "l2": float(model.l2),
        "hidden_dim": int(model.hidden_dim),
        "epochs": int(model.epochs),
        "learning_rate": float(model.learning_rate),
        "random_state": int(model.random_state),
        "feature_names": list(model.feature_names),
        "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
        "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
        "w1": [] if model.w1_ is None else np.asarray(model.w1_, dtype=float).tolist(),
        "b1": [] if model.b1_ is None else np.asarray(model.b1_, dtype=float).tolist(),
        "w2": [] if model.w2_ is None else np.asarray(model.w2_, dtype=float).tolist(),
        "b2": float(model.b2_),
        "fallback_value": None if model.fallback_value_ is None else float(model.fallback_value_),
    }


def deserialize_quantile_model(payload: dict[str, object]) -> QuantileLinearModel | MLPQuantileModel:
    model_type = str(payload.get("model_type", "quantile_linear"))
    if model_type == "mlp_quantile":
        model = MLPQuantileModel(
            quantile=float(payload.get("quantile", 0.5)),
            l2=float(payload.get("l2", 1.0)),
            hidden_dim=int(payload.get("hidden_dim", 24)),
            epochs=int(payload.get("epochs", 120)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            random_state=int(payload.get("random_state", 7)),
        )
        model.feature_names = [str(item) for item in payload.get("feature_names", [])]
        model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
        model.std_ = np.asarray(payload.get("std", []), dtype=float)
        model.w1_ = np.asarray(payload.get("w1", []), dtype=float)
        model.b1_ = np.asarray(payload.get("b1", []), dtype=float)
        model.w2_ = np.asarray(payload.get("w2", []), dtype=float)
        model.b2_ = float(payload.get("b2", 0.0))
        fallback_value = payload.get("fallback_value")
        model.fallback_value_ = None if fallback_value is None else float(fallback_value)
        return model

    model = QuantileLinearModel(
        quantile=float(payload.get("quantile", 0.5)),
        l2=float(payload.get("l2", 1.0)),
        max_iter=int(payload.get("max_iter", 300)),
    )
    model.feature_names = [str(item) for item in payload.get("feature_names", [])]
    model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
    model.std_ = np.asarray(payload.get("std", []), dtype=float)
    model.coef_ = np.asarray(payload.get("coef", []), dtype=float)
    model.intercept_ = float(payload.get("intercept", 0.0))
    fallback_value = payload.get("fallback_value")
    model.fallback_value_ = None if fallback_value is None else float(fallback_value)
    return model


def serialize_quantile_bundle(
    q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> list[dict[str, object]]:
    return [serialize_quantile_model(model) for model in q_models]


def deserialize_quantile_bundle(
    payload: object,
) -> tuple[
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
]:
    items = payload if isinstance(payload, list) else []
    models = [deserialize_quantile_model(item) for item in items if isinstance(item, dict)]
    if len(models) != 5:
        raise ValueError("invalid quantile model bundle")
    return tuple(models)  # type: ignore[return-value]
