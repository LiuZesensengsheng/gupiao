from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

from src.domain.entities import BinaryMetrics


def _as_float_array(frame: pd.DataFrame, columns: List[str]) -> np.ndarray:
    return frame[columns].astype(float).to_numpy(copy=True)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> BinaryMetrics:
    if len(y_true) == 0:
        return BinaryMetrics.empty()

    y_true = y_true.astype(float)
    y_prob = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    pred = (y_prob >= 0.5).astype(float)

    accuracy = float((pred == y_true).mean())
    brier = float(np.mean((y_prob - y_true) ** 2))
    base_rate = float(np.mean(y_true))

    pos = int((y_true == 1).sum())
    neg = int((y_true == 0).sum())
    if pos == 0 or neg == 0:
        auc = np.nan
    else:
        ranks = rankdata(y_prob)
        auc = float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))

    return BinaryMetrics(n=len(y_true), accuracy=accuracy, brier=brier, auc=auc, base_rate=base_rate)


class LogisticBinaryModel:
    def __init__(self, l2: float = 1.0, max_iter: int = 400) -> None:
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.feature_names: List[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.fallback_prob_: float | None = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> "LogisticBinaryModel":
        train = df.dropna(subset=feature_cols + [target_col]).copy()
        if train.empty:
            raise ValueError("No rows available for training after dropping NaN.")

        y = train[target_col].astype(float).to_numpy()
        x = _as_float_array(train, feature_cols)

        self.feature_names = list(feature_cols)
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        xs = (x - self.mean_) / self.std_

        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        if positives == 0 or negatives == 0:
            self.fallback_prob_ = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
            self.coef_ = np.zeros(xs.shape[1], dtype=float)
            self.intercept_ = float(np.log(self.fallback_prob_ / (1 - self.fallback_prob_)))
            return self

        p0 = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
        init = np.zeros(xs.shape[1] + 1, dtype=float)
        init[-1] = np.log(p0 / (1 - p0))

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:-1]
            b = params[-1]
            z = xs @ w + b
            p = np.clip(expit(z), 1e-8, 1 - 1e-8)
            loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
            loss = loss + 0.5 * self.l2 * float(np.sum(w * w))

            grad_common = (p - y) / len(y)
            grad_w = xs.T @ grad_common + self.l2 * w
            grad_b = float(np.sum(grad_common))
            grad = np.concatenate([grad_w, [grad_b]])
            return float(loss), grad

        opt = minimize(
            fun=lambda p: objective(p)[0],
            x0=init,
            jac=lambda p: objective(p)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iter},
        )

        if not opt.success:
            self.fallback_prob_ = p0
            self.coef_ = np.zeros(xs.shape[1], dtype=float)
            self.intercept_ = init[-1]
            return self

        self.coef_ = opt.x[:-1].astype(float)
        self.intercept_ = float(opt.x[-1])
        self.fallback_prob_ = None
        return self

    def predict_proba(self, df: pd.DataFrame, feature_cols: List[str] | None = None) -> np.ndarray:
        if feature_cols is None:
            feature_cols = self.feature_names
        if self.mean_ is None or self.std_ is None or self.coef_ is None:
            raise ValueError("Model is not trained.")
        x = _as_float_array(df, feature_cols)
        xs = (x - self.mean_) / self.std_

        if self.fallback_prob_ is not None:
            return np.full(xs.shape[0], self.fallback_prob_, dtype=float)
        z = xs @ self.coef_ + self.intercept_
        return np.clip(expit(z), 1e-6, 1 - 1e-6)

    def top_drivers(self, row: pd.Series, top_n: int = 3) -> List[str]:
        if self.mean_ is None or self.std_ is None or self.coef_ is None:
            return []
        values = row[self.feature_names].astype(float).to_numpy()
        xs = (values - self.mean_) / self.std_
        contrib = xs * self.coef_
        order = np.argsort(np.abs(contrib))[::-1][:top_n]
        drivers: List[str] = []
        for idx in order:
            drivers.append(f"{self.feature_names[idx]}({float(contrib[idx]):+.2f})")
        return drivers
