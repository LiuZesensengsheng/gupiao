from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from src.domain.entities import BinaryMetrics


def _as_float_array(frame: pd.DataFrame, columns: List[str]) -> np.ndarray:
    return frame[columns].astype(float).to_numpy(copy=True)


def _torch_is_available() -> bool:
    return torch is not None


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
        return self.fit_prepared(x=x, y=y, feature_cols=feature_cols)

    def fit_prepared(self, *, x: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> "LogisticBinaryModel":
        if len(y) == 0 or np.asarray(x).size == 0:
            raise ValueError("No rows available for training after dropping NaN.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

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


class QuantileLinearModel:
    def __init__(self, quantile: float, l2: float = 1.0, max_iter: int = 300) -> None:
        self.quantile = float(np.clip(float(quantile), 1e-3, 1.0 - 1e-3))
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.feature_names: List[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.fallback_value_: float | None = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> "QuantileLinearModel":
        train = df.dropna(subset=feature_cols + [target_col]).copy()
        if train.empty:
            raise ValueError("No rows available for quantile training after dropping NaN.")

        y = train[target_col].astype(float).to_numpy()
        x = _as_float_array(train, feature_cols)
        return self.fit_prepared(x=x, y=y, feature_cols=feature_cols)

    def fit_prepared(self, *, x: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> "QuantileLinearModel":
        if len(y) == 0 or np.asarray(x).size == 0:
            raise ValueError("No rows available for quantile training after dropping NaN.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.feature_names = list(feature_cols)
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        xs = (x - self.mean_) / self.std_

        q_value = float(np.quantile(y, self.quantile))
        if len(y) < 25 or float(np.nanstd(y)) < 1e-10:
            self.fallback_value_ = q_value
            self.coef_ = np.zeros(xs.shape[1], dtype=float)
            self.intercept_ = q_value
            return self

        init = np.zeros(xs.shape[1] + 1, dtype=float)
        init[-1] = q_value

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            w = params[:-1]
            b = params[-1]
            pred = xs @ w + b
            residual = y - pred
            loss = np.where(
                residual >= 0.0,
                self.quantile * residual,
                (self.quantile - 1.0) * residual,
            ).mean()
            loss = float(loss) + 0.5 * self.l2 * float(np.sum(w * w))

            grad_pred = np.where(
                residual > 0.0,
                -self.quantile,
                1.0 - self.quantile,
            )
            grad_w = xs.T @ grad_pred / len(y) + self.l2 * w
            grad_b = float(np.mean(grad_pred))
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
            self.fallback_value_ = q_value
            self.coef_ = np.zeros(xs.shape[1], dtype=float)
            self.intercept_ = q_value
            return self

        self.coef_ = opt.x[:-1].astype(float)
        self.intercept_ = float(opt.x[-1])
        self.fallback_value_ = None
        return self

    def predict(self, df: pd.DataFrame, feature_cols: List[str] | None = None) -> np.ndarray:
        if feature_cols is None:
            feature_cols = self.feature_names
        if self.mean_ is None or self.std_ is None or self.coef_ is None:
            raise ValueError("Quantile model is not trained.")
        x = _as_float_array(df, feature_cols)
        xs = (x - self.mean_) / self.std_
        if self.fallback_value_ is not None:
            return np.full(xs.shape[0], float(self.fallback_value_), dtype=float)
        return (xs @ self.coef_ + self.intercept_).astype(float)


class _BaseMLPModel:
    def __init__(
        self,
        *,
        l2: float = 1.0,
        hidden_dim: int = 24,
        epochs: int = 120,
        learning_rate: float = 0.03,
        random_state: int = 7,
        device: str = "auto",
        use_torch: bool | None = None,
    ) -> None:
        self.l2 = float(l2)
        self.hidden_dim = int(max(4, hidden_dim))
        self.epochs = int(max(20, epochs))
        self.learning_rate = float(max(1e-4, learning_rate))
        self.random_state = int(random_state)
        self.device = str(device or "auto").strip().lower() or "auto"
        self.use_torch = use_torch
        self.feature_names: List[str] = []
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.w1_: np.ndarray | None = None
        self.b1_: np.ndarray | None = None
        self.w2_: np.ndarray | None = None
        self.b2_: float = 0.0
        self.training_backend_: str = "numpy"
        self.training_device_: str = "cpu"

    def _prepare_x(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        x = _as_float_array(df, feature_cols)
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Model is not trained.")
        return (x - self.mean_) / self.std_

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        scale1 = 1.0 / np.sqrt(max(1, n_features))
        scale2 = 1.0 / np.sqrt(max(1, self.hidden_dim))
        self.w1_ = rng.normal(0.0, scale1, size=(n_features, self.hidden_dim)).astype(float)
        self.b1_ = np.zeros(self.hidden_dim, dtype=float)
        self.w2_ = rng.normal(0.0, scale2, size=self.hidden_dim).astype(float)
        self.b2_ = 0.0

    def _forward(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.w1_ is None or self.b1_ is None or self.w2_ is None:
            raise ValueError("Model is not initialized.")
        hidden_linear = xs @ self.w1_ + self.b1_
        hidden = np.tanh(hidden_linear)
        output = hidden @ self.w2_ + float(self.b2_)
        return hidden, output

    def _should_use_torch(self) -> bool:
        if self.device == "numpy":
            return False
        if self.use_torch is False:
            return False
        return _torch_is_available()

    def _resolve_torch_device(self) -> str | None:
        if not self._should_use_torch():
            return None
        if torch is None:
            return None
        if self.device in {"auto", "cuda", "gpu"}:
            if torch.cuda.is_available():
                return "cuda"
            if self.device in {"cuda", "gpu"}:
                raise RuntimeError("Torch CUDA requested but no CUDA device is available.")
            return "cpu"
        if self.device == "cpu":
            return "cpu"
        raise ValueError(f"Unsupported MLP device: {self.device}")

    def _torch_parameters(self, n_features: int, *, device: str) -> tuple[object, object, object, object]:
        if torch is None:
            raise RuntimeError("Torch is not installed.")
        rng = np.random.default_rng(self.random_state)
        scale1 = 1.0 / np.sqrt(max(1, n_features))
        scale2 = 1.0 / np.sqrt(max(1, self.hidden_dim))
        w1 = torch.tensor(
            rng.normal(0.0, scale1, size=(n_features, self.hidden_dim)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        b1 = torch.zeros(self.hidden_dim, dtype=torch.float32, device=device, requires_grad=True)
        w2 = torch.tensor(
            rng.normal(0.0, scale2, size=self.hidden_dim),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        b2 = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
        return w1, b1, w2, b2

    def _torch_forward(self, xs: object, *, w1: object, b1: object, w2: object, b2: object) -> tuple[object, object]:
        if torch is None:
            raise RuntimeError("Torch is not installed.")
        hidden_linear = xs @ w1 + b1
        hidden = torch.tanh(hidden_linear)
        output = hidden @ w2 + b2
        return hidden, output

    def _store_torch_parameters(self, *, w1: object, b1: object, w2: object, b2: object, device: str) -> None:
        if torch is None:
            raise RuntimeError("Torch is not installed.")
        self.w1_ = w1.detach().cpu().numpy().astype(float)
        self.b1_ = b1.detach().cpu().numpy().astype(float)
        self.w2_ = w2.detach().cpu().numpy().astype(float)
        self.b2_ = float(b2.detach().cpu().item())
        self.training_backend_ = "torch"
        self.training_device_ = str(device)

    def _fit_with_torch(self, *, x: np.ndarray, y: np.ndarray, trainer: str) -> bool:
        device = self._resolve_torch_device()
        if device is None:
            return False
        if torch is None:
            return False
        try:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
            xs = torch.tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=device)
            ys = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32, device=device)
            w1, b1, w2, b2 = self._torch_parameters(xs.shape[1], device=device)
            optimizer = torch.optim.Adam((w1, b1, w2, b2), lr=self.learning_rate)
            for _ in range(self.epochs):
                optimizer.zero_grad()
                hidden, output = self._torch_forward(xs, w1=w1, b1=b1, w2=w2, b2=b2)
                if trainer == "binary":
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(output, ys)
                elif trainer == "quantile":
                    residual = ys - output
                    loss = torch.maximum(self.quantile * residual, (self.quantile - 1.0) * residual).mean()
                else:
                    raise ValueError(f"Unsupported torch trainer: {trainer}")
                loss = loss + 0.5 * self.l2 * (torch.sum(w1 * w1) + torch.sum(w2 * w2))
                loss.backward()
                optimizer.step()
            self._store_torch_parameters(w1=w1, b1=b1, w2=w2, b2=b2, device=device)
            return True
        except Exception:
            if self.device in {"cuda", "gpu", "cpu"}:
                raise
            self.training_backend_ = "numpy"
            self.training_device_ = "cpu"
            return False


class MLPBinaryModel(_BaseMLPModel):
    def __init__(
        self,
        l2: float = 1.0,
        hidden_dim: int = 24,
        epochs: int = 120,
        learning_rate: float = 0.03,
        random_state: int = 7,
        device: str = "auto",
        use_torch: bool | None = None,
    ) -> None:
        super().__init__(
            l2=l2,
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            random_state=random_state,
            device=device,
            use_torch=use_torch,
        )
        self.fallback_prob_: float | None = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> "MLPBinaryModel":
        train = df.dropna(subset=feature_cols + [target_col]).copy()
        if train.empty:
            raise ValueError("No rows available for MLP binary training after dropping NaN.")

        y = train[target_col].astype(float).to_numpy()
        x = _as_float_array(train, feature_cols)
        return self.fit_prepared(x=x, y=y, feature_cols=feature_cols)

    def fit_prepared(self, *, x: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> "MLPBinaryModel":
        if len(y) == 0 or np.asarray(x).size == 0:
            raise ValueError("No rows available for MLP binary training after dropping NaN.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.feature_names = list(feature_cols)
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        xs = (x - self.mean_) / self.std_

        positives = int((y == 1).sum())
        negatives = int((y == 0).sum())
        if positives == 0 or negatives == 0:
            self.fallback_prob_ = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
            self._init_params(xs.shape[1])
            self.training_backend_ = "numpy"
            self.training_device_ = "cpu"
            return self

        if self._fit_with_torch(x=xs, y=y, trainer="binary"):
            self.fallback_prob_ = None
            return self

        self._init_params(xs.shape[1])
        for _ in range(self.epochs):
            hidden, logits = self._forward(xs)
            prob = np.clip(expit(logits), 1e-6, 1 - 1e-6)
            grad_logits = (prob - y) / len(y)
            grad_w2 = hidden.T @ grad_logits + self.l2 * self.w2_
            grad_b2 = float(np.sum(grad_logits))
            grad_hidden = np.outer(grad_logits, self.w2_) * (1.0 - hidden * hidden)
            grad_w1 = xs.T @ grad_hidden + self.l2 * self.w1_
            grad_b1 = np.sum(grad_hidden, axis=0)

            self.w2_ = self.w2_ - self.learning_rate * grad_w2
            self.b2_ = float(self.b2_ - self.learning_rate * grad_b2)
            self.w1_ = self.w1_ - self.learning_rate * grad_w1
            self.b1_ = self.b1_ - self.learning_rate * grad_b1

        self.fallback_prob_ = None
        self.training_backend_ = "numpy"
        self.training_device_ = "cpu"
        return self

    def predict_proba(self, df: pd.DataFrame, feature_cols: List[str] | None = None) -> np.ndarray:
        if feature_cols is None:
            feature_cols = self.feature_names
        xs = self._prepare_x(df, feature_cols)
        if self.fallback_prob_ is not None:
            return np.full(xs.shape[0], float(self.fallback_prob_), dtype=float)
        _, logits = self._forward(xs)
        return np.clip(expit(logits), 1e-6, 1 - 1e-6)


class MLPQuantileModel(_BaseMLPModel):
    def __init__(
        self,
        quantile: float,
        l2: float = 1.0,
        hidden_dim: int = 24,
        epochs: int = 120,
        learning_rate: float = 0.03,
        random_state: int = 7,
        device: str = "auto",
        use_torch: bool | None = None,
    ) -> None:
        super().__init__(
            l2=l2,
            hidden_dim=hidden_dim,
            epochs=epochs,
            learning_rate=learning_rate,
            random_state=random_state,
            device=device,
            use_torch=use_torch,
        )
        self.quantile = float(np.clip(float(quantile), 1e-3, 1.0 - 1e-3))
        self.fallback_value_: float | None = None

    def fit(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> "MLPQuantileModel":
        train = df.dropna(subset=feature_cols + [target_col]).copy()
        if train.empty:
            raise ValueError("No rows available for MLP quantile training after dropping NaN.")

        y = train[target_col].astype(float).to_numpy()
        x = _as_float_array(train, feature_cols)
        return self.fit_prepared(x=x, y=y, feature_cols=feature_cols)

    def fit_prepared(self, *, x: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> "MLPQuantileModel":
        if len(y) == 0 or np.asarray(x).size == 0:
            raise ValueError("No rows available for MLP quantile training after dropping NaN.")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        self.feature_names = list(feature_cols)
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        xs = (x - self.mean_) / self.std_

        q_value = float(np.quantile(y, self.quantile))
        if len(y) < 25 or float(np.nanstd(y)) < 1e-10:
            self.fallback_value_ = q_value
            self._init_params(xs.shape[1])
            self.b2_ = q_value
            self.training_backend_ = "numpy"
            self.training_device_ = "cpu"
            return self

        if self._fit_with_torch(x=xs, y=y, trainer="quantile"):
            self.fallback_value_ = None
            return self

        self._init_params(xs.shape[1])
        self.b2_ = q_value
        for _ in range(self.epochs):
            hidden, pred = self._forward(xs)
            residual = y - pred
            grad_pred = np.where(
                residual > 0.0,
                -self.quantile,
                1.0 - self.quantile,
            ) / len(y)
            grad_w2 = hidden.T @ grad_pred + self.l2 * self.w2_
            grad_b2 = float(np.sum(grad_pred))
            grad_hidden = np.outer(grad_pred, self.w2_) * (1.0 - hidden * hidden)
            grad_w1 = xs.T @ grad_hidden + self.l2 * self.w1_
            grad_b1 = np.sum(grad_hidden, axis=0)

            self.w2_ = self.w2_ - self.learning_rate * grad_w2
            self.b2_ = float(self.b2_ - self.learning_rate * grad_b2)
            self.w1_ = self.w1_ - self.learning_rate * grad_w1
            self.b1_ = self.b1_ - self.learning_rate * grad_b1

        self.fallback_value_ = None
        self.training_backend_ = "numpy"
        self.training_device_ = "cpu"
        return self

    def predict(self, df: pd.DataFrame, feature_cols: List[str] | None = None) -> np.ndarray:
        if feature_cols is None:
            feature_cols = self.feature_names
        xs = self._prepare_x(df, feature_cols)
        if self.fallback_value_ is not None:
            return np.full(xs.shape[0], float(self.fallback_value_), dtype=float)
        _, pred = self._forward(xs)
        return np.asarray(pred, dtype=float)
