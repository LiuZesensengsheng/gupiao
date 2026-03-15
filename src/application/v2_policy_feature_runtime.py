from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from src.application.v2_contracts import CompositeState, StockForecastState


@dataclass(frozen=True)
class PolicyFeatureRuntimeDependencies:
    clip: Callable[[float, float, float], float]
    alpha_opportunity_metrics: Callable[[Iterable[StockForecastState]], dict[str, float]]
    candidate_stocks_from_state: Callable[[CompositeState], list[StockForecastState]]
    candidate_risk_snapshot: Callable[[Iterable[StockForecastState]], dict[str, float]]


def policy_feature_names() -> list[str]:
    return [
        "mkt_up_1d",
        "mkt_up_20d",
        "mkt_drawdown_risk",
        "mkt_liquidity_stress",
        "cross_fund_flow",
        "cross_margin_risk_on",
        "cross_breadth",
        "cross_leader_participation",
        "cross_weak_ratio",
        "top_sector_up_20d",
        "top_sector_relative_strength",
        "top_stock_up_20d",
        "top_stock_tradeability",
        "top_stock_excess_vs_sector",
        "alpha_headroom",
        "alpha_breadth",
        "alpha_top_score",
        "alpha_avg_top3",
        "alpha_median_score",
        "candidate_shortlist_ratio",
        "candidate_shortlist_size_norm",
        "candidate_alpha_breadth",
        "candidate_durability",
    ]


def policy_feature_vector(
    state: CompositeState,
    *,
    deps: PolicyFeatureRuntimeDependencies,
) -> np.ndarray:
    top_sector = state.sectors[0] if state.sectors else None
    top_stock = state.stocks[0] if state.stocks else None
    alpha_metrics = deps.alpha_opportunity_metrics(state.stocks)
    candidate_selection = getattr(state, "candidate_selection", None)
    candidate_stocks = deps.candidate_stocks_from_state(state)
    candidate_alpha_metrics = deps.alpha_opportunity_metrics(candidate_stocks)
    candidate_risk = deps.candidate_risk_snapshot(candidate_stocks)
    shortlist_ratio = float(
        getattr(candidate_selection, "shortlist_ratio", 0.0)
        or (len(candidate_stocks) / max(1, len(state.stocks)))
    )
    shortlist_size_norm = float(
        deps.clip(len(candidate_stocks) / max(4.0, min(16.0, len(state.stocks) / 8.0)), 0.0, 1.0)
    )
    return np.asarray(
        [
            float(state.market.up_1d_prob),
            float(state.market.up_20d_prob),
            float(state.market.drawdown_risk),
            float(state.market.liquidity_stress),
            float(state.cross_section.fund_flow_strength),
            float(state.cross_section.margin_risk_on_score),
            float(state.cross_section.breadth_strength),
            float(state.cross_section.leader_participation),
            float(state.cross_section.weak_stock_ratio),
            0.0 if top_sector is None else float(top_sector.up_20d_prob),
            0.0 if top_sector is None else float(top_sector.relative_strength),
            0.0 if top_stock is None else float(top_stock.up_20d_prob),
            0.0 if top_stock is None else float(top_stock.tradeability_score),
            0.0 if top_stock is None else float(top_stock.excess_vs_sector_prob),
            float(alpha_metrics["alpha_headroom"]),
            float(alpha_metrics["breadth_ratio"]),
            float(alpha_metrics["top_score"]),
            float(alpha_metrics["avg_top3"]),
            float(alpha_metrics["median_score"]),
            shortlist_ratio,
            shortlist_size_norm,
            float(candidate_alpha_metrics["breadth_ratio"]),
            float(candidate_risk["durability_score"]),
        ],
        dtype=float,
    )


def fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.size == 0 or y.size == 0:
        return 0.0, np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    X_aug = np.hstack([ones, X])
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weight.size != X.shape[0]:
            raise ValueError("sample_weight dimension mismatch")
        weight = np.sqrt(np.clip(weight, 1e-9, None)).reshape(-1, 1)
        X_aug = X_aug * weight
        y = y * weight.reshape(-1)
    reg = np.eye(X_aug.shape[1], dtype=float) * float(max(0.0, l2))
    reg[0, 0] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    return float(coef[0]), np.asarray(coef[1:], dtype=float)


def predict_ridge(features: np.ndarray, intercept: float, coef: np.ndarray) -> float:
    return float(intercept + np.dot(np.asarray(features, dtype=float), np.asarray(coef, dtype=float)))


def normalize_coef_vector(coef: object, expected_dim: int) -> np.ndarray:
    arr = np.asarray(coef, dtype=float).reshape(-1)
    target_dim = max(0, int(expected_dim))
    if arr.size == target_dim:
        return arr
    if arr.size > target_dim:
        return np.asarray(arr[:target_dim], dtype=float)
    out = np.zeros(target_dim, dtype=float)
    if arr.size > 0:
        out[: arr.size] = arr
    return out


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
