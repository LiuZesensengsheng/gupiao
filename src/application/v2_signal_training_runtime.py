from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from src.application.v2_leader_runtime import build_leader_score_snapshots
from src.application.v2_policy_feature_runtime import fit_ridge_regression, predict_ridge, r2_score

_ROLES = ["leader", "core", "follower", "rebound", "laggard"]
_PHASES = ["emerging", "strengthening", "crowded", "diverging", "fading"]
_EXIT_SOURCES = ["shortlist", "top_conviction", "theme_role", "candidate_score"]


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(number):
        return float(default)
    return float(number)


def _normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def _rank_percentile(rank: int, size: int) -> float:
    if size <= 1 or rank <= 0:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - ((rank - 1) / max(1, size - 1)))))


def _dcg(values: Iterable[float]) -> float:
    total = 0.0
    for idx, value in enumerate(values, start=1):
        total += float(value) / math.log2(idx + 1.0)
    return float(total)


def leader_model_feature_names() -> list[str]:
    return [
        "negative_score",
        "candidate_score",
        "conviction_score",
        "theme_percentile",
        "theme_size_norm",
        "role_downgrade",
        "hard_negative",
        *[f"role_{item}" for item in _ROLES],
        *[f"phase_{item}" for item in _PHASES],
    ]


def exit_model_feature_names() -> list[str]:
    return [
        "negative_score",
        "candidate_score",
        "conviction_score",
        "hold_score",
        "theme_percentile",
        "theme_size_norm",
        "role_downgrade",
        "hard_negative",
        *[f"role_{item}" for item in _ROLES],
        *[f"phase_{item}" for item in _PHASES],
        *[f"source_{item}" for item in _EXIT_SOURCES],
    ]


def exit_candidate_label(
    *,
    score: float,
    threshold: float = 0.50,
) -> str:
    value = _clip01(score)
    base = _clip01(threshold)
    if value >= max(0.78, base + 0.22):
        return "exit_fast"
    if value >= max(0.60, base + 0.08):
        return "reduce"
    if value >= max(0.44, base - 0.06):
        return "watch"
    return "keep"


def runtime_exit_hold_score(
    *,
    role: str,
    phase: str,
    negative_score: float,
    candidate_score: float,
    conviction_score: float,
) -> float:
    role_score = {
        "leader": 1.00,
        "core": 0.85,
        "follower": 0.55,
        "rebound": 0.45,
        "laggard": 0.20,
    }.get(_normalize_text(role), 0.40)
    phase_score = {
        "strengthening": 1.00,
        "emerging": 0.80,
        "crowded": 0.55,
        "diverging": 0.35,
        "fading": 0.10,
    }.get(_normalize_text(phase), 0.45)
    return _clip01(
        0.34 * float(conviction_score)
        + 0.26 * float(candidate_score)
        + 0.20 * role_score
        + 0.12 * phase_score
        + 0.08 * (1.0 - float(negative_score))
    )


def _runtime_exit_sample_source(
    *,
    symbol: str,
    role: str,
    candidate_score: float,
    shortlisted: set[str],
    top_actionable: set[str],
) -> str:
    if symbol in shortlisted:
        return "shortlist"
    if symbol in top_actionable:
        return "top_conviction"
    if role in {"leader", "core"}:
        return "theme_role"
    if float(candidate_score) >= 0.56:
        return "candidate_score"
    return ""


def build_exit_behavior_runtime_rows(
    *,
    state: object | None,
    candidate_limit: int = 8,
) -> dict[str, dict[str, object]]:
    if state is None:
        return {}
    snapshots = build_leader_score_snapshots(state=state)
    if not snapshots:
        return {}

    shortlisted = {
        str(item).strip().lower()
        for item in getattr(getattr(state, "candidate_selection", None), "shortlisted_symbols", []) or []
        if str(item).strip()
    }
    grouped: dict[str, list[object]] = {}
    for item in snapshots:
        grouped.setdefault(str(item.theme), []).append(item)

    rows: dict[str, dict[str, object]] = {}
    for group in grouped.values():
        ranked_group = sorted(
            group,
            key=lambda item: (float(item.conviction_score), float(item.candidate_score), -float(item.negative_score)),
            reverse=True,
        )
        top_actionable = {
            str(item.symbol).strip().lower()
            for item in ranked_group[: max(2, min(int(candidate_limit), len(ranked_group)))]
        }
        for item in group:
            symbol = str(item.symbol).strip().lower()
            role = _normalize_text(item.role)
            phase = _normalize_text(item.theme_phase)
            rows[symbol] = {
                "symbol": symbol,
                "theme": str(item.theme),
                "role": role,
                "theme_phase": phase,
                "negative_score": float(item.negative_score),
                "candidate_score": float(item.candidate_score),
                "conviction_score": float(item.conviction_score),
                "hold_score": runtime_exit_hold_score(
                    role=role,
                    phase=phase,
                    negative_score=float(item.negative_score),
                    candidate_score=float(item.candidate_score),
                    conviction_score=float(item.conviction_score),
                ),
                "theme_percentile": _rank_percentile(int(item.theme_rank), int(item.theme_size)),
                "theme_size": int(item.theme_size),
                "role_downgrade": bool(item.role_downgrade),
                "hard_negative": bool(item.hard_negative),
                "sample_source": _runtime_exit_sample_source(
                    symbol=symbol,
                    role=role,
                    candidate_score=float(item.candidate_score),
                    shortlisted=shortlisted,
                    top_actionable=top_actionable,
                ),
            }
    return rows


def rank_exit_candidates(
    *,
    model_payload: dict[str, object] | None,
    rows_by_symbol: dict[str, dict[str, object]],
    symbols: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    threshold = exit_behavior_threshold(model_payload)
    symbol_filter = {
        str(item).strip().lower()
        for item in (symbols or [])
        if str(item).strip()
    }
    ranked: list[dict[str, object]] = []
    for symbol, row in rows_by_symbol.items():
        symbol_key = str(symbol).strip().lower()
        if symbol_filter and symbol_key not in symbol_filter:
            continue
        score = score_exit_behavior_row(
            model_payload=model_payload,
            row=row,
        )
        if score is None:
            continue
        ranked.append(
            {
                "symbol": symbol_key,
                "score": float(score),
                "label": exit_candidate_label(score=float(score), threshold=threshold),
                "row": dict(row),
            }
        )
    ranked.sort(key=lambda item: (float(item["score"]), str(item["symbol"])), reverse=True)
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = int(idx)
    return ranked


def _normalize_coef(model_payload: dict[str, object], feature_count: int) -> np.ndarray:
    raw = model_payload.get("coef", [])
    values = [_safe_float(item, 0.0) for item in raw] if isinstance(raw, list) else []
    if len(values) < feature_count:
        values.extend([0.0] * (feature_count - len(values)))
    return np.asarray(values[:feature_count], dtype=float)


def exit_behavior_threshold(model_payload: dict[str, object] | None) -> float:
    if not isinstance(model_payload, dict):
        return 0.50
    return _clip01(_safe_float(model_payload.get("threshold", 0.50), 0.50))


def score_exit_behavior_row(
    *,
    model_payload: dict[str, object] | None,
    row: dict[str, object],
) -> float | None:
    if not isinstance(model_payload, dict):
        return None
    feature_names = model_payload.get("feature_names", [])
    if not isinstance(feature_names, list) or not feature_names:
        return None
    vector = _exit_feature_vector(row)
    coef = _normalize_coef(model_payload, vector.size)
    intercept = _safe_float(model_payload.get("intercept", 0.0), 0.0)
    return _clip01(float(intercept + np.dot(vector, coef)))


def _leader_feature_vector(row: dict[str, object]) -> np.ndarray:
    role = _normalize_text(row.get("role"))
    phase = _normalize_text(row.get("theme_phase"))
    return np.asarray(
        [
            _safe_float(row.get("negative_score"), 0.0),
            _safe_float(row.get("candidate_score"), 0.0),
            _safe_float(row.get("conviction_score"), 0.0),
            _safe_float(row.get("theme_percentile"), 0.0),
            _clip01(_safe_float(row.get("theme_size"), 0.0) / 20.0),
            1.0 if bool(row.get("role_downgrade", False)) else 0.0,
            1.0 if bool(row.get("hard_negative", False)) else 0.0,
            *[1.0 if role == item else 0.0 for item in _ROLES],
            *[1.0 if phase == item else 0.0 for item in _PHASES],
        ],
        dtype=float,
    )


def _exit_feature_vector(row: dict[str, object]) -> np.ndarray:
    role = _normalize_text(row.get("role"))
    phase = _normalize_text(row.get("theme_phase"))
    source = _normalize_text(row.get("sample_source"))
    return np.asarray(
        [
            _safe_float(row.get("negative_score"), 0.0),
            _safe_float(row.get("candidate_score"), 0.0),
            _safe_float(row.get("conviction_score"), 0.0),
            _safe_float(row.get("hold_score"), 0.0),
            _safe_float(row.get("theme_percentile"), 0.0),
            _clip01(_safe_float(row.get("theme_size"), 0.0) / 20.0),
            1.0 if bool(row.get("role_downgrade", False)) else 0.0,
            1.0 if bool(row.get("hard_negative", False)) else 0.0,
            *[1.0 if role == item else 0.0 for item in _ROLES],
            *[1.0 if phase == item else 0.0 for item in _PHASES],
            *[1.0 if source == item else 0.0 for item in _EXIT_SOURCES],
        ],
        dtype=float,
    )


def _empty_model(*, model_name: str, feature_names: list[str], target_name: str, threshold: float | None = None) -> dict[str, object]:
    model = {
        "model_name": model_name,
        "model_type": "ridge_linear",
        "feature_names": list(feature_names),
        "target_name": target_name,
        "intercept": 0.0,
        "coef": [0.0] * len(feature_names),
        "train_rows": 0,
        "train_r2": 0.0,
        "train_metrics": {},
        "evaluation_rows": 0,
        "evaluation_metrics": {},
    }
    if threshold is not None:
        model["threshold"] = float(threshold)
    return model


def _leader_filter_target(row: dict[str, object]) -> float:
    bucket = _normalize_text(row.get("leader_bucket"))
    future_score = _safe_float(row.get("future_theme_score"), 0.0)
    future_excess_5 = _safe_float(row.get("future_excess_5d_vs_sector"), 0.0)
    if bool(row.get("is_true_leader", False)):
        return 1.0
    if bucket in {"true_leader", "contender"}:
        return 1.0
    if future_score >= 0.72 and future_excess_5 >= -0.01:
        return 1.0
    return 0.0


def _leader_filter_sample_weight(row: dict[str, object]) -> float:
    return float(
        max(
            0.1,
            1.0
            + 1.5 * (1.0 if bool(row.get("is_true_leader", False)) else 0.0)
            + 0.9 * (1.0 if bool(row.get("hard_negative", False)) else 0.0)
            + 0.45 * max(0.0, _safe_float(row.get("future_theme_score"), 0.0) - 0.50),
        )
    )


def _leader_rank_sample_weight(row: dict[str, object]) -> float:
    return float(
        1.0
        + 1.2 * (1.0 if bool(row.get("is_true_leader", False)) else 0.0)
        + 0.6 * max(0.0, _safe_float(row.get("future_excess_5d_vs_sector"), 0.0))
        + 0.4 * max(0.0, _safe_float(row.get("future_excess_20d_vs_sector"), 0.0))
    )


def _leader_rank_training_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [
        row
        for row in rows
        if bool(row.get("is_true_leader", False))
        or not bool(row.get("hard_negative", False))
        or _normalize_text(row.get("leader_bucket")) == "contender"
    ]
    return selected if len(selected) >= max(8, min(3, len(rows))) else list(rows)


def _evaluate_leader_filter_predictions(
    rows: list[dict[str, object]],
    preds: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    if not rows:
        return {
            "target_mean": 0.0,
            "pass_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "true_leader_survival_recall": 0.0,
            "hard_negative_filter_rate": 0.0,
        }
    clipped = np.asarray([_clip01(value) for value in preds.tolist()], dtype=float)
    actual = np.asarray([_leader_filter_target(row) for row in rows], dtype=float)
    predicted = (clipped >= float(threshold)).astype(float)
    tp = float(np.sum((predicted == 1.0) & (actual == 1.0)))
    fp = float(np.sum((predicted == 1.0) & (actual == 0.0)))
    fn = float(np.sum((predicted == 0.0) & (actual == 1.0)))
    true_leader_mask = np.asarray([1.0 if bool(row.get("is_true_leader", False)) else 0.0 for row in rows], dtype=float) == 1.0
    hard_negative_mask = (
        np.asarray([1.0 if bool(row.get("hard_negative", False)) else 0.0 for row in rows], dtype=float) == 1.0
    )
    return {
        "target_mean": float(np.mean(actual)),
        "pass_rate": float(np.mean(predicted)),
        "precision": float(tp / max(1.0, tp + fp)),
        "recall": float(tp / max(1.0, tp + fn)),
        "true_leader_survival_recall": float(np.mean(predicted[true_leader_mask])) if np.any(true_leader_mask) else 0.0,
        "hard_negative_filter_rate": float(np.mean(1.0 - predicted[hard_negative_mask]))
        if np.any(hard_negative_mask)
        else 0.0,
    }


def _two_stage_group_rankings(
    rows: list[dict[str, object]],
    filter_preds: np.ndarray,
    rank_preds: np.ndarray,
    *,
    threshold: float,
) -> dict[tuple[str, str], list[tuple[dict[str, object], float, float, bool]]]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, object], float, float, bool]]] = {}
    for row, filter_pred, rank_pred in zip(rows, filter_preds.tolist(), rank_preds.tolist()):
        key = (str(row.get("date", "")), str(row.get("theme", "")))
        grouped.setdefault(key, []).append(
            (
                row,
                float(filter_pred),
                float(rank_pred),
                float(filter_pred) >= float(threshold),
            )
        )
    ranked_groups: dict[tuple[str, str], list[tuple[dict[str, object], float, float, bool]]] = {}
    for key, group in grouped.items():
        passed = [item for item in group if item[3]]
        rejected = [item for item in group if not item[3]]
        if not passed and group:
            fallback = max(group, key=lambda item: (item[1], item[2]))
            passed = [fallback]
            rejected = [item for item in group if item is not fallback]
        passed.sort(key=lambda item: (item[2], item[1]), reverse=True)
        rejected.sort(key=lambda item: (item[1], item[2]), reverse=True)
        ranked_groups[key] = passed + rejected
    return ranked_groups


def _evaluate_two_stage_leader_predictions(
    rows: list[dict[str, object]],
    filter_preds: np.ndarray,
    rank_preds: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    ranked_groups = _two_stage_group_rankings(rows, filter_preds, rank_preds, threshold=threshold)
    top1_hits = 0.0
    top3_hits = 0.0
    ndcg_values: list[float] = []
    group_count = 0
    fallback_groups = 0
    avg_pass_count = 0.0
    for group in ranked_groups.values():
        if len(group) < 2:
            continue
        passed_count = sum(1 for _, filter_pred, _, _ in group if float(filter_pred) >= float(threshold))
        avg_pass_count += float(passed_count)
        if passed_count == 0:
            fallback_groups += 1
        true_leaders = {str(row.get("symbol", "")) for row, _, _, _ in group if bool(row.get("is_true_leader", False))}
        if not true_leaders:
            continue
        top_symbols = [str(row.get("symbol", "")) for row, _, _, _ in group[:3]]
        top1_hits += 1.0 if top_symbols and top_symbols[0] in true_leaders else 0.0
        top3_hits += 1.0 if set(top_symbols) & true_leaders else 0.0
        predicted_relevance = [_safe_float(row.get("future_theme_score"), 0.0) for row, _, _, _ in group[:3]]
        ideal_relevance = sorted((_safe_float(row.get("future_theme_score"), 0.0) for row, _, _, _ in group), reverse=True)[:3]
        idcg = _dcg(ideal_relevance)
        ndcg_values.append(0.0 if idcg <= 1e-9 else _dcg(predicted_relevance) / idcg)
        group_count += 1
    return {
        "group_count": float(group_count),
        "top1_hit_rate": float(top1_hits / max(1, group_count)),
        "top3_recall": float(top3_hits / max(1, group_count)),
        "ndcg_at_3": float(sum(ndcg_values) / max(1, len(ndcg_values))),
        "avg_pass_count": float(avg_pass_count / max(1, group_count)),
        "fallback_group_rate": float(fallback_groups / max(1, group_count)),
    }


def _select_leader_filter_threshold(
    *,
    fit_rows: list[dict[str, object]],
    filter_preds: np.ndarray,
    rank_preds: np.ndarray,
) -> float:
    if filter_preds.size <= 0:
        return 0.50
    candidate_thresholds = {0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65}
    quantiles = np.quantile(np.asarray([_clip01(value) for value in filter_preds.tolist()], dtype=float), [0.25, 0.35, 0.50, 0.65, 0.75])
    for value in quantiles.tolist():
        candidate_thresholds.add(round(_clip01(float(value)), 4))
    best_threshold = 0.50
    best_score = float("-inf")
    for threshold in sorted(candidate_thresholds):
        filter_metrics = _evaluate_leader_filter_predictions(fit_rows, filter_preds, threshold=threshold)
        two_stage_metrics = _evaluate_two_stage_leader_predictions(
            fit_rows,
            filter_preds,
            rank_preds,
            threshold=threshold,
        )
        score = (
            1.30 * float(two_stage_metrics.get("top1_hit_rate", 0.0))
            + 0.35 * float(two_stage_metrics.get("top3_recall", 0.0))
            + 0.15 * float(two_stage_metrics.get("ndcg_at_3", 0.0))
            + 0.20 * float(filter_metrics.get("hard_negative_filter_rate", 0.0))
            + 0.25 * float(filter_metrics.get("true_leader_survival_recall", 0.0))
            - 0.08 * abs(float(filter_metrics.get("pass_rate", 0.0)) - 0.55)
        )
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return _clip01(best_threshold)


def _evaluate_leader_predictions(rows: list[dict[str, object]], preds: np.ndarray) -> dict[str, float]:
    grouped: dict[tuple[str, str], list[tuple[dict[str, object], float]]] = {}
    for row, pred in zip(rows, preds.tolist()):
        grouped.setdefault((str(row.get("date", "")), str(row.get("theme", ""))), []).append((row, float(pred)))
    top1_hits = 0.0
    top3_hits = 0.0
    ndcg_values: list[float] = []
    group_count = 0
    for group in grouped.values():
        if len(group) < 2:
            continue
        true_leaders = {str(row.get("symbol", "")) for row, _ in group if bool(row.get("is_true_leader", False))}
        if not true_leaders:
            continue
        ranked = sorted(group, key=lambda item: item[1], reverse=True)
        top_symbols = [str(row.get("symbol", "")) for row, _ in ranked[:3]]
        top1_hits += 1.0 if top_symbols and top_symbols[0] in true_leaders else 0.0
        top3_hits += 1.0 if set(top_symbols) & true_leaders else 0.0
        predicted_relevance = [_safe_float(row.get("future_theme_score"), 0.0) for row, _ in ranked[:3]]
        ideal_relevance = sorted((_safe_float(row.get("future_theme_score"), 0.0) for row, _ in group), reverse=True)[:3]
        idcg = _dcg(ideal_relevance)
        ndcg_values.append(0.0 if idcg <= 1e-9 else _dcg(predicted_relevance) / idcg)
        group_count += 1
    return {
        "group_count": float(group_count),
        "top1_hit_rate": float(top1_hits / max(1, group_count)),
        "top3_recall": float(top3_hits / max(1, group_count)),
        "ndcg_at_3": float(sum(ndcg_values) / max(1, len(ndcg_values))),
    }


def _evaluate_exit_predictions(rows: list[dict[str, object]], preds: np.ndarray, *, threshold: float) -> dict[str, float]:
    if not rows:
        return {
            "target_mean": 0.0,
            "positive_rate": 0.0,
            "predicted_positive_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": 0.0,
            "rank_corr": 0.0,
            "top_bucket_avg_target": 0.0,
            "bottom_bucket_avg_target": 0.0,
            "top_bucket_positive_rate": 0.0,
            "bucket_monotonicity": 0.0,
            "avg_positive_score": 0.0,
            "avg_negative_score": 0.0,
        }
    clipped = np.asarray([_clip01(value) for value in preds.tolist()], dtype=float)
    actual_target = np.asarray(
        [_clip01(_safe_float(row.get("future_drag_score", row.get("exit_pressure_score", 0.0)), 0.0)) for row in rows],
        dtype=float,
    )
    actual = np.asarray([1.0 if bool(row.get("should_exit_early", False)) else 0.0 for row in rows], dtype=float)
    predicted = (clipped >= float(threshold)).astype(float)
    tp = float(np.sum((predicted == 1.0) & (actual == 1.0)))
    fp = float(np.sum((predicted == 1.0) & (actual == 0.0)))
    fn = float(np.sum((predicted == 0.0) & (actual == 1.0)))
    tn = float(np.sum((predicted == 0.0) & (actual == 0.0)))
    positive_mask = actual == 1.0
    negative_mask = actual == 0.0
    if clipped.size >= 2 and float(np.std(clipped)) > 1e-12 and float(np.std(actual_target)) > 1e-12:
        pred_rank = np.argsort(np.argsort(clipped)).astype(float)
        target_rank = np.argsort(np.argsort(actual_target)).astype(float)
        rank_corr = float(np.corrcoef(pred_rank, target_rank)[0, 1])
    else:
        rank_corr = 0.0
    descending = np.argsort(-clipped)
    bucket_count = min(5, int(len(rows)))
    bucket_size = max(1, int(np.ceil(len(rows) / max(1, bucket_count))))
    bucket_means: list[float] = []
    bucket_positive_rates: list[float] = []
    for idx in range(0, len(rows), bucket_size):
        bucket_idx = descending[idx: idx + bucket_size]
        if bucket_idx.size <= 0:
            continue
        bucket_means.append(float(np.mean(actual_target[bucket_idx])))
        bucket_positive_rates.append(float(np.mean(actual[bucket_idx])))
    monotonic_hits = 0
    monotonic_pairs = 0
    for left, right in zip(bucket_means, bucket_means[1:]):
        monotonic_pairs += 1
        if float(left) >= float(right):
            monotonic_hits += 1
    return {
        "target_mean": float(np.mean(actual_target)),
        "positive_rate": float(np.mean(actual)),
        "predicted_positive_rate": float(np.mean(predicted)),
        "precision": float(tp / max(1.0, tp + fp)),
        "recall": float(tp / max(1.0, tp + fn)),
        "accuracy": float((tp + tn) / max(1.0, tp + tn + fp + fn)),
        "rank_corr": float(rank_corr),
        "top_bucket_avg_target": float(bucket_means[0]) if bucket_means else 0.0,
        "bottom_bucket_avg_target": float(bucket_means[-1]) if bucket_means else 0.0,
        "top_bucket_positive_rate": float(bucket_positive_rates[0]) if bucket_positive_rates else 0.0,
        "bucket_monotonicity": float(monotonic_hits / max(1, monotonic_pairs)),
        "avg_positive_score": float(np.mean(clipped[positive_mask])) if np.any(positive_mask) else 0.0,
        "avg_negative_score": float(np.mean(clipped[negative_mask])) if np.any(negative_mask) else 0.0,
    }


def train_leader_rank_model(
    *,
    fit_rows: list[dict[str, object]],
    evaluation_rows: list[dict[str, object]] | None = None,
    l2: float = 1.0,
) -> dict[str, object]:
    feature_names = leader_model_feature_names()
    if not fit_rows:
        model = _empty_model(
            model_name="leader_rank_v2_two_stage",
            feature_names=feature_names,
            target_name="future_theme_score",
        )
        model["leader_filter_model"] = _empty_model(
            model_name="leader_filter_v1",
            feature_names=feature_names,
            target_name="is_actionable_leader",
            threshold=0.50,
        )
        model["leader_two_stage_manifest"] = {
            "filter_threshold": 0.50,
            "rank_train_rows": 0,
            "fit_filter_metrics": {},
            "evaluation_filter_metrics": {},
        }
        return model

    evaluation_rows = list(evaluation_rows or [])
    X = np.asarray([_leader_feature_vector(row) for row in fit_rows], dtype=float)
    filter_y = np.asarray([_leader_filter_target(row) for row in fit_rows], dtype=float)
    filter_weight = np.asarray([_leader_filter_sample_weight(row) for row in fit_rows], dtype=float)
    filter_intercept, filter_coef = fit_ridge_regression(X, filter_y, l2=l2, sample_weight=filter_weight)
    fit_filter_pred_raw = np.asarray([predict_ridge(row, filter_intercept, filter_coef) for row in X], dtype=float)
    fit_filter_pred = np.asarray([_clip01(value) for value in fit_filter_pred_raw.tolist()], dtype=float)

    rank_fit_rows = _leader_rank_training_rows(fit_rows)
    X_rank = np.asarray([_leader_feature_vector(row) for row in rank_fit_rows], dtype=float)
    rank_y = np.asarray([_safe_float(row.get("future_theme_score"), 0.0) for row in rank_fit_rows], dtype=float)
    rank_weight = np.asarray([_leader_rank_sample_weight(row) for row in rank_fit_rows], dtype=float)
    rank_intercept, rank_coef = fit_ridge_regression(X_rank, rank_y, l2=l2, sample_weight=rank_weight)
    fit_rank_pred = np.asarray([predict_ridge(row, rank_intercept, rank_coef) for row in X], dtype=float)
    rank_train_pred = np.asarray([predict_ridge(row, rank_intercept, rank_coef) for row in X_rank], dtype=float)

    threshold = _select_leader_filter_threshold(
        fit_rows=fit_rows,
        filter_preds=fit_filter_pred,
        rank_preds=fit_rank_pred,
    )

    evaluation_X = np.asarray([_leader_feature_vector(row) for row in evaluation_rows], dtype=float)
    eval_filter_pred = np.asarray(
        [_clip01(predict_ridge(row, filter_intercept, filter_coef)) for row in evaluation_X],
        dtype=float,
    )
    eval_rank_pred = np.asarray(
        [predict_ridge(row, rank_intercept, rank_coef) for row in evaluation_X],
        dtype=float,
    )

    fit_filter_metrics = _evaluate_leader_filter_predictions(fit_rows, fit_filter_pred, threshold=threshold)
    evaluation_filter_metrics = _evaluate_leader_filter_predictions(
        evaluation_rows,
        eval_filter_pred,
        threshold=threshold,
    )
    return {
        "model_name": "leader_rank_v2_two_stage",
        "model_type": "ridge_linear",
        "feature_names": feature_names,
        "target_name": "future_theme_score",
        "intercept": float(rank_intercept),
        "coef": [float(value) for value in rank_coef.tolist()],
        "train_rows": int(len(fit_rows)),
        "train_r2": float(r2_score(rank_y, rank_train_pred)),
        "train_metrics": _evaluate_two_stage_leader_predictions(
            fit_rows,
            fit_filter_pred,
            fit_rank_pred,
            threshold=threshold,
        ),
        "evaluation_rows": int(len(evaluation_rows)),
        "evaluation_metrics": _evaluate_two_stage_leader_predictions(
            evaluation_rows,
            eval_filter_pred,
            eval_rank_pred,
            threshold=threshold,
        ),
        "leader_filter_model": {
            "model_name": "leader_filter_v1",
            "model_type": "ridge_linear",
            "feature_names": feature_names,
            "target_name": "is_actionable_leader",
            "threshold": float(threshold),
            "intercept": float(filter_intercept),
            "coef": [float(value) for value in filter_coef.tolist()],
            "train_rows": int(len(fit_rows)),
            "train_r2": float(r2_score(filter_y, fit_filter_pred)),
            "train_metrics": fit_filter_metrics,
            "evaluation_rows": int(len(evaluation_rows)),
            "evaluation_metrics": evaluation_filter_metrics,
        },
        "leader_two_stage_manifest": {
            "filter_threshold": float(threshold),
            "rank_train_rows": int(len(rank_fit_rows)),
            "fit_filter_metrics": fit_filter_metrics,
            "evaluation_filter_metrics": evaluation_filter_metrics,
            "rank_stage_train_metrics": _evaluate_leader_predictions(rank_fit_rows, rank_train_pred),
        },
    }


def train_exit_behavior_model(
    *,
    fit_rows: list[dict[str, object]],
    evaluation_rows: list[dict[str, object]] | None = None,
    l2: float = 1.0,
    threshold: float = 0.50,
) -> dict[str, object]:
    feature_names = exit_model_feature_names()
    if not fit_rows:
        return _empty_model(
            model_name="exit_behavior_v1",
            feature_names=feature_names,
            target_name="future_drag_score",
            threshold=threshold,
        )

    X = np.asarray([_exit_feature_vector(row) for row in fit_rows], dtype=float)
    y = np.asarray(
        [_clip01(_safe_float(row.get("future_drag_score", row.get("exit_pressure_score", 0.0)), 0.0)) for row in fit_rows],
        dtype=float,
    )
    sample_weight = np.asarray(
        [
            max(
                0.1,
                _safe_float(
                    row.get(
                        "sample_weight",
                        1.0
                        + 0.8 * _safe_float(row.get("future_drag_score", row.get("exit_pressure_score", 0.0)), 0.0)
                        + 0.35 * (1.0 if bool(row.get("should_exit_early", False)) else 0.0),
                    ),
                    1.0,
                ),
            )
            for row in fit_rows
        ],
        dtype=float,
    )
    intercept, coef = fit_ridge_regression(X, y, l2=l2, sample_weight=sample_weight)
    train_pred = np.asarray([predict_ridge(row, intercept, coef) for row in X], dtype=float)
    evaluation_rows = list(evaluation_rows or [])
    eval_pred = np.asarray(
        [predict_ridge(_exit_feature_vector(row), intercept, coef) for row in evaluation_rows],
        dtype=float,
    )
    return {
        "model_name": "exit_behavior_v1",
        "model_type": "ridge_linear",
        "feature_names": feature_names,
        "target_name": "future_drag_score",
        "threshold": float(threshold),
        "intercept": float(intercept),
        "coef": [float(value) for value in coef.tolist()],
        "train_rows": int(len(fit_rows)),
        "train_r2": float(r2_score(y, np.asarray([_clip01(value) for value in train_pred], dtype=float))),
        "train_metrics": _evaluate_exit_predictions(fit_rows, train_pred, threshold=threshold),
        "evaluation_rows": int(len(evaluation_rows)),
        "evaluation_metrics": _evaluate_exit_predictions(evaluation_rows, eval_pred, threshold=threshold),
    }


def build_signal_training_artifacts(
    *,
    leader_fit_rows: list[dict[str, object]],
    leader_evaluation_rows: list[dict[str, object]] | None = None,
    exit_fit_rows: list[dict[str, object]],
    exit_evaluation_rows: list[dict[str, object]] | None = None,
    l2: float = 1.0,
) -> dict[str, object]:
    leader_model = train_leader_rank_model(
        fit_rows=leader_fit_rows,
        evaluation_rows=leader_evaluation_rows,
        l2=l2,
    )
    exit_model = train_exit_behavior_model(
        fit_rows=exit_fit_rows,
        evaluation_rows=exit_evaluation_rows,
        l2=l2,
    )
    return {
        "signal_training_manifest": {
            "leader_fit_rows": int(len(leader_fit_rows)),
            "leader_evaluation_rows": int(len(leader_evaluation_rows or [])),
            "leader_eval_top1_hit_rate": float(leader_model.get("evaluation_metrics", {}).get("top1_hit_rate", 0.0)),
            "leader_eval_top3_recall": float(leader_model.get("evaluation_metrics", {}).get("top3_recall", 0.0)),
            "leader_eval_ndcg_at_3": float(leader_model.get("evaluation_metrics", {}).get("ndcg_at_3", 0.0)),
            "leader_eval_filter_precision": float(
                leader_model.get("leader_filter_model", {}).get("evaluation_metrics", {}).get("precision", 0.0)
            ),
            "leader_eval_filter_recall": float(
                leader_model.get("leader_filter_model", {}).get("evaluation_metrics", {}).get("recall", 0.0)
            ),
            "leader_eval_filter_pass_rate": float(
                leader_model.get("leader_filter_model", {}).get("evaluation_metrics", {}).get("pass_rate", 0.0)
            ),
            "leader_eval_true_leader_survival_recall": float(
                leader_model.get("leader_filter_model", {})
                .get("evaluation_metrics", {})
                .get("true_leader_survival_recall", 0.0)
            ),
            "leader_eval_hard_negative_filter_rate": float(
                leader_model.get("leader_filter_model", {})
                .get("evaluation_metrics", {})
                .get("hard_negative_filter_rate", 0.0)
            ),
            "leader_rank_train_rows": int(leader_model.get("leader_two_stage_manifest", {}).get("rank_train_rows", 0)),
            "exit_fit_rows": int(len(exit_fit_rows)),
            "exit_evaluation_rows": int(len(exit_evaluation_rows or [])),
            "exit_eval_rank_corr": float(exit_model.get("evaluation_metrics", {}).get("rank_corr", 0.0)),
            "exit_eval_top_bucket_avg_target": float(
                exit_model.get("evaluation_metrics", {}).get("top_bucket_avg_target", 0.0)
            ),
            "exit_eval_bottom_bucket_avg_target": float(
                exit_model.get("evaluation_metrics", {}).get("bottom_bucket_avg_target", 0.0)
            ),
            "exit_eval_bucket_monotonicity": float(
                exit_model.get("evaluation_metrics", {}).get("bucket_monotonicity", 0.0)
            ),
            "exit_eval_precision": float(exit_model.get("evaluation_metrics", {}).get("precision", 0.0)),
            "exit_eval_recall": float(exit_model.get("evaluation_metrics", {}).get("recall", 0.0)),
            "exit_eval_accuracy": float(exit_model.get("evaluation_metrics", {}).get("accuracy", 0.0)),
        },
        "leader_rank_model": leader_model,
        "exit_behavior_model": exit_model,
    }
