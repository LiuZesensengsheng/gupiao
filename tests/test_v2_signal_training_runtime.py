from __future__ import annotations

import numpy as np

from src.application.v2_signal_training_runtime import (
    _evaluate_exit_predictions,
    _exit_feature_vector,
    _select_exit_behavior_threshold,
    build_signal_training_artifacts,
    exit_model_feature_names,
    exit_candidate_label,
)


def test_build_signal_training_artifacts_trains_leader_and_exit_models() -> None:
    leader_fit_rows = [
        {
            "date": "2026-03-01",
            "symbol": "AAA",
            "theme": "chips",
            "role": "leader",
            "leader_bucket": "true_leader",
            "leader_tri_label": "confirmed_leader",
            "theme_phase": "strengthening",
            "theme_percentile": 1.0,
            "theme_size": 3,
            "negative_score": 0.10,
            "candidate_score": 0.86,
            "conviction_score": 0.90,
            "role_downgrade": False,
            "hard_negative": False,
            "future_theme_score": 1.05,
            "future_theme_percentile": 1.0,
            "future_excess_5d_vs_sector": 0.08,
            "future_excess_20d_vs_sector": 0.12,
            "is_true_leader": True,
            "is_possible_leader": False,
            "is_confirmed_leader": True,
        },
        {
            "date": "2026-03-01",
            "symbol": "BBB",
            "theme": "chips",
            "role": "core",
            "leader_bucket": "contender",
            "leader_tri_label": "possible_leader",
            "theme_phase": "strengthening",
            "theme_percentile": 0.5,
            "theme_size": 3,
            "negative_score": 0.20,
            "candidate_score": 0.62,
            "conviction_score": 0.66,
            "role_downgrade": False,
            "hard_negative": False,
            "future_theme_score": 0.72,
            "future_theme_percentile": 0.5,
            "future_excess_5d_vs_sector": 0.02,
            "future_excess_20d_vs_sector": 0.04,
            "is_true_leader": False,
            "is_possible_leader": True,
            "is_confirmed_leader": False,
        },
        {
            "date": "2026-03-01",
            "symbol": "CCC",
            "theme": "chips",
            "role": "laggard",
            "leader_bucket": "hard_negative",
            "leader_tri_label": "not_leader",
            "theme_phase": "fading",
            "theme_percentile": 0.0,
            "theme_size": 3,
            "negative_score": 0.72,
            "candidate_score": 0.30,
            "conviction_score": 0.22,
            "role_downgrade": True,
            "hard_negative": True,
            "future_theme_score": 0.12,
            "future_theme_percentile": 0.0,
            "future_excess_5d_vs_sector": -0.08,
            "future_excess_20d_vs_sector": -0.12,
            "is_true_leader": False,
            "is_possible_leader": False,
            "is_confirmed_leader": False,
        },
    ]
    leader_eval_rows = [dict(item) for item in leader_fit_rows]

    exit_fit_rows = [
        {
            "date": "2026-03-01",
            "symbol": "AAA",
            "theme": "chips",
            "role": "leader",
            "theme_phase": "strengthening",
            "theme_percentile": 1.0,
            "theme_size": 3,
            "sample_source": "shortlist",
            "negative_score": 0.10,
            "candidate_score": 0.86,
            "conviction_score": 0.90,
            "hold_score": 0.92,
            "role_downgrade": False,
            "hard_negative": False,
            "breakout_quality_score": 0.86,
            "exhaustion_reversal_risk": 0.12,
            "pullback_reclaim_score": 0.74,
            "distance_to_20d_high": 0.03,
            "distance_to_20d_low": 0.18,
            "volume_breakout_ratio": 1.9,
            "breakdown_below_20_low": 0.0,
            "should_exit_early": False,
            "exit_label": "keep",
            "future_drag_score": 0.08,
            "exit_pressure_score": 0.08,
            "path_failure_score": 0.04,
            "sample_weight": 1.1,
        },
        {
            "date": "2026-03-01",
            "symbol": "BBB",
            "theme": "chips",
            "role": "core",
            "theme_phase": "crowded",
            "theme_percentile": 0.5,
            "theme_size": 3,
            "sample_source": "top_conviction",
            "negative_score": 0.34,
            "candidate_score": 0.58,
            "conviction_score": 0.54,
            "hold_score": 0.52,
            "role_downgrade": False,
            "hard_negative": False,
            "breakout_quality_score": 0.42,
            "exhaustion_reversal_risk": 0.38,
            "pullback_reclaim_score": 0.44,
            "distance_to_20d_high": -0.03,
            "distance_to_20d_low": 0.08,
            "volume_breakout_ratio": 1.1,
            "breakdown_below_20_low": 0.0,
            "should_exit_early": True,
            "exit_label": "reduce",
            "future_drag_score": 0.56,
            "exit_pressure_score": 0.56,
            "path_failure_score": 0.52,
            "sample_weight": 1.4,
        },
        {
            "date": "2026-03-01",
            "symbol": "CCC",
            "theme": "chips",
            "role": "laggard",
            "theme_phase": "fading",
            "theme_percentile": 0.0,
            "theme_size": 3,
            "sample_source": "theme_role",
            "negative_score": 0.72,
            "candidate_score": 0.30,
            "conviction_score": 0.22,
            "hold_score": 0.12,
            "role_downgrade": True,
            "hard_negative": True,
            "breakout_quality_score": 0.10,
            "exhaustion_reversal_risk": 0.88,
            "pullback_reclaim_score": 0.08,
            "distance_to_20d_high": -0.15,
            "distance_to_20d_low": -0.02,
            "volume_breakout_ratio": 0.56,
            "breakdown_below_20_low": 1.0,
            "should_exit_early": True,
            "exit_label": "exit_fast",
            "future_drag_score": 0.92,
            "exit_pressure_score": 0.92,
            "path_failure_score": 0.88,
            "sample_weight": 1.8,
        },
    ]
    exit_eval_rows = [dict(item) for item in exit_fit_rows]

    payload = build_signal_training_artifacts(
        leader_fit_rows=leader_fit_rows,
        leader_evaluation_rows=leader_eval_rows,
        exit_fit_rows=exit_fit_rows,
        exit_evaluation_rows=exit_eval_rows,
        l2=0.5,
    )

    leader_model = payload["leader_rank_model"]
    exit_model = payload["exit_behavior_model"]

    assert leader_model["train_rows"] == 3
    assert leader_model["evaluation_rows"] == 3
    assert len(leader_model["coef"]) == len(leader_model["feature_names"])
    assert leader_model["leader_filter_model"]["train_rows"] == 3
    assert leader_model["leader_two_stage_manifest"]["rank_train_rows"] >= 2
    assert leader_model["leader_filter_model"]["evaluation_metrics"]["true_leader_survival_recall"] >= 0.5
    assert leader_model["leader_filter_model"]["evaluation_metrics"]["possible_leader_recall"] >= 0.5
    assert leader_model["leader_filter_model"]["evaluation_metrics"]["not_leader_filter_rate"] >= 0.5
    assert leader_model["evaluation_metrics"]["top1_hit_rate"] >= 0.5
    assert leader_model["evaluation_metrics"]["possible_recall_at_k"] >= 0.5
    assert leader_model["evaluation_metrics"]["not_leader_avoid_rate"] >= 0.5

    assert exit_model["train_rows"] == 3
    assert exit_model["evaluation_rows"] == 3
    assert exit_model["target_name"] == "exit_pressure_score"
    assert len(exit_model["coef"]) == len(exit_model["feature_names"])
    assert "exhaustion_reversal_risk" in exit_model["feature_names"]
    assert "breakdown_below_20_low" in exit_model["feature_names"]
    assert exit_model["evaluation_metrics"]["recall"] >= 0.5
    assert exit_model["evaluation_metrics"]["rank_corr"] >= 0.5
    assert exit_model["evaluation_metrics"]["label_accuracy"] >= 0.66
    assert exit_model["evaluation_metrics"]["reduce_or_worse_recall"] >= 0.5
    assert exit_model["evaluation_metrics"]["path_corr"] >= 0.5
    assert (
        exit_model["evaluation_metrics"]["top_bucket_avg_target"]
        >= exit_model["evaluation_metrics"]["bottom_bucket_avg_target"]
    )

    summary = payload["signal_training_manifest"]
    assert summary["leader_fit_rows"] == 3
    assert summary["leader_eval_filter_possible_recall"] >= 0.5
    assert summary["leader_eval_filter_not_leader_rate"] >= 0.5
    assert summary["leader_eval_possible_recall_at_k"] >= 0.5
    assert summary["leader_eval_true_leader_survival_recall"] >= 0.5
    assert summary["exit_fit_rows"] == 3
    assert summary["exit_eval_rank_corr"] >= 0.5
    assert summary["exit_eval_label_accuracy"] >= 0.66
    assert summary["exit_eval_reduce_or_worse_recall"] >= 0.5


def test_exit_candidate_label_maps_continuous_score_to_watch_reduce_and_exit_fast() -> None:
    assert exit_candidate_label(score=0.35, threshold=0.5) == "keep"
    assert exit_candidate_label(score=0.48, threshold=0.5) == "watch"
    assert exit_candidate_label(score=0.66, threshold=0.5) == "reduce"
    assert exit_candidate_label(score=0.86, threshold=0.5) == "exit_fast"


def test_exit_candidate_label_uses_observable_breakdown_for_exit_fast() -> None:
    assert (
        exit_candidate_label(
            score=0.70,
            threshold=0.68,
            phase="strengthening",
            breakdown_below_20_low=1.0,
        )
        == "exit_fast"
    )


def test_exit_feature_vector_downweights_technical_signals_for_healthy_contexts() -> None:
    names = exit_model_feature_names()
    healthy = {
        "role": "leader",
        "theme_phase": "strengthening",
        "negative_score": 0.10,
        "candidate_score": 0.84,
        "conviction_score": 0.82,
        "hold_score": 0.86,
        "theme_percentile": 1.0,
        "theme_size": 5,
        "role_downgrade": False,
        "hard_negative": False,
        "sample_source": "shortlist",
        "exhaustion_reversal_risk": 0.80,
        "pullback_reclaim_score": 0.70,
        "breakout_quality_score": 0.75,
        "distance_to_20d_high": 0.02,
        "distance_to_20d_low": 0.12,
        "volume_breakout_ratio": 1.8,
        "breakdown_below_20_low": 1.0,
    }
    fragile = dict(healthy)
    fragile.update(
        {
            "role": "laggard",
            "theme_phase": "fading",
            "negative_score": 0.72,
            "candidate_score": 0.28,
            "conviction_score": 0.24,
            "hold_score": 0.18,
            "role_downgrade": True,
            "hard_negative": True,
        }
    )
    healthy_vec = _exit_feature_vector(healthy)
    fragile_vec = _exit_feature_vector(fragile)
    tech_names = [
        "exhaustion_reversal_risk",
        "pullback_reclaim_score",
        "breakout_quality_score",
        "distance_to_20d_high",
        "distance_to_20d_low",
        "volume_breakout_ratio",
        "breakdown_below_20_low",
    ]
    for name in tech_names:
        idx = names.index(name)
        assert abs(float(fragile_vec[idx])) > abs(float(healthy_vec[idx]))


def test_select_exit_behavior_threshold_can_choose_more_conservative_cutoff_when_low_threshold_overfires() -> None:
    fit_rows = [
        {
            "theme_phase": "crowded",
            "role": "core",
            "role_downgrade": False,
            "hard_negative": False,
            "should_exit_early": False,
            "exit_label": "keep",
            "path_failure_score": 0.08,
            "future_drag_score": 0.18,
        },
        {
            "theme_phase": "crowded",
            "role": "core",
            "role_downgrade": False,
            "hard_negative": False,
            "should_exit_early": False,
            "exit_label": "keep",
            "path_failure_score": 0.10,
            "future_drag_score": 0.22,
        },
        {
            "theme_phase": "crowded",
            "role": "core",
            "role_downgrade": False,
            "hard_negative": False,
            "should_exit_early": False,
            "exit_label": "keep",
            "path_failure_score": 0.12,
            "future_drag_score": 0.24,
        },
        {
            "theme_phase": "fading",
            "role": "laggard",
            "role_downgrade": True,
            "hard_negative": True,
            "should_exit_early": True,
            "exit_label": "reduce",
            "path_failure_score": 0.58,
            "future_drag_score": 0.74,
        },
        {
            "theme_phase": "fading",
            "role": "laggard",
            "role_downgrade": True,
            "hard_negative": True,
            "should_exit_early": True,
            "exit_label": "exit_fast",
            "path_failure_score": 0.74,
            "future_drag_score": 0.88,
        },
        {
            "theme_phase": "diverging",
            "role": "rebound",
            "role_downgrade": True,
            "hard_negative": False,
            "should_exit_early": True,
            "exit_label": "reduce",
            "path_failure_score": 0.49,
            "future_drag_score": 0.68,
        },
    ]
    preds = np.asarray([0.58, 0.60, 0.62, 0.76, 0.84, 0.72], dtype=float)

    threshold = _select_exit_behavior_threshold(fit_rows=fit_rows, preds=preds)

    assert threshold >= 0.68


def test_evaluate_exit_predictions_does_not_leak_future_path_scores_into_predicted_labels() -> None:
    row = {
        "theme_phase": "strengthening",
        "role": "core",
        "role_downgrade": False,
        "hard_negative": False,
        "should_exit_early": False,
        "exit_label": "watch",
        "breakdown_below_20_low": 0.0,
        "exhaustion_reversal_risk": 0.0,
        "distance_to_20d_high": -0.12,
    }
    low_future = dict(row)
    low_future.update(
        {
            "path_failure_score": 0.0,
            "breakdown_path_score": 0.0,
            "rebound_failure_score": 0.0,
        }
    )
    high_future = dict(row)
    high_future.update(
        {
            "path_failure_score": 0.95,
            "breakdown_path_score": 0.95,
            "rebound_failure_score": 0.95,
        }
    )

    low_metrics = _evaluate_exit_predictions([low_future], np.asarray([0.62], dtype=float), threshold=0.68)
    high_metrics = _evaluate_exit_predictions([high_future], np.asarray([0.62], dtype=float), threshold=0.68)

    assert low_metrics["predicted_positive_rate"] == high_metrics["predicted_positive_rate"]
    assert low_metrics["label_accuracy"] == high_metrics["label_accuracy"]
