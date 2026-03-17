from __future__ import annotations

from src.application.v2_signal_training_runtime import (
    build_signal_training_artifacts,
    exit_candidate_label,
)


def test_build_signal_training_artifacts_trains_leader_and_exit_models() -> None:
    leader_fit_rows = [
        {
            "date": "2026-03-01",
            "symbol": "AAA",
            "theme": "chips",
            "role": "leader",
            "theme_phase": "strengthening",
            "theme_percentile": 1.0,
            "theme_size": 3,
            "negative_score": 0.10,
            "candidate_score": 0.86,
            "conviction_score": 0.90,
            "role_downgrade": False,
            "hard_negative": False,
            "future_theme_score": 1.05,
            "future_excess_5d_vs_sector": 0.08,
            "future_excess_20d_vs_sector": 0.12,
            "is_true_leader": True,
        },
        {
            "date": "2026-03-01",
            "symbol": "BBB",
            "theme": "chips",
            "role": "core",
            "theme_phase": "strengthening",
            "theme_percentile": 0.5,
            "theme_size": 3,
            "negative_score": 0.20,
            "candidate_score": 0.62,
            "conviction_score": 0.66,
            "role_downgrade": False,
            "hard_negative": False,
            "future_theme_score": 0.72,
            "future_excess_5d_vs_sector": 0.02,
            "future_excess_20d_vs_sector": 0.04,
            "is_true_leader": False,
        },
        {
            "date": "2026-03-01",
            "symbol": "CCC",
            "theme": "chips",
            "role": "laggard",
            "theme_phase": "fading",
            "theme_percentile": 0.0,
            "theme_size": 3,
            "negative_score": 0.72,
            "candidate_score": 0.30,
            "conviction_score": 0.22,
            "role_downgrade": True,
            "hard_negative": True,
            "future_theme_score": 0.12,
            "future_excess_5d_vs_sector": -0.08,
            "future_excess_20d_vs_sector": -0.12,
            "is_true_leader": False,
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
            "should_exit_early": False,
            "future_drag_score": 0.08,
            "exit_pressure_score": 0.08,
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
            "should_exit_early": True,
            "future_drag_score": 0.56,
            "exit_pressure_score": 0.56,
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
            "should_exit_early": True,
            "future_drag_score": 0.92,
            "exit_pressure_score": 0.92,
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
    assert leader_model["evaluation_metrics"]["top1_hit_rate"] >= 0.5

    assert exit_model["train_rows"] == 3
    assert exit_model["evaluation_rows"] == 3
    assert len(exit_model["coef"]) == len(exit_model["feature_names"])
    assert exit_model["evaluation_metrics"]["recall"] >= 0.5
    assert exit_model["evaluation_metrics"]["rank_corr"] >= 0.5
    assert (
        exit_model["evaluation_metrics"]["top_bucket_avg_target"]
        >= exit_model["evaluation_metrics"]["bottom_bucket_avg_target"]
    )

    summary = payload["signal_training_manifest"]
    assert summary["leader_fit_rows"] == 3
    assert summary["exit_fit_rows"] == 3
    assert summary["exit_eval_rank_corr"] >= 0.5


def test_exit_candidate_label_maps_continuous_score_to_watch_reduce_and_exit_fast() -> None:
    assert exit_candidate_label(score=0.35, threshold=0.5) == "keep"
    assert exit_candidate_label(score=0.48, threshold=0.5) == "watch"
    assert exit_candidate_label(score=0.66, threshold=0.5) == "reduce"
    assert exit_candidate_label(score=0.86, threshold=0.5) == "exit_fast"
