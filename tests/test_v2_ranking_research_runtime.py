from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.application.v2_contracts import (
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    MainlineState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
    StockRoleSnapshot,
    ThemeEpisode,
)
from src.application.v2_leader_runtime import (
    build_leader_artifact_payloads,
    build_research_label_artifact_payloads,
)
from src.application.v2_ranking_research_runtime import (
    RankingResearchDependencies,
    persist_v2_ranking_research_artifacts,
    run_v2_ranking_research,
)
from src.application.v2_signal_training_runtime import build_signal_training_artifacts
from src.interfaces.presenters.v2_ranking_research_presenters import (
    write_v2_ranking_research_dashboard,
    write_v2_ranking_research_report,
)


def _leader_state(as_of_date: str) -> CompositeState:
    return CompositeState(
        market=MarketForecastState(
            as_of_date=as_of_date,
            up_1d_prob=0.57,
            up_5d_prob=0.61,
            up_20d_prob=0.64,
            trend_state="trend",
            drawdown_risk=0.18,
            volatility_regime="normal",
            liquidity_stress=0.16,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date=as_of_date,
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.04,
            fund_flow_strength=0.12,
            margin_risk_on_score=0.10,
            breadth_strength=0.22,
            leader_participation=0.64,
            weak_stock_ratio=0.18,
        ),
        sectors=[SectorForecastState("chips", 0.61, 0.67, 0.18, 0.20, 0.18)],
        stocks=[
            StockForecastState("AAA", "chips", 0.61, 0.69, 0.74, 0.67, 0.14, 0.93, alpha_score=0.88),
            StockForecastState("BBB", "chips", 0.57, 0.61, 0.66, 0.58, 0.10, 0.89, alpha_score=0.72),
            StockForecastState("CCC", "chips", 0.49, 0.46, 0.52, 0.47, 0.04, 0.73, alpha_score=0.41),
        ],
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["AAA", "BBB", "CCC"],
            shortlisted_sectors=["chips"],
            total_scored=3,
            shortlist_size=3,
            shortlist_ratio=1.0,
            selection_mode="macro_sector_ranking",
            selection_notes=["test shortlist"],
        ),
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        mainlines=[
            MainlineState(
                name="chips",
                conviction=0.68,
                breadth=0.36,
                leadership=0.30,
                catalyst_strength=0.24,
                event_risk_level=0.16,
                sectors=["chips"],
                representative_symbols=["AAA", "BBB"],
            )
        ],
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="strengthening",
                conviction=0.69,
                breadth=0.34,
                leadership=0.32,
                event_risk=0.18,
                sectors=["chips"],
                representative_symbols=["AAA", "BBB"],
            )
        ],
        stock_role_states={
            "AAA": StockRoleSnapshot(symbol="AAA", theme="chips", role="leader"),
            "BBB": StockRoleSnapshot(symbol="BBB", theme="chips", role="core"),
            "CCC": StockRoleSnapshot(symbol="CCC", theme="chips", role="laggard", previous_role="core", role_downgrade=True),
        },
    )


def _make_trajectory() -> SimpleNamespace:
    prepared = SimpleNamespace(
        stock_frames={
            "AAA": pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02")],
                    "excess_ret_5_vs_sector": [0.14, 0.16],
                    "excess_ret_20_vs_sector": [0.22, 0.24],
                    "fwd_ret_1": [0.03, 0.02],
                    "excess_ret_1_vs_mkt": [0.01, 0.01],
                }
            ),
            "BBB": pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02")],
                    "excess_ret_5_vs_sector": [0.05, 0.06],
                    "excess_ret_20_vs_sector": [0.08, 0.09],
                    "fwd_ret_1": [0.01, 0.01],
                    "excess_ret_1_vs_mkt": [0.0, 0.0],
                }
            ),
            "CCC": pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-02")],
                    "excess_ret_5_vs_sector": [-0.07, -0.09],
                    "excess_ret_20_vs_sector": [-0.11, -0.14],
                    "fwd_ret_1": [-0.03, -0.04],
                    "excess_ret_1_vs_mkt": [-0.02, -0.03],
                }
            ),
        }
    )
    return SimpleNamespace(
        prepared=prepared,
        steps=[
            SimpleNamespace(date=pd.Timestamp("2026-03-01"), composite_state=_leader_state("2026-03-01")),
            SimpleNamespace(date=pd.Timestamp("2026-03-02"), composite_state=_leader_state("2026-03-02")),
        ],
    )


def _split_trajectory(trajectory: SimpleNamespace, split_mode: str = "purged_wf", embargo_days: int = 20) -> tuple[object, object, object]:
    del split_mode, embargo_days
    return (
        SimpleNamespace(prepared=trajectory.prepared, steps=[trajectory.steps[0]]),
        SimpleNamespace(prepared=trajectory.prepared, steps=[]),
        SimpleNamespace(prepared=trajectory.prepared, steps=[trajectory.steps[1]]),
    )


def test_run_v2_ranking_research_builds_metrics_and_candidates(tmp_path: Path) -> None:
    progress: list[tuple[str, str]] = []
    trajectory = _make_trajectory()
    deps = RankingResearchDependencies(
        emit_progress_fn=lambda stage, message: progress.append((stage, message)),
        load_or_build_v2_backtest_trajectory_fn=lambda **_: trajectory,
        split_research_trajectory_fn=_split_trajectory,
        trajectory_step_count_fn=lambda item: 0 if item is None else len(getattr(item, "steps", []) or []),
        build_leader_artifact_payloads_fn=build_leader_artifact_payloads,
        build_research_label_artifact_payloads_fn=build_research_label_artifact_payloads,
        build_signal_training_artifacts_fn=build_signal_training_artifacts,
    )

    result = run_v2_ranking_research(
        dependencies=deps,
        strategy_id="alpha_v2",
        candidate_limit=2,
        top_k=2,
        signal_l2=0.5,
        cache_root=str(tmp_path / "cache"),
    )

    assert result.strategy_id == "alpha_v2"
    assert result.fit_steps == 1
    assert result.evaluation_steps == 1
    assert result.evaluation_scope == "holdout"
    assert len(result.leader_candidates) == 2
    assert result.leader_candidates[0]["symbol"] == "AAA"
    assert result.signal_training_manifest["leader_fit_rows"] == 3
    assert result.signal_training_manifest["leader_rank_train_rows"] >= 2
    assert "leader_filter_model" in result.leader_rank_model
    assert result.signal_training_manifest["exit_fit_rows"] == 3
    assert result.summary()["leader_metrics"]["candidate_recall_at_k"] >= 1.0
    assert any(stage == "ranking" and "排序研究完成" in message for stage, message in progress)

    artifacts = persist_v2_ranking_research_artifacts(result, artifact_root=str(tmp_path / "artifacts"))
    report_path = write_v2_ranking_research_report(tmp_path / "ranking.md", result)
    dashboard_path = write_v2_ranking_research_dashboard(tmp_path / "ranking.html", result)

    assert Path(artifacts["manifest_path"]).exists()
    assert Path(artifacts["leader_candidates_path"]).exists()
    assert report_path.read_text(encoding="utf-8").find("AAA") >= 0
    assert dashboard_path.read_text(encoding="utf-8").find("V2 Ranking Research") >= 0


def test_run_v2_ranking_research_handles_empty_trajectory() -> None:
    deps = RankingResearchDependencies(
        emit_progress_fn=lambda stage, message: None,
        load_or_build_v2_backtest_trajectory_fn=lambda **_: None,
        split_research_trajectory_fn=lambda trajectory, split_mode="purged_wf", embargo_days=20: (None, None, None),
        trajectory_step_count_fn=lambda item: 0 if item is None else len(getattr(item, "steps", []) or []),
        build_leader_artifact_payloads_fn=build_leader_artifact_payloads,
        build_research_label_artifact_payloads_fn=build_research_label_artifact_payloads,
        build_signal_training_artifacts_fn=build_signal_training_artifacts,
    )

    result = run_v2_ranking_research(
        dependencies=deps,
        strategy_id="alpha_v2",
    )

    assert result.trajectory_steps == 0
    assert result.fit_steps == 0
    assert result.evaluation_steps == 0
    assert result.leader_candidates == []
    assert result.signal_training_manifest["leader_fit_rows"] == 0
    assert result.signal_training_manifest["leader_rank_train_rows"] == 0
    assert result.signal_training_manifest["exit_fit_rows"] == 0


def test_run_v2_ranking_research_reuses_result_cache(tmp_path: Path) -> None:
    progress: list[tuple[str, str]] = []
    trajectory = _make_trajectory()
    setattr(trajectory, "_decorated_trajectory_cache_key", "trajectory-cache-key")
    seen = {
        "leader": 0,
        "labels": 0,
        "signal": 0,
    }

    def counted_leader_payloads(**kwargs: object) -> dict[str, object]:
        seen["leader"] += 1
        return build_leader_artifact_payloads(**kwargs)

    def counted_label_payloads(**kwargs: object) -> dict[str, object]:
        seen["labels"] += 1
        return build_research_label_artifact_payloads(**kwargs)

    def counted_signal_payloads(**kwargs: object) -> dict[str, object]:
        seen["signal"] += 1
        return build_signal_training_artifacts(**kwargs)

    deps = RankingResearchDependencies(
        emit_progress_fn=lambda stage, message: progress.append((stage, message)),
        load_or_build_v2_backtest_trajectory_fn=lambda **_: trajectory,
        split_research_trajectory_fn=_split_trajectory,
        trajectory_step_count_fn=lambda item: 0 if item is None else len(getattr(item, "steps", []) or []),
        build_leader_artifact_payloads_fn=counted_leader_payloads,
        build_research_label_artifact_payloads_fn=counted_label_payloads,
        build_signal_training_artifacts_fn=counted_signal_payloads,
    )

    first = run_v2_ranking_research(
        dependencies=deps,
        strategy_id="alpha_v2",
        cache_root=str(tmp_path / "cache"),
        candidate_limit=2,
        top_k=2,
    )
    second = run_v2_ranking_research(
        dependencies=deps,
        strategy_id="alpha_v2",
        cache_root=str(tmp_path / "cache"),
        candidate_limit=2,
        top_k=2,
    )

    assert first.summary() == second.summary()
    assert "leader_filter_model" in first.leader_rank_model
    assert seen["leader"] == 1
    assert seen["labels"] == 3
    assert seen["signal"] == 1
    assert any(stage == "cache" and "ranking research" in message for stage, message in progress)
