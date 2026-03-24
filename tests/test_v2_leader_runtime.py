from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

from src.application.v2_contracts import (
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    InfoAggregateState,
    MainlineState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
    StockRoleSnapshot,
    ThemeEpisode,
)
from src.application.v2_leader_runtime import (
    apply_leader_candidate_overlay,
    build_leader_artifact_payloads,
    build_leader_score_snapshots,
    evaluate_leader_candidates,
    top_leader_candidates,
)


def _leader_state() -> CompositeState:
    return CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-12",
            up_1d_prob=0.57,
            up_5d_prob=0.61,
            up_20d_prob=0.64,
            trend_state="trend",
            drawdown_risk=0.18,
            volatility_regime="normal",
            liquidity_stress=0.16,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-12",
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


def test_build_leader_score_snapshots_separates_strong_and_weak_names() -> None:
    state = _leader_state()

    snapshots = build_leader_score_snapshots(state=state)
    snapshot_map = {item.symbol: item for item in snapshots}

    assert snapshots[0].symbol == "AAA"
    assert snapshot_map["AAA"].candidate_score > snapshot_map["BBB"].candidate_score > snapshot_map["CCC"].candidate_score
    assert snapshot_map["AAA"].conviction_score > snapshot_map["CCC"].conviction_score
    assert snapshot_map["CCC"].hard_negative is True
    assert "theme strengthening" in snapshot_map["AAA"].reasons
    assert "role downgrade active" in snapshot_map["CCC"].reasons


def test_build_leader_score_snapshots_uses_info_and_shortlist_support() -> None:
    state = _leader_state()
    state = replace(
        state,
        stocks=[
            replace(state.stocks[0], alpha_score=0.60, tradeability_score=0.87),
            replace(state.stocks[1], excess_vs_sector_prob=0.53, tradeability_score=0.80, alpha_score=0.64),
            state.stocks[2],
        ],
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["AAA", "BBB", "CCC"],
            shortlisted_sectors=["chips"],
            total_scored=300,
            shortlist_size=2,
            shortlist_ratio=2.0 / 300.0,
            selection_mode="macro_sector_ranking",
            selection_notes=["macro shortlist"],
        ),
        market_info_state=InfoAggregateState(
            info_prob_5d=0.58,
            info_prob_20d=0.60,
            source_diversity=0.40,
        ),
        stock_info_states={
            "AAA": InfoAggregateState(
                catalyst_strength=0.58,
                coverage_ratio=0.84,
                coverage_confidence=0.76,
                info_prob_5d=0.74,
                info_prob_20d=0.76,
                source_diversity=0.72,
                event_risk_level=0.08,
            ),
            "BBB": InfoAggregateState(
                catalyst_strength=0.04,
                coverage_ratio=0.18,
                coverage_confidence=0.16,
                info_prob_5d=0.44,
                info_prob_20d=0.45,
                source_diversity=0.14,
                event_risk_level=0.62,
                negative_event_risk=0.28,
            ),
        },
        theme_episodes=[
            ThemeEpisode(
                theme="chips",
                phase="crowded",
                conviction=0.72,
                breadth=0.30,
                leadership=0.29,
                catalyst_strength=0.26,
                event_risk=0.46,
                crowding=0.64,
                capital_support=0.42,
                macro_alignment=0.38,
                viewpoint_score=0.08,
                viewpoint_conflict=0.31,
                sectors=["chips"],
                representative_symbols=["AAA", "BBB"],
            )
        ],
    )

    snapshots = build_leader_score_snapshots(state=state)
    snapshot_map = {item.symbol: item for item in snapshots}

    assert snapshot_map["AAA"].negative_score < snapshot_map["BBB"].negative_score
    assert snapshot_map["AAA"].candidate_score > snapshot_map["BBB"].candidate_score
    assert "info catalysts aligned" in snapshot_map["AAA"].reasons
    assert "macro shortlist confirmed" in snapshot_map["AAA"].reasons


def test_evaluate_leader_candidates_uses_future_theme_relative_returns() -> None:
    state = _leader_state()
    trajectory = SimpleNamespace(
        prepared=SimpleNamespace(
            stock_frames={
                "AAA": pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2026-03-12")],
                        "excess_ret_5_vs_sector": [0.16],
                        "excess_ret_20_vs_sector": [0.24],
                    }
                ),
                "BBB": pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2026-03-12")],
                        "excess_ret_5_vs_sector": [0.06],
                        "excess_ret_20_vs_sector": [0.08],
                    }
                ),
                "CCC": pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2026-03-12")],
                        "excess_ret_5_vs_sector": [-0.09],
                        "excess_ret_20_vs_sector": [-0.14],
                    }
                ),
            }
        ),
        steps=[
            SimpleNamespace(
                date=pd.Timestamp("2026-03-12"),
                composite_state=state,
            )
        ],
    )

    evaluation = evaluate_leader_candidates(trajectory=trajectory, top_k=2)

    assert evaluation["theme_group_count"] == 1
    assert evaluation["candidate_recall_at_k"] == 1.0
    assert evaluation["conviction_precision_at_1"] == 1.0
    assert evaluation["ndcg_at_k"] > 0.95
    assert evaluation["hard_negative_survival_recall"] == 1.0


def test_build_leader_artifact_payloads_returns_manifest_and_candidates() -> None:
    state = _leader_state()
    candidates = top_leader_candidates(state=state, limit=2)
    payloads = build_leader_artifact_payloads(state=state, trajectory=None, limit=2)

    assert payloads["leader_manifest"]["candidate_count"] == 2
    assert payloads["leader_manifest"]["snapshot_count"] >= 3
    assert len(payloads["leader_candidates"]) == 2
    assert payloads["leader_candidates"][0]["symbol"] == candidates[0].symbol


def test_apply_leader_candidate_overlay_can_promote_tail_into_core() -> None:
    state = _leader_state()
    state = state.__class__(
        **{
                **state.__dict__,
                "candidate_selection": state.candidate_selection.__class__(
                    shortlisted_symbols=["BBB", "CCC", "AAA"],
                    shortlisted_sectors=["chips"],
                    total_scored=3,
                    shortlist_size=2,
                    shortlist_ratio=2.0 / 3.0,
                    selection_mode="macro_sector_ranking",
                selection_notes=["base shortlist"],
            ),
        }
    )

    updated = apply_leader_candidate_overlay(state=state)

    assert updated.candidate_selection.shortlisted_symbols[:2] == ["AAA", "BBB"]
    assert set(updated.candidate_selection.shortlisted_symbols[:2]) == {"AAA", "BBB"}
    assert updated.candidate_selection.shortlisted_symbols[2] == "CCC"
    assert updated.candidate_selection.shortlisted_sectors == ["chips"]
    assert updated.candidate_selection.sector_slots == {"chips": 2}
    assert any("Leader overlay reprioritized" in note for note in updated.candidate_selection.selection_notes)
    assert any("Promoted into core: AAA" in note for note in updated.candidate_selection.selection_notes)
    assert any("Demoted from core: CCC" in note for note in updated.candidate_selection.selection_notes)


def test_apply_leader_candidate_overlay_can_use_learned_model() -> None:
    state = _leader_state()
    state = state.__class__(
        **{
            **state.__dict__,
            "stocks": [
                replace(state.stocks[0], up_5d_prob=0.56, up_20d_prob=0.58, tradeability_score=0.82, alpha_score=0.55),
                replace(state.stocks[1], up_5d_prob=0.67, up_20d_prob=0.70, tradeability_score=0.93, alpha_score=0.84),
                state.stocks[2],
            ],
            "candidate_selection": state.candidate_selection.__class__(
                shortlisted_symbols=["BBB", "AAA", "CCC"],
                shortlisted_sectors=["chips"],
                total_scored=3,
                shortlist_size=1,
                shortlist_ratio=1.0 / 3.0,
                selection_mode="macro_sector_ranking",
            ),
        }
    )

    without_model = apply_leader_candidate_overlay(state=state)
    assert without_model.candidate_selection.shortlisted_symbols[:2] == ["BBB", "AAA"]

    learned_model = {
        "feature_names": [f"f{i}" for i in range(17)],
        "intercept": 0.05,
        "coef": [-0.4, 0.1, 0.1, 0.1, 0.0, -0.3, -0.6, 0.9, -0.3, -0.1, -0.2, -0.5, 0.2, 0.35, -0.25, -0.35, -0.5],
        "train_rows": 50,
        "leader_filter_model": {
            "feature_names": [f"f{i}" for i in range(17)],
            "intercept": 0.05,
            "coef": [-0.55, 0.08, 0.12, 0.1, 0.0, -0.35, -0.8, 1.0, -0.45, -0.15, -0.25, -0.6, 0.2, 0.45, -0.3, -0.4, -0.6],
            "threshold": 0.52,
            "train_rows": 50,
        },
    }
    updated = apply_leader_candidate_overlay(state=state, leader_rank_model=learned_model)

    assert updated.candidate_selection.shortlisted_symbols[:2] == ["AAA", "BBB"]
    assert any("Learned leader overlay reprioritized" in note for note in updated.candidate_selection.selection_notes)


def test_apply_leader_candidate_overlay_keeps_strong_fading_leader_inside_core() -> None:
    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-12",
            up_1d_prob=0.57,
            up_5d_prob=0.61,
            up_20d_prob=0.64,
            trend_state="trend",
            drawdown_risk=0.18,
            volatility_regime="normal",
            liquidity_stress=0.16,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-12",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.04,
            fund_flow_strength=0.12,
            margin_risk_on_score=0.10,
            breadth_strength=0.22,
            leader_participation=0.64,
            weak_stock_ratio=0.18,
        ),
        sectors=[
            SectorForecastState("shipping", 0.58, 0.63, 0.14, 0.16, 0.18),
            SectorForecastState("chips", 0.62, 0.68, 0.18, 0.18, 0.18),
        ],
        stocks=[
            StockForecastState("FADER", "shipping", 0.61, 0.68, 0.70, 0.64, 0.11, 0.93, alpha_score=0.88),
            StockForecastState("STRONG", "chips", 0.58, 0.64, 0.67, 0.60, 0.11, 0.91, alpha_score=0.76),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        mainlines=[
            MainlineState(
                name="shipping",
                conviction=0.22,
                breadth=0.20,
                leadership=0.14,
                catalyst_strength=0.18,
                event_risk_level=0.58,
                sectors=["shipping"],
                representative_symbols=["FADER"],
            ),
            MainlineState(
                name="chips",
                conviction=0.70,
                breadth=0.36,
                leadership=0.31,
                catalyst_strength=0.26,
                event_risk_level=0.18,
                sectors=["chips"],
                representative_symbols=["STRONG"],
            ),
        ],
        theme_episodes=[
            ThemeEpisode(
                theme="shipping",
                phase="fading",
                conviction=0.46,
                breadth=0.28,
                leadership=0.21,
                event_risk=0.54,
                crowding=0.18,
                sectors=["shipping"],
                representative_symbols=["FADER"],
            ),
            ThemeEpisode(
                theme="chips",
                phase="strengthening",
                conviction=0.66,
                breadth=0.34,
                leadership=0.29,
                event_risk=0.18,
                crowding=0.20,
                sectors=["chips"],
                representative_symbols=["STRONG"],
            ),
        ],
        stock_role_states={
            "FADER": StockRoleSnapshot(symbol="FADER", theme="shipping", role="leader"),
            "STRONG": StockRoleSnapshot(symbol="STRONG", theme="chips", role="leader"),
        },
        candidate_selection=CandidateSelectionState(
            shortlisted_symbols=["FADER", "STRONG"],
            shortlisted_sectors=["shipping", "chips"],
            total_scored=2,
            shortlist_size=2,
            shortlist_ratio=1.0,
            selection_mode="macro_sector_ranking",
            selection_notes=["base shortlist"],
        ),
    )

    updated = apply_leader_candidate_overlay(state=state)

    assert set(updated.candidate_selection.shortlisted_symbols[:2]) == {"FADER", "STRONG"}
