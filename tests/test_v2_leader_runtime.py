from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.application.v2_contracts import (
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


def test_apply_leader_candidate_overlay_reorders_without_changing_membership() -> None:
    state = _leader_state()
    state = state.__class__(
        **{
                **state.__dict__,
                "candidate_selection": state.candidate_selection.__class__(
                    shortlisted_symbols=["BBB", "AAA", "CCC"],
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
    assert any("Leader overlay reprioritized" in note for note in updated.candidate_selection.selection_notes)
