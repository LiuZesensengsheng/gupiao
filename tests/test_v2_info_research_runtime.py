from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.application.v2_backtest_metrics_runtime import panel_slice_metrics
from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    InfoAggregateState,
    InfoItem,
    MarketForecastState,
    StockForecastState,
)
from src.application.v2_info_research_runtime import (
    InfoResearchDependencies,
    run_v2_info_research,
)
from src.application.v2_info_shadow_runtime import (
    InfoShadowRuntimeDependencies,
    build_shadow_scored_rows_for_step,
)
from src.interfaces.presenters.v2_info_research_presenters import (
    write_v2_info_research_dashboard,
    write_v2_info_research_report,
)


def _base_state(as_of_date: str) -> CompositeState:
    return CompositeState(
        market=MarketForecastState(
            as_of_date=as_of_date,
            up_1d_prob=0.55,
            up_5d_prob=0.58,
            up_20d_prob=0.62,
            trend_state="trend",
            drawdown_risk=0.12,
            volatility_regime="normal",
            liquidity_stress=0.10,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date=as_of_date,
            large_vs_small_bias=0.01,
            growth_vs_value_bias=0.03,
            fund_flow_strength=0.12,
            margin_risk_on_score=0.10,
            breadth_strength=0.18,
            leader_participation=0.61,
            weak_stock_ratio=0.22,
        ),
        sectors=[],
        stocks=[
            StockForecastState("AAA", "chips", 0.60, 0.66, 0.71, 0.61, 0.12, 0.95, alpha_score=0.80),
            StockForecastState("BBB", "chips", 0.52, 0.56, 0.60, 0.54, 0.06, 0.84, alpha_score=0.35),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
        stock_info_states={
            "AAA": InfoAggregateState(item_count=2, negative_event_risk=0.12),
            "BBB": InfoAggregateState(item_count=0, negative_event_risk=0.0),
        },
    )


def test_build_shadow_scored_rows_for_step_exposes_extended_horizons() -> None:
    state = _base_state("2026-03-20")
    stock_frames = {
        "AAA": pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-03-20")],
                "excess_ret_1_vs_mkt": [0.02],
                "excess_ret_2_vs_mkt": [0.03],
                "excess_ret_3_vs_mkt": [0.05],
                "excess_ret_5_vs_mkt": [0.08],
                "excess_ret_10_vs_mkt": [0.11],
                "excess_ret_20_vs_sector": [0.16],
            }
        ),
        "BBB": pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-03-20")],
                "excess_ret_1_vs_mkt": [-0.01],
                "excess_ret_2_vs_mkt": [-0.02],
                "excess_ret_3_vs_mkt": [-0.03],
                "excess_ret_5_vs_mkt": [-0.04],
                "excess_ret_10_vs_mkt": [-0.06],
                "excess_ret_20_vs_sector": [-0.09],
            }
        ),
    }
    deps = InfoShadowRuntimeDependencies(
        build_sector_map_from_state=lambda state: {},
        build_info_state_maps=lambda **kwargs: (InfoAggregateState(), {}, {}),
        predict_info_shadow_prob=lambda **kwargs: (0.5, {}),
        blend_probability=lambda *args, **kwargs: 0.5,
        build_mainline_states=lambda **kwargs: [],
        stock_policy_score=lambda stock: float(stock.alpha_score),
        compose_shadow_stock_score=lambda stock, info_state: float(stock.alpha_score) + 0.05 * float(info_state.item_count),
        safe_float=lambda value, default=0.0: default if pd.isna(value) else float(value),
        info_aggregate_state_cls=InfoAggregateState,
        info_feature_frame=lambda **kwargs: pd.DataFrame(),
        fit_info_shadow_model=lambda *args, **kwargs: None,
        info_shadow_feature_columns=[],
        panel_slice_metrics=panel_slice_metrics,
    )

    rows, event_day = build_shadow_scored_rows_for_step(
        state=state,
        stock_frames=stock_frames,
        date=pd.Timestamp("2026-03-20"),
        deps=deps,
    )

    assert event_day is True
    assert {"realized_ret_2d", "realized_ret_3d", "realized_ret_10d"}.issubset(rows.columns)
    assert rows.loc[rows["symbol"] == "AAA", "realized_ret_10d"].iloc[0] == 0.11


def test_run_v2_info_research_and_presenters(tmp_path: Path) -> None:
    trajectory = SimpleNamespace(
        prepared=SimpleNamespace(
            stock_frames={
                "AAA": pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2026-03-19"), pd.Timestamp("2026-03-20")],
                        "excess_ret_1_vs_mkt": [0.01, 0.03],
                        "excess_ret_2_vs_mkt": [0.02, 0.04],
                        "excess_ret_3_vs_mkt": [0.03, 0.05],
                        "excess_ret_5_vs_mkt": [0.05, 0.07],
                        "excess_ret_10_vs_mkt": [0.08, 0.10],
                        "excess_ret_20_vs_sector": [0.11, 0.14],
                    }
                ),
                "BBB": pd.DataFrame(
                    {
                        "date": [pd.Timestamp("2026-03-19"), pd.Timestamp("2026-03-20")],
                        "excess_ret_1_vs_mkt": [-0.01, -0.02],
                        "excess_ret_2_vs_mkt": [-0.02, -0.03],
                        "excess_ret_3_vs_mkt": [-0.03, -0.04],
                        "excess_ret_5_vs_mkt": [-0.04, -0.05],
                        "excess_ret_10_vs_mkt": [-0.05, -0.07],
                        "excess_ret_20_vs_sector": [-0.08, -0.10],
                    }
                ),
            }
        ),
        steps=[
            SimpleNamespace(date=pd.Timestamp("2026-03-19"), composite_state=_base_state("2026-03-19")),
            SimpleNamespace(date=pd.Timestamp("2026-03-20"), composite_state=_base_state("2026-03-20")),
        ],
    )
    info_items = [
        InfoItem(
            date="2026-03-19",
            target_type="stock",
            target="AAA",
            horizon="both",
            direction="bullish",
            info_type="announcement",
            title="AAA signed a contract",
            source_subset="announcements",
            event_tag="contract_win",
            publish_datetime="2026-03-19T08:30:00",
        ),
        InfoItem(
            date="2026-03-20",
            target_type="stock",
            target="BBB",
            horizon="both",
            direction="bearish",
            info_type="news",
            title="BBB weak tape",
            source_subset="market_news",
            event_tag="guidance_negative",
            publish_datetime="",
        ),
    ]
    progress: list[tuple[str, str]] = []
    fit_calls = {"count": 0}
    deps = InfoResearchDependencies(
        emit_progress_fn=lambda stage, message: progress.append((stage, message)),
        load_or_build_v2_backtest_trajectory_fn=lambda **_: trajectory,
        split_research_trajectory_fn=lambda trajectory, split_mode="purged_wf", embargo_days=20: (
            SimpleNamespace(prepared=trajectory.prepared, steps=[]),
            SimpleNamespace(prepared=trajectory.prepared, steps=[trajectory.steps[0]]),
            SimpleNamespace(prepared=trajectory.prepared, steps=[trajectory.steps[1]]),
        ),
        trajectory_step_count_fn=lambda item: 0 if item is None else len(getattr(item, "steps", []) or []),
        load_v2_runtime_settings_fn=lambda **_: {},
        resolve_v2_universe_settings_fn=lambda settings, cache_root: dict(settings),
        load_v2_info_items_for_date_fn=lambda **_: ("input/info.csv", info_items),
        fit_v2_info_shadow_models_fn=lambda **_: (
            fit_calls.__setitem__("count", fit_calls["count"] + 1) or
            {"1d": SimpleNamespace(mode="rule", samples=2), "5d": SimpleNamespace(mode="rule", samples=2), "20d": SimpleNamespace(mode="rule", samples=2)},
            {"1d": SimpleNamespace(mode="rule", samples=1), "5d": SimpleNamespace(mode="rule", samples=1), "20d": SimpleNamespace(mode="rule", samples=1)},
        ),
        enrich_state_with_info_fn=lambda state, **kwargs: state,
        build_shadow_scored_rows_for_step_fn=lambda state, stock_frames, date: (
            pd.DataFrame(
                {
                    "symbol": ["AAA", "BBB"],
                    "score": [0.90, 0.10],
                    "realized_ret_1d": [0.03, -0.02],
                    "realized_ret_2d": [0.04, -0.03],
                    "realized_ret_3d": [0.05, -0.04],
                    "realized_ret_5d": [0.07, -0.05],
                    "realized_ret_10d": [0.10, -0.07],
                    "realized_ret_20d": [0.14, -0.10],
                }
            ),
            True,
        ),
        stock_policy_score_fn=lambda stock: float(stock.alpha_score),
        panel_slice_metrics_fn=panel_slice_metrics,
        filter_info_items_by_source_subset_fn=lambda items, subset: [item for item in items if item.source_subset == subset],
        event_tag_counts_fn=lambda items: {item.event_tag: sum(1 for candidate in items if candidate.event_tag == item.event_tag) for item in items if item.event_tag},
        info_source_breakdown_fn=lambda items: {
            "market_news": sum(1 for item in items if item.source_subset == "market_news"),
            "announcements": sum(1 for item in items if item.source_subset == "announcements"),
            "research": sum(1 for item in items if item.source_subset == "research"),
        },
    )

    result = run_v2_info_research(
        dependencies=deps,
        strategy_id="alpha_v2",
        cache_root=str(tmp_path / "cache"),
        horizons=("1d", "2d", "3d", "5d", "10d", "20d"),
        min_tag_count=1,
        max_tag_count=5,
    )

    assert result.strategy_id == "alpha_v2"
    assert result.info_manifest["info_item_count"] == 2
    assert result.timestamp_variants["timestamped_only"]["item_count"] == 1
    assert result.source_variants["all_info"]["horizon_metrics"]["20d"]["shadow_rank_ic"] > 0.0
    assert result.tag_variants[0]["event_tag"] in {"contract_win", "guidance_negative"}
    assert fit_calls["count"] == 3
    assert any(stage == "info-research" and "完成" in message for stage, message in progress)
    assert any(stage == "cache" and "variant 缓存" in message for stage, message in progress)

    report_path = write_v2_info_research_report(tmp_path / "info_research.md", result)
    dashboard_path = write_v2_info_research_dashboard(tmp_path / "info_research.html", result)

    assert report_path.read_text(encoding="utf-8").find("Source Variants") >= 0
    assert dashboard_path.read_text(encoding="utf-8").find("V2 Info Research") >= 0
