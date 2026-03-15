from __future__ import annotations

import ast
from pathlib import Path

from src.application.v2_contracts import LearnedPolicyModel, PolicySpec, V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult
from src.artifact_registry.v2_registry import publish_v2_research_artifacts
from src.reporting.view_models import build_daily_report_view_model, build_research_report_view_model
from src.review_analytics.summaries import summarize_daily_run


def _make_backtest(total_return: float = 0.20, annual_return: float = 0.18) -> V2BacktestSummary:
    return V2BacktestSummary(
        start_date="2024-01-01",
        end_date="2024-12-31",
        n_days=200,
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=-0.10,
        avg_turnover=0.22,
        total_cost=0.02,
    )


def test_artifact_registry_does_not_directly_import_legacy_services() -> None:
    module_path = Path("src/artifact_registry/v2_registry.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "src.application" not in imported_modules


def test_daily_report_view_model_uses_summary_shape() -> None:
    from src.application.v2_contracts import (
        CompositeState,
        CrossSectionForecastState,
        DailyRunResult,
        MarketForecastState,
        PolicyDecision,
    )
    from src.application.v2_services import build_strategy_snapshot
    import tempfile

    baseline = _make_backtest()
    learning_result = V2PolicyLearningResult(
        model=LearnedPolicyModel(
            feature_names=["x1"],
            exposure_intercept=0.55,
            exposure_coef=[0.1],
            position_intercept=2.5,
            position_coef=[0.05],
            turnover_intercept=0.20,
            turnover_coef=[0.01],
            train_rows=88,
            train_r2_exposure=0.33,
            train_r2_positions=0.25,
            train_r2_turnover=0.19,
        ),
        baseline=baseline,
        learned=_make_backtest(0.24, 0.22),
    )
    calibration = V2CalibrationResult(
        best_policy=PolicySpec(),
        best_score=0.12,
        baseline=baseline,
        calibrated=_make_backtest(0.22, 0.20),
        trials=[],
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = publish_v2_research_artifacts(
            strategy_id="swing_v2",
            artifact_root=tmp_dir,
            publish_forecast_models=False,
            settings={
                "config_path": "config/api.json",
                "source": "local",
                "watchlist": "config/watchlist.json",
                "universe_file": "config/universe_smoke_5.json",
                "universe_limit": 5,
                "symbols": ["000001.SZ"],
                "symbol_count": 1,
                "start": "2024-01-01",
                "end": "2024-12-31",
            },
            baseline=baseline,
            calibration=calibration,
            learning=learning_result,
        )
        result = DailyRunResult(
            snapshot=build_strategy_snapshot(strategy_id="swing_v2", universe_id="demo_universe"),
            composite_state=CompositeState(
                market=MarketForecastState(
                    as_of_date="2026-03-01",
                    up_1d_prob=0.5,
                    up_5d_prob=0.5,
                    up_20d_prob=0.5,
                    trend_state="trend",
                    drawdown_risk=0.1,
                    volatility_regime="normal",
                    liquidity_stress=0.1,
                ),
                cross_section=CrossSectionForecastState(
                    as_of_date="2026-03-01",
                    large_vs_small_bias=0.0,
                    growth_vs_value_bias=0.0,
                    fund_flow_strength=0.0,
                    margin_risk_on_score=0.0,
                    breadth_strength=0.0,
                    leader_participation=0.0,
                    weak_stock_ratio=0.0,
                ),
                sectors=[],
                stocks=[],
                strategy_mode="trend_follow",
                risk_regime="risk_on",
            ),
            policy_decision=PolicyDecision(
                target_exposure=0.8,
                target_position_count=3,
                rebalance_now=True,
                rebalance_intensity=0.5,
                intraday_t_allowed=False,
                turnover_cap=0.2,
            ),
            trade_actions=[],
        )

        daily_vm = build_daily_report_view_model(result)
        research_vm = build_research_report_view_model(
            strategy_id="swing_v2",
            baseline=baseline,
            calibration=calibration,
            learning=learning_result,
            artifacts=paths,
        )

        assert daily_vm.strategy_id == summarize_daily_run(result)["strategy_id"]
        assert daily_vm.strategy_mode == "trend_follow"
        assert daily_vm.market_forecasts == []
        assert "candidate_selection" in daily_vm.dynamic_universe
        assert research_vm.strategy_id == "swing_v2"
        assert research_vm.release_gate_passed is (
            str(paths.get("release_gate_passed", "false")).strip().lower() == "true"
        )
        assert research_vm.learning_model["train_rows"] == 88


def test_presenters_keep_single_public_v2_entries() -> None:
    markdown_path = Path("src/interfaces/presenters/markdown_reports.py")
    html_path = Path("src/interfaces/presenters/html_dashboard.py")

    markdown_tree = ast.parse(markdown_path.read_text(encoding="utf-8"))
    html_tree = ast.parse(html_path.read_text(encoding="utf-8"))

    markdown_defs = [node.name for node in markdown_tree.body if isinstance(node, ast.FunctionDef)]
    html_defs = [node.name for node in html_tree.body if isinstance(node, ast.FunctionDef)]

    assert markdown_defs.count("write_v2_daily_report") == 1
    assert markdown_defs.count("write_v2_research_report") == 1
    assert markdown_defs.count("write_v2_daily_report_from_view_model") == 1
    assert markdown_defs.count("write_v2_research_report_from_view_model") == 1

    assert html_defs.count("write_v2_daily_dashboard") == 0
    assert html_defs.count("write_v2_research_dashboard") == 1
    assert html_defs.count("write_v2_daily_dashboard_from_view_model") == 0
    assert html_defs.count("write_v2_research_dashboard_from_view_model") == 1
