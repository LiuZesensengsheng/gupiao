from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.application import v2_policy_feature_runtime as policy_feature_runtime
from src.application.v2_artifact_runtime import (
    load_policy_model_from_path as load_policy_model_from_path_runtime,
)
from src.application.v2_backtest_runtime import (
    execute_v2_backtest_trajectory as execute_v2_backtest_trajectory_runtime,
    simulate_execution_day as simulate_execution_day_runtime,
)
from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    LearnedPolicyModel,
    MarketForecastState,
    PolicyInput,
    PolicySpec,
    SectorForecastState,
    StockForecastState,
)
from src.application.v2_daily_runtime import (
    DailyCacheKeyDependencies,
    daily_result_cache_key as daily_result_cache_key_runtime,
    daily_result_cache_path as daily_result_cache_path_runtime,
    file_mtime_token as file_mtime_token_runtime,
)
from src.application.v2_daily_snapshot_runtime import (
    build_strategy_snapshot as build_strategy_snapshot_runtime,
    resolve_manifest_path,
)
from src.application.v2_policy_runtime import (
    apply_policy as apply_policy_runtime,
    build_trade_actions as build_trade_actions_runtime,
    policy_spec_from_model as policy_spec_from_model_runtime,
)
from src.application.v2_runtime_settings import (
    load_v2_runtime_settings as load_v2_runtime_settings_runtime,
    resolve_v2_universe_settings as resolve_v2_universe_settings_runtime,
)
from src.contracts.artifacts import CURRENT_ARTIFACT_VERSION


def test_daily_snapshot_runtime_keeps_facade_contract() -> None:
    runtime_snapshot = build_strategy_snapshot_runtime(strategy_id="alpha_v2", universe_id="demo")

    from src.application.v2_services import build_strategy_snapshot

    facade_snapshot = build_strategy_snapshot(strategy_id="alpha_v2", universe_id="demo")

    assert facade_snapshot == runtime_snapshot
    assert (
        resolve_manifest_path(
            strategy_id="alpha_v2",
            artifact_root="artifacts/v2",
            run_id="20260314_010203",
            snapshot_path=None,
        )
        == Path("artifacts/v2") / "alpha_v2" / "20260314_010203" / "research_manifest.json"
    )


def test_runtime_settings_module_keeps_facade_contract(tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(
        json.dumps({"stocks": [{"symbol": "000001.SZ", "name": "PingAn", "sector": "閾惰"}]}),
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "common": {
                    "source": "local",
                    "watchlist": "config/watchlist.json",
                    "universe_limit": 5,
                    "use_info_fusion": False,
                },
                "daily": {
                    "start": "2024-01-01",
                    "end": "2024-12-31",
                },
            }
        ),
        encoding="utf-8",
    )

    from src.application.v2_services import _load_v2_runtime_settings, _resolve_v2_universe_settings

    runtime_settings = load_v2_runtime_settings_runtime(
        config_path=str(config_path),
        universe_file=str(universe_path),
        source="local",
    )
    facade_settings = _load_v2_runtime_settings(
        config_path=str(config_path),
        universe_file=str(universe_path),
        source="local",
    )
    assert facade_settings == runtime_settings

    runtime_resolved = resolve_v2_universe_settings_runtime(
        settings=runtime_settings,
        cache_root=str(tmp_path / "cache"),
    )
    facade_resolved = _resolve_v2_universe_settings(
        settings=facade_settings,
        cache_root=str(tmp_path / "cache"),
    )
    assert facade_resolved == runtime_resolved
    assert facade_resolved["symbols"] == ["000001.SZ"]


def test_daily_cache_runtime_keeps_facade_contract(tmp_path: Path) -> None:
    import src.application.v2_services as legacy

    config_path = tmp_path / "runtime.json"
    watchlist = tmp_path / "watchlist.json"
    universe_file = tmp_path / "universe.json"
    margin_market = tmp_path / "margin_market.csv"
    margin_stock = tmp_path / "margin_stock.csv"
    info_file = tmp_path / "info.json"
    manifest_path = tmp_path / "manifest.json"
    for path in (config_path, watchlist, universe_file, margin_market, margin_stock, info_file, manifest_path):
        path.write_text("{}", encoding="utf-8")

    settings = {
        "config_path": str(config_path),
        "source": "local",
        "watchlist": str(watchlist),
        "universe_file": str(universe_file),
        "universe_limit": 20,
        "universe_tier": "favorites_16",
        "source_universe_manifest_path": str(manifest_path),
        "start": "2024-01-01",
        "end": "2024-12-31",
        "min_train_days": 40,
        "step_days": 5,
        "l2": 0.2,
        "max_positions": 8,
        "use_margin_features": True,
        "margin_market_file": str(margin_market),
        "margin_stock_file": str(margin_stock),
        "use_us_index_context": True,
        "us_index_source": "akshare",
        "use_us_sector_etf_context": False,
        "use_cn_etf_context": False,
        "cn_etf_source": "akshare",
        "event_file": str(info_file),
        "info_hash": "abc",
        "use_info_fusion": True,
        "info_shadow_only": False,
        "info_source_mode": "layered",
        "info_types": ["news"],
        "info_subsets": ["market_news"],
        "announcement_event_tags": ["earnings"],
    }

    runtime_key = daily_result_cache_key_runtime(
        strategy_id="alpha_v2",
        settings=settings,
        artifact_root=str(tmp_path / "artifacts"),
        run_id="20260315_010203",
        snapshot_path="",
        allow_retrain=False,
        deps=DailyCacheKeyDependencies(
            resolve_manifest_path=legacy._resolve_manifest_path,
            resolve_info_file_from_settings=legacy._resolve_info_file_from_settings,
        ),
    )
    facade_key = legacy._daily_result_cache_key(
        strategy_id="alpha_v2",
        settings=settings,
        artifact_root=str(tmp_path / "artifacts"),
        run_id="20260315_010203",
        snapshot_path="",
        allow_retrain=False,
    )
    assert facade_key == runtime_key
    assert file_mtime_token_runtime(info_file) == legacy._file_mtime_token(info_file)
    assert legacy._daily_result_cache_path(cache_root=str(tmp_path), cache_key=facade_key) == daily_result_cache_path_runtime(
        cache_root=str(tmp_path),
        cache_key=runtime_key,
    )


def test_policy_artifact_runtime_keeps_facade_contract(tmp_path: Path) -> None:
    payload_path = tmp_path / "latest_policy_model.json"
    payload_path.write_text(
        json.dumps(
            {
                "artifact_type": "learned_policy_model",
                "artifact_version": CURRENT_ARTIFACT_VERSION,
                "feature_names": ["x1"],
                "exposure_intercept": 0.5,
                "exposure_coef": [0.1],
                "position_intercept": 2.0,
                "position_coef": [0.1],
                "turnover_intercept": 0.2,
                "turnover_coef": [0.05],
                "train_rows": 64,
                "train_r2_exposure": 0.2,
                "train_r2_positions": 0.18,
                "train_r2_turnover": 0.12,
            }
        ),
        encoding="utf-8",
    )

    runtime_model = load_policy_model_from_path_runtime(
        payload_path,
        load_json_dict=lambda path_like: json.loads(Path(path_like).read_text(encoding="utf-8")),
    )

    from src.application.v2_services import _load_policy_model_from_path
    from src.artifact_registry.v2_registry import load_published_v2_policy_model

    facade_model = _load_policy_model_from_path(payload_path)

    strategy_dir = tmp_path / "alpha_v2"
    strategy_dir.mkdir()
    (strategy_dir / "latest_policy_model.json").write_text(payload_path.read_text(encoding="utf-8"), encoding="utf-8")
    registry_model = load_published_v2_policy_model(strategy_id="alpha_v2", artifact_root=str(tmp_path))

    assert runtime_model == facade_model
    assert registry_model == facade_model


def test_policy_runtime_keeps_facade_contract() -> None:
    import src.application.v2_services as legacy

    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-14",
            up_1d_prob=0.57,
            up_5d_prob=0.59,
            up_20d_prob=0.61,
            trend_state="trend",
            drawdown_risk=0.22,
            volatility_regime="normal",
            liquidity_stress=0.18,
            up_2d_prob=0.58,
            up_3d_prob=0.59,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-14",
            large_vs_small_bias=0.03,
            growth_vs_value_bias=0.02,
            fund_flow_strength=0.08,
            margin_risk_on_score=0.12,
            breadth_strength=0.14,
            leader_participation=0.16,
            weak_stock_ratio=0.22,
        ),
        sectors=[
            SectorForecastState("鏈夎壊", 0.58, 0.62, 0.14, 0.28, 0.20),
            SectorForecastState("鍖栧伐", 0.55, 0.57, 0.08, 0.24, 0.18),
        ],
        stocks=[
            StockForecastState("000630.SZ", "鏈夎壊", 0.57, 0.60, 0.64, 0.58, 0.12, 0.88),
            StockForecastState("600438.SH", "鍖栧伐", 0.54, 0.56, 0.58, 0.54, 0.08, 0.82),
        ],
        strategy_mode="trend_following",
        risk_regime="risk_on",
    )
    policy_input = PolicyInput(
        composite_state=state,
        current_weights={"000630.SZ": 0.18},
        current_cash=0.82,
        total_equity=1.0,
        current_holding_days={"000630.SZ": 8},
    )
    feature_names = legacy._policy_feature_names()
    model = LearnedPolicyModel(
        feature_names=feature_names,
        exposure_intercept=0.62,
        exposure_coef=[0.0] * len(feature_names),
        position_intercept=3.0,
        position_coef=[0.0] * len(feature_names),
        turnover_intercept=0.24,
        turnover_coef=[0.0] * len(feature_names),
        train_rows=32,
        train_r2_exposure=0.12,
        train_r2_positions=0.08,
        train_r2_turnover=0.05,
    )

    runtime_decision = apply_policy_runtime(
        policy_input,
        deps=legacy._policy_runtime_dependencies(),
    )
    facade_decision = legacy.apply_policy(policy_input)
    assert facade_decision == runtime_decision

    runtime_actions = build_trade_actions_runtime(
        decision=runtime_decision,
        current_weights=policy_input.current_weights,
    )
    facade_actions = legacy.build_trade_actions(
        decision=facade_decision,
        current_weights=policy_input.current_weights,
    )
    assert facade_actions == runtime_actions

    runtime_spec = policy_spec_from_model_runtime(
        state=state,
        model=model,
        deps=legacy._policy_runtime_dependencies(),
    )
    facade_spec = legacy._policy_spec_from_model(state=state, model=model)
    assert isinstance(facade_spec, PolicySpec)
    assert facade_spec == runtime_spec


def test_backtest_execution_runtime_keeps_facade_contract() -> None:
    import pandas as pd
    import src.application.v2_services as legacy

    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-14",
            up_1d_prob=0.56,
            up_5d_prob=0.58,
            up_20d_prob=0.60,
            trend_state="trend",
            drawdown_risk=0.20,
            volatility_regime="normal",
            liquidity_stress=0.18,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-14",
            large_vs_small_bias=0.01,
            growth_vs_value_bias=0.02,
            fund_flow_strength=0.06,
            margin_risk_on_score=0.10,
            breadth_strength=0.12,
            leader_participation=0.15,
            weak_stock_ratio=0.22,
        ),
        sectors=[SectorForecastState("鏈夎壊", 0.57, 0.61, 0.12, 0.22, 0.18)],
        stocks=[StockForecastState("000630.SZ", "鏈夎壊", 0.57, 0.60, 0.63, 0.57, 0.10, 0.86)],
        strategy_mode="trend_following",
        risk_regime="risk_on",
    )
    decision = legacy.apply_policy(
        PolicyInput(
            composite_state=state,
            current_weights={"000630.SZ": 0.10},
            current_cash=0.90,
            total_equity=1.0,
            current_holding_days={"000630.SZ": 5},
        )
    )
    date = pd.Timestamp("2026-03-14")
    next_date = pd.Timestamp("2026-03-17")
    stock_frames = {
        "000630.SZ": pd.DataFrame(
            [
                {
                    "date": date,
                    "open": 10.0,
                    "close": 10.2,
                    "low": 9.95,
                    "high": 10.3,
                    "ret_1": 0.02,
                    "fwd_ret_1": 0.03,
                }
            ]
        )
    }
    runtime_sim = simulate_execution_day_runtime(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"000630.SZ": 0.10},
        current_cash=0.90,
        stock_states=state.stocks,
        stock_frames=stock_frames,
        total_commission_rate=0.00015,
        base_slippage_rate=0.0002,
        deps=legacy._backtest_execution_dependencies(),
    )
    facade_sim = legacy._simulate_execution_day(
        date=date,
        next_date=next_date,
        decision=decision,
        current_weights={"000630.SZ": 0.10},
        current_cash=0.90,
        stock_states=state.stocks,
        stock_frames=stock_frames,
        total_commission_rate=0.00015,
        base_slippage_rate=0.0002,
    )
    assert facade_sim == runtime_sim

    trajectory = SimpleNamespace(
        prepared=SimpleNamespace(
            stock_frames=stock_frames,
            market_valid=pd.DataFrame([{"date": date, "mkt_fwd_ret_1": 0.01}]),
            settings={"universe_tier": "favorites_16"},
        ),
        steps=[
            SimpleNamespace(
                date=date,
                next_date=next_date,
                composite_state=state,
                stock_states=state.stocks,
                horizon_metrics={
                    "20d": {
                        "rank_ic": 0.12,
                        "top_decile_return": 0.03,
                        "top_bottom_spread": 0.05,
                        "top_k_hit_rate": 0.60,
                    }
                },
            )
        ],
    )
    runtime_summary, runtime_rows = execute_v2_backtest_trajectory_runtime(
        trajectory,
        deps=legacy._backtest_execution_dependencies(),
        capture_learning_rows=True,
    )
    facade_summary, facade_rows = legacy._execute_v2_backtest_trajectory(
        trajectory,
        capture_learning_rows=True,
    )
    assert facade_summary == runtime_summary
    assert facade_rows == runtime_rows


def test_policy_feature_runtime_keeps_facade_contract() -> None:
    import src.application.v2_services as legacy

    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-14",
            up_1d_prob=0.57,
            up_5d_prob=0.59,
            up_20d_prob=0.61,
            trend_state="trend",
            drawdown_risk=0.22,
            volatility_regime="normal",
            liquidity_stress=0.18,
            up_2d_prob=0.58,
            up_3d_prob=0.59,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-14",
            large_vs_small_bias=0.03,
            growth_vs_value_bias=0.02,
            fund_flow_strength=0.08,
            margin_risk_on_score=0.12,
            breadth_strength=0.14,
            leader_participation=0.16,
            weak_stock_ratio=0.22,
        ),
        sectors=[
            SectorForecastState("閾捐矾A", 0.58, 0.62, 0.14, 0.28, 0.20),
            SectorForecastState("閾捐矾B", 0.55, 0.57, 0.08, 0.24, 0.18),
        ],
        stocks=[
            StockForecastState("000630.SZ", "閾捐矾A", 0.57, 0.60, 0.64, 0.58, 0.12, 0.88),
            StockForecastState("600438.SH", "閾捐矾B", 0.54, 0.56, 0.58, 0.54, 0.08, 0.82),
        ],
        strategy_mode="trend_following",
        risk_regime="risk_on",
    )

    feature_deps = legacy._policy_feature_runtime_dependencies()
    runtime_features = policy_feature_runtime.policy_feature_vector(state, deps=feature_deps)
    facade_features = legacy._policy_feature_vector(state)
    assert policy_feature_runtime.policy_feature_names() == legacy._policy_feature_names()
    assert runtime_features.tolist() == pytest.approx(facade_features.tolist())

    X = [[1.0, 0.5], [0.8, 0.3], [0.6, 0.1], [0.4, -0.1]]
    y = [0.7, 0.6, 0.45, 0.2]
    weights = [1.0, 0.8, 1.2, 1.0]
    runtime_intercept, runtime_coef = policy_feature_runtime.fit_ridge_regression(X, y, l2=0.5, sample_weight=weights)
    facade_intercept, facade_coef = legacy._fit_ridge_regression(X, y, l2=0.5, sample_weight=weights)
    assert runtime_intercept == pytest.approx(facade_intercept)
    assert runtime_coef.tolist() == pytest.approx(facade_coef.tolist())
    assert policy_feature_runtime.predict_ridge(X[0], runtime_intercept, runtime_coef) == pytest.approx(
        legacy._predict_ridge(X[0], facade_intercept, facade_coef)
    )
    assert policy_feature_runtime.normalize_coef_vector([1.0], 3).tolist() == pytest.approx(
        legacy._normalize_coef_vector([1.0], 3).tolist()
    )
    assert policy_feature_runtime.r2_score(y, y) == pytest.approx(legacy._r2_score(y, y))
