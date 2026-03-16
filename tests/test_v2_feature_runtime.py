from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.application.v2_contracts import (
    CompositeState,
    CrossSectionForecastState,
    MarketForecastState,
    SectorForecastState,
    StockForecastState,
)
from src.application.v2_feature_runtime import (
    BlendedBinaryModel,
    BlendedQuantileModel,
    ForecastRuntimeDependencies,
    _fit_quantile_model_family,
    _prepare_target_arrays,
    _resolve_training_window_days,
    _slice_prepared_target_arrays,
    _should_emit_block_progress,
    _training_row_window,
    make_forecast_backend,
    tensorize_temporal_frame,
)
from src.application.v2_forecast_model_runtime import ReturnQuantileProfile, fit_quantile_quintet
from src.infrastructure.modeling import QuantileLinearModel
from src.application.v2_learning_target_runtime import (
    LearningTargetDependencies,
    derive_learning_targets,
)


def _forecast_runtime_dependencies() -> ForecastRuntimeDependencies:
    return ForecastRuntimeDependencies(
        emit_progress=lambda *_: None,
        format_elapsed=lambda elapsed: f"{float(elapsed):.2f}s",
        build_date_slice_index=lambda frame, **_: (frame, {}),
        build_market_and_cross_section_from_prebuilt_frame=lambda **_: (None, None),
        build_stock_states_from_panel_slice=lambda **_: ([], pd.DataFrame()),
        build_sector_states=lambda *_args, **_kwargs: [],
        stock_policy_score=lambda *_: 0.0,
        compose_state=lambda **_: None,
        panel_horizon_metrics=lambda *_: {},
        fit_quantile_quintet=lambda *_args, **_kwargs: None,
        fit_mlp_quantile_quintet=lambda *_args, **_kwargs: None,
        logistic_model_cls=object,
        mlp_model_cls=object,
        quantile_model_cls=object,
        mlp_quantile_model_cls=object,
        trajectory_step_cls=SimpleNamespace,
        backtest_trajectory_cls=SimpleNamespace,
    )


def _safe_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if parsed == parsed else float(default)
    except Exception:
        return float(default)


def test_tensorize_temporal_frame_builds_grouped_lags() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB", "BBB"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "f1": [1.0, 2.0, 10.0, 20.0],
            "f2": [3.0, 4.0, 30.0, 40.0],
        }
    )

    tensor_frame, lag_cols = tensorize_temporal_frame(
        frame,
        feature_cols=["f1", "f2"],
        group_col="symbol",
        lag_depth=2,
    )

    assert lag_cols == ["f1__lag0", "f2__lag0", "f1__lag1", "f2__lag1"]
    aaa_lag = tensor_frame.loc[tensor_frame["symbol"] == "AAA", "f1__lag1"].tolist()
    bbb_lag = tensor_frame.loc[tensor_frame["symbol"] == "BBB", "f2__lag1"].tolist()
    assert np.isnan(aaa_lag[0])
    assert aaa_lag[1] == 1.0
    assert np.isnan(bbb_lag[0])
    assert bbb_lag[1] == 30.0


def test_make_forecast_backend_selects_supported_backend() -> None:
    deps = _forecast_runtime_dependencies()

    linear_backend = make_forecast_backend("linear", deps=deps)
    deep_backend = make_forecast_backend("deep", deps=deps)
    hybrid_backend = make_forecast_backend("hybrid", deps=deps)

    assert linear_backend.name == "linear"
    assert deep_backend.name == "deep"
    assert hybrid_backend.name == "hybrid"

    with pytest.raises(ValueError, match="Unsupported forecast backend"):
        make_forecast_backend("unknown", deps=deps)


def test_blended_binary_model_averages_different_feature_views() -> None:
    class _StubBinaryModel:
        def __init__(self, value: float) -> None:
            self.value = float(value)
            self.calls: list[list[str]] = []

        def predict_proba(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
            self.calls.append([] if feature_cols is None else list(feature_cols))
            return np.full(len(df), self.value, dtype=float)

    primary = _StubBinaryModel(0.60)
    secondary = _StubBinaryModel(0.40)
    model = BlendedBinaryModel(
        primary_model=primary,
        primary_feature_cols=["f_raw"],
        secondary_model=secondary,
        secondary_feature_cols=["f_tensor"],
        primary_weight=0.65,
        secondary_weight=0.35,
    )

    got = model.predict_proba(pd.DataFrame({"f_raw": [1.0, 2.0], "f_tensor": [3.0, 4.0]}))

    assert got == pytest.approx(np.array([0.53, 0.53]))
    assert primary.calls == [["f_raw"]]
    assert secondary.calls == [["f_tensor"]]


def test_blended_quantile_model_averages_different_feature_views() -> None:
    class _StubQuantileModel:
        def __init__(self, value: float) -> None:
            self.value = float(value)
            self.calls: list[list[str]] = []

        def predict(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
            self.calls.append([] if feature_cols is None else list(feature_cols))
            return np.full(len(df), self.value, dtype=float)

    primary = _StubQuantileModel(0.03)
    secondary = _StubQuantileModel(0.01)
    model = BlendedQuantileModel(
        primary_model=primary,
        primary_feature_cols=["f_raw"],
        secondary_model=secondary,
        secondary_feature_cols=["f_tensor"],
        primary_weight=0.65,
        secondary_weight=0.35,
    )

    got = model.predict(pd.DataFrame({"f_raw": [1.0], "f_tensor": [2.0]}))

    assert got == pytest.approx(np.array([0.023]))
    assert primary.calls == [["f_raw"]]
    assert secondary.calls == [["f_tensor"]]


def test_should_emit_block_progress_throttles_large_runs() -> None:
    emitted = [idx for idx in range(1, 13) if _should_emit_block_progress(block_idx=idx, total_blocks=12)]

    assert emitted == [1, 5, 10, 12]


def test_should_emit_block_progress_keeps_small_runs_verbose() -> None:
    emitted = [idx for idx in range(1, 6) if _should_emit_block_progress(block_idx=idx, total_blocks=5)]

    assert emitted == [1, 2, 3, 4, 5]


def test_slice_prepared_target_arrays_matches_prefix_filtering() -> None:
    frame = pd.DataFrame(
        {
            "f1": [1.0, np.nan, 3.0, 4.0],
            "f2": [10.0, 20.0, 30.0, 40.0],
            "target_a": [1.0, 0.0, np.nan, 1.0],
            "target_b": [0.5, np.nan, 0.7, 0.8],
        }
    )

    prepared = _prepare_target_arrays(
        frame,
        feature_cols=["f1", "f2"],
        target_cols=["target_a", "target_b"],
    )

    x_a, y_a = _slice_prepared_target_arrays(prepared, end=3, target_col="target_a")
    x_b, y_b = _slice_prepared_target_arrays(prepared, end=3, target_col="target_b")

    expected_a = frame.iloc[:3].dropna(subset=["f1", "f2", "target_a"])
    expected_b = frame.iloc[:3].dropna(subset=["f1", "f2", "target_b"])

    assert np.array_equal(x_a, expected_a[["f1", "f2"]].astype(float).to_numpy())
    assert np.array_equal(y_a, expected_a["target_a"].astype(float).to_numpy())
    assert np.array_equal(x_b, expected_b[["f1", "f2"]].astype(float).to_numpy())
    assert np.array_equal(y_b, expected_b["target_b"].astype(float).to_numpy())


def test_fit_quantile_model_family_matches_dataframe_training_path() -> None:
    frame = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.5, 4.2, 5.1, 6.3, 7.4, 8.0],
            "f2": [0.5, 1.2, 1.8, 2.6, 3.1, 3.7, 4.4, 4.9],
            "target": [0.02, 0.01, 0.03, 0.025, 0.04, 0.038, 0.05, 0.048],
        }
    )

    prepared = _prepare_target_arrays(
        frame,
        feature_cols=["f1", "f2"],
        target_cols=["target"],
    )
    x, y = _slice_prepared_target_arrays(prepared, end=len(frame), target_col="target")

    fitted_from_arrays = _fit_quantile_model_family(
        model_cls=QuantileLinearModel,
        l2=0.8,
        feature_cols=["f1", "f2"],
        x=x,
        y=y,
    )
    fitted_from_frame = fit_quantile_quintet(
        frame,
        feature_cols=["f1", "f2"],
        target_col="target",
        l2=0.8,
    )

    probe = frame.iloc[[1, 5, 7]]
    for prepared_model, frame_model in zip(fitted_from_arrays, fitted_from_frame):
        assert prepared_model.predict(probe, ["f1", "f2"]) == pytest.approx(
            frame_model.predict(probe, ["f1", "f2"])
        )


def test_resolve_training_window_days_enforces_minimum_train_window() -> None:
    settings = {"training_window_days": 120}

    assert _resolve_training_window_days(settings=settings, min_train_days=240) == 240
    assert _resolve_training_window_days(settings={"training_window_days": 480}, min_train_days=240) == 480
    assert _resolve_training_window_days(settings={"training_window_days": 0}, min_train_days=240) is None


def test_training_row_window_uses_expanding_or_rolling_bounds() -> None:
    dates = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx) for idx in range(6)]
    bounds = {
        dates[0]: (0, 2),
        dates[1]: (2, 4),
        dates[2]: (4, 6),
        dates[3]: (6, 8),
        dates[4]: (8, 10),
        dates[5]: (10, 12),
    }

    assert _training_row_window(bounds=bounds, dates=dates, block_start=4, training_window_days=None) == (0, 8)
    assert _training_row_window(bounds=bounds, dates=dates, block_start=4, training_window_days=3) == (2, 8)


def test_derive_learning_targets_uses_generated_universe_path() -> None:
    date = pd.Timestamp("2026-03-14")
    state = CompositeState(
        market=MarketForecastState(
            as_of_date="2026-03-14",
            up_1d_prob=0.58,
            up_5d_prob=0.60,
            up_20d_prob=0.62,
            trend_state="trend",
            drawdown_risk=0.18,
            volatility_regime="normal",
            liquidity_stress=0.12,
        ),
        cross_section=CrossSectionForecastState(
            as_of_date="2026-03-14",
            large_vs_small_bias=0.02,
            growth_vs_value_bias=0.01,
            fund_flow_strength=0.04,
            margin_risk_on_score=0.06,
            breadth_strength=0.12,
            leader_participation=0.14,
            weak_stock_ratio=0.24,
        ),
        sectors=[SectorForecastState("科技", 0.58, 0.61, 0.08, 0.22, 0.15)],
        stocks=[
            StockForecastState("AAA", "科技", 0.62, 0.66, 0.70, 0.60, 0.12, 0.90, alpha_score=0.74),
            StockForecastState("BBB", "科技", 0.58, 0.61, 0.64, 0.56, 0.08, 0.82, alpha_score=0.68),
            StockForecastState("CCC", "医药", 0.54, 0.58, 0.60, 0.53, 0.06, 0.78, alpha_score=0.62),
            StockForecastState("DDD", "消费", 0.52, 0.55, 0.57, 0.51, 0.05, 0.75, alpha_score=0.59),
        ],
        strategy_mode="trend_follow",
        risk_regime="risk_on",
    )
    stock_frames = {
        "AAA": pd.DataFrame([{"date": date, "fwd_ret_1": 0.018, "excess_ret_1_vs_mkt": 0.012, "excess_ret_5_vs_mkt": 0.022, "excess_ret_20_vs_sector": 0.040}]),
        "BBB": pd.DataFrame([{"date": date, "fwd_ret_1": 0.014, "excess_ret_1_vs_mkt": 0.010, "excess_ret_5_vs_mkt": 0.018, "excess_ret_20_vs_sector": 0.032}]),
        "CCC": pd.DataFrame([{"date": date, "fwd_ret_1": 0.009, "excess_ret_1_vs_mkt": 0.004, "excess_ret_5_vs_mkt": 0.009, "excess_ret_20_vs_sector": 0.016}]),
        "DDD": pd.DataFrame([{"date": date, "fwd_ret_1": 0.006, "excess_ret_1_vs_mkt": 0.002, "excess_ret_5_vs_mkt": 0.004, "excess_ret_20_vs_sector": 0.010}]),
    }
    horizon_metrics = {"20d": {"rank_ic": 0.14, "top_bottom_spread": 0.08, "top_k_hit_rate": 0.68}}
    deps = LearningTargetDependencies(
        stock_policy_score=lambda stock: float(getattr(stock, "alpha_score", 0.0)),
        safe_float=_safe_float,
        alpha_opportunity_metrics=lambda stocks: {"alpha_headroom": 0.05, "breadth_ratio": 0.12, "top_score": 0.74},
        signal_unit=lambda value, scale: float(np.clip(float(value) / max(float(scale), 1e-9), -1.0, 1.0)),
        normalize_universe_tier=lambda tier: str(tier or "").strip().lower(),
        clip=lambda value, lo, hi: float(np.clip(float(value), float(lo), float(hi))),
    )

    exposure, positions, turnover, sample_weight = derive_learning_targets(
        state=state,
        stock_frames=stock_frames,
        date=date,
        horizon_metrics=horizon_metrics,
        universe_tier="generated_80",
        deps=deps,
    )

    assert exposure == pytest.approx(0.722236875)
    assert positions == pytest.approx(5.0)
    assert turnover == pytest.approx(0.3022896875)
    assert sample_weight == pytest.approx(3.0168375)
