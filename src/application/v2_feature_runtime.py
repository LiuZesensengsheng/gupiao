from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
import pandas as pd

from src.application.v2_forecast_model_runtime import QUANTILE_LEVELS


@dataclass(frozen=True)
class ForecastRuntimeDependencies:
    emit_progress: Callable[[str, str], None]
    format_elapsed: Callable[[float], str]
    build_date_slice_index: Callable[..., tuple[pd.DataFrame, dict[pd.Timestamp, tuple[int, int]]]]
    build_market_and_cross_section_from_prebuilt_frame: Callable[..., tuple[object, object]]
    build_stock_states_from_panel_slice: Callable[..., tuple[list[object], pd.DataFrame]]
    build_sector_states: Callable[..., list[object]]
    stock_policy_score: Callable[[object], float]
    compose_state: Callable[..., object]
    panel_horizon_metrics: Callable[[pd.DataFrame], dict[str, dict[str, float]]]
    fit_quantile_quintet: Callable[..., object]
    fit_mlp_quantile_quintet: Callable[..., object]
    logistic_model_cls: Callable[..., Any]
    mlp_model_cls: Callable[..., Any]
    quantile_model_cls: Callable[..., Any]
    mlp_quantile_model_cls: Callable[..., Any]
    trajectory_step_cls: type
    backtest_trajectory_cls: type


class ForecastBackend(Protocol):
    name: str

    def build_trajectory(
        self,
        prepared: object,
        *,
        retrain_days: int = 20,
    ) -> object:
        ...        


MARKET_BINARY_TARGETS = [
    "mkt_target_1d_up",
    "mkt_target_2d_up",
    "mkt_target_3d_up",
    "mkt_target_5d_up",
    "mkt_target_20d_up",
]

PANEL_BINARY_TARGETS = [
    "target_1d_excess_mkt_up",
    "target_2d_excess_mkt_up",
    "target_3d_excess_mkt_up",
    "target_5d_excess_mkt_up",
    "target_20d_excess_sector_up",
]

PANEL_QUANTILE_TARGETS = [
    "excess_ret_1_vs_mkt",
    "excess_ret_20_vs_sector",
]


@dataclass(frozen=True)
class _PreparedTargetArrays:
    feature_cols: tuple[str, ...]
    x_all: np.ndarray
    feature_valid: np.ndarray
    target_values: dict[str, np.ndarray]


def _prepare_target_arrays(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_cols: list[str],
) -> _PreparedTargetArrays:
    if df.empty:
        return _PreparedTargetArrays(
            feature_cols=tuple(feature_cols),
            x_all=np.empty((0, len(feature_cols)), dtype=float),
            feature_valid=np.empty(0, dtype=bool),
            target_values={target: np.empty(0, dtype=float) for target in target_cols},
        )

    work = df[feature_cols + target_cols].astype(float)
    feature_frame = work[feature_cols]
    return _PreparedTargetArrays(
        feature_cols=tuple(feature_cols),
        x_all=feature_frame.to_numpy(copy=True),
        feature_valid=(~feature_frame.isna().any(axis=1)).to_numpy(),
        target_values={
            target_col: work[target_col].to_numpy(copy=True)
            for target_col in target_cols
        },
    )


def _slice_prepared_target_arrays(
    prepared: _PreparedTargetArrays,
    *,
    start: int = 0,
    end: int,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    hi = max(0, min(int(end), prepared.x_all.shape[0]))
    lo = max(0, min(int(start), hi))
    if hi <= lo:
        return np.empty((0, len(prepared.feature_cols)), dtype=float), np.empty(0, dtype=float)

    y_slice = prepared.target_values[target_col][lo:hi]
    valid_mask = prepared.feature_valid[lo:hi] & ~np.isnan(y_slice)
    if not valid_mask.any():
        return np.empty((0, len(prepared.feature_cols)), dtype=float), np.empty(0, dtype=float)
    x_slice = prepared.x_all[lo:hi]
    return x_slice[valid_mask], y_slice[valid_mask].astype(float, copy=False)


def _slice_prepared_target_family(
    prepared: _PreparedTargetArrays,
    *,
    start: int = 0,
    end: int,
    target_cols: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        target_col: _slice_prepared_target_arrays(
            prepared,
            start=start,
            end=end,
            target_col=target_col,
        )
        for target_col in target_cols
    }


def _fit_quantile_model_family(
    *,
    model_cls: Callable[..., Any],
    l2: float,
    feature_cols: list[str],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[Any, Any, Any, Any, Any]:
    if len(y) == 0 or np.asarray(x).size == 0:
        raise ValueError("No rows available for quantile training after dropping NaN.")
    return tuple(
        model_cls(quantile=quantile, l2=float(l2)).fit_prepared(
            x=x,
            y=y,
            feature_cols=feature_cols,
        )
        for quantile in QUANTILE_LEVELS
    )


def _prepare_binary_training_family(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_cols: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    prepared = _prepare_target_arrays(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
    return _slice_prepared_target_family(
        prepared,
        end=prepared.x_all.shape[0],
        target_cols=target_cols,
    )


def _fit_binary_model_family(
    *,
    model_cls: Callable[..., Any],
    l2: float,
    feature_cols: list[str],
    prepared_targets: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for target_col, (x, y) in prepared_targets.items():
        model = model_cls(l2=float(l2))
        fit_prepared = getattr(model, "fit_prepared", None)
        if callable(fit_prepared):
            fit_prepared(x=x, y=y, feature_cols=feature_cols)
        else:
            raise AttributeError(f"{type(model).__name__} is missing fit_prepared")
        models[target_col] = model
    return models


def _history_slice(frame: pd.DataFrame, end: int) -> pd.DataFrame:
    return frame.iloc[:end]


def _should_emit_block_progress(*, block_idx: int, total_blocks: int) -> bool:
    if total_blocks <= 10:
        return True
    if block_idx in {1, total_blocks}:
        return True
    return block_idx % 5 == 0


def _resolve_training_window_days(
    *,
    settings: object,
    min_train_days: int,
) -> int | None:
    if not isinstance(settings, dict):
        return None
    raw_window = settings.get("training_window_days")
    if raw_window is None:
        return None
    try:
        parsed = int(raw_window)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return max(int(min_train_days), parsed)


def _training_row_window(
    *,
    bounds: dict[pd.Timestamp, tuple[int, int]],
    dates: list[pd.Timestamp],
    block_start: int,
    training_window_days: int | None,
) -> tuple[int, int]:
    end_date = dates[block_start - 1]
    end_row = bounds.get(end_date, (0, 0))[1]
    if training_window_days is None:
        return 0, end_row
    window_start_idx = max(0, int(block_start) - int(training_window_days))
    start_date = dates[window_start_idx]
    start_row = bounds.get(start_date, (0, 0))[0]
    return start_row, end_row


def tensorize_temporal_frame(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    group_col: str | None,
    lag_depth: int = 3,
) -> tuple[pd.DataFrame, list[str]]:
    if frame.empty or not feature_cols:
        return frame.copy(), []
    lag_depth = max(1, int(lag_depth))
    if group_col is None:
        work = frame.sort_values("date").copy()
        grouped = None
    else:
        work = frame.sort_values([group_col, "date"]).copy()
        grouped = work.groupby(group_col, observed=True, sort=False)

    out_cols: list[str] = []
    lag_frames: list[pd.DataFrame] = []
    for lag in range(lag_depth):
        if grouped is None:
            shifted = work[feature_cols].shift(lag)
        else:
            shifted = grouped[feature_cols].shift(lag)
        new_cols = [f"{col}__lag{lag}" for col in feature_cols]
        shifted = shifted.copy()
        shifted.columns = new_cols
        lag_frames.append(shifted)
        out_cols.extend(new_cols)
    if lag_frames:
        work = pd.concat([work] + lag_frames, axis=1)
    return work, out_cols


HYBRID_LINEAR_WEIGHT = 0.65
HYBRID_DEEP_WEIGHT = 0.35


class BlendedBinaryModel:
    def __init__(
        self,
        *,
        primary_model: Any,
        primary_feature_cols: list[str],
        secondary_model: Any,
        secondary_feature_cols: list[str],
        primary_weight: float = HYBRID_LINEAR_WEIGHT,
        secondary_weight: float = HYBRID_DEEP_WEIGHT,
    ) -> None:
        total_weight = max(float(primary_weight) + float(secondary_weight), 1e-9)
        self.primary_model = primary_model
        self.primary_feature_cols = list(primary_feature_cols)
        self.secondary_model = secondary_model
        self.secondary_feature_cols = list(secondary_feature_cols)
        self.primary_weight = float(primary_weight) / total_weight
        self.secondary_weight = float(secondary_weight) / total_weight

    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
        primary = np.asarray(
            self.primary_model.predict_proba(df, self.primary_feature_cols),
            dtype=float,
        )
        secondary = np.asarray(
            self.secondary_model.predict_proba(df, self.secondary_feature_cols),
            dtype=float,
        )
        blended = self.primary_weight * primary + self.secondary_weight * secondary
        return np.clip(blended, 1e-6, 1.0 - 1e-6)


class BlendedQuantileModel:
    def __init__(
        self,
        *,
        primary_model: Any,
        primary_feature_cols: list[str],
        secondary_model: Any,
        secondary_feature_cols: list[str],
        primary_weight: float = HYBRID_LINEAR_WEIGHT,
        secondary_weight: float = HYBRID_DEEP_WEIGHT,
    ) -> None:
        total_weight = max(float(primary_weight) + float(secondary_weight), 1e-9)
        self.primary_model = primary_model
        self.primary_feature_cols = list(primary_feature_cols)
        self.secondary_model = secondary_model
        self.secondary_feature_cols = list(secondary_feature_cols)
        self.primary_weight = float(primary_weight) / total_weight
        self.secondary_weight = float(secondary_weight) / total_weight

    def predict(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
        primary = np.asarray(
            self.primary_model.predict(df, self.primary_feature_cols),
            dtype=float,
        )
        secondary = np.asarray(
            self.secondary_model.predict(df, self.secondary_feature_cols),
            dtype=float,
        )
        return self.primary_weight * primary + self.secondary_weight * secondary


def _blend_binary_model_family(
    *,
    primary_models: dict[str, Any],
    primary_feature_cols: list[str],
    secondary_models: dict[str, Any],
    secondary_feature_cols: list[str],
    primary_weight: float = HYBRID_LINEAR_WEIGHT,
    secondary_weight: float = HYBRID_DEEP_WEIGHT,
) -> dict[str, BlendedBinaryModel]:
    return {
        target_col: BlendedBinaryModel(
            primary_model=primary_models[target_col],
            primary_feature_cols=primary_feature_cols,
            secondary_model=secondary_models[target_col],
            secondary_feature_cols=secondary_feature_cols,
            primary_weight=primary_weight,
            secondary_weight=secondary_weight,
        )
        for target_col in primary_models
        if target_col in secondary_models
    }


def _blend_quantile_model_family(
    *,
    primary_models: tuple[Any, Any, Any, Any, Any],
    primary_feature_cols: list[str],
    secondary_models: tuple[Any, Any, Any, Any, Any],
    secondary_feature_cols: list[str],
    primary_weight: float = HYBRID_LINEAR_WEIGHT,
    secondary_weight: float = HYBRID_DEEP_WEIGHT,
) -> tuple[BlendedQuantileModel, BlendedQuantileModel, BlendedQuantileModel, BlendedQuantileModel, BlendedQuantileModel]:
    return tuple(
        BlendedQuantileModel(
            primary_model=primary_model,
            primary_feature_cols=primary_feature_cols,
            secondary_model=secondary_model,
            secondary_feature_cols=secondary_feature_cols,
            primary_weight=primary_weight,
            secondary_weight=secondary_weight,
        )
        for primary_model, secondary_model in zip(primary_models, secondary_models)
    )


class LinearForecastBackend:
    name = "linear"

    def __init__(self, deps: ForecastRuntimeDependencies) -> None:
        self._deps = deps

    def build_trajectory(
        self,
        prepared: object,
        *,
        retrain_days: int = 20,
    ) -> object:
        settings = getattr(prepared, "settings")
        market_valid = getattr(prepared, "market_valid")
        panel = getattr(prepared, "panel")
        market_feature_cols = getattr(prepared, "market_feature_cols")
        feature_cols = getattr(prepared, "feature_cols")
        dates = getattr(prepared, "dates")
        min_train_days = int(settings["min_train_days"])
        training_window_days = _resolve_training_window_days(settings=settings, min_train_days=min_train_days)
        steps: list[object] = []
        market_sorted, market_bounds = self._deps.build_date_slice_index(
            market_valid,
            sort_cols=["date"],
        )
        panel_sorted, panel_bounds = self._deps.build_date_slice_index(
            panel,
            sort_cols=["date", "symbol"],
        )
        market_training_arrays = _prepare_target_arrays(
            market_sorted,
            feature_cols=market_feature_cols,
            target_cols=MARKET_BINARY_TARGETS,
        )
        panel_training_arrays = _prepare_target_arrays(
            panel_sorted,
            feature_cols=feature_cols,
            target_cols=PANEL_BINARY_TARGETS + PANEL_QUANTILE_TARGETS,
        )
        block_starts = list(range(min_train_days, len(dates) - 1, max(1, int(retrain_days))))
        self._deps.emit_progress(
            "trajectory",
            f"backend={self.name} building trajectory: blocks={len(block_starts)}, dates={len(dates)}, universe={len(getattr(prepared, 'stock_frames'))}, training_window_days={training_window_days or 'expanding'}",
        )
        trajectory_started = time.perf_counter()

        for block_idx, block_start in enumerate(block_starts, start=1):
            elapsed = time.perf_counter() - trajectory_started
            completed = max(0, block_idx - 1)
            eta = 0.0 if completed <= 0 else (elapsed / completed) * (len(block_starts) - completed)
            if _should_emit_block_progress(block_idx=block_idx, total_blocks=len(block_starts)):
                self._deps.emit_progress(
                    "trajectory",
                    f"backend={self.name} training block {block_idx}/{len(block_starts)}: cutoff {pd.Timestamp(dates[block_start - 1]).date()} | elapsed={self._deps.format_elapsed(elapsed)} | eta={self._deps.format_elapsed(eta)}",
                )
            market_train_start, market_train_end = _training_row_window(
                bounds=market_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            if market_train_end <= market_train_start:
                continue
            market_models = _fit_binary_model_family(
                model_cls=self._deps.logistic_model_cls,
                l2=float(settings["l2"]),
                feature_cols=market_feature_cols,
                prepared_targets=_slice_prepared_target_family(
                    market_training_arrays,
                    start=market_train_start,
                    end=market_train_end,
                    target_cols=MARKET_BINARY_TARGETS,
                ),
            )
            panel_train_start, panel_train_end = _training_row_window(
                bounds=panel_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            if panel_train_end <= panel_train_start:
                continue
            panel_models = _fit_binary_model_family(
                model_cls=self._deps.logistic_model_cls,
                l2=float(settings["l2"]),
                feature_cols=feature_cols,
                prepared_targets=_slice_prepared_target_family(
                    panel_training_arrays,
                    start=panel_train_start,
                    end=panel_train_end,
                    target_cols=PANEL_BINARY_TARGETS,
                ),
            )
            short_x, short_y = _slice_prepared_target_arrays(
                panel_training_arrays,
                start=panel_train_start,
                end=panel_train_end,
                target_col="excess_ret_1_vs_mkt",
            )
            panel_short_q_models = _fit_quantile_model_family(
                model_cls=self._deps.quantile_model_cls,
                feature_cols=feature_cols,
                l2=float(settings["l2"]),
                x=short_x,
                y=short_y,
            )
            mid_x, mid_y = _slice_prepared_target_arrays(
                panel_training_arrays,
                start=panel_train_start,
                end=panel_train_end,
                target_col="excess_ret_20_vs_sector",
            )
            panel_mid_q_models = _fit_quantile_model_family(
                model_cls=self._deps.quantile_model_cls,
                feature_cols=feature_cols,
                l2=float(settings["l2"]),
                x=mid_x,
                y=mid_y,
            )

            block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
            for idx in range(block_start, block_end):
                date = dates[idx]
                next_date = dates[idx + 1]
                market_start, market_end = market_bounds.get(date, (0, 0))
                market_row = market_sorted.iloc[market_start:market_end]
                if market_row.empty:
                    continue
                mkt_short = float(market_models["mkt_target_1d_up"].predict_proba(market_row, market_feature_cols)[0])
                mkt_two = float(market_models["mkt_target_2d_up"].predict_proba(market_row, market_feature_cols)[0])
                mkt_three = float(market_models["mkt_target_3d_up"].predict_proba(market_row, market_feature_cols)[0])
                mkt_five = float(market_models["mkt_target_5d_up"].predict_proba(market_row, market_feature_cols)[0])
                mkt_mid = float(market_models["mkt_target_20d_up"].predict_proba(market_row, market_feature_cols)[0])
                market_state, cross_section = self._deps.build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=_history_slice(market_sorted, market_end),
                    market_short_prob=mkt_short,
                    market_two_prob=mkt_two,
                    market_three_prob=mkt_three,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_start, panel_end = panel_bounds.get(date, (0, 0))
                panel_row = panel_sorted.iloc[panel_start:panel_end]
                stock_states, scored_rows = self._deps.build_stock_states_from_panel_slice(
                    panel_row=panel_row,
                    feature_cols=feature_cols,
                    short_model=panel_models["target_1d_excess_mkt_up"],
                    two_model=panel_models["target_2d_excess_mkt_up"],
                    three_model=panel_models["target_3d_excess_mkt_up"],
                    five_model=panel_models["target_5d_excess_mkt_up"],
                    mid_model=panel_models["target_20d_excess_sector_up"],
                    short_q_models=panel_short_q_models,
                    mid_q_models=panel_mid_q_models,
                )
                if not stock_states:
                    continue
                sector_states = self._deps.build_sector_states(
                    stock_states,
                    stock_score_fn=self._deps.stock_policy_score,
                )
                composite_state = self._deps.compose_state(
                    market=market_state,
                    sectors=sector_states,
                    stocks=stock_states,
                    cross_section=cross_section,
                )
                steps.append(
                    self._deps.trajectory_step_cls(
                        date=date,
                        next_date=next_date,
                        composite_state=composite_state,
                        stock_states=stock_states,
                        horizon_metrics=self._deps.panel_horizon_metrics(scored_rows),
                    )
                )

        return self._deps.backtest_trajectory_cls(prepared=prepared, steps=steps)


class DeepForecastBackend:
    name = "deep"

    def __init__(self, deps: ForecastRuntimeDependencies) -> None:
        self._deps = deps

    def build_trajectory(
        self,
        prepared: object,
        *,
        retrain_days: int = 20,
    ) -> object:
        settings = getattr(prepared, "settings")
        market_valid = getattr(prepared, "market_valid")
        panel = getattr(prepared, "panel")
        dates = getattr(prepared, "dates")
        min_train_days = int(settings["min_train_days"])
        training_window_days = _resolve_training_window_days(settings=settings, min_train_days=min_train_days)
        steps: list[object] = []
        market_valid_sorted, market_valid_bounds = self._deps.build_date_slice_index(
            market_valid,
            sort_cols=["date"],
        )

        tensor_market, tensor_market_cols = tensorize_temporal_frame(
            market_valid,
            feature_cols=getattr(prepared, "market_feature_cols"),
            group_col=None,
            lag_depth=3,
        )
        self._deps.emit_progress("trajectory", f"backend={self.name} 宸插畬鎴愬競鍦烘椂搴忓紶閲忓寲: cols={len(tensor_market_cols)}")
        tensor_panel, tensor_panel_cols = tensorize_temporal_frame(
            panel,
            feature_cols=getattr(prepared, "feature_cols"),
            group_col="symbol",
            lag_depth=3,
        )
        self._deps.emit_progress("trajectory", f"backend={self.name} 宸插畬鎴愪釜鑲℃椂搴忓紶閲忓寲: cols={len(tensor_panel_cols)}")
        tensor_market_sorted, tensor_market_bounds = self._deps.build_date_slice_index(
            tensor_market,
            sort_cols=["date"],
        )
        tensor_panel_sorted, tensor_panel_bounds = self._deps.build_date_slice_index(
            tensor_panel,
            sort_cols=["date", "symbol"],
        )
        tensor_market_training_arrays = _prepare_target_arrays(
            tensor_market_sorted,
            feature_cols=tensor_market_cols,
            target_cols=MARKET_BINARY_TARGETS,
        )
        tensor_panel_training_arrays = _prepare_target_arrays(
            tensor_panel_sorted,
            feature_cols=tensor_panel_cols,
            target_cols=PANEL_BINARY_TARGETS + PANEL_QUANTILE_TARGETS,
        )
        block_starts = list(range(min_train_days, len(dates) - 1, max(1, int(retrain_days))))
        self._deps.emit_progress(
            "trajectory",
            f"backend={self.name} building trajectory: blocks={len(block_starts)}, dates={len(dates)}, universe={len(getattr(prepared, 'stock_frames'))}, training_window_days={training_window_days or 'expanding'}",
        )
        trajectory_started = time.perf_counter()

        for block_idx, block_start in enumerate(block_starts, start=1):
            elapsed = time.perf_counter() - trajectory_started
            completed = max(0, block_idx - 1)
            eta = 0.0 if completed <= 0 else (elapsed / completed) * (len(block_starts) - completed)
            if _should_emit_block_progress(block_idx=block_idx, total_blocks=len(block_starts)):
                self._deps.emit_progress(
                    "trajectory",
                    f"backend={self.name} training block {block_idx}/{len(block_starts)}: cutoff {pd.Timestamp(dates[block_start - 1]).date()} | elapsed={self._deps.format_elapsed(elapsed)} | eta={self._deps.format_elapsed(eta)}",
                )
            market_train_start, market_train_end = _training_row_window(
                bounds=tensor_market_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            if market_train_end <= market_train_start:
                continue
            market_models = _fit_binary_model_family(
                model_cls=self._deps.mlp_model_cls,
                l2=float(settings["l2"]),
                feature_cols=tensor_market_cols,
                prepared_targets=_slice_prepared_target_family(
                    tensor_market_training_arrays,
                    start=market_train_start,
                    end=market_train_end,
                    target_cols=MARKET_BINARY_TARGETS,
                ),
            )
            panel_train_start, panel_train_end = _training_row_window(
                bounds=tensor_panel_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            if panel_train_end <= panel_train_start:
                continue
            panel_models = _fit_binary_model_family(
                model_cls=self._deps.mlp_model_cls,
                l2=float(settings["l2"]),
                feature_cols=tensor_panel_cols,
                prepared_targets=_slice_prepared_target_family(
                    tensor_panel_training_arrays,
                    start=panel_train_start,
                    end=panel_train_end,
                    target_cols=PANEL_BINARY_TARGETS,
                ),
            )
            short_x, short_y = _slice_prepared_target_arrays(
                tensor_panel_training_arrays,
                start=panel_train_start,
                end=panel_train_end,
                target_col="excess_ret_1_vs_mkt",
            )
            panel_short_q_models = _fit_quantile_model_family(
                model_cls=self._deps.mlp_quantile_model_cls,
                feature_cols=tensor_panel_cols,
                l2=float(settings["l2"]),
                x=short_x,
                y=short_y,
            )
            mid_x, mid_y = _slice_prepared_target_arrays(
                tensor_panel_training_arrays,
                start=panel_train_start,
                end=panel_train_end,
                target_col="excess_ret_20_vs_sector",
            )
            panel_mid_q_models = _fit_quantile_model_family(
                model_cls=self._deps.mlp_quantile_model_cls,
                feature_cols=tensor_panel_cols,
                l2=float(settings["l2"]),
                x=mid_x,
                y=mid_y,
            )

            block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
            for idx in range(block_start, block_end):
                date = dates[idx]
                next_date = dates[idx + 1]
                market_start, market_end = tensor_market_bounds.get(date, (0, 0))
                market_row = tensor_market_sorted.iloc[market_start:market_end]
                if market_row.empty:
                    continue
                mkt_short = float(market_models["mkt_target_1d_up"].predict_proba(market_row, tensor_market_cols)[0])
                mkt_two = float(market_models["mkt_target_2d_up"].predict_proba(market_row, tensor_market_cols)[0])
                mkt_three = float(market_models["mkt_target_3d_up"].predict_proba(market_row, tensor_market_cols)[0])
                mkt_five = float(market_models["mkt_target_5d_up"].predict_proba(market_row, tensor_market_cols)[0])
                mkt_mid = float(market_models["mkt_target_20d_up"].predict_proba(market_row, tensor_market_cols)[0])
                market_hist_end = market_valid_bounds.get(date, (0, 0))[1]
                market_state, cross_section = self._deps.build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=_history_slice(market_valid_sorted, market_hist_end),
                    market_short_prob=mkt_short,
                    market_two_prob=mkt_two,
                    market_three_prob=mkt_three,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_start, panel_end = tensor_panel_bounds.get(date, (0, 0))
                panel_row = tensor_panel_sorted.iloc[panel_start:panel_end]
                stock_states, scored_rows = self._deps.build_stock_states_from_panel_slice(
                    panel_row=panel_row,
                    feature_cols=tensor_panel_cols,
                    short_model=panel_models["target_1d_excess_mkt_up"],
                    two_model=panel_models["target_2d_excess_mkt_up"],
                    three_model=panel_models["target_3d_excess_mkt_up"],
                    five_model=panel_models["target_5d_excess_mkt_up"],
                    mid_model=panel_models["target_20d_excess_sector_up"],
                    short_q_models=panel_short_q_models,
                    mid_q_models=panel_mid_q_models,
                )
                if not stock_states:
                    continue
                sector_states = self._deps.build_sector_states(
                    stock_states,
                    stock_score_fn=self._deps.stock_policy_score,
                )
                composite_state = self._deps.compose_state(
                    market=market_state,
                    sectors=sector_states,
                    stocks=stock_states,
                    cross_section=cross_section,
                )
                steps.append(
                    self._deps.trajectory_step_cls(
                        date=date,
                        next_date=next_date,
                        composite_state=composite_state,
                        stock_states=stock_states,
                        horizon_metrics=self._deps.panel_horizon_metrics(scored_rows),
                    )
                )

        return self._deps.backtest_trajectory_cls(prepared=prepared, steps=steps)


class HybridForecastBackend:
    name = "hybrid"

    def __init__(self, deps: ForecastRuntimeDependencies) -> None:
        self._deps = deps

    def build_trajectory(
        self,
        prepared: object,
        *,
        retrain_days: int = 20,
    ) -> object:
        settings = getattr(prepared, "settings")
        market_valid = getattr(prepared, "market_valid")
        panel = getattr(prepared, "panel")
        market_feature_cols = getattr(prepared, "market_feature_cols")
        feature_cols = getattr(prepared, "feature_cols")
        dates = getattr(prepared, "dates")
        min_train_days = int(settings["min_train_days"])
        training_window_days = _resolve_training_window_days(settings=settings, min_train_days=min_train_days)
        steps: list[object] = []
        market_valid_sorted, market_valid_bounds = self._deps.build_date_slice_index(
            market_valid,
            sort_cols=["date"],
        )
        panel_sorted, panel_bounds = self._deps.build_date_slice_index(
            panel,
            sort_cols=["date", "symbol"],
        )
        tensor_market, tensor_market_cols = tensorize_temporal_frame(
            market_valid,
            feature_cols=market_feature_cols,
            group_col=None,
            lag_depth=3,
        )
        tensor_panel, tensor_panel_cols = tensorize_temporal_frame(
            panel,
            feature_cols=feature_cols,
            group_col="symbol",
            lag_depth=3,
        )
        tensor_market_sorted, tensor_market_bounds = self._deps.build_date_slice_index(
            tensor_market,
            sort_cols=["date"],
        )
        tensor_panel_sorted, tensor_panel_bounds = self._deps.build_date_slice_index(
            tensor_panel,
            sort_cols=["date", "symbol"],
        )
        market_training_arrays = _prepare_target_arrays(
            market_valid_sorted,
            feature_cols=market_feature_cols,
            target_cols=MARKET_BINARY_TARGETS,
        )
        panel_training_arrays = _prepare_target_arrays(
            panel_sorted,
            feature_cols=feature_cols,
            target_cols=PANEL_BINARY_TARGETS + PANEL_QUANTILE_TARGETS,
        )
        tensor_market_training_arrays = _prepare_target_arrays(
            tensor_market_sorted,
            feature_cols=tensor_market_cols,
            target_cols=MARKET_BINARY_TARGETS,
        )
        tensor_panel_training_arrays = _prepare_target_arrays(
            tensor_panel_sorted,
            feature_cols=tensor_panel_cols,
            target_cols=PANEL_BINARY_TARGETS + PANEL_QUANTILE_TARGETS,
        )
        block_starts = list(range(min_train_days, len(dates) - 1, max(1, int(retrain_days))))
        self._deps.emit_progress(
            "trajectory",
            f"backend={self.name} building trajectory: blocks={len(block_starts)}, dates={len(dates)}, universe={len(getattr(prepared, 'stock_frames'))}, training_window_days={training_window_days or 'expanding'}",
        )
        trajectory_started = time.perf_counter()

        for block_idx, block_start in enumerate(block_starts, start=1):
            elapsed = time.perf_counter() - trajectory_started
            completed = max(0, block_idx - 1)
            eta = 0.0 if completed <= 0 else (elapsed / completed) * (len(block_starts) - completed)
            if _should_emit_block_progress(block_idx=block_idx, total_blocks=len(block_starts)):
                self._deps.emit_progress(
                    "trajectory",
                    f"backend={self.name} training block {block_idx}/{len(block_starts)}: cutoff {pd.Timestamp(dates[block_start - 1]).date()} | elapsed={self._deps.format_elapsed(elapsed)} | eta={self._deps.format_elapsed(eta)}",
                )
            raw_market_train_start, raw_market_train_end = _training_row_window(
                bounds=market_valid_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            tensor_market_train_start, tensor_market_train_end = _training_row_window(
                bounds=tensor_market_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            if raw_market_train_end <= raw_market_train_start or tensor_market_train_end <= tensor_market_train_start:
                continue
            market_models_linear = _fit_binary_model_family(
                model_cls=self._deps.logistic_model_cls,
                l2=float(settings["l2"]),
                feature_cols=market_feature_cols,
                prepared_targets=_slice_prepared_target_family(
                    market_training_arrays,
                    start=raw_market_train_start,
                    end=raw_market_train_end,
                    target_cols=MARKET_BINARY_TARGETS,
                ),
            )
            market_models_deep = _fit_binary_model_family(
                model_cls=self._deps.mlp_model_cls,
                l2=float(settings["l2"]),
                feature_cols=tensor_market_cols,
                prepared_targets=_slice_prepared_target_family(
                    tensor_market_training_arrays,
                    start=tensor_market_train_start,
                    end=tensor_market_train_end,
                    target_cols=MARKET_BINARY_TARGETS,
                ),
            )
            market_models = _blend_binary_model_family(
                primary_models=market_models_linear,
                primary_feature_cols=market_feature_cols,
                secondary_models=market_models_deep,
                secondary_feature_cols=tensor_market_cols,
            )
            raw_panel_train_start, raw_panel_train_end = _training_row_window(
                bounds=panel_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            tensor_panel_train_start, tensor_panel_train_end = _training_row_window(
                bounds=tensor_panel_bounds,
                dates=dates,
                block_start=block_start,
                training_window_days=training_window_days,
            )
            if raw_panel_train_end <= raw_panel_train_start or tensor_panel_train_end <= tensor_panel_train_start:
                continue
            panel_models_linear = _fit_binary_model_family(
                model_cls=self._deps.logistic_model_cls,
                l2=float(settings["l2"]),
                feature_cols=feature_cols,
                prepared_targets=_slice_prepared_target_family(
                    panel_training_arrays,
                    start=raw_panel_train_start,
                    end=raw_panel_train_end,
                    target_cols=PANEL_BINARY_TARGETS,
                ),
            )
            panel_models_deep = _fit_binary_model_family(
                model_cls=self._deps.mlp_model_cls,
                l2=float(settings["l2"]),
                feature_cols=tensor_panel_cols,
                prepared_targets=_slice_prepared_target_family(
                    tensor_panel_training_arrays,
                    start=tensor_panel_train_start,
                    end=tensor_panel_train_end,
                    target_cols=PANEL_BINARY_TARGETS,
                ),
            )
            panel_models = _blend_binary_model_family(
                primary_models=panel_models_linear,
                primary_feature_cols=feature_cols,
                secondary_models=panel_models_deep,
                secondary_feature_cols=tensor_panel_cols,
            )
            raw_short_x, raw_short_y = _slice_prepared_target_arrays(
                panel_training_arrays,
                start=raw_panel_train_start,
                end=raw_panel_train_end,
                target_col="excess_ret_1_vs_mkt",
            )
            tensor_short_x, tensor_short_y = _slice_prepared_target_arrays(
                tensor_panel_training_arrays,
                start=tensor_panel_train_start,
                end=tensor_panel_train_end,
                target_col="excess_ret_1_vs_mkt",
            )
            panel_short_q_linear = _fit_quantile_model_family(
                model_cls=self._deps.quantile_model_cls,
                feature_cols=feature_cols,
                l2=float(settings["l2"]),
                x=raw_short_x,
                y=raw_short_y,
            )
            panel_short_q_deep = _fit_quantile_model_family(
                model_cls=self._deps.mlp_quantile_model_cls,
                feature_cols=tensor_panel_cols,
                l2=float(settings["l2"]),
                x=tensor_short_x,
                y=tensor_short_y,
            )
            panel_short_q_models = _blend_quantile_model_family(
                primary_models=panel_short_q_linear,
                primary_feature_cols=feature_cols,
                secondary_models=panel_short_q_deep,
                secondary_feature_cols=tensor_panel_cols,
            )
            raw_mid_x, raw_mid_y = _slice_prepared_target_arrays(
                panel_training_arrays,
                start=raw_panel_train_start,
                end=raw_panel_train_end,
                target_col="excess_ret_20_vs_sector",
            )
            tensor_mid_x, tensor_mid_y = _slice_prepared_target_arrays(
                tensor_panel_training_arrays,
                start=tensor_panel_train_start,
                end=tensor_panel_train_end,
                target_col="excess_ret_20_vs_sector",
            )
            panel_mid_q_linear = _fit_quantile_model_family(
                model_cls=self._deps.quantile_model_cls,
                feature_cols=feature_cols,
                l2=float(settings["l2"]),
                x=raw_mid_x,
                y=raw_mid_y,
            )
            panel_mid_q_deep = _fit_quantile_model_family(
                model_cls=self._deps.mlp_quantile_model_cls,
                feature_cols=tensor_panel_cols,
                l2=float(settings["l2"]),
                x=tensor_mid_x,
                y=tensor_mid_y,
            )
            panel_mid_q_models = _blend_quantile_model_family(
                primary_models=panel_mid_q_linear,
                primary_feature_cols=feature_cols,
                secondary_models=panel_mid_q_deep,
                secondary_feature_cols=tensor_panel_cols,
            )

            block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
            for idx in range(block_start, block_end):
                date = dates[idx]
                next_date = dates[idx + 1]
                market_start, market_end = tensor_market_bounds.get(date, (0, 0))
                market_row = tensor_market_sorted.iloc[market_start:market_end]
                if market_row.empty:
                    continue
                mkt_short = float(market_models["mkt_target_1d_up"].predict_proba(market_row)[0])
                mkt_two = float(market_models["mkt_target_2d_up"].predict_proba(market_row)[0])
                mkt_three = float(market_models["mkt_target_3d_up"].predict_proba(market_row)[0])
                mkt_five = float(market_models["mkt_target_5d_up"].predict_proba(market_row)[0])
                mkt_mid = float(market_models["mkt_target_20d_up"].predict_proba(market_row)[0])
                market_hist_end = market_valid_bounds.get(date, (0, 0))[1]
                market_state, cross_section = self._deps.build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=_history_slice(market_valid_sorted, market_hist_end),
                    market_short_prob=mkt_short,
                    market_two_prob=mkt_two,
                    market_three_prob=mkt_three,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_start, panel_end = tensor_panel_bounds.get(date, (0, 0))
                panel_row = tensor_panel_sorted.iloc[panel_start:panel_end]
                stock_states, scored_rows = self._deps.build_stock_states_from_panel_slice(
                    panel_row=panel_row,
                    feature_cols=tensor_panel_cols,
                    short_model=panel_models["target_1d_excess_mkt_up"],
                    two_model=panel_models["target_2d_excess_mkt_up"],
                    three_model=panel_models["target_3d_excess_mkt_up"],
                    five_model=panel_models["target_5d_excess_mkt_up"],
                    mid_model=panel_models["target_20d_excess_sector_up"],
                    short_q_models=panel_short_q_models,
                    mid_q_models=panel_mid_q_models,
                )
                if not stock_states:
                    continue
                sector_states = self._deps.build_sector_states(
                    stock_states,
                    stock_score_fn=self._deps.stock_policy_score,
                )
                composite_state = self._deps.compose_state(
                    market=market_state,
                    sectors=sector_states,
                    stocks=stock_states,
                    cross_section=cross_section,
                )
                steps.append(
                    self._deps.trajectory_step_cls(
                        date=date,
                        next_date=next_date,
                        composite_state=composite_state,
                        stock_states=stock_states,
                        horizon_metrics=self._deps.panel_horizon_metrics(scored_rows),
                    )
                )

        return self._deps.backtest_trajectory_cls(prepared=prepared, steps=steps)


def make_forecast_backend(
    name: str | None,
    *,
    deps: ForecastRuntimeDependencies,
) -> ForecastBackend:
    backend = (str(name).strip().lower() if name is not None else "linear") or "linear"
    if backend == "linear":
        return LinearForecastBackend(deps=deps)
    if backend == "deep":
        return DeepForecastBackend(deps=deps)
    if backend == "hybrid":
        return HybridForecastBackend(deps=deps)
    raise ValueError(f"Unsupported forecast backend: {backend}")
