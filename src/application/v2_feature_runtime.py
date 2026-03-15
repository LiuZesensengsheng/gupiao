from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import pandas as pd


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
        steps: list[object] = []
        market_sorted, market_bounds = self._deps.build_date_slice_index(
            market_valid,
            sort_cols=["date"],
        )
        panel_sorted, panel_bounds = self._deps.build_date_slice_index(
            panel,
            sort_cols=["date", "symbol"],
        )
        block_starts = list(range(min_train_days, len(dates) - 1, max(1, int(retrain_days))))
        self._deps.emit_progress(
            "trajectory",
            f"backend={self.name} 寮€濮嬫瀯寤鸿建杩? blocks={len(block_starts)}, dates={len(dates)}, universe={len(getattr(prepared, 'stock_frames'))}",
        )
        trajectory_started = time.perf_counter()

        for block_idx, block_start in enumerate(block_starts, start=1):
            elapsed = time.perf_counter() - trajectory_started
            completed = max(0, block_idx - 1)
            eta = 0.0 if completed <= 0 else (elapsed / completed) * (len(block_starts) - completed)
            self._deps.emit_progress(
                "trajectory",
                f"backend={self.name} 璁粌鍧?{block_idx}/{len(block_starts)}: 鎴 {pd.Timestamp(dates[block_start - 1]).date()} | elapsed={self._deps.format_elapsed(elapsed)} | eta={self._deps.format_elapsed(eta)}",
            )
            train_cutoff = market_bounds.get(dates[block_start - 1], (0, 0))[1]
            market_train = market_sorted.iloc[:train_cutoff].copy()
            if market_train.empty:
                continue
            market_short_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_1d_up",
            )
            market_two_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_2d_up",
            )
            market_three_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_3d_up",
            )
            market_five_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_5d_up",
            )
            market_mid_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_20d_up",
            )
            panel_cutoff = panel_bounds.get(dates[block_start - 1], (0, 0))[1]
            panel_train = panel_sorted.iloc[:panel_cutoff].copy()
            if panel_train.empty:
                continue
            panel_short_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_1d_excess_mkt_up",
            )
            panel_two_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_2d_excess_mkt_up",
            )
            panel_three_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_3d_excess_mkt_up",
            )
            panel_five_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_5d_excess_mkt_up",
            )
            panel_mid_model = self._deps.logistic_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_20d_excess_sector_up",
            )
            panel_short_q_models = self._deps.fit_quantile_quintet(
                panel_train,
                feature_cols=feature_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=float(settings["l2"]),
            )
            panel_mid_q_models = self._deps.fit_quantile_quintet(
                panel_train,
                feature_cols=feature_cols,
                target_col="excess_ret_20_vs_sector",
                l2=float(settings["l2"]),
            )

            block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
            for idx in range(block_start, block_end):
                date = dates[idx]
                next_date = dates[idx + 1]
                market_start, market_end = market_bounds.get(date, (0, 0))
                market_row = market_sorted.iloc[market_start:market_end].copy()
                if market_row.empty:
                    continue
                mkt_short = float(market_short_model.predict_proba(market_row, market_feature_cols)[0])
                mkt_two = float(market_two_model.predict_proba(market_row, market_feature_cols)[0])
                mkt_three = float(market_three_model.predict_proba(market_row, market_feature_cols)[0])
                mkt_five = float(market_five_model.predict_proba(market_row, market_feature_cols)[0])
                mkt_mid = float(market_mid_model.predict_proba(market_row, market_feature_cols)[0])
                market_state, cross_section = self._deps.build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=market_sorted.iloc[:market_end].copy(),
                    market_short_prob=mkt_short,
                    market_two_prob=mkt_two,
                    market_three_prob=mkt_three,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_start, panel_end = panel_bounds.get(date, (0, 0))
                panel_row = panel_sorted.iloc[panel_start:panel_end].copy()
                stock_states, scored_rows = self._deps.build_stock_states_from_panel_slice(
                    panel_row=panel_row,
                    feature_cols=feature_cols,
                    short_model=panel_short_model,
                    two_model=panel_two_model,
                    three_model=panel_three_model,
                    five_model=panel_five_model,
                    mid_model=panel_mid_model,
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
        block_starts = list(range(min_train_days, len(dates) - 1, max(1, int(retrain_days))))
        self._deps.emit_progress(
            "trajectory",
            f"backend={self.name} 寮€濮嬫瀯寤鸿建杩? blocks={len(block_starts)}, dates={len(dates)}, universe={len(getattr(prepared, 'stock_frames'))}",
        )
        trajectory_started = time.perf_counter()

        for block_idx, block_start in enumerate(block_starts, start=1):
            elapsed = time.perf_counter() - trajectory_started
            completed = max(0, block_idx - 1)
            eta = 0.0 if completed <= 0 else (elapsed / completed) * (len(block_starts) - completed)
            self._deps.emit_progress(
                "trajectory",
                f"backend={self.name} 璁粌鍧?{block_idx}/{len(block_starts)}: 鎴 {pd.Timestamp(dates[block_start - 1]).date()} | elapsed={self._deps.format_elapsed(elapsed)} | eta={self._deps.format_elapsed(eta)}",
            )
            train_cutoff = tensor_market_bounds.get(dates[block_start - 1], (0, 0))[1]
            market_train = tensor_market_sorted.iloc[:train_cutoff].copy()
            if market_train.empty:
                continue
            market_short_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_1d_up",
            )
            market_two_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_2d_up",
            )
            market_three_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_3d_up",
            )
            market_five_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_5d_up",
            )
            market_mid_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_20d_up",
            )
            panel_cutoff = tensor_panel_bounds.get(dates[block_start - 1], (0, 0))[1]
            panel_train = tensor_panel_sorted.iloc[:panel_cutoff].copy()
            if panel_train.empty:
                continue
            panel_short_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_1d_excess_mkt_up",
            )
            panel_two_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_2d_excess_mkt_up",
            )
            panel_three_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_3d_excess_mkt_up",
            )
            panel_five_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_5d_excess_mkt_up",
            )
            panel_mid_model = self._deps.mlp_model_cls(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_20d_excess_sector_up",
            )
            panel_short_q_models = self._deps.fit_mlp_quantile_quintet(
                panel_train,
                feature_cols=tensor_panel_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=float(settings["l2"]),
            )
            panel_mid_q_models = self._deps.fit_mlp_quantile_quintet(
                panel_train,
                feature_cols=tensor_panel_cols,
                target_col="excess_ret_20_vs_sector",
                l2=float(settings["l2"]),
            )

            block_end = min(block_start + max(1, int(retrain_days)), len(dates) - 1)
            for idx in range(block_start, block_end):
                date = dates[idx]
                next_date = dates[idx + 1]
                market_start, market_end = tensor_market_bounds.get(date, (0, 0))
                market_row = tensor_market_sorted.iloc[market_start:market_end].copy()
                if market_row.empty:
                    continue
                mkt_short = float(market_short_model.predict_proba(market_row, tensor_market_cols)[0])
                mkt_two = float(market_two_model.predict_proba(market_row, tensor_market_cols)[0])
                mkt_three = float(market_three_model.predict_proba(market_row, tensor_market_cols)[0])
                mkt_five = float(market_five_model.predict_proba(market_row, tensor_market_cols)[0])
                mkt_mid = float(market_mid_model.predict_proba(market_row, tensor_market_cols)[0])
                market_hist_end = market_valid_bounds.get(date, (0, 0))[1]
                market_state, cross_section = self._deps.build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=market_valid_sorted.iloc[:market_hist_end].copy(),
                    market_short_prob=mkt_short,
                    market_two_prob=mkt_two,
                    market_three_prob=mkt_three,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_start, panel_end = tensor_panel_bounds.get(date, (0, 0))
                panel_row = tensor_panel_sorted.iloc[panel_start:panel_end].copy()
                stock_states, scored_rows = self._deps.build_stock_states_from_panel_slice(
                    panel_row=panel_row,
                    feature_cols=tensor_panel_cols,
                    short_model=panel_short_model,
                    two_model=panel_two_model,
                    three_model=panel_three_model,
                    five_model=panel_five_model,
                    mid_model=panel_mid_model,
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
    raise ValueError(f"Unsupported forecast backend: {backend}")
