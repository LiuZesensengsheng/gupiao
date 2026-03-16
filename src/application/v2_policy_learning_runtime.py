from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np

from src.application.v2_contracts import (
    LearnedPolicyModel,
    PolicySpec,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)

_COARSE_CALIBRATION_MIN_STEPS = 96
_COARSE_CALIBRATION_FRACTION = 0.30
_COARSE_CALIBRATION_KEEP = 6


@dataclass(frozen=True)
class PolicyLearningDependencies:
    emit_progress: Callable[[str, str], None]
    policy_objective_score: Callable[[V2BacktestSummary], float]
    run_v2_backtest_live: Callable[..., V2BacktestSummary]
    run_v2_backtest_core: Callable[..., tuple[V2BacktestSummary, list[dict[str, float]]]]
    policy_feature_names: Callable[[], list[str]]
    fit_ridge_regression: Callable[..., tuple[float, np.ndarray]]
    predict_ridge: Callable[[np.ndarray, float, np.ndarray], float]
    r2_score: Callable[[np.ndarray, np.ndarray], float]


def _policy_spec_key(spec: PolicySpec) -> tuple[float, float, float, int, int, int, float, float, float]:
    return (
        float(spec.risk_on_exposure),
        float(spec.cautious_exposure),
        float(spec.risk_off_exposure),
        int(spec.risk_on_positions),
        int(spec.cautious_positions),
        int(spec.risk_off_positions),
        float(spec.risk_on_turnover_cap),
        float(spec.cautious_turnover_cap),
        float(spec.risk_off_turnover_cap),
    )


def _trajectory_step_count(trajectory: Any | None) -> int:
    steps = getattr(trajectory, "steps", None)
    if steps is None:
        return 0
    try:
        return int(len(steps))
    except Exception:
        return 0


def _build_coarse_calibration_trajectory(trajectory: Any | None) -> Any | None:
    if trajectory is None:
        return None
    steps = getattr(trajectory, "steps", None)
    if steps is None:
        return None
    rows = list(steps)
    total_steps = len(rows)
    if total_steps < _COARSE_CALIBRATION_MIN_STEPS:
        return None
    coarse_steps = max(int(np.ceil(total_steps * _COARSE_CALIBRATION_FRACTION)), _COARSE_CALIBRATION_MIN_STEPS // 2)
    coarse_steps = min(total_steps - 24, coarse_steps)
    if coarse_steps <= 0 or coarse_steps >= total_steps:
        return None
    return type(trajectory)(
        prepared=getattr(trajectory, "prepared", None),
        steps=list(rows[:coarse_steps]),
    )


def calibrate_v2_policy(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    baseline: V2BacktestSummary | None = None,
    trajectory: Any | None = None,
    retrain_days: int = 20,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    deps: PolicyLearningDependencies,
) -> V2CalibrationResult:
    baseline_spec = PolicySpec()
    baseline = baseline or deps.run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        policy_spec=baseline_spec,
        trajectory=trajectory,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    exposure_sets = [
        (0.75, 0.50, 0.25),
        (0.85, 0.60, 0.35),
        (0.90, 0.65, 0.35),
    ]
    position_sets = [
        (3, 2, 1),
        (4, 3, 2),
        (5, 3, 1),
        (4, 4, 2),
    ]
    turnover_sets = [
        (0.40, 0.28, 0.20),
        (0.34, 0.24, 0.16),
        (0.45, 0.32, 0.22),
    ]
    candidates: list[PolicySpec] = []
    seen_specs: set[tuple[float, float, float, int, int, int, float, float, float]] = set()
    for exp in exposure_sets:
        for pos in position_sets:
            for turn in turnover_sets:
                spec = PolicySpec(
                    risk_on_exposure=float(exp[0]),
                    cautious_exposure=float(exp[1]),
                    risk_off_exposure=float(exp[2]),
                    risk_on_positions=int(pos[0]),
                    cautious_positions=int(pos[1]),
                    risk_off_positions=int(pos[2]),
                    risk_on_turnover_cap=float(turn[0]),
                    cautious_turnover_cap=float(turn[1]),
                    risk_off_turnover_cap=float(turn[2]),
                )
                key = _policy_spec_key(spec)
                if key in seen_specs:
                    continue
                seen_specs.add(key)
                candidates.append(spec)
    best_spec = baseline_spec
    best_summary = baseline
    best_score = deps.policy_objective_score(baseline)
    trials: list[dict[str, object]] = [
        {
            "policy": asdict(baseline_spec),
            "summary": asdict(baseline),
            "score": float(best_score),
        }
    ]
    baseline_key = _policy_spec_key(baseline_spec)
    candidate_specs = [spec for spec in candidates if _policy_spec_key(spec) != baseline_key]
    total_candidates = len(candidate_specs)
    coarse_trajectory = _build_coarse_calibration_trajectory(trajectory)
    full_candidate_specs = list(candidate_specs)
    if coarse_trajectory is not None and total_candidates > _COARSE_CALIBRATION_KEEP:
        coarse_step_count = _trajectory_step_count(coarse_trajectory)
        deps.emit_progress(
            "calibration",
            f"开始参数粗筛: candidates={total_candidates}, coarse_steps={coarse_step_count}",
        )
        coarse_ranked: list[tuple[float, PolicySpec]] = []
        for spec in candidate_specs:
            coarse_summary = deps.run_v2_backtest_live(
                strategy_id=strategy_id,
                config_path=config_path,
                source=source,
                universe_file=universe_file,
                universe_limit=universe_limit,
                universe_tier=universe_tier,
                dynamic_universe=dynamic_universe,
                generator_target_size=generator_target_size,
                generator_coarse_size=generator_coarse_size,
                generator_theme_aware=generator_theme_aware,
                generator_use_concepts=generator_use_concepts,
                policy_spec=spec,
                trajectory=coarse_trajectory,
                retrain_days=retrain_days,
                cache_root=cache_root,
                refresh_cache=refresh_cache,
                forecast_backend=forecast_backend,
                use_us_index_context=use_us_index_context,
                us_index_source=us_index_source,
            )
            coarse_ranked.append((float(deps.policy_objective_score(coarse_summary)), spec))
        coarse_ranked.sort(key=lambda item: item[0], reverse=True)
        full_candidate_specs = [spec for _, spec in coarse_ranked[: min(total_candidates, _COARSE_CALIBRATION_KEEP)]]
        deps.emit_progress(
            "calibration",
            f"粗筛完成: kept={len(full_candidate_specs)}/{total_candidates}",
        )
    deps.emit_progress(
        "calibration",
        f"开始参数搜索: candidates={total_candidates}, full_eval={len(full_candidate_specs)}",
    )
    for idx, spec in enumerate(full_candidate_specs, start=1):
        deps.emit_progress(
            "calibration",
            f"评估候选 {idx}/{len(full_candidate_specs)}: exposure={spec.risk_on_exposure:.2f}, positions={spec.risk_on_positions}, turnover={spec.risk_on_turnover_cap:.2f}",
        )
        summary = deps.run_v2_backtest_live(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            dynamic_universe=dynamic_universe,
            generator_target_size=generator_target_size,
            generator_coarse_size=generator_coarse_size,
            generator_theme_aware=generator_theme_aware,
            generator_use_concepts=generator_use_concepts,
            policy_spec=spec,
            trajectory=trajectory,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        score = deps.policy_objective_score(summary)
        trials.append(
            {
                "policy": asdict(spec),
                "summary": asdict(summary),
                "score": float(score),
            }
        )
        if score > best_score:
            best_score = float(score)
            best_spec = spec
            best_summary = summary
            deps.emit_progress("calibration", f"发现更优参数: score={best_score:.4f}")
    return V2CalibrationResult(
        best_policy=best_spec,
        best_score=float(best_score),
        baseline=baseline,
        calibrated=best_summary,
        trials=trials,
    )


def learn_v2_policy_model(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    l2: float = 1.0,
    baseline: V2BacktestSummary | None = None,
    trajectory: Any | None = None,
    fit_trajectory: Any | None = None,
    evaluation_trajectory: Any | None = None,
    retrain_days: int = 20,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    deps: PolicyLearningDependencies,
) -> V2PolicyLearningResult:
    fit_trajectory = fit_trajectory or trajectory
    evaluation_trajectory = evaluation_trajectory or trajectory
    baseline = baseline or deps.run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        trajectory=evaluation_trajectory,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    _, rows = deps.run_v2_backtest_core(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        capture_learning_rows=True,
        trajectory=fit_trajectory,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    feature_names = deps.policy_feature_names()
    if not rows:
        model = LearnedPolicyModel(
            feature_names=feature_names,
            exposure_intercept=0.60,
            exposure_coef=[0.0] * len(feature_names),
            position_intercept=3.0,
            position_coef=[0.0] * len(feature_names),
            turnover_intercept=0.22,
            turnover_coef=[0.0] * len(feature_names),
            train_rows=0,
            train_r2_exposure=0.0,
            train_r2_positions=0.0,
            train_r2_turnover=0.0,
        )
        learned_summary = deps.run_v2_backtest_live(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            dynamic_universe=dynamic_universe,
            generator_target_size=generator_target_size,
            generator_coarse_size=generator_coarse_size,
            generator_theme_aware=generator_theme_aware,
            generator_use_concepts=generator_use_concepts,
            learned_policy=model,
            trajectory=evaluation_trajectory,
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        return V2PolicyLearningResult(model=model, baseline=baseline, learned=learned_summary)

    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)
    y_exposure = np.asarray([float(row["target_exposure"]) for row in rows], dtype=float)
    y_positions = np.asarray([float(row["target_positions"]) for row in rows], dtype=float)
    y_turnover = np.asarray([float(row["target_turnover"]) for row in rows], dtype=float)
    sample_weight = np.asarray([float(row.get("sample_weight", 1.0)) for row in rows], dtype=float)

    exp_b, exp_w = deps.fit_ridge_regression(X, y_exposure, l2=l2, sample_weight=sample_weight)
    pos_b, pos_w = deps.fit_ridge_regression(X, y_positions, l2=l2, sample_weight=sample_weight)
    turn_b, turn_w = deps.fit_ridge_regression(X, y_turnover, l2=l2, sample_weight=sample_weight)

    pred_exp = np.asarray([deps.predict_ridge(row, exp_b, exp_w) for row in X], dtype=float)
    pred_pos = np.asarray([deps.predict_ridge(row, pos_b, pos_w) for row in X], dtype=float)
    pred_turn = np.asarray([deps.predict_ridge(row, turn_b, turn_w) for row in X], dtype=float)

    model = LearnedPolicyModel(
        feature_names=feature_names,
        exposure_intercept=float(exp_b),
        exposure_coef=[float(x) for x in exp_w.tolist()],
        position_intercept=float(pos_b),
        position_coef=[float(x) for x in pos_w.tolist()],
        turnover_intercept=float(turn_b),
        turnover_coef=[float(x) for x in turn_w.tolist()],
        train_rows=int(len(rows)),
        train_r2_exposure=float(deps.r2_score(y_exposure, pred_exp)),
        train_r2_positions=float(deps.r2_score(y_positions, pred_pos)),
        train_r2_turnover=float(deps.r2_score(y_turnover, pred_turn)),
    )
    learned_summary = deps.run_v2_backtest_live(
        strategy_id=strategy_id,
        config_path=config_path,
        source=source,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        learned_policy=model,
        trajectory=evaluation_trajectory,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    return V2PolicyLearningResult(
        model=model,
        baseline=baseline,
        learned=learned_summary,
    )


def baseline_only_calibration(
    baseline: V2BacktestSummary,
    *,
    deps: PolicyLearningDependencies,
) -> V2CalibrationResult:
    baseline_spec = PolicySpec()
    score = deps.policy_objective_score(baseline)
    return V2CalibrationResult(
        best_policy=baseline_spec,
        best_score=float(score),
        baseline=baseline,
        calibrated=baseline,
        trials=[
            {
                "policy": asdict(baseline_spec),
                "summary": asdict(baseline),
                "score": float(score),
            }
        ],
    )


def placeholder_learning_result(baseline: V2BacktestSummary, *, deps: PolicyLearningDependencies) -> V2PolicyLearningResult:
    feature_names = deps.policy_feature_names()
    model = LearnedPolicyModel(
        feature_names=feature_names,
        exposure_intercept=0.60,
        exposure_coef=[0.0] * len(feature_names),
        position_intercept=3.0,
        position_coef=[0.0] * len(feature_names),
        turnover_intercept=0.22,
        turnover_coef=[0.0] * len(feature_names),
        train_rows=0,
        train_r2_exposure=0.0,
        train_r2_positions=0.0,
        train_r2_turnover=0.0,
    )
    return V2PolicyLearningResult(
        model=model,
        baseline=baseline,
        learned=baseline,
    )
