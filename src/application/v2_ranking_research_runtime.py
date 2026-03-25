from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


@dataclass(frozen=True)
class RankingResearchDependencies:
    emit_progress_fn: Callable[[str, str], None]
    load_or_build_v2_backtest_trajectory_fn: Callable[..., Any]
    split_research_trajectory_fn: Callable[..., tuple[Any, Any, Any]]
    trajectory_step_count_fn: Callable[[Any], int]
    build_leader_artifact_payloads_fn: Callable[..., dict[str, object]]
    build_research_label_artifact_payloads_fn: Callable[..., dict[str, object]]
    build_signal_training_artifacts_fn: Callable[..., dict[str, object]]


@dataclass(frozen=True)
class V2RankingResearchResult:
    strategy_id: str
    forecast_backend: str
    retrain_days: int
    training_window_days: int | None
    split_mode: str
    embargo_days: int
    fit_scope: str
    evaluation_scope: str
    top_k: int
    candidate_limit: int
    leader_min_theme_size: int
    exit_min_theme_size: int
    exit_candidate_limit: int
    signal_l2: float
    trajectory_steps: int
    fit_steps: int
    evaluation_steps: int
    evaluation_window_start: str
    evaluation_window_end: str
    latest_state_date: str
    leader_manifest: dict[str, object]
    leader_candidates: list[dict[str, object]]
    full_training_label_manifest: dict[str, object]
    fit_training_label_manifest: dict[str, object]
    evaluation_training_label_manifest: dict[str, object]
    signal_training_manifest: dict[str, object]
    leader_rank_model: dict[str, object]
    exit_behavior_model: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        return asdict(self)

    def summary(self) -> dict[str, object]:
        leader_eval = dict(self.leader_manifest.get("evaluation", {}))
        return {
            "strategy_id": self.strategy_id,
            "forecast_backend": self.forecast_backend,
            "trajectory_steps": int(self.trajectory_steps),
            "fit_steps": int(self.fit_steps),
            "evaluation_steps": int(self.evaluation_steps),
            "fit_scope": self.fit_scope,
            "evaluation_scope": self.evaluation_scope,
            "evaluation_window": {
                "start": self.evaluation_window_start,
                "end": self.evaluation_window_end,
            },
            "leader_metrics": {
                "candidate_recall_at_k": float(leader_eval.get("candidate_recall_at_k", 0.0)),
                "conviction_precision_at_1": float(leader_eval.get("conviction_precision_at_1", 0.0)),
                "confirmed_precision_at_1": float(leader_eval.get("confirmed_precision_at_1", 0.0)),
                "possible_recall_at_k": float(leader_eval.get("possible_recall_at_k", 0.0)),
                "not_leader_avoid_rate": float(leader_eval.get("not_leader_avoid_rate", 0.0)),
                "ndcg_at_k": float(leader_eval.get("ndcg_at_k", 0.0)),
                "hard_negative_survival_recall": float(leader_eval.get("hard_negative_survival_recall", 0.0)),
                "hard_negative_filter_rate": float(leader_eval.get("hard_negative_filter_rate", 0.0)),
            },
            "signal_metrics": {
                "leader_eval_top1_hit_rate": float(
                    self.signal_training_manifest.get("leader_eval_top1_hit_rate", 0.0)
                ),
                "leader_eval_top3_recall": float(
                    self.signal_training_manifest.get("leader_eval_top3_recall", 0.0)
                ),
                "leader_eval_possible_recall_at_k": float(
                    self.signal_training_manifest.get("leader_eval_possible_recall_at_k", 0.0)
                ),
                "leader_eval_confirmed_precision_at_1": float(
                    self.signal_training_manifest.get("leader_eval_confirmed_precision_at_1", 0.0)
                ),
                "leader_eval_not_leader_avoid_rate": float(
                    self.signal_training_manifest.get("leader_eval_not_leader_avoid_rate", 0.0)
                ),
                "leader_eval_ndcg_at_3": float(
                    self.signal_training_manifest.get("leader_eval_ndcg_at_3", 0.0)
                ),
                "leader_eval_filter_possible_recall": float(
                    self.signal_training_manifest.get("leader_eval_filter_possible_recall", 0.0)
                ),
                "leader_eval_filter_confirmed_recall": float(
                    self.signal_training_manifest.get("leader_eval_filter_confirmed_recall", 0.0)
                ),
                "leader_eval_filter_not_leader_rate": float(
                    self.signal_training_manifest.get("leader_eval_filter_not_leader_rate", 0.0)
                ),
                "exit_eval_rank_corr": float(self.signal_training_manifest.get("exit_eval_rank_corr", 0.0)),
                "exit_eval_path_corr": float(self.signal_training_manifest.get("exit_eval_path_corr", 0.0)),
                "exit_eval_precision": float(self.signal_training_manifest.get("exit_eval_precision", 0.0)),
                "exit_eval_recall": float(self.signal_training_manifest.get("exit_eval_recall", 0.0)),
                "exit_eval_accuracy": float(self.signal_training_manifest.get("exit_eval_accuracy", 0.0)),
                "exit_eval_label_accuracy": float(
                    self.signal_training_manifest.get("exit_eval_label_accuracy", 0.0)
                ),
                "exit_eval_keep_precision": float(
                    self.signal_training_manifest.get("exit_eval_keep_precision", 0.0)
                ),
                "exit_eval_watch_or_worse_recall": float(
                    self.signal_training_manifest.get("exit_eval_watch_or_worse_recall", 0.0)
                ),
                "exit_eval_reduce_or_worse_recall": float(
                    self.signal_training_manifest.get("exit_eval_reduce_or_worse_recall", 0.0)
                ),
                "exit_eval_exit_fast_recall": float(
                    self.signal_training_manifest.get("exit_eval_exit_fast_recall", 0.0)
                ),
            },
            "candidate_count": int(len(self.leader_candidates)),
            "top_candidate_symbols": [str(item.get("symbol", "")) for item in self.leader_candidates[:5]],
            "label_coverage": {
                "confirmed": int(self.full_training_label_manifest.get("leader_confirmed_count", 0)),
                "possible": int(self.full_training_label_manifest.get("leader_possible_count", 0)),
                "not_leader": int(self.full_training_label_manifest.get("leader_not_count", 0)),
            },
        }


def _trajectory_steps(trajectory: object | None) -> list[object]:
    return list(getattr(trajectory, "steps", []) or [])


def _clone_trajectory(trajectory: object | None, steps: list[object]) -> object | None:
    if trajectory is None or not steps:
        return None
    prepared = getattr(trajectory, "prepared", None)
    try:
        return type(trajectory)(
            prepared=prepared,
            steps=list(steps),
        )
    except TypeError:
        return SimpleNamespace(
            prepared=prepared,
            steps=list(steps),
        )


def _trajectory_window(trajectory: object | None) -> tuple[str, str]:
    steps = _trajectory_steps(trajectory)
    if not steps:
        return "", ""
    return str(getattr(steps[0], "date", ""))[:10], str(getattr(steps[-1], "date", ""))[:10]


def _last_state_date(trajectory: object | None) -> str:
    steps = _trajectory_steps(trajectory)
    if not steps:
        return ""
    last_step = steps[-1]
    composite_state = getattr(last_step, "composite_state", None)
    market_state = getattr(composite_state, "market", None)
    market_date = str(getattr(market_state, "as_of_date", "") or "").strip()
    if market_date:
        return market_date
    return str(getattr(last_step, "date", ""))[:10]


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ranking_cache_root(cache_root: str) -> Path:
    root = Path(str(cache_root)) / "ranking_research"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _trajectory_identity(trajectory: object | None) -> dict[str, object]:
    steps = _trajectory_steps(trajectory)
    return {
        "decorated_cache_key": str(getattr(trajectory, "_decorated_trajectory_cache_key", "") or ""),
        "raw_cache_key": str(getattr(trajectory, "_raw_trajectory_cache_key", "") or ""),
        "step_count": int(len(steps)),
        "window": _trajectory_window(trajectory),
        "latest_state_date": _last_state_date(trajectory),
    }


def _ranking_result_cache_key(
    *,
    strategy_id: str,
    forecast_backend: str,
    retrain_days: int,
    training_window_days: int | None,
    split_mode: str,
    embargo_days: int,
    top_k: int,
    candidate_limit: int,
    leader_min_theme_size: int,
    exit_min_theme_size: int,
    exit_candidate_limit: int,
    signal_l2: float,
    trajectory: object | None,
) -> str:
    payload = {
        "version": "v2-ranking-research-cache-19",
        "strategy_id": str(strategy_id),
        "forecast_backend": str(forecast_backend),
        "retrain_days": int(retrain_days),
        "training_window_days": None if training_window_days is None else int(training_window_days),
        "split_mode": str(split_mode),
        "embargo_days": int(embargo_days),
        "top_k": int(top_k),
        "candidate_limit": int(candidate_limit),
        "leader_min_theme_size": int(leader_min_theme_size),
        "exit_min_theme_size": int(exit_min_theme_size),
        "exit_candidate_limit": int(exit_candidate_limit),
        "signal_l2": float(signal_l2),
        "trajectory": _trajectory_identity(trajectory),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _ranking_result_cache_path(*, cache_root: str, cache_key: str) -> Path:
    return _ranking_cache_root(cache_root) / f"{cache_key}.pkl"


def _load_cached_ranking_result(*, cache_root: str, cache_key: str) -> V2RankingResearchResult | None:
    path = _ranking_result_cache_path(cache_root=cache_root, cache_key=cache_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            cached = pickle.load(f)
    except Exception:
        return None
    return cached if isinstance(cached, V2RankingResearchResult) else None


def _store_cached_ranking_result(
    *,
    cache_root: str,
    cache_key: str,
    result: V2RankingResearchResult,
) -> None:
    path = _ranking_result_cache_path(cache_root=cache_root, cache_key=cache_key)
    with path.open("wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_v2_ranking_research(
    *,
    dependencies: RankingResearchDependencies,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    info_file: str | None = None,
    info_lookback_days: int | None = None,
    info_half_life_days: float | None = None,
    use_info_fusion: bool | None = None,
    use_learned_info_fusion: bool | None = None,
    info_shadow_only: bool | None = None,
    info_types: str | None = None,
    info_source_mode: str | None = None,
    info_subsets: str | None = None,
    info_cutoff_time: str | None = None,
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
    training_window_days: int | None = 480,
    split_mode: str = "purged_wf",
    embargo_days: int = 20,
    top_k: int = 3,
    candidate_limit: int = 16,
    leader_min_theme_size: int = 3,
    exit_min_theme_size: int = 2,
    exit_candidate_limit: int = 8,
    signal_l2: float = 1.0,
) -> V2RankingResearchResult:
    dependencies.emit_progress_fn("ranking", f"载入排序研究轨迹: backend={forecast_backend}")
    trajectory = dependencies.load_or_build_v2_backtest_trajectory_fn(
        config_path=config_path,
        source=source,
        start_date=start_date,
        end_date=end_date,
        lookback_years=lookback_years,
        universe_file=universe_file,
        universe_limit=universe_limit,
        universe_tier=universe_tier,
        info_file=info_file,
        info_lookback_days=info_lookback_days,
        info_half_life_days=info_half_life_days,
        use_info_fusion=use_info_fusion,
        use_learned_info_fusion=use_learned_info_fusion,
        info_shadow_only=info_shadow_only,
        info_types=info_types,
        info_source_mode=info_source_mode,
        info_subsets=info_subsets,
        info_cutoff_time=info_cutoff_time,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        dynamic_universe=dynamic_universe,
        generator_target_size=generator_target_size,
        generator_coarse_size=generator_coarse_size,
        generator_theme_aware=generator_theme_aware,
        generator_use_concepts=generator_use_concepts,
        retrain_days=retrain_days,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        training_window_days=training_window_days,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    trajectory_steps = dependencies.trajectory_step_count_fn(trajectory)
    dependencies.emit_progress_fn("ranking", f"轨迹载入完成: steps={trajectory_steps}")
    cache_key = _ranking_result_cache_key(
        strategy_id=strategy_id,
        forecast_backend=forecast_backend,
        retrain_days=retrain_days,
        training_window_days=training_window_days,
        split_mode=split_mode,
        embargo_days=embargo_days,
        top_k=top_k,
        candidate_limit=candidate_limit,
        leader_min_theme_size=leader_min_theme_size,
        exit_min_theme_size=exit_min_theme_size,
        exit_candidate_limit=exit_candidate_limit,
        signal_l2=signal_l2,
        trajectory=trajectory,
    )
    if not refresh_cache:
        cached_result = _load_cached_ranking_result(
            cache_root=cache_root,
            cache_key=cache_key,
        )
        if cached_result is not None:
            dependencies.emit_progress_fn("cache", f"命中 ranking research 缓存: key={cache_key}")
            return cached_result

    if trajectory is None:
        train_trajectory = None
        validation_trajectory = None
        holdout_trajectory = None
    else:
        train_trajectory, validation_trajectory, holdout_trajectory = dependencies.split_research_trajectory_fn(
            trajectory,
            split_mode=split_mode,
            embargo_days=embargo_days,
        )
    fit_steps = _trajectory_steps(train_trajectory) + _trajectory_steps(validation_trajectory)
    fit_trajectory = _clone_trajectory(trajectory, fit_steps)
    fit_scope = "fit_split"
    if fit_trajectory is None:
        fit_trajectory = trajectory
        fit_scope = "full_fallback"

    evaluation_trajectory = holdout_trajectory
    evaluation_scope = "holdout"
    if not _trajectory_steps(evaluation_trajectory):
        evaluation_trajectory = trajectory
        evaluation_scope = "full_fallback"

    dependencies.emit_progress_fn(
        "ranking",
        (
            "样本切分完成: "
            f"fit_steps={dependencies.trajectory_step_count_fn(fit_trajectory)}, "
            f"evaluation_steps={dependencies.trajectory_step_count_fn(evaluation_trajectory)}"
        ),
    )

    evaluation_steps = _trajectory_steps(evaluation_trajectory)
    latest_state = None if not evaluation_steps else getattr(evaluation_steps[-1], "composite_state", None)

    leader_started = time.perf_counter()
    dependencies.emit_progress_fn("ranking", "开始评估 leader 候选")
    leader_payloads = dependencies.build_leader_artifact_payloads_fn(
        state=latest_state,
        trajectory=evaluation_trajectory,
        top_k=top_k,
        limit=candidate_limit,
    )
    dependencies.emit_progress_fn(
        "ranking",
        f"leader 候选评估完成: elapsed={time.perf_counter() - leader_started:.1f}s",
    )

    label_started = time.perf_counter()
    dependencies.emit_progress_fn("ranking", "开始构建训练标签")
    full_label_payloads = dependencies.build_research_label_artifact_payloads_fn(
        trajectory=trajectory,
        min_leader_theme_size=leader_min_theme_size,
        min_exit_theme_size=exit_min_theme_size,
        exit_candidate_limit=exit_candidate_limit,
    )
    fit_label_payloads = dependencies.build_research_label_artifact_payloads_fn(
        trajectory=fit_trajectory,
        min_leader_theme_size=leader_min_theme_size,
        min_exit_theme_size=exit_min_theme_size,
        exit_candidate_limit=exit_candidate_limit,
    )
    evaluation_label_payloads = dependencies.build_research_label_artifact_payloads_fn(
        trajectory=evaluation_trajectory,
        min_leader_theme_size=leader_min_theme_size,
        min_exit_theme_size=exit_min_theme_size,
        exit_candidate_limit=exit_candidate_limit,
    )
    dependencies.emit_progress_fn(
        "ranking",
        f"训练标签构建完成: elapsed={time.perf_counter() - label_started:.1f}s",
    )

    leader_fit_rows = [
        dict(item)
        for item in fit_label_payloads.get("leader_training_labels", [])
        if isinstance(item, dict)
    ]
    leader_evaluation_rows = [
        dict(item)
        for item in evaluation_label_payloads.get("leader_training_labels", [])
        if isinstance(item, dict)
    ]
    exit_fit_rows = [
        dict(item)
        for item in fit_label_payloads.get("exit_training_labels", [])
        if isinstance(item, dict)
    ]
    exit_evaluation_rows = [
        dict(item)
        for item in evaluation_label_payloads.get("exit_training_labels", [])
        if isinstance(item, dict)
    ]

    dependencies.emit_progress_fn(
        "ranking",
        (
            "开始训练轻量排序模型: "
            f"leader_fit={len(leader_fit_rows)}, "
            f"leader_eval={len(leader_evaluation_rows)}, "
            f"exit_fit={len(exit_fit_rows)}, "
            f"exit_eval={len(exit_evaluation_rows)}"
        ),
    )
    signal_started = time.perf_counter()
    signal_payloads = dependencies.build_signal_training_artifacts_fn(
        leader_fit_rows=leader_fit_rows,
        leader_evaluation_rows=leader_evaluation_rows,
        exit_fit_rows=exit_fit_rows,
        exit_evaluation_rows=exit_evaluation_rows,
        l2=signal_l2,
    )
    dependencies.emit_progress_fn(
        "ranking",
        f"轻量排序模型训练完成: elapsed={time.perf_counter() - signal_started:.1f}s",
    )

    evaluation_window_start, evaluation_window_end = _trajectory_window(evaluation_trajectory)
    result = V2RankingResearchResult(
        strategy_id=strategy_id,
        forecast_backend=forecast_backend,
        retrain_days=int(retrain_days),
        training_window_days=training_window_days,
        split_mode=split_mode,
        embargo_days=int(embargo_days),
        fit_scope=fit_scope,
        evaluation_scope=evaluation_scope,
        top_k=int(top_k),
        candidate_limit=int(candidate_limit),
        leader_min_theme_size=int(leader_min_theme_size),
        exit_min_theme_size=int(exit_min_theme_size),
        exit_candidate_limit=int(exit_candidate_limit),
        signal_l2=float(signal_l2),
        trajectory_steps=int(trajectory_steps),
        fit_steps=int(dependencies.trajectory_step_count_fn(fit_trajectory)),
        evaluation_steps=int(dependencies.trajectory_step_count_fn(evaluation_trajectory)),
        evaluation_window_start=evaluation_window_start,
        evaluation_window_end=evaluation_window_end,
        latest_state_date=_last_state_date(evaluation_trajectory),
        leader_manifest=dict(leader_payloads.get("leader_manifest", {})),
        leader_candidates=[
            dict(item)
            for item in leader_payloads.get("leader_candidates", [])
            if isinstance(item, dict)
        ],
        full_training_label_manifest=dict(full_label_payloads.get("training_label_manifest", {})),
        fit_training_label_manifest=dict(fit_label_payloads.get("training_label_manifest", {})),
        evaluation_training_label_manifest=dict(evaluation_label_payloads.get("training_label_manifest", {})),
        signal_training_manifest=dict(signal_payloads.get("signal_training_manifest", {})),
        leader_rank_model=dict(signal_payloads.get("leader_rank_model", {})),
        exit_behavior_model=dict(signal_payloads.get("exit_behavior_model", {})),
    )
    try:
        _store_cached_ranking_result(
            cache_root=cache_root,
            cache_key=cache_key,
            result=result,
        )
        dependencies.emit_progress_fn("cache", f"ranking research 缓存已写入: key={cache_key}")
    except Exception:
        pass
    dependencies.emit_progress_fn("ranking", "排序研究完成")
    return result


def persist_v2_ranking_research_artifacts(
    result: V2RankingResearchResult,
    *,
    artifact_root: str = "artifacts/v2",
) -> dict[str, str]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(str(artifact_root)) / result.strategy_id / "ranking_research" / run_id
    manifest_path = run_dir / "ranking_research_manifest.json"
    candidates_path = run_dir / "leader_candidates.json"
    leader_model_path = run_dir / "leader_rank_model.json"
    exit_model_path = run_dir / "exit_behavior_model.json"

    manifest_payload = result.summary()
    manifest_payload["result"] = result.to_payload()
    _json_dump(manifest_path, manifest_payload)
    _json_dump(candidates_path, result.leader_candidates)
    _json_dump(leader_model_path, result.leader_rank_model)
    _json_dump(exit_model_path, result.exit_behavior_model)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "leader_candidates_path": str(candidates_path),
        "leader_rank_model_path": str(leader_model_path),
        "exit_behavior_model_path": str(exit_model_path),
    }
