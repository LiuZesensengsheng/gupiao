from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from typing import Any, Mapping, Self


def _filtered_kwargs(cls: type[object], values: Mapping[str, object]) -> dict[str, object]:
    names = {field.name for field in fields(cls)}
    return {name: values[name] for name in names if name in values}


@dataclass(frozen=True)
class _BaseRuntimeOptions:
    strategy_id: str = "swing_v2"
    config_path: str = "config/api.json"
    source: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    lookback_years: int | None = None
    universe_tier: str | None = None
    universe_file: str | None = None
    universe_limit: int | None = None
    dynamic_universe: bool | None = None
    generator_target_size: int | None = None
    generator_coarse_size: int | None = None
    generator_theme_aware: bool | None = None
    generator_use_concepts: bool | None = None
    info_file: str | None = None
    info_lookback_days: int | None = None
    info_half_life_days: float | None = None
    use_info_fusion: bool | None = None
    use_learned_info_fusion: bool | None = None
    info_shadow_only: bool | None = None
    info_types: str | None = None
    info_source_mode: str | None = None
    info_subsets: str | None = None
    info_cutoff_time: str | None = None
    external_signals: bool | None = None
    event_file: str | None = None
    capital_flow_file: str | None = None
    macro_file: str | None = None
    use_us_index_context: bool | None = None
    us_index_source: str | None = None

    @classmethod
    def from_kwargs(cls, **kwargs: object) -> Self:
        return cls(**_filtered_kwargs(cls, kwargs))

    def common_kwargs(self) -> dict[str, object]:
        return {
            "strategy_id": self.strategy_id,
            "config_path": self.config_path,
            "source": self.source,
            "universe_tier": self.universe_tier,
            "universe_file": self.universe_file,
            "universe_limit": self.universe_limit,
            "dynamic_universe": self.dynamic_universe,
            "generator_target_size": self.generator_target_size,
            "generator_coarse_size": self.generator_coarse_size,
            "generator_theme_aware": self.generator_theme_aware,
            "generator_use_concepts": self.generator_use_concepts,
            "info_file": self.info_file,
            "info_lookback_days": self.info_lookback_days,
            "info_half_life_days": self.info_half_life_days,
            "use_info_fusion": self.use_info_fusion,
            "use_learned_info_fusion": self.use_learned_info_fusion,
            "info_shadow_only": self.info_shadow_only,
            "info_types": self.info_types,
            "info_source_mode": self.info_source_mode,
            "info_subsets": self.info_subsets,
            "info_cutoff_time": self.info_cutoff_time,
            "external_signals": self.external_signals,
            "event_file": self.event_file,
            "capital_flow_file": self.capital_flow_file,
            "macro_file": self.macro_file,
            "use_us_index_context": self.use_us_index_context,
            "us_index_source": self.us_index_source,
        }


@dataclass(frozen=True)
class ResearchRunOptions(_BaseRuntimeOptions):
    artifact_root: str = "artifacts/v2"
    cache_root: str = "artifacts/v2/cache"
    refresh_cache: bool = False
    retrain_days: int = 20
    forecast_backend: str = "linear"
    training_window_days: int | None = 480
    skip_calibration: bool = False
    skip_learning: bool = False
    split_mode: str = "purged_wf"
    embargo_days: int = 20
    publish_forecast_models: bool = True

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> Self:
        return cls(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            start_date=getattr(args, "start_date", None),
            end_date=getattr(args, "end_date", None),
            lookback_years=getattr(args, "lookback_years", None),
            universe_tier=args.universe_tier,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
            dynamic_universe=args.dynamic_universe,
            generator_target_size=args.generator_target_size,
            generator_coarse_size=args.generator_coarse_size,
            generator_theme_aware=args.generator_theme_aware,
            generator_use_concepts=args.generator_use_concepts,
            info_file=args.info_file,
            info_lookback_days=args.info_lookback_days,
            info_half_life_days=args.info_half_life_days,
            use_info_fusion=args.use_info_fusion,
            use_learned_info_fusion=getattr(args, "use_learned_info_fusion", None),
            info_shadow_only=args.info_shadow_only,
            info_types=args.info_types,
            info_source_mode=args.info_source_mode,
            info_subsets=args.info_subsets,
            info_cutoff_time=getattr(args, "info_cutoff_time", None),
            external_signals=args.external_signals,
            event_file=args.event_file,
            capital_flow_file=args.capital_flow_file,
            macro_file=args.macro_file,
            use_us_index_context=args.use_us_index_context,
            us_index_source=args.us_index_source,
            artifact_root=str(args.artifact_root),
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            retrain_days=int(args.retrain_days),
            forecast_backend=str(args.forecast_backend),
            training_window_days=args.training_window_days,
            skip_calibration=bool(getattr(args, "light", False) or args.skip_calibration),
            skip_learning=bool(getattr(args, "light", False) or args.skip_learning),
            split_mode=str(args.split_mode),
            embargo_days=int(args.embargo_days),
            publish_forecast_models=bool(args.publish_forecast_models),
        )

    def workflow_kwargs(self) -> dict[str, object]:
        payload = self.common_kwargs()
        payload.update(
            {
                "cache_root": self.cache_root,
                "refresh_cache": self.refresh_cache,
                "retrain_days": self.retrain_days,
                "forecast_backend": self.forecast_backend,
                "training_window_days": self.training_window_days,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "lookback_years": self.lookback_years,
                "skip_calibration": self.skip_calibration,
                "skip_learning": self.skip_learning,
                "split_mode": self.split_mode,
                "embargo_days": self.embargo_days,
            }
        )
        return payload

    def publish_kwargs(self) -> dict[str, object]:
        payload = self.common_kwargs()
        payload.update(
            {
                "artifact_root": self.artifact_root,
                "cache_root": self.cache_root,
                "retrain_days": self.retrain_days,
                "forecast_backend": self.forecast_backend,
                "training_window_days": self.training_window_days,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "lookback_years": self.lookback_years,
                "publish_forecast_models": self.publish_forecast_models,
                "split_mode": self.split_mode,
                "embargo_days": self.embargo_days,
            }
        )
        return payload


@dataclass(frozen=True)
class RankingResearchOptions(_BaseRuntimeOptions):
    artifact_root: str = "artifacts/v2"
    cache_root: str = "artifacts/v2/cache"
    refresh_cache: bool = False
    retrain_days: int = 20
    forecast_backend: str = "linear"
    training_window_days: int | None = 480
    split_mode: str = "purged_wf"
    embargo_days: int = 20
    top_k: int = 3
    candidate_limit: int = 16
    leader_min_theme_size: int = 3
    exit_min_theme_size: int = 2
    exit_candidate_limit: int = 8
    signal_l2: float = 1.0

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> Self:
        return cls(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            start_date=getattr(args, "start_date", None),
            end_date=getattr(args, "end_date", None),
            lookback_years=getattr(args, "lookback_years", None),
            universe_tier=args.universe_tier,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
            dynamic_universe=args.dynamic_universe,
            generator_target_size=args.generator_target_size,
            generator_coarse_size=args.generator_coarse_size,
            generator_theme_aware=args.generator_theme_aware,
            generator_use_concepts=args.generator_use_concepts,
            info_file=args.info_file,
            info_lookback_days=args.info_lookback_days,
            info_half_life_days=args.info_half_life_days,
            use_info_fusion=args.use_info_fusion,
            use_learned_info_fusion=getattr(args, "use_learned_info_fusion", None),
            info_shadow_only=args.info_shadow_only,
            info_types=args.info_types,
            info_source_mode=args.info_source_mode,
            info_subsets=args.info_subsets,
            info_cutoff_time=getattr(args, "info_cutoff_time", None),
            external_signals=args.external_signals,
            event_file=args.event_file,
            capital_flow_file=args.capital_flow_file,
            macro_file=args.macro_file,
            use_us_index_context=args.use_us_index_context,
            us_index_source=args.us_index_source,
            artifact_root=str(args.artifact_root),
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            retrain_days=int(args.retrain_days),
            forecast_backend=str(args.forecast_backend),
            training_window_days=args.training_window_days,
            split_mode=str(args.split_mode),
            embargo_days=int(args.embargo_days),
            top_k=int(args.top_k),
            candidate_limit=int(args.candidate_limit),
            leader_min_theme_size=int(args.leader_min_theme_size),
            exit_min_theme_size=int(args.exit_min_theme_size),
            exit_candidate_limit=int(args.exit_candidate_limit),
            signal_l2=float(args.signal_l2),
        )

    def workflow_kwargs(self) -> dict[str, object]:
        payload = self.common_kwargs()
        payload.update(
            {
                "cache_root": self.cache_root,
                "refresh_cache": self.refresh_cache,
                "retrain_days": self.retrain_days,
                "forecast_backend": self.forecast_backend,
                "training_window_days": self.training_window_days,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "lookback_years": self.lookback_years,
                "split_mode": self.split_mode,
                "embargo_days": self.embargo_days,
                "top_k": self.top_k,
                "candidate_limit": self.candidate_limit,
                "leader_min_theme_size": self.leader_min_theme_size,
                "exit_min_theme_size": self.exit_min_theme_size,
                "exit_candidate_limit": self.exit_candidate_limit,
                "signal_l2": self.signal_l2,
            }
        )
        return payload


@dataclass(frozen=True)
class DailyRunOptions(_BaseRuntimeOptions):
    artifact_root: str = "artifacts/v2"
    cache_root: str = "artifacts/v2/cache"
    refresh_cache: bool = False
    run_id: str | None = None
    snapshot_path: str | None = None
    allow_retrain: bool = False
    disable_learned_policy: bool = False

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> Self:
        return cls(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            universe_tier=args.universe_tier,
            universe_file=args.universe_file,
            universe_limit=args.universe_limit,
            dynamic_universe=args.dynamic_universe,
            generator_target_size=args.generator_target_size,
            generator_coarse_size=args.generator_coarse_size,
            generator_theme_aware=args.generator_theme_aware,
            generator_use_concepts=args.generator_use_concepts,
            info_file=args.info_file,
            info_lookback_days=args.info_lookback_days,
            info_half_life_days=args.info_half_life_days,
            use_info_fusion=args.use_info_fusion,
            use_learned_info_fusion=getattr(args, "use_learned_info_fusion", None),
            info_shadow_only=args.info_shadow_only,
            info_types=args.info_types,
            info_source_mode=args.info_source_mode,
            info_subsets=args.info_subsets,
            info_cutoff_time=getattr(args, "info_cutoff_time", None),
            external_signals=args.external_signals,
            event_file=args.event_file,
            capital_flow_file=args.capital_flow_file,
            macro_file=args.macro_file,
            use_us_index_context=args.use_us_index_context,
            us_index_source=args.us_index_source,
            artifact_root=str(args.artifact_root),
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            run_id=args.run_id,
            snapshot_path=args.snapshot_path,
            allow_retrain=bool(args.allow_retrain),
            disable_learned_policy=bool(getattr(args, "disable_learned_policy", False)),
        )

    def workflow_kwargs(self) -> dict[str, object]:
        payload = self.common_kwargs()
        payload.update(
            {
                "artifact_root": self.artifact_root,
                "cache_root": self.cache_root,
                "refresh_cache": self.refresh_cache,
                "run_id": self.run_id,
                "snapshot_path": self.snapshot_path,
                "allow_retrain": self.allow_retrain,
                "disable_learned_policy": self.disable_learned_policy,
            }
        )
        return payload


@dataclass(frozen=True)
class ResearchMatrixOptions:
    strategy_id: str = "swing_v2"
    config_path: str = "config/api.json"
    source: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    lookback_years: int | None = None
    artifact_root: str = "artifacts/v2"
    cache_root: str = "artifacts/v2/cache"
    refresh_cache: bool = False
    retrain_days: int = 20
    forecast_backend: str = "linear"
    training_window_days: int | None = 480
    split_mode: str = "purged_wf"
    embargo_days: int = 20
    universe_tiers: tuple[str, ...] = ("favorites_16", "generated_80", "generated_150", "generated_300")

    @classmethod
    def from_kwargs(cls, **kwargs: object) -> Self:
        values = _filtered_kwargs(cls, kwargs)
        if "universe_tiers" in values and isinstance(values["universe_tiers"], list):
            values["universe_tiers"] = tuple(str(item) for item in values["universe_tiers"])
        return cls(**values)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> Self:
        return cls(
            strategy_id=str(args.strategy),
            config_path=str(args.config),
            source=args.source,
            start_date=getattr(args, "start_date", None),
            end_date=getattr(args, "end_date", None),
            lookback_years=getattr(args, "lookback_years", None),
            artifact_root=str(args.artifact_root),
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            retrain_days=int(args.retrain_days),
            forecast_backend=str(args.forecast_backend),
            training_window_days=args.training_window_days,
            split_mode=str(args.split_mode),
            embargo_days=int(args.embargo_days),
            universe_tiers=tuple(str(item) for item in args.tiers),
        )

    def workflow_kwargs(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "config_path": self.config_path,
            "source": self.source,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "lookback_years": self.lookback_years,
            "artifact_root": self.artifact_root,
            "cache_root": self.cache_root,
            "refresh_cache": self.refresh_cache,
            "retrain_days": self.retrain_days,
            "forecast_backend": self.forecast_backend,
            "training_window_days": self.training_window_days,
            "split_mode": self.split_mode,
            "embargo_days": self.embargo_days,
            "universe_tiers": list(self.universe_tiers),
        }
