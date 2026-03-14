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
    info_shadow_only: bool | None = None
    info_types: str | None = None
    info_source_mode: str | None = None
    info_subsets: str | None = None
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
            "info_shadow_only": self.info_shadow_only,
            "info_types": self.info_types,
            "info_source_mode": self.info_source_mode,
            "info_subsets": self.info_subsets,
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
    forecast_backend: str = "linear"
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
            info_shadow_only=args.info_shadow_only,
            info_types=args.info_types,
            info_source_mode=args.info_source_mode,
            info_subsets=args.info_subsets,
            external_signals=args.external_signals,
            event_file=args.event_file,
            capital_flow_file=args.capital_flow_file,
            macro_file=args.macro_file,
            use_us_index_context=args.use_us_index_context,
            us_index_source=args.us_index_source,
            artifact_root=str(args.artifact_root),
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            forecast_backend=str(args.forecast_backend),
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
                "forecast_backend": self.forecast_backend,
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
                "forecast_backend": self.forecast_backend,
                "publish_forecast_models": self.publish_forecast_models,
                "split_mode": self.split_mode,
                "embargo_days": self.embargo_days,
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
            info_shadow_only=args.info_shadow_only,
            info_types=args.info_types,
            info_source_mode=args.info_source_mode,
            info_subsets=args.info_subsets,
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
            }
        )
        return payload


@dataclass(frozen=True)
class ResearchMatrixOptions:
    strategy_id: str = "swing_v2"
    config_path: str = "config/api.json"
    source: str | None = None
    artifact_root: str = "artifacts/v2"
    cache_root: str = "artifacts/v2/cache"
    refresh_cache: bool = False
    forecast_backend: str = "linear"
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
            artifact_root=str(args.artifact_root),
            cache_root=str(args.cache_root),
            refresh_cache=bool(args.refresh_cache),
            forecast_backend=str(args.forecast_backend),
            split_mode=str(args.split_mode),
            embargo_days=int(args.embargo_days),
            universe_tiers=tuple(str(item) for item in args.tiers),
        )

    def workflow_kwargs(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "config_path": self.config_path,
            "source": self.source,
            "artifact_root": self.artifact_root,
            "cache_root": self.cache_root,
            "refresh_cache": self.refresh_cache,
            "forecast_backend": self.forecast_backend,
            "split_mode": self.split_mode,
            "embargo_days": self.embargo_days,
            "universe_tiers": list(self.universe_tiers),
        }
