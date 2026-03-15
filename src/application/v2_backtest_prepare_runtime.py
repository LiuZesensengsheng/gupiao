from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class PreparedV2BacktestData:
    settings: dict[str, object]
    market_valid: pd.DataFrame
    market_feature_cols: list[str]
    panel: pd.DataFrame
    feature_cols: list[str]
    stock_frames: dict[str, pd.DataFrame]
    dates: list[pd.Timestamp]


@dataclass(frozen=True)
class TrajectoryStep:
    date: pd.Timestamp
    next_date: pd.Timestamp
    composite_state: object
    stock_states: list[object]
    horizon_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class BacktestTrajectory:
    prepared: PreparedV2BacktestData | object
    steps: list[TrajectoryStep] | list[object]


@dataclass(frozen=True)
class BacktestPrepareDependencies:
    load_v2_runtime_settings: Callable[..., dict[str, object]]
    resolve_v2_universe_settings: Callable[..., dict[str, object]]
    build_prepared_backtest_cache_key: Callable[[dict[str, object]], str]
    prepared_backtest_cache_path: Callable[..., Path]
    load_pickle_cache: Callable[[object], object]
    store_pickle_cache: Callable[[object, object], object]
    emit_progress: Callable[[str, str], None]
    load_watchlist: Callable[[str], tuple[object, object, object]]
    build_candidate_universe: Callable[..., object]
    load_symbol_daily: Callable[..., pd.DataFrame]
    make_market_feature_frame: Callable[[pd.DataFrame], pd.DataFrame]
    build_market_context_features: Callable[..., object]
    build_stock_panel_dataset: Callable[..., object]
    market_feature_columns: list[str]
    make_forecast_backend: Callable[[str | None], object]
    prepare_v2_backtest_data: Callable[..., object]
    build_v2_backtest_trajectory_from_prepared: Callable[..., object]
    parse_boolish: Callable[[object, bool], bool]


def trajectory_cache_key(
    *,
    config_path: str,
    source: str | None,
    universe_file: str | None,
    universe_limit: int | None,
    universe_tier: str | None,
    retrain_days: int,
    forecast_backend: str,
    use_us_index_context: bool,
    us_index_source: str,
    use_us_sector_etf_context: bool,
    use_cn_etf_context: bool,
    cn_etf_source: str,
) -> str:
    payload = {
        "version": "v2-trajectory-cache-2",
        "config_path": str(Path(config_path).resolve()),
        "source": "" if source is None else str(source),
        "universe_file": "" if universe_file is None else str(Path(universe_file).resolve()),
        "universe_limit": -1 if universe_limit is None else int(universe_limit),
        "universe_tier": "" if universe_tier is None else str(universe_tier),
        "retrain_days": int(retrain_days),
        "forecast_backend": str(forecast_backend),
        "use_us_index_context": bool(use_us_index_context),
        "us_index_source": str(us_index_source),
        "use_us_sector_etf_context": bool(use_us_sector_etf_context),
        "use_cn_etf_context": bool(use_cn_etf_context),
        "cn_etf_source": str(cn_etf_source),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def trajectory_cache_path(
    *,
    cache_root: str,
    cache_key: str,
) -> Path:
    root = Path(str(cache_root))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{cache_key}.pkl"


def slice_backtest_trajectory(
    trajectory: object,
    *,
    start: int,
    end: int,
) -> object:
    steps = list(getattr(trajectory, "steps"))
    n_steps = len(steps)
    lo = max(0, min(int(start), n_steps))
    hi = max(lo, min(int(end), n_steps))
    return type(trajectory)(
        prepared=getattr(trajectory, "prepared"),
        steps=list(steps[lo:hi]),
    )


def split_research_trajectory(
    trajectory: object,
    split_mode: str = "purged_wf",
    embargo_days: int = 20,
) -> tuple[object, object, object]:
    mode = str(split_mode).strip().lower() or "purged_wf"
    n_steps = len(getattr(trajectory, "steps"))
    if n_steps <= 2:
        empty = slice_backtest_trajectory(trajectory, start=0, end=0)
        holdout = slice_backtest_trajectory(trajectory, start=0, end=n_steps)
        return empty, empty, holdout

    if mode in {"simple", "legacy"}:
        train_end = max(1, int(n_steps * 0.60))
        remaining = max(2, n_steps - train_end)
        validation_len = max(1, remaining // 2)
        holdout_start = min(n_steps - 1, train_end + validation_len)
        if holdout_start <= train_end:
            holdout_start = min(n_steps - 1, train_end + 1)
        if holdout_start >= n_steps:
            holdout_start = max(1, n_steps - 1)
        validation = slice_backtest_trajectory(trajectory, start=train_end, end=holdout_start)
        holdout = slice_backtest_trajectory(trajectory, start=holdout_start, end=n_steps)
        train = slice_backtest_trajectory(trajectory, start=0, end=train_end)
        if not getattr(validation, "steps"):
            validation = slice_backtest_trajectory(trajectory, start=max(0, holdout_start - 1), end=holdout_start)
        if not getattr(holdout, "steps"):
            holdout = slice_backtest_trajectory(trajectory, start=max(0, n_steps - 1), end=n_steps)
        return train, validation, holdout

    if mode != "purged_wf":
        raise ValueError(f"Unsupported split mode: {split_mode}")

    embargo_steps = max(0, int(embargo_days))
    train_end = max(1, int(n_steps * 0.60))
    validation_end_target = max(train_end + 1, int(n_steps * 0.80))
    validation_start = min(n_steps, train_end + embargo_steps)
    validation_end = min(n_steps, validation_end_target)
    holdout_start = min(n_steps, validation_end + embargo_steps)

    if validation_start >= validation_end:
        validation_start = min(n_steps - 2, train_end)
        validation_end = min(n_steps - 1, max(validation_start + 1, validation_end_target))
    if holdout_start <= validation_end:
        holdout_start = min(n_steps - 1, validation_end + 1)
    if holdout_start >= n_steps:
        holdout_start = max(validation_end, n_steps - 1)

    train = slice_backtest_trajectory(trajectory, start=0, end=train_end)
    validation = slice_backtest_trajectory(trajectory, start=validation_start, end=validation_end)
    holdout = slice_backtest_trajectory(trajectory, start=holdout_start, end=n_steps)
    if not getattr(validation, "steps"):
        validation = slice_backtest_trajectory(
            trajectory,
            start=max(0, holdout_start - 1),
            end=max(holdout_start, 1),
        )
    if not getattr(holdout, "steps"):
        holdout = slice_backtest_trajectory(trajectory, start=max(0, n_steps - 1), end=n_steps)
    return train, validation, holdout


def prepare_v2_backtest_data(
    *,
    config_path: str,
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    prepared_dataclass: type | None = None,
    deps: BacktestPrepareDependencies,
) -> object | None:
    settings = deps.load_v2_runtime_settings(
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
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    settings["refresh_cache"] = bool(refresh_cache)
    settings = deps.resolve_v2_universe_settings(settings=settings, cache_root=cache_root)
    prepared_cache_key = deps.build_prepared_backtest_cache_key(settings)
    prepared_cache_path = deps.prepared_backtest_cache_path(
        cache_root=cache_root,
        cache_key=prepared_cache_key,
    )
    if not refresh_cache:
        cached_prepared = deps.load_pickle_cache(prepared_cache_path)
        if cached_prepared is not None:
            deps.emit_progress("cache", "命中 prepared data 缓存")
            return cached_prepared

    market_security, _, _ = deps.load_watchlist(str(settings["watchlist"]))
    universe = deps.build_candidate_universe(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    stocks = universe.rows
    if not stocks:
        return None

    market_raw = deps.load_symbol_daily(
        symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_feat_base = deps.make_market_feature_frame(market_raw)
    market_context = deps.build_market_context_features(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_dates=market_feat_base["date"],
        use_margin_features=bool(settings["use_margin_features"]),
        margin_market_file=str(settings["margin_market_file"]),
        use_us_index_context=bool(settings.get("use_us_index_context", False)),
        us_index_source=str(settings.get("us_index_source", "akshare")),
        use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
        use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
        cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
    )
    market_frame = market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")
    market_feature_cols = list(deps.market_feature_columns) + list(market_context.feature_columns)
    market_valid = market_frame.dropna(
        subset=market_feature_cols + [
            "mkt_target_1d_up",
            "mkt_target_2d_up",
            "mkt_target_3d_up",
            "mkt_target_5d_up",
            "mkt_target_20d_up",
        ]
    ).sort_values("date").copy()
    if market_valid.empty:
        return None

    panel_bundle = deps.build_stock_panel_dataset(
        stock_securities=stocks,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_frame=market_frame,
        extra_market_cols=list(market_context.feature_columns),
        use_margin_features=bool(settings["use_margin_features"]),
        margin_stock_file=str(settings["margin_stock_file"]),
    )
    panel = panel_bundle.frame
    feature_cols = list(panel_bundle.feature_columns)
    if panel.empty or not feature_cols:
        return None

    stock_frames = {
        str(symbol): frame.sort_values("date").copy()
        for symbol, frame in panel.groupby("symbol", observed=True)
    }
    common_dates = set(pd.to_datetime(market_valid["date"])) & set(pd.to_datetime(panel["date"]))
    dates = sorted(pd.Timestamp(d) for d in common_dates)
    min_train_days = int(settings["min_train_days"])
    if len(dates) <= min_train_days + 1:
        return None

    if prepared_dataclass is None:
        raise ValueError("prepared_dataclass is required")
    prepared = prepared_dataclass(
        settings=settings,
        market_valid=market_valid,
        market_feature_cols=market_feature_cols,
        panel=panel,
        feature_cols=feature_cols,
        stock_frames=stock_frames,
        dates=dates,
    )
    try:
        deps.store_pickle_cache(prepared_cache_path, prepared)
        deps.emit_progress("cache", "prepared data 缓存已写入")
    except Exception:
        pass
    return prepared


def build_v2_backtest_trajectory_from_prepared(
    prepared: object,
    *,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
    deps: BacktestPrepareDependencies,
) -> object:
    backend = deps.make_forecast_backend(forecast_backend)
    return backend.build_trajectory(prepared, retrain_days=retrain_days)


def load_or_build_v2_backtest_trajectory(
    *,
    config_path: str,
    source: str | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    retrain_days: int = 20,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    deps: BacktestPrepareDependencies,
) -> object | None:
    backend = deps.make_forecast_backend(forecast_backend)
    settings = deps.load_v2_runtime_settings(
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
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    settings["refresh_cache"] = bool(refresh_cache)
    settings = deps.resolve_v2_universe_settings(settings=settings, cache_root=cache_root)
    cache_key = trajectory_cache_key(
        config_path=str(settings.get("config_path", config_path)),
        source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
        universe_file=str(settings.get("universe_file", universe_file))
        if settings.get("universe_file", universe_file) is not None
        else None,
        universe_limit=int(settings.get("universe_limit")) if settings.get("universe_limit") is not None else universe_limit,
        universe_tier=str(settings.get("universe_tier", universe_tier)),
        retrain_days=retrain_days,
        forecast_backend=backend.name,
        use_us_index_context=bool(settings.get("use_us_index_context", False)),
        us_index_source=str(settings.get("us_index_source", "akshare")),
        use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
        use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
        cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
    )
    cache_path = trajectory_cache_path(cache_root=cache_root, cache_key=cache_key)
    if not refresh_cache and cache_path.exists():
        deps.emit_progress("cache", f"命中轨迹缓存: backend={backend.name}")
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if cached is not None:
                return cached
        except Exception:
            pass
    else:
        deps.emit_progress("cache", f"轨迹缓存未命中: backend={backend.name}，准备重建")

    deps.emit_progress("research", "开始准备研究数据")
    prepared = deps.prepare_v2_backtest_data(
        config_path=str(settings.get("config_path", config_path)),
        source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
        universe_file=str(settings.get("universe_file", universe_file))
        if settings.get("universe_file", universe_file) is not None
        else None,
        universe_limit=int(settings.get("universe_limit")) if settings.get("universe_limit") is not None else universe_limit,
        universe_tier=str(settings.get("universe_tier", universe_tier)),
        dynamic_universe=deps.parse_boolish(settings.get("dynamic_universe_enabled", False), False),
        generator_target_size=int(settings.get("generator_target_size", settings.get("universe_limit", 0)) or 0),
        generator_coarse_size=int(settings.get("generator_coarse_size", 0) or 0),
        generator_theme_aware=deps.parse_boolish(settings.get("generator_theme_aware", True), True),
        generator_use_concepts=deps.parse_boolish(settings.get("generator_use_concepts", True), True),
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        use_us_index_context=bool(settings.get("use_us_index_context", False)),
        us_index_source=str(settings.get("us_index_source", "akshare")),
    )
    if prepared is None:
        return None
    deps.emit_progress("research", "开始构建预测轨迹")
    trajectory = deps.build_v2_backtest_trajectory_from_prepared(
        prepared,
        retrain_days=retrain_days,
        forecast_backend=backend.name,
    )
    try:
        with cache_path.open("wb") as f:
            pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
        deps.emit_progress("cache", f"轨迹缓存已写入: backend={backend.name}")
    except Exception:
        pass
    return trajectory
