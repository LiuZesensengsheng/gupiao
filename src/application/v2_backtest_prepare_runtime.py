from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, replace
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
    load_v2_info_items_for_date: Callable[..., tuple[str, list[object]]]
    enrich_state_with_info: Callable[..., object]
    attach_external_signals_to_composite_state: Callable[..., tuple[object, dict[str, object]]]
    attach_insight_memory_to_state: Callable[..., object]
    apply_leader_candidate_overlay: Callable[..., object]
    resolve_latest_leader_rank_model: Callable[..., dict[str, object] | None]
    sha256_file: Callable[[object], str]


def _merge_info_overlay_state(
    state: object,
    overlay_state: object | None,
) -> object:
    if overlay_state is None:
        return state
    try:
        return replace(
            state,
            market_info_state=getattr(overlay_state, "market_info_state", getattr(state, "market_info_state", None)),
            sector_info_states=dict(getattr(overlay_state, "sector_info_states", {}) or {}),
            stock_info_states=dict(getattr(overlay_state, "stock_info_states", {}) or {}),
        )
    except Exception:
        return state


def trajectory_cache_key(
    *,
    raw_cache_key: str,
    overlay_cache_token: str = "",
) -> str:
    payload = {
        "version": "v2-trajectory-cache-10",
        "raw_cache_key": str(raw_cache_key),
        "overlay_cache_token": str(overlay_cache_token),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def raw_trajectory_cache_key(
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
    training_window_days: int | None,
    start: str = "",
    end: str = "",
) -> str:
    payload = {
        "version": "v2-trajectory-raw-cache-4",
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
        "training_window_days": None if training_window_days is None else int(training_window_days),
        "start": str(start or ""),
        "end": str(end or ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _normalized_path(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return text


def _hashed_path(value: object, *, deps: BacktestPrepareDependencies) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(deps.sha256_file(text))
    except Exception:
        return ""


def _research_overlay_cache_token(
    *,
    settings: dict[str, object],
    deps: BacktestPrepareDependencies,
) -> str:
    leader_rank_model = deps.resolve_latest_leader_rank_model()
    payload = {
        "use_info_fusion": bool(settings.get("use_info_fusion", False)),
        "use_learned_info_fusion": bool(settings.get("use_learned_info_fusion", settings.get("use_learned_news_fusion", False))),
        "external_signals": bool(settings.get("external_signals", False)),
        "enable_insight_memory": bool(settings.get("enable_insight_memory", False)),
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "info_types": [str(item) for item in settings.get("info_types", [])],
        "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
        "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
        "info_lookback_days": int(settings.get("info_lookback_days", 45)),
        "event_lookback_days": int(settings.get("event_lookback_days", settings.get("info_lookback_days", 45))),
        "capital_flow_lookback_days": int(settings.get("capital_flow_lookback_days", 20)),
        "macro_lookback_days": int(settings.get("macro_lookback_days", 60)),
        "info_half_life_days": float(settings.get("info_half_life_days", 10.0)),
        "info_cutoff_time": str(settings.get("info_cutoff_time", "23:59:59")),
        "external_signal_version": str(settings.get("external_signal_version", "v1")),
        "info_file": _normalized_path(settings.get("info_file", "")),
        "info_file_hash": _hashed_path(settings.get("info_file", ""), deps=deps),
        "event_file": _normalized_path(settings.get("event_file", "")),
        "event_file_hash": _hashed_path(settings.get("event_file", ""), deps=deps),
        "news_file": _normalized_path(settings.get("news_file", "")),
        "news_file_hash": _hashed_path(settings.get("news_file", ""), deps=deps),
        "capital_flow_file": _normalized_path(settings.get("capital_flow_file", "")),
        "capital_flow_file_hash": _hashed_path(settings.get("capital_flow_file", ""), deps=deps),
        "macro_file": _normalized_path(settings.get("macro_file", "")),
        "macro_file_hash": _hashed_path(settings.get("macro_file", ""), deps=deps),
        "insight_notes_dir": _normalized_path(settings.get("insight_notes_dir", "")),
        "insight_notes_hash": _hashed_path(settings.get("insight_notes_dir", ""), deps=deps),
        "leader_rank_model_hash": hashlib.sha256(
            json.dumps(leader_rank_model or {}, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:24],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _step_as_of_date(step: object) -> pd.Timestamp:
    state = getattr(step, "composite_state", None)
    market = getattr(state, "market", None)
    market_as_of = getattr(market, "as_of_date", "")
    try:
        if str(market_as_of).strip():
            return pd.Timestamp(market_as_of).normalize()
    except Exception:
        pass
    return pd.Timestamp(getattr(step, "date")).normalize()


def _replace_trajectory_step(
    step: object,
    *,
    composite_state: object,
    stock_states: list[object],
) -> object:
    try:
        return replace(
            step,
            composite_state=composite_state,
            stock_states=stock_states,
        )
    except Exception:
        return type(step)(
            date=getattr(step, "date"),
            next_date=getattr(step, "next_date"),
            composite_state=composite_state,
            stock_states=stock_states,
            horizon_metrics=getattr(step, "horizon_metrics"),
        )


def _replace_trajectory_steps(
    trajectory: object,
    *,
    steps: list[object],
) -> object:
    try:
        return replace(trajectory, steps=steps)
    except Exception:
        prepared = getattr(trajectory, "prepared", None)
        if prepared is not None:
            try:
                return type(trajectory)(
                    prepared=prepared,
                    steps=steps,
                )
            except Exception:
                pass
        try:
            cloned = type(trajectory)()
            if prepared is not None:
                setattr(cloned, "prepared", prepared)
            setattr(cloned, "steps", steps)
            return cloned
        except Exception:
            setattr(trajectory, "steps", steps)
            return trajectory


def _attach_trajectory_cache_identity(
    trajectory: object | None,
    *,
    raw_cache_key: str,
    decorated_cache_key: str,
) -> object | None:
    if trajectory is None:
        return None
    for attr, value in (
        ("_raw_trajectory_cache_key", str(raw_cache_key)),
        ("_decorated_trajectory_cache_key", str(decorated_cache_key)),
    ):
        try:
            setattr(trajectory, attr, value)
        except Exception:
            try:
                object.__setattr__(trajectory, attr, value)
            except Exception:
                pass
    return trajectory


def decorate_research_trajectory(
    trajectory: object,
    *,
    settings: dict[str, object],
    deps: BacktestPrepareDependencies,
) -> object:
    steps = getattr(trajectory, "steps", None)
    if not steps:
        return trajectory

    use_info_fusion = bool(settings.get("use_info_fusion", False))
    use_learned_info_fusion = bool(settings.get("use_learned_info_fusion", settings.get("use_learned_news_fusion", False)))
    info_shadow_only = bool(settings.get("info_shadow_only", False))
    info_shadow_requested = bool(use_info_fusion or use_learned_info_fusion or info_shadow_only)
    use_external_signals = bool(settings.get("external_signals", False))
    use_insight_memory = bool(settings.get("enable_insight_memory", False))
    if not (info_shadow_requested or use_external_signals or use_insight_memory):
        return trajectory

    step_list = list(steps)
    deps.emit_progress(
        "trajectory",
        (
            "decorating research states: "
            f"steps={len(step_list)}, "
            f"info={use_info_fusion}, learned_info={use_learned_info_fusion}, shadow_only={info_shadow_only}, external={use_external_signals}, insight={use_insight_memory}"
        ),
    )
    info_items_by_date: dict[str, list[object]] = {}
    previous_roles: dict[str, object] = {}
    decorated_steps: list[object] = []
    leader_rank_model = deps.resolve_latest_leader_rank_model()

    def _load_info_items(as_of_date: pd.Timestamp) -> list[object]:
        key = str(as_of_date.date())
        cached = info_items_by_date.get(key)
        if cached is not None:
            return cached
        try:
            _, loaded = deps.load_v2_info_items_for_date(
                settings=settings,
                as_of_date=as_of_date,
                learned_window=False,
            )
        except Exception:
            loaded = []
        info_items_by_date[key] = list(loaded)
        return info_items_by_date[key]

    for idx, step in enumerate(step_list, start=1):
        state = getattr(step, "composite_state", None)
        if state is None:
            decorated_steps.append(step)
            continue
        as_of_date = _step_as_of_date(step)
        info_items = _load_info_items(as_of_date)
        decorated_state = state
        if info_shadow_requested and info_items:
            shadow_state = deps.enrich_state_with_info(
                state=decorated_state,
                as_of_date=as_of_date,
                info_items=info_items,
                settings=settings,
            )
            decorated_state = _merge_info_overlay_state(decorated_state, shadow_state)
            if use_info_fusion and not info_shadow_only:
                decorated_state = shadow_state
        if use_external_signals:
            decorated_state, _ = deps.attach_external_signals_to_composite_state(
                state=decorated_state,
                settings=settings,
                as_of_date=as_of_date,
                info_items=info_items,
            )
        if use_insight_memory:
            if previous_roles:
                try:
                    decorated_state = replace(decorated_state, stock_role_states=dict(previous_roles))
                except Exception:
                    pass
            decorated_state = deps.attach_insight_memory_to_state(
                state=decorated_state,
                settings=settings,
                as_of_date=as_of_date,
                info_items=info_items,
            )
            previous_roles = dict(getattr(decorated_state, "stock_role_states", {}) or {})
        else:
            previous_roles = {}
        decorated_state = deps.apply_leader_candidate_overlay(
            state=decorated_state,
            leader_rank_model=leader_rank_model,
        )
        decorated_steps.append(
            _replace_trajectory_step(
                step,
                composite_state=decorated_state,
                stock_states=list(getattr(decorated_state, "stocks", getattr(step, "stock_states", [])) or []),
            )
        )
        if idx == len(step_list) or (len(step_list) > 20 and idx % 25 == 0):
            deps.emit_progress(
                "trajectory",
                f"decorated research states {idx}/{len(step_list)}",
            )

    return _replace_trajectory_steps(trajectory, steps=decorated_steps)


def trajectory_cache_path(
    *,
    cache_root: str,
    cache_key: str,
    layer: str = "decorated",
) -> Path:
    root = Path(str(cache_root))
    if str(layer).strip().lower() == "raw":
        root = root / "raw_trajectory"
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
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
    info_file: str | None = None,
    info_lookback_days: int | None = None,
    info_half_life_days: float | None = None,
    use_info_fusion: bool | None = None,
    info_shadow_only: bool | None = None,
    info_types: str | None = None,
    info_source_mode: str | None = None,
    info_subsets: str | None = None,
    info_cutoff_time: str | None = None,
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    training_window_days: int | None = None,
    prepared_dataclass: type | None = None,
    deps: BacktestPrepareDependencies,
) -> object | None:
    settings = deps.load_v2_runtime_settings(
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
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        training_window_days=training_window_days,
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
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_years: int | None = None,
    universe_file: str | None = None,
    universe_limit: int | None = None,
    universe_tier: str | None = None,
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
    training_window_days: int | None = None,
    deps: BacktestPrepareDependencies,
) -> object | None:
    backend = deps.make_forecast_backend(forecast_backend)
    settings = deps.load_v2_runtime_settings(
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
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        training_window_days=training_window_days,
    )
    settings["refresh_cache"] = bool(refresh_cache)
    settings = deps.resolve_v2_universe_settings(settings=settings, cache_root=cache_root)
    raw_cache_key = raw_trajectory_cache_key(
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
        training_window_days=(
            int(settings.get("training_window_days"))
            if settings.get("training_window_days") is not None
            else None
        ),
        start=str(settings.get("start", "")),
        end=str(settings.get("end", "")),
    )
    decorated_cache_key = trajectory_cache_key(
        raw_cache_key=raw_cache_key,
        overlay_cache_token=_research_overlay_cache_token(
            settings=settings,
            deps=deps,
        ),
    )
    raw_cache_path = trajectory_cache_path(
        cache_root=cache_root,
        cache_key=raw_cache_key,
        layer="raw",
    )
    decorated_cache_path = trajectory_cache_path(
        cache_root=cache_root,
        cache_key=decorated_cache_key,
        layer="decorated",
    )
    if not refresh_cache:
        cached_trajectory = deps.load_pickle_cache(decorated_cache_path)
        if cached_trajectory is not None:
            deps.emit_progress("cache", f"命中轨迹缓存: backend={backend.name}")
            return _attach_trajectory_cache_identity(
                cached_trajectory,
                raw_cache_key=raw_cache_key,
                decorated_cache_key=decorated_cache_key,
            )
    raw_trajectory = None
    if not refresh_cache:
        raw_trajectory = deps.load_pickle_cache(raw_cache_path)
        if raw_trajectory is not None:
            deps.emit_progress("cache", f"命中 raw 轨迹缓存: backend={backend.name}")
    if raw_trajectory is None:
        deps.emit_progress("cache", f"轨迹缓存未命中: backend={backend.name}，准备重建")
        deps.emit_progress("research", "开始准备研究数据")
        prepared = deps.prepare_v2_backtest_data(
            config_path=str(settings.get("config_path", config_path)),
            source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
            start_date=str(settings.get("start", "")) or None,
            end_date=str(settings.get("end", "")) or None,
            lookback_years=(
                int(settings.get("lookback_years"))
                if settings.get("lookback_years") is not None
                else lookback_years
            ),
            universe_file=str(settings.get("universe_file", universe_file))
            if settings.get("universe_file", universe_file) is not None
            else None,
            universe_limit=int(settings.get("universe_limit")) if settings.get("universe_limit") is not None else universe_limit,
            universe_tier=str(settings.get("universe_tier", universe_tier)),
            info_file=str(settings.get("info_file", info_file)) if settings.get("info_file", info_file) is not None else None,
            info_lookback_days=(
                int(settings.get("info_lookback_days"))
                if settings.get("info_lookback_days") is not None
                else info_lookback_days
            ),
            info_half_life_days=(
                float(settings.get("info_half_life_days"))
                if settings.get("info_half_life_days") is not None
                else info_half_life_days
            ),
            use_info_fusion=deps.parse_boolish(settings.get("use_info_fusion", False), False),
            info_shadow_only=deps.parse_boolish(settings.get("info_shadow_only", True), True),
            info_types=settings.get("info_types", info_types),
            info_source_mode=str(settings.get("info_source_mode", info_source_mode or "layered")),
            info_subsets=settings.get("info_subsets", info_subsets),
            external_signals=deps.parse_boolish(settings.get("external_signals", True), True),
            event_file=str(settings.get("event_file", event_file)) if settings.get("event_file", event_file) is not None else None,
            capital_flow_file=(
                str(settings.get("capital_flow_file", capital_flow_file))
                if settings.get("capital_flow_file", capital_flow_file) is not None
                else None
            ),
            macro_file=str(settings.get("macro_file", macro_file)) if settings.get("macro_file", macro_file) is not None else None,
            dynamic_universe=deps.parse_boolish(settings.get("dynamic_universe_enabled", False), False),
            generator_target_size=int(settings.get("generator_target_size", settings.get("universe_limit", 0)) or 0),
            generator_coarse_size=int(settings.get("generator_coarse_size", 0) or 0),
            generator_theme_aware=deps.parse_boolish(settings.get("generator_theme_aware", True), True),
            generator_use_concepts=deps.parse_boolish(settings.get("generator_use_concepts", True), True),
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
            training_window_days=(
                int(settings.get("training_window_days"))
                if settings.get("training_window_days") is not None
                else training_window_days
            ),
        )
        if prepared is None:
            return None
        deps.emit_progress("research", "开始构建预测轨迹")
        raw_trajectory = deps.build_v2_backtest_trajectory_from_prepared(
            prepared,
            retrain_days=retrain_days,
            forecast_backend=backend.name,
        )
        try:
            deps.store_pickle_cache(raw_cache_path, raw_trajectory)
            deps.emit_progress("cache", f"raw 轨迹缓存已写入: backend={backend.name}")
        except Exception:
            pass
    else:
        deps.emit_progress("trajectory", "开始叠加 research overlay")
    trajectory = decorate_research_trajectory(
        raw_trajectory,
        settings=settings,
        deps=deps,
    )
    trajectory = _attach_trajectory_cache_identity(
        trajectory,
        raw_cache_key=raw_cache_key,
        decorated_cache_key=decorated_cache_key,
    )
    try:
        deps.store_pickle_cache(decorated_cache_path, trajectory)
        deps.emit_progress("cache", f"轨迹缓存已写入: backend={backend.name}")
    except Exception:
        pass
    return trajectory
