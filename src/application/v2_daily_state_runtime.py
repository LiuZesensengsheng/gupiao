from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CompositeState,
    InfoDivergenceRecord,
    InfoItem,
    InfoSignalRecord,
    LearnedPolicyModel,
    SectorForecastState,
    StrategySnapshot,
)


@dataclass(frozen=True)
class DailyUniverseContext:
    market_security: object
    current_holdings: list[object]
    stocks: list[object]
    sector_map: dict[str, str]


@dataclass(frozen=True)
class DailyStateRuntimeDependencies:
    emit_progress: Callable[[str, str], None]
    load_watchlist: Callable[[str], tuple[object, list[object], dict[str, str]]]
    build_candidate_universe: Callable[..., object]
    daily_universe_context_cls: type
    run_quant_pipeline: Callable[..., tuple[object, list[object]]]
    load_symbol_daily: Callable[..., pd.DataFrame]
    build_market_and_cross_section_states: Callable[..., tuple[object, object]]
    safe_float: Callable[[object, float], float]
    return_quantile_profile_cls: type
    build_sector_daily_frames: Callable[..., dict[str, pd.DataFrame]]
    run_sector_forecast: Callable[..., list[object]]
    build_stock_states_from_rows: Callable[..., list[object]]
    compose_state: Callable[..., CompositeState]
    path_from_manifest_entry: Callable[..., Path | None]
    load_json_dict: Callable[[object], dict[str, object]]
    decode_composite_state: Callable[[object], CompositeState | None]
    load_frozen_forecast_bundle: Callable[[object], dict[str, object]]
    score_live_composite_state_from_frozen_bundle: Callable[..., tuple[CompositeState | None, list[object]]]
    load_v2_info_items_for_date: Callable[..., tuple[str, list[InfoItem]]]
    enrich_state_with_info: Callable[..., CompositeState]
    sha256_file: Callable[[object], str]
    stable_json_hash: Callable[[object], str]
    top_negative_events: Callable[..., list[InfoSignalRecord]]
    top_positive_stock_signals: Callable[..., list[InfoSignalRecord]]
    quant_info_divergence_rows: Callable[..., list[InfoDivergenceRecord]]
    attach_external_signals_to_composite_state: Callable[..., tuple[CompositeState, dict[str, object]]]
    attach_insight_memory_to_state: Callable[..., CompositeState]


def build_daily_universe_context(
    settings: dict[str, object],
    *,
    deps: DailyStateRuntimeDependencies,
) -> object:
    deps.emit_progress("daily", "加载观察池与候选股票池")
    market_security, current_holdings, base_sector_map = deps.load_watchlist(str(settings["watchlist"]))
    universe = deps.build_candidate_universe(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    stocks = universe.rows or current_holdings
    sector_map = {
        stock.symbol: (stock.sector or base_sector_map.get(stock.symbol, "其他"))
        for stock in stocks
    }
    return deps.daily_universe_context_cls(
        market_security=market_security,
        current_holdings=current_holdings,
        stocks=stocks,
        sector_map=sector_map,
    )


def build_daily_composite_state(
    *,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path | None,
    snapshot: StrategySnapshot,
    allow_retrain: bool,
    universe_ctx: object,
    deps: DailyStateRuntimeDependencies,
) -> tuple[CompositeState, list[object]]:
    stock_rows: list[object] = []
    if allow_retrain:
        deps.emit_progress("daily", f"开始量化预测: universe={len(universe_ctx.stocks)}")
        market_forecast, stock_rows = deps.run_quant_pipeline(
            market_security=universe_ctx.market_security,
            stock_securities=universe_ctx.stocks,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
            min_train_days=int(settings["min_train_days"]),
            step_days=int(settings["step_days"]),
            l2=float(settings["l2"]),
            max_positions=int(settings["max_positions"]),
            use_margin_features=bool(settings["use_margin_features"]),
            margin_market_file=str(settings["margin_market_file"]),
            margin_stock_file=str(settings["margin_stock_file"]),
            enable_walk_forward_eval=False,
            progress_callback=lambda message: deps.emit_progress("daily", message),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
        )
        deps.emit_progress("daily", "开始构建市场状态与横截面状态")
        market_raw = deps.load_symbol_daily(
            symbol=universe_ctx.market_security.symbol,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
        )
        market_state, cross_section = deps.build_market_and_cross_section_states(
            market_symbol=universe_ctx.market_security.symbol,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
            use_margin_features=bool(settings["use_margin_features"]),
            margin_market_file=str(settings["margin_market_file"]),
            market_short_prob=float(market_forecast.short_prob),
            market_two_prob=deps.safe_float(getattr(market_forecast, "two_prob", np.nan), np.nan),
            market_three_prob=deps.safe_float(getattr(market_forecast, "three_prob", np.nan), np.nan),
            market_five_prob=float(market_forecast.five_prob),
            market_mid_prob=float(market_forecast.mid_prob),
            market_short_profile=deps.return_quantile_profile_cls(
                expected_return=float(deps.safe_float(getattr(market_forecast, "short_expected_ret", 0.0), 0.0)),
                q10=float(deps.safe_float(getattr(market_forecast, "short_q10", np.nan), np.nan)),
                q30=float(deps.safe_float(getattr(market_forecast, "short_q30", np.nan), np.nan)),
                q20=float(deps.safe_float(getattr(market_forecast, "short_q20", np.nan), np.nan)),
                q50=float(deps.safe_float(getattr(market_forecast, "short_q50", np.nan), np.nan)),
                q70=float(deps.safe_float(getattr(market_forecast, "short_q70", np.nan), np.nan)),
                q80=float(deps.safe_float(getattr(market_forecast, "short_q80", np.nan), np.nan)),
                q90=float(deps.safe_float(getattr(market_forecast, "short_q90", np.nan), np.nan)),
            ),
            market_mid_profile=deps.return_quantile_profile_cls(
                expected_return=float(deps.safe_float(getattr(market_forecast, "mid_expected_ret", 0.0), 0.0)),
                q10=float(deps.safe_float(getattr(market_forecast, "mid_q10", np.nan), np.nan)),
                q30=float(deps.safe_float(getattr(market_forecast, "mid_q30", np.nan), np.nan)),
                q20=float(deps.safe_float(getattr(market_forecast, "mid_q20", np.nan), np.nan)),
                q50=float(deps.safe_float(getattr(market_forecast, "mid_q50", np.nan), np.nan)),
                q70=float(deps.safe_float(getattr(market_forecast, "mid_q70", np.nan), np.nan)),
                q80=float(deps.safe_float(getattr(market_forecast, "mid_q80", np.nan), np.nan)),
                q90=float(deps.safe_float(getattr(market_forecast, "mid_q90", np.nan), np.nan)),
            ),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
            use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
            use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
            cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
        )
        deps.emit_progress("daily", "开始独立板块预测")
        sector_frames = deps.build_sector_daily_frames(
            stock_securities=universe_ctx.stocks,
            sector_map=universe_ctx.sector_map,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
        )
        sector_records = deps.run_sector_forecast(
            sector_frames=sector_frames,
            market_raw=market_raw,
            l2=float(settings["l2"]),
        )
        sectors = [
            SectorForecastState(
                sector=item.sector,
                up_5d_prob=float(item.up_5d_prob),
                up_20d_prob=float(item.up_20d_prob),
                relative_strength=float(item.excess_vs_market_prob - 0.5),
                rotation_speed=float(item.rotation_speed),
                crowding_score=float(item.crowding_score),
            )
            for item in sector_records
        ]
        sector_strength_map = {item.sector: float(item.excess_vs_market_prob - 0.5) for item in sector_records}
        stocks_state = deps.build_stock_states_from_rows(
            stock_rows,
            universe_ctx.sector_map,
            sector_strength_map=sector_strength_map,
        )
        deps.emit_progress("daily", "开始策略决策与交易计划生成")
        composite_state = deps.compose_state(
            market=market_state,
            sectors=sectors,
            stocks=stocks_state,
            cross_section=cross_section,
        )
        return composite_state, stock_rows

    if not manifest or manifest_path is None:
        raise RuntimeError(
            "daily-run default mode requires a published research manifest. "
            "Use `research-run` first, pass `--snapshot-path/--run-id`, or set `--allow-retrain`."
        )
    frozen_state_path = deps.path_from_manifest_entry(
        manifest.get("frozen_daily_state"),
        run_dir=manifest_path.parent,
    )
    frozen_bundle_path = deps.path_from_manifest_entry(
        manifest.get("frozen_forecast_bundle"),
        run_dir=manifest_path.parent,
    )
    frozen_state_payload = deps.load_json_dict(frozen_state_path) if frozen_state_path is not None else {}
    composite_payload = frozen_state_payload.get("composite_state")
    composite_state = deps.decode_composite_state(composite_payload)
    if composite_state is None:
        raise RuntimeError(
            f"Snapshot {manifest_path} does not contain usable frozen daily state. "
            "Re-run research with `--publish-forecast-models` or set `--allow-retrain`."
        )
    deps.emit_progress("daily", f"已加载发布快照: run_id={snapshot.run_id or 'NA'}")
    frozen_bundle = deps.load_frozen_forecast_bundle(frozen_bundle_path) if frozen_bundle_path is not None else {}
    if frozen_bundle:
        live_composite_state, live_stock_rows = deps.score_live_composite_state_from_frozen_bundle(
            bundle=frozen_bundle,
            settings=settings,
            universe_ctx=universe_ctx,
        )
        if live_composite_state is not None:
            composite_state = live_composite_state
            stock_rows = live_stock_rows
            deps.emit_progress("daily", "loaded frozen forecast models and refreshed live scores")
    return composite_state, stock_rows


def attach_daily_info_overlay(
    *,
    snapshot: StrategySnapshot,
    settings: dict[str, object],
    composite_state: CompositeState,
    symbol_names: dict[str, str],
    deps: DailyStateRuntimeDependencies,
) -> tuple[
    CompositeState,
    str,
    str,
    bool,
    int,
    list[InfoSignalRecord],
    list[InfoSignalRecord],
    list[InfoDivergenceRecord],
    list[InfoItem],
]:
    info_hash = str(snapshot.info_hash or settings.get("info_hash", ""))
    info_manifest_path = str(snapshot.info_manifest_path or settings.get("info_manifest_path", ""))
    info_shadow_enabled = bool(snapshot.info_shadow_enabled)
    info_item_count = 0
    top_negative_info_events: list[InfoSignalRecord] = []
    top_positive_info_signals: list[InfoSignalRecord] = []
    quant_info_divergence: list[InfoDivergenceRecord] = []
    info_items: list[InfoItem] = []
    if bool(settings.get("use_info_fusion", False)):
        info_file_path, info_items = deps.load_v2_info_items_for_date(
            settings=settings,
            as_of_date=pd.Timestamp(composite_state.market.as_of_date),
            learned_window=False,
        )
        if info_items:
            composite_state = deps.enrich_state_with_info(
                state=composite_state,
                as_of_date=pd.Timestamp(composite_state.market.as_of_date),
                info_items=info_items,
                settings=settings,
            )
            info_shadow_enabled = True
            info_item_count = len(info_items)
            if not info_hash:
                info_hash = deps.sha256_file(info_file_path) or deps.stable_json_hash([asdict(item) for item in info_items])
            top_negative_info_events = deps.top_negative_events(
                info_items,
                as_of_date=pd.Timestamp(composite_state.market.as_of_date),
                half_life_days=float(settings.get("info_half_life_days", 10.0)),
            )
            top_positive_info_signals = deps.top_positive_stock_signals(
                composite_state,
                symbol_names=symbol_names,
            )
            quant_info_divergence = deps.quant_info_divergence_rows(
                composite_state,
                symbol_names=symbol_names,
            )
    return (
        composite_state,
        info_hash,
        info_manifest_path,
        info_shadow_enabled,
        info_item_count,
        top_negative_info_events,
        top_positive_info_signals,
        quant_info_divergence,
        info_items,
    )


def attach_daily_external_signal_overlay(
    *,
    snapshot: StrategySnapshot,
    settings: dict[str, object],
    composite_state: CompositeState,
    info_items: list[InfoItem],
    allow_rebuild: bool,
    deps: DailyStateRuntimeDependencies,
) -> tuple[CompositeState, str, str, bool, dict[str, object], dict[str, object]]:
    manifest_path = str(snapshot.external_signal_manifest_path or settings.get("external_signal_manifest", ""))
    version = str(snapshot.external_signal_version or settings.get("external_signal_version", "v1"))
    enabled = bool(snapshot.external_signal_enabled or settings.get("external_signals", True))
    capital_flow_snapshot = dict(snapshot.capital_flow_snapshot or asdict(composite_state.capital_flow_state))
    macro_context_snapshot = dict(snapshot.macro_context_snapshot or asdict(composite_state.macro_context_state))

    if allow_rebuild or not snapshot.external_signal_enabled:
        composite_state, package = deps.attach_external_signals_to_composite_state(
            state=composite_state,
            settings=settings,
            as_of_date=pd.Timestamp(composite_state.market.as_of_date),
            info_items=info_items,
        )
        manifest_path = str(manifest_path or settings.get("external_signal_manifest", ""))
        version = str(package.get("manifest", {}).get("external_signal_version", version))
        enabled = bool(package.get("manifest", {}).get("external_signal_enabled", enabled))
        capital_flow_snapshot = dict(package.get("capital_flow_snapshot", capital_flow_snapshot))
        macro_context_snapshot = dict(package.get("macro_context_snapshot", macro_context_snapshot))
    return (
        composite_state,
        manifest_path,
        version,
        enabled,
        capital_flow_snapshot,
        macro_context_snapshot,
    )


def attach_daily_insight_overlay(
    *,
    settings: dict[str, object],
    composite_state: CompositeState,
    info_items: list[InfoItem],
    deps: DailyStateRuntimeDependencies,
) -> CompositeState:
    return deps.attach_insight_memory_to_state(
        state=composite_state,
        settings=settings,
        as_of_date=pd.Timestamp(composite_state.market.as_of_date),
        info_items=info_items,
    )
