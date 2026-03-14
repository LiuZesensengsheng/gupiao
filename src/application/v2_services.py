from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from src.application.v2_contracts import (
    CapitalFlowState,
    CompositeState,
    CrossSectionForecastState,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
    DailyRunResult,
    InfoAggregateState,
    InfoDivergenceRecord,
    InfoItem,
    InfoSignalRecord,
    HorizonForecast,
    LearnedPolicyModel,
    MarketFactsState,
    MarketForecastState,
    MarketSentimentState,
    MacroContextState,
    PolicyDecision,
    PolicyInput,
    PolicySpec,
    PredictionReviewState,
    PredictionReviewWindow,
    SectorForecastState,
    StockForecastState,
    StrategySnapshot,
)
from src.contracts.artifacts import (
    DatasetManifest,
    ForecastBundle,
    LearnedPolicyArtifact,
    ResearchManifest,
    add_artifact_metadata,
)
from src.application.v2_external_signal_support import (
    attach_external_signals_to_state,
    build_external_signal_package,
    ensure_external_signal_manifest_path,
    merge_external_signal_manifest_summary,
)
from src.application.v2_daily_runtime import (
    build_daily_symbol_names as _build_daily_symbol_names_external,
    load_daily_cached_result as _load_daily_cached_result_external,
)
from src.application.v2_backtest_cache_support import (
    build_prepared_backtest_cache_key as _build_prepared_backtest_cache_key_external,
    load_pickle_cache as _load_pickle_cache_external,
    prepared_backtest_cache_path as _prepared_backtest_cache_path_external,
    store_pickle_cache as _store_pickle_cache_external,
)
from src.application.v2_candidate_selection import (
    build_candidate_selection_state as _build_candidate_selection_state_external,
    candidate_risk_snapshot as _candidate_risk_snapshot_external,
    candidate_stocks_from_state as _candidate_stocks_from_state_external,
)
from src.application.v2_mainline_support import (
    build_mainline_states as _build_mainline_states_external,
    dominant_mainline_sectors as _dominant_mainline_sectors_external,
)
from src.application.v2_publish_support import (
    load_backtest_payload_for_run as _load_backtest_payload_for_run_external,
    load_backtest_payload_from_manifest as _load_backtest_payload_from_manifest_external,
    pass_default_switch_gate as _pass_default_switch_gate_external,
    pass_release_gate as _pass_release_gate_external,
    summary_from_payload as _summary_from_payload_external,
    tier_latest_manifest_path as _tier_latest_manifest_path_external,
    tier_latest_policy_path as _tier_latest_policy_path_external,
)
from src.application.v2_snapshot_support import (
    build_frozen_daily_state_payload as _build_frozen_daily_state_payload_external,
    decode_composite_state as _decode_composite_state_external,
    load_research_manifest_for_daily as _load_research_manifest_for_daily_external,
    resolve_manifest_entry_path as _path_from_manifest_entry_external,
    serialize_composite_state as _serialize_composite_state_external,
)
from src.application.v2_universe_generator import generate_dynamic_universe
from src.application.v2_sector_support import (
    allocate_sector_slots as _allocate_sector_slots_external,
    allocate_with_sector_budgets as _allocate_with_sector_budgets_external,
    build_sector_states as _build_sector_states_external,
    cap_sector_budgets as _cap_sector_budgets_external,
    ranked_sector_budgets_with_alpha as _ranked_sector_budgets_with_alpha_external,
)
from src.application.watchlist import load_watchlist
from src.domain.entities import TradeAction
from src.domain.news import blend_probability
from src.domain.policies import blend_horizon_score, decide_market_state
from src.infrastructure.discovery import (
    build_candidate_universe,
    build_predefined_universe,
    normalize_universe_tier,
)
from src.infrastructure.cross_section_forecast import forecast_cross_section_state
from src.infrastructure.features import (
    MARKET_FEATURE_COLUMNS,
    make_market_feature_frame,
)
from src.infrastructure.info_repository import load_v2_info_items
from src.infrastructure.forecast_engine import run_quant_pipeline
from src.infrastructure.v2_info_fusion import (
    build_info_state_maps,
    event_tag_counts,
    quant_info_divergence_rows,
    top_negative_events,
    top_positive_stock_signals,
)
from src.infrastructure.market_context import build_market_context_features
from src.infrastructure.market_data import load_symbol_daily, set_tushare_token
from src.infrastructure.modeling import (
    LogisticBinaryModel,
    MLPBinaryModel,
    MLPQuantileModel,
    QuantileLinearModel,
)
from src.infrastructure.panel_dataset import build_stock_live_panel_dataset, build_stock_panel_dataset
from src.infrastructure.sector_data import build_sector_daily_frames
from src.infrastructure.sector_forecast import run_sector_forecast
from src.infrastructure.strategy_memory import remember_daily_run, remember_research_run


def _clip(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _coalesce(primary: object, secondary: object, default: object) -> object:
    if primary is not None:
        return primary
    if secondary is not None:
        return secondary
    return default


def _configure_v2_tushare_token(
    *,
    explicit_token: str | None = None,
    daily: dict[str, object] | None = None,
    common: dict[str, object] | None = None,
) -> None:
    candidates: list[object] = [explicit_token]
    if isinstance(daily, dict):
        candidates.append(daily.get("tushare_token"))
    if isinstance(common, dict):
        candidates.append(common.get("tushare_token"))
    for candidate in candidates:
        if candidate is None:
            continue
        token = str(candidate).strip()
        if token:
            set_tushare_token(token)
            os.environ["TUSHARE_TOKEN"] = token
            return


def _parse_boolish(value: object, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _parse_csv_tokens(value: object, default: Iterable[str]) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        out = [str(item).strip() for item in value if str(item).strip()]
        return out or [str(item).strip() for item in default if str(item).strip()]
    text = str(value).strip()
    if not text:
        return [str(item).strip() for item in default if str(item).strip()]
    return [item.strip() for item in text.split(",") if item.strip()]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:
        return float(default)
    return out


def _signal_unit(value: object, scale: float) -> float:
    denom = max(1e-9, float(scale))
    return _clip(_safe_float(value, 0.0) / denom, -1.0, 1.0)


def _is_main_board_symbol(symbol: object) -> bool:
    text = str(symbol or "").strip().upper()
    if not text or "." not in text:
        return False
    code, market = text.split(".", 1)
    if market == "SH":
        return code.startswith(("600", "601", "603", "605"))
    if market == "SZ":
        return code.startswith(("000", "001", "002"))
    return False


_DEFAULT_SPLIT_MODE = "purged_wf"
_DEFAULT_EMBARGO_DAYS = 20
_RELEASE_GATE_THRESHOLD = {
    "excess_annual_return_min": 0.0,
    "information_ratio_min": 0.30,
    "max_drawdown_worse_limit": 0.05,
}
_DEFAULT_SWITCH_GATE_THRESHOLD = {
    "excess_annual_return_delta_min": 0.02,
    "information_ratio_delta_min": 0.10,
    "max_drawdown_worse_limit": 0.02,
}
_INFO_SHADOW_FEATURE_COLUMNS = [
    "q_logit",
    "i_logit",
    "q_minus_i",
    "negative_event_risk",
    "item_count_log",
]


def _stable_json_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _sha256_file(path_like: object) -> str:
    path = Path(str(path_like))
    if not path.exists():
        return ""
    h = hashlib.sha256()
    if path.is_dir():
        for file in sorted(p for p in path.rglob("*") if p.is_file()):
            rel = str(file.relative_to(path)).replace("\\", "/")
            h.update(rel.encode("utf-8"))
            with file.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
        return h.hexdigest()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class _InfoShadowModel:
    mode: str
    samples: int
    feature_cols: list[str]
    model: LogisticBinaryModel | None = None


def _clip_prob(p: float) -> float:
    return float(np.clip(float(p), 1e-6, 1.0 - 1e-6))


def _logit_prob(p: float) -> float:
    clipped = _clip_prob(p)
    return float(np.log(clipped / (1.0 - clipped)))


def _resolve_info_file_from_settings(settings: dict[str, object]) -> str:
    event_file = str(settings.get("event_file", "")).strip()
    if event_file and Path(event_file).exists():
        return event_file
    info_file = str(settings.get("info_file", "")).strip()
    if info_file and Path(info_file).exists():
        return info_file
    news_file = str(settings.get("news_file", "")).strip()
    if news_file and Path(news_file).exists():
        return news_file
    return info_file or news_file


def _load_v2_info_items_for_date(
    *,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    learned_window: bool = False,
) -> tuple[str, list[InfoItem]]:
    info_file = _resolve_info_file_from_settings(settings)
    if not info_file:
        return "", []
    lookback_days = int(
        settings.get(
            "learned_info_lookback_days" if learned_window else "event_lookback_days",
            settings.get("event_lookback_days", settings.get("info_lookback_days", 45)),
        )
    )
    items = load_v2_info_items(
        info_file,
        as_of_date=as_of_date.normalize(),
        lookback_days=lookback_days,
        source_mode=str(settings.get("info_source_mode", "layered")),
        info_types=settings.get("info_types", ("news", "announcement", "research")),
        info_subsets=settings.get("info_subsets", ("market_news", "announcements", "research")),
        announcement_event_tags=settings.get(
            "announcement_event_tags",
            (
                "earnings_positive",
                "earnings_negative",
                "guidance_positive",
                "guidance_negative",
                "contract_win",
                "contract_loss",
                "regulatory_positive",
                "regulatory_negative",
                "share_reduction",
                "share_increase",
                "trading_halt",
                "delisting_risk",
            ),
        ),
    )
    return info_file, items


def _info_feature_frame(
    *,
    quant_prob: Sequence[float],
    info_prob: Sequence[float],
    negative_event_risk: Sequence[float],
    item_count: Sequence[float],
) -> pd.DataFrame:
    q = np.asarray([_clip_prob(float(item)) for item in quant_prob], dtype=float)
    i = np.asarray([_clip_prob(float(item)) for item in info_prob], dtype=float)
    return pd.DataFrame(
        {
            "q_logit": [_logit_prob(float(item)) for item in q],
            "i_logit": [_logit_prob(float(item)) for item in i],
            "q_minus_i": q - i,
            "negative_event_risk": np.asarray(negative_event_risk, dtype=float),
            "item_count_log": np.log1p(np.asarray(item_count, dtype=float)),
        }
    )


def _fit_info_shadow_model(
    frame: pd.DataFrame,
    *,
    target_col: str,
    l2: float,
    min_samples: int,
) -> _InfoShadowModel:
    valid = frame.dropna(subset=_INFO_SHADOW_FEATURE_COLUMNS + [target_col]).copy()
    if len(valid) < int(min_samples):
        return _InfoShadowModel(mode="rule", samples=int(len(valid)), feature_cols=list(_INFO_SHADOW_FEATURE_COLUMNS))
    model = LogisticBinaryModel(l2=float(l2)).fit(valid, _INFO_SHADOW_FEATURE_COLUMNS, target_col)
    return _InfoShadowModel(
        mode="learned",
        samples=int(len(valid)),
        feature_cols=list(_INFO_SHADOW_FEATURE_COLUMNS),
        model=model,
    )


def _predict_info_shadow_prob(
    *,
    quant_prob: float,
    info_prob: float,
    negative_event_risk: float,
    item_count: int,
    score: float,
    strength: float,
    model: _InfoShadowModel | None,
) -> tuple[float, str]:
    if model is not None and model.mode == "learned" and model.model is not None:
        frame = _info_feature_frame(
            quant_prob=[quant_prob],
            info_prob=[info_prob],
            negative_event_risk=[negative_event_risk],
            item_count=[item_count],
        )
        prob = float(model.model.predict_proba(frame, model.feature_cols)[0])
        return _clip_prob(prob), "learned"
    return float(blend_probability(quant_prob, score, sentiment_strength=strength)), "rule"


def _compose_shadow_stock_score(
    *,
    stock: StockForecastState,
    info_state: InfoAggregateState,
) -> float:
    return float(
        0.15 * float(info_state.shadow_prob_1d)
        + 0.25 * float(info_state.shadow_prob_5d)
        + 0.60 * float(info_state.shadow_prob_20d)
        + 0.08 * float(stock.tradeability_score)
        - 0.08 * float(info_state.negative_event_risk)
    )


def _build_sector_map_from_state(state: CompositeState) -> dict[str, str]:
    return {str(stock.symbol): str(stock.sector) for stock in state.stocks}


def _enrich_state_with_info(
    *,
    state: CompositeState,
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
    settings: dict[str, object],
    stock_models: dict[str, _InfoShadowModel] | None = None,
    market_models: dict[str, _InfoShadowModel] | None = None,
) -> CompositeState:
    sector_map = _build_sector_map_from_state(state)
    market_info_state, sector_info_states, stock_info_states = build_info_state_maps(
        info_items=info_items,
        as_of_date=as_of_date,
        stock_symbols=sector_map.keys(),
        sector_map=sector_map,
        market_to_stock_carry=0.35,
        info_half_life_days=float(settings.get("info_half_life_days", 10.0)),
        market_info_strength=float(settings.get("market_info_strength", 0.9)),
        stock_info_strength=float(settings.get("stock_info_strength", 1.1)),
    )

    market_shadow_1d, _ = _predict_info_shadow_prob(
        quant_prob=float(state.market.up_1d_prob),
        info_prob=float(market_info_state.info_prob_1d),
        negative_event_risk=float(market_info_state.negative_event_risk),
        item_count=int(market_info_state.item_count),
        score=float(market_info_state.short_score),
        strength=float(settings.get("market_info_strength", 0.9)),
        model=None if market_models is None else market_models.get("1d"),
    )
    market_shadow_5d, _ = _predict_info_shadow_prob(
        quant_prob=float(state.market.up_5d_prob),
        info_prob=float(market_info_state.info_prob_5d),
        negative_event_risk=float(market_info_state.negative_event_risk),
        item_count=int(market_info_state.item_count),
        score=float(market_info_state.short_score),
        strength=0.9 * float(settings.get("market_info_strength", 0.9)),
        model=None if market_models is None else market_models.get("5d"),
    )
    market_shadow_20d, _ = _predict_info_shadow_prob(
        quant_prob=float(state.market.up_20d_prob),
        info_prob=float(market_info_state.info_prob_20d),
        negative_event_risk=float(market_info_state.negative_event_risk),
        item_count=int(market_info_state.item_count),
        score=float(market_info_state.mid_score),
        strength=float(settings.get("market_info_strength", 0.9)),
        model=None if market_models is None else market_models.get("20d"),
    )
    market_info_state = InfoAggregateState(
        **{
            **asdict(market_info_state),
            "shadow_prob_1d": float(market_shadow_1d),
            "shadow_prob_5d": float(market_shadow_5d),
            "shadow_prob_20d": float(market_shadow_20d),
        }
    )

    updated_sector_states: dict[str, InfoAggregateState] = {}
    for sector in state.sectors:
        current = sector_info_states.get(sector.sector, InfoAggregateState())
        shadow_1d = float(blend_probability(0.5, current.short_score, sentiment_strength=0.7 * float(settings.get("stock_info_strength", 1.1))))
        shadow_5d = float(blend_probability(float(sector.up_5d_prob), current.short_score, sentiment_strength=0.6 * float(settings.get("stock_info_strength", 1.1))))
        shadow_20d = float(blend_probability(float(sector.up_20d_prob), current.mid_score, sentiment_strength=0.6 * float(settings.get("stock_info_strength", 1.1))))
        updated_sector_states[sector.sector] = InfoAggregateState(
            **{
                **asdict(current),
                "shadow_prob_1d": shadow_1d,
                "shadow_prob_5d": shadow_5d,
                "shadow_prob_20d": shadow_20d,
            }
        )

    updated_stock_states: dict[str, InfoAggregateState] = {}
    for stock in state.stocks:
        current = stock_info_states.get(stock.symbol, InfoAggregateState())
        shadow_1d, _ = _predict_info_shadow_prob(
            quant_prob=float(stock.up_1d_prob),
            info_prob=float(current.info_prob_1d),
            negative_event_risk=float(current.negative_event_risk),
            item_count=int(current.item_count),
            score=float(current.short_score),
            strength=float(settings.get("stock_info_strength", 1.1)),
            model=None if stock_models is None else stock_models.get("1d"),
        )
        shadow_5d, _ = _predict_info_shadow_prob(
            quant_prob=float(stock.up_5d_prob),
            info_prob=float(current.info_prob_5d),
            negative_event_risk=float(current.negative_event_risk),
            item_count=int(current.item_count),
            score=float(current.short_score),
            strength=0.9 * float(settings.get("stock_info_strength", 1.1)),
            model=None if stock_models is None else stock_models.get("5d"),
        )
        shadow_20d, _ = _predict_info_shadow_prob(
            quant_prob=float(stock.up_20d_prob),
            info_prob=float(current.info_prob_20d),
            negative_event_risk=float(current.negative_event_risk),
            item_count=int(current.item_count),
            score=float(current.mid_score),
            strength=float(settings.get("stock_info_strength", 1.1)),
            model=None if stock_models is None else stock_models.get("20d"),
        )
        updated_stock_states[stock.symbol] = InfoAggregateState(
            **{
                **asdict(current),
                "shadow_prob_1d": float(shadow_1d),
                "shadow_prob_5d": float(shadow_5d),
                "shadow_prob_20d": float(shadow_20d),
            }
        )
    updated_mainlines = _build_mainline_states_external(
        market=state.market,
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=state.stocks,
        stock_score_fn=_stock_policy_score,
        sector_info_states=updated_sector_states,
        stock_info_states=updated_stock_states,
        capital_flow_state=state.capital_flow_state,
        macro_context_state=state.macro_context_state,
    )
    return CompositeState(
        market=state.market,
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=state.stocks,
        strategy_mode=state.strategy_mode,
        risk_regime=state.risk_regime,
        candidate_selection=getattr(state, "candidate_selection", None) or state.candidate_selection,
        mainlines=updated_mainlines,
        market_info_state=market_info_state,
        sector_info_states=updated_sector_states,
        stock_info_states=updated_stock_states,
        capital_flow_state=state.capital_flow_state,
        macro_context_state=state.macro_context_state,
    )


def _build_external_signal_package_for_date(
    *,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
) -> dict[str, object]:
    if not bool(settings.get("external_signals", True)):
        return {
            "capital_flow_state": CapitalFlowState(),
            "macro_context_state": MacroContextState(),
            "capital_flow_snapshot": asdict(CapitalFlowState()),
            "macro_context_snapshot": asdict(MacroContextState()),
            "manifest": {
                "as_of_date": str(as_of_date.date()),
                "external_signal_version": str(settings.get("external_signal_version", "v1")),
                "external_signal_enabled": False,
                "sources": {},
                "windows": {},
                "coverage": {},
                "event_summary": {},
                "capital_flow_snapshot": asdict(CapitalFlowState()),
                "macro_context_snapshot": asdict(MacroContextState()),
            },
        }
    return build_external_signal_package(
        settings=settings,
        as_of_date=as_of_date,
        info_items=info_items,
    )


def _attach_external_signals_to_composite_state(
    *,
    state: CompositeState,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
) -> tuple[CompositeState, dict[str, object]]:
    package = _build_external_signal_package_for_date(
        settings=settings,
        as_of_date=as_of_date,
        info_items=info_items,
    )
    enriched_state = attach_external_signals_to_state(
        state=state,
        capital_flow_state=package["capital_flow_state"],
        macro_context_state=package["macro_context_state"],
    )
    refreshed_mainlines = _build_mainline_states_external(
        market=enriched_state.market,
        cross_section=enriched_state.cross_section,
        sectors=enriched_state.sectors,
        stocks=enriched_state.stocks,
        stock_score_fn=_stock_policy_score,
        sector_info_states=getattr(enriched_state, "sector_info_states", {}),
        stock_info_states=getattr(enriched_state, "stock_info_states", {}),
        capital_flow_state=enriched_state.capital_flow_state,
        macro_context_state=enriched_state.macro_context_state,
    )
    return (
        replace(enriched_state, mainlines=refreshed_mainlines),
        package,
    )


def _load_json_dict(path_like: object) -> dict[str, object]:
    path = Path(str(path_like))
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _resolve_manifest_path(
    *,
    strategy_id: str,
    artifact_root: str,
    run_id: str | None,
    snapshot_path: str | None,
) -> Path:
    if snapshot_path is not None and str(snapshot_path).strip():
        path = Path(str(snapshot_path))
        if path.is_dir():
            return path / "research_manifest.json"
        return path
    if run_id is not None and str(run_id).strip():
        return Path(str(artifact_root)) / str(strategy_id) / str(run_id).strip() / "research_manifest.json"
    return Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json"


def _path_from_manifest_entry(entry: object, *, run_dir: Path) -> Path | None:
    return _path_from_manifest_entry_external(entry, run_dir=run_dir)


def _compose_run_snapshot_hash(
    *,
    run_id: str,
    strategy_id: str,
    config_hash: str,
    policy_hash: str,
    universe_hash: str,
    model_hashes: dict[str, str],
) -> str:
    payload = {
        "run_id": str(run_id),
        "strategy_id": str(strategy_id),
        "config_hash": str(config_hash),
        "policy_hash": str(policy_hash),
        "universe_hash": str(universe_hash),
        "model_hashes": {str(k): str(v) for k, v in sorted(model_hashes.items())},
    }
    return _stable_json_hash(payload)


def _emit_progress(stage: str, message: str) -> None:
    print(f"[V2][{stage}] {message}")


def _trajectory_step_count(trajectory: object) -> int:
    steps = getattr(trajectory, "steps", [])
    try:
        return int(len(steps))
    except TypeError:
        return 0


def _format_elapsed(seconds: float) -> str:
    secs = max(0, int(round(float(seconds))))
    minutes, remain = divmod(secs, 60)
    if minutes <= 0:
        return f"{remain}s"
    return f"{minutes}m{remain:02d}s"


def _policy_feature_names() -> list[str]:
    return [
        "mkt_up_1d",
        "mkt_up_20d",
        "mkt_drawdown_risk",
        "mkt_liquidity_stress",
        "cross_fund_flow",
        "cross_margin_risk_on",
        "cross_breadth",
        "cross_leader_participation",
        "cross_weak_ratio",
        "top_sector_up_20d",
        "top_sector_relative_strength",
        "top_stock_up_20d",
        "top_stock_tradeability",
        "top_stock_excess_vs_sector",
        "alpha_headroom",
        "alpha_breadth",
        "alpha_top_score",
        "alpha_avg_top3",
        "alpha_median_score",
        "candidate_shortlist_ratio",
        "candidate_shortlist_size_norm",
        "candidate_alpha_breadth",
        "candidate_durability",
    ]


def _policy_feature_vector(state: CompositeState) -> np.ndarray:
    top_sector = state.sectors[0] if state.sectors else None
    top_stock = state.stocks[0] if state.stocks else None
    alpha_metrics = _alpha_opportunity_metrics(state.stocks)
    candidate_selection = getattr(state, "candidate_selection", None)
    candidate_stocks = _candidate_stocks_from_state_external(state)
    candidate_alpha_metrics = _alpha_opportunity_metrics(candidate_stocks)
    candidate_risk = _candidate_risk_snapshot_external(candidate_stocks)
    shortlist_ratio = float(
        getattr(candidate_selection, "shortlist_ratio", 0.0)
        or (len(candidate_stocks) / max(1, len(state.stocks)))
    )
    shortlist_size_norm = float(
        _clip(len(candidate_stocks) / max(4.0, min(16.0, len(state.stocks) / 8.0)), 0.0, 1.0)
    )
    return np.asarray(
        [
            float(state.market.up_1d_prob),
            float(state.market.up_20d_prob),
            float(state.market.drawdown_risk),
            float(state.market.liquidity_stress),
            float(state.cross_section.fund_flow_strength),
            float(state.cross_section.margin_risk_on_score),
            float(state.cross_section.breadth_strength),
            float(state.cross_section.leader_participation),
            float(state.cross_section.weak_stock_ratio),
            0.0 if top_sector is None else float(top_sector.up_20d_prob),
            0.0 if top_sector is None else float(top_sector.relative_strength),
            0.0 if top_stock is None else float(top_stock.up_20d_prob),
            0.0 if top_stock is None else float(top_stock.tradeability_score),
            0.0 if top_stock is None else float(top_stock.excess_vs_sector_prob),
            float(alpha_metrics["alpha_headroom"]),
            float(alpha_metrics["breadth_ratio"]),
            float(alpha_metrics["top_score"]),
            float(alpha_metrics["avg_top3"]),
            float(alpha_metrics["median_score"]),
            shortlist_ratio,
            shortlist_size_norm,
            float(candidate_alpha_metrics["breadth_ratio"]),
            float(candidate_risk["durability_score"]),
        ],
        dtype=float,
    )


def _fit_ridge_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    if X.size == 0 or y.size == 0:
        return 0.0, np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1), dtype=float)
    X_aug = np.hstack([ones, X])
    if sample_weight is not None:
        weight = np.asarray(sample_weight, dtype=float).reshape(-1)
        if weight.size != X.shape[0]:
            raise ValueError("sample_weight dimension mismatch")
        weight = np.sqrt(np.clip(weight, 1e-9, None)).reshape(-1, 1)
        X_aug = X_aug * weight
        y = y * weight.reshape(-1)
    reg = np.eye(X_aug.shape[1], dtype=float) * float(max(0.0, l2))
    reg[0, 0] = 0.0
    coef = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)
    return float(coef[0]), np.asarray(coef[1:], dtype=float)


def _predict_ridge(features: np.ndarray, intercept: float, coef: np.ndarray) -> float:
    return float(intercept + np.dot(np.asarray(features, dtype=float), np.asarray(coef, dtype=float)))


def _normalize_coef_vector(coef: object, expected_dim: int) -> np.ndarray:
    arr = np.asarray(coef, dtype=float).reshape(-1)
    target_dim = max(0, int(expected_dim))
    if arr.size == target_dim:
        return arr
    if arr.size > target_dim:
        return np.asarray(arr[:target_dim], dtype=float)
    out = np.zeros(target_dim, dtype=float)
    if arr.size > 0:
        out[: arr.size] = arr
    return out


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


@dataclass(frozen=True)
class _ReturnQuantileProfile:
    expected_return: float
    q10: float
    q30: float
    q20: float
    q50: float
    q70: float
    q80: float
    q90: float


def _fit_quantile_quintet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel, QuantileLinearModel]:
    return (
        QuantileLinearModel(quantile=0.10, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.30, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.50, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.70, l2=l2).fit(df, feature_cols, target_col),
        QuantileLinearModel(quantile=0.90, l2=l2).fit(df, feature_cols, target_col),
    )


def _fit_mlp_quantile_quintet(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_col: str,
    l2: float,
) -> tuple[MLPQuantileModel, MLPQuantileModel, MLPQuantileModel, MLPQuantileModel, MLPQuantileModel]:
    return (
        MLPQuantileModel(quantile=0.10, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.30, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.50, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.70, l2=l2).fit(df, feature_cols, target_col),
        MLPQuantileModel(quantile=0.90, l2=l2).fit(df, feature_cols, target_col),
    )


def _predict_quantile_profile(
    row: pd.DataFrame,
    *,
    feature_cols: list[str],
    q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> _ReturnQuantileProfile:
    raw = [float(model.predict(row, feature_cols)[0]) for model in q_models]
    q10, q30, q50, q70, q90 = [float(x) for x in np.maximum.accumulate(np.asarray(raw, dtype=float))]
    q20 = float(0.5 * (q10 + q30))
    q80 = float(0.5 * (q70 + q90))
    return _ReturnQuantileProfile(
        expected_return=float(0.10 * q10 + 0.20 * q30 + 0.40 * q50 + 0.20 * q70 + 0.10 * q90),
        q10=q10,
        q30=q30,
        q20=q20,
        q50=q50,
        q70=q70,
        q80=q80,
        q90=q90,
    )


def _predict_quantile_profiles(
    frame: pd.DataFrame,
    *,
    feature_cols: list[str],
    q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["expected_return", "q10", "q30", "q20", "q50", "q70", "q80", "q90"],
        )
    raw = np.column_stack([model.predict(frame, feature_cols) for model in q_models]).astype(float)
    ordered = np.maximum.accumulate(raw, axis=1)
    q10 = ordered[:, 0]
    q30 = ordered[:, 1]
    q50 = ordered[:, 2]
    q70 = ordered[:, 3]
    q90 = ordered[:, 4]
    q20 = 0.5 * (q10 + q30)
    q80 = 0.5 * (q70 + q90)
    expected = 0.10 * q10 + 0.20 * q30 + 0.40 * q50 + 0.20 * q70 + 0.10 * q90
    return pd.DataFrame(
        {
            "expected_return": expected.astype(float),
            "q10": q10.astype(float),
            "q30": q30.astype(float),
            "q20": q20.astype(float),
            "q50": q50.astype(float),
            "q70": q70.astype(float),
            "q80": q80.astype(float),
            "q90": q90.astype(float),
        },
        index=frame.index,
    )


def _serialize_binary_model(model: LogisticBinaryModel | MLPBinaryModel) -> dict[str, object]:
    if isinstance(model, LogisticBinaryModel):
        return {
            "model_type": "logistic_linear",
            "l2": float(model.l2),
            "max_iter": int(model.max_iter),
            "feature_names": list(model.feature_names),
            "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
            "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
            "coef": [] if model.coef_ is None else np.asarray(model.coef_, dtype=float).tolist(),
            "intercept": float(model.intercept_),
            "fallback_prob": None if model.fallback_prob_ is None else float(model.fallback_prob_),
        }
    return {
        "model_type": "mlp_binary",
        "l2": float(model.l2),
        "hidden_dim": int(model.hidden_dim),
        "epochs": int(model.epochs),
        "learning_rate": float(model.learning_rate),
        "random_state": int(model.random_state),
        "feature_names": list(model.feature_names),
        "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
        "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
        "w1": [] if model.w1_ is None else np.asarray(model.w1_, dtype=float).tolist(),
        "b1": [] if model.b1_ is None else np.asarray(model.b1_, dtype=float).tolist(),
        "w2": [] if model.w2_ is None else np.asarray(model.w2_, dtype=float).tolist(),
        "b2": float(model.b2_),
        "fallback_prob": None if model.fallback_prob_ is None else float(model.fallback_prob_),
    }


def _deserialize_binary_model(payload: dict[str, object]) -> LogisticBinaryModel | MLPBinaryModel:
    model_type = str(payload.get("model_type", "logistic_linear"))
    if model_type == "mlp_binary":
        model = MLPBinaryModel(
            l2=float(payload.get("l2", 1.0)),
            hidden_dim=int(payload.get("hidden_dim", 24)),
            epochs=int(payload.get("epochs", 120)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            random_state=int(payload.get("random_state", 7)),
        )
        model.feature_names = [str(item) for item in payload.get("feature_names", [])]
        model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
        model.std_ = np.asarray(payload.get("std", []), dtype=float)
        model.w1_ = np.asarray(payload.get("w1", []), dtype=float)
        model.b1_ = np.asarray(payload.get("b1", []), dtype=float)
        model.w2_ = np.asarray(payload.get("w2", []), dtype=float)
        model.b2_ = float(payload.get("b2", 0.0))
        fallback_prob = payload.get("fallback_prob")
        model.fallback_prob_ = None if fallback_prob is None else float(fallback_prob)
        return model

    model = LogisticBinaryModel(
        l2=float(payload.get("l2", 1.0)),
        max_iter=int(payload.get("max_iter", 400)),
    )
    model.feature_names = [str(item) for item in payload.get("feature_names", [])]
    model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
    model.std_ = np.asarray(payload.get("std", []), dtype=float)
    model.coef_ = np.asarray(payload.get("coef", []), dtype=float)
    model.intercept_ = float(payload.get("intercept", 0.0))
    fallback_prob = payload.get("fallback_prob")
    model.fallback_prob_ = None if fallback_prob is None else float(fallback_prob)
    return model


def _serialize_quantile_model(model: QuantileLinearModel | MLPQuantileModel) -> dict[str, object]:
    if isinstance(model, QuantileLinearModel):
        return {
            "model_type": "quantile_linear",
            "quantile": float(model.quantile),
            "l2": float(model.l2),
            "max_iter": int(model.max_iter),
            "feature_names": list(model.feature_names),
            "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
            "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
            "coef": [] if model.coef_ is None else np.asarray(model.coef_, dtype=float).tolist(),
            "intercept": float(model.intercept_),
            "fallback_value": None if model.fallback_value_ is None else float(model.fallback_value_),
        }
    return {
        "model_type": "mlp_quantile",
        "quantile": float(model.quantile),
        "l2": float(model.l2),
        "hidden_dim": int(model.hidden_dim),
        "epochs": int(model.epochs),
        "learning_rate": float(model.learning_rate),
        "random_state": int(model.random_state),
        "feature_names": list(model.feature_names),
        "mean": [] if model.mean_ is None else np.asarray(model.mean_, dtype=float).tolist(),
        "std": [] if model.std_ is None else np.asarray(model.std_, dtype=float).tolist(),
        "w1": [] if model.w1_ is None else np.asarray(model.w1_, dtype=float).tolist(),
        "b1": [] if model.b1_ is None else np.asarray(model.b1_, dtype=float).tolist(),
        "w2": [] if model.w2_ is None else np.asarray(model.w2_, dtype=float).tolist(),
        "b2": float(model.b2_),
        "fallback_value": None if model.fallback_value_ is None else float(model.fallback_value_),
    }


def _deserialize_quantile_model(payload: dict[str, object]) -> QuantileLinearModel | MLPQuantileModel:
    model_type = str(payload.get("model_type", "quantile_linear"))
    if model_type == "mlp_quantile":
        model = MLPQuantileModel(
            quantile=float(payload.get("quantile", 0.5)),
            l2=float(payload.get("l2", 1.0)),
            hidden_dim=int(payload.get("hidden_dim", 24)),
            epochs=int(payload.get("epochs", 120)),
            learning_rate=float(payload.get("learning_rate", 0.03)),
            random_state=int(payload.get("random_state", 7)),
        )
        model.feature_names = [str(item) for item in payload.get("feature_names", [])]
        model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
        model.std_ = np.asarray(payload.get("std", []), dtype=float)
        model.w1_ = np.asarray(payload.get("w1", []), dtype=float)
        model.b1_ = np.asarray(payload.get("b1", []), dtype=float)
        model.w2_ = np.asarray(payload.get("w2", []), dtype=float)
        model.b2_ = float(payload.get("b2", 0.0))
        fallback_value = payload.get("fallback_value")
        model.fallback_value_ = None if fallback_value is None else float(fallback_value)
        return model

    model = QuantileLinearModel(
        quantile=float(payload.get("quantile", 0.5)),
        l2=float(payload.get("l2", 1.0)),
        max_iter=int(payload.get("max_iter", 300)),
    )
    model.feature_names = [str(item) for item in payload.get("feature_names", [])]
    model.mean_ = np.asarray(payload.get("mean", []), dtype=float)
    model.std_ = np.asarray(payload.get("std", []), dtype=float)
    model.coef_ = np.asarray(payload.get("coef", []), dtype=float)
    model.intercept_ = float(payload.get("intercept", 0.0))
    fallback_value = payload.get("fallback_value")
    model.fallback_value_ = None if fallback_value is None else float(fallback_value)
    return model


def _serialize_quantile_bundle(
    q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> list[dict[str, object]]:
    return [_serialize_quantile_model(model) for model in q_models]


def _deserialize_quantile_bundle(
    payload: object,
) -> tuple[
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
    QuantileLinearModel | MLPQuantileModel,
]:
    items = payload if isinstance(payload, list) else []
    models = [_deserialize_quantile_model(item) for item in items if isinstance(item, dict)]
    if len(models) != 5:
        raise ValueError("invalid quantile model bundle")
    return tuple(models)  # type: ignore[return-value]


_HORIZON_SCALE = {
    "1d": 0.035,
    "2d": 0.050,
    "3d": 0.065,
    "5d": 0.095,
    "10d": 0.145,
    "20d": 0.220,
}


def _next_session_label(as_of_date: object) -> str:
    ts = pd.Timestamp(as_of_date)
    if pd.isna(ts):
        return ""
    return str((ts + pd.offsets.BDay(1)).date())


def _blend_quantile_profiles(
    left: _ReturnQuantileProfile,
    right: _ReturnQuantileProfile,
    *,
    left_weight: float,
) -> _ReturnQuantileProfile:
    w = float(np.clip(left_weight, 0.0, 1.0))
    r = 1.0 - w
    return _ReturnQuantileProfile(
        expected_return=float(w * left.expected_return + r * right.expected_return),
        q10=float(w * left.q10 + r * right.q10),
        q30=float(w * left.q30 + r * right.q30),
        q20=float(w * left.q20 + r * right.q20),
        q50=float(w * left.q50 + r * right.q50),
        q70=float(w * left.q70 + r * right.q70),
        q80=float(w * left.q80 + r * right.q80),
        q90=float(w * left.q90 + r * right.q90),
    )


def _synthetic_quantile_profile(*, prob: float, horizon_key: str) -> _ReturnQuantileProfile:
    scale = float(_HORIZON_SCALE.get(str(horizon_key), 0.08))
    expected = float(np.clip((float(prob) - 0.5) * 1.2 * scale, -scale, scale))
    spread = max(scale * 0.55, 0.01)
    q50 = expected
    q10 = expected - 0.85 * spread
    q30 = expected - 0.40 * spread
    q70 = expected + 0.40 * spread
    q90 = expected + 0.85 * spread
    return _ReturnQuantileProfile(
        expected_return=float(expected),
        q10=float(q10),
        q30=float(q30),
        q20=float(0.5 * (q10 + q30)),
        q50=float(q50),
        q70=float(q70),
        q80=float(0.5 * (q70 + q90)),
        q90=float(q90),
    )


def _intrinsic_confidence(
    *,
    up_prob: float,
    horizon_probs: dict[str, float],
    info_state: InfoAggregateState | None = None,
    calibration_prior: dict[str, float] | None = None,
    tradability_status: str = "normal",
) -> tuple[float, str]:
    p = float(up_prob)
    probability_edge = float(np.clip(abs(p - 0.5) / 0.22, 0.0, 1.0))
    local_probs = [float(v) for v in horizon_probs.values() if v == v]
    if local_probs:
        dispersion = float(np.std(local_probs))
        consistency = float(np.clip(1.0 - dispersion / 0.12, 0.0, 1.0))
    else:
        consistency = 0.5
    coverage = float(getattr(info_state, "coverage_confidence", 0.55) if info_state is not None else 0.55)
    calibration = 0.58
    if calibration_prior:
        hit_rate = float(calibration_prior.get("top_k_hit_rate", calibration_prior.get("hit_rate", 0.55)))
        rank_ic = float(calibration_prior.get("rank_ic", 0.05))
        calibration = float(np.clip(0.55 * hit_rate + 0.45 * (0.5 + rank_ic), 0.0, 1.0))
    status_penalty = 0.0
    if tradability_status == "data_insufficient":
        status_penalty = 0.18
    elif tradability_status in {"halted", "delisted"}:
        status_penalty = 0.40
    confidence = float(
        np.clip(
            0.34 * calibration
            + 0.28 * consistency
            + 0.20 * coverage
            + 0.18 * probability_edge
            - status_penalty,
            0.05,
            0.98,
        )
    )
    if confidence >= 0.78:
        tone = "高"
    elif confidence >= 0.62:
        tone = "中高"
    elif confidence >= 0.46:
        tone = "中"
    else:
        tone = "偏低"
    return confidence, f"{tone}置信度: 校准稳定性、信号一致性和样本覆盖共同支持。"


def _build_horizon_forecasts(
    *,
    latest_close: float,
    horizon_probs: dict[str, float],
    short_profile: _ReturnQuantileProfile | None,
    mid_profile: _ReturnQuantileProfile | None,
    info_state: InfoAggregateState | None = None,
    calibration_priors: dict[str, dict[str, float]] | None = None,
    tradability_status: str = "normal",
) -> dict[str, HorizonForecast]:
    base_short = short_profile or _synthetic_quantile_profile(prob=float(horizon_probs.get("1d", 0.5)), horizon_key="1d")
    base_mid = mid_profile or _synthetic_quantile_profile(prob=float(horizon_probs.get("20d", 0.5)), horizon_key="20d")
    five_profile = _blend_quantile_profiles(base_short, base_mid, left_weight=0.35)
    profile_map = {
        "1d": base_short,
        "2d": _blend_quantile_profiles(base_short, five_profile, left_weight=0.75),
        "3d": _blend_quantile_profiles(base_short, five_profile, left_weight=0.60),
        "5d": five_profile,
        "10d": _blend_quantile_profiles(five_profile, base_mid, left_weight=0.45),
        "20d": base_mid,
    }
    out: dict[str, HorizonForecast] = {}
    for key, label in [("1d", "未来1日"), ("2d", "未来2日"), ("3d", "未来3日"), ("5d", "未来5日"), ("10d", "未来10日"), ("20d", "未来20日")]:
        profile = profile_map[key]
        up_prob = float(horizon_probs.get(key, 0.5))
        confidence, reason = _intrinsic_confidence(
            up_prob=up_prob,
            horizon_probs=horizon_probs,
            info_state=info_state,
            calibration_prior=None if calibration_priors is None else calibration_priors.get(key),
            tradability_status=tradability_status,
        )
        base_price = float(latest_close)
        price_low = np.nan if base_price != base_price else float(base_price * (1.0 + float(profile.q10)))
        price_mid = np.nan if base_price != base_price else float(base_price * (1.0 + float(profile.q50)))
        price_high = np.nan if base_price != base_price else float(base_price * (1.0 + float(profile.q90)))
        out[key] = HorizonForecast(
            horizon_days=int(key.rstrip("d")),
            label=label,
            up_prob=up_prob,
            expected_return=float(profile.expected_return),
            q10=float(profile.q10),
            q50=float(profile.q50),
            q90=float(profile.q90),
            price_low=price_low,
            price_mid=price_mid,
            price_high=price_high,
            confidence=float(confidence),
            confidence_reason=str(reason),
        )
    return out


def _market_facts_from_row(row: pd.Series) -> MarketFactsState:
    coverage = int(round(float(_safe_float(row.get("breadth_coverage", 0.0), 0.0))))
    return MarketFactsState(
        sample_coverage=max(0, coverage),
        advancers=int(round(float(_safe_float(row.get("breadth_advancers", 0.0), 0.0)))),
        decliners=int(round(float(_safe_float(row.get("breadth_decliners", 0.0), 0.0)))),
        flats=int(round(float(_safe_float(row.get("breadth_flats", 0.0), 0.0)))),
        limit_up_count=int(round(float(_safe_float(row.get("breadth_limit_up_count", 0.0), 0.0)))),
        limit_down_count=int(round(float(_safe_float(row.get("breadth_limit_down_count", 0.0), 0.0)))),
        new_high_count=int(round(float(_safe_float(row.get("breadth_new_high_count", 0.0), 0.0)))),
        new_low_count=int(round(float(_safe_float(row.get("breadth_new_low_count", 0.0), 0.0)))),
        median_return=float(_safe_float(row.get("breadth_median_return", 0.0), 0.0)),
        sample_amount=float(_safe_float(row.get("breadth_sample_amount", 0.0), 0.0)),
        amount_z20=float(_safe_float(row.get("breadth_amount_z20", 0.0), 0.0)),
    )


def _sentiment_stage(score: float) -> str:
    if score >= 78.0:
        return "过热"
    if score >= 64.0:
        return "亢奋"
    if score >= 54.0:
        return "回暖"
    if score >= 42.0:
        return "修复"
    return "冰点"


def _pct_text(value: float, *, signed: bool = False) -> str:
    fmt = f"{float(value) * 100:+.1f}%" if signed else f"{float(value) * 100:.1f}%"
    return fmt


def _num_text(value: float, digits: int = 2, *, signed: bool = False) -> str:
    fmt = f"{float(value):+.{digits}f}" if signed else f"{float(value):.{digits}f}"
    return fmt


def _build_market_sentiment_state(
    *,
    market: MarketForecastState,
    cross_section: CrossSectionForecastState,
    capital_flow: CapitalFlowState,
    macro: MacroContextState,
) -> MarketSentimentState:
    facts = market.market_facts
    advance_balance = float(np.clip((facts.advancers - facts.decliners) / max(1, facts.sample_coverage), -1.0, 1.0))
    limit_balance = float(
        np.clip((facts.limit_up_count - facts.limit_down_count) / max(1, facts.sample_coverage), -1.0, 1.0)
    )
    high_low_balance = float(
        np.clip((facts.new_high_count - facts.new_low_count) / max(1, facts.sample_coverage), -1.0, 1.0)
    )
    score = float(
        np.clip(
            50.0
            + 14.0 * float(cross_section.breadth_strength)
            + 12.0 * advance_balance
            + 8.0 * limit_balance
            + 8.0 * high_low_balance
            + 10.0 * (float(capital_flow.turnover_heat) - 0.5)
            + 6.0 * float(capital_flow.margin_balance_change)
            + 4.0 * float(capital_flow.northbound_net_flow)
            + 6.0 * (float(market.up_5d_prob) - 0.5)
            - 10.0 * float(market.drawdown_risk)
            - 6.0 * max(0.0, float(macro.fx_pressure))
            - 5.0 * max(0.0, float(macro.commodity_pressure)),
            0.0,
            100.0,
        )
    )
    drivers: list[tuple[float, str]] = [
        (abs(float(cross_section.breadth_strength)), f"市场宽度强度 {_pct_text(float(cross_section.breadth_strength), signed=True)}"),
        (abs(advance_balance), f"涨跌家数差 {facts.advancers}/{facts.decliners}"),
        (abs(limit_balance), f"涨跌停差 {facts.limit_up_count}/{facts.limit_down_count}"),
        (abs(high_low_balance), f"新高/新低 {facts.new_high_count}/{facts.new_low_count}"),
        (abs(float(capital_flow.turnover_heat) - 0.5), f"成交热度 {_pct_text(float(capital_flow.turnover_heat))}"),
        (abs(float(capital_flow.margin_balance_change)), f"两融变化 {_pct_text(float(capital_flow.margin_balance_change), signed=True)}"),
        (abs(float(capital_flow.northbound_net_flow)), f"北向强度 {_num_text(float(capital_flow.northbound_net_flow), 2, signed=True)}"),
    ]
    ordered = [text for _, text in sorted(drivers, key=lambda item: item[0], reverse=True)[:4]]
    stage = _sentiment_stage(score)
    summary = f"{stage}阶段，下一交易日情绪分 {score:.0f}/100。"
    return MarketSentimentState(score=score, stage=stage, drivers=ordered, summary=summary)


def _build_date_slice_index(
    frame: pd.DataFrame,
    *,
    sort_cols: list[str],
) -> tuple[pd.DataFrame, dict[pd.Timestamp, tuple[int, int]]]:
    if frame.empty:
        return frame.copy(), {}
    work = frame.sort_values(sort_cols).reset_index(drop=True).copy()
    dates = pd.to_datetime(work["date"]).to_numpy()
    bounds: dict[pd.Timestamp, tuple[int, int]] = {}
    start = 0
    n = len(work)
    while start < n:
        date = pd.Timestamp(dates[start])
        end = start + 1
        while end < n and pd.Timestamp(dates[end]) == date:
            end += 1
        bounds[date] = (start, end)
        start = end
    return work, bounds


def _distributional_score(
    *,
    short_prob: float,
    two_prob: float,
    three_prob: float,
    five_prob: float,
    mid_prob: float,
    short_expected_ret: float,
    mid_expected_ret: float,
) -> float:
    base_score = float(
        0.14 * float(short_prob)
        + 0.18 * float(two_prob)
        + 0.22 * float(three_prob)
        + 0.28 * float(five_prob)
        + 0.18 * float(mid_prob)
    )
    short_ret_score = float(np.clip(0.5 + float(short_expected_ret) / 0.06, 0.0, 1.0))
    two_ret_score = float(
        np.clip(0.5 + (0.75 * float(short_expected_ret) + 0.25 * float(mid_expected_ret)) / 0.08, 0.0, 1.0)
    )
    three_ret_score = float(
        np.clip(0.5 + (0.60 * float(short_expected_ret) + 0.40 * float(mid_expected_ret)) / 0.10, 0.0, 1.0)
    )
    five_ret_score = float(
        np.clip(0.5 + (0.35 * float(short_expected_ret) + 0.65 * float(mid_expected_ret)) / 0.12, 0.0, 1.0)
    )
    mid_ret_score = float(np.clip(0.5 + float(mid_expected_ret) / 0.20, 0.0, 1.0))
    dist_score = float(
        0.14 * short_ret_score
        + 0.18 * two_ret_score
        + 0.22 * three_ret_score
        + 0.24 * five_ret_score
        + 0.22 * mid_ret_score
    )
    return float(0.4 * base_score + 0.6 * dist_score)


def _is_actionable_status(status: str) -> bool:
    return str(status) not in {"halted", "delisted"}


def _status_tradeability_limit(status: str) -> float:
    status = str(status)
    if status in {"halted", "delisted"}:
        return 0.0
    if status == "data_insufficient":
        return 0.35
    return 1.0


def _status_score_penalty(status: str) -> float:
    status = str(status)
    if status == "halted":
        return 1.0
    if status == "delisted":
        return 1.5
    if status == "data_insufficient":
        return 0.08
    return 0.0


def _alpha_score_components(stock: StockForecastState) -> dict[str, float]:
    base_alpha_score = float(getattr(stock, "alpha_score", 0.0))
    up_1d = float(getattr(stock, "up_1d_prob", 0.5))
    up_5d = float(getattr(stock, "up_5d_prob", 0.5))
    up_20d = float(getattr(stock, "up_20d_prob", 0.5))
    up_2d = float(getattr(stock, "up_2d_prob", 0.65 * up_1d + 0.35 * up_5d))
    up_3d = float(getattr(stock, "up_3d_prob", 0.35 * up_1d + 0.65 * up_5d))
    excess_vs_sector = float(getattr(stock, "excess_vs_sector_prob", 0.5))
    event_impact = float(getattr(stock, "event_impact_score", 0.5))
    tradeability_score = float(getattr(stock, "tradeability_score", 0.5))
    if abs(base_alpha_score) <= 1e-12:
        base_components = {
            "short": 0.08 * up_1d,
            "two": 0.10 * up_2d,
            "three": 0.13 * up_3d,
            "five": 0.20 * up_5d,
            "mid": 0.20 * up_20d,
            "excess": 0.18 * excess_vs_sector,
            "tradeability": 0.07 * tradeability_score,
            "event": 0.04 * event_impact,
        }
        base_alpha_score = float(sum(base_components.values()))
    else:
        base_components = {}

    horizon_dispersion = float(
        _clip(
            0.20 * abs(up_1d - up_2d)
            + 0.25 * abs(up_2d - up_3d)
            + 0.25 * abs(up_3d - up_5d)
            + 0.30 * abs(up_5d - up_20d),
            0.0,
            1.0,
        )
    )
    execution_risk = float(_clip(1.0 - tradeability_score, 0.0, 1.0))
    event_risk = float(_clip((0.55 - event_impact) / 0.55, 0.0, 1.0))
    medium_edge = float(_clip(0.58 * (up_20d - 0.50) + 0.42 * (up_5d - 0.50), 0.0, 0.35))
    sector_edge = float(_clip(excess_vs_sector - 0.50, 0.0, 0.30))
    trend_alignment = float(
        _clip(
            0.55 * max(0.0, up_3d - up_1d)
            + 0.75 * max(0.0, up_5d - up_2d)
            + 0.90 * max(0.0, up_20d - up_5d),
            0.0,
            1.0,
        )
    )
    stability_bonus = float(_clip((0.16 - horizon_dispersion) / 0.16, 0.0, 1.0))
    quality_bonus = float(_clip(0.65 * tradeability_score + 0.35 * event_impact - 0.55, 0.0, 0.35))
    reversal_penalty = float(_clip(up_1d - max(up_5d, up_20d), 0.0, 0.35))
    weak_mid_penalty = float(_clip(0.52 - up_20d, 0.0, 0.20))
    risk_penalty = float(
        0.16 * horizon_dispersion
        + 0.12 * execution_risk
        + 0.08 * event_risk
        + 0.10 * reversal_penalty
        + 0.08 * weak_mid_penalty
    )
    selection_bonus = float(
        0.18 * medium_edge
        + 0.14 * sector_edge
        + 0.10 * trend_alignment
        + 0.08 * stability_bonus
        + 0.06 * quality_bonus
    )
    status_penalty = float(_status_score_penalty(getattr(stock, "tradability_status", "normal")))
    raw = dict(base_components)
    raw["base_alpha_score"] = float(base_alpha_score)
    raw["medium_edge"] = medium_edge
    raw["sector_edge"] = sector_edge
    raw["trend_alignment"] = trend_alignment
    raw["stability_bonus"] = stability_bonus
    raw["quality_bonus"] = quality_bonus
    raw["selection_bonus"] = selection_bonus
    raw["horizon_dispersion"] = horizon_dispersion
    raw["execution_risk"] = execution_risk
    raw["event_risk"] = event_risk
    raw["reversal_penalty"] = reversal_penalty
    raw["weak_mid_penalty"] = weak_mid_penalty
    raw["risk_penalty"] = risk_penalty
    raw["status_penalty"] = status_penalty
    raw["alpha_score"] = float(base_alpha_score + selection_bonus - risk_penalty - status_penalty)
    return raw


def _build_stock_states_from_panel_slice(
    *,
    panel_row: pd.DataFrame,
    feature_cols: list[str],
    short_model: LogisticBinaryModel | MLPBinaryModel,
    two_model: LogisticBinaryModel | MLPBinaryModel,
    three_model: LogisticBinaryModel | MLPBinaryModel,
    five_model: LogisticBinaryModel | MLPBinaryModel,
    mid_model: LogisticBinaryModel | MLPBinaryModel,
    short_q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
    mid_q_models: tuple[
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
        QuantileLinearModel | MLPQuantileModel,
    ],
) -> tuple[list[StockForecastState], pd.DataFrame]:
    if panel_row.empty:
        return [], pd.DataFrame()

    short_probs = short_model.predict_proba(panel_row, feature_cols)
    two_probs = two_model.predict_proba(panel_row, feature_cols)
    three_probs = three_model.predict_proba(panel_row, feature_cols)
    five_probs = five_model.predict_proba(panel_row, feature_cols)
    mid_probs = mid_model.predict_proba(panel_row, feature_cols)
    short_profiles = _predict_quantile_profiles(
        panel_row,
        feature_cols=feature_cols,
        q_models=short_q_models,
    )
    mid_profiles = _predict_quantile_profiles(
        panel_row,
        feature_cols=feature_cols,
        q_models=mid_q_models,
    )

    symbols = panel_row["symbol"].astype(str).to_numpy()
    sectors = panel_row.get("sector", pd.Series(["其他"] * len(panel_row), index=panel_row.index)).fillna("其他").astype(str).to_numpy()
    if "tradability_status" in panel_row.columns:
        statuses = panel_row["tradability_status"].fillna("normal").astype(str).to_numpy()
    else:
        statuses = np.full(len(panel_row), "normal", dtype=object)
    realized_1d_arr = panel_row.get("excess_ret_1_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_2d_arr = panel_row.get("excess_ret_2_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_3d_arr = panel_row.get("excess_ret_3_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_5d_arr = panel_row.get("excess_ret_5_vs_mkt", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    realized_20d_arr = panel_row.get("excess_ret_20_vs_sector", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    latest_close_arr = panel_row.get("close", pd.Series(np.nan, index=panel_row.index)).astype(float).to_numpy()
    short_expected_arr = short_profiles["expected_return"].to_numpy(dtype=float)
    mid_expected_arr = mid_profiles["expected_return"].to_numpy(dtype=float)

    interim_rows: list[dict[str, float | str]] = []
    sector_mid_total: dict[str, float] = {}
    sector_counts: dict[str, int] = {}

    for idx, symbol in enumerate(symbols):
        sector = sectors[idx]
        short_prob = float(short_probs[idx])
        two_prob = float(two_probs[idx])
        three_prob = float(three_probs[idx])
        five_prob = float(five_probs[idx])
        mid_prob = float(mid_probs[idx])
        short_expected_ret = float(short_expected_arr[idx])
        mid_expected_ret = float(mid_expected_arr[idx])
        interim_rows.append(
            {
                "idx": idx,
                "symbol": symbol,
                "sector": sector,
                "short_prob": short_prob,
                "two_prob": two_prob,
                "three_prob": three_prob,
                "five_prob": five_prob,
                "mid_prob": mid_prob,
                "short_expected_ret": short_expected_ret,
                "mid_expected_ret": mid_expected_ret,
                "score": _distributional_score(
                    short_prob=short_prob,
                    two_prob=two_prob,
                    three_prob=three_prob,
                    five_prob=five_prob,
                    mid_prob=mid_prob,
                    short_expected_ret=short_expected_ret,
                    mid_expected_ret=mid_expected_ret,
                ),
            }
        )
        sector_mid_total[sector] = sector_mid_total.get(sector, 0.0) + mid_prob
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    sector_avg_mid = {
        sector: sector_mid_total[sector] / max(1, sector_counts[sector])
        for sector in sector_mid_total
    }
    scored_rows: list[dict[str, float | str]] = []
    out: list[StockForecastState] = []
    for item in interim_rows:
        idx = int(item["idx"])
        short_prob = float(item["short_prob"])
        two_prob = float(item["two_prob"])
        three_prob = float(item["three_prob"])
        five_prob = float(item["five_prob"])
        mid_prob = float(item["mid_prob"])
        short_expected_ret = float(item["short_expected_ret"])
        mid_expected_ret = float(item["mid_expected_ret"])
        sector = str(item["sector"])
        status = str(statuses[idx] or "normal")
        tradeability = _clip(
            1.0
            - (
                0.18 * abs(short_prob - two_prob)
                + 0.22 * abs(two_prob - three_prob)
                + 0.25 * abs(three_prob - five_prob)
                + 0.35 * abs(five_prob - mid_prob)
            ),
            0.0,
            1.0,
        )
        tradeability = min(tradeability, _status_tradeability_limit(status))
        expected_anchor = float(np.clip(mid_expected_ret / 0.20, -0.5, 0.5))
        event_impact = float(_clip(0.5 + short_expected_ret / 0.06, 0.0, 1.0))
        if status in {"halted", "delisted"}:
            event_impact = 0.0
        out.append(
            StockForecastState(
                symbol=str(item["symbol"]),
                sector=sector,
                up_1d_prob=short_prob,
                up_2d_prob=two_prob,
                up_3d_prob=three_prob,
                up_5d_prob=five_prob,
                up_10d_prob=float(0.45 * five_prob + 0.55 * mid_prob),
                up_20d_prob=mid_prob,
                excess_vs_sector_prob=float(
                    _clip(
                        mid_prob
                        - sector_avg_mid.get(sector, mid_prob)
                        + 0.5
                        + 0.1 * expected_anchor,
                        0.0,
                        1.0,
                    )
                ),
                event_impact_score=event_impact,
                tradeability_score=float(tradeability),
                alpha_score=float(item["score"]),
                tradability_status=status,
                latest_close=float(latest_close_arr[idx]),
                horizon_forecasts=_build_horizon_forecasts(
                    latest_close=float(latest_close_arr[idx]),
                    horizon_probs={
                        "1d": short_prob,
                        "2d": two_prob,
                        "3d": three_prob,
                        "5d": five_prob,
                        "10d": float(0.45 * five_prob + 0.55 * mid_prob),
                        "20d": mid_prob,
                    },
                    short_profile=_ReturnQuantileProfile(
                        expected_return=float(short_profiles["expected_return"].iloc[idx]),
                        q10=float(short_profiles["q10"].iloc[idx]),
                        q30=float(short_profiles["q30"].iloc[idx]),
                        q20=float(short_profiles["q20"].iloc[idx]),
                        q50=float(short_profiles["q50"].iloc[idx]),
                        q70=float(short_profiles["q70"].iloc[idx]),
                        q80=float(short_profiles["q80"].iloc[idx]),
                        q90=float(short_profiles["q90"].iloc[idx]),
                    ),
                    mid_profile=_ReturnQuantileProfile(
                        expected_return=float(mid_profiles["expected_return"].iloc[idx]),
                        q10=float(mid_profiles["q10"].iloc[idx]),
                        q30=float(mid_profiles["q30"].iloc[idx]),
                        q20=float(mid_profiles["q20"].iloc[idx]),
                        q50=float(mid_profiles["q50"].iloc[idx]),
                        q70=float(mid_profiles["q70"].iloc[idx]),
                        q80=float(mid_profiles["q80"].iloc[idx]),
                        q90=float(mid_profiles["q90"].iloc[idx]),
                    ),
                    tradability_status=status,
                ),
            )
        )
        realized_1d = np.nan
        realized_2d = np.nan
        realized_3d = np.nan
        realized_5d = np.nan
        realized_20d = np.nan
        realized_1d = float(realized_1d_arr[idx])
        realized_2d = float(realized_2d_arr[idx])
        realized_3d = float(realized_3d_arr[idx])
        realized_5d = float(realized_5d_arr[idx])
        realized_20d = float(realized_20d_arr[idx])
        scored_rows.append(
            {
                "symbol": str(item["symbol"]),
                "score": float(item["score"]),
                "realized_ret_1d": float(realized_1d),
                "realized_ret_2d": float(realized_2d),
                "realized_ret_3d": float(realized_3d),
                "realized_ret_5d": float(realized_5d),
                "realized_ret_20d": float(realized_20d),
            }
        )
    out.sort(
        key=lambda stock: (
            _distributional_score(
                short_prob=stock.up_1d_prob,
                two_prob=getattr(stock, "up_2d_prob", 0.5),
                three_prob=getattr(stock, "up_3d_prob", 0.5),
                five_prob=stock.up_5d_prob,
                mid_prob=stock.up_20d_prob,
                short_expected_ret=float(stock.event_impact_score - 0.5) * 0.06,
                mid_expected_ret=float(stock.excess_vs_sector_prob - 0.5) * 0.20,
            ),
            stock.up_20d_prob,
            stock.tradeability_score,
        ),
        reverse=True,
    )
    return out, pd.DataFrame(scored_rows)


def _panel_slice_metrics(
    scored_rows: pd.DataFrame,
    *,
    realized_col: str = "realized_ret_20d",
    top_k: int = 3,
) -> tuple[float, float, float, float]:
    if scored_rows.empty:
        return 0.0, 0.0, 0.0, 0.0
    frame = scored_rows.dropna(subset=["score", realized_col]).copy()
    if len(frame) < 2:
        return 0.0, 0.0, 0.0, 0.0
    rank_ic = float(frame["score"].corr(frame[realized_col], method="spearman"))
    if rank_ic != rank_ic:
        rank_ic = 0.0
    bucket_n = max(1, int(np.ceil(len(frame) * 0.1)))
    ranked = frame.sort_values("score", ascending=False).reset_index(drop=True)
    top_bucket = ranked.head(bucket_n)
    bottom_bucket = ranked.tail(bucket_n)
    top_decile_return = float(top_bucket[realized_col].mean()) if not top_bucket.empty else 0.0
    bottom_return = float(bottom_bucket[realized_col].mean()) if not bottom_bucket.empty else 0.0
    top_bottom_spread = float(top_decile_return - bottom_return)
    top_k_n = max(1, min(int(top_k), len(ranked)))
    top_k_hit_rate = float((ranked.head(top_k_n)[realized_col] > 0.0).mean())
    return rank_ic, top_decile_return, top_bottom_spread, top_k_hit_rate


def _panel_horizon_metrics(scored_rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    mapping = {
        "1d": "realized_ret_1d",
        "2d": "realized_ret_2d",
        "3d": "realized_ret_3d",
        "5d": "realized_ret_5d",
        "20d": "realized_ret_20d",
    }
    out: dict[str, dict[str, float]] = {}
    for horizon, realized_col in mapping.items():
        rank_ic, top_decile_ret, top_bottom_spread, top_k_hit_rate = _panel_slice_metrics(
            scored_rows,
            realized_col=realized_col,
        )
        out[horizon] = {
            "rank_ic": float(rank_ic),
            "top_decile_return": float(top_decile_ret),
            "top_bottom_spread": float(top_bottom_spread),
            "top_k_hit_rate": float(top_k_hit_rate),
        }
    return out


def build_strategy_snapshot(
    *,
    strategy_id: str,
    universe_id: str = "top_liquid_200",
    universe_size: int = 0,
    universe_generation_rule: str = "",
    source_universe_manifest_path: str = "",
    info_manifest_path: str = "",
    info_hash: str = "",
    info_shadow_enabled: bool = False,
    external_signal_manifest_path: str = "",
    external_signal_version: str = "",
    external_signal_enabled: bool = False,
    capital_flow_snapshot: dict[str, object] | None = None,
    macro_context_snapshot: dict[str, object] | None = None,
    generator_manifest_path: str = "",
    generator_version: str = "",
    generator_hash: str = "",
    coarse_pool_size: int = 0,
    refined_pool_size: int = 0,
    selected_pool_size: int = 0,
    theme_allocations: list[dict[str, object]] | None = None,
    run_id: str = "",
    data_window: str = "",
    model_hashes: dict[str, str] | None = None,
    policy_hash: str = "",
    universe_hash: str = "",
    created_at: str = "",
    snapshot_hash: str = "",
    config_hash: str = "",
    manifest_path: str = "",
    use_us_index_context: bool = False,
    us_index_source: str = "",
) -> StrategySnapshot:
    return StrategySnapshot(
        strategy_id=str(strategy_id).strip() or "swing_v2",
        universe_id=str(universe_id).strip() or "top_liquid_200",
        feature_set_version="fset_v2_core",
        market_model_id="mkt_lr_v2",
        sector_model_id="sector_lr_v2",
        stock_model_id="stock_lr_v2",
        cross_section_model_id="cross_section_v2",
        policy_version="policy_v2_rules",
        execution_version="exec_t1_v2",
        universe_size=int(universe_size),
        universe_generation_rule=str(universe_generation_rule),
        source_universe_manifest_path=str(source_universe_manifest_path),
        info_manifest_path=str(info_manifest_path),
        info_hash=str(info_hash),
        info_shadow_enabled=bool(info_shadow_enabled),
        external_signal_manifest_path=str(external_signal_manifest_path),
        external_signal_version=str(external_signal_version),
        external_signal_enabled=bool(external_signal_enabled),
        capital_flow_snapshot=dict(capital_flow_snapshot or {}),
        macro_context_snapshot=dict(macro_context_snapshot or {}),
        generator_manifest_path=str(generator_manifest_path),
        generator_version=str(generator_version),
        generator_hash=str(generator_hash),
        coarse_pool_size=int(coarse_pool_size),
        refined_pool_size=int(refined_pool_size),
        selected_pool_size=int(selected_pool_size),
        theme_allocations=list(theme_allocations or []),
        run_id=str(run_id),
        data_window=str(data_window),
        model_hashes=dict(model_hashes or {}),
        policy_hash=str(policy_hash),
        universe_hash=str(universe_hash),
        created_at=str(created_at),
        snapshot_hash=str(snapshot_hash),
        config_hash=str(config_hash),
        manifest_path=str(manifest_path),
        use_us_index_context=bool(use_us_index_context),
        us_index_source=str(us_index_source),
    )


def _load_v2_runtime_settings(
    *,
    config_path: str,
    source: str | None = None,
    tushare_token: str | None = None,
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
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    path = Path(config_path)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            payload = raw

    common = payload.get("common", {}) if isinstance(payload.get("common"), dict) else {}
    daily = payload.get("daily", {}) if isinstance(payload.get("daily"), dict) else {}
    _configure_v2_tushare_token(explicit_token=tushare_token, daily=daily, common=common)

    def pick(key: str, default: object) -> object:
        return _coalesce(daily.get(key), common.get(key), default)

    resolved_universe_limit = int(
        universe_limit
        if universe_limit is not None
        else int(pick("universe_limit", 5))
    )
    default_dynamic_universe = resolved_universe_limit >= 150
    requested_dynamic_universe = (
        bool(dynamic_universe)
        if dynamic_universe is not None
        else _parse_boolish(pick("dynamic_universe_enabled", default_dynamic_universe), default_dynamic_universe)
    )
    default_universe_file = (
        str(pick("generated_universe_base_file", "config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json"))
        if requested_dynamic_universe
        else str(pick("universe_file", "config/universe_smoke_5.json"))
    )
    resolved_universe_file = (
        str(universe_file).strip()
        if universe_file is not None and str(universe_file).strip()
        else default_universe_file
    )
    resolved_universe_tier = (
        str(universe_tier).strip()
        if universe_tier is not None and str(universe_tier).strip()
        else ("" if resolved_universe_file else str(pick("universe_tier", "")))
    )
    resolved_generator_target_size = int(
        generator_target_size
        if generator_target_size is not None
        else int(pick("dynamic_universe_target_size", resolved_universe_limit or 300))
    )
    resolved_generator_coarse_size = int(
        generator_coarse_size
        if generator_coarse_size is not None
        else int(pick("dynamic_universe_coarse_size", max(1000, resolved_generator_target_size * 3)))
    )

    return {
        "config_path": str(config_path),
        "watchlist": str(pick("watchlist", "config/watchlist.json")),
        "source": str(source).strip() if source is not None and str(source).strip() else str(pick("source", "auto")),
        "data_dir": str(pick("data_dir", "data")),
        "start": str(pick("start", "2018-01-01")),
        "end": str(pick("end", "2099-12-31")),
        "min_train_days": int(pick("min_train_days", 240)),
        "step_days": int(pick("step_days", 20)),
        "l2": float(pick("l2", 0.8)),
        "max_positions": int(pick("max_positions", 5)),
        "use_margin_features": bool(pick("use_margin_features", True)),
        "margin_market_file": str(pick("margin_market_file", "input/margin_market.csv")),
        "margin_stock_file": str(pick("margin_stock_file", "input/margin_stock.csv")),
        "use_us_index_context": (
            bool(use_us_index_context)
            if use_us_index_context is not None
            else _parse_boolish(pick("use_us_index_context", False), False)
        ),
        "us_index_source": (
            str(us_index_source).strip()
            if us_index_source is not None and str(us_index_source).strip()
            else str(pick("us_index_source", "akshare")).strip()
        ),
        "use_us_sector_etf_context": _parse_boolish(pick("use_us_sector_etf_context", False), False),
        "use_cn_etf_context": _parse_boolish(pick("use_cn_etf_context", False), False),
        "cn_etf_source": str(pick("cn_etf_source", "akshare")).strip(),
        "universe_tier": resolved_universe_tier,
        "active_default_universe_tier": str(pick("active_default_universe_tier", "favorites_16")),
        "candidate_default_universe_tier": str(pick("candidate_default_universe_tier", "generated_80")),
        "favorites_universe_file": str(pick("favorites_universe_file", "config/universe_favorites.json")),
        "generated_universe_base_file": str(
            pick("generated_universe_base_file", "config/universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json")
        ),
        "baseline_reference_run_id": str(pick("baseline_reference_run_id", "20260308_211808")),
        "news_file": str(pick("news_file", "input/news_parts")),
        "info_file": (
            str(info_file).strip()
            if info_file is not None and str(info_file).strip()
            else (
                str(event_file).strip()
                if event_file is not None and str(event_file).strip()
                else str(pick("info_file", pick("news_file", "input/info_parts")))
            )
        ),
        "event_file": (
            str(event_file).strip()
            if event_file is not None and str(event_file).strip()
            else str(pick("event_file", pick("info_file", "input/info_parts")))
        ),
        "info_lookback_days": int(
            info_lookback_days
            if info_lookback_days is not None
            else int(pick("info_lookback_days", pick("news_lookback_days", 45)))
        ),
        "event_lookback_days": int(
            pick(
                "event_lookback_days",
                info_lookback_days if info_lookback_days is not None else pick("info_lookback_days", 45),
            )
        ),
        "learned_info_lookback_days": int(pick("learned_info_lookback_days", pick("learned_news_lookback_days", 720))),
        "info_half_life_days": float(
            info_half_life_days
            if info_half_life_days is not None
            else float(pick("info_half_life_days", pick("news_half_life_days", 10.0)))
        ),
        "capital_flow_file": (
            str(capital_flow_file).strip()
            if capital_flow_file is not None and str(capital_flow_file).strip()
            else str(pick("capital_flow_file", ""))
        ),
        "macro_file": (
            str(macro_file).strip()
            if macro_file is not None and str(macro_file).strip()
            else str(pick("macro_file", ""))
        ),
        "capital_flow_lookback_days": int(pick("capital_flow_lookback_days", 20)),
        "macro_lookback_days": int(pick("macro_lookback_days", 60)),
        "dynamic_universe_enabled": (
            bool(dynamic_universe)
            if dynamic_universe is not None
            else requested_dynamic_universe
        ),
        "generator_target_size": resolved_generator_target_size,
        "generator_coarse_size": resolved_generator_coarse_size,
        "generator_theme_aware": (
            bool(generator_theme_aware)
            if generator_theme_aware is not None
            else _parse_boolish(pick("generator_theme_aware", True), True)
        ),
        "generator_use_concepts": (
            bool(generator_use_concepts)
            if generator_use_concepts is not None
            else _parse_boolish(pick("generator_use_concepts", True), True)
        ),
        "main_board_only_universe": _parse_boolish(
            pick("main_board_only_universe", pick("main_board_only_recommendations", False)),
            False,
        ),
        "main_board_only_recommendations": _parse_boolish(
            pick("main_board_only_recommendations", False),
            False,
        ),
        "dynamic_universe_min_history_days": int(pick("dynamic_universe_min_history_days", 480)),
        "dynamic_universe_min_recent_amount": float(pick("dynamic_universe_min_recent_amount", 2.0e7)),
        "dynamic_universe_theme_cap_ratio": float(pick("dynamic_universe_theme_cap_ratio", 0.16)),
        "dynamic_universe_theme_floor_count": int(pick("dynamic_universe_theme_floor_count", 2)),
        "dynamic_universe_turnover_quality_weight": float(pick("dynamic_universe_turnover_quality_weight", 0.25)),
        "dynamic_universe_theme_weight": float(pick("dynamic_universe_theme_weight", 0.18)),
        "external_signals": (
            bool(external_signals)
            if external_signals is not None
            else _parse_boolish(pick("external_signals", True), True)
        ),
        "external_signal_version": str(pick("external_signal_version", "v1")),
        "event_risk_cutoff": float(pick("event_risk_cutoff", 0.55)),
        "catalyst_boost_cap": float(pick("catalyst_boost_cap", 0.12)),
        "flow_exposure_cap": float(pick("flow_exposure_cap", 0.08)),
        "market_info_strength": float(pick("market_info_strength", pick("market_news_strength", 0.9))),
        "stock_info_strength": float(pick("stock_info_strength", pick("stock_news_strength", 1.1))),
        "use_info_fusion": (
            bool(use_info_fusion)
            if use_info_fusion is not None
            else _parse_boolish(pick("use_info_fusion", False), False)
        ),
        "use_learned_info_fusion": _parse_boolish(pick("use_learned_info_fusion", pick("use_learned_news_fusion", True)), True),
        "learned_info_min_samples": int(pick("learned_info_min_samples", pick("learned_news_min_samples", 80))),
        "learned_info_l2": float(pick("learned_info_l2", pick("learned_news_l2", 0.8))),
        "learned_info_holdout_ratio": float(pick("learned_info_holdout_ratio", pick("learned_holdout_ratio", 0.2))),
        "info_source_mode": (
            str(info_source_mode).strip()
            if info_source_mode is not None and str(info_source_mode).strip()
            else str(pick("info_source_mode", "layered")).strip()
        ),
        "info_shadow_only": (
            bool(info_shadow_only)
            if info_shadow_only is not None
            else _parse_boolish(pick("info_shadow_only", True), True)
        ),
        "info_types": _parse_csv_tokens(
            info_types if info_types is not None else pick("info_types", "news,announcement,research"),
            default=("news", "announcement", "research"),
        ),
        "info_subsets": _parse_csv_tokens(
            info_subsets if info_subsets is not None else pick("info_subsets", "market_news,announcements,research"),
            default=("market_news", "announcements", "research"),
        ),
        "announcement_event_tags": _parse_csv_tokens(
            pick(
                "announcement_event_tags",
                "earnings_positive,earnings_negative,guidance_positive,guidance_negative,contract_win,contract_loss,regulatory_positive,regulatory_negative,share_reduction,share_increase,trading_halt,delisting_risk",
            ),
            default=(
                "earnings_positive",
                "earnings_negative",
                "guidance_positive",
                "guidance_negative",
                "contract_win",
                "contract_loss",
                "regulatory_positive",
                "regulatory_negative",
                "share_reduction",
                "share_increase",
                "trading_halt",
                "delisting_risk",
            ),
        ),
        "universe_file": resolved_universe_file,
        "universe_limit": resolved_universe_limit,
    }


def _extract_universe_rows(payload: object) -> list[dict[str, str]]:
    if isinstance(payload, list):
        return [
            {
                "symbol": str(item),
                "name": str(item),
                "sector": "其他",
            }
            for item in payload
        ]
    if isinstance(payload, dict):
        raw_rows = payload.get("stocks", [])
        if isinstance(raw_rows, list):
            out: list[dict[str, str]] = []
            for item in raw_rows:
                if not isinstance(item, dict):
                    continue
                symbol = str(item.get("symbol", "")).strip()
                if not symbol:
                    continue
                out.append(
                    {
                        "symbol": symbol,
                        "name": str(item.get("name", symbol)),
                        "sector": str(item.get("sector", "其他")),
                    }
                )
            return out
    return []


def _hydrate_universe_metadata(
    *,
    universe_file: str,
    universe_limit: int,
    universe_tier: str = "",
    universe_generation_rule: str = "",
) -> dict[str, object]:
    path = Path(str(universe_file))
    rows: list[dict[str, str]] = []
    payload = _load_json_dict(path)
    if payload:
        rows = _extract_universe_rows(payload)
    elif path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            rows = _extract_universe_rows(raw)
        except Exception:
            rows = []
    symbols = [str(item["symbol"]) for item in rows]
    universe_id = str(universe_tier).strip() or path.stem or "v2_universe"
    generation_rule = str(universe_generation_rule).strip()
    if not generation_rule and payload:
        generation_rule = str(payload.get("generation_rule", ""))
    if not generation_rule:
        generation_rule = "external_universe_file"
    return {
        "universe_tier": str(universe_tier).strip(),
        "universe_id": universe_id,
        "universe_size": int(len(symbols) if symbols else max(0, int(universe_limit))),
        "universe_generation_rule": generation_rule,
        "source_universe_manifest_path": str(path.resolve()) if path.exists() else str(path),
        "symbols": symbols,
        "symbol_count": int(len(symbols)),
        "universe_hash": _sha256_file(path) or _stable_json_hash(symbols),
    }


def _resolve_v2_universe_settings(
    *,
    settings: dict[str, object],
    cache_root: str,
) -> dict[str, object]:
    resolved = dict(settings)
    requested_tier = str(resolved.get("universe_tier", "")).strip()
    dynamic_universe_enabled = _parse_boolish(resolved.get("dynamic_universe_enabled", False), False)
    generator_target_size = int(
        resolved.get("generator_target_size", resolved.get("universe_limit", 300)) or resolved.get("universe_limit", 300)
    )
    generator_source_file = str(resolved.get("universe_file", "")).strip()
    if requested_tier and dynamic_universe_enabled:
        normalized_tier = normalize_universe_tier(requested_tier)
        if normalized_tier.startswith("generated_"):
            if not generator_source_file:
                generator_source_file = str(resolved.get("generated_universe_base_file", generator_source_file)).strip()
            tier_digits = "".join(ch for ch in normalized_tier if ch.isdigit())
            if tier_digits:
                generator_target_size = int(tier_digits)
    if dynamic_universe_enabled and (generator_source_file or requested_tier):
        dynamic_result = generate_dynamic_universe(
            universe_file=generator_source_file,
            data_dir=str(resolved.get("data_dir", "")),
            cache_root=str(cache_root),
            target_size=max(1, int(generator_target_size)),
            coarse_size=max(generator_target_size, int(resolved.get("generator_coarse_size", 1000))),
            theme_aware=_parse_boolish(resolved.get("generator_theme_aware", True), True),
            use_concepts=_parse_boolish(resolved.get("generator_use_concepts", True), True),
            end_date=str(resolved.get("end", "")),
            min_history_days=int(resolved.get("dynamic_universe_min_history_days", 480)),
            min_recent_amount=float(resolved.get("dynamic_universe_min_recent_amount", 2.0e7)),
            theme_cap_ratio=float(resolved.get("dynamic_universe_theme_cap_ratio", 0.16)),
            theme_floor_count=int(resolved.get("dynamic_universe_theme_floor_count", 2)),
            turnover_quality_weight=float(resolved.get("dynamic_universe_turnover_quality_weight", 0.25)),
            theme_weight=float(resolved.get("dynamic_universe_theme_weight", 0.18)),
            main_board_only=_parse_boolish(resolved.get("main_board_only_universe", False), False),
            refresh_cache=_parse_boolish(resolved.get("refresh_cache", False), False),
        )
        manifest = dynamic_result.generator_manifest
        selected_symbols = [
            str(item.get("symbol", ""))
            for item in dynamic_result.selected_300
            if str(item.get("symbol", "")).strip()
        ]
        manifest_path_text = str(manifest.manifest_path)
        universe_manifest_path = (
            manifest_path_text.replace(".generator.json", ".json")
            if manifest_path_text.endswith(".generator.json")
            else str(generator_source_file)
        )
        resolved["universe_tier"] = str(requested_tier)
        resolved["universe_file"] = universe_manifest_path
        resolved["universe_limit"] = int(len(selected_symbols))
        resolved["universe_id"] = f"dynamic_{int(generator_target_size)}"
        resolved["universe_size"] = int(len(selected_symbols))
        resolved["universe_generation_rule"] = f"{manifest.generator_version}: coarse->{manifest.coarse_pool_size} select->{manifest.selected_pool_size}"
        resolved["source_universe_manifest_path"] = str(manifest.source_universe_path or generator_source_file)
        resolved["symbols"] = selected_symbols
        resolved["symbol_count"] = int(len(selected_symbols))
        resolved["universe_hash"] = str(manifest.generator_hash)
        resolved["generator_manifest_path"] = str(manifest.manifest_path)
        resolved["generator_version"] = str(manifest.generator_version)
        resolved["generator_hash"] = str(manifest.generator_hash)
        resolved["coarse_pool_size"] = int(manifest.coarse_pool_size)
        resolved["refined_pool_size"] = int(manifest.refined_pool_size)
        resolved["selected_pool_size"] = int(manifest.selected_pool_size)
        resolved["theme_allocations"] = [asdict(item) for item in manifest.theme_allocations]
        return resolved

    if requested_tier:
        normalized_tier = normalize_universe_tier(requested_tier)
        catalog_dir = Path(str(cache_root)) / "universe_catalog"
        manifest_token = _stable_json_hash(
            {
                "tier": normalized_tier,
                "data_dir": str(resolved.get("data_dir", "")),
                "favorites_universe_file": str(resolved.get("favorites_universe_file", "")),
                "generated_universe_base_file": str(resolved.get("generated_universe_base_file", "")),
            }
        )[:12]
        manifest_path = catalog_dir / f"{normalized_tier}_{manifest_token}.json"
        built = build_predefined_universe(
            tier_id=normalized_tier,
            data_dir=str(resolved.get("data_dir", "")),
            favorites_file=str(resolved.get("favorites_universe_file", "")),
            generated_base_file=str(resolved.get("generated_universe_base_file", "")),
            output_path=manifest_path,
            exclude_symbols=(),
        )
        resolved["universe_tier"] = normalized_tier
        resolved["universe_file"] = str(manifest_path.resolve())
        resolved["universe_limit"] = int(built.universe_size or len(built.rows))
        resolved["universe_id"] = str(built.universe_id or normalized_tier)
        resolved["universe_size"] = int(built.universe_size or len(built.rows))
        resolved["universe_generation_rule"] = str(built.generation_rule)
        resolved["source_universe_manifest_path"] = str(built.manifest_path or manifest_path.resolve())
        resolved["symbols"] = [str(item.symbol) for item in built.rows]
        resolved["symbol_count"] = int(len(built.rows))
        resolved["universe_hash"] = _sha256_file(manifest_path) or _stable_json_hash(resolved["symbols"])
        return resolved

    hydrated = _hydrate_universe_metadata(
        universe_file=str(resolved.get("universe_file", "")),
        universe_limit=int(resolved.get("universe_limit", 0)),
        universe_generation_rule=str(resolved.get("universe_generation_rule", "")),
    )
    resolved.update(hydrated)
    return resolved


def _build_market_and_cross_section_states(
    *,
    market_symbol: str,
    source: str,
    data_dir: str,
    start: str,
    end: str,
    use_margin_features: bool,
    margin_market_file: str,
    use_us_index_context: bool,
    us_index_source: str,
    use_us_sector_etf_context: bool = False,
    use_cn_etf_context: bool = False,
    cn_etf_source: str = "akshare",
    market_short_prob: float,
    market_two_prob: float | None,
    market_three_prob: float | None,
    market_five_prob: float | None,
    market_mid_prob: float,
    market_short_profile: _ReturnQuantileProfile | None = None,
    market_mid_profile: _ReturnQuantileProfile | None = None,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    market_raw = load_symbol_daily(
        symbol=market_symbol,
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=source,
        data_dir=data_dir,
        start=start,
        end=end,
        market_dates=market_feat_base["date"],
        use_margin_features=use_margin_features,
        margin_market_file=margin_market_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        use_us_sector_etf_context=use_us_sector_etf_context,
        use_cn_etf_context=use_cn_etf_context,
        cn_etf_source=cn_etf_source,
    )
    market_frame = market_feat_base.merge(market_context.frame, on="date", how="left")
    latest = market_frame.sort_values("date").iloc[-1]
    state = decide_market_state(float(market_short_prob), float(market_mid_prob))
    cross_section_record = forecast_cross_section_state(market_frame)

    mkt_vol_20 = float(latest.get("mkt_volatility_20", 0.0))
    if mkt_vol_20 != mkt_vol_20:
        mkt_vol_20 = 0.0
    mkt_vol_60 = float(latest.get("mkt_volatility_60", mkt_vol_20))
    if mkt_vol_60 != mkt_vol_60:
        mkt_vol_60 = mkt_vol_20
    mkt_vol_60 = max(1e-6, mkt_vol_60)
    if mkt_vol_20 >= mkt_vol_60 * 1.15:
        volatility_regime = "high"
    elif mkt_vol_20 <= mkt_vol_60 * 0.85:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    latest_close = float(_safe_float(latest.get("close", np.nan), np.nan))
    horizon_probs = {
        "1d": float(market_short_prob),
        "2d": float(
            market_two_prob
            if market_two_prob is not None
            else (0.65 * market_short_prob + 0.35 * (market_five_prob if market_five_prob is not None else market_mid_prob))
        ),
        "3d": float(
            market_three_prob
            if market_three_prob is not None
            else (0.35 * market_short_prob + 0.65 * (market_five_prob if market_five_prob is not None else market_mid_prob))
        ),
        "5d": float(
            market_five_prob if market_five_prob is not None else (0.6 * market_short_prob + 0.4 * market_mid_prob)
        ),
        "10d": float(
            0.45 * (market_five_prob if market_five_prob is not None else (0.6 * market_short_prob + 0.4 * market_mid_prob))
            + 0.55 * market_mid_prob
        ),
        "20d": float(market_mid_prob),
    }
    market_facts = _market_facts_from_row(latest)

    market = MarketForecastState(
        as_of_date=str(latest["date"].date()),
        up_1d_prob=horizon_probs["1d"],
        up_2d_prob=horizon_probs["2d"],
        up_3d_prob=horizon_probs["3d"],
        up_5d_prob=horizon_probs["5d"],
        up_10d_prob=horizon_probs["10d"],
        up_20d_prob=float(market_mid_prob),
        trend_state=str(state.state_code),
        drawdown_risk=_clip(abs(float(latest.get("mkt_drawdown_20", 0.0) or 0.0)), 0.0, 1.0),
        volatility_regime=volatility_regime,
        liquidity_stress=_clip(0.5 - float(cross_section_record.breadth_strength), 0.0, 1.0),
        latest_close=latest_close,
        horizon_forecasts=_build_horizon_forecasts(
            latest_close=latest_close,
            horizon_probs=horizon_probs,
            short_profile=market_short_profile,
            mid_profile=market_mid_profile,
        ),
        market_facts=market_facts,
    )
    cross_section = CrossSectionForecastState(
        as_of_date=str(cross_section_record.as_of_date.date()),
        large_vs_small_bias=float(cross_section_record.large_vs_small_bias),
        growth_vs_value_bias=float(cross_section_record.growth_vs_value_bias),
        fund_flow_strength=float(cross_section_record.fund_flow_strength),
        margin_risk_on_score=float(cross_section_record.margin_risk_on_score),
        breadth_strength=float(cross_section_record.breadth_strength),
        leader_participation=float(cross_section_record.leader_participation),
        weak_stock_ratio=float(cross_section_record.weak_stock_ratio),
    )
    return market, cross_section


def _build_stock_states_from_rows(
    rows: list[object],
    sector_map: dict[str, str],
    *,
    sector_strength_map: dict[str, float] | None = None,
) -> list[StockForecastState]:
    out: list[StockForecastState] = []
    sector_mid_map: dict[str, float] = {}
    sector_counts: dict[str, int] = {}
    for row in rows:
        sector = sector_map.get(row.symbol, "其他")
        sector_mid_map[sector] = sector_mid_map.get(sector, 0.0) + float(row.mid_prob)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    sector_avg_mid = {
        sector: sector_mid_map[sector] / max(1, sector_counts[sector])
        for sector in sector_mid_map
    }

    for row in rows:
        sector = sector_map.get(row.symbol, "其他")
        status = str(getattr(row, "tradability_status", "normal") or "normal")
        two_prob = _safe_float(
            getattr(row, "two_prob", 0.65 * float(row.short_prob) + 0.35 * float(getattr(row, "five_prob", row.mid_prob))),
            0.5,
        )
        three_prob = _safe_float(
            getattr(row, "three_prob", 0.35 * float(row.short_prob) + 0.65 * float(getattr(row, "five_prob", row.mid_prob))),
            0.5,
        )
        five_prob = _safe_float(getattr(row, "five_prob", 0.6 * float(row.short_prob) + 0.4 * float(row.mid_prob)), 0.5)
        tradeability = _clip(
            1.0
            - (
                0.18 * abs(float(row.short_prob) - two_prob)
                + 0.22 * abs(two_prob - three_prob)
                + 0.25 * abs(three_prob - five_prob)
                + 0.35 * abs(five_prob - float(row.mid_prob))
            ),
            0.0,
            1.0,
        )
        tradeability = min(tradeability, _status_tradeability_limit(status))
        short_expected_ret = _safe_float(getattr(row, "short_expected_ret", 0.0), 0.0)
        mid_expected_ret = _safe_float(getattr(row, "mid_expected_ret", 0.0), 0.0)
        sector_excess_anchor = 0.0 if sector_strength_map is None else float(sector_strength_map.get(sector, 0.0))
        expected_anchor = float(np.clip(mid_expected_ret / 0.20, -0.5, 0.5))
        event_impact = float(_clip(0.5 + short_expected_ret / 0.06, 0.0, 1.0))
        if status in {"halted", "delisted"}:
            event_impact = 0.0
        latest_close = float(_safe_float(getattr(row, "latest_close", getattr(row, "close", np.nan)), np.nan))
        short_profile = _ReturnQuantileProfile(
            expected_return=float(short_expected_ret),
            q10=float(_safe_float(getattr(row, "short_q10", np.nan), np.nan)),
            q30=float(_safe_float(getattr(row, "short_q30", np.nan), np.nan)),
            q20=float(_safe_float(getattr(row, "short_q20", np.nan), np.nan)),
            q50=float(_safe_float(getattr(row, "short_q50", np.nan), np.nan)),
            q70=float(_safe_float(getattr(row, "short_q70", np.nan), np.nan)),
            q80=float(_safe_float(getattr(row, "short_q80", np.nan), np.nan)),
            q90=float(_safe_float(getattr(row, "short_q90", np.nan), np.nan)),
        )
        mid_profile = _ReturnQuantileProfile(
            expected_return=float(mid_expected_ret),
            q10=float(_safe_float(getattr(row, "mid_q10", np.nan), np.nan)),
            q30=float(_safe_float(getattr(row, "mid_q30", np.nan), np.nan)),
            q20=float(_safe_float(getattr(row, "mid_q20", np.nan), np.nan)),
            q50=float(_safe_float(getattr(row, "mid_q50", np.nan), np.nan)),
            q70=float(_safe_float(getattr(row, "mid_q70", np.nan), np.nan)),
            q80=float(_safe_float(getattr(row, "mid_q80", np.nan), np.nan)),
            q90=float(_safe_float(getattr(row, "mid_q90", np.nan), np.nan)),
        )
        horizon_probs = {
            "1d": float(row.short_prob),
            "2d": float(two_prob),
            "3d": float(three_prob),
            "5d": float(five_prob),
            "10d": float(0.45 * five_prob + 0.55 * float(row.mid_prob)),
            "20d": float(row.mid_prob),
        }
        out.append(
            StockForecastState(
                symbol=row.symbol,
                sector=sector,
                up_1d_prob=float(row.short_prob),
                up_2d_prob=float(two_prob),
                up_3d_prob=float(three_prob),
                up_5d_prob=float(five_prob),
                up_10d_prob=float(horizon_probs["10d"]),
                up_20d_prob=float(row.mid_prob),
                excess_vs_sector_prob=float(
                    _clip(
                        float(row.mid_prob)
                        - sector_avg_mid.get(sector, float(row.mid_prob))
                        + 0.5
                        + 0.2 * sector_excess_anchor
                        + 0.1 * expected_anchor,
                        0.0,
                        1.0,
                    )
                ),
                event_impact_score=event_impact,
                tradeability_score=float(tradeability),
                alpha_score=float(
                    _distributional_score(
                        short_prob=float(row.short_prob),
                        two_prob=float(two_prob),
                        three_prob=float(three_prob),
                        five_prob=float(five_prob),
                        mid_prob=float(row.mid_prob),
                        short_expected_ret=float(short_expected_ret),
                        mid_expected_ret=float(mid_expected_ret),
                    )
                ),
                tradability_status=status,
                latest_close=latest_close,
                horizon_forecasts=_build_horizon_forecasts(
                    latest_close=latest_close,
                    horizon_probs=horizon_probs,
                    short_profile=short_profile,
                    mid_profile=mid_profile,
                    tradability_status=status,
                ),
            )
        )
    out.sort(
        key=lambda item: (_stock_policy_score(item), item.up_20d_prob, item.excess_vs_sector_prob),
        reverse=True,
    )
    return out


def compose_state(
    *,
    market: MarketForecastState,
    sectors: list[SectorForecastState],
    stocks: list[StockForecastState],
    cross_section: CrossSectionForecastState,
) -> CompositeState:
    risk_score = max(
        float(market.drawdown_risk),
        float(market.liquidity_stress),
        float(cross_section.weak_stock_ratio),
    )
    if risk_score >= 0.60:
        risk_regime = "risk_off"
    elif risk_score >= 0.40:
        risk_regime = "cautious"
    else:
        risk_regime = "risk_on"

    if market.trend_state == "trend" and cross_section.breadth_strength >= 0.10:
        strategy_mode = "trend_follow"
    elif market.trend_state == "range":
        strategy_mode = "range_rotation"
    else:
        strategy_mode = "defensive"

    ordered_sectors = sorted(sectors, key=lambda item: (item.up_20d_prob, item.relative_strength), reverse=True)
    ordered_stocks = sorted(
        stocks,
        key=lambda item: (_stock_policy_score(item), item.up_20d_prob, item.excess_vs_sector_prob),
        reverse=True,
    )
    mainlines = _build_mainline_states_external(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        stock_score_fn=_stock_policy_score,
    )
    candidate_selection = _build_candidate_selection_state_external(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        mainlines=mainlines,
        strategy_mode=strategy_mode,
        risk_regime=risk_regime,
        stock_score_fn=_stock_policy_score,
    )
    return CompositeState(
        market=market,
        cross_section=cross_section,
        sectors=ordered_sectors,
        stocks=ordered_stocks,
        strategy_mode=strategy_mode,
        risk_regime=risk_regime,
        candidate_selection=candidate_selection,
        mainlines=mainlines,
    )


def _profile_from_horizon_map(
    forecasts: dict[str, HorizonForecast],
    key: str,
) -> _ReturnQuantileProfile | None:
    item = forecasts.get(key)
    if item is None:
        return None
    q10 = float(item.q10)
    q50 = float(item.q50)
    q90 = float(item.q90)
    q30 = float((2.0 * q10 + q50) / 3.0)
    q70 = float((q50 + 2.0 * q90) / 3.0)
    return _ReturnQuantileProfile(
        expected_return=float(item.expected_return),
        q10=q10,
        q30=q30,
        q20=float(0.5 * (q10 + q30)),
        q50=q50,
        q70=q70,
        q80=float(0.5 * (q70 + q90)),
        q90=q90,
    )


def _load_prediction_review_context(
    *,
    manifest: dict[str, object],
    manifest_path: Path | None,
) -> tuple[PredictionReviewState, dict[str, dict[str, float]]]:
    if manifest_path is None:
        return PredictionReviewState(), {}
    backtest_path = _path_from_manifest_entry(manifest.get("backtest_summary"), run_dir=manifest_path.parent)
    payload = _load_json_dict(backtest_path)
    if not payload:
        return PredictionReviewState(), {}
    learned = payload.get("learned")
    baseline = payload.get("baseline")
    summary = learned if isinstance(learned, dict) else (baseline if isinstance(baseline, dict) else payload)
    raw_horizon_metrics = summary.get("horizon_metrics", {}) if isinstance(summary, dict) else {}
    calibration_priors = {
        str(key): {str(metric): float(value) for metric, value in raw.items()}
        for key, raw in raw_horizon_metrics.items()
        if isinstance(raw, dict)
    }
    if "10d" not in calibration_priors and ("5d" in calibration_priors or "20d" in calibration_priors):
        five = calibration_priors.get("5d", {})
        twenty = calibration_priors.get("20d", {})
        calibration_priors["10d"] = {
            "rank_ic": float(0.45 * float(five.get("rank_ic", 0.0)) + 0.55 * float(twenty.get("rank_ic", 0.0))),
            "top_k_hit_rate": float(
                0.45 * float(five.get("top_k_hit_rate", 0.0)) + 0.55 * float(twenty.get("top_k_hit_rate", 0.0))
            ),
            "top_bottom_spread": float(
                0.45 * float(five.get("top_bottom_spread", 0.0)) + 0.55 * float(twenty.get("top_bottom_spread", 0.0))
            ),
        }

    curve = [float(item) for item in summary.get("nav_curve", [])] if isinstance(summary, dict) else []
    excess_curve = [float(item) for item in summary.get("excess_nav_curve", [])] if isinstance(summary, dict) else []
    curve_dates = [str(item) for item in summary.get("curve_dates", [])] if isinstance(summary, dict) else []

    def _window_from_curve(window: int) -> PredictionReviewWindow:
        if window in {5, 20} and f"{window}d" in calibration_priors:
            metrics = calibration_priors[f"{window}d"]
            return PredictionReviewWindow(
                window_days=int(window),
                label=f"近{window}日预测命中参考",
                hit_rate=float(metrics.get("top_k_hit_rate", 0.0)),
                avg_edge=float(metrics.get("top_bottom_spread", 0.0)),
                realized_return=0.0,
                sample_size=int(summary.get("n_days", 0)),
                note=f"研究期 {window} 日横截面命中率与头尾价差。",
            )
        if len(curve) < max(2, window + 1):
            return PredictionReviewWindow(window_days=int(window), label=f"近{window}日表现参考")
        recent_nav = curve[-(window + 1) :]
        base_nav = recent_nav[0]
        realized_return = 0.0 if abs(base_nav) <= 1e-12 else float(recent_nav[-1] / base_nav - 1.0)
        daily_rets = [
            float(nxt / prev - 1.0)
            for prev, nxt in zip(recent_nav[:-1], recent_nav[1:])
            if abs(prev) > 1e-12
        ]
        hit_rate = float(sum(1 for item in daily_rets if item > 0.0) / max(1, len(daily_rets)))
        avg_edge = 0.0
        if len(excess_curve) >= len(curve):
            recent_excess = excess_curve[-(window + 1) :]
            base_excess = recent_excess[0]
            if abs(base_excess) > 1e-12:
                avg_edge = float(recent_excess[-1] / base_excess - 1.0)
        return PredictionReviewWindow(
            window_days=int(window),
            label=f"近{window}日策略表现",
            hit_rate=hit_rate,
            avg_edge=avg_edge,
            realized_return=realized_return,
            sample_size=int(len(daily_rets)),
            note=f"截至 {(curve_dates[-1] if curve_dates else '最近一期')} 的滚动净值表现。",
        )

    review = PredictionReviewState(
        windows={
            "5d": _window_from_curve(5),
            "20d": _window_from_curve(20),
            "60d": _window_from_curve(60),
        },
        notes=[f"复盘参考来自研究 run_id={str(manifest.get('run_id', '')).strip() or 'NA'}"],
    )
    return review, calibration_priors


def _stock_reason_bundle(
    *,
    stock: StockForecastState,
    info_state: InfoAggregateState,
    state: CompositeState,
    rank: int,
    policy: PolicyDecision,
) -> tuple[list[str], list[str], list[str], str, str, str, str]:
    forecasts = dict(getattr(stock, "horizon_forecasts", {}))
    one_day = forecasts.get("1d")
    five_day = forecasts.get("5d")
    twenty_day = forecasts.get("20d")
    selection_reasons: list[str] = []
    if twenty_day is not None and float(twenty_day.up_prob) >= 0.60:
        selection_reasons.append(f"20日上涨概率 {twenty_day.up_prob * 100:.1f}%，中段趋势占优")
    if five_day is not None and float(five_day.up_prob) >= 0.58:
        selection_reasons.append(f"5日上涨概率 {five_day.up_prob * 100:.1f}%，短波段延续性较好")
    if float(stock.excess_vs_sector_prob) >= 0.55:
        selection_reasons.append(f"行业内相对强度 {stock.excess_vs_sector_prob * 100:.1f}%，位于板块前排")
    if float(stock.tradeability_score) >= 0.80:
        selection_reasons.append(f"量价结构稳定，交易一致性 {stock.tradeability_score * 100:.1f}%")
    if float(info_state.catalyst_strength) >= 0.55:
        selection_reasons.append(f"催化强度 {info_state.catalyst_strength * 100:.1f}%，对信号有加成")
    if float(state.cross_section.breadth_strength) >= 0.08:
        selection_reasons.append("市场宽度配合度尚可，前排信号更容易兑现")
    if not selection_reasons:
        selection_reasons.append("综合排序靠前，趋势、相对强弱和交易结构较均衡")

    alpha_parts = _alpha_score_components(stock)
    ranking_reasons: list[str] = []
    if float(alpha_parts.get("selection_bonus", 0.0)) > 0.08:
        ranking_reasons.append("趋势延续与稳定性加分较高")
    if float(alpha_parts.get("sector_edge", 0.0)) > 0.03:
        ranking_reasons.append("行业内相对强度为排名提供支持")
    if float(alpha_parts.get("medium_edge", 0.0)) > 0.03:
        ranking_reasons.append("中期空间优于大部分候选")
    if float(alpha_parts.get("stability_bonus", 0.0)) > 0.30:
        ranking_reasons.append("多周期信号一致，分歧不大")
    if not ranking_reasons:
        ranking_reasons.append("综合排序稳定，处于当前候选前列")

    risk_flags: list[str] = []
    if float(stock.up_1d_prob) > float(stock.up_5d_prob) + 0.06:
        risk_flags.append("短线偏热，次日容易先冲后分化")
    if float(stock.tradeability_score) < 0.72:
        risk_flags.append("量价承接一般，追价性价比不高")
    if float(info_state.negative_event_risk) >= 0.35:
        risk_flags.append("信息面负事件风险偏高")
    if one_day is not None and float(one_day.confidence) < 0.48:
        risk_flags.append("次日预测置信度偏低，宜轻仓观察")
    if str(stock.tradability_status) != "normal":
        risk_flags.append(f"交易状态受限: {stock.tradability_status}")
    if not risk_flags:
        risk_flags.append("当前未见明显硬性风险，但仍需服从仓位纪律")

    invalidation = "若下一交易日"
    if one_day is not None and one_day.price_low == one_day.price_low:
        invalidation += f"收盘跌破 {one_day.price_low:.2f}"
    else:
        invalidation += "低于预期下沿"
    invalidation += "，且 5 日上涨概率回落到 50% 下方，则本次信号失效。"

    target_weight = float(policy.symbol_target_weights.get(stock.symbol, 0.0))
    desired_weight = float(policy.desired_symbol_target_weights.get(stock.symbol, target_weight))
    action_reason = ""
    weight_reason = ""
    blocked_reason = ""
    if target_weight > 0.0:
        action_reason = f"排序第 {rank}，进入当前 {policy.target_position_count} 个目标持仓。"
        seat_weight = float(policy.target_exposure / max(1, policy.target_position_count))
        weight_reason = (
            f"目标权重 {target_weight * 100:.2f}%，在总仓位 {policy.target_exposure * 100:.1f}% 下属于"
            f"{'主仓' if target_weight >= max(0.15, seat_weight) else '辅助仓'}配置。"
        )
        if desired_weight > target_weight + 1e-6:
            weight_reason += " 受风险约束后权重有所收缩。"
    else:
        if rank > int(policy.target_position_count):
            blocked_reason = f"当前只开 {policy.target_position_count} 个仓位，这只排位靠前但未进入前 {policy.target_position_count}。"
        elif float(policy.sector_budgets.get(stock.sector, 0.0)) <= 0.0:
            blocked_reason = f"{stock.sector} 当前未分配预算，本次只保留跟踪。"
        else:
            blocked_reason = "综合排序仍不错，但没有超过本轮实际执行门槛。"
    return (
        selection_reasons[:3],
        ranking_reasons[:3],
        risk_flags[:3],
        invalidation,
        action_reason,
        weight_reason,
        blocked_reason,
    )


def _decorate_composite_state_for_reporting(
    *,
    state: CompositeState,
    policy: PolicyDecision,
    calibration_priors: dict[str, dict[str, float]],
    reporting_market: MarketForecastState | None = None,
    reporting_cross_section: CrossSectionForecastState | None = None,
) -> CompositeState:
    base_market = reporting_market or state.market
    base_cross_section = reporting_cross_section or state.cross_section
    updated_market = replace(
        base_market,
        horizon_forecasts=_build_horizon_forecasts(
            latest_close=float(getattr(base_market, "latest_close", np.nan)),
            horizon_probs={
                "1d": float(base_market.up_1d_prob),
                "2d": float(base_market.up_2d_prob),
                "3d": float(base_market.up_3d_prob),
                "5d": float(base_market.up_5d_prob),
                "10d": float(getattr(base_market, "up_10d_prob", 0.45 * base_market.up_5d_prob + 0.55 * base_market.up_20d_prob)),
                "20d": float(base_market.up_20d_prob),
            },
            short_profile=_profile_from_horizon_map(dict(getattr(base_market, "horizon_forecasts", {})), "1d"),
            mid_profile=_profile_from_horizon_map(dict(getattr(base_market, "horizon_forecasts", {})), "20d"),
            calibration_priors=calibration_priors,
        ),
        sentiment=_build_market_sentiment_state(
            market=base_market,
            cross_section=base_cross_section,
            capital_flow=state.capital_flow_state,
            macro=state.macro_context_state,
        ),
    )
    reporting_state = replace(state, market=updated_market, cross_section=base_cross_section)
    ordered_candidates = _candidate_stocks_from_state_external(reporting_state)
    rank_map = {stock.symbol: idx for idx, stock in enumerate(ordered_candidates, start=1)}
    updated_stocks: list[StockForecastState] = []
    for stock in reporting_state.stocks:
        info_state = reporting_state.stock_info_states.get(stock.symbol, InfoAggregateState())
        refreshed_forecasts = _build_horizon_forecasts(
            latest_close=float(getattr(stock, "latest_close", np.nan)),
            horizon_probs={
                "1d": float(stock.up_1d_prob),
                "2d": float(stock.up_2d_prob),
                "3d": float(stock.up_3d_prob),
                "5d": float(stock.up_5d_prob),
                "10d": float(getattr(stock, "up_10d_prob", 0.45 * stock.up_5d_prob + 0.55 * stock.up_20d_prob)),
                "20d": float(stock.up_20d_prob),
            },
            short_profile=_profile_from_horizon_map(dict(getattr(stock, "horizon_forecasts", {})), "1d"),
            mid_profile=_profile_from_horizon_map(dict(getattr(stock, "horizon_forecasts", {})), "20d"),
            info_state=info_state,
            calibration_priors=calibration_priors,
            tradability_status=str(getattr(stock, "tradability_status", "normal")),
        )
        reasons, ranking, risks, invalidation, action_reason, weight_reason, blocked_reason = _stock_reason_bundle(
            stock=stock,
            info_state=info_state,
            state=reporting_state,
            rank=int(rank_map.get(stock.symbol, len(rank_map) + 1)),
            policy=policy,
        )
        updated_stocks.append(
            replace(
                stock,
                horizon_forecasts=refreshed_forecasts,
                selection_reasons=reasons,
                ranking_reasons=ranking,
                risk_flags=risks,
                invalidation_rule=invalidation,
                action_reason=action_reason,
                weight_reason=weight_reason,
                blocked_reason=blocked_reason,
            )
        )
    updated_stocks.sort(key=lambda item: rank_map.get(item.symbol, len(rank_map) + 999))
    return replace(reporting_state, market=updated_market, cross_section=base_cross_section, stocks=updated_stocks)


def _filter_state_for_recommendation_scope(
    *,
    state: CompositeState,
    main_board_only: bool,
) -> CompositeState:
    if not main_board_only:
        return state
    filtered_stocks = [stock for stock in state.stocks if _is_main_board_symbol(stock.symbol)]
    if not filtered_stocks:
        return state
    filtered_selection = _build_candidate_selection_state_external(
        market=state.market,
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=filtered_stocks,
        mainlines=state.mainlines,
        strategy_mode=state.strategy_mode,
        risk_regime=state.risk_regime,
        stock_score_fn=_stock_policy_score,
    )
    selection_notes = list(filtered_selection.selection_notes or [])
    selection_notes.append("Recommendation scope limited to main-board listings only.")
    filtered_selection = replace(filtered_selection, selection_notes=selection_notes)
    filtered_symbols = {stock.symbol for stock in filtered_stocks}
    filtered_info_states = {
        symbol: payload
        for symbol, payload in state.stock_info_states.items()
        if symbol in filtered_symbols
    }
    return replace(
        state,
        stocks=filtered_stocks,
        candidate_selection=filtered_selection,
        stock_info_states=filtered_info_states,
    )


def _build_live_market_reporting_overlay(
    *,
    settings: dict[str, object],
    universe_ctx: _DailyUniverseContext,
    state: CompositeState,
) -> tuple[MarketForecastState | None, CrossSectionForecastState | None]:
    try:
        market_state, cross_section = _build_market_and_cross_section_states(
            market_symbol=str(getattr(universe_ctx.market_security, "symbol", "")),
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
            use_margin_features=bool(settings.get("use_margin_features", False)),
            margin_market_file=str(settings.get("margin_market_file", "")),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
            use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
            use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
            cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
            market_short_prob=float(state.market.up_1d_prob),
            market_two_prob=float(getattr(state.market, "up_2d_prob", 0.5)),
            market_three_prob=float(getattr(state.market, "up_3d_prob", 0.5)),
            market_five_prob=float(state.market.up_5d_prob),
            market_mid_prob=float(state.market.up_20d_prob),
            market_short_profile=_profile_from_horizon_map(dict(getattr(state.market, "horizon_forecasts", {})), "1d"),
            market_mid_profile=_profile_from_horizon_map(dict(getattr(state.market, "horizon_forecasts", {})), "20d"),
        )
        return market_state, cross_section
    except Exception:
        return None, None


def _ranked_sector_budgets(sectors: Iterable[SectorForecastState], *, target_exposure: float) -> dict[str, float]:
    rows = list(sectors)
    if not rows:
        return {}
    raw = [max(0.0, float(item.up_20d_prob) - 0.50) + max(0.0, float(item.relative_strength)) for item in rows]
    total = sum(raw)
    if total <= 1e-9:
        equal = float(target_exposure) / float(len(rows))
        return {item.sector: equal for item in rows}
    return {item.sector: float(target_exposure) * score / total for item, score in zip(rows, raw)}


def _alpha_opportunity_metrics(stocks: Iterable[StockForecastState]) -> dict[str, float]:
    actionable = [
        stock for stock in stocks
        if _is_actionable_status(getattr(stock, "tradability_status", "normal"))
    ]
    if not actionable:
        return {
            "top_score": 0.0,
            "avg_top3": 0.0,
            "median_score": 0.0,
            "breadth_ratio": 0.0,
            "strong_count": 0.0,
            "alpha_headroom": 0.0,
        }
    scores = sorted((_stock_policy_score(stock) for stock in actionable), reverse=True)
    top_slice = scores[: min(3, len(scores))]
    top_score = float(scores[0])
    avg_top3 = float(sum(top_slice) / max(1, len(top_slice)))
    median_score = float(np.median(scores))
    strong_cut = max(0.56, median_score + 0.03)
    strong_count = int(sum(1 for score in scores if score >= strong_cut))
    breadth_ratio = float(strong_count / max(1, len(scores)))
    alpha_headroom = float(max(0.0, avg_top3 - max(0.54, median_score)))
    return {
        "top_score": top_score,
        "avg_top3": avg_top3,
        "median_score": median_score,
        "breadth_ratio": breadth_ratio,
        "strong_count": float(strong_count),
        "alpha_headroom": alpha_headroom,
    }


def _mainline_preference_maps(
    mainlines: Iterable[MainlineState],
    *,
    risk_cutoff: float,
) -> tuple[dict[str, float], dict[str, float], list[MainlineState]]:
    confirmed: list[MainlineState] = []
    sector_boosts: dict[str, float] = {}
    symbol_boosts: dict[str, float] = {}
    rows = list(mainlines or [])[:3]
    top_conviction = max((float(getattr(item, "conviction", 0.0)) for item in rows), default=0.0)
    cutoff = max(0.30, top_conviction - 0.08)
    for rank, mainline in enumerate(rows):
        conviction = float(getattr(mainline, "conviction", 0.0))
        event_risk = float(getattr(mainline, "event_risk_level", 0.0))
        if conviction < cutoff or event_risk >= float(risk_cutoff):
            continue
        leadership = float(getattr(mainline, "leadership", 0.0))
        catalyst = float(getattr(mainline, "catalyst_strength", 0.0))
        boost = float(
            _clip(
                0.05
                + 0.28 * max(0.0, conviction - cutoff)
                + 0.06 * leadership
                + 0.05 * catalyst
                - 0.02 * rank,
                0.03,
                0.16,
            )
        )
        confirmed.append(mainline)
        for sector in getattr(mainline, "sectors", []):
            sector_key = str(sector)
            sector_boosts[sector_key] = max(sector_boosts.get(sector_key, 0.0), boost)
        for symbol in getattr(mainline, "representative_symbols", []):
            symbol_key = str(symbol)
            symbol_boosts[symbol_key] = max(symbol_boosts.get(symbol_key, 0.0), boost + 0.03)
    return sector_boosts, symbol_boosts, confirmed


def _ranked_sector_budgets_with_alpha(
    *,
    sectors: Iterable[SectorForecastState],
    stocks: Iterable[StockForecastState],
    target_exposure: float,
    sector_score_adjustments: dict[str, float] | None = None,
) -> dict[str, float]:
    return _ranked_sector_budgets_with_alpha_external(
        sectors=sectors,
        stocks=stocks,
        target_exposure=target_exposure,
        stock_score_fn=_stock_policy_score,
        sector_score_adjustments=sector_score_adjustments,
    )


def _cap_sector_budgets(
    *,
    sector_budgets: dict[str, float],
    target_exposure: float,
    risk_regime: str,
    breadth_strength: float,
) -> tuple[dict[str, float], list[str]]:
    return _cap_sector_budgets_external(
        sector_budgets=sector_budgets,
        target_exposure=target_exposure,
        risk_regime=risk_regime,
        breadth_strength=breadth_strength,
    )


def _stock_policy_score(stock: StockForecastState) -> float:
    return float(_alpha_score_components(stock)["alpha_score"])


def _policy_objective_score(summary: V2BacktestSummary) -> float:
    excess_alpha = float(summary.excess_annual_return)
    ir_term = float(summary.information_ratio)
    drawdown_penalty = 0.80 * abs(float(summary.max_drawdown))
    turnover_penalty = 0.40 * float(summary.avg_turnover)
    cost_penalty = 0.60 * float(summary.total_cost)
    score = excess_alpha + ir_term - drawdown_penalty - turnover_penalty - cost_penalty
    return float(score)


def _allocate_sector_slots(
    *,
    sector_budgets: dict[str, float],
    available_by_sector: dict[str, list[tuple[StockForecastState, float]]],
    total_slots: int,
    sector_strengths: dict[str, float] | None = None,
) -> dict[str, int]:
    return _allocate_sector_slots_external(
        sector_budgets=sector_budgets,
        available_by_sector=available_by_sector,
        total_slots=total_slots,
        sector_strengths=sector_strengths,
    )


def _allocate_with_sector_budgets(
    *,
    stocks: list[StockForecastState],
    sector_budgets: dict[str, float],
    target_position_count: int,
    sector_strengths: dict[str, float] | None = None,
    max_single_position: float = 0.35,
    symbol_score_adjustments: dict[str, float] | None = None,
) -> dict[str, float]:
    return _allocate_with_sector_budgets_external(
        stocks=stocks,
        sector_budgets=sector_budgets,
        target_position_count=target_position_count,
        stock_score_fn=_stock_policy_score,
        sector_strengths=sector_strengths,
        max_single_position=max_single_position,
        symbol_score_adjustments=symbol_score_adjustments,
    )


def _finalize_target_weights(
    *,
    desired_weights: dict[str, float],
    current_weights: dict[str, float],
    current_holding_days: dict[str, int],
    stocks: list[StockForecastState],
    target_exposure: float,
    min_trade_delta: float,
    min_holding_days: int,
) -> tuple[dict[str, float], list[str]]:
    adjusted = {symbol: max(0.0, float(weight)) for symbol, weight in desired_weights.items()}
    state_map = {item.symbol: item for item in stocks}
    notes: list[str] = []
    locked_symbols: set[str] = set()

    all_symbols = sorted(set(adjusted) | set(current_weights))
    for symbol in all_symbols:
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        state = state_map.get(symbol)
        status = "data_insufficient" if state is None else str(getattr(state, "tradability_status", "normal") or "normal")
        target = max(0.0, float(adjusted.get(symbol, 0.0)))

        if state is None and current > 1e-9:
            adjusted[symbol] = current
            locked_symbols.add(symbol)
            notes.append(f"{symbol}: missing state, holding frozen.")
            continue
        if not _is_actionable_status(status):
            if current > 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
                notes.append(f"{symbol}: {status}, holding frozen.")
            else:
                adjusted.pop(symbol, None)
                notes.append(f"{symbol}: {status}, new entry blocked.")
            continue
        if status == "data_insufficient":
            if current <= 1e-9 and target > 1e-9:
                adjusted.pop(symbol, None)
                notes.append(f"{symbol}: data insufficient, new entry blocked.")
                continue
            if target > current + 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
                notes.append(f"{symbol}: data insufficient, add-on blocked.")
                continue
        holding_days = int(max(0, current_holding_days.get(symbol, 0)))
        if current > 1e-9 and holding_days < int(min_holding_days) and target < current - 1e-9:
            adjusted[symbol] = current
            locked_symbols.add(symbol)
            notes.append(
                f"{symbol}: minimum holding window active ({holding_days}/{int(min_holding_days)}d), sell blocked."
            )
            continue

    for symbol in sorted(set(adjusted) | set(current_weights)):
        current = max(0.0, float(current_weights.get(symbol, 0.0)))
        target = max(0.0, float(adjusted.get(symbol, 0.0)))
        if abs(target - current) < float(min_trade_delta):
            if abs(target - current) > 1e-9:
                notes.append(f"{symbol}: rebalance gap below threshold.")
            if current > 1e-9:
                adjusted[symbol] = current
                locked_symbols.add(symbol)
            else:
                adjusted.pop(symbol, None)

    locked_total = float(sum(max(0.0, float(adjusted.get(symbol, 0.0))) for symbol in locked_symbols))
    free_symbols = [
        symbol for symbol, weight in adjusted.items()
        if symbol not in locked_symbols and float(weight) > 1e-9
    ]
    free_total = float(sum(float(adjusted[symbol]) for symbol in free_symbols))
    free_budget = max(0.0, float(target_exposure) - locked_total)
    if free_total > free_budget + 1e-9 and free_total > 1e-9:
        scale = float(free_budget / free_total) if free_budget > 1e-9 else 0.0
        for symbol in free_symbols:
            adjusted[symbol] = float(adjusted[symbol]) * scale
        notes.append("Actionable targets scaled down to respect target exposure after frozen holdings.")

    adjusted = {
        symbol: float(weight)
        for symbol, weight in adjusted.items()
        if float(weight) > 1e-6
    }
    return adjusted, notes


def _apply_external_signal_weight_tilts(
    *,
    weights: dict[str, float],
    state: CompositeState,
    target_exposure: float,
    risk_cutoff: float,
    catalyst_boost_cap: float,
) -> tuple[dict[str, float], list[str]]:
    adjusted = {str(symbol): max(0.0, float(weight)) for symbol, weight in weights.items() if float(weight) > 1e-9}
    if not adjusted:
        return adjusted, []
    notes: list[str] = []
    stock_map = {item.symbol: item for item in state.stocks}
    for symbol in list(adjusted):
        info_state = state.stock_info_states.get(symbol, InfoAggregateState())
        event_risk = float(info_state.event_risk_level)
        catalyst = float(info_state.catalyst_strength)
        alpha_advantage = 0.0
        stock = stock_map.get(symbol)
        if stock is not None:
            alpha_source = getattr(stock, "alpha_score", None)
            if alpha_source is None:
                try:
                    alpha_source = _stock_policy_score(stock)
                except Exception:
                    alpha_source = 0.55
            alpha_advantage = max(0.0, float(alpha_source) - 0.55)
        if event_risk >= float(risk_cutoff):
            adjusted[symbol] *= max(0.0, 1.0 - min(0.85, event_risk))
            notes.append(f"{symbol}: event risk above cutoff, target trimmed.")
        elif catalyst > 0.0 and alpha_advantage > 0.0:
            boost = min(float(catalyst_boost_cap), 0.35 * catalyst + 0.80 * alpha_advantage)
            adjusted[symbol] *= 1.0 + boost
            notes.append(f"{symbol}: catalyst aligned with alpha, target boosted.")
    total = float(sum(adjusted.values()))
    if total <= 1e-9:
        return {}, notes
    scale = float(target_exposure) / total if target_exposure > 1e-9 else 0.0
    return (
        {
            symbol: float(weight) * scale
            for symbol, weight in adjusted.items()
            if float(weight) * scale > 1e-6
        },
        notes,
    )


def _enforce_single_name_cap(
    *,
    weights: dict[str, float],
    max_single_position: float,
) -> dict[str, float]:
    adjusted = {
        str(symbol): max(0.0, float(weight))
        for symbol, weight in weights.items()
        if float(weight) > 1e-9
    }
    cap = max(0.0, float(max_single_position))
    if not adjusted or cap <= 1e-9:
        return adjusted
    for symbol in list(adjusted):
        adjusted[symbol] = min(adjusted[symbol], cap)
    return {
        symbol: float(weight)
        for symbol, weight in adjusted.items()
        if float(weight) > 1e-6
    }


def _sector_budgets_from_weights(
    *,
    symbol_weights: dict[str, float],
    stocks: list[StockForecastState],
) -> dict[str, float]:
    state_map = {item.symbol: item for item in stocks}
    out: dict[str, float] = {}
    for symbol, weight in symbol_weights.items():
        if float(weight) <= 1e-9:
            continue
        sector = state_map.get(symbol).sector if state_map.get(symbol) is not None else "其他"
        out[sector] = out.get(sector, 0.0) + float(weight)
    return out


def _advance_holding_days(
    *,
    prev_holding_days: dict[str, int],
    prev_weights: dict[str, float],
    next_weights: dict[str, float],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for symbol, weight in next_weights.items():
        if float(weight) <= 1e-9:
            continue
        prev_weight = max(0.0, float(prev_weights.get(symbol, 0.0)))
        if prev_weight > 1e-9:
            out[symbol] = int(max(1, int(prev_holding_days.get(symbol, 0)) + 1))
        else:
            out[symbol] = 1
    return out


def apply_policy(
    policy_input: PolicyInput,
    *,
    policy_spec: PolicySpec | None = None,
) -> PolicyDecision:
    policy_spec = policy_spec or PolicySpec()
    state = policy_input.composite_state
    market = state.market
    cross = state.cross_section
    min_holding_days = 5

    target_exposure = 0.0
    target_position_count = 1
    turnover_cap = float(policy_spec.risk_off_turnover_cap)
    intraday_t_allowed = False
    risk_notes: list[str] = []
    candidate_stocks = _candidate_stocks_from_state_external(state)
    candidate_selection = getattr(state, "candidate_selection", None)
    alpha_metrics = _alpha_opportunity_metrics(candidate_stocks)
    candidate_risk = _candidate_risk_snapshot_external(candidate_stocks)
    mainlines = list(getattr(state, "mainlines", []))
    dominant_mainline_sectors = _dominant_mainline_sectors_external(mainlines)
    mainline_sector_boosts, mainline_symbol_boosts, confirmed_mainlines = _mainline_preference_maps(
        mainlines,
        risk_cutoff=float(policy_spec.event_risk_cutoff),
    )
    alpha_headroom = float(alpha_metrics["alpha_headroom"])
    alpha_breadth = float(alpha_metrics["breadth_ratio"])
    top_alpha = float(alpha_metrics["top_score"])
    market_info = getattr(state, "market_info_state", InfoAggregateState())
    capital_flow = getattr(state, "capital_flow_state", CapitalFlowState())
    macro_context = getattr(state, "macro_context_state", MacroContextState())
    near_term_stack = float(
        0.20 * float(market.up_1d_prob)
        + 0.22 * float(getattr(market, "up_2d_prob", 0.5))
        + 0.24 * float(getattr(market, "up_3d_prob", 0.5))
        + 0.34 * float(market.up_5d_prob)
    )

    if state.risk_regime == "risk_on":
        regime_floor = 0.45
        target_position_count = int(policy_spec.risk_on_positions)
        turnover_cap = float(policy_spec.risk_on_turnover_cap)
        intraday_t_allowed = state.strategy_mode == "range_rotation"
    elif state.risk_regime == "cautious":
        regime_floor = 0.35
        target_position_count = int(policy_spec.cautious_positions)
        turnover_cap = float(policy_spec.cautious_turnover_cap)
    else:
        regime_floor = 0.25
        target_position_count = int(policy_spec.risk_off_positions)
        turnover_cap = float(policy_spec.risk_off_turnover_cap)
        risk_notes.append("Risk-off regime: hard floor reduced, but not forced into deep cash.")

    alpha_base_exposure = float(
        _clip(
            0.25
            + 1.60 * alpha_headroom
            + 0.55 * alpha_breadth
            + 0.35 * max(0.0, top_alpha - 0.55),
            regime_floor,
            0.95,
        )
    )
    target_exposure = float(alpha_base_exposure)

    if near_term_stack < 0.50:
        target_exposure *= 0.95
        risk_notes.append("Near-term market stack below 0.50: mild exposure trim.")
    if float(market_info.event_risk_level) >= float(policy_spec.event_risk_cutoff):
        target_exposure *= 0.90
        target_position_count = max(1, target_position_count - 1)
        turnover_cap = min(turnover_cap, 0.20)
        risk_notes.append("Event risk elevated: exposure trimmed and concentration reduced.")
    if macro_context.macro_risk_level == "high":
        target_exposure *= 0.88
        turnover_cap = min(turnover_cap, 0.18)
        risk_notes.append("Macro risk high: exposure trimmed and turnover capped.")
    elif macro_context.macro_risk_level == "elevated":
        target_exposure *= 0.94
        risk_notes.append("Macro risk elevated: mild exposure trim.")
    if capital_flow.flow_regime in {"outflow", "strong_outflow"}:
        flow_penalty = float(policy_spec.flow_exposure_cap) * (1.0 if capital_flow.flow_regime == "strong_outflow" else 0.65)
        target_exposure = max(regime_floor, target_exposure - flow_penalty)
        turnover_cap = min(turnover_cap, 0.22 if capital_flow.flow_regime == "outflow" else 0.18)
        risk_notes.append(f"Capital flow {capital_flow.flow_regime}: exposure trimmed.")
    elif capital_flow.flow_regime in {"inflow", "strong_inflow"} and state.risk_regime != "risk_off":
        flow_boost = float(policy_spec.flow_exposure_cap) * (0.60 if capital_flow.flow_regime == "inflow" else 1.0)
        target_exposure = min(1.0, target_exposure + flow_boost)
        turnover_cap = min(0.45, turnover_cap + 0.02)
        risk_notes.append(f"Capital flow {capital_flow.flow_regime}: measured exposure boost.")
    if market.drawdown_risk >= 0.50:
        target_exposure *= 0.90
        turnover_cap = min(turnover_cap, 0.22)
        risk_notes.append("Drawdown risk elevated: mild exposure trim.")
    elif market.drawdown_risk >= 0.35:
        target_exposure *= 0.94
        turnover_cap = min(turnover_cap, 0.20)
        risk_notes.append("Intermediate drawdown risk: extra exposure trim.")
    if cross.fund_flow_strength < 0.0:
        target_exposure *= 0.94
        risk_notes.append("Fund flow weak: mild exposure trim.")
    if market.volatility_regime == "high":
        target_exposure *= 0.90
        turnover_cap = min(turnover_cap, 0.24)
        risk_notes.append("High volatility regime: exposure trimmed, not capped aggressively.")
    elif market.volatility_regime == "low" and state.risk_regime == "risk_on" and cross.breadth_strength > 0.15 and near_term_stack >= 0.56:
        target_exposure = min(1.0, target_exposure * 1.05)
        risk_notes.append("Low volatility with strong near-term stack: exposure nudged up.")
    if cross.large_vs_small_bias < -0.05:
        target_position_count = min(target_position_count + 1, 5)
        risk_notes.append("Large-cap bias weak: diversify more positions.")
    if cross.growth_vs_value_bias < -0.08:
        turnover_cap = min(turnover_cap, 0.24)
        risk_notes.append("Growth style weak: turnover capped conservatively.")

    if alpha_headroom <= 0.01 or alpha_breadth < 0.05:
        target_exposure *= 0.90
        target_position_count = max(1, target_position_count - 1)
        turnover_cap = min(turnover_cap, 0.22)
        risk_notes.append("Cross-sectional alpha weak: exposure trimmed.")
    elif (
        alpha_headroom >= 0.02
        and alpha_breadth >= 0.08
        and cross.breadth_strength >= 0.10
        and market.liquidity_stress <= 0.60
    ):
        alpha_boost = min(0.12, 0.70 * alpha_headroom + 0.18 * alpha_breadth)
        target_exposure = min(1.0, target_exposure + alpha_boost)
        if top_alpha >= 0.62:
            target_position_count = min(5, target_position_count + 1)
        risk_notes.append("Cross-sectional alpha strong: exposure boosted.")

    if top_alpha >= 0.64 and cross.breadth_strength >= 0.12 and state.risk_regime != "risk_off":
        turnover_cap = min(0.45, turnover_cap + 0.03)
        risk_notes.append("Top alpha concentration supports measured rotation.")

    if mainlines:
        top_mainline = mainlines[0]
        if float(top_mainline.event_risk_level) >= float(policy_spec.event_risk_cutoff):
            target_exposure *= 0.94
            turnover_cap = min(turnover_cap, 0.20)
            risk_notes.append(f"Mainline {top_mainline.name} is risk-watched: exposure trimmed.")
        elif (
            float(top_mainline.conviction) >= 0.62
            and float(top_mainline.catalyst_strength) >= 0.24
            and state.risk_regime != "risk_off"
        ):
            target_exposure = min(1.0, target_exposure + 0.03)
            target_position_count = min(5, target_position_count + 1)
            risk_notes.append(f"Mainline {top_mainline.name} confirmed: measured exposure support.")
    if confirmed_mainlines:
        target_position_count = max(target_position_count, min(4, len(confirmed_mainlines) + 1))
        if state.risk_regime != "risk_off":
            turnover_cap = min(0.40, turnover_cap + 0.02)
        risk_notes.append(
            "Mainline budgets prioritized: " + ", ".join(str(item.name) for item in confirmed_mainlines[:3])
        )

    if float(candidate_risk["fragile_ratio"]) >= 0.35:
        target_exposure *= 0.92
        turnover_cap = min(turnover_cap, 0.20)
        risk_notes.append("Candidate set fragile: exposure and turnover trimmed.")
    elif float(candidate_risk["fragile_ratio"]) >= 0.20 and state.risk_regime != "risk_on":
        target_exposure *= 0.95
        risk_notes.append("Candidate set mildly fragile under cautious regime: exposure trimmed.")
    if float(candidate_risk["reversal_ratio"]) >= 0.25:
        turnover_cap = min(turnover_cap, 0.18)
        risk_notes.append("Short-term reversal risk elevated across candidates: turnover capped.")
    if (
        float(candidate_risk["durability_score"]) <= 0.54
        and (float(market.drawdown_risk) >= 0.30 or float(cross.weak_stock_ratio) >= 0.48)
    ):
        target_exposure *= 0.94
        risk_notes.append("Candidate durability soft in a fragile tape: extra exposure trim.")

    if candidate_selection is not None and len(candidate_stocks) < len(state.stocks):
        risk_notes.append(
            f"Candidate shortlist active: {len(candidate_stocks)}/{len(state.stocks)} names after macro-sector screening."
        )
    if candidate_stocks:
        target_position_count = min(target_position_count, len(candidate_stocks))

    target_exposure = _clip(target_exposure, regime_floor, 1.0)
    max_single_position = 0.35
    if market.volatility_regime == "high":
        max_single_position = min(max_single_position, 0.24)
    if cross.large_vs_small_bias < -0.05:
        max_single_position = min(max_single_position, 0.22)
    if cross.growth_vs_value_bias < -0.08:
        max_single_position = min(max_single_position, 0.20)
    if float(candidate_risk["fragile_ratio"]) >= 0.20:
        max_single_position = min(max_single_position, 0.20 if state.risk_regime == "risk_on" else 0.18)
        risk_notes.append("Candidate fragility keeps single-name sizing conservative.")
    if (
        candidate_selection is not None
        and int(getattr(candidate_selection, "total_scored", 0)) >= 120
        and int(getattr(candidate_selection, "shortlist_size", 0)) >= 10
    ):
        target_position_count = max(target_position_count, 3 if state.risk_regime != "risk_off" else 2)
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_on" else 0.16)
        risk_notes.append("Large-universe shortlist: concentration spread across more names.")
    if float(market.drawdown_risk) >= 0.35 or float(cross.weak_stock_ratio) >= 0.50:
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_on" else 0.16)
        risk_notes.append("Fragile tape: single-name cap tightened.")
    if alpha_breadth >= 0.12 and alpha_headroom >= 0.02:
        target_position_count = max(target_position_count, 2)
        max_single_position = min(max_single_position, 0.18 if state.risk_regime == "risk_off" else 0.22)
        risk_notes.append("Alpha breadth strong: concentration reduced across more names.")
    target_position_count = int(np.clip(target_position_count, 1, 5))
    candidate_sector_names = set(
        getattr(candidate_selection, "shortlisted_sectors", []) if candidate_selection is not None else []
    )
    if dominant_mainline_sectors:
        candidate_sector_names.update(dominant_mainline_sectors)
    policy_sectors = [
        sector for sector in state.sectors
        if not candidate_sector_names or str(sector.sector) in candidate_sector_names
    ]
    if not policy_sectors:
        policy_sectors = list(state.sectors)
    if mainline_sector_boosts:
        policy_sectors = sorted(
            policy_sectors,
            key=lambda sector: (
                float(mainline_sector_boosts.get(str(sector.sector), 0.0)),
                float(sector.up_20d_prob),
                float(sector.relative_strength),
            ),
            reverse=True,
        )
    if not candidate_stocks:
        candidate_stocks = list(state.stocks)
    desired_sector_budgets = _ranked_sector_budgets_with_alpha(
        sectors=policy_sectors[: max(1, target_position_count)],
        stocks=candidate_stocks,
        target_exposure=target_exposure,
        sector_score_adjustments=mainline_sector_boosts,
    )
    desired_sector_budgets, sector_cap_notes = _cap_sector_budgets(
        sector_budgets=desired_sector_budgets,
        target_exposure=target_exposure,
        risk_regime=state.risk_regime,
        breadth_strength=float(cross.breadth_strength),
    )
    risk_notes.extend(sector_cap_notes)
    desired_symbol_target_weights = _allocate_with_sector_budgets(
        stocks=candidate_stocks,
        sector_budgets=desired_sector_budgets,
        target_position_count=int(target_position_count),
        sector_strengths={
            sector: float(weight) / max(float(target_exposure), 1e-9)
            for sector, weight in desired_sector_budgets.items()
        },
        max_single_position=float(max_single_position),
        symbol_score_adjustments=mainline_symbol_boosts,
    )
    desired_symbol_target_weights, external_signal_notes = _apply_external_signal_weight_tilts(
        weights=desired_symbol_target_weights,
        state=state,
        target_exposure=float(target_exposure),
        risk_cutoff=float(policy_spec.event_risk_cutoff),
        catalyst_boost_cap=float(policy_spec.catalyst_boost_cap),
    )
    desired_symbol_target_weights = _enforce_single_name_cap(
        weights=desired_symbol_target_weights,
        max_single_position=float(max_single_position),
    )
    risk_notes.extend(external_signal_notes)
    symbol_target_weights, execution_notes = _finalize_target_weights(
        desired_weights=desired_symbol_target_weights,
        current_weights=policy_input.current_weights,
        current_holding_days=policy_input.current_holding_days,
        stocks=state.stocks,
        target_exposure=target_exposure,
        min_trade_delta=min(0.02, 0.25 * float(turnover_cap)),
        min_holding_days=min_holding_days,
    )
    symbol_target_weights = _enforce_single_name_cap(
        weights=symbol_target_weights,
        max_single_position=float(max_single_position),
    )
    risk_notes.extend(execution_notes)
    sector_budgets = _sector_budgets_from_weights(
        symbol_weights=symbol_target_weights,
        stocks=state.stocks,
    )

    current_total = sum(max(0.0, float(v)) for v in policy_input.current_weights.values())
    rebalance_gap = abs(float(target_exposure) - float(current_total))
    rebalance_now = rebalance_gap >= 0.05
    rebalance_intensity = _clip(rebalance_gap / max(0.05, turnover_cap), 0.0, 1.0)

    return PolicyDecision(
        target_exposure=float(target_exposure),
        target_position_count=int(target_position_count),
        rebalance_now=bool(rebalance_now),
        rebalance_intensity=float(rebalance_intensity),
        intraday_t_allowed=bool(intraday_t_allowed),
        turnover_cap=float(turnover_cap),
        sector_budgets=sector_budgets,
        desired_sector_budgets=desired_sector_budgets,
        symbol_target_weights=symbol_target_weights,
        desired_symbol_target_weights=desired_symbol_target_weights,
        execution_notes=execution_notes,
        risk_notes=risk_notes,
    )


def build_trade_actions(
    *,
    decision: PolicyDecision,
    current_weights: dict[str, float],
) -> list[TradeAction]:
    all_symbols = sorted(set(current_weights) | set(decision.symbol_target_weights))
    actions: list[TradeAction] = []
    for symbol in all_symbols:
        current_weight = max(0.0, float(current_weights.get(symbol, 0.0)))
        target_weight = max(0.0, float(decision.symbol_target_weights.get(symbol, 0.0)))
        delta_weight = float(target_weight - current_weight)
        if delta_weight > 0.02:
            action = "BUY"
        elif delta_weight < -0.02:
            action = "SELL"
        else:
            action = "HOLD"

        note = ""
        if action == "HOLD" and abs(delta_weight) > 1e-9:
            note = "below_rebalance_threshold"

        actions.append(
            TradeAction(
                symbol=symbol,
                name=symbol,
                action=action,
                current_weight=float(current_weight),
                target_weight=float(target_weight),
                delta_weight=float(delta_weight),
                note=note,
            )
        )
    actions.sort(key=lambda item: (abs(float(item.delta_weight)), float(item.target_weight)), reverse=True)
    return actions


def _policy_spec_from_model(
    *,
    state: CompositeState,
    model: LearnedPolicyModel,
) -> PolicySpec:
    features = _policy_feature_vector(state)
    exposure_coef = _normalize_coef_vector(model.exposure_coef, features.size)
    position_coef = _normalize_coef_vector(model.position_coef, features.size)
    turnover_coef = _normalize_coef_vector(model.turnover_coef, features.size)
    exposure = _clip(
        _predict_ridge(features, model.exposure_intercept, exposure_coef),
        0.20,
        0.95,
    )
    positions = int(
        round(
            _clip(
                _predict_ridge(features, model.position_intercept, position_coef),
                1.0,
                6.0,
            )
        )
    )
    turnover_cap = _clip(
        _predict_ridge(features, model.turnover_intercept, turnover_coef),
        0.10,
        0.45,
    )
    cautious_exposure = _clip(0.5 * (exposure + 0.35), 0.30, exposure)
    risk_off_exposure = _clip(0.5 * cautious_exposure, 0.20, 0.40)
    cautious_positions = min(positions, max(1, positions - 1))
    risk_off_positions = max(1, positions - 2)
    cautious_turnover = _clip(min(turnover_cap, 0.85 * turnover_cap), 0.10, turnover_cap)
    risk_off_turnover = _clip(min(cautious_turnover, 0.70 * turnover_cap), 0.08, cautious_turnover)
    return PolicySpec(
        risk_on_exposure=float(exposure),
        cautious_exposure=float(cautious_exposure),
        risk_off_exposure=float(risk_off_exposure),
        risk_on_positions=int(positions),
        cautious_positions=int(cautious_positions),
        risk_off_positions=int(risk_off_positions),
        risk_on_turnover_cap=float(turnover_cap),
        cautious_turnover_cap=float(cautious_turnover),
        risk_off_turnover_cap=float(risk_off_turnover),
    )


def _simulate_execution_day(
    *,
    date: pd.Timestamp,
    next_date: pd.Timestamp,
    decision: PolicyDecision,
    current_weights: dict[str, float],
    current_cash: float,
    stock_states: list[StockForecastState],
    stock_frames: dict[str, pd.DataFrame],
    total_commission_rate: float,
    base_slippage_rate: float,
) -> tuple[float, float, float, float, float, dict[str, float], float]:
    state_map = {item.symbol: item for item in stock_states}
    symbols = sorted(set(current_weights) | set(decision.symbol_target_weights))
    raw_deltas = {
        symbol: float(decision.symbol_target_weights.get(symbol, 0.0)) - float(current_weights.get(symbol, 0.0))
        for symbol in symbols
    }

    executed_deltas: dict[str, float] = {}
    fill_ratios: list[float] = []
    slippage_rates: list[float] = []
    slippage_amounts: list[float] = []
    total_turnover_budget = float(max(0.0, decision.turnover_cap))
    used_turnover = 0.0

    ordered_symbols = sorted(symbols, key=lambda sym: abs(raw_deltas.get(sym, 0.0)), reverse=True)
    for symbol in ordered_symbols:
        delta = float(raw_deltas.get(symbol, 0.0))
        if abs(delta) <= 1e-4:
            continue
        state = state_map.get(symbol)
        frame = stock_frames.get(symbol)
        day_row = None
        if frame is not None:
            day_row = frame[frame["date"] == date]
        status = "normal"
        if state is not None:
            status = str(getattr(state, "tradability_status", "normal") or "normal")
        elif frame is None or day_row is None or day_row.empty:
            status = "halted"
        if not _is_actionable_status(status):
            continue
        if status == "data_insufficient" and delta > 0.0:
            # Keep execution conservative even if an upstream caller accidentally tries to add risk.
            continue
        if day_row is not None and not day_row.empty:
            latest = day_row.iloc[0]
            close_px = _safe_float(latest.get("close"), np.nan)
            open_px = _safe_float(latest.get("open"), np.nan)
            low_px = _safe_float(latest.get("low"), np.nan)
            high_px = _safe_float(latest.get("high"), np.nan)
            ret_1 = _safe_float(latest.get("ret_1"), np.nan)
            if close_px == close_px and ret_1 == ret_1:
                prev_close = close_px / max(1e-9, 1.0 + ret_1)
                limit_up_px = prev_close * 1.098
                limit_down_px = prev_close * 0.902
                # Single rebalance per trading day is already T+1 compatible; this only blocks orders
                # when the instrument appears pinned at the daily price limit for the full session.
                if delta > 0.0 and low_px == low_px and low_px >= limit_up_px:
                    continue
                if delta < 0.0 and high_px == high_px and high_px <= limit_down_px:
                    continue
        tradeability = 0.45 if state is None else _clip(float(state.tradeability_score), 0.10, 1.0)
        tradeability = min(tradeability, _status_tradeability_limit(status))
        liquidity_cap = 0.03 + 0.12 * tradeability
        remaining_turnover = max(0.0, total_turnover_budget - used_turnover)
        max_abs_trade = min(abs(delta), liquidity_cap, remaining_turnover)
        if max_abs_trade <= 1e-6:
            continue
        fill_ratio = max_abs_trade / max(abs(delta), 1e-9)
        executed = float(np.sign(delta) * max_abs_trade)
        executed_deltas[symbol] = executed
        fill_ratios.append(float(fill_ratio))
        impact = max_abs_trade / max(liquidity_cap, 1e-6)
        open_gap_penalty = 0.0
        intraday_range_penalty = 0.0
        if day_row is not None and not day_row.empty:
            latest = day_row.iloc[0]
            close_px = _safe_float(latest.get("close"), np.nan)
            open_px = _safe_float(latest.get("open"), np.nan)
            low_px = _safe_float(latest.get("low"), np.nan)
            high_px = _safe_float(latest.get("high"), np.nan)
            ret_1 = _safe_float(latest.get("ret_1"), np.nan)
            if close_px == close_px and ret_1 == ret_1:
                prev_close = close_px / max(1e-9, 1.0 + ret_1)
                if open_px == open_px:
                    open_gap = float(open_px / max(prev_close, 1e-9) - 1.0)
                    open_gap_penalty = max(0.0, open_gap) if delta > 0.0 else max(0.0, -open_gap)
                if low_px == low_px and high_px == high_px:
                    intraday_range_penalty = max(0.0, float(high_px - low_px) / max(prev_close, 1e-9))
        slippage_rate = float(
            base_slippage_rate
            * (
                0.65
                + 0.7 * impact
                + 0.35 * (1.0 - tradeability)
                + 0.40 * open_gap_penalty
                + 0.10 * intraday_range_penalty
            )
        )
        slippage_rates.append(slippage_rate)
        slippage_amounts.append(float(abs(executed) * slippage_rate))
        used_turnover += abs(executed)

    executed_weights = {symbol: max(0.0, float(weight)) for symbol, weight in current_weights.items()}
    for symbol, delta in executed_deltas.items():
        executed_weights[symbol] = max(0.0, float(executed_weights.get(symbol, 0.0)) + float(delta))
    executed_weights = {
        symbol: float(weight)
        for symbol, weight in executed_weights.items()
        if float(weight) > 1e-6
    }

    invested_after_trade = float(sum(executed_weights.values()))
    if invested_after_trade > 1.0:
        scale = 1.0 / invested_after_trade
        executed_weights = {symbol: float(weight) * scale for symbol, weight in executed_weights.items()}
        invested_after_trade = 1.0
    cash_after_trade = max(0.0, 1.0 - invested_after_trade)

    gross_end_value = float(cash_after_trade)
    position_values: dict[str, float] = {}
    for symbol, weight in executed_weights.items():
        state = state_map.get(symbol)
        status = "normal" if state is None else str(getattr(state, "tradability_status", "normal") or "normal")
        frame = stock_frames.get(symbol)
        if frame is None:
            realized_ret = -0.30 if status == "delisted" else 0.0
        else:
            row = frame[frame["date"] == date]
            if row.empty:
                realized_ret = -0.30 if status == "delisted" else 0.0
            else:
                realized_ret = float(row.iloc[0]["fwd_ret_1"])
                if status == "delisted":
                    realized_ret = min(realized_ret, -0.20)
        value = float(weight) * (1.0 + realized_ret)
        position_values[symbol] = value
        gross_end_value += value

    commission_cost = float(used_turnover * total_commission_rate)
    slippage_cost = float(sum(slippage_amounts)) if slippage_amounts else 0.0
    total_cost = float(commission_cost + slippage_cost)
    net_end_value = max(1e-9, gross_end_value - total_cost)

    next_weights = {
        symbol: float(value) / net_end_value
        for symbol, value in position_values.items()
        if float(value) > 1e-9
    }
    next_cash = max(0.0, float(cash_after_trade - total_cost) / net_end_value)
    daily_return = float(net_end_value - 1.0)
    avg_fill_ratio = float(np.mean(fill_ratios)) if fill_ratios else 0.0
    avg_slippage_bps = float(np.mean(slippage_rates) * 10000.0) if slippage_rates else 0.0
    return (
        daily_return,
        float(used_turnover),
        float(total_cost),
        avg_fill_ratio,
        avg_slippage_bps,
        next_weights,
        next_cash,
    )


def _to_v2_backtest_summary(
    *,
    returns: list[float],
    benchmark_returns: list[float] | None = None,
    turnovers: list[float],
    costs: list[float],
    gross_returns: list[float],
    fill_ratios: list[float],
    slippage_bps: list[float],
    rank_ics: list[float] | None = None,
    top_decile_returns: list[float] | None = None,
    top_bottom_spreads: list[float] | None = None,
    top_k_hit_rates: list[float] | None = None,
    horizon_metrics: dict[str, dict[str, list[float]]] | None = None,
    dates: list[pd.Timestamp],
) -> V2BacktestSummary:
    if not returns or not dates:
        return V2BacktestSummary(
            start_date="",
            end_date="",
            n_days=0,
            total_return=0.0,
            annual_return=0.0,
            max_drawdown=0.0,
            avg_turnover=0.0,
            total_cost=0.0,
            avg_rank_ic=0.0,
            avg_top_decile_return=0.0,
            avg_top_bottom_spread=0.0,
            avg_top_k_hit_rate=0.0,
            horizon_metrics={},
        )
    ret_arr = np.asarray(returns, dtype=float)
    nav = np.cumprod(1.0 + ret_arr)
    bench_arr = np.asarray(benchmark_returns if benchmark_returns is not None else np.zeros_like(ret_arr), dtype=float)
    if bench_arr.shape != ret_arr.shape:
        bench_arr = np.resize(bench_arr, ret_arr.shape)
    benchmark_nav = np.cumprod(1.0 + bench_arr)
    excess_ret_arr = (1.0 + ret_arr) / np.maximum(1.0 + bench_arr, 1e-9) - 1.0
    excess_nav = np.cumprod(1.0 + excess_ret_arr)
    gross_nav = np.cumprod(1.0 + np.asarray(gross_returns, dtype=float)) if gross_returns else nav
    peak = np.maximum.accumulate(nav)
    drawdown = nav / np.maximum(peak, 1e-12) - 1.0
    excess_peak = np.maximum.accumulate(excess_nav)
    total_return = float(nav[-1] - 1.0)
    benchmark_total_return = float(benchmark_nav[-1] - 1.0)
    excess_total_return = float(excess_nav[-1] - 1.0)
    gross_total_return = float(gross_nav[-1] - 1.0)
    n_days = len(returns)
    annual_return = float((1.0 + total_return) ** (252.0 / max(1, n_days)) - 1.0)
    benchmark_annual_return = float((1.0 + benchmark_total_return) ** (252.0 / max(1, n_days)) - 1.0)
    excess_annual_return = float((1.0 + excess_total_return) ** (252.0 / max(1, n_days)) - 1.0)
    annual_vol = float(np.std(ret_arr, ddof=0) * np.sqrt(252.0))
    win_rate = float(np.mean(ret_arr > 0.0))
    excess_drawdown = excess_nav / np.maximum(excess_peak, 1e-12) - 1.0
    excess_vol = float(np.std(excess_ret_arr, ddof=0))
    information_ratio = 0.0 if excess_vol <= 1e-12 else float(np.mean(excess_ret_arr) / excess_vol * np.sqrt(252.0))
    horizon_summary: dict[str, dict[str, float]] = {}
    if horizon_metrics:
        for horizon, metric_map in horizon_metrics.items():
            horizon_summary[horizon] = {
                key: float(np.mean(values)) if values else 0.0
                for key, values in metric_map.items()
            }
    return V2BacktestSummary(
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        n_days=int(n_days),
        total_return=float(total_return),
        annual_return=float(annual_return),
        max_drawdown=float(np.min(drawdown)),
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        total_cost=float(np.sum(costs)) if costs else 0.0,
        gross_total_return=float(gross_total_return),
        annual_vol=float(annual_vol),
        win_rate=float(win_rate),
        trade_days=int(sum(1 for item in turnovers if float(item) > 1e-9)),
        avg_fill_ratio=float(np.mean(fill_ratios)) if fill_ratios else 0.0,
        avg_slippage_bps=float(np.mean(slippage_bps)) if slippage_bps else 0.0,
        avg_rank_ic=float(np.mean(rank_ics)) if rank_ics else 0.0,
        avg_top_decile_return=float(np.mean(top_decile_returns)) if top_decile_returns else 0.0,
        avg_top_bottom_spread=float(np.mean(top_bottom_spreads)) if top_bottom_spreads else 0.0,
        avg_top_k_hit_rate=float(np.mean(top_k_hit_rates)) if top_k_hit_rates else 0.0,
        horizon_metrics=horizon_summary,
        benchmark_total_return=float(benchmark_total_return),
        benchmark_annual_return=float(benchmark_annual_return),
        excess_total_return=float(excess_total_return),
        excess_annual_return=float(excess_annual_return),
        excess_max_drawdown=float(np.min(excess_drawdown)),
        information_ratio=float(information_ratio),
        nav_curve=[float(x) for x in nav.tolist()],
        benchmark_nav_curve=[float(x) for x in benchmark_nav.tolist()],
        excess_nav_curve=[float(x) for x in excess_nav.tolist()],
        curve_dates=[str(item.date()) for item in dates],
    )


def _build_market_and_cross_section_from_prebuilt_frame(
    *,
    market_frame: pd.DataFrame,
    market_short_prob: float,
    market_two_prob: float | None,
    market_three_prob: float | None,
    market_five_prob: float | None,
    market_mid_prob: float,
    market_short_profile: _ReturnQuantileProfile | None = None,
    market_mid_profile: _ReturnQuantileProfile | None = None,
) -> tuple[MarketForecastState, CrossSectionForecastState]:
    latest = market_frame.sort_values("date").iloc[-1]
    state = decide_market_state(float(market_short_prob), float(market_mid_prob))
    cross_section_record = forecast_cross_section_state(market_frame)

    mkt_vol_20 = float(latest.get("mkt_volatility_20", 0.0))
    if mkt_vol_20 != mkt_vol_20:
        mkt_vol_20 = 0.0
    mkt_vol_60 = float(latest.get("mkt_volatility_60", mkt_vol_20))
    if mkt_vol_60 != mkt_vol_60:
        mkt_vol_60 = mkt_vol_20
    mkt_vol_60 = max(1e-6, mkt_vol_60)
    if mkt_vol_20 >= mkt_vol_60 * 1.15:
        volatility_regime = "high"
    elif mkt_vol_20 <= mkt_vol_60 * 0.85:
        volatility_regime = "low"
    else:
        volatility_regime = "normal"

    drawdown_raw = float(latest.get("mkt_drawdown_20", 0.0))
    if drawdown_raw != drawdown_raw:
        drawdown_raw = 0.0
    latest_close = float(_safe_float(latest.get("close", np.nan), np.nan))
    horizon_probs = {
        "1d": float(market_short_prob),
        "2d": float(
            market_two_prob
            if market_two_prob is not None
            else (0.65 * market_short_prob + 0.35 * (market_five_prob if market_five_prob is not None else market_mid_prob))
        ),
        "3d": float(
            market_three_prob
            if market_three_prob is not None
            else (0.35 * market_short_prob + 0.65 * (market_five_prob if market_five_prob is not None else market_mid_prob))
        ),
        "5d": float(
            market_five_prob if market_five_prob is not None else (0.6 * market_short_prob + 0.4 * market_mid_prob)
        ),
        "10d": float(
            0.45 * (market_five_prob if market_five_prob is not None else (0.6 * market_short_prob + 0.4 * market_mid_prob))
            + 0.55 * market_mid_prob
        ),
        "20d": float(market_mid_prob),
    }
    market = MarketForecastState(
        as_of_date=str(latest["date"].date()),
        up_1d_prob=horizon_probs["1d"],
        up_2d_prob=horizon_probs["2d"],
        up_3d_prob=horizon_probs["3d"],
        up_5d_prob=horizon_probs["5d"],
        up_10d_prob=horizon_probs["10d"],
        up_20d_prob=float(market_mid_prob),
        trend_state=str(state.state_code),
        drawdown_risk=_clip(abs(drawdown_raw), 0.0, 1.0),
        volatility_regime=volatility_regime,
        liquidity_stress=_clip(0.5 - float(cross_section_record.breadth_strength), 0.0, 1.0),
        latest_close=latest_close,
        horizon_forecasts=_build_horizon_forecasts(
            latest_close=latest_close,
            horizon_probs=horizon_probs,
            short_profile=market_short_profile,
            mid_profile=market_mid_profile,
        ),
        market_facts=_market_facts_from_row(latest),
    )
    cross_section = CrossSectionForecastState(
        as_of_date=str(cross_section_record.as_of_date.date()),
        large_vs_small_bias=float(cross_section_record.large_vs_small_bias),
        growth_vs_value_bias=float(cross_section_record.growth_vs_value_bias),
        fund_flow_strength=float(cross_section_record.fund_flow_strength),
        margin_risk_on_score=float(cross_section_record.margin_risk_on_score),
        breadth_strength=float(cross_section_record.breadth_strength),
        leader_participation=float(cross_section_record.leader_participation),
        weak_stock_ratio=float(cross_section_record.weak_stock_ratio),
    )
    return market, cross_section


def _derive_learning_targets(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    horizon_metrics: dict[str, dict[str, float]] | None = None,
    universe_tier: str | None = None,
) -> tuple[float, float, float, float]:
    ranked = sorted(state.stocks, key=_stock_policy_score, reverse=True)
    realized: list[float] = []
    realized_excess_1d: list[float] = []
    realized_excess_5d: list[float] = []
    realized_excess_20d_sector: list[float] = []
    for stock in ranked[:4]:
        frame = stock_frames.get(stock.symbol)
        if frame is None:
            continue
        row = frame[frame["date"] == date]
        if row.empty:
            continue
        realized.append(_safe_float(row.iloc[0].get("fwd_ret_1"), 0.0))
        realized_excess_1d.append(_safe_float(row.iloc[0].get("excess_ret_1_vs_mkt"), 0.0))
        realized_excess_5d.append(_safe_float(row.iloc[0].get("excess_ret_5_vs_mkt"), 0.0))
        realized_excess_20d_sector.append(_safe_float(row.iloc[0].get("excess_ret_20_vs_sector"), 0.0))
    lead_ret = float(np.mean(realized)) if realized else 0.0
    lead_excess_1d = float(np.mean(realized_excess_1d)) if realized_excess_1d else 0.0
    lead_excess_5d = float(np.mean(realized_excess_5d)) if realized_excess_5d else 0.0
    lead_excess_20d_sector = float(np.mean(realized_excess_20d_sector)) if realized_excess_20d_sector else 0.0
    alpha_metrics = _alpha_opportunity_metrics(state.stocks)
    opportunity_signal = float(
        0.45 * _signal_unit(alpha_metrics["alpha_headroom"], 0.04)
        + 0.30 * _signal_unit(alpha_metrics["breadth_ratio"] - 0.08, 0.12)
        + 0.25 * _signal_unit(alpha_metrics["top_score"] - 0.56, 0.10)
    )
    quality_20d = {} if horizon_metrics is None else dict(horizon_metrics.get("20d", {}))
    ranking_signal = float(
        0.55 * _signal_unit(quality_20d.get("rank_ic", 0.0), 0.12)
        + 0.30 * _signal_unit(quality_20d.get("top_bottom_spread", 0.0), 0.08)
        + 0.15 * _signal_unit(_safe_float(quality_20d.get("top_k_hit_rate", 0.5), 0.5) - 0.5, 0.20)
    )
    realized_alpha_signal = float(
        0.50 * _signal_unit(lead_excess_1d, 0.02)
        + 0.30 * _signal_unit(lead_excess_5d, 0.04)
        + 0.20 * _signal_unit(lead_excess_20d_sector, 0.08)
    )
    composite_signal = float(
        0.45 * realized_alpha_signal
        + 0.30 * ranking_signal
        + 0.25 * opportunity_signal
    )

    if normalize_universe_tier(universe_tier) == "generated_80":
        regime_floor = 0.45 if state.risk_regime == "risk_on" else (0.35 if state.risk_regime == "cautious" else 0.25)
        exposure = float(_clip(0.58 + 0.22 * composite_signal, regime_floor, 0.92))
        if state.market.drawdown_risk > 0.45:
            exposure *= 0.90
        if state.market.volatility_regime == "high":
            exposure *= 0.92
        if state.cross_section.weak_stock_ratio > 0.55:
            exposure *= 0.92
        if state.cross_section.breadth_strength < 0.05:
            exposure *= 0.95
        exposure = float(_clip(exposure, regime_floor, 0.92))

        positions = 3
        if alpha_metrics["breadth_ratio"] >= 0.10:
            positions += 1
        if alpha_metrics["alpha_headroom"] >= 0.02 and state.cross_section.breadth_strength >= 0.10:
            positions += 1
        if composite_signal <= -0.20 or state.cross_section.weak_stock_ratio >= 0.60:
            positions -= 1
        if state.market.volatility_regime == "high":
            positions = min(positions, 4)
        positions = int(np.clip(positions, 1, 5))

        turnover = float(
            0.18
            + 0.07 * max(0.0, realized_alpha_signal)
            + 0.04 * max(0.0, ranking_signal)
            + 0.05 * max(0.0, opportunity_signal)
            + 0.03 * abs(composite_signal)
        )
        if composite_signal < -0.10:
            turnover = min(turnover, 0.18)
        if state.market.drawdown_risk > 0.45:
            turnover = min(turnover, 0.18)
        if state.market.volatility_regime == "high":
            turnover = min(turnover, 0.24)
        turnover = float(_clip(turnover, 0.12, 0.32))
        sample_weight = float(
            1.0
            + 1.2 * abs(composite_signal)
            + 0.8 * max(0.0, realized_alpha_signal)
            + 0.5 * max(0.0, ranking_signal)
            + 0.6 * max(0.0, opportunity_signal)
        )
        return float(exposure), float(positions), float(turnover), sample_weight

    if lead_ret >= 0.008:
        exposure = 0.85
    elif lead_ret >= 0.0:
        exposure = 0.60
    else:
        exposure = 0.35

    breadth_bonus = 1 if float(state.cross_section.breadth_strength) > 0.05 else 0
    weakness_penalty = 1 if float(state.cross_section.weak_stock_ratio) > 0.55 else 0
    positions = int(np.clip(3 + breadth_bonus - weakness_penalty + (1 if lead_ret >= 0.012 else 0), 1, 5))

    turnover = 0.18
    if abs(lead_ret) >= 0.01:
        turnover = 0.32
    elif abs(lead_ret) >= 0.004:
        turnover = 0.25
    if float(state.market.drawdown_risk) > 0.45:
        turnover = min(turnover, 0.18)

    sample_weight = float(1.0 + 1.5 * abs(lead_ret))
    return float(exposure), float(positions), float(turnover), sample_weight


@dataclass(frozen=True)
class _PreparedV2BacktestData:
    settings: dict[str, object]
    market_valid: pd.DataFrame
    market_feature_cols: list[str]
    panel: pd.DataFrame
    feature_cols: list[str]
    stock_frames: dict[str, pd.DataFrame]
    dates: list[pd.Timestamp]


@dataclass(frozen=True)
class _TrajectoryStep:
    date: pd.Timestamp
    next_date: pd.Timestamp
    composite_state: CompositeState
    stock_states: list[StockForecastState]
    horizon_metrics: dict[str, dict[str, float]]


@dataclass(frozen=True)
class _BacktestTrajectory:
    prepared: _PreparedV2BacktestData
    steps: list[_TrajectoryStep]


class ForecastBackend(Protocol):
    name: str

    def build_trajectory(
        self,
        prepared: _PreparedV2BacktestData,
        *,
        retrain_days: int = 20,
        ) -> _BacktestTrajectory:
        ...


def _slice_backtest_trajectory(
    trajectory: _BacktestTrajectory,
    *,
    start: int,
    end: int,
) -> _BacktestTrajectory:
    n_steps = len(trajectory.steps)
    lo = max(0, min(int(start), n_steps))
    hi = max(lo, min(int(end), n_steps))
    return _BacktestTrajectory(
        prepared=trajectory.prepared,
        steps=list(trajectory.steps[lo:hi]),
    )


def _split_research_trajectory(
    trajectory: _BacktestTrajectory,
    split_mode: str = _DEFAULT_SPLIT_MODE,
    embargo_days: int = _DEFAULT_EMBARGO_DAYS,
) -> tuple[_BacktestTrajectory, _BacktestTrajectory, _BacktestTrajectory]:
    mode = (str(split_mode).strip().lower() or _DEFAULT_SPLIT_MODE)
    n_steps = len(trajectory.steps)
    if n_steps <= 2:
        empty = _slice_backtest_trajectory(trajectory, start=0, end=0)
        holdout = _slice_backtest_trajectory(trajectory, start=0, end=n_steps)
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
        validation = _slice_backtest_trajectory(trajectory, start=train_end, end=holdout_start)
        holdout = _slice_backtest_trajectory(trajectory, start=holdout_start, end=n_steps)
        train = _slice_backtest_trajectory(trajectory, start=0, end=train_end)
        if not validation.steps:
            validation = _slice_backtest_trajectory(trajectory, start=max(0, holdout_start - 1), end=holdout_start)
        if not holdout.steps:
            holdout = _slice_backtest_trajectory(trajectory, start=max(0, n_steps - 1), end=n_steps)
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

    train = _slice_backtest_trajectory(trajectory, start=0, end=train_end)
    validation = _slice_backtest_trajectory(trajectory, start=validation_start, end=validation_end)
    holdout = _slice_backtest_trajectory(trajectory, start=holdout_start, end=n_steps)
    if not validation.steps:
        validation = _slice_backtest_trajectory(
            trajectory,
            start=max(0, holdout_start - 1),
            end=max(holdout_start, 1),
        )
    if not holdout.steps:
        holdout = _slice_backtest_trajectory(trajectory, start=max(0, n_steps - 1), end=n_steps)
    return train, validation, holdout


def _fit_v2_info_shadow_models(
    *,
    trajectory: _BacktestTrajectory,
    settings: dict[str, object],
    info_items: list[InfoItem],
) -> tuple[dict[str, _InfoShadowModel], dict[str, _InfoShadowModel]]:
    stock_rows_by_horizon: dict[str, list[dict[str, float]]] = {"1d": [], "5d": [], "20d": []}
    market_rows_by_horizon: dict[str, list[dict[str, float]]] = {"1d": [], "5d": [], "20d": []}
    for step in trajectory.steps:
        enriched_state = _enrich_state_with_info(
            state=step.composite_state,
            as_of_date=pd.Timestamp(step.date),
            info_items=info_items,
            settings=settings,
        )
        market_row = trajectory.prepared.market_valid[trajectory.prepared.market_valid["date"] == step.date]
        if not market_row.empty:
            market_info = enriched_state.market_info_state
            row = market_row.iloc[0]
            market_rows_by_horizon["1d"].append(
                {
                    "quant_prob": float(enriched_state.market.up_1d_prob),
                    "info_prob": float(market_info.info_prob_1d),
                    "negative_event_risk": float(market_info.negative_event_risk),
                    "item_count": float(market_info.item_count),
                    "y": 1.0 if _safe_float(row.get("mkt_fwd_ret_1"), 0.0) > 0.0 else 0.0,
                }
            )
            market_rows_by_horizon["5d"].append(
                {
                    "quant_prob": float(enriched_state.market.up_5d_prob),
                    "info_prob": float(market_info.info_prob_5d),
                    "negative_event_risk": float(market_info.negative_event_risk),
                    "item_count": float(market_info.item_count),
                    "y": 1.0 if _safe_float(row.get("mkt_fwd_ret_5"), 0.0) > 0.0 else 0.0,
                }
            )
            market_rows_by_horizon["20d"].append(
                {
                    "quant_prob": float(enriched_state.market.up_20d_prob),
                    "info_prob": float(market_info.info_prob_20d),
                    "negative_event_risk": float(market_info.negative_event_risk),
                    "item_count": float(market_info.item_count),
                    "y": 1.0 if _safe_float(row.get("mkt_fwd_ret_20"), 0.0) > 0.0 else 0.0,
                }
            )
        for stock in enriched_state.stocks:
            info_state = enriched_state.stock_info_states.get(stock.symbol, InfoAggregateState())
            frame = trajectory.prepared.stock_frames.get(stock.symbol)
            if frame is None:
                continue
            row = frame[frame["date"] == step.date]
            if row.empty:
                continue
            payload = row.iloc[0]
            stock_rows_by_horizon["1d"].append(
                {
                    "quant_prob": float(stock.up_1d_prob),
                    "info_prob": float(info_state.info_prob_1d),
                    "negative_event_risk": float(info_state.negative_event_risk),
                    "item_count": float(info_state.item_count),
                    "y": 1.0 if _safe_float(payload.get("excess_ret_1_vs_mkt"), 0.0) > 0.0 else 0.0,
                }
            )
            stock_rows_by_horizon["5d"].append(
                {
                    "quant_prob": float(stock.up_5d_prob),
                    "info_prob": float(info_state.info_prob_5d),
                    "negative_event_risk": float(info_state.negative_event_risk),
                    "item_count": float(info_state.item_count),
                    "y": 1.0 if _safe_float(payload.get("excess_ret_5_vs_mkt"), 0.0) > 0.0 else 0.0,
                }
            )
            stock_rows_by_horizon["20d"].append(
                {
                    "quant_prob": float(stock.up_20d_prob),
                    "info_prob": float(info_state.info_prob_20d),
                    "negative_event_risk": float(info_state.negative_event_risk),
                    "item_count": float(info_state.item_count),
                    "y": 1.0 if _safe_float(payload.get("excess_ret_20_vs_sector"), 0.0) > 0.0 else 0.0,
                }
            )

    def _fit_bucket(bucket: list[dict[str, float]]) -> _InfoShadowModel:
        if not bucket:
            return _InfoShadowModel(mode="rule", samples=0, feature_cols=list(_INFO_SHADOW_FEATURE_COLUMNS))
        frame = _info_feature_frame(
            quant_prob=[row["quant_prob"] for row in bucket],
            info_prob=[row["info_prob"] for row in bucket],
            negative_event_risk=[row["negative_event_risk"] for row in bucket],
            item_count=[row["item_count"] for row in bucket],
        )
        frame["y"] = [float(row["y"]) for row in bucket]
        return _fit_info_shadow_model(
            frame,
            target_col="y",
            l2=float(settings.get("learned_info_l2", 0.8)),
            min_samples=int(settings.get("learned_info_min_samples", 80)),
        )

    return (
        {horizon: _fit_bucket(bucket) for horizon, bucket in stock_rows_by_horizon.items()},
        {horizon: _fit_bucket(bucket) for horizon, bucket in market_rows_by_horizon.items()},
    )


def _build_shadow_scored_rows_for_step(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, float | str]] = []
    event_day = False
    for stock in state.stocks:
        info_state = state.stock_info_states.get(stock.symbol, InfoAggregateState())
        frame = stock_frames.get(stock.symbol)
        if frame is None:
            continue
        row = frame[frame["date"] == date]
        if row.empty:
            continue
        payload = row.iloc[0]
        if info_state.item_count > 0 or info_state.negative_event_risk > 0.10:
            event_day = True
        rows.append(
            {
                "symbol": str(stock.symbol),
                "score": _compose_shadow_stock_score(stock=stock, info_state=info_state),
                "realized_ret_1d": _safe_float(payload.get("excess_ret_1_vs_mkt"), np.nan),
                "realized_ret_5d": _safe_float(payload.get("excess_ret_5_vs_mkt"), np.nan),
                "realized_ret_20d": _safe_float(payload.get("excess_ret_20_vs_sector"), np.nan),
            }
        )
    return pd.DataFrame(rows), bool(event_day)


def _filter_info_items_by_source_subset(items: Iterable[InfoItem], subset: str) -> list[InfoItem]:
    target = str(subset).strip()
    if not target:
        return list(items)
    return [item for item in items if str(getattr(item, "source_subset", "")) == target]


def _info_source_breakdown(items: Iterable[InfoItem]) -> dict[str, int]:
    counts = {
        "market_news": 0,
        "announcements": 0,
        "research": 0,
    }
    for item in items:
        subset = str(getattr(item, "source_subset", "")).strip()
        if subset in counts:
            counts[subset] = int(counts[subset] + 1)
    return counts


def _build_info_shadow_variant(
    *,
    validation_trajectory: _BacktestTrajectory,
    holdout_trajectory: _BacktestTrajectory,
    settings: dict[str, object],
    info_items: list[InfoItem],
) -> dict[str, object]:
    stock_models, market_models = _fit_v2_info_shadow_models(
        trajectory=validation_trajectory,
        settings=settings,
        info_items=info_items,
    )

    holdout_shadow_metrics: dict[str, list[float]] = {
        "1d_rank_ic": [],
        "5d_rank_ic": [],
        "20d_rank_ic": [],
        "20d_top_bottom_spread": [],
        "event_day_hit_rate_shadow": [],
        "event_day_hit_rate_quant": [],
    }
    stock_score_deltas: list[dict[str, object]] = []
    coverage_steps = 0
    market_items_total = 0
    stock_coverage_total = 0.0
    last_state: CompositeState | None = None
    last_date: pd.Timestamp | None = None

    for step in holdout_trajectory.steps:
        enriched_state = _enrich_state_with_info(
            state=step.composite_state,
            as_of_date=pd.Timestamp(step.date),
            info_items=info_items,
            settings=settings,
            stock_models=stock_models,
            market_models=market_models,
        )
        shadow_rows, event_day = _build_shadow_scored_rows_for_step(
            state=enriched_state,
            stock_frames=holdout_trajectory.prepared.stock_frames,
            date=step.date,
        )
        rank_ic_1d, _, _, top_k_1d = _panel_slice_metrics(shadow_rows, realized_col="realized_ret_1d")
        rank_ic_5d, _, _, _ = _panel_slice_metrics(shadow_rows, realized_col="realized_ret_5d")
        rank_ic_20d, _, spread_20d, top_k_20d = _panel_slice_metrics(shadow_rows, realized_col="realized_ret_20d")
        holdout_shadow_metrics["1d_rank_ic"].append(float(rank_ic_1d))
        holdout_shadow_metrics["5d_rank_ic"].append(float(rank_ic_5d))
        holdout_shadow_metrics["20d_rank_ic"].append(float(rank_ic_20d))
        holdout_shadow_metrics["20d_top_bottom_spread"].append(float(spread_20d))
        if event_day:
            holdout_shadow_metrics["event_day_hit_rate_shadow"].append(float(top_k_20d))
            holdout_shadow_metrics["event_day_hit_rate_quant"].append(float(step.horizon_metrics["20d"]["top_k_hit_rate"]))
        if enriched_state.market_info_state.item_count > 0:
            coverage_steps += 1
        market_items_total += int(enriched_state.market_info_state.item_count)
        covered_names = sum(1 for item in enriched_state.stock_info_states.values() if item.item_count > 0)
        stock_coverage_total += float(covered_names / max(1, len(enriched_state.stock_info_states)))
        for stock in enriched_state.stocks:
            info_state = enriched_state.stock_info_states.get(stock.symbol, InfoAggregateState())
            quant_score = _stock_policy_score(stock)
            shadow_score = _compose_shadow_stock_score(stock=stock, info_state=info_state)
            stock_score_deltas.append(
                {
                    "symbol": str(stock.symbol),
                    "sector": str(stock.sector),
                    "quant_score": float(quant_score),
                    "shadow_score": float(shadow_score),
                    "delta": float(shadow_score - quant_score),
                    "negative_event_risk": float(info_state.negative_event_risk),
                    "item_count": int(info_state.item_count),
                }
            )
        last_state = enriched_state
        last_date = pd.Timestamp(step.date)

    stock_score_deltas.sort(key=lambda item: float(item["delta"]), reverse=True)
    top_positive = stock_score_deltas[:5]
    top_negative = sorted(stock_score_deltas, key=lambda item: float(item["delta"]))[:5]
    coverage_summary = {
        "holdout_step_coverage_ratio": float(coverage_steps / max(1, len(holdout_trajectory.steps))),
        "avg_market_item_count": float(market_items_total / max(1, len(holdout_trajectory.steps))),
        "avg_stock_coverage_ratio": float(stock_coverage_total / max(1, len(holdout_trajectory.steps))),
        "market_coverage_ratio": float(coverage_steps / max(1, len(holdout_trajectory.steps))),
        "stock_coverage_ratio": float(stock_coverage_total / max(1, len(holdout_trajectory.steps))),
    }
    return {
        "market_shadow_modes": {horizon: model.mode for horizon, model in market_models.items()},
        "stock_shadow_modes": {horizon: model.mode for horizon, model in stock_models.items()},
        "model_samples": {
            "market": {horizon: int(model.samples) for horizon, model in market_models.items()},
            "stock": {horizon: int(model.samples) for horizon, model in stock_models.items()},
        },
        "avg_1d_rank_ic": float(np.mean(holdout_shadow_metrics["1d_rank_ic"])) if holdout_shadow_metrics["1d_rank_ic"] else 0.0,
        "avg_5d_rank_ic": float(np.mean(holdout_shadow_metrics["5d_rank_ic"])) if holdout_shadow_metrics["5d_rank_ic"] else 0.0,
        "avg_20d_rank_ic": float(np.mean(holdout_shadow_metrics["20d_rank_ic"])) if holdout_shadow_metrics["20d_rank_ic"] else 0.0,
        "avg_20d_top_bottom_spread": float(np.mean(holdout_shadow_metrics["20d_top_bottom_spread"])) if holdout_shadow_metrics["20d_top_bottom_spread"] else 0.0,
        "event_day_hit_rate": float(np.mean(holdout_shadow_metrics["event_day_hit_rate_shadow"])) if holdout_shadow_metrics["event_day_hit_rate_shadow"] else 0.0,
        "quant_event_day_hit_rate": float(np.mean(holdout_shadow_metrics["event_day_hit_rate_quant"])) if holdout_shadow_metrics["event_day_hit_rate_quant"] else 0.0,
        "coverage_summary": coverage_summary,
        "top_positive_stock_deltas": top_positive,
        "top_negative_stock_deltas": top_negative,
        "last_market_info_state": {} if last_state is None else asdict(last_state.market_info_state),
        "last_date": "" if last_date is None else str(last_date.date()),
    }


def _build_info_shadow_report(
    *,
    validation_trajectory: _BacktestTrajectory,
    holdout_trajectory: _BacktestTrajectory,
    settings: dict[str, object],
    info_items: list[InfoItem],
) -> dict[str, object]:
    all_variant = _build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=info_items,
    )
    market_news_variant = _build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=_filter_info_items_by_source_subset(info_items, "market_news"),
    )
    announcement_variant = _build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=_filter_info_items_by_source_subset(info_items, "announcements"),
    )
    research_variant = _build_info_shadow_variant(
        validation_trajectory=validation_trajectory,
        holdout_trajectory=holdout_trajectory,
        settings=settings,
        info_items=_filter_info_items_by_source_subset(info_items, "research"),
    )
    report = {
        "info_shadow_enabled": bool(settings.get("use_info_fusion", False)),
        "shadow_only": bool(settings.get("info_shadow_only", True)),
        "market_shadow_modes": dict(all_variant.get("market_shadow_modes", {})),
        "stock_shadow_modes": dict(all_variant.get("stock_shadow_modes", {})),
        "model_samples": dict(all_variant.get("model_samples", {})),
        "quant_only": {
            "avg_20d_rank_ic": float(np.mean([float(step.horizon_metrics["20d"]["rank_ic"]) for step in holdout_trajectory.steps])) if holdout_trajectory.steps else 0.0,
            "avg_20d_top_bottom_spread": float(np.mean([float(step.horizon_metrics["20d"]["top_bottom_spread"]) for step in holdout_trajectory.steps])) if holdout_trajectory.steps else 0.0,
            "event_day_hit_rate": float(all_variant.get("quant_event_day_hit_rate", 0.0)),
        },
        "quant_plus_info_shadow": {
            key: value
            for key, value in all_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "market_news_only": {
            key: value
            for key, value in market_news_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "announcements_only": {
            key: value
            for key, value in announcement_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "research_only": {
            key: value
            for key, value in research_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "all_info_combined": {
            key: value
            for key, value in all_variant.items()
            if key in {"avg_1d_rank_ic", "avg_5d_rank_ic", "avg_20d_rank_ic", "avg_20d_top_bottom_spread", "event_day_hit_rate"}
        },
        "coverage_summary": dict(all_variant.get("coverage_summary", {})),
        "top_positive_stock_deltas": list(all_variant.get("top_positive_stock_deltas", [])),
        "top_negative_stock_deltas": list(all_variant.get("top_negative_stock_deltas", [])),
        "event_tag_distribution": event_tag_counts(info_items),
        "info_source_breakdown": _info_source_breakdown(info_items),
        "last_market_info_state": dict(all_variant.get("last_market_info_state", {})),
        "last_date": str(all_variant.get("last_date", "")),
    }
    return report


def _build_info_manifest_payload(
    *,
    settings: dict[str, object],
    info_file: str,
    info_items: list[InfoItem],
    as_of_date: pd.Timestamp,
    config_hash: str,
    shadow_enabled: bool,
    shadow_report: dict[str, object] | None = None,
) -> dict[str, object]:
    counts: dict[str, int] = {}
    for item in info_items:
        counts[item.info_type] = int(counts.get(item.info_type, 0) + 1)
    source_breakdown = _info_source_breakdown(info_items)
    date_window = {
        "start": "",
        "end": "",
    }
    if info_items:
        date_window = {
            "start": str(min(item.date for item in info_items)),
            "end": str(max(item.date for item in info_items)),
        }
    info_hash = _sha256_file(info_file) if info_file else ""
    if not info_hash:
        info_hash = _stable_json_hash([asdict(item) for item in info_items])
    return {
        "info_file": str(info_file),
        "info_hash": str(info_hash),
        "info_item_count": int(len(info_items)),
        "info_type_counts": counts,
        "info_source_breakdown": source_breakdown,
        "market_news_count": int(source_breakdown.get("market_news", 0)),
        "announcement_count": int(source_breakdown.get("announcements", 0)),
        "research_count": int(source_breakdown.get("research", 0)),
        "date_window": date_window,
        "coverage_summary": {} if shadow_report is None else dict(shadow_report.get("coverage_summary", {})),
        "market_coverage_ratio": float(
            (shadow_report or {}).get("coverage_summary", {}).get("market_coverage_ratio", 0.0)
        ),
        "stock_coverage_ratio": float(
            (shadow_report or {}).get("coverage_summary", {}).get("stock_coverage_ratio", 0.0)
        ),
        "config_hash": str(config_hash),
        "info_shadow_enabled": bool(shadow_enabled),
        "info_shadow_only": bool(settings.get("info_shadow_only", True)),
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "info_types": [str(item) for item in settings.get("info_types", [])],
        "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
        "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
        "as_of_date": str(as_of_date.date()),
    }


def _empty_v2_backtest_result() -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    return (
        _to_v2_backtest_summary(
            returns=[],
            turnovers=[],
            costs=[],
            gross_returns=[],
            fill_ratios=[],
            slippage_bps=[],
            dates=[],
        ),
        [],
    )


def _prepare_v2_backtest_data(
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
) -> _PreparedV2BacktestData | None:
    settings = _load_v2_runtime_settings(
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
    settings = _resolve_v2_universe_settings(settings=settings, cache_root=cache_root)
    prepared_cache_key = _build_prepared_backtest_cache_key_external(settings)
    prepared_cache_path = _prepared_backtest_cache_path_external(
        cache_root=cache_root,
        cache_key=prepared_cache_key,
    )
    if not refresh_cache:
        cached_prepared = _load_pickle_cache_external(prepared_cache_path)
        if cached_prepared is not None:
            _emit_progress("cache", "命中 prepared data 缓存")
            return cached_prepared
    market_security, _, _ = load_watchlist(str(settings["watchlist"]))
    universe = build_candidate_universe(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        universe_file=str(settings["universe_file"]),
        candidate_limit=max(5, int(settings["universe_limit"])),
        exclude_symbols=[market_security.symbol],
    )
    stocks = universe.rows
    if not stocks:
        return None

    market_raw = load_symbol_daily(
        symbol=market_security.symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
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
    market_feature_cols = list(MARKET_FEATURE_COLUMNS) + list(market_context.feature_columns)
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

    panel_bundle = build_stock_panel_dataset(
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

    prepared = _PreparedV2BacktestData(
        settings=settings,
        market_valid=market_valid,
        market_feature_cols=market_feature_cols,
        panel=panel,
        feature_cols=feature_cols,
        stock_frames=stock_frames,
        dates=dates,
    )
    try:
        _store_pickle_cache_external(prepared_cache_path, prepared)
        _emit_progress("cache", "prepared data 缓存已写入")
    except Exception:
        pass
    return prepared


def _build_frozen_linear_forecast_bundle(
    prepared: _PreparedV2BacktestData,
) -> dict[str, object]:
    settings = prepared.settings
    market_train = prepared.market_valid.copy()
    panel_train = prepared.panel.copy()

    market_models = {
        "1d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(market_train, prepared.market_feature_cols, "mkt_target_1d_up")
        ),
        "2d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(market_train, prepared.market_feature_cols, "mkt_target_2d_up")
        ),
        "3d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(market_train, prepared.market_feature_cols, "mkt_target_3d_up")
        ),
        "5d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(market_train, prepared.market_feature_cols, "mkt_target_5d_up")
        ),
        "20d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(market_train, prepared.market_feature_cols, "mkt_target_20d_up")
        ),
    }
    market_quantiles = {
        "1d": _serialize_quantile_bundle(
            _fit_quantile_quintet(
                market_train,
                feature_cols=prepared.market_feature_cols,
                target_col="mkt_fwd_ret_1",
                l2=float(settings["l2"]),
            )
        ),
        "20d": _serialize_quantile_bundle(
            _fit_quantile_quintet(
                market_train,
                feature_cols=prepared.market_feature_cols,
                target_col="mkt_fwd_ret_20",
                l2=float(settings["l2"]),
            )
        ),
    }
    stock_models = {
        "1d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(panel_train, prepared.feature_cols, "target_1d_excess_mkt_up")
        ),
        "2d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(panel_train, prepared.feature_cols, "target_2d_excess_mkt_up")
        ),
        "3d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(panel_train, prepared.feature_cols, "target_3d_excess_mkt_up")
        ),
        "5d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(panel_train, prepared.feature_cols, "target_5d_excess_mkt_up")
        ),
        "20d": _serialize_binary_model(
            LogisticBinaryModel(l2=float(settings["l2"])).fit(panel_train, prepared.feature_cols, "target_20d_excess_sector_up")
        ),
    }
    stock_quantiles = {
        "1d": _serialize_quantile_bundle(
            _fit_quantile_quintet(
                panel_train,
                feature_cols=prepared.feature_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=float(settings["l2"]),
            )
        ),
        "20d": _serialize_quantile_bundle(
            _fit_quantile_quintet(
                panel_train,
                feature_cols=prepared.feature_cols,
                target_col="excess_ret_20_vs_sector",
                l2=float(settings["l2"]),
            )
        ),
    }
    return {
        "format_version": 1,
        "backend": "linear",
        "created_from_end_date": str(pd.Timestamp(max(prepared.dates)).date()),
        "market_feature_cols": list(prepared.market_feature_cols),
        "panel_feature_cols": list(prepared.feature_cols),
        "market_models": market_models,
        "market_quantiles": market_quantiles,
        "stock_models": stock_models,
        "stock_quantiles": stock_quantiles,
    }


def _load_frozen_forecast_bundle(path_like: object) -> dict[str, object]:
    payload = _load_json_dict(path_like)
    if not payload:
        return {}
    if str(payload.get("backend", "")).strip().lower() not in {"linear", "deep"}:
        return {}
    ForecastBundle.from_payload(payload)
    return payload


def _build_live_market_frame(
    *,
    settings: dict[str, object],
    market_symbol: str,
) -> pd.DataFrame:
    market_raw = load_symbol_daily(
        symbol=market_symbol,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
    )
    market_feat_base = make_market_feature_frame(market_raw)
    market_context = build_market_context_features(
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_dates=market_feat_base["date"],
        use_margin_features=bool(settings.get("use_margin_features", False)),
        margin_market_file=str(settings.get("margin_market_file", "")),
        use_us_index_context=bool(settings.get("use_us_index_context", False)),
        us_index_source=str(settings.get("us_index_source", "akshare")),
        use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
        use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
        cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
    )
    return market_feat_base.merge(market_context.frame, on="date", how="left", validate="1:1")


def _score_live_composite_state_from_frozen_bundle(
    *,
    bundle: dict[str, object],
    settings: dict[str, object],
    universe_ctx: _DailyUniverseContext,
) -> tuple[CompositeState | None, list[object]]:
    if str(bundle.get("backend", "")).strip().lower() != "linear":
        return None, []

    market_feature_cols = [str(item) for item in bundle.get("market_feature_cols", [])]
    panel_feature_cols = [str(item) for item in bundle.get("panel_feature_cols", [])]
    if not market_feature_cols or not panel_feature_cols:
        return None, []

    market_frame = _build_live_market_frame(
        settings=settings,
        market_symbol=str(getattr(universe_ctx.market_security, "symbol", "")),
    )
    if market_frame.empty:
        return None, []
    latest_market = market_frame.sort_values("date").iloc[[-1]].copy()
    if any(col not in latest_market.columns for col in market_feature_cols):
        return None, []
    latest_market = latest_market.dropna(subset=market_feature_cols)
    if latest_market.empty:
        return None, []

    market_models_raw = bundle.get("market_models", {})
    market_quantiles_raw = bundle.get("market_quantiles", {})
    if not isinstance(market_models_raw, dict) or not isinstance(market_quantiles_raw, dict):
        return None, []
    market_short_model = _deserialize_binary_model(dict(market_models_raw.get("1d", {})))
    market_two_model = _deserialize_binary_model(dict(market_models_raw.get("2d", {})))
    market_three_model = _deserialize_binary_model(dict(market_models_raw.get("3d", {})))
    market_five_model = _deserialize_binary_model(dict(market_models_raw.get("5d", {})))
    market_mid_model = _deserialize_binary_model(dict(market_models_raw.get("20d", {})))
    market_short_q = _deserialize_quantile_bundle(market_quantiles_raw.get("1d"))
    market_mid_q = _deserialize_quantile_bundle(market_quantiles_raw.get("20d"))

    mkt_short = float(market_short_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_two = float(market_two_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_three = float(market_three_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_five = float(market_five_model.predict_proba(latest_market, market_feature_cols)[0])
    mkt_mid = float(market_mid_model.predict_proba(latest_market, market_feature_cols)[0])
    market_short_profile = _predict_quantile_profile(
        latest_market,
        feature_cols=market_feature_cols,
        q_models=market_short_q,
    )
    market_mid_profile = _predict_quantile_profile(
        latest_market,
        feature_cols=market_feature_cols,
        q_models=market_mid_q,
    )
    market_state, cross_section = _build_market_and_cross_section_from_prebuilt_frame(
        market_frame=market_frame,
        market_short_prob=mkt_short,
        market_two_prob=mkt_two,
        market_three_prob=mkt_three,
        market_five_prob=mkt_five,
        market_mid_prob=mkt_mid,
        market_short_profile=market_short_profile,
        market_mid_profile=market_mid_profile,
    )

    panel_bundle = build_stock_live_panel_dataset(
        stock_securities=universe_ctx.stocks,
        source=str(settings["source"]),
        data_dir=str(settings["data_dir"]),
        start=str(settings["start"]),
        end=str(settings["end"]),
        market_frame=market_frame,
        extra_market_cols=[col for col in market_frame.columns if col.startswith("us_") or col.startswith("cn_") or col.startswith("breadth_") or col.startswith("fin_") or col.startswith("sec_")],
        use_margin_features=bool(settings.get("use_margin_features", False)),
        margin_stock_file=str(settings.get("margin_stock_file", "")),
    )
    live_panel = panel_bundle.frame
    if live_panel.empty or any(col not in live_panel.columns for col in panel_feature_cols):
        return None, []
    latest_panel_date = pd.to_datetime(live_panel["date"], errors="coerce").dropna().max()
    latest_panel = live_panel[live_panel["date"] == latest_panel_date].copy()
    latest_panel = latest_panel.dropna(subset=panel_feature_cols)
    if latest_panel.empty:
        return None, []

    stock_models_raw = bundle.get("stock_models", {})
    stock_quantiles_raw = bundle.get("stock_quantiles", {})
    if not isinstance(stock_models_raw, dict) or not isinstance(stock_quantiles_raw, dict):
        return None, []
    stock_states, _ = _build_stock_states_from_panel_slice(
        panel_row=latest_panel,
        feature_cols=panel_feature_cols,
        short_model=_deserialize_binary_model(dict(stock_models_raw.get("1d", {}))),
        two_model=_deserialize_binary_model(dict(stock_models_raw.get("2d", {}))),
        three_model=_deserialize_binary_model(dict(stock_models_raw.get("3d", {}))),
        five_model=_deserialize_binary_model(dict(stock_models_raw.get("5d", {}))),
        mid_model=_deserialize_binary_model(dict(stock_models_raw.get("20d", {}))),
        short_q_models=_deserialize_quantile_bundle(stock_quantiles_raw.get("1d")),
        mid_q_models=_deserialize_quantile_bundle(stock_quantiles_raw.get("20d")),
    )
    if not stock_states:
        return None, []

    sector_states = _build_sector_states_external(
        stock_states,
        stock_score_fn=_stock_policy_score,
    )
    composite_state = compose_state(
        market=market_state,
        sectors=sector_states,
        stocks=stock_states,
        cross_section=cross_section,
    )
    return composite_state, []


def _tensorize_temporal_frame(
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

    def build_trajectory(
        self,
        prepared: _PreparedV2BacktestData,
        *,
        retrain_days: int = 20,
    ) -> _BacktestTrajectory:
        settings = prepared.settings
        market_valid = prepared.market_valid
        panel = prepared.panel
        market_feature_cols = prepared.market_feature_cols
        feature_cols = prepared.feature_cols
        dates = prepared.dates
        min_train_days = int(settings["min_train_days"])
        steps: list[_TrajectoryStep] = []
        market_sorted, market_bounds = _build_date_slice_index(
            market_valid,
            sort_cols=["date"],
        )
        panel_sorted, panel_bounds = _build_date_slice_index(
            panel,
            sort_cols=["date", "symbol"],
        )
        block_starts = list(range(min_train_days, len(dates) - 1, max(1, int(retrain_days))))
        _emit_progress(
            "trajectory",
            f"backend={self.name} 开始构建轨迹: blocks={len(block_starts)}, dates={len(dates)}, universe={len(prepared.stock_frames)}",
        )
        trajectory_started = time.perf_counter()

        for block_idx, block_start in enumerate(block_starts, start=1):
            elapsed = time.perf_counter() - trajectory_started
            completed = max(0, block_idx - 1)
            eta = 0.0 if completed <= 0 else (elapsed / completed) * (len(block_starts) - completed)
            _emit_progress(
                "trajectory",
                f"backend={self.name} 训练块 {block_idx}/{len(block_starts)}: 截止 {pd.Timestamp(dates[block_start - 1]).date()} | elapsed={_format_elapsed(elapsed)} | eta={_format_elapsed(eta)}",
            )
            train_cutoff = market_bounds.get(dates[block_start - 1], (0, 0))[1]
            market_train = market_sorted.iloc[:train_cutoff].copy()
            if market_train.empty:
                continue
            market_short_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_1d_up",
            )
            market_two_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_2d_up",
            )
            market_three_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_3d_up",
            )
            market_five_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_5d_up",
            )
            market_mid_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                market_feature_cols,
                "mkt_target_20d_up",
            )
            panel_cutoff = panel_bounds.get(dates[block_start - 1], (0, 0))[1]
            panel_train = panel_sorted.iloc[:panel_cutoff].copy()
            if panel_train.empty:
                continue
            panel_short_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_1d_excess_mkt_up",
            )
            panel_two_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_2d_excess_mkt_up",
            )
            panel_three_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_3d_excess_mkt_up",
            )
            panel_five_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_5d_excess_mkt_up",
            )
            panel_mid_model = LogisticBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                feature_cols,
                "target_20d_excess_sector_up",
            )
            panel_short_q_models = _fit_quantile_quintet(
                panel_train,
                feature_cols=feature_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=float(settings["l2"]),
            )
            panel_mid_q_models = _fit_quantile_quintet(
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
                market_state, cross_section = _build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=market_sorted.iloc[:market_end].copy(),
                    market_short_prob=mkt_short,
                    market_two_prob=mkt_two,
                    market_three_prob=mkt_three,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_start, panel_end = panel_bounds.get(date, (0, 0))
                panel_row = panel_sorted.iloc[panel_start:panel_end].copy()
                stock_states, scored_rows = _build_stock_states_from_panel_slice(
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
                sector_states = _build_sector_states_external(
                    stock_states,
                    stock_score_fn=_stock_policy_score,
                )
                composite_state = compose_state(
                    market=market_state,
                    sectors=sector_states,
                    stocks=stock_states,
                    cross_section=cross_section,
                )
                steps.append(
                    _TrajectoryStep(
                        date=date,
                        next_date=next_date,
                        composite_state=composite_state,
                        stock_states=stock_states,
                        horizon_metrics=_panel_horizon_metrics(scored_rows),
                    )
                )

        return _BacktestTrajectory(prepared=prepared, steps=steps)


class DeepForecastBackend:
    name = "deep"

    def build_trajectory(
        self,
        prepared: _PreparedV2BacktestData,
        *,
        retrain_days: int = 20,
    ) -> _BacktestTrajectory:
        settings = prepared.settings
        market_valid = prepared.market_valid
        panel = prepared.panel
        dates = prepared.dates
        min_train_days = int(settings["min_train_days"])
        steps: list[_TrajectoryStep] = []
        market_valid_sorted, market_valid_bounds = _build_date_slice_index(
            market_valid,
            sort_cols=["date"],
        )

        tensor_market, tensor_market_cols = _tensorize_temporal_frame(
            market_valid,
            feature_cols=prepared.market_feature_cols,
            group_col=None,
            lag_depth=3,
        )
        _emit_progress("trajectory", f"backend={self.name} 已完成市场时序张量化: cols={len(tensor_market_cols)}")
        tensor_panel, tensor_panel_cols = _tensorize_temporal_frame(
            panel,
            feature_cols=prepared.feature_cols,
            group_col="symbol",
            lag_depth=3,
        )
        _emit_progress("trajectory", f"backend={self.name} 已完成个股时序张量化: cols={len(tensor_panel_cols)}")
        tensor_market_sorted, tensor_market_bounds = _build_date_slice_index(
            tensor_market,
            sort_cols=["date"],
        )
        tensor_panel_sorted, tensor_panel_bounds = _build_date_slice_index(
            tensor_panel,
            sort_cols=["date", "symbol"],
        )
        block_starts = list(range(min_train_days, len(dates) - 1, max(1, int(retrain_days))))
        _emit_progress(
            "trajectory",
            f"backend={self.name} 开始构建轨迹: blocks={len(block_starts)}, dates={len(dates)}, universe={len(prepared.stock_frames)}",
        )
        trajectory_started = time.perf_counter()

        for block_idx, block_start in enumerate(block_starts, start=1):
            elapsed = time.perf_counter() - trajectory_started
            completed = max(0, block_idx - 1)
            eta = 0.0 if completed <= 0 else (elapsed / completed) * (len(block_starts) - completed)
            _emit_progress(
                "trajectory",
                f"backend={self.name} 训练块 {block_idx}/{len(block_starts)}: 截止 {pd.Timestamp(dates[block_start - 1]).date()} | elapsed={_format_elapsed(elapsed)} | eta={_format_elapsed(eta)}",
            )
            train_cutoff = tensor_market_bounds.get(dates[block_start - 1], (0, 0))[1]
            market_train = tensor_market_sorted.iloc[:train_cutoff].copy()
            if market_train.empty:
                continue
            market_short_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_1d_up",
            )
            market_two_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_2d_up",
            )
            market_three_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_3d_up",
            )
            market_five_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_5d_up",
            )
            market_mid_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                market_train,
                tensor_market_cols,
                "mkt_target_20d_up",
            )
            panel_cutoff = tensor_panel_bounds.get(dates[block_start - 1], (0, 0))[1]
            panel_train = tensor_panel_sorted.iloc[:panel_cutoff].copy()
            if panel_train.empty:
                continue
            panel_short_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_1d_excess_mkt_up",
            )
            panel_two_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_2d_excess_mkt_up",
            )
            panel_three_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_3d_excess_mkt_up",
            )
            panel_five_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_5d_excess_mkt_up",
            )
            panel_mid_model = MLPBinaryModel(l2=float(settings["l2"])).fit(
                panel_train,
                tensor_panel_cols,
                "target_20d_excess_sector_up",
            )
            panel_short_q_models = _fit_mlp_quantile_quintet(
                panel_train,
                feature_cols=tensor_panel_cols,
                target_col="excess_ret_1_vs_mkt",
                l2=float(settings["l2"]),
            )
            panel_mid_q_models = _fit_mlp_quantile_quintet(
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
                market_state, cross_section = _build_market_and_cross_section_from_prebuilt_frame(
                    market_frame=market_valid_sorted.iloc[:market_hist_end].copy(),
                    market_short_prob=mkt_short,
                    market_two_prob=mkt_two,
                    market_three_prob=mkt_three,
                    market_five_prob=mkt_five,
                    market_mid_prob=mkt_mid,
                )
                panel_start, panel_end = tensor_panel_bounds.get(date, (0, 0))
                panel_row = tensor_panel_sorted.iloc[panel_start:panel_end].copy()
                stock_states, scored_rows = _build_stock_states_from_panel_slice(
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
                sector_states = _build_sector_states_external(
                    stock_states,
                    stock_score_fn=_stock_policy_score,
                )
                composite_state = compose_state(
                    market=market_state,
                    sectors=sector_states,
                    stocks=stock_states,
                    cross_section=cross_section,
                )
                steps.append(
                    _TrajectoryStep(
                        date=date,
                        next_date=next_date,
                        composite_state=composite_state,
                        stock_states=stock_states,
                        horizon_metrics=_panel_horizon_metrics(scored_rows),
                    )
                )

        return _BacktestTrajectory(prepared=prepared, steps=steps)


def _make_forecast_backend(name: str | None) -> ForecastBackend:
    backend = (str(name).strip().lower() if name is not None else "linear") or "linear"
    if backend == "linear":
        return LinearForecastBackend()
    if backend == "deep":
        return DeepForecastBackend()
    raise ValueError(f"Unsupported forecast backend: {backend}")


def _trajectory_cache_key(
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


def _trajectory_cache_path(
    *,
    cache_root: str,
    cache_key: str,
) -> Path:
    root = Path(str(cache_root))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{cache_key}.pkl"


def _file_mtime_token(path_like: object) -> int:
    try:
        path = Path(str(path_like))
    except Exception:
        return 0
    if not path.exists():
        return 0
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return 0


def _daily_result_cache_key(
    *,
    strategy_id: str,
    settings: dict[str, object],
    artifact_root: str,
    run_id: str = "",
    snapshot_path: str = "",
    allow_retrain: bool = False,
) -> str:
    policy_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    manifest_path = _resolve_manifest_path(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        run_id=run_id,
        snapshot_path=snapshot_path,
    )
    payload = {
        "version": "v2-daily-cache-4",
        "strategy_id": str(strategy_id),
        "config_path": str(Path(str(settings.get("config_path", ""))).resolve()),
        "source": str(settings.get("source", "")),
        "watchlist": str(Path(str(settings.get("watchlist", ""))).resolve()),
        "watchlist_mtime": _file_mtime_token(settings.get("watchlist", "")),
        "universe_file": str(Path(str(settings.get("universe_file", ""))).resolve()),
        "universe_mtime": _file_mtime_token(settings.get("universe_file", "")),
        "universe_limit": int(settings.get("universe_limit", 0)),
        "universe_tier": str(settings.get("universe_tier", "")),
        "source_universe_manifest_path": str(settings.get("source_universe_manifest_path", "")),
        "source_universe_manifest_mtime": _file_mtime_token(settings.get("source_universe_manifest_path", "")),
        "start": str(settings.get("start", "")),
        "end": str(settings.get("end", "")),
        "min_train_days": int(settings.get("min_train_days", 0)),
        "step_days": int(settings.get("step_days", 0)),
        "l2": float(settings.get("l2", 0.0)),
        "max_positions": int(settings.get("max_positions", 0)),
        "use_margin_features": bool(settings.get("use_margin_features", False)),
        "margin_market_file": str(settings.get("margin_market_file", "")),
        "margin_market_mtime": _file_mtime_token(settings.get("margin_market_file", "")),
        "margin_stock_file": str(settings.get("margin_stock_file", "")),
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
        "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
        "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
        "margin_stock_mtime": _file_mtime_token(settings.get("margin_stock_file", "")),
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "info_file": str(Path(str(_resolve_info_file_from_settings(settings) or "")).resolve()) if _resolve_info_file_from_settings(settings) else "",
        "info_file_mtime": _file_mtime_token(_resolve_info_file_from_settings(settings)),
        "info_hash": str(settings.get("info_hash", "")),
        "use_info_fusion": bool(settings.get("use_info_fusion", False)),
        "info_shadow_only": bool(settings.get("info_shadow_only", True)),
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "info_types": [str(item) for item in settings.get("info_types", [])],
        "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
        "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
        "published_policy_path": str(policy_path.resolve()),
        "published_policy_mtime": _file_mtime_token(policy_path),
        "run_id": str(run_id),
        "snapshot_path": str(snapshot_path),
        "allow_retrain": bool(allow_retrain),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_mtime": _file_mtime_token(manifest_path),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _daily_result_cache_path(
    *,
    cache_root: str,
    cache_key: str,
) -> Path:
    root = Path(str(cache_root))
    root.mkdir(parents=True, exist_ok=True)
    return root / f"daily_{cache_key}.pkl"


def _load_or_build_v2_backtest_trajectory(
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
) -> _BacktestTrajectory | None:
    backend = _make_forecast_backend(forecast_backend)
    settings = _load_v2_runtime_settings(
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
    settings = _resolve_v2_universe_settings(settings=settings, cache_root=cache_root)
    cache_key = _trajectory_cache_key(
        config_path=str(settings.get("config_path", config_path)),
        source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
        universe_file=str(settings.get("universe_file", universe_file))
        if settings.get("universe_file", universe_file) is not None
        else None,
        universe_limit=(
            int(settings.get("universe_limit"))
            if settings.get("universe_limit") is not None
            else universe_limit
        ),
        universe_tier=str(settings.get("universe_tier", universe_tier)),
        retrain_days=retrain_days,
        forecast_backend=backend.name,
        use_us_index_context=bool(settings.get("use_us_index_context", False)),
        us_index_source=str(settings.get("us_index_source", "akshare")),
        use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
        use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
        cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
    )
    cache_path = _trajectory_cache_path(cache_root=cache_root, cache_key=cache_key)
    if not refresh_cache and cache_path.exists():
        _emit_progress("cache", f"命中轨迹缓存: backend={backend.name}")
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if cached is not None:
                return cached
        except Exception:
            pass
    else:
        _emit_progress("cache", f"轨迹缓存未命中: backend={backend.name}，准备重建")

    _emit_progress("research", "开始准备研究数据")
    prepared = _prepare_v2_backtest_data(
        config_path=str(settings.get("config_path", config_path)),
        source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
        universe_file=str(settings.get("universe_file", universe_file))
        if settings.get("universe_file", universe_file) is not None
        else None,
        universe_limit=(
            int(settings.get("universe_limit"))
            if settings.get("universe_limit") is not None
            else universe_limit
        ),
        universe_tier=str(settings.get("universe_tier", universe_tier)),
        dynamic_universe=_parse_boolish(settings.get("dynamic_universe_enabled", False), False),
        generator_target_size=int(settings.get("generator_target_size", settings.get("universe_limit", 0)) or 0),
        generator_coarse_size=int(settings.get("generator_coarse_size", 0) or 0),
        generator_theme_aware=_parse_boolish(settings.get("generator_theme_aware", True), True),
        generator_use_concepts=_parse_boolish(settings.get("generator_use_concepts", True), True),
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        use_us_index_context=bool(settings.get("use_us_index_context", False)),
        us_index_source=str(settings.get("us_index_source", "akshare")),
    )
    if prepared is None:
        return None
    _emit_progress("research", "开始构建预测轨迹")
    trajectory = _build_v2_backtest_trajectory_from_prepared(
        prepared,
        retrain_days=retrain_days,
        forecast_backend=backend.name,
    )
    try:
        with cache_path.open("wb") as f:
            pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
        _emit_progress("cache", f"轨迹缓存已写入: backend={backend.name}")
    except Exception:
        pass
    return trajectory


def _build_v2_backtest_trajectory_from_prepared(
    prepared: _PreparedV2BacktestData,
    *,
    retrain_days: int = 20,
    forecast_backend: str = "linear",
) -> _BacktestTrajectory:
    backend = _make_forecast_backend(forecast_backend)
    return backend.build_trajectory(prepared, retrain_days=retrain_days)


def _execute_v2_backtest_trajectory(
    trajectory: _BacktestTrajectory,
    *,
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    capture_learning_rows: bool = False,
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    _ = retrain_days
    commission_rate = max(0.0, float(commission_bps)) / 10000.0
    slippage_rate = max(0.0, float(slippage_bps)) / 10000.0
    returns: list[float] = []
    benchmark_returns: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    costs: list[float] = []
    fill_ratios: list[float] = []
    slippage_cost_bps: list[float] = []
    rank_ics: list[float] = []
    top_decile_returns: list[float] = []
    top_bottom_spreads: list[float] = []
    top_k_hit_rates: list[float] = []
    horizon_metric_series: dict[str, dict[str, list[float]]] = {}
    out_dates: list[pd.Timestamp] = []
    prev_weights: dict[str, float] = {}
    prev_holding_days: dict[str, int] = {}
    prev_cash = 1.0
    learning_rows: list[dict[str, float]] = []

    for step in trajectory.steps:
        rank_ics.append(float(step.horizon_metrics["20d"]["rank_ic"]))
        top_decile_returns.append(float(step.horizon_metrics["20d"]["top_decile_return"]))
        top_bottom_spreads.append(float(step.horizon_metrics["20d"]["top_bottom_spread"]))
        top_k_hit_rates.append(float(step.horizon_metrics["20d"]["top_k_hit_rate"]))
        for horizon, metric_map in step.horizon_metrics.items():
            horizon_bucket = horizon_metric_series.setdefault(
                horizon,
                {"rank_ic": [], "top_decile_return": [], "top_bottom_spread": [], "top_k_hit_rate": []},
            )
            for name, value in metric_map.items():
                horizon_bucket.setdefault(name, []).append(float(value))

        active_policy_spec = policy_spec
        if learned_policy is not None:
            active_policy_spec = _policy_spec_from_model(
                state=step.composite_state,
                model=learned_policy,
            )
        decision = apply_policy(
            PolicyInput(
                composite_state=step.composite_state,
                current_weights=prev_weights,
                current_cash=max(0.0, prev_cash),
                total_equity=1.0,
                current_holding_days=prev_holding_days,
            ),
            policy_spec=active_policy_spec,
        )
        gross_ret = float(
            sum(
                float(weight) * _safe_float(
                    trajectory.prepared.stock_frames[symbol][trajectory.prepared.stock_frames[symbol]["date"] == step.date].iloc[0]["fwd_ret_1"],
                    0.0,
                )
                for symbol, weight in decision.symbol_target_weights.items()
                if symbol in trajectory.prepared.stock_frames
                and not trajectory.prepared.stock_frames[symbol][trajectory.prepared.stock_frames[symbol]["date"] == step.date].empty
            )
        )
        daily_ret, turnover, cost, fill_ratio, slip_bps, next_weights, next_cash = _simulate_execution_day(
            date=step.date,
            next_date=step.next_date,
            decision=decision,
            current_weights=prev_weights,
            current_cash=prev_cash,
            stock_states=step.stock_states,
            stock_frames=trajectory.prepared.stock_frames,
            total_commission_rate=commission_rate,
            base_slippage_rate=slippage_rate,
        )
        benchmark_row = trajectory.prepared.market_valid[trajectory.prepared.market_valid["date"] == step.date]
        benchmark_ret = 0.0
        if not benchmark_row.empty:
            benchmark_ret = _safe_float(benchmark_row.iloc[0].get("mkt_fwd_ret_1", 0.0), 0.0)
        returns.append(float(daily_ret))
        benchmark_returns.append(float(benchmark_ret))
        gross_returns.append(float(gross_ret))
        turnovers.append(turnover)
        costs.append(cost)
        fill_ratios.append(fill_ratio)
        slippage_cost_bps.append(slip_bps)
        out_dates.append(step.next_date)
        old_weights = dict(prev_weights)
        prev_weights = next_weights
        prev_holding_days = _advance_holding_days(
            prev_holding_days=prev_holding_days,
            prev_weights=old_weights,
            next_weights=next_weights,
        )
        prev_cash = next_cash

        if capture_learning_rows:
            target_exposure, target_positions, target_turnover, sample_weight = _derive_learning_targets(
                state=step.composite_state,
                stock_frames=trajectory.prepared.stock_frames,
                date=step.date,
                horizon_metrics=step.horizon_metrics,
                universe_tier=trajectory.prepared.settings.get("universe_tier"),
            )
            row = {
                name: float(value)
                for name, value in zip(_policy_feature_names(), _policy_feature_vector(step.composite_state))
            }
            row.update(
                {
                    "target_exposure": float(target_exposure),
                    "target_positions": float(target_positions),
                    "target_turnover": float(target_turnover),
                    "sample_weight": float(sample_weight),
                }
            )
            learning_rows.append(row)

    return (
        _to_v2_backtest_summary(
            returns=returns,
            benchmark_returns=benchmark_returns,
            turnovers=turnovers,
            costs=costs,
            gross_returns=gross_returns,
            fill_ratios=fill_ratios,
            slippage_bps=slippage_cost_bps,
            rank_ics=rank_ics,
            top_decile_returns=top_decile_returns,
            top_bottom_spreads=top_bottom_spreads,
            top_k_hit_rates=top_k_hit_rates,
            horizon_metrics=horizon_metric_series,
            dates=out_dates,
        ),
        learning_rows,
    )


def _run_v2_backtest_core(
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
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    capture_learning_rows: bool = False,
    trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
) -> tuple[V2BacktestSummary, list[dict[str, float]]]:
    _ = strategy_id
    if trajectory is None:
        trajectory = _load_or_build_v2_backtest_trajectory(
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
            retrain_days=retrain_days,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        if trajectory is None:
            return _empty_v2_backtest_result()
    return _execute_v2_backtest_trajectory(
        trajectory,
        policy_spec=policy_spec,
        learned_policy=learned_policy,
        retrain_days=retrain_days,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        capture_learning_rows=capture_learning_rows,
    )


def run_v2_backtest_live(
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
    policy_spec: PolicySpec | None = None,
    learned_policy: LearnedPolicyModel | None = None,
    retrain_days: int = 20,
    commission_bps: float = 1.5,
    slippage_bps: float = 2.0,
    trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
) -> V2BacktestSummary:
    summary, _ = _run_v2_backtest_core(
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
        policy_spec=policy_spec,
        learned_policy=learned_policy,
        retrain_days=retrain_days,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        capture_learning_rows=False,
        trajectory=trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    return summary


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
    trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
) -> V2CalibrationResult:
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

    baseline_spec = PolicySpec()
    baseline = baseline or run_v2_backtest_live(
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
    best_score = _policy_objective_score(baseline)
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
    _emit_progress("calibration", f"开始参数搜索: candidates={total_candidates}")
    for idx, spec in enumerate(candidate_specs, start=1):
        _emit_progress(
            "calibration",
            f"评估候选 {idx}/{total_candidates}: exposure={spec.risk_on_exposure:.2f}, positions={spec.risk_on_positions}, turnover={spec.risk_on_turnover_cap:.2f}",
        )
        summary = run_v2_backtest_live(
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
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        score = _policy_objective_score(summary)
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
            _emit_progress("calibration", f"发现更优参数: score={best_score:.4f}")
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
    trajectory: _BacktestTrajectory | None = None,
    fit_trajectory: _BacktestTrajectory | None = None,
    evaluation_trajectory: _BacktestTrajectory | None = None,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
) -> V2PolicyLearningResult:
    fit_trajectory = fit_trajectory or trajectory
    evaluation_trajectory = evaluation_trajectory or trajectory
    baseline = baseline or run_v2_backtest_live(
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
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    _, rows = _run_v2_backtest_core(
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
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    feature_names = _policy_feature_names()
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
        learned_summary = run_v2_backtest_live(
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

    exp_b, exp_w = _fit_ridge_regression(X, y_exposure, l2=l2, sample_weight=sample_weight)
    pos_b, pos_w = _fit_ridge_regression(X, y_positions, l2=l2, sample_weight=sample_weight)
    turn_b, turn_w = _fit_ridge_regression(X, y_turnover, l2=l2, sample_weight=sample_weight)

    pred_exp = np.asarray([_predict_ridge(row, exp_b, exp_w) for row in X], dtype=float)
    pred_pos = np.asarray([_predict_ridge(row, pos_b, pos_w) for row in X], dtype=float)
    pred_turn = np.asarray([_predict_ridge(row, turn_b, turn_w) for row in X], dtype=float)

    model = LearnedPolicyModel(
        feature_names=feature_names,
        exposure_intercept=float(exp_b),
        exposure_coef=[float(x) for x in exp_w.tolist()],
        position_intercept=float(pos_b),
        position_coef=[float(x) for x in pos_w.tolist()],
        turnover_intercept=float(turn_b),
        turnover_coef=[float(x) for x in turn_w.tolist()],
        train_rows=int(len(rows)),
        train_r2_exposure=float(_r2_score(y_exposure, pred_exp)),
        train_r2_positions=float(_r2_score(y_positions, pred_pos)),
        train_r2_turnover=float(_r2_score(y_turnover, pred_turn)),
    )
    learned_summary = run_v2_backtest_live(
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


def _baseline_only_calibration(baseline: V2BacktestSummary) -> V2CalibrationResult:
    baseline_spec = PolicySpec()
    score = _policy_objective_score(baseline)
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


def _placeholder_learning_result(baseline: V2BacktestSummary) -> V2PolicyLearningResult:
    model = LearnedPolicyModel(
        feature_names=_policy_feature_names(),
        exposure_intercept=0.60,
        exposure_coef=[0.0] * len(_policy_feature_names()),
        position_intercept=3.0,
        position_coef=[0.0] * len(_policy_feature_names()),
        turnover_intercept=0.22,
        turnover_coef=[0.0] * len(_policy_feature_names()),
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


def _run_v2_research_workflow_impl(
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
    info_file: str | None = None,
    info_lookback_days: int | None = None,
    info_half_life_days: float | None = None,
    use_info_fusion: bool | None = None,
    info_shadow_only: bool | None = None,
    info_types: str | None = None,
    info_source_mode: str | None = None,
    info_subsets: str | None = None,
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    skip_calibration: bool = False,
    skip_learning: bool = False,
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    split_mode: str = _DEFAULT_SPLIT_MODE,
    embargo_days: int = _DEFAULT_EMBARGO_DAYS,
) -> tuple[V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult]:
    _ = (external_signals, event_file, capital_flow_file, macro_file)
    _emit_progress("research", f"载入研究轨迹: backend={forecast_backend}")
    trajectory = _load_or_build_v2_backtest_trajectory(
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
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    if trajectory is None:
        empty_summary = run_v2_backtest_live(
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
            trajectory=None,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        return empty_summary, _baseline_only_calibration(empty_summary), _placeholder_learning_result(empty_summary)
    _, validation_trajectory, holdout_trajectory = _split_research_trajectory(
        trajectory,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )
    _emit_progress(
        "research",
        f"样本切分完成(mode={split_mode}, embargo={embargo_days}d): validation={_trajectory_step_count(validation_trajectory)}, holdout={_trajectory_step_count(holdout_trajectory)}",
    )
    _emit_progress("research", "开始回放 holdout 基线")
    baseline = run_v2_backtest_live(
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
        trajectory=holdout_trajectory,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        forecast_backend=forecast_backend,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    validation_baseline = None
    if not skip_calibration:
        _emit_progress("research", "开始回放 validation 基线")
        validation_baseline = run_v2_backtest_live(
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
            trajectory=validation_trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
    calibration = (
        _baseline_only_calibration(baseline)
        if skip_calibration
        else calibrate_v2_policy(
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
            baseline=validation_baseline if validation_baseline is not None else baseline,
            trajectory=validation_trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
    )
    if not skip_calibration:
        _emit_progress("research", "参数搜索完成，开始 holdout 复核")
        holdout_calibrated = run_v2_backtest_live(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_file=universe_file,
            universe_limit=universe_limit,
            universe_tier=universe_tier,
            policy_spec=calibration.best_policy,
            trajectory=holdout_trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
        calibration = V2CalibrationResult(
            best_policy=calibration.best_policy,
            best_score=calibration.best_score,
            baseline=baseline,
            calibrated=holdout_calibrated,
            trials=calibration.trials,
        )
    else:
        _emit_progress("research", "已跳过参数搜索")
    learning = (
        _placeholder_learning_result(baseline)
        if skip_learning
        else learn_v2_policy_model(
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
            baseline=baseline,
            trajectory=holdout_trajectory,
            fit_trajectory=validation_trajectory,
            evaluation_trajectory=holdout_trajectory,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
        )
    )
    if skip_learning:
        _emit_progress("research", "已跳过学习型策略")
    else:
        _emit_progress("research", "学习型策略评估完成")
    return baseline, calibration, learning


def _run_v2_research_matrix_impl(
    *,
    strategy_id: str = "swing_v2",
    config_path: str = "config/api.json",
    source: str | None = None,
    artifact_root: str = "artifacts/v2",
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    forecast_backend: str = "linear",
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    split_mode: str = _DEFAULT_SPLIT_MODE,
    embargo_days: int = _DEFAULT_EMBARGO_DAYS,
    universe_tiers: Iterable[str] = ("favorites_16", "generated_80", "generated_150", "generated_300"),
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    normalized_tiers = [normalize_universe_tier(item) for item in universe_tiers]
    for tier_id in normalized_tiers:
        _emit_progress("matrix", f"开始研究矩阵档位: {tier_id}")
        baseline, calibration, learning = _run_v2_research_workflow_impl(
            strategy_id=strategy_id,
            config_path=config_path,
            source=source,
            universe_tier=tier_id,
            cache_root=cache_root,
            refresh_cache=refresh_cache,
            forecast_backend=forecast_backend,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
            split_mode=split_mode,
            embargo_days=embargo_days,
        )
        artifacts = _publish_v2_research_artifacts_impl(
            strategy_id=strategy_id,
            artifact_root=artifact_root,
            config_path=config_path,
            source=source,
            universe_tier=tier_id,
            baseline=baseline,
            calibration=calibration,
            learning=learning,
            cache_root=cache_root,
            forecast_backend=forecast_backend,
            publish_forecast_models=True,
            use_us_index_context=use_us_index_context,
            us_index_source=us_index_source,
            split_mode=split_mode,
            embargo_days=embargo_days,
        )
        rows.append(
            {
                "universe_tier": tier_id,
                "run_id": artifacts.get("run_id", ""),
                "release_gate_passed": artifacts.get("release_gate_passed", "false"),
                "default_switch_gate_passed": artifacts.get("default_switch_gate_passed", "false"),
                "annual_return": float(learning.learned.annual_return),
                "excess_annual_return": float(learning.learned.excess_annual_return),
                "information_ratio": float(learning.learned.information_ratio),
                "max_drawdown": float(learning.learned.max_drawdown),
                "avg_turnover": float(learning.learned.avg_turnover),
                "total_cost": float(learning.learned.total_cost),
                "baseline_annual_return": float(baseline.annual_return),
                "baseline_excess_annual_return": float(baseline.excess_annual_return),
                "baseline_information_ratio": float(baseline.information_ratio),
                "baseline_max_drawdown": float(baseline.max_drawdown),
                "research_manifest": artifacts.get("research_manifest", ""),
            }
        )
    return {
        "strategy_id": str(strategy_id),
        "split_mode": str(split_mode),
        "embargo_days": int(embargo_days),
        "forecast_backend": str(forecast_backend),
        "rows": rows,
    }


def _load_published_v2_policy_model_impl(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
) -> LearnedPolicyModel | None:
    model_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    return _load_policy_model_from_path(model_path)


def _load_policy_model_from_path(model_path: Path) -> LearnedPolicyModel | None:
    if not model_path.exists():
        return None
    payload = _load_json_dict(model_path)
    if not isinstance(payload, dict):
        return None
    return LearnedPolicyArtifact.from_payload(payload).model


def _with_backtest_metadata(
    summary: V2BacktestSummary,
    *,
    run_id: str,
    snapshot_hash: str,
    config_hash: str,
) -> V2BacktestSummary:
    payload = asdict(summary)
    payload["run_id"] = str(run_id)
    payload["snapshot_hash"] = str(snapshot_hash)
    payload["config_hash"] = str(config_hash)
    return V2BacktestSummary(**payload)


def _decode_composite_state(payload: object) -> CompositeState | None:
    return _decode_composite_state_external(payload)


def _serialize_composite_state(state: CompositeState) -> dict[str, object]:
    return _serialize_composite_state_external(state)


def _build_frozen_daily_state_payload(
    *,
    trajectory: _BacktestTrajectory | None,
    split_mode: str,
    embargo_days: int,
) -> dict[str, object]:
    return _build_frozen_daily_state_payload_external(
        trajectory=trajectory,
        split_mode=split_mode,
        embargo_days=embargo_days,
        split_trajectory=_split_research_trajectory,
    )


def _pass_release_gate(
    *,
    baseline: V2BacktestSummary,
    candidate: V2BacktestSummary,
) -> tuple[bool, list[str]]:
    return _pass_release_gate_external(
        baseline=baseline,
        candidate=candidate,
        threshold=_RELEASE_GATE_THRESHOLD,
    )


def _tier_latest_manifest_path(*, artifact_root: str, strategy_id: str, universe_tier: str) -> Path:
    return _tier_latest_manifest_path_external(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        universe_tier=universe_tier,
    )


def _tier_latest_policy_path(*, artifact_root: str, strategy_id: str, universe_tier: str) -> Path:
    return _tier_latest_policy_path_external(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        universe_tier=universe_tier,
    )


def _summary_from_payload(template: V2BacktestSummary, payload: dict[str, object]) -> V2BacktestSummary:
    return _summary_from_payload_external(template, payload)


def _load_backtest_payload_from_manifest(manifest_payload: dict[str, object], manifest_path: Path) -> dict[str, object]:
    return _load_backtest_payload_from_manifest_external(
        manifest_payload,
        manifest_path,
        path_from_manifest_entry=lambda entry: _path_from_manifest_entry(entry, run_dir=manifest_path.parent),
        load_json_dict=_load_json_dict,
    )


def _load_backtest_payload_for_run(
    *,
    artifact_root: str,
    strategy_id: str,
    run_id: str,
) -> dict[str, object]:
    return _load_backtest_payload_for_run_external(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        run_id=run_id,
        load_json_dict=_load_json_dict,
    )


def _pass_default_switch_gate(
    *,
    baseline_reference: V2BacktestSummary,
    candidate: V2BacktestSummary,
) -> tuple[bool, list[str], dict[str, float]]:
    return _pass_default_switch_gate_external(
        baseline_reference=baseline_reference,
        candidate=candidate,
        threshold=_DEFAULT_SWITCH_GATE_THRESHOLD,
    )


def _load_research_manifest_for_daily(
    *,
    strategy_id: str,
    artifact_root: str,
    run_id: str | None,
    snapshot_path: str | None,
) -> tuple[dict[str, object], Path]:
    return _load_research_manifest_for_daily_external(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        run_id=run_id,
        snapshot_path=snapshot_path,
        resolve_manifest_path=_resolve_manifest_path,
        load_json_dict=_load_json_dict,
    )


def _build_snapshot_from_manifest(
    *,
    strategy_id: str,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
) -> StrategySnapshot:
    dataset_path = _path_from_manifest_entry(manifest.get("dataset_manifest"), run_dir=manifest_path.parent)
    dataset_manifest = _load_json_dict(dataset_path) if dataset_path is not None else {}
    config_hash = str(manifest.get("config_hash", "")) or _stable_json_hash(settings)
    model_hashes_raw = manifest.get("model_hashes", {})
    model_hashes = (
        {str(k): str(v) for k, v in model_hashes_raw.items()}
        if isinstance(model_hashes_raw, dict)
        else {}
    )
    policy_hash = str(manifest.get("policy_hash", ""))
    universe_hash = str(manifest.get("universe_hash", ""))
    run_id = str(manifest.get("run_id", ""))
    snapshot_hash = str(manifest.get("snapshot_hash", "")) or _compose_run_snapshot_hash(
        run_id=run_id,
        strategy_id=strategy_id,
        config_hash=config_hash,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        model_hashes=model_hashes,
    )
    universe_file = str(dataset_manifest.get("universe_file", settings.get("universe_file", "")))
    universe_id = str(dataset_manifest.get("universe_id", "")).strip() or Path(universe_file).stem or Path(str(settings.get("universe_file", ""))).stem or "v2_universe"
    start = str(dataset_manifest.get("start", settings.get("start", "")))
    end = str(dataset_manifest.get("end", settings.get("end", "")))
    data_window = f"{start}~{end}" if start or end else ""
    return build_strategy_snapshot(
        strategy_id=strategy_id,
        universe_id=universe_id,
        universe_size=int(dataset_manifest.get("universe_size", dataset_manifest.get("symbol_count", 0)) or 0),
        universe_generation_rule=str(dataset_manifest.get("universe_generation_rule", "")),
        source_universe_manifest_path=str(
            dataset_manifest.get("source_universe_manifest_path", dataset_manifest.get("universe_file", ""))
        ),
        info_manifest_path=str(manifest.get("info_manifest", "")),
        info_hash=str(manifest.get("info_hash", dataset_manifest.get("info_hash", ""))),
        info_shadow_enabled=_parse_boolish(manifest.get("info_shadow_enabled", dataset_manifest.get("info_shadow_enabled", False)), False),
        external_signal_manifest_path=str(manifest.get("external_signal_manifest", dataset_manifest.get("external_signal_manifest", ""))),
        external_signal_version=str(manifest.get("external_signal_version", dataset_manifest.get("external_signal_version", "v1"))),
        external_signal_enabled=_parse_boolish(
            manifest.get("external_signal_enabled", dataset_manifest.get("external_signal_enabled", False)),
            False,
        ),
        capital_flow_snapshot=dict(manifest.get("capital_flow_snapshot", dataset_manifest.get("capital_flow_snapshot", {}))),
        macro_context_snapshot=dict(manifest.get("macro_context_snapshot", dataset_manifest.get("macro_context_snapshot", {}))),
        generator_manifest_path=str(manifest.get("generator_manifest", dataset_manifest.get("generator_manifest", ""))),
        generator_version=str(manifest.get("generator_version", dataset_manifest.get("generator_version", ""))),
        generator_hash=str(manifest.get("generator_hash", dataset_manifest.get("generator_hash", ""))),
        coarse_pool_size=int(manifest.get("coarse_pool_size", dataset_manifest.get("coarse_pool_size", 0)) or 0),
        refined_pool_size=int(manifest.get("refined_pool_size", dataset_manifest.get("refined_pool_size", 0)) or 0),
        selected_pool_size=int(manifest.get("selected_pool_size", dataset_manifest.get("selected_pool_size", 0)) or 0),
        theme_allocations=[
            dict(item)
            for item in manifest.get("theme_allocations", dataset_manifest.get("theme_allocations", []))
            if isinstance(item, dict)
        ],
        run_id=run_id,
        data_window=data_window,
        model_hashes=model_hashes,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        created_at=str(manifest.get("created_at", "")),
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
        manifest_path=str(manifest_path.resolve()),
        use_us_index_context=_parse_boolish(
            dataset_manifest.get("use_us_index_context", manifest.get("use_us_index_context", False)),
            False,
        ),
        us_index_source=str(dataset_manifest.get("us_index_source", manifest.get("us_index_source", ""))),
    )


def _publish_v2_research_artifacts_impl(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
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
    info_file: str | None = None,
    info_lookback_days: int | None = None,
    info_half_life_days: float | None = None,
    use_info_fusion: bool | None = None,
    info_shadow_only: bool | None = None,
    info_types: str | None = None,
    info_source_mode: str | None = None,
    info_subsets: str | None = None,
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    settings: dict[str, object] | None = None,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    cache_root: str = "artifacts/v2/cache",
    forecast_backend: str = "linear",
    publish_forecast_models: bool = True,
    split_mode: str = _DEFAULT_SPLIT_MODE,
    embargo_days: int = _DEFAULT_EMBARGO_DAYS,
    update_latest: bool = True,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
) -> dict[str, str]:
    settings = settings or _load_v2_runtime_settings(
        config_path=config_path,
        source=source,
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
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    settings = _resolve_v2_universe_settings(settings=dict(settings), cache_root=cache_root)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    created_at = datetime.now().isoformat(timespec="seconds")
    base_dir = Path(str(artifact_root)) / str(strategy_id) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    universe_path = Path(str(settings.get("universe_file", "")))
    symbols = [str(item) for item in settings.get("symbols", [])]
    universe_tier_value = str(settings.get("universe_tier", "")).strip()
    universe_id = str(settings.get("universe_id", "")).strip() or universe_tier_value or universe_path.stem or "v2_universe"
    universe_size = int(settings.get("universe_size", len(symbols)) or len(symbols))
    universe_generation_rule = str(settings.get("universe_generation_rule", "")).strip() or "external_universe_file"
    source_universe_manifest_path = str(
        settings.get("source_universe_manifest_path", settings.get("universe_file", ""))
    )
    active_default_universe_tier = str(settings.get("active_default_universe_tier", "favorites_16")).strip()
    candidate_default_universe_tier = str(settings.get("candidate_default_universe_tier", "generated_80")).strip()
    baseline_reference_run_id = str(settings.get("baseline_reference_run_id", "")).strip()

    config_hash = _stable_json_hash(settings)
    learning_manifest = add_artifact_metadata(
        asdict(learning.model),
        artifact_type="learned_policy_model",
    )
    policy_hash = _stable_json_hash(learning_manifest)
    universe_hash = str(settings.get("universe_hash", "")) or _sha256_file(universe_path) or _stable_json_hash(symbols)
    model_hashes = {
        "market_model": _sha256_text("mkt_lr_v2"),
        "sector_model": _sha256_text("sector_lr_v2"),
        "stock_model": _sha256_text("stock_lr_v2"),
        "cross_section_model": _sha256_text("cross_section_v2"),
        "learned_policy_model": policy_hash,
    }
    snapshot_hash = _compose_run_snapshot_hash(
        run_id=run_id,
        strategy_id=strategy_id,
        config_hash=config_hash,
        policy_hash=policy_hash,
        universe_hash=universe_hash,
        model_hashes=model_hashes,
    )
    baseline_meta = _with_backtest_metadata(
        baseline,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
    )
    calibrated_meta = _with_backtest_metadata(
        calibration.calibrated,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
    )
    learned_meta = _with_backtest_metadata(
        learning.learned,
        run_id=run_id,
        snapshot_hash=snapshot_hash,
        config_hash=config_hash,
    )

    def _window_payload(trajectory: _BacktestTrajectory | None) -> dict[str, object]:
        if trajectory is None or not trajectory.steps:
            return {"start": "", "end": "", "n_steps": 0}
        return {
            "start": str(trajectory.steps[0].date.date()),
            "end": str(trajectory.steps[-1].next_date.date()),
            "n_steps": int(len(trajectory.steps)),
        }

    trajectory = None
    frozen_daily_state: dict[str, object] = {}
    forecast_models_manifest: dict[str, object] = {}
    frozen_forecast_bundle: dict[str, object] = {}
    train_window = {"start": "", "end": "", "n_steps": 0}
    validation_window = {"start": "", "end": "", "n_steps": 0}
    holdout_window = {"start": "", "end": "", "n_steps": 0}
    regime_counts: dict[str, int] = {}
    if publish_forecast_models:
        trajectory = _load_or_build_v2_backtest_trajectory(
            config_path=str(settings.get("config_path", config_path)),
            source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
            universe_file=str(settings.get("universe_file", universe_file))
            if settings.get("universe_file", universe_file) is not None
            else None,
            universe_limit=(
                int(settings.get("universe_limit"))
                if settings.get("universe_limit") is not None
                else (int(universe_limit) if universe_limit is not None else None)
            ),
            universe_tier=str(settings.get("universe_tier", universe_tier)),
            cache_root=cache_root,
            refresh_cache=False,
            forecast_backend=forecast_backend,
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
        )
        if trajectory is not None:
            train_traj, validation_traj, holdout_traj = _split_research_trajectory(
                trajectory,
                split_mode=split_mode,
                embargo_days=embargo_days,
            )
            train_window = _window_payload(train_traj)
            validation_window = _window_payload(validation_traj)
            holdout_window = _window_payload(holdout_traj)
            frozen_daily_state = _build_frozen_daily_state_payload(
                trajectory=trajectory,
                split_mode=split_mode,
                embargo_days=embargo_days,
            )
            for step in holdout_traj.steps:
                regime = str(step.composite_state.risk_regime or "unknown")
                regime_counts[regime] = int(regime_counts.get(regime, 0)) + 1
            if (
                str(forecast_backend).strip().lower() == "linear"
                and trajectory.prepared is not None
                and hasattr(trajectory.prepared, "market_valid")
                and hasattr(trajectory.prepared, "panel")
                and hasattr(trajectory.prepared, "market_feature_cols")
                and hasattr(trajectory.prepared, "feature_cols")
                and hasattr(trajectory.prepared, "dates")
            ):
                frozen_forecast_bundle = _build_frozen_linear_forecast_bundle(trajectory.prepared)
        forecast_models_manifest = add_artifact_metadata(
            {
            "run_id": run_id,
            "strategy_id": str(strategy_id),
            "forecast_backend": str(forecast_backend),
            "split_mode": str(split_mode),
            "embargo_days": int(embargo_days),
            "use_us_index_context": bool(settings.get("use_us_index_context", False)),
            "us_index_source": str(settings.get("us_index_source", "akshare")),
            "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
            "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
            "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
            "model_hashes": model_hashes,
            "data_window": {
                "start": str(settings.get("start", "")),
                "end": str(settings.get("end", "")),
            },
            "regime_counts": regime_counts,
            "frozen_bundle_ready": bool(frozen_forecast_bundle),
            },
            artifact_type="forecast_models_manifest",
        )

    dataset_path = base_dir / "dataset_manifest.json"
    calibration_path = base_dir / "policy_calibration.json"
    learning_path = base_dir / "learned_policy_model.json"
    forecast_models_path = base_dir / "forecast_models_manifest.json"
    frozen_forecast_bundle_path = base_dir / "frozen_forecast_bundle.json"
    frozen_state_path = base_dir / "frozen_daily_state.json"
    backtest_path = base_dir / "backtest_summary.json"
    consistency_path = base_dir / "consistency_checklist.json"
    rolling_oos_path = base_dir / "rolling_oos_report.json"
    info_manifest_path = base_dir / "info_manifest.json"
    info_shadow_report_path = base_dir / "info_shadow_report.json"
    external_signal_manifest_path = ensure_external_signal_manifest_path(base_dir)
    manifest_path = base_dir / "research_manifest.json"
    latest_policy_path = Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    latest_manifest_path = Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json"
    tier_latest_policy_path = _tier_latest_policy_path(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        universe_tier=universe_tier_value,
    )
    tier_latest_manifest_path = _tier_latest_manifest_path(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        universe_tier=universe_tier_value,
    )

    info_shadow_enabled = False
    info_shadow_report: dict[str, object] = {
        "info_shadow_enabled": False,
        "shadow_only": bool(settings.get("info_shadow_only", True)),
        "quant_only": {},
        "quant_plus_info_shadow": {},
        "market_news_only": {},
        "announcements_only": {},
        "research_only": {},
        "all_info_combined": {},
        "coverage_summary": {},
        "top_positive_stock_deltas": [],
        "top_negative_stock_deltas": [],
        "event_tag_distribution": {},
        "info_source_breakdown": {},
        "last_market_info_state": {},
        "last_date": "",
        "market_shadow_modes": {},
        "stock_shadow_modes": {},
        "model_samples": {"market": {}, "stock": {}},
    }
    info_file_path = _resolve_info_file_from_settings(settings)
    info_as_of_date = pd.Timestamp(learned_meta.end_date or baseline_meta.end_date or settings.get("end", "today")).normalize()
    info_items: list[InfoItem] = []
    if bool(settings.get("use_info_fusion", False)):
        if trajectory is None:
            trajectory = _load_or_build_v2_backtest_trajectory(
                config_path=str(settings.get("config_path", config_path)),
                source=str(settings.get("source", source)) if settings.get("source", source) is not None else None,
                universe_file=str(settings.get("universe_file", universe_file))
                if settings.get("universe_file", universe_file) is not None
                else None,
                universe_limit=(
                    int(settings.get("universe_limit"))
                    if settings.get("universe_limit") is not None
                    else (int(universe_limit) if universe_limit is not None else None)
                ),
                universe_tier=str(settings.get("universe_tier", universe_tier)),
                dynamic_universe=_parse_boolish(settings.get("dynamic_universe_enabled", False), False),
                generator_target_size=int(settings.get("generator_target_size", settings.get("universe_limit", 0)) or 0),
                generator_coarse_size=int(settings.get("generator_coarse_size", 0) or 0),
                generator_theme_aware=_parse_boolish(settings.get("generator_theme_aware", True), True),
                generator_use_concepts=_parse_boolish(settings.get("generator_use_concepts", True), True),
                cache_root=cache_root,
                refresh_cache=False,
                forecast_backend=forecast_backend,
                use_us_index_context=bool(settings.get("use_us_index_context", False)),
                us_index_source=str(settings.get("us_index_source", "akshare")),
            )
        info_file_path, info_items = _load_v2_info_items_for_date(
            settings=settings,
            as_of_date=info_as_of_date,
            learned_window=True,
        )
        if trajectory is not None and info_items:
            _, validation_traj, holdout_traj = _split_research_trajectory(
                trajectory,
                split_mode=split_mode,
                embargo_days=embargo_days,
            )
            info_shadow_report = _build_info_shadow_report(
                validation_trajectory=validation_traj,
                holdout_trajectory=holdout_traj,
                settings=settings,
                info_items=info_items,
            )
            info_shadow_enabled = True
    info_manifest = _build_info_manifest_payload(
        settings=settings,
        info_file=info_file_path,
        info_items=info_items,
        as_of_date=info_as_of_date,
        config_hash=config_hash,
        shadow_enabled=info_shadow_enabled,
        shadow_report=info_shadow_report,
    )
    info_hash = str(info_manifest.get("info_hash", ""))
    external_signal_package = _build_external_signal_package_for_date(
        settings=settings,
        as_of_date=info_as_of_date,
        info_items=info_items,
    )
    external_signal_manifest = dict(external_signal_package.get("manifest", {}))
    info_manifest = merge_external_signal_manifest_summary(
        info_manifest=info_manifest,
        external_signal_manifest=external_signal_manifest,
    )

    if publish_forecast_models and frozen_daily_state:
        frozen_composite = _decode_composite_state(frozen_daily_state.get("composite_state"))
        if frozen_composite is not None:
            if bool(settings.get("use_info_fusion", False)) and info_items:
                frozen_composite = _enrich_state_with_info(
                    state=frozen_composite,
                    as_of_date=info_as_of_date,
                    info_items=info_items,
                    settings=settings,
                )
            frozen_composite, _ = _attach_external_signals_to_composite_state(
                state=frozen_composite,
                settings=settings,
                as_of_date=info_as_of_date,
                info_items=info_items,
            )
            frozen_daily_state["composite_state"] = _serialize_composite_state(frozen_composite)
        frozen_daily_state = add_artifact_metadata(
            frozen_daily_state,
            artifact_type="frozen_daily_state",
        )
    if publish_forecast_models and frozen_forecast_bundle:
        frozen_forecast_bundle = add_artifact_metadata(
            frozen_forecast_bundle,
            artifact_type="forecast_bundle",
        )

    dataset_manifest = add_artifact_metadata(
        {
        "strategy_id": str(strategy_id),
        "config_path": str(settings.get("config_path", "")),
        "source": str(settings.get("source", "")),
        "watchlist": str(settings.get("watchlist", "")),
        "universe_tier": universe_tier_value,
        "universe_id": universe_id,
        "universe_size": int(universe_size),
        "universe_generation_rule": universe_generation_rule,
        "source_universe_manifest_path": source_universe_manifest_path,
        "universe_file": str(settings.get("universe_file", "")),
        "universe_limit": int(settings.get("universe_limit", 0)),
        "dynamic_universe_enabled": bool(settings.get("dynamic_universe_enabled", False)),
        "generator_manifest": str(settings.get("generator_manifest_path", "")),
        "generator_version": str(settings.get("generator_version", "")),
        "generator_hash": str(settings.get("generator_hash", "")),
        "coarse_pool_size": int(settings.get("coarse_pool_size", 0)),
        "refined_pool_size": int(settings.get("refined_pool_size", 0)),
        "selected_pool_size": int(settings.get("selected_pool_size", 0)),
        "theme_allocations": [dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
        "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
        "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
        "start": str(settings.get("start", "")),
        "end": str(settings.get("end", "")),
        "symbols": symbols,
        "symbol_count": int(settings.get("symbol_count", len(symbols))),
        "universe_hash": universe_hash,
        "config_hash": config_hash,
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "info_file": str(info_file_path),
        "event_file": str(settings.get("event_file", info_file_path)),
        "info_hash": info_hash,
        "info_shadow_enabled": bool(info_shadow_enabled),
        "info_shadow_only": bool(settings.get("info_shadow_only", True)),
        "info_item_count": int(info_manifest.get("info_item_count", 0)),
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
        "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
        "capital_flow_file": str(settings.get("capital_flow_file", "")),
        "macro_file": str(settings.get("macro_file", "")),
        "external_signal_manifest": str(external_signal_manifest_path),
        "external_signal_version": str(settings.get("external_signal_version", "v1")),
        "external_signal_enabled": bool(settings.get("external_signals", True)),
        "event_lookback_days": int(settings.get("event_lookback_days", settings.get("info_lookback_days", 45))),
        "capital_flow_lookback_days": int(settings.get("capital_flow_lookback_days", 20)),
        "macro_lookback_days": int(settings.get("macro_lookback_days", 60)),
        "event_risk_cutoff": float(settings.get("event_risk_cutoff", 0.55)),
        "catalyst_boost_cap": float(settings.get("catalyst_boost_cap", 0.12)),
        "flow_exposure_cap": float(settings.get("flow_exposure_cap", 0.08)),
        "capital_flow_snapshot": dict(external_signal_package.get("capital_flow_snapshot", {})),
        "macro_context_snapshot": dict(external_signal_package.get("macro_context_snapshot", {})),
        "active_default_universe_tier": active_default_universe_tier,
        "candidate_default_universe_tier": candidate_default_universe_tier,
        },
        artifact_type="dataset_manifest",
    )
    calibration_manifest = add_artifact_metadata(
        {
        "best_score": float(calibration.best_score),
        "best_policy": asdict(calibration.best_policy),
        "trials": calibration.trials,
        "policy_hash": _stable_json_hash(asdict(calibration.best_policy)),
        },
        artifact_type="policy_calibration",
    )
    backtest_manifest = add_artifact_metadata(
        {
        "baseline": asdict(baseline_meta),
        "calibrated": asdict(calibrated_meta),
        "learned": asdict(learned_meta),
        },
        artifact_type="backtest_summary",
    )
    consistency_manifest = add_artifact_metadata(
        {
        "run_id": run_id,
        "universe_tier": universe_tier_value,
        "universe_id": universe_id,
        "universe_size": int(universe_size),
        "split_mode": str(split_mode),
        "embargo_days": int(embargo_days),
        "train_window": train_window,
        "validation_window": validation_window,
        "holdout_window": holdout_window,
        "snapshot_hash": snapshot_hash,
        "config_hash": config_hash,
        "policy_hash": policy_hash,
        "universe_hash": universe_hash,
        "model_hashes": model_hashes,
        "info_hash": info_hash,
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "external_signal_enabled": bool(settings.get("external_signals", True)),
        "external_signal_version": str(settings.get("external_signal_version", "v1")),
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        },
        artifact_type="consistency_checklist",
    )
    rolling_oos_manifest = add_artifact_metadata(
        {
        "run_id": run_id,
        "universe_tier": universe_tier_value,
        "windows": [
            {
                "name": "window_1",
                "start": learned_meta.start_date,
                "end": learned_meta.end_date,
                "excess_annual_return": float(learned_meta.excess_annual_return),
                "information_ratio": float(learned_meta.information_ratio),
                "max_drawdown": float(learned_meta.max_drawdown),
            },
            {
                "name": "window_2",
                "start": calibrated_meta.start_date,
                "end": calibrated_meta.end_date,
                "excess_annual_return": float(calibrated_meta.excess_annual_return),
                "information_ratio": float(calibrated_meta.information_ratio),
                "max_drawdown": float(calibrated_meta.max_drawdown),
            },
        ],
        "regime_breakdown": regime_counts,
        },
        artifact_type="rolling_oos_report",
    )
    latest_policy_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_path.write_text(json.dumps(dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    learning_path.write_text(json.dumps(learning_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if publish_forecast_models:
        forecast_models_path.write_text(json.dumps(forecast_models_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        frozen_forecast_bundle_path.write_text(json.dumps(frozen_forecast_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        frozen_state_path.write_text(json.dumps(frozen_daily_state, ensure_ascii=False, indent=2), encoding="utf-8")
    backtest_path.write_text(json.dumps(backtest_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    consistency_path.write_text(json.dumps(consistency_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    rolling_oos_path.write_text(json.dumps(rolling_oos_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    info_manifest_path.write_text(json.dumps(info_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    info_shadow_report_path.write_text(json.dumps(info_shadow_report, ensure_ascii=False, indent=2), encoding="utf-8")
    external_signal_manifest_path.write_text(
        json.dumps(external_signal_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    gate_ok, gate_reasons = _pass_release_gate(
        baseline=baseline_meta,
        candidate=learned_meta,
    )
    previous_manifest = _load_json_dict(tier_latest_manifest_path)
    previous_gate_ok = False
    previous_reason = "missing previous same-tier latest manifest"
    if previous_manifest:
        previous_backtest = _load_backtest_payload_from_manifest(previous_manifest, tier_latest_manifest_path)
        prev_baseline_payload = previous_backtest.get("baseline", {}) if isinstance(previous_backtest, dict) else {}
        prev_learned_payload = previous_backtest.get("learned", {}) if isinstance(previous_backtest, dict) else {}
        if isinstance(prev_baseline_payload, dict) and isinstance(prev_learned_payload, dict):
            prev_baseline = _summary_from_payload(baseline_meta, prev_baseline_payload)
            prev_learned = _summary_from_payload(learned_meta, prev_learned_payload)
            previous_gate_ok, previous_reasons = _pass_release_gate(
                baseline=prev_baseline,
                candidate=prev_learned,
            )
            previous_reason = "" if previous_gate_ok else "; ".join(previous_reasons)
    release_gate_passed = bool(gate_ok and previous_gate_ok)
    release_gate = {
        "current_passed": bool(gate_ok),
        "current_reasons": gate_reasons,
        "previous_passed": bool(previous_gate_ok),
        "previous_reason": previous_reason,
        "require_two_consecutive": True,
        "passed": bool(release_gate_passed),
    }
    baseline_reference_payload = _load_backtest_payload_for_run(
        artifact_root=artifact_root,
        strategy_id=strategy_id,
        run_id=baseline_reference_run_id,
    )
    baseline_reference_learned_payload = (
        baseline_reference_payload.get("learned", {})
        if isinstance(baseline_reference_payload, dict)
        else {}
    )
    baseline_reference_summary = (
        _summary_from_payload(learned_meta, baseline_reference_learned_payload)
        if isinstance(baseline_reference_learned_payload, dict) and baseline_reference_learned_payload
        else learned_meta
    )
    switch_current_ok = False
    switch_current_reasons = ["switch gate skipped: not candidate default universe tier"]
    switch_previous_ok = False
    switch_previous_reason = "missing previous same-tier switch gate"
    switch_deltas = {
        "excess_annual_return_delta": 0.0,
        "information_ratio_delta": 0.0,
        "max_drawdown_diff": 0.0,
    }
    if universe_tier_value == candidate_default_universe_tier and baseline_reference_run_id:
        switch_current_ok, switch_current_reasons, switch_deltas = _pass_default_switch_gate(
            baseline_reference=baseline_reference_summary,
            candidate=learned_meta,
        )
        if previous_manifest:
            previous_switch_gate = previous_manifest.get("default_switch_gate", {})
            if isinstance(previous_switch_gate, dict):
                switch_previous_ok = bool(previous_switch_gate.get("current_passed", False))
                switch_previous_reason = "" if switch_previous_ok else str(previous_switch_gate.get("current_reasons", ""))
    default_switch_gate_passed = bool(release_gate_passed and switch_current_ok and switch_previous_ok)
    default_switch_gate = {
        "baseline_reference_run_id": baseline_reference_run_id,
        "current_passed": bool(switch_current_ok),
        "current_reasons": switch_current_reasons,
        "previous_passed": bool(switch_previous_ok),
        "previous_reason": switch_previous_reason,
        "require_two_consecutive": True,
        "deltas": switch_deltas,
        "passed": bool(default_switch_gate_passed),
    }

    manifest = add_artifact_metadata(
        {
        "run_id": run_id,
        "strategy_id": str(strategy_id),
        "created_at": created_at,
        "baseline_reference_run_id": baseline_reference_run_id,
        "universe_tier": universe_tier_value,
        "universe_id": universe_id,
        "universe_size": int(universe_size),
        "universe_generation_rule": universe_generation_rule,
        "source_universe_manifest_path": source_universe_manifest_path,
        "info_hash": info_hash,
        "info_shadow_enabled": bool(info_shadow_enabled),
        "info_source_mode": str(settings.get("info_source_mode", "layered")),
        "info_subsets": [str(item) for item in settings.get("info_subsets", [])],
        "announcement_event_tags": [str(item) for item in settings.get("announcement_event_tags", [])],
        "external_signal_version": str(settings.get("external_signal_version", "v1")),
        "external_signal_enabled": bool(settings.get("external_signals", True)),
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "use_us_sector_etf_context": bool(settings.get("use_us_sector_etf_context", False)),
        "use_cn_etf_context": bool(settings.get("use_cn_etf_context", False)),
        "cn_etf_source": str(settings.get("cn_etf_source", "akshare")),
        "dynamic_universe_enabled": bool(settings.get("dynamic_universe_enabled", False)),
        "generator_manifest": str(settings.get("generator_manifest_path", "")),
        "generator_version": str(settings.get("generator_version", "")),
        "generator_hash": str(settings.get("generator_hash", "")),
        "coarse_pool_size": int(settings.get("coarse_pool_size", 0)),
        "refined_pool_size": int(settings.get("refined_pool_size", 0)),
        "selected_pool_size": int(settings.get("selected_pool_size", 0)),
        "theme_allocations": [dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
        "capital_flow_snapshot": dict(external_signal_package.get("capital_flow_snapshot", {})),
        "macro_context_snapshot": dict(external_signal_package.get("macro_context_snapshot", {})),
        "use_us_index_context": bool(settings.get("use_us_index_context", False)),
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "data_window": {
            "start": str(settings.get("start", "")),
            "end": str(settings.get("end", "")),
        },
        "config_hash": config_hash,
        "snapshot_hash": snapshot_hash,
        "policy_hash": policy_hash,
        "universe_hash": universe_hash,
        "model_hashes": model_hashes,
        "split_mode": str(split_mode),
        "embargo_days": int(embargo_days),
        "dataset_manifest": str(dataset_path),
        "policy_calibration": str(calibration_path),
        "learned_policy_model": str(learning_path),
        "forecast_models_manifest": str(forecast_models_path) if publish_forecast_models else "",
        "frozen_forecast_bundle": str(frozen_forecast_bundle_path) if publish_forecast_models else "",
        "frozen_daily_state": str(frozen_state_path) if publish_forecast_models else "",
        "backtest_summary": str(backtest_path),
        "consistency_checklist": str(consistency_path),
        "rolling_oos_report": str(rolling_oos_path),
        "info_manifest": str(info_manifest_path),
        "info_shadow_report": str(info_shadow_report_path),
        "external_signal_manifest": str(external_signal_manifest_path),
        "published_policy_model": str(latest_policy_path),
        "latest_research_manifest": str(latest_manifest_path),
        "tier_published_policy_model": str(tier_latest_policy_path),
        "tier_latest_research_manifest": str(tier_latest_manifest_path),
        "release_gate": release_gate,
        "default_switch_gate": default_switch_gate,
        },
        artifact_type="research_manifest",
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tier_latest_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if gate_ok:
        tier_latest_policy_path.write_text(json.dumps(learning_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    allow_default_latest_update = bool(
        not universe_tier_value or universe_tier_value == active_default_universe_tier
    )
    if update_latest and release_gate_passed and allow_default_latest_update:
        latest_policy_path.write_text(json.dumps(learning_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    elif update_latest and release_gate_passed and not allow_default_latest_update:
        note = "当前为非默认股票池，本次只更新同 tier latest，不覆盖默认 latest"
        if default_switch_gate_passed:
            note += "；default_switch_gate 已通过，切换 `active_default_universe_tier` 后可升级默认"
        _emit_progress("publish", note)
    elif not release_gate_passed:
        _emit_progress("publish", "门禁未通过，本次不更新 latest policy/manifest")

    memory_path = remember_research_run(
        memory_root=Path(str(artifact_root)) / "memory",
        strategy_id=strategy_id,
        run_id=run_id,
        baseline=baseline_meta,
        calibration=calibration,
        learning=learning,
        release_gate_passed=release_gate_passed,
        universe_id=universe_id,
        universe_tier=universe_tier_value,
        universe_size=int(universe_size),
        external_signal_version=str(settings.get("external_signal_version", "v1")),
        external_signal_enabled=bool(settings.get("external_signals", True)),
    )

    return {
        "run_dir": str(base_dir),
        "run_id": run_id,
        "baseline_reference_run_id": baseline_reference_run_id,
        "universe_tier": universe_tier_value,
        "universe_id": universe_id,
        "universe_size": str(universe_size),
        "source_universe_manifest_path": source_universe_manifest_path,
        "info_manifest": str(info_manifest_path),
        "info_shadow_report": str(info_shadow_report_path),
        "info_hash": info_hash,
        "info_item_count": str(info_manifest.get("info_item_count", 0)),
        "info_shadow_enabled": "true" if info_shadow_enabled else "false",
        "external_signal_manifest": str(external_signal_manifest_path),
        "external_signal_version": str(settings.get("external_signal_version", "v1")),
        "external_signal_enabled": "true" if bool(settings.get("external_signals", True)) else "false",
        "generator_manifest": str(settings.get("generator_manifest_path", "")),
        "generator_version": str(settings.get("generator_version", "")),
        "generator_hash": str(settings.get("generator_hash", "")),
        "coarse_pool_size": str(settings.get("coarse_pool_size", 0)),
        "refined_pool_size": str(settings.get("refined_pool_size", 0)),
        "selected_pool_size": str(settings.get("selected_pool_size", 0)),
        "use_us_index_context": "true" if bool(settings.get("use_us_index_context", False)) else "false",
        "us_index_source": str(settings.get("us_index_source", "akshare")),
        "dataset_manifest": str(dataset_path),
        "policy_calibration": str(calibration_path),
        "learned_policy_model": str(learning_path),
        "forecast_models_manifest": str(forecast_models_path) if publish_forecast_models else "",
        "frozen_forecast_bundle": str(frozen_forecast_bundle_path) if publish_forecast_models else "",
        "frozen_daily_state": str(frozen_state_path) if publish_forecast_models else "",
        "backtest_summary": str(backtest_path),
        "consistency_checklist": str(consistency_path),
        "rolling_oos_report": str(rolling_oos_path),
        "research_manifest": str(manifest_path),
        "published_policy_model": str(latest_policy_path),
        "strategy_memory": str(memory_path),
        "capital_flow_snapshot": json.dumps(external_signal_package.get("capital_flow_snapshot", {}), ensure_ascii=False),
        "macro_context_snapshot": json.dumps(external_signal_package.get("macro_context_snapshot", {}), ensure_ascii=False),
        "release_gate_passed": "true" if release_gate_passed else "false",
        "default_switch_gate_passed": "true" if default_switch_gate_passed else "false",
        "snapshot_hash": snapshot_hash,
        "config_hash": config_hash,
    }


@dataclass(frozen=True)
class _DailySnapshotContext:
    settings: dict[str, object]
    manifest: dict[str, object]
    manifest_path: Path | None
    snapshot: StrategySnapshot
    resolved_run_id: str


@dataclass(frozen=True)
class _DailyUniverseContext:
    market_security: object
    current_holdings: list[object]
    stocks: list[object]
    sector_map: dict[str, str]


def _load_daily_cached_result(
    *,
    cache_path: Path,
    refresh_cache: bool,
    memory_root: Path,
) -> DailyRunResult | None:
    return _load_daily_cached_result_external(
        cache_path=cache_path,
        refresh_cache=refresh_cache,
        memory_root=memory_root,
        emit_progress=_emit_progress,
    )


def _hydrate_daily_settings_from_dataset_manifest(
    *,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path,
    universe_tier: str | None,
    universe_file: str | None,
) -> dict[str, object]:
    dataset_path = _path_from_manifest_entry(manifest.get("dataset_manifest"), run_dir=manifest_path.parent)
    dataset_manifest = _load_json_dict(dataset_path) if dataset_path is not None else {}
    if not dataset_manifest:
        return settings
    DatasetManifest.from_payload(dataset_manifest)

    manifest_universe_tier = str(dataset_manifest.get("universe_tier", "")).strip()
    manifest_universe_file = str(dataset_manifest.get("universe_file", "")).strip()
    requested_universe_tier = str(settings.get("universe_tier", "")).strip()
    requested_universe_file = str(settings.get("universe_file", "")).strip()
    if universe_tier is not None and requested_universe_tier and manifest_universe_tier and requested_universe_tier != manifest_universe_tier:
        raise ValueError(
            f"universe tier mismatch: requested={requested_universe_tier}, manifest={manifest_universe_tier}"
        )
    if universe_file is not None and requested_universe_file and manifest_universe_file:
        requested_path = str(Path(requested_universe_file).resolve())
        manifest_path_resolved = str(Path(manifest_universe_file).resolve())
        if requested_path != manifest_path_resolved:
            raise ValueError(
                f"universe file mismatch: requested={requested_path}, manifest={manifest_path_resolved}"
            )

    hydrated = dict(settings)
    hydrated["universe_file"] = str(dataset_manifest.get("universe_file", hydrated.get("universe_file", "")))
    hydrated["universe_limit"] = int(dataset_manifest.get("universe_limit", hydrated.get("universe_limit", 0)))
    hydrated["universe_tier"] = str(dataset_manifest.get("universe_tier", hydrated.get("universe_tier", "")))
    hydrated["universe_id"] = str(dataset_manifest.get("universe_id", hydrated.get("universe_id", "")))
    hydrated["universe_size"] = int(
        dataset_manifest.get("universe_size", dataset_manifest.get("symbol_count", hydrated.get("universe_size", 0)))
    )
    hydrated["universe_generation_rule"] = str(
        dataset_manifest.get("universe_generation_rule", hydrated.get("universe_generation_rule", ""))
    )
    hydrated["source_universe_manifest_path"] = str(
        dataset_manifest.get("source_universe_manifest_path", hydrated.get("source_universe_manifest_path", ""))
    )
    hydrated["symbols"] = [
        str(item)
        for item in dataset_manifest.get("symbols", hydrated.get("symbols", []))
        if str(item).strip()
    ]
    hydrated["symbol_count"] = int(dataset_manifest.get("symbol_count", len(hydrated["symbols"])))
    hydrated["universe_hash"] = str(dataset_manifest.get("universe_hash", hydrated.get("universe_hash", "")))
    hydrated["dynamic_universe_enabled"] = _parse_boolish(
        dataset_manifest.get("dynamic_universe_enabled", hydrated.get("dynamic_universe_enabled", False)),
        False,
    )
    hydrated["generator_manifest_path"] = str(
        dataset_manifest.get("generator_manifest", hydrated.get("generator_manifest_path", ""))
    )
    hydrated["generator_version"] = str(dataset_manifest.get("generator_version", hydrated.get("generator_version", "")))
    hydrated["generator_hash"] = str(dataset_manifest.get("generator_hash", hydrated.get("generator_hash", "")))
    hydrated["coarse_pool_size"] = int(dataset_manifest.get("coarse_pool_size", hydrated.get("coarse_pool_size", 0)))
    hydrated["refined_pool_size"] = int(dataset_manifest.get("refined_pool_size", hydrated.get("refined_pool_size", 0)))
    hydrated["selected_pool_size"] = int(dataset_manifest.get("selected_pool_size", hydrated.get("selected_pool_size", 0)))
    hydrated["theme_allocations"] = [
        dict(item)
        for item in dataset_manifest.get("theme_allocations", hydrated.get("theme_allocations", []))
        if isinstance(item, dict)
    ]
    hydrated["info_file"] = str(dataset_manifest.get("info_file", hydrated.get("info_file", "")))
    hydrated["event_file"] = str(dataset_manifest.get("event_file", hydrated.get("event_file", hydrated.get("info_file", ""))))
    hydrated["info_hash"] = str(dataset_manifest.get("info_hash", hydrated.get("info_hash", "")))
    hydrated["info_shadow_enabled"] = _parse_boolish(
        dataset_manifest.get("info_shadow_enabled", hydrated.get("info_shadow_enabled", False)),
        False,
    )
    hydrated["capital_flow_file"] = str(dataset_manifest.get("capital_flow_file", hydrated.get("capital_flow_file", "")))
    hydrated["macro_file"] = str(dataset_manifest.get("macro_file", hydrated.get("macro_file", "")))
    hydrated["external_signals"] = _parse_boolish(
        dataset_manifest.get("external_signal_enabled", hydrated.get("external_signals", True)),
        True,
    )
    hydrated["external_signal_version"] = str(
        dataset_manifest.get("external_signal_version", hydrated.get("external_signal_version", "v1"))
    )
    hydrated["event_lookback_days"] = int(dataset_manifest.get("event_lookback_days", hydrated.get("event_lookback_days", hydrated.get("info_lookback_days", 45))))
    hydrated["capital_flow_lookback_days"] = int(
        dataset_manifest.get("capital_flow_lookback_days", hydrated.get("capital_flow_lookback_days", 20))
    )
    hydrated["macro_lookback_days"] = int(
        dataset_manifest.get("macro_lookback_days", hydrated.get("macro_lookback_days", 60))
    )
    hydrated["event_risk_cutoff"] = float(dataset_manifest.get("event_risk_cutoff", hydrated.get("event_risk_cutoff", 0.55)))
    hydrated["catalyst_boost_cap"] = float(dataset_manifest.get("catalyst_boost_cap", hydrated.get("catalyst_boost_cap", 0.12)))
    hydrated["flow_exposure_cap"] = float(dataset_manifest.get("flow_exposure_cap", hydrated.get("flow_exposure_cap", 0.08)))
    hydrated["info_source_mode"] = str(dataset_manifest.get("info_source_mode", hydrated.get("info_source_mode", "layered")))
    hydrated["use_us_index_context"] = _parse_boolish(
        dataset_manifest.get("use_us_index_context", manifest.get("use_us_index_context", False)),
        False,
    )
    hydrated["us_index_source"] = str(
        dataset_manifest.get("us_index_source", manifest.get("us_index_source", "akshare"))
    )
    hydrated["use_us_sector_etf_context"] = _parse_boolish(
        dataset_manifest.get("use_us_sector_etf_context", manifest.get("use_us_sector_etf_context", False)),
        False,
    )
    hydrated["use_cn_etf_context"] = _parse_boolish(
        dataset_manifest.get("use_cn_etf_context", manifest.get("use_cn_etf_context", False)),
        False,
    )
    hydrated["cn_etf_source"] = str(
        dataset_manifest.get("cn_etf_source", manifest.get("cn_etf_source", "akshare"))
    )
    hydrated["info_subsets"] = [
        str(item)
        for item in dataset_manifest.get("info_subsets", hydrated.get("info_subsets", []))
        if str(item).strip()
    ]
    hydrated["announcement_event_tags"] = [
        str(item)
        for item in dataset_manifest.get("announcement_event_tags", hydrated.get("announcement_event_tags", []))
        if str(item).strip()
    ]
    return hydrated


def _is_daily_universe_override_mismatch(exc: Exception) -> bool:
    text = str(exc)
    return "universe tier mismatch:" in text or "universe file mismatch:" in text


def _build_daily_snapshot_context(
    *,
    strategy_id: str,
    config_path: str,
    source: str | None,
    universe_file: str | None,
    universe_limit: int | None,
    universe_tier: str | None,
    dynamic_universe: bool | None = None,
    generator_target_size: int | None = None,
    generator_coarse_size: int | None = None,
    generator_theme_aware: bool | None = None,
    generator_use_concepts: bool | None = None,
    info_file: str | None,
    info_lookback_days: int | None,
    info_half_life_days: float | None,
    use_info_fusion: bool | None,
    info_shadow_only: bool | None,
    info_types: str | None,
    info_source_mode: str | None,
    info_subsets: str | None,
    external_signals: bool | None,
    event_file: str | None,
    capital_flow_file: str | None,
    macro_file: str | None,
    use_us_index_context: bool | None,
    us_index_source: str | None,
    artifact_root: str,
    cache_root: str,
    refresh_cache: bool = False,
    run_id: str | None,
    snapshot_path: str | None,
    allow_retrain: bool,
) -> _DailySnapshotContext:
    settings = _load_v2_runtime_settings(
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
        info_file=info_file,
        info_lookback_days=info_lookback_days,
        info_half_life_days=info_half_life_days,
        use_info_fusion=use_info_fusion,
        info_shadow_only=info_shadow_only,
        info_types=info_types,
        info_source_mode=info_source_mode,
        info_subsets=info_subsets,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
    )
    settings["refresh_cache"] = bool(refresh_cache)
    settings = _resolve_v2_universe_settings(settings=settings, cache_root=cache_root)

    manifest: dict[str, object] = {}
    manifest_path: Path | None = None
    try:
        manifest, manifest_path = _load_research_manifest_for_daily(
            strategy_id=strategy_id,
            artifact_root=artifact_root,
            run_id=run_id,
            snapshot_path=snapshot_path,
        )
        if manifest:
            ResearchManifest.from_payload(manifest)
    except Exception:
        if not allow_retrain:
            raise

    resolved_run_id = ""
    if manifest:
        resolved_run_id = str(manifest.get("run_id", "")).strip()
    elif run_id is not None:
        resolved_run_id = str(run_id).strip()

    if manifest and manifest_path is not None:
        try:
            settings = _hydrate_daily_settings_from_dataset_manifest(
                settings=settings,
                manifest=manifest,
                manifest_path=manifest_path,
                universe_tier=universe_tier,
                universe_file=universe_file,
            )
            snapshot = _build_snapshot_from_manifest(
                strategy_id=strategy_id,
                settings=settings,
                manifest=manifest,
                manifest_path=manifest_path,
            )
        except ValueError as exc:
            explicit_universe_override = universe_tier is not None or universe_file is not None
            if not (allow_retrain and explicit_universe_override and _is_daily_universe_override_mismatch(exc)):
                raise
            _emit_progress(
                "load-strategy-snapshot",
                "detected explicit universe override; bypassing published snapshot and rebuilding daily state.",
            )
            manifest = {}
            manifest_path = None
            resolved_run_id = ""
            data_window = f"{settings.get('start', '')}~{settings.get('end', '')}"
            snapshot = build_strategy_snapshot(
                strategy_id=strategy_id,
                universe_id=str(settings.get("universe_id", "")).strip()
                or Path(str(settings["universe_file"])).stem
                or "v2_universe",
                universe_size=int(settings.get("universe_size", settings.get("symbol_count", 0)) or 0),
                universe_generation_rule=str(settings.get("universe_generation_rule", "")),
                source_universe_manifest_path=str(
                    settings.get("source_universe_manifest_path", settings.get("universe_file", ""))
                ),
                info_manifest_path=str(settings.get("info_manifest_path", "")),
                info_hash=str(settings.get("info_hash", "")),
                info_shadow_enabled=_parse_boolish(settings.get("info_shadow_enabled", False), False),
                external_signal_manifest_path=str(settings.get("external_signal_manifest", "")),
                external_signal_version=str(settings.get("external_signal_version", "v1")),
                external_signal_enabled=_parse_boolish(settings.get("external_signals", False), False),
                capital_flow_snapshot=dict(settings.get("capital_flow_snapshot", {})),
                macro_context_snapshot=dict(settings.get("macro_context_snapshot", {})),
                generator_manifest_path=str(settings.get("generator_manifest_path", "")),
                generator_version=str(settings.get("generator_version", "")),
                generator_hash=str(settings.get("generator_hash", "")),
                coarse_pool_size=int(settings.get("coarse_pool_size", 0)),
                refined_pool_size=int(settings.get("refined_pool_size", 0)),
                selected_pool_size=int(settings.get("selected_pool_size", 0)),
                theme_allocations=[dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
                run_id="",
                data_window=data_window,
                model_hashes={},
                policy_hash="",
                universe_hash=str(settings.get("universe_hash", "")),
                created_at=str(pd.Timestamp.now().isoformat()),
                snapshot_hash="",
                config_hash=_stable_json_hash(settings),
                manifest_path="",
                use_us_index_context=_parse_boolish(settings.get("use_us_index_context", False), False),
                us_index_source=str(settings.get("us_index_source", "")),
            )
    else:
        data_window = f"{settings.get('start', '')}~{settings.get('end', '')}"
        snapshot = build_strategy_snapshot(
            strategy_id=strategy_id,
            universe_id=str(settings.get("universe_id", "")).strip() or Path(str(settings["universe_file"])).stem or "v2_universe",
            universe_size=int(settings.get("universe_size", settings.get("symbol_count", 0)) or 0),
            universe_generation_rule=str(settings.get("universe_generation_rule", "")),
            source_universe_manifest_path=str(
                settings.get("source_universe_manifest_path", settings.get("universe_file", ""))
            ),
            info_manifest_path=str(settings.get("info_manifest_path", "")),
            info_hash=str(settings.get("info_hash", "")),
            info_shadow_enabled=_parse_boolish(settings.get("info_shadow_enabled", False), False),
            external_signal_manifest_path=str(settings.get("external_signal_manifest", "")),
            external_signal_version=str(settings.get("external_signal_version", "v1")),
            external_signal_enabled=_parse_boolish(settings.get("external_signals", True), True),
            capital_flow_snapshot=dict(settings.get("capital_flow_snapshot", {})),
            macro_context_snapshot=dict(settings.get("macro_context_snapshot", {})),
            generator_manifest_path=str(settings.get("generator_manifest_path", "")),
            generator_version=str(settings.get("generator_version", "")),
            generator_hash=str(settings.get("generator_hash", "")),
            coarse_pool_size=int(settings.get("coarse_pool_size", 0)),
            refined_pool_size=int(settings.get("refined_pool_size", 0)),
            selected_pool_size=int(settings.get("selected_pool_size", 0)),
            theme_allocations=[dict(item) for item in settings.get("theme_allocations", []) if isinstance(item, dict)],
            run_id=resolved_run_id,
            data_window=data_window,
            config_hash=_stable_json_hash(settings),
            universe_hash=str(settings.get("universe_hash", "")) or _sha256_file(settings.get("universe_file", "")),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
        )

    return _DailySnapshotContext(
        settings=settings,
        manifest=manifest,
        manifest_path=manifest_path,
        snapshot=snapshot,
        resolved_run_id=resolved_run_id,
    )


def _build_daily_universe_context(settings: dict[str, object]) -> _DailyUniverseContext:
    _emit_progress("daily", "加载观察池与候选股票池")
    market_security, current_holdings, base_sector_map = load_watchlist(str(settings["watchlist"]))
    universe = build_candidate_universe(
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
    return _DailyUniverseContext(
        market_security=market_security,
        current_holdings=current_holdings,
        stocks=stocks,
        sector_map=sector_map,
    )


def _build_daily_composite_state(
    *,
    settings: dict[str, object],
    manifest: dict[str, object],
    manifest_path: Path | None,
    snapshot: StrategySnapshot,
    allow_retrain: bool,
    universe_ctx: _DailyUniverseContext,
) -> tuple[CompositeState, list[object]]:
    stock_rows: list[object] = []
    if allow_retrain:
        _emit_progress("daily", f"开始量化预测: universe={len(universe_ctx.stocks)}")
        market_forecast, stock_rows = run_quant_pipeline(
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
            progress_callback=lambda message: _emit_progress("daily", message),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
        )
        _emit_progress("daily", "开始构建市场状态与横截面状态")
        market_raw = load_symbol_daily(
            symbol=universe_ctx.market_security.symbol,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
        )
        market_state, cross_section = _build_market_and_cross_section_states(
            market_symbol=universe_ctx.market_security.symbol,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
            use_margin_features=bool(settings["use_margin_features"]),
            margin_market_file=str(settings["margin_market_file"]),
            market_short_prob=float(market_forecast.short_prob),
            market_two_prob=_safe_float(getattr(market_forecast, "two_prob", np.nan), np.nan),
            market_three_prob=_safe_float(getattr(market_forecast, "three_prob", np.nan), np.nan),
            market_five_prob=float(market_forecast.five_prob),
            market_mid_prob=float(market_forecast.mid_prob),
            market_short_profile=_ReturnQuantileProfile(
                expected_return=float(_safe_float(getattr(market_forecast, "short_expected_ret", 0.0), 0.0)),
                q10=float(_safe_float(getattr(market_forecast, "short_q10", np.nan), np.nan)),
                q30=float(_safe_float(getattr(market_forecast, "short_q30", np.nan), np.nan)),
                q20=float(_safe_float(getattr(market_forecast, "short_q20", np.nan), np.nan)),
                q50=float(_safe_float(getattr(market_forecast, "short_q50", np.nan), np.nan)),
                q70=float(_safe_float(getattr(market_forecast, "short_q70", np.nan), np.nan)),
                q80=float(_safe_float(getattr(market_forecast, "short_q80", np.nan), np.nan)),
                q90=float(_safe_float(getattr(market_forecast, "short_q90", np.nan), np.nan)),
            ),
            market_mid_profile=_ReturnQuantileProfile(
                expected_return=float(_safe_float(getattr(market_forecast, "mid_expected_ret", 0.0), 0.0)),
                q10=float(_safe_float(getattr(market_forecast, "mid_q10", np.nan), np.nan)),
                q30=float(_safe_float(getattr(market_forecast, "mid_q30", np.nan), np.nan)),
                q20=float(_safe_float(getattr(market_forecast, "mid_q20", np.nan), np.nan)),
                q50=float(_safe_float(getattr(market_forecast, "mid_q50", np.nan), np.nan)),
                q70=float(_safe_float(getattr(market_forecast, "mid_q70", np.nan), np.nan)),
                q80=float(_safe_float(getattr(market_forecast, "mid_q80", np.nan), np.nan)),
                q90=float(_safe_float(getattr(market_forecast, "mid_q90", np.nan), np.nan)),
            ),
            use_us_index_context=bool(settings.get("use_us_index_context", False)),
            us_index_source=str(settings.get("us_index_source", "akshare")),
            use_us_sector_etf_context=bool(settings.get("use_us_sector_etf_context", False)),
            use_cn_etf_context=bool(settings.get("use_cn_etf_context", False)),
            cn_etf_source=str(settings.get("cn_etf_source", "akshare")),
        )
        _emit_progress("daily", "开始独立板块预测")
        sector_frames = build_sector_daily_frames(
            stock_securities=universe_ctx.stocks,
            sector_map=universe_ctx.sector_map,
            source=str(settings["source"]),
            data_dir=str(settings["data_dir"]),
            start=str(settings["start"]),
            end=str(settings["end"]),
        )
        sector_records = run_sector_forecast(
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
        stocks_state = _build_stock_states_from_rows(
            stock_rows,
            universe_ctx.sector_map,
            sector_strength_map=sector_strength_map,
        )
        _emit_progress("daily", "开始策略决策与交易计划生成")
        composite_state = compose_state(
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
    frozen_state_path = _path_from_manifest_entry(
        manifest.get("frozen_daily_state"),
        run_dir=manifest_path.parent,
    )
    frozen_bundle_path = _path_from_manifest_entry(
        manifest.get("frozen_forecast_bundle"),
        run_dir=manifest_path.parent,
    )
    frozen_state_payload = _load_json_dict(frozen_state_path) if frozen_state_path is not None else {}
    composite_payload = frozen_state_payload.get("composite_state")
    composite_state = _decode_composite_state(composite_payload)
    if composite_state is None:
        raise RuntimeError(
            f"Snapshot {manifest_path} does not contain usable frozen daily state. "
            "Re-run research with `--publish-forecast-models` or set `--allow-retrain`."
        )
    _emit_progress("daily", f"已加载发布快照: run_id={snapshot.run_id or 'NA'}")
    frozen_bundle = _load_frozen_forecast_bundle(frozen_bundle_path) if frozen_bundle_path is not None else {}
    if frozen_bundle:
        live_composite_state, live_stock_rows = _score_live_composite_state_from_frozen_bundle(
            bundle=frozen_bundle,
            settings=settings,
            universe_ctx=universe_ctx,
        )
        if live_composite_state is not None:
            composite_state = live_composite_state
            stock_rows = live_stock_rows
            _emit_progress("daily", "loaded frozen forecast models and refreshed live scores")
    return composite_state, stock_rows


def _build_daily_symbol_names(
    *,
    current_holdings: list[object],
    stocks: list[object],
    stock_rows: list[object],
    composite_state: CompositeState,
) -> dict[str, str]:
    return _build_daily_symbol_names_external(
        current_holdings=current_holdings,
        stocks=stocks,
        stock_rows=stock_rows,
        composite_state=composite_state,
    )


def _attach_daily_info_overlay(
    *,
    snapshot: StrategySnapshot,
    settings: dict[str, object],
    composite_state: CompositeState,
    symbol_names: dict[str, str],
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
        info_file_path, info_items = _load_v2_info_items_for_date(
            settings=settings,
            as_of_date=pd.Timestamp(composite_state.market.as_of_date),
            learned_window=False,
        )
        if info_items:
            composite_state = _enrich_state_with_info(
                state=composite_state,
                as_of_date=pd.Timestamp(composite_state.market.as_of_date),
                info_items=info_items,
                settings=settings,
            )
            info_shadow_enabled = True
            info_item_count = len(info_items)
            if not info_hash:
                info_hash = _sha256_file(info_file_path) or _stable_json_hash([asdict(item) for item in info_items])
            top_negative_info_events = top_negative_events(
                info_items,
                as_of_date=pd.Timestamp(composite_state.market.as_of_date),
                half_life_days=float(settings.get("info_half_life_days", 10.0)),
            )
            top_positive_info_signals = top_positive_stock_signals(
                composite_state,
                symbol_names=symbol_names,
            )
            quant_info_divergence = quant_info_divergence_rows(
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


def _attach_daily_external_signal_overlay(
    *,
    snapshot: StrategySnapshot,
    settings: dict[str, object],
    composite_state: CompositeState,
    info_items: list[InfoItem],
    allow_rebuild: bool,
) -> tuple[CompositeState, str, str, bool, dict[str, object], dict[str, object]]:
    manifest_path = str(snapshot.external_signal_manifest_path or settings.get("external_signal_manifest", ""))
    version = str(snapshot.external_signal_version or settings.get("external_signal_version", "v1"))
    enabled = bool(snapshot.external_signal_enabled or settings.get("external_signals", True))
    capital_flow_snapshot = dict(snapshot.capital_flow_snapshot or asdict(composite_state.capital_flow_state))
    macro_context_snapshot = dict(snapshot.macro_context_snapshot or asdict(composite_state.macro_context_state))

    if allow_rebuild or not snapshot.external_signal_enabled:
        composite_state, package = _attach_external_signals_to_composite_state(
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


def _resolve_daily_policy_model(
    *,
    strategy_id: str,
    artifact_root: str,
    manifest: dict[str, object],
    manifest_path: Path | None,
) -> LearnedPolicyModel | None:
    learned_policy = None
    if manifest and manifest_path is not None:
        model_path = _path_from_manifest_entry(
            manifest.get("learned_policy_model"),
            run_dir=manifest_path.parent,
        )
        if model_path is not None:
            learned_policy = _load_policy_model_from_path(model_path)
    if learned_policy is None:
        learned_policy = _load_published_v2_policy_model_impl(
            strategy_id=strategy_id,
            artifact_root=artifact_root,
        )
    return learned_policy


def _run_daily_v2_live_impl(
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
    info_file: str | None = None,
    info_lookback_days: int | None = None,
    info_half_life_days: float | None = None,
    use_info_fusion: bool | None = None,
    info_shadow_only: bool | None = None,
    info_types: str | None = None,
    info_source_mode: str | None = None,
    info_subsets: str | None = None,
    external_signals: bool | None = None,
    event_file: str | None = None,
    capital_flow_file: str | None = None,
    macro_file: str | None = None,
    use_us_index_context: bool | None = None,
    us_index_source: str | None = None,
    artifact_root: str = "artifacts/v2",
    cache_root: str = "artifacts/v2/cache",
    refresh_cache: bool = False,
    run_id: str | None = None,
    snapshot_path: str | None = None,
    allow_retrain: bool = False,
) -> DailyRunResult:
    memory_root = Path(str(artifact_root)) / "memory"
    snapshot_ctx = _build_daily_snapshot_context(
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
        info_file=info_file,
        info_lookback_days=info_lookback_days,
        info_half_life_days=info_half_life_days,
        use_info_fusion=use_info_fusion,
        info_shadow_only=info_shadow_only,
        info_types=info_types,
        info_source_mode=info_source_mode,
        info_subsets=info_subsets,
        external_signals=external_signals,
        event_file=event_file,
        capital_flow_file=capital_flow_file,
        macro_file=macro_file,
        use_us_index_context=use_us_index_context,
        us_index_source=us_index_source,
        artifact_root=artifact_root,
        cache_root=cache_root,
        refresh_cache=refresh_cache,
        run_id=run_id,
        snapshot_path=snapshot_path,
        allow_retrain=allow_retrain,
    )
    settings = snapshot_ctx.settings

    cache_key = _daily_result_cache_key(
        strategy_id=strategy_id,
        settings=settings,
        artifact_root=artifact_root,
        run_id=snapshot_ctx.resolved_run_id,
        snapshot_path=str(snapshot_path or ""),
        allow_retrain=allow_retrain,
    )
    cache_path = _daily_result_cache_path(
        cache_root=cache_root,
        cache_key=cache_key,
    )
    cached = _load_daily_cached_result(
        cache_path=cache_path,
        refresh_cache=refresh_cache,
        memory_root=memory_root,
    )
    if cached is not None:
        return cached

    snapshot = snapshot_ctx.snapshot
    manifest = snapshot_ctx.manifest
    manifest_path = snapshot_ctx.manifest_path
    universe_ctx = _build_daily_universe_context(settings)
    composite_state, stock_rows = _build_daily_composite_state(
        settings=settings,
        manifest=manifest,
        manifest_path=manifest_path,
        snapshot=snapshot,
        allow_retrain=allow_retrain,
        universe_ctx=universe_ctx,
    )

    current_weights = {}
    if universe_ctx.current_holdings:
        equal_weight = 1.0 / float(len(universe_ctx.current_holdings))
        current_weights = {item.symbol: float(equal_weight) for item in universe_ctx.current_holdings}
    symbol_names = _build_daily_symbol_names(
        current_holdings=universe_ctx.current_holdings,
        stocks=universe_ctx.stocks,
        stock_rows=stock_rows,
        composite_state=composite_state,
    )
    (
        composite_state,
        info_hash,
        info_manifest_path,
        info_shadow_enabled,
        info_item_count,
        top_negative_info_events,
        top_positive_info_signals,
        quant_info_divergence,
        info_items,
    ) = _attach_daily_info_overlay(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        symbol_names=symbol_names,
    )
    (
        composite_state,
        external_signal_manifest_path,
        external_signal_version,
        external_signal_enabled,
        capital_flow_snapshot,
        macro_context_snapshot,
    ) = _attach_daily_external_signal_overlay(
        snapshot=snapshot,
        settings=settings,
        composite_state=composite_state,
        info_items=info_items,
        allow_rebuild=allow_retrain,
    )
    composite_state = _filter_state_for_recommendation_scope(
        state=composite_state,
        main_board_only=_parse_boolish(settings.get("main_board_only_recommendations", False), False),
    )

    learned_policy = _resolve_daily_policy_model(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    active_policy_spec = None
    if learned_policy is not None:
        active_policy_spec = _policy_spec_from_model(
            state=composite_state,
            model=learned_policy,
        )
    active_policy_spec = replace(
        active_policy_spec or PolicySpec(),
        event_risk_cutoff=float(settings.get("event_risk_cutoff", 0.55)),
        catalyst_boost_cap=float(settings.get("catalyst_boost_cap", 0.12)),
        flow_exposure_cap=float(settings.get("flow_exposure_cap", 0.08)),
    )

    policy_decision = apply_policy(
        PolicyInput(
            composite_state=composite_state,
            current_weights=current_weights,
            current_cash=max(0.0, 1.0 - sum(current_weights.values())),
            total_equity=1.0,
            current_holding_days={symbol: 5 for symbol in current_weights},
        ),
        policy_spec=active_policy_spec,
    )
    trade_actions = build_trade_actions(
        decision=policy_decision,
        current_weights=current_weights,
    )
    prediction_review, calibration_priors = _load_prediction_review_context(
        manifest=manifest,
        manifest_path=manifest_path,
    )
    reporting_market, reporting_cross_section = _build_live_market_reporting_overlay(
        settings=settings,
        universe_ctx=universe_ctx,
        state=composite_state,
    )
    composite_state = _decorate_composite_state_for_reporting(
        state=composite_state,
        policy=policy_decision,
        calibration_priors=calibration_priors,
        reporting_market=reporting_market,
        reporting_cross_section=reporting_cross_section,
    )
    result = DailyRunResult(
        snapshot=snapshot,
        composite_state=composite_state,
        policy_decision=policy_decision,
        trade_actions=trade_actions,
        symbol_names=symbol_names,
        info_hash=info_hash,
        info_manifest_path=info_manifest_path,
        info_shadow_enabled=info_shadow_enabled,
        info_item_count=info_item_count,
        external_signal_manifest_path=external_signal_manifest_path,
        external_signal_version=external_signal_version,
        external_signal_enabled=external_signal_enabled,
        capital_flow_snapshot=capital_flow_snapshot,
        macro_context_snapshot=macro_context_snapshot,
        generator_manifest_path=snapshot.generator_manifest_path,
        generator_version=snapshot.generator_version,
        generator_hash=snapshot.generator_hash,
        coarse_pool_size=snapshot.coarse_pool_size,
        refined_pool_size=snapshot.refined_pool_size,
        selected_pool_size=snapshot.selected_pool_size,
        theme_allocations=list(snapshot.theme_allocations),
        top_negative_info_events=top_negative_info_events,
        top_positive_info_signals=top_positive_info_signals,
        quant_info_divergence=quant_info_divergence,
        run_id=snapshot.run_id,
        snapshot_hash=snapshot.snapshot_hash,
        config_hash=snapshot.config_hash,
        manifest_path=snapshot.manifest_path,
        prediction_review=prediction_review,
    )
    result = remember_daily_run(
        memory_root=memory_root,
        result=result,
    )
    try:
        with cache_path.open("wb") as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        _emit_progress("daily", "日运行缓存已写入")
    except Exception:
        pass
    return result


def run_v2_research_workflow(**kwargs: object) -> tuple[V2BacktestSummary, V2CalibrationResult, V2PolicyLearningResult]:
    from src.workflows.research_workflow import run_v2_research_workflow as _run

    return _run(**kwargs)


def run_v2_research_matrix(**kwargs: object) -> dict[str, object]:
    from src.workflows.research_workflow import run_v2_research_matrix as _run

    return _run(**kwargs)


def load_published_v2_policy_model(
    *,
    strategy_id: str,
    artifact_root: str = "artifacts/v2",
) -> LearnedPolicyModel | None:
    from src.artifact_registry.v2_registry import load_published_v2_policy_model as _load

    return _load(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
    )


def publish_v2_research_artifacts(
    *,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    settings: dict[str, object] | None = None,
    **kwargs: object,
) -> dict[str, str]:
    from src.artifact_registry.v2_registry import publish_v2_research_artifacts as _publish

    return _publish(
        baseline=baseline,
        calibration=calibration,
        learning=learning,
        settings=settings,
        **kwargs,
    )


def run_daily_v2_live(**kwargs: object) -> DailyRunResult:
    from src.workflows.daily_workflow import run_daily_v2_live as _run

    return _run(**kwargs)


def summarize_daily_run(result: DailyRunResult) -> dict[str, object]:
    def _display_name(symbol: str) -> str:
        return str(result.symbol_names.get(symbol, symbol))

    policy_payload = asdict(result.policy_decision)
    policy_payload["symbol_target_weights"] = {
        _display_name(symbol): weight
        for symbol, weight in result.policy_decision.symbol_target_weights.items()
    }
    policy_payload["desired_symbol_target_weights"] = {
        _display_name(symbol): weight
        for symbol, weight in result.policy_decision.desired_symbol_target_weights.items()
    }
    return {
        "strategy_id": result.snapshot.strategy_id,
        "universe_id": result.snapshot.universe_id,
        "universe_size": result.snapshot.universe_size,
        "universe_generation_rule": result.snapshot.universe_generation_rule,
        "source_universe_manifest_path": result.snapshot.source_universe_manifest_path,
        "info_hash": result.info_hash or result.snapshot.info_hash,
        "info_manifest_path": result.info_manifest_path or result.snapshot.info_manifest_path,
        "info_shadow_enabled": bool(result.info_shadow_enabled or result.snapshot.info_shadow_enabled),
        "info_item_count": int(result.info_item_count),
        "external_signal_manifest_path": result.external_signal_manifest_path or result.snapshot.external_signal_manifest_path,
        "external_signal_version": result.external_signal_version or result.snapshot.external_signal_version,
        "external_signal_enabled": bool(result.external_signal_enabled or result.snapshot.external_signal_enabled),
        "capital_flow_snapshot": dict(result.capital_flow_snapshot or result.snapshot.capital_flow_snapshot),
        "macro_context_snapshot": dict(result.macro_context_snapshot or result.snapshot.macro_context_snapshot),
        "generator_manifest_path": result.generator_manifest_path or result.snapshot.generator_manifest_path,
        "generator_version": result.generator_version or result.snapshot.generator_version,
        "generator_hash": result.generator_hash or result.snapshot.generator_hash,
        "coarse_pool_size": int(result.coarse_pool_size or result.snapshot.coarse_pool_size),
        "refined_pool_size": int(result.refined_pool_size or result.snapshot.refined_pool_size),
        "selected_pool_size": int(result.selected_pool_size or result.snapshot.selected_pool_size),
        "theme_allocations": [dict(item) for item in (result.theme_allocations or result.snapshot.theme_allocations)],
        "use_us_index_context": bool(result.snapshot.use_us_index_context),
        "us_index_source": str(result.snapshot.us_index_source),
        "run_id": result.run_id or result.snapshot.run_id,
        "snapshot_hash": result.snapshot_hash or result.snapshot.snapshot_hash,
        "config_hash": result.config_hash or result.snapshot.config_hash,
        "manifest_path": result.manifest_path or result.snapshot.manifest_path,
        "memory_path": result.memory_path,
        "memory_recall": asdict(result.memory_recall),
        "prediction_review": asdict(result.prediction_review),
        "strategy_mode": result.composite_state.strategy_mode,
        "risk_regime": result.composite_state.risk_regime,
        "market": asdict(result.composite_state.market),
        "market_info_state": asdict(result.composite_state.market_info_state),
        "capital_flow_state": asdict(result.composite_state.capital_flow_state),
        "macro_context_state": asdict(result.composite_state.macro_context_state),
        "mainlines": [asdict(item) for item in getattr(result.composite_state, "mainlines", [])],
        "policy": policy_payload,
        "top_negative_info_events": [asdict(item) for item in result.top_negative_info_events],
        "top_positive_info_signals": [asdict(item) for item in result.top_positive_info_signals],
        "quant_info_divergence": [asdict(item) for item in result.quant_info_divergence],
        "trade_plan": [
            {
                "stock": _display_name(action.symbol),
                "action": action.action,
                "current_weight": action.current_weight,
                "target_weight": action.target_weight,
                "delta_weight": action.delta_weight,
                "note": action.note,
            }
            for action in result.trade_actions
        ],
    }


def summarize_v2_backtest(
    result: V2BacktestSummary,
    *,
    run_id: str | None = None,
    snapshot_hash: str | None = None,
    config_hash: str | None = None,
) -> dict[str, object]:
    payload = asdict(result)
    payload.pop("nav_curve", None)
    payload.pop("benchmark_nav_curve", None)
    payload.pop("excess_nav_curve", None)
    payload.pop("curve_dates", None)
    payload["run_id"] = str(run_id if run_id is not None else result.run_id)
    payload["snapshot_hash"] = str(snapshot_hash if snapshot_hash is not None else result.snapshot_hash)
    payload["config_hash"] = str(config_hash if config_hash is not None else result.config_hash)
    return payload


def summarize_v2_calibration(result: V2CalibrationResult) -> dict[str, object]:
    return {
        "best_score": float(result.best_score),
        "best_policy": asdict(result.best_policy),
        "baseline": summarize_v2_backtest(result.baseline),
        "calibrated": summarize_v2_backtest(result.calibrated),
        "trial_count": len(result.trials),
    }


def summarize_v2_policy_learning(result: V2PolicyLearningResult) -> dict[str, object]:
    return {
        "model": asdict(result.model),
        "baseline": summarize_v2_backtest(result.baseline),
        "learned": summarize_v2_backtest(result.learned),
    }
