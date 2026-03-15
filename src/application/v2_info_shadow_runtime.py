from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from src.application.v2_contracts import CompositeState, InfoAggregateState, InfoItem
from src.domain.news import blend_probability
from src.infrastructure.info_repository import load_v2_info_items
from src.infrastructure.modeling import LogisticBinaryModel


INFO_SHADOW_FEATURE_COLUMNS = [
    "q_logit",
    "i_logit",
    "q_minus_i",
    "negative_event_risk",
    "item_count_log",
]


@dataclass(frozen=True)
class InfoShadowModel:
    mode: str
    samples: int
    feature_cols: list[str]
    model: LogisticBinaryModel | None = None


@dataclass(frozen=True)
class InfoShadowRuntimeDependencies:
    build_sector_map_from_state: Callable[[CompositeState], dict[str, str]]
    build_info_state_maps: Callable[..., tuple[InfoAggregateState, dict[str, InfoAggregateState], dict[str, InfoAggregateState]]]
    predict_info_shadow_prob: Callable[..., tuple[float, dict[str, float]]]
    blend_probability: Callable[..., float]
    build_mainline_states: Callable[..., list[object]]
    stock_policy_score: Callable[[object], float]
    compose_shadow_stock_score: Callable[..., float]
    safe_float: Callable[[object, float], float]
    info_aggregate_state_cls: type[InfoAggregateState]
    info_feature_frame: Callable[..., pd.DataFrame]
    fit_info_shadow_model: Callable[..., object]
    info_shadow_feature_columns: list[str]
    panel_slice_metrics: Callable[..., tuple[float, float, float, float]]


def clip_prob(p: float) -> float:
    return float(np.clip(float(p), 1e-6, 1.0 - 1e-6))


def logit_prob(p: float) -> float:
    clipped = clip_prob(p)
    return float(np.log(clipped / (1.0 - clipped)))


def resolve_info_file_from_settings(settings: dict[str, object]) -> str:
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


def load_v2_info_items_for_date(
    *,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    learned_window: bool = False,
) -> tuple[str, list[InfoItem]]:
    info_file = resolve_info_file_from_settings(settings)
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


def info_feature_frame(
    *,
    quant_prob: Iterable[float],
    info_prob: Iterable[float],
    negative_event_risk: Iterable[float],
    item_count: Iterable[float],
) -> pd.DataFrame:
    q = np.asarray([clip_prob(float(item)) for item in quant_prob], dtype=float)
    i = np.asarray([clip_prob(float(item)) for item in info_prob], dtype=float)
    return pd.DataFrame(
        {
            "q_logit": [logit_prob(float(item)) for item in q],
            "i_logit": [logit_prob(float(item)) for item in i],
            "q_minus_i": q - i,
            "negative_event_risk": np.asarray(list(negative_event_risk), dtype=float),
            "item_count_log": np.log1p(np.asarray(list(item_count), dtype=float)),
        }
    )


def fit_info_shadow_model(
    frame: pd.DataFrame,
    *,
    target_col: str,
    l2: float,
    min_samples: int,
) -> InfoShadowModel:
    valid = frame.dropna(subset=INFO_SHADOW_FEATURE_COLUMNS + [target_col]).copy()
    if len(valid) < int(min_samples):
        return InfoShadowModel(mode="rule", samples=int(len(valid)), feature_cols=list(INFO_SHADOW_FEATURE_COLUMNS))
    model = LogisticBinaryModel(l2=float(l2)).fit(valid, INFO_SHADOW_FEATURE_COLUMNS, target_col)
    return InfoShadowModel(
        mode="learned",
        samples=int(len(valid)),
        feature_cols=list(INFO_SHADOW_FEATURE_COLUMNS),
        model=model,
    )


def predict_info_shadow_prob(
    *,
    quant_prob: float,
    info_prob: float,
    negative_event_risk: float,
    item_count: int,
    score: float,
    strength: float,
    model: InfoShadowModel | None,
) -> tuple[float, str]:
    if model is not None and model.mode == "learned" and model.model is not None:
        frame = info_feature_frame(
            quant_prob=[quant_prob],
            info_prob=[info_prob],
            negative_event_risk=[negative_event_risk],
            item_count=[item_count],
        )
        prob = float(model.model.predict_proba(frame, model.feature_cols)[0])
        return clip_prob(prob), "learned"
    return float(blend_probability(quant_prob, score, sentiment_strength=strength)), "rule"


def enrich_state_with_info(
    *,
    state: CompositeState,
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
    settings: dict[str, object],
    stock_models: dict[str, object] | None = None,
    market_models: dict[str, object] | None = None,
    deps: InfoShadowRuntimeDependencies,
) -> CompositeState:
    sector_map = deps.build_sector_map_from_state(state)
    market_info_state, sector_info_states, stock_info_states = deps.build_info_state_maps(
        info_items=info_items,
        as_of_date=as_of_date,
        stock_symbols=sector_map.keys(),
        sector_map=sector_map,
        market_to_stock_carry=0.35,
        info_half_life_days=float(settings.get("info_half_life_days", 10.0)),
        market_info_strength=float(settings.get("market_info_strength", 0.9)),
        stock_info_strength=float(settings.get("stock_info_strength", 1.1)),
    )

    market_shadow_1d, _ = deps.predict_info_shadow_prob(
        quant_prob=float(state.market.up_1d_prob),
        info_prob=float(market_info_state.info_prob_1d),
        negative_event_risk=float(market_info_state.negative_event_risk),
        item_count=int(market_info_state.item_count),
        score=float(market_info_state.short_score),
        strength=float(settings.get("market_info_strength", 0.9)),
        model=None if market_models is None else market_models.get("1d"),
    )
    market_shadow_5d, _ = deps.predict_info_shadow_prob(
        quant_prob=float(state.market.up_5d_prob),
        info_prob=float(market_info_state.info_prob_5d),
        negative_event_risk=float(market_info_state.negative_event_risk),
        item_count=int(market_info_state.item_count),
        score=float(market_info_state.short_score),
        strength=0.9 * float(settings.get("market_info_strength", 0.9)),
        model=None if market_models is None else market_models.get("5d"),
    )
    market_shadow_20d, _ = deps.predict_info_shadow_prob(
        quant_prob=float(state.market.up_20d_prob),
        info_prob=float(market_info_state.info_prob_20d),
        negative_event_risk=float(market_info_state.negative_event_risk),
        item_count=int(market_info_state.item_count),
        score=float(market_info_state.mid_score),
        strength=float(settings.get("market_info_strength", 0.9)),
        model=None if market_models is None else market_models.get("20d"),
    )
    market_info_state = deps.info_aggregate_state_cls(
        **{
            **asdict(market_info_state),
            "shadow_prob_1d": float(market_shadow_1d),
            "shadow_prob_5d": float(market_shadow_5d),
            "shadow_prob_20d": float(market_shadow_20d),
        }
    )

    updated_sector_states: dict[str, InfoAggregateState] = {}
    for sector in state.sectors:
        current = sector_info_states.get(sector.sector, deps.info_aggregate_state_cls())
        shadow_1d = float(
            deps.blend_probability(0.5, current.short_score, sentiment_strength=0.7 * float(settings.get("stock_info_strength", 1.1)))
        )
        shadow_5d = float(
            deps.blend_probability(
                float(sector.up_5d_prob),
                current.short_score,
                sentiment_strength=0.6 * float(settings.get("stock_info_strength", 1.1)),
            )
        )
        shadow_20d = float(
            deps.blend_probability(
                float(sector.up_20d_prob),
                current.mid_score,
                sentiment_strength=0.6 * float(settings.get("stock_info_strength", 1.1)),
            )
        )
        updated_sector_states[sector.sector] = deps.info_aggregate_state_cls(
            **{
                **asdict(current),
                "shadow_prob_1d": shadow_1d,
                "shadow_prob_5d": shadow_5d,
                "shadow_prob_20d": shadow_20d,
            }
        )

    updated_stock_states: dict[str, InfoAggregateState] = {}
    for stock in state.stocks:
        current = stock_info_states.get(stock.symbol, deps.info_aggregate_state_cls())
        shadow_1d, _ = deps.predict_info_shadow_prob(
            quant_prob=float(stock.up_1d_prob),
            info_prob=float(current.info_prob_1d),
            negative_event_risk=float(current.negative_event_risk),
            item_count=int(current.item_count),
            score=float(current.short_score),
            strength=float(settings.get("stock_info_strength", 1.1)),
            model=None if stock_models is None else stock_models.get("1d"),
        )
        shadow_5d, _ = deps.predict_info_shadow_prob(
            quant_prob=float(stock.up_5d_prob),
            info_prob=float(current.info_prob_5d),
            negative_event_risk=float(current.negative_event_risk),
            item_count=int(current.item_count),
            score=float(current.short_score),
            strength=0.9 * float(settings.get("stock_info_strength", 1.1)),
            model=None if stock_models is None else stock_models.get("5d"),
        )
        shadow_20d, _ = deps.predict_info_shadow_prob(
            quant_prob=float(stock.up_20d_prob),
            info_prob=float(current.info_prob_20d),
            negative_event_risk=float(current.negative_event_risk),
            item_count=int(current.item_count),
            score=float(current.mid_score),
            strength=float(settings.get("stock_info_strength", 1.1)),
            model=None if stock_models is None else stock_models.get("20d"),
        )
        updated_stock_states[stock.symbol] = deps.info_aggregate_state_cls(
            **{
                **asdict(current),
                "shadow_prob_1d": float(shadow_1d),
                "shadow_prob_5d": float(shadow_5d),
                "shadow_prob_20d": float(shadow_20d),
            }
        )

    updated_mainlines = deps.build_mainline_states(
        market=state.market,
        cross_section=state.cross_section,
        sectors=state.sectors,
        stocks=state.stocks,
        stock_score_fn=deps.stock_policy_score,
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


def fit_v2_info_shadow_models(
    *,
    trajectory: object,
    settings: dict[str, object],
    info_items: list[InfoItem],
    deps: InfoShadowRuntimeDependencies,
) -> tuple[dict[str, object], dict[str, object]]:
    stock_rows_by_horizon: dict[str, list[dict[str, float]]] = {"1d": [], "5d": [], "20d": []}
    market_rows_by_horizon: dict[str, list[dict[str, float]]] = {"1d": [], "5d": [], "20d": []}
    for step in trajectory.steps:
        enriched_state = enrich_state_with_info(
            state=step.composite_state,
            as_of_date=pd.Timestamp(step.date),
            info_items=info_items,
            settings=settings,
            deps=deps,
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
                    "y": 1.0 if deps.safe_float(row.get("mkt_fwd_ret_1"), 0.0) > 0.0 else 0.0,
                }
            )
            market_rows_by_horizon["5d"].append(
                {
                    "quant_prob": float(enriched_state.market.up_5d_prob),
                    "info_prob": float(market_info.info_prob_5d),
                    "negative_event_risk": float(market_info.negative_event_risk),
                    "item_count": float(market_info.item_count),
                    "y": 1.0 if deps.safe_float(row.get("mkt_fwd_ret_5"), 0.0) > 0.0 else 0.0,
                }
            )
            market_rows_by_horizon["20d"].append(
                {
                    "quant_prob": float(enriched_state.market.up_20d_prob),
                    "info_prob": float(market_info.info_prob_20d),
                    "negative_event_risk": float(market_info.negative_event_risk),
                    "item_count": float(market_info.item_count),
                    "y": 1.0 if deps.safe_float(row.get("mkt_fwd_ret_20"), 0.0) > 0.0 else 0.0,
                }
            )
        for stock in enriched_state.stocks:
            info_state = enriched_state.stock_info_states.get(stock.symbol, deps.info_aggregate_state_cls())
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
                    "y": 1.0 if deps.safe_float(payload.get("excess_ret_1_vs_mkt"), 0.0) > 0.0 else 0.0,
                }
            )
            stock_rows_by_horizon["5d"].append(
                {
                    "quant_prob": float(stock.up_5d_prob),
                    "info_prob": float(info_state.info_prob_5d),
                    "negative_event_risk": float(info_state.negative_event_risk),
                    "item_count": float(info_state.item_count),
                    "y": 1.0 if deps.safe_float(payload.get("excess_ret_5_vs_mkt"), 0.0) > 0.0 else 0.0,
                }
            )
            stock_rows_by_horizon["20d"].append(
                {
                    "quant_prob": float(stock.up_20d_prob),
                    "info_prob": float(info_state.info_prob_20d),
                    "negative_event_risk": float(info_state.negative_event_risk),
                    "item_count": float(info_state.item_count),
                    "y": 1.0 if deps.safe_float(payload.get("excess_ret_20_vs_sector"), 0.0) > 0.0 else 0.0,
                }
            )

    def _fit_bucket(bucket: list[dict[str, float]]) -> object:
        if not bucket:
            return {"mode": "rule", "samples": 0, "feature_cols": list(deps.info_shadow_feature_columns)}
        frame = deps.info_feature_frame(
            quant_prob=[row["quant_prob"] for row in bucket],
            info_prob=[row["info_prob"] for row in bucket],
            negative_event_risk=[row["negative_event_risk"] for row in bucket],
            item_count=[row["item_count"] for row in bucket],
        )
        frame["y"] = [float(row["y"]) for row in bucket]
        return deps.fit_info_shadow_model(
            frame,
            target_col="y",
            l2=float(settings.get("learned_info_l2", 0.8)),
            min_samples=int(settings.get("learned_info_min_samples", 80)),
        )

    return (
        {horizon: _fit_bucket(bucket) for horizon, bucket in stock_rows_by_horizon.items()},
        {horizon: _fit_bucket(bucket) for horizon, bucket in market_rows_by_horizon.items()},
    )


def build_shadow_scored_rows_for_step(
    *,
    state: CompositeState,
    stock_frames: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    deps: InfoShadowRuntimeDependencies,
) -> tuple[pd.DataFrame, bool]:
    rows: list[dict[str, float | str]] = []
    event_day = False
    for stock in state.stocks:
        info_state = state.stock_info_states.get(stock.symbol, deps.info_aggregate_state_cls())
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
                "score": deps.compose_shadow_stock_score(stock=stock, info_state=info_state),
                "realized_ret_1d": deps.safe_float(payload.get("excess_ret_1_vs_mkt"), np.nan),
                "realized_ret_5d": deps.safe_float(payload.get("excess_ret_5_vs_mkt"), np.nan),
                "realized_ret_20d": deps.safe_float(payload.get("excess_ret_20_vs_sector"), np.nan),
            }
        )
    return pd.DataFrame(rows), bool(event_day)


def build_info_shadow_variant(
    *,
    validation_trajectory: object,
    holdout_trajectory: object,
    settings: dict[str, object],
    info_items: list[InfoItem],
    deps: InfoShadowRuntimeDependencies,
) -> dict[str, object]:
    stock_models, market_models = fit_v2_info_shadow_models(
        trajectory=validation_trajectory,
        settings=settings,
        info_items=info_items,
        deps=deps,
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
        enriched_state = enrich_state_with_info(
            state=step.composite_state,
            as_of_date=pd.Timestamp(step.date),
            info_items=info_items,
            settings=settings,
            stock_models=stock_models,
            market_models=market_models,
            deps=deps,
        )
        shadow_rows, event_day = build_shadow_scored_rows_for_step(
            state=enriched_state,
            stock_frames=holdout_trajectory.prepared.stock_frames,
            date=step.date,
            deps=deps,
        )
        rank_ic_1d, _, _, top_k_1d = deps.panel_slice_metrics(shadow_rows, realized_col="realized_ret_1d")
        rank_ic_5d, _, _, _ = deps.panel_slice_metrics(shadow_rows, realized_col="realized_ret_5d")
        rank_ic_20d, _, spread_20d, top_k_20d = deps.panel_slice_metrics(shadow_rows, realized_col="realized_ret_20d")
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
            info_state = enriched_state.stock_info_states.get(stock.symbol, deps.info_aggregate_state_cls())
            quant_score = deps.stock_policy_score(stock)
            shadow_score = deps.compose_shadow_stock_score(stock=stock, info_state=info_state)
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
