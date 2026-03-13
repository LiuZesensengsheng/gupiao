from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.application.v2_contracts import (
    CapitalFlowState,
    CandidateSelectionState,
    CompositeState,
    CrossSectionForecastState,
    HorizonForecast,
    InfoAggregateState,
    MainlineState,
    MarketFactsState,
    MarketForecastState,
    MarketSentimentState,
    MacroContextState,
    SectorForecastState,
    StockForecastState,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_RELATIVE_ROOTS = {"artifacts", "config", "data", "docs", "input", "reports", "src", "tests"}


def resolve_manifest_entry_path(entry: object, *, run_dir: Path) -> Path | None:
    if entry is None:
        return None
    text = str(entry).strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    repo_candidate = (_REPO_ROOT / path).resolve()
    run_candidate = (run_dir / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    if repo_candidate.exists():
        return repo_candidate
    if run_candidate.exists():
        return run_candidate
    first = path.parts[0].lower() if path.parts else ""
    if first in _REPO_RELATIVE_ROOTS:
        return repo_candidate
    return run_candidate


def decode_composite_state(payload: object) -> CompositeState | None:
    if not isinstance(payload, dict):
        return None
    market_raw = payload.get("market")
    cross_raw = payload.get("cross_section")
    sectors_raw = payload.get("sectors")
    stocks_raw = payload.get("stocks")
    if not isinstance(market_raw, dict) or not isinstance(cross_raw, dict):
        return None
    if not isinstance(sectors_raw, list) or not isinstance(stocks_raw, list):
        return None
    try:
        def _decode_horizon_map(raw: object) -> dict[str, HorizonForecast]:
            if not isinstance(raw, dict):
                return {}
            out: dict[str, HorizonForecast] = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    try:
                        out[str(key)] = HorizonForecast(**value)
                    except Exception:
                        continue
            return out

        def _decode_market(raw: dict[str, object]) -> MarketForecastState:
            payload = dict(raw)
            payload["horizon_forecasts"] = _decode_horizon_map(payload.get("horizon_forecasts"))
            market_facts_raw = payload.get("market_facts")
            if isinstance(market_facts_raw, dict):
                payload["market_facts"] = MarketFactsState(**market_facts_raw)
            sentiment_raw = payload.get("sentiment")
            if isinstance(sentiment_raw, dict):
                payload["sentiment"] = MarketSentimentState(**sentiment_raw)
            return MarketForecastState(**payload)

        def _decode_stock(raw: dict[str, object]) -> StockForecastState:
            payload = dict(raw)
            payload["horizon_forecasts"] = _decode_horizon_map(payload.get("horizon_forecasts"))
            return StockForecastState(**payload)

        market = _decode_market(market_raw)
        cross = CrossSectionForecastState(**cross_raw)
        sectors = [SectorForecastState(**item) for item in sectors_raw if isinstance(item, dict)]
        stocks = [_decode_stock(item) for item in stocks_raw if isinstance(item, dict)]
        candidate_selection_raw = payload.get("candidate_selection", payload.get("candidate_selection_state", {}))
        mainlines_raw = payload.get("mainlines", [])
        market_info_raw = payload.get("market_info_state", {})
        sector_info_raw = payload.get("sector_info_states", {})
        stock_info_raw = payload.get("stock_info_states", {})
        return CompositeState(
            market=market,
            cross_section=cross,
            sectors=sectors,
            stocks=stocks,
            strategy_mode=str(payload.get("strategy_mode", "")),
            risk_regime=str(payload.get("risk_regime", "")),
            candidate_selection=CandidateSelectionState(**candidate_selection_raw)
            if isinstance(candidate_selection_raw, dict)
            else CandidateSelectionState(),
            mainlines=[MainlineState(**item) for item in mainlines_raw if isinstance(item, dict)],
            market_info_state=InfoAggregateState(**market_info_raw) if isinstance(market_info_raw, dict) else InfoAggregateState(),
            sector_info_states={
                str(key): InfoAggregateState(**value)
                for key, value in sector_info_raw.items()
                if isinstance(value, dict)
            } if isinstance(sector_info_raw, dict) else {},
            stock_info_states={
                str(key): InfoAggregateState(**value)
                for key, value in stock_info_raw.items()
                if isinstance(value, dict)
            } if isinstance(stock_info_raw, dict) else {},
            capital_flow_state=CapitalFlowState(**payload.get("capital_flow_state", {}))
            if isinstance(payload.get("capital_flow_state"), dict)
            else CapitalFlowState(),
            macro_context_state=MacroContextState(**payload.get("macro_context_state", {}))
            if isinstance(payload.get("macro_context_state"), dict)
            else MacroContextState(),
        )
    except Exception:
        return None


def _serialize_horizon_map(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        return {}
    payload: dict[str, object] = {}
    for key, value in raw.items():
        item = value if isinstance(value, HorizonForecast) else HorizonForecast(**value) if isinstance(value, dict) else None
        if item is None:
            continue
        payload[str(key)] = asdict(item)
    return payload


def _serialize_market_state(raw: object) -> dict[str, object]:
    market = raw if isinstance(raw, MarketForecastState) else None
    market_facts = getattr(raw, "market_facts", MarketFactsState())
    sentiment = getattr(raw, "sentiment", MarketSentimentState())
    return {
        "as_of_date": str(getattr(raw, "as_of_date", "")),
        "up_1d_prob": float(getattr(raw, "up_1d_prob", 0.5)),
        "up_5d_prob": float(getattr(raw, "up_5d_prob", 0.5)),
        "up_20d_prob": float(getattr(raw, "up_20d_prob", 0.5)),
        "trend_state": str(getattr(raw, "trend_state", "neutral")),
        "drawdown_risk": float(getattr(raw, "drawdown_risk", 0.0)),
        "volatility_regime": str(getattr(raw, "volatility_regime", "neutral")),
        "liquidity_stress": float(getattr(raw, "liquidity_stress", 0.0)),
        "up_2d_prob": float(getattr(raw, "up_2d_prob", 0.5)),
        "up_3d_prob": float(getattr(raw, "up_3d_prob", 0.5)),
        "up_10d_prob": float(getattr(raw, "up_10d_prob", 0.5)),
        "latest_close": float(getattr(raw, "latest_close", float("nan"))),
        "horizon_forecasts": _serialize_horizon_map(getattr(raw, "horizon_forecasts", {})),
        "market_facts": asdict(market_facts) if isinstance(market_facts, MarketFactsState) else asdict(MarketFactsState()),
        "sentiment": asdict(sentiment) if isinstance(sentiment, MarketSentimentState) else asdict(MarketSentimentState()),
    } if market is not None or raw is not None else asdict(MarketForecastState(
        as_of_date="",
        up_1d_prob=0.5,
        up_5d_prob=0.5,
        up_20d_prob=0.5,
        trend_state="neutral",
        drawdown_risk=0.0,
        volatility_regime="neutral",
        liquidity_stress=0.0,
    ))


def _serialize_stock_state(raw: object) -> dict[str, object]:
    return {
        "symbol": str(getattr(raw, "symbol", "")),
        "sector": str(getattr(raw, "sector", "")),
        "up_1d_prob": float(getattr(raw, "up_1d_prob", 0.5)),
        "up_5d_prob": float(getattr(raw, "up_5d_prob", 0.5)),
        "up_20d_prob": float(getattr(raw, "up_20d_prob", 0.5)),
        "excess_vs_sector_prob": float(getattr(raw, "excess_vs_sector_prob", 0.5)),
        "event_impact_score": float(getattr(raw, "event_impact_score", 0.0)),
        "tradeability_score": float(getattr(raw, "tradeability_score", 0.0)),
        "alpha_score": float(getattr(raw, "alpha_score", 0.0)),
        "tradability_status": str(getattr(raw, "tradability_status", "normal")),
        "up_2d_prob": float(getattr(raw, "up_2d_prob", 0.5)),
        "up_3d_prob": float(getattr(raw, "up_3d_prob", 0.5)),
        "up_10d_prob": float(getattr(raw, "up_10d_prob", 0.5)),
        "latest_close": float(getattr(raw, "latest_close", float("nan"))),
        "horizon_forecasts": _serialize_horizon_map(getattr(raw, "horizon_forecasts", {})),
        "selection_reasons": [str(item) for item in getattr(raw, "selection_reasons", [])],
        "ranking_reasons": [str(item) for item in getattr(raw, "ranking_reasons", [])],
        "risk_flags": [str(item) for item in getattr(raw, "risk_flags", [])],
        "invalidation_rule": str(getattr(raw, "invalidation_rule", "")),
        "action_reason": str(getattr(raw, "action_reason", "")),
        "weight_reason": str(getattr(raw, "weight_reason", "")),
        "blocked_reason": str(getattr(raw, "blocked_reason", "")),
    }


def serialize_composite_state(state: CompositeState) -> dict[str, object]:
    return {
        "market": _serialize_market_state(getattr(state, "market", None)),
        "cross_section": asdict(getattr(state, "cross_section")),
        "sectors": [asdict(item) for item in getattr(state, "sectors", [])],
        "stocks": [_serialize_stock_state(item) for item in getattr(state, "stocks", [])],
        "strategy_mode": str(getattr(state, "strategy_mode", "")),
        "risk_regime": str(getattr(state, "risk_regime", "")),
        "candidate_selection": asdict(getattr(state, "candidate_selection", CandidateSelectionState())),
        "mainlines": [asdict(item) for item in getattr(state, "mainlines", [])],
        "market_info_state": asdict(getattr(state, "market_info_state", InfoAggregateState())),
        "sector_info_states": {
            str(key): asdict(value)
            for key, value in getattr(state, "sector_info_states", {}).items()
        },
        "stock_info_states": {
            str(key): asdict(value)
            for key, value in getattr(state, "stock_info_states", {}).items()
        },
        "capital_flow_state": asdict(getattr(state, "capital_flow_state", CapitalFlowState())),
        "macro_context_state": asdict(getattr(state, "macro_context_state", MacroContextState())),
    }


def build_frozen_daily_state_payload(
    *,
    trajectory: object | None,
    split_mode: str,
    embargo_days: int,
    split_trajectory: Callable[..., tuple[object, object, object]],
) -> dict[str, object]:
    if trajectory is None or not getattr(trajectory, "steps", None):
        return {}
    _, _, holdout = split_trajectory(
        trajectory,
        split_mode=split_mode,
        embargo_days=embargo_days,
    )
    steps = getattr(holdout, "steps", None) or getattr(trajectory, "steps", None) or []
    if not steps:
        return {}
    last_step = steps[-1]
    return {
        "as_of_date": str(last_step.date.date()),
        "next_date": str(last_step.next_date.date()),
        "composite_state": serialize_composite_state(last_step.composite_state),
    }


def load_research_manifest_for_daily(
    *,
    strategy_id: str,
    artifact_root: str,
    run_id: str | None,
    snapshot_path: str | None,
    resolve_manifest_path: Callable[..., Path],
    load_json_dict: Callable[[object], dict[str, object]],
) -> tuple[dict[str, object], Path]:
    manifest_path = resolve_manifest_path(
        strategy_id=strategy_id,
        artifact_root=artifact_root,
        run_id=run_id,
        snapshot_path=snapshot_path,
    )
    payload = load_json_dict(manifest_path)
    if not payload:
        raise FileNotFoundError(
            f"Missing research manifest for daily-run: {manifest_path}. "
            "Run `research-run` first or pass `--allow-retrain`."
        )
    manifest_run_id = str(payload.get("run_id", "")).strip()
    requested_run_id = "" if run_id is None else str(run_id).strip()
    if requested_run_id and manifest_run_id and manifest_run_id != requested_run_id:
        raise ValueError(
            f"run_id mismatch: requested={requested_run_id}, manifest={manifest_run_id} ({manifest_path})"
        )
    manifest_strategy = str(payload.get("strategy_id", "")).strip()
    if manifest_strategy and manifest_strategy != str(strategy_id):
        raise ValueError(
            f"strategy mismatch in manifest: requested={strategy_id}, manifest={manifest_strategy}"
        )
    return payload, manifest_path
