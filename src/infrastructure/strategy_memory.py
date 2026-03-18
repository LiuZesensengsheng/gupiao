from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from src.application.v2_contracts import (
    DailyRunResult,
    StrategyMemoryRecall,
    V2BacktestSummary,
    V2CalibrationResult,
    V2PolicyLearningResult,
)

_MAX_RESEARCH_ENTRIES = 24
_MAX_DAILY_ENTRIES = 90
_RECALL_WINDOW = 5


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _memory_path(memory_root: str | Path, strategy_id: str) -> Path:
    root = Path(memory_root)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{str(strategy_id).strip() or 'default'}_memory.json"


def _load_payload(path: Path, strategy_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "strategy_id": str(strategy_id),
            "updated_at": "",
            "research_runs": [],
            "daily_runs": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("strategy_id", str(strategy_id))
    payload.setdefault("updated_at", "")
    payload.setdefault("research_runs", [])
    payload.setdefault("daily_runs", [])
    return payload


def _save_payload(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = _utc_now_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_by_key(rows: list[dict[str, Any]], row: dict[str, Any], *, key_fields: tuple[str, ...], limit: int) -> list[dict[str, Any]]:
    row_key = tuple(str(row.get(field, "")) for field in key_fields)
    filtered = [
        item
        for item in rows
        if tuple(str(item.get(field, "")) for field in key_fields) != row_key
    ]
    filtered.append(row)
    filtered = sorted(
        filtered,
        key=lambda item: (
            str(item.get("as_of_date", "")),
            str(item.get("end_date", "")),
            str(item.get("recorded_at", "")),
            str(item.get("run_id", "")),
        ),
    )
    return filtered[-int(limit):]


def _pct_text(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def _float_text(value: float) -> str:
    return f"{float(value):.2f}"


def _top_labels(counter: Counter[str], *, limit: int = 3) -> list[str]:
    return [label for label, _ in counter.most_common(limit) if str(label).strip()]


def build_memory_recall(payload: dict[str, Any]) -> StrategyMemoryRecall:
    research_runs = [item for item in payload.get("research_runs", []) if isinstance(item, dict)]
    daily_runs = [item for item in payload.get("daily_runs", []) if isinstance(item, dict)]
    recent_daily = daily_runs[-_RECALL_WINDOW:]
    latest_research = research_runs[-1] if research_runs else {}

    exposures = [float(item.get("target_exposure", 0.0)) for item in recent_daily]
    average_target_exposure = sum(exposures) / len(exposures) if exposures else 0.0
    exposure_trend = exposures[-1] - exposures[0] if len(exposures) >= 2 else 0.0
    rebalance_ratio = (
        sum(1 for item in recent_daily if bool(item.get("rebalance_now", False))) / len(recent_daily)
        if recent_daily
        else 0.0
    )

    symbol_counter: Counter[str] = Counter()
    sector_counter: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    positive_counter: Counter[str] = Counter()
    event_risk_counter: Counter[str] = Counter()
    catalyst_counter: Counter[str] = Counter()
    flow_regimes: list[str] = []
    macro_risk_counter: Counter[str] = Counter()
    for item in recent_daily:
        symbol_counter.update([str(x) for x in item.get("top_symbols", []) if str(x).strip()])
        sector_counter.update([str(x) for x in item.get("top_sectors", []) if str(x).strip()])
        risk_counter.update([str(x) for x in item.get("risk_tags", []) if str(x).strip()])
        positive_counter.update([str(x) for x in item.get("positive_tags", []) if str(x).strip()])
        event_risk_counter.update([str(x) for x in item.get("event_risk_tags", []) if str(x).strip()])
        catalyst_counter.update([str(x) for x in item.get("catalyst_tags", []) if str(x).strip()])
        if str(item.get("flow_regime", "")).strip():
            flow_regimes.append(str(item.get("flow_regime", "")).strip())
        if str(item.get("macro_risk_level", "")).strip():
            macro_risk_counter.update([str(item.get("macro_risk_level", "")).strip()])

    recurring_symbols = _top_labels(symbol_counter)
    recurring_sectors = _top_labels(sector_counter)
    recurring_risk_tags = _top_labels(risk_counter)
    recurring_positive_tags = _top_labels(positive_counter)
    recurring_event_risk_tags = _top_labels(event_risk_counter)
    recurring_catalyst_tags = _top_labels(catalyst_counter)
    recent_flow_regimes = [label for label in flow_regimes[-3:] if str(label).strip()]
    recurring_macro_risk_levels = _top_labels(macro_risk_counter)

    narrative: list[str] = []
    if latest_research:
        gate_text = "通过" if bool(latest_research.get("release_gate_passed", False)) else "未通过"
        narrative.append(
            "最近一次研究 "
            f"run_id={latest_research.get('run_id', 'NA')}，"
            f"超额年化 {_pct_text(float(latest_research.get('excess_annual_return', 0.0)))}，"
            f"IR {_float_text(float(latest_research.get('information_ratio', 0.0)))}，"
            f"release gate {gate_text}。"
        )
    if recent_daily:
        if exposure_trend > 0.01:
            trend_text = "近几次仓位有上调倾向"
        elif exposure_trend < -0.01:
            trend_text = "近几次仓位有下调倾向"
        else:
            trend_text = "近几次仓位整体稳定"
        narrative.append(
            f"近 {len(recent_daily)} 次日运行平均目标仓位 {_pct_text(average_target_exposure)}，"
            f"调仓触发占比 {_pct_text(rebalance_ratio)}，{trend_text}。"
        )
    if recurring_symbols:
        narrative.append(f"高频关注标的: {', '.join(recurring_symbols)}。")
    if recurring_risk_tags:
        narrative.append(f"重复出现的风险标签: {', '.join(recurring_risk_tags)}。")
    if recurring_positive_tags:
        narrative.append(f"持续出现的正向线索: {', '.join(recurring_positive_tags)}。")
    if recurring_event_risk_tags:
        narrative.append(f"高频事件风险: {', '.join(recurring_event_risk_tags)}。")
    if recurring_catalyst_tags:
        narrative.append(f"高频催化标签: {', '.join(recurring_catalyst_tags)}。")
    if recent_flow_regimes:
        narrative.append(f"近期资金状态: {', '.join(recent_flow_regimes)}。")
    if recurring_macro_risk_levels:
        narrative.append(f"宏观风险持续性: {', '.join(recurring_macro_risk_levels)}。")

    return StrategyMemoryRecall(
        memory_path=str(payload.get("memory_path", "")),
        updated_at=str(payload.get("updated_at", "")),
        latest_research_run_id=str(latest_research.get("run_id", "")),
        latest_research_end_date=str(latest_research.get("end_date", "")),
        latest_research_release_gate_passed=bool(latest_research.get("release_gate_passed", False)),
        latest_research_excess_annual_return=float(latest_research.get("excess_annual_return", 0.0)),
        latest_research_information_ratio=float(latest_research.get("information_ratio", 0.0)),
        recent_daily_run_count=int(len(recent_daily)),
        average_target_exposure=float(average_target_exposure),
        exposure_trend=float(exposure_trend),
        rebalance_ratio=float(rebalance_ratio),
        recurring_symbols=recurring_symbols,
        recurring_sectors=recurring_sectors,
        recurring_risk_tags=recurring_risk_tags,
        recurring_positive_tags=recurring_positive_tags,
        recurring_event_risk_tags=recurring_event_risk_tags,
        recurring_catalyst_tags=recurring_catalyst_tags,
        recent_flow_regimes=recent_flow_regimes,
        recurring_macro_risk_levels=recurring_macro_risk_levels,
        narrative=narrative,
    )


def remember_research_run(
    *,
    memory_root: str | Path,
    strategy_id: str,
    run_id: str,
    baseline: V2BacktestSummary,
    calibration: V2CalibrationResult,
    learning: V2PolicyLearningResult,
    release_gate_passed: bool,
    universe_id: str = "",
    universe_tier: str = "",
    universe_size: int = 0,
    external_signal_version: str = "",
    external_signal_enabled: bool = False,
) -> Path:
    path = _memory_path(memory_root, strategy_id)
    payload = _load_payload(path, strategy_id)
    payload["memory_path"] = str(path)
    row = {
        "recorded_at": _utc_now_iso(),
        "run_id": str(run_id),
        "end_date": str(learning.learned.end_date or calibration.calibrated.end_date or baseline.end_date),
        "universe_id": str(universe_id),
        "universe_tier": str(universe_tier),
        "universe_size": int(universe_size),
        "release_gate_passed": bool(release_gate_passed),
        "baseline_excess_annual_return": float(baseline.excess_annual_return),
        "calibrated_excess_annual_return": float(calibration.calibrated.excess_annual_return),
        "excess_annual_return": float(learning.learned.excess_annual_return),
        "information_ratio": float(learning.learned.information_ratio),
        "max_drawdown": float(learning.learned.max_drawdown),
        "annual_return": float(learning.learned.annual_return),
        "avg_turnover": float(learning.learned.avg_turnover),
        "external_signal_version": str(external_signal_version),
        "external_signal_enabled": bool(external_signal_enabled),
    }
    payload["research_runs"] = _upsert_by_key(
        list(payload.get("research_runs", [])),
        row,
        key_fields=("run_id",),
        limit=_MAX_RESEARCH_ENTRIES,
    )
    _save_payload(path, payload)
    return path


def remember_daily_run(
    *,
    memory_root: str | Path,
    result: DailyRunResult,
) -> DailyRunResult:
    strategy_id = str(result.snapshot.strategy_id)
    path = _memory_path(memory_root, strategy_id)
    payload = _load_payload(path, strategy_id)
    payload["memory_path"] = str(path)

    target_weights = sorted(
        result.policy_decision.symbol_target_weights.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    top_symbols = [str(symbol) for symbol, weight in target_weights if float(weight) > 0.0][:5]
    sector_weights = sorted(
        result.policy_decision.sector_budgets.items(),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    top_sectors = [str(sector) for sector, weight in sector_weights if float(weight) > 0.0][:3]
    risk_tags = [
        str(item.event_tag)
        for item in result.top_negative_info_events
        if str(item.event_tag).strip()
    ] or [str(note) for note in result.policy_decision.risk_notes if str(note).strip()]
    positive_tags = [
        str(item.target_name or item.target)
        for item in result.top_positive_info_signals
        if str(item.target_name or item.target).strip()
    ]
    event_risk_tags = [
        str(item.event_tag)
        for item in result.top_negative_info_events
        if str(item.event_tag).strip()
    ]
    catalyst_tags = [
        str(item.target_name or item.target)
        for item in result.top_positive_info_signals
        if str(item.target_name or item.target).strip()
    ]

    row = {
        "recorded_at": _utc_now_iso(),
        "run_id": str(result.run_id or result.snapshot.run_id),
        "as_of_date": str(result.composite_state.market.as_of_date),
        "strategy_mode": str(result.composite_state.strategy_mode),
        "risk_regime": str(result.composite_state.risk_regime),
        "target_exposure": float(result.policy_decision.target_exposure),
        "target_position_count": int(result.policy_decision.target_position_count),
        "rebalance_now": bool(result.policy_decision.rebalance_now),
        "rebalance_intensity": float(result.policy_decision.rebalance_intensity),
        "turnover_cap": float(result.policy_decision.turnover_cap),
        "top_symbols": top_symbols,
        "top_sectors": top_sectors,
        "risk_tags": risk_tags[:5],
        "positive_tags": positive_tags[:5],
        "event_risk_tags": event_risk_tags[:5],
        "catalyst_tags": catalyst_tags[:5],
        "flow_regime": str(result.composite_state.capital_flow_state.flow_regime),
        "macro_risk_level": str(result.composite_state.macro_context_state.macro_risk_level),
        "trade_actions": [
            {
                "symbol": str(action.symbol),
                "action": str(action.action),
                "target_weight": float(action.target_weight),
                "delta_weight": float(action.delta_weight),
            }
            for action in result.trade_actions[:10]
        ],
    }
    payload["daily_runs"] = _upsert_by_key(
        list(payload.get("daily_runs", [])),
        row,
        key_fields=("as_of_date", "run_id"),
        limit=_MAX_DAILY_ENTRIES,
    )
    _save_payload(path, payload)
    recall = build_memory_recall(payload)
    recall = replace(recall, memory_path=str(path))
    return replace(result, memory_path=str(path), memory_recall=recall)
