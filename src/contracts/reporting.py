from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DailyReportViewModel:
    strategy_id: str
    run_id: str
    strategy_mode: str
    risk_regime: str
    as_of_date: str
    next_session: str
    external_signal_enabled: bool
    metadata: dict[str, object] = field(default_factory=dict)
    market_summary: dict[str, object] = field(default_factory=dict)
    market_forecasts: list[dict[str, object]] = field(default_factory=list)
    dynamic_universe: dict[str, object] = field(default_factory=dict)
    memory_summary: dict[str, object] = field(default_factory=dict)
    external_signals: dict[str, object] = field(default_factory=dict)
    mainlines: list[dict[str, object]] = field(default_factory=list)
    top_recommendations: list[dict[str, object]] = field(default_factory=list)
    explanation_cards: list[dict[str, object]] = field(default_factory=list)
    prediction_review: dict[str, object] = field(default_factory=dict)
    trade_actions: list[dict[str, object]] = field(default_factory=list)
    policy: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchReportViewModel:
    strategy_id: str
    run_id: str
    release_gate_passed: bool
    artifacts: dict[str, object] = field(default_factory=dict)
    baseline: dict[str, object] = field(default_factory=dict)
    calibration: dict[str, object] = field(default_factory=dict)
    learned: dict[str, object] = field(default_factory=dict)
    comparison_metrics: list[dict[str, object]] = field(default_factory=list)
    horizon_metrics: list[dict[str, object]] = field(default_factory=list)
    calibration_trials: list[dict[str, object]] = field(default_factory=list)
    learning_model: dict[str, object] = field(default_factory=dict)
    info_shadow_summary: dict[str, object] = field(default_factory=dict)
