from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DailyReportViewModel:
    strategy_id: str
    run_id: str
    strategy_mode: str
    risk_regime: str
    external_signal_enabled: bool
    trade_plan: list[dict[str, object]] = field(default_factory=list)
    policy: dict[str, object] = field(default_factory=dict)
    market: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchReportViewModel:
    strategy_id: str
    run_id: str
    release_gate_passed: bool
    baseline: dict[str, object] = field(default_factory=dict)
    calibration: dict[str, object] = field(default_factory=dict)
    learned: dict[str, object] = field(default_factory=dict)
