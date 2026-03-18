from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.application.v2_contracts import V2BacktestSummary


def pass_release_gate(
    *,
    baseline: V2BacktestSummary,
    candidate: V2BacktestSummary,
    threshold: dict[str, float],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(candidate.excess_annual_return) <= float(threshold["excess_annual_return_min"]):
        reasons.append("holdout excess_annual_return <= 0")
    if float(candidate.information_ratio) <= float(threshold["information_ratio_min"]):
        reasons.append("holdout information_ratio <= 0.30")
    drawdown_diff = abs(float(candidate.max_drawdown)) - abs(float(baseline.max_drawdown))
    if drawdown_diff > float(threshold["max_drawdown_worse_limit"]):
        reasons.append("holdout max_drawdown worse than baseline by > 5pp")
    return (len(reasons) == 0), reasons


def tier_latest_manifest_path(*, artifact_root: str, strategy_id: str, universe_tier: str) -> Path:
    suffix = str(universe_tier).strip() or "custom"
    if not str(universe_tier).strip():
        return Path(str(artifact_root)) / str(strategy_id) / "latest_research_manifest.json"
    return Path(str(artifact_root)) / str(strategy_id) / f"latest_research_manifest.{suffix}.json"


def tier_latest_policy_path(*, artifact_root: str, strategy_id: str, universe_tier: str) -> Path:
    suffix = str(universe_tier).strip() or "custom"
    if not str(universe_tier).strip():
        return Path(str(artifact_root)) / str(strategy_id) / "latest_policy_model.json"
    return Path(str(artifact_root)) / str(strategy_id) / f"latest_policy_model.{suffix}.json"


def summary_from_payload(template: V2BacktestSummary, payload: dict[str, object]) -> V2BacktestSummary:
    base = asdict(template)
    return V2BacktestSummary(
        **{
            **base,
            **{k: v for k, v in payload.items() if k in base},
        }
    )


def load_backtest_payload_from_manifest(
    manifest_payload: dict[str, object],
    manifest_path: Path,
    *,
    path_from_manifest_entry: Callable[[object], Path | None],
    load_json_dict: Callable[[object], dict[str, object]],
) -> dict[str, object]:
    backtest_path = path_from_manifest_entry(manifest_payload.get("backtest_summary"))
    if backtest_path is None:
        return {}
    loaded = load_json_dict(backtest_path)
    return loaded if isinstance(loaded, dict) else {}


def load_backtest_payload_for_run(
    *,
    artifact_root: str,
    strategy_id: str,
    run_id: str,
    load_json_dict: Callable[[object], dict[str, object]],
) -> dict[str, object]:
    if not str(run_id).strip():
        return {}
    backtest_path = Path(str(artifact_root)) / str(strategy_id) / str(run_id).strip() / "backtest_summary.json"
    loaded = load_json_dict(backtest_path)
    return loaded if isinstance(loaded, dict) else {}


def pass_default_switch_gate(
    *,
    baseline_reference: V2BacktestSummary,
    candidate: V2BacktestSummary,
    threshold: dict[str, float],
) -> tuple[bool, list[str], dict[str, float]]:
    reasons: list[str] = []
    excess_delta = float(candidate.excess_annual_return) - float(baseline_reference.excess_annual_return)
    ir_delta = float(candidate.information_ratio) - float(baseline_reference.information_ratio)
    drawdown_diff = abs(float(candidate.max_drawdown)) - abs(float(baseline_reference.max_drawdown))
    if not (
        excess_delta >= float(threshold["excess_annual_return_delta_min"])
        or ir_delta >= float(threshold["information_ratio_delta_min"])
    ):
        reasons.append("switch gate unmet: excess_annual_return < +2pp and information_ratio < +0.10 vs baseline reference")
    if drawdown_diff > float(threshold["max_drawdown_worse_limit"]):
        reasons.append("switch gate unmet: max_drawdown worse than baseline reference by > 2pp")
    deltas = {
        "excess_annual_return_delta": float(excess_delta),
        "information_ratio_delta": float(ir_delta),
        "max_drawdown_diff": float(drawdown_diff),
    }
    return (len(reasons) == 0), reasons, deltas
