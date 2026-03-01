from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class WorkflowStage:
    name: str
    purpose: str
    produces: str


@dataclass(frozen=True)
class WorkflowBlueprint:
    name: str
    stages: List[WorkflowStage]


def build_research_run_blueprint() -> WorkflowBlueprint:
    return WorkflowBlueprint(
        name="research-run",
        stages=[
            WorkflowStage("prepare-data", "normalize and cache raw market facts", "normalized data artifacts"),
            WorkflowStage("build-dataset", "freeze training samples and labels", "dataset artifact"),
            WorkflowStage("train-forecast-models", "fit market/sector/stock/cross-section models", "forecast model artifacts"),
            WorkflowStage("compose-state", "validate state composition contracts", "state-composition diagnostics"),
            WorkflowStage("calibrate-policy", "fit or tune policy parameters against a fixed forecast layer", "policy artifact"),
            WorkflowStage("run-backtest", "simulate execution with frozen strategy snapshot", "backtest artifact"),
            WorkflowStage("publish-strategy-snapshot", "publish immutable strategy config for daily production", "strategy snapshot"),
        ],
    )


def build_daily_run_blueprint() -> WorkflowBlueprint:
    return WorkflowBlueprint(
        name="daily-run",
        stages=[
            WorkflowStage("prepare-data", "update local market facts and caches", "fresh normalized data"),
            WorkflowStage("load-strategy-snapshot", "load the published production strategy", "strategy snapshot"),
            WorkflowStage("generate-forecast-state", "score the latest market state with frozen models", "forecast state"),
            WorkflowStage("compose-state", "merge all forecast outputs into a single investable state", "composite state"),
            WorkflowStage("apply-policy", "turn state into target exposure and target weights", "policy decision"),
            WorkflowStage("build-trade-plan", "compare current holdings to target portfolio", "trade plan"),
            WorkflowStage("render-report", "render markdown/html outputs", "daily report"),
        ],
    )


def describe_v2_stack() -> str:
    return (
        "V2 = forecast layer (market/sector/stock/style-flow-breadth) "
        "+ policy layer (exposure/concentration/rebalance/trade plan)."
    )
