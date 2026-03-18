from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import pandas as pd

from src.application.v2_contracts import CapitalFlowState, CompositeState, InfoItem, MacroContextState
from src.application.v2_external_signal_support import (
    attach_external_signals_to_state,
    build_external_signal_package,
)


@dataclass(frozen=True)
class ExternalSignalRuntimeDependencies:
    build_mainline_states: Callable[..., list[object]]
    stock_policy_score: Callable[[object], float]


def build_external_signal_package_for_date(
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


def attach_external_signals_to_composite_state(
    *,
    state: CompositeState,
    settings: dict[str, object],
    as_of_date: pd.Timestamp,
    info_items: list[InfoItem],
    deps: ExternalSignalRuntimeDependencies,
) -> tuple[CompositeState, dict[str, object]]:
    package = build_external_signal_package_for_date(
        settings=settings,
        as_of_date=as_of_date,
        info_items=info_items,
    )
    enriched_state = attach_external_signals_to_state(
        state=state,
        capital_flow_state=package["capital_flow_state"],
        macro_context_state=package["macro_context_state"],
    )
    refreshed_mainlines = deps.build_mainline_states(
        market=enriched_state.market,
        cross_section=enriched_state.cross_section,
        sectors=enriched_state.sectors,
        stocks=enriched_state.stocks,
        stock_score_fn=deps.stock_policy_score,
        sector_info_states=getattr(enriched_state, "sector_info_states", {}),
        stock_info_states=getattr(enriched_state, "stock_info_states", {}),
        capital_flow_state=enriched_state.capital_flow_state,
        macro_context_state=enriched_state.macro_context_state,
    )
    return replace(enriched_state, mainlines=refreshed_mainlines), package
