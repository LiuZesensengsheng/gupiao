from __future__ import annotations

from src.application import v2_services as legacy
from src.contracts.runtime import ResearchMatrixOptions, ResearchRunOptions


def run_v2_research_workflow(
    *,
    options: ResearchRunOptions | None = None,
    **kwargs: object,
):
    resolved = options or ResearchRunOptions.from_kwargs(**kwargs)
    return legacy._run_v2_research_workflow_impl(**resolved.workflow_kwargs())


def run_v2_research_matrix(
    *,
    options: ResearchMatrixOptions | None = None,
    **kwargs: object,
):
    resolved = options or ResearchMatrixOptions.from_kwargs(**kwargs)
    return legacy._run_v2_research_matrix_impl(**resolved.workflow_kwargs())
