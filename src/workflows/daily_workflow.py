from __future__ import annotations

from src.application import v2_services as legacy
from src.contracts.runtime import DailyRunOptions


def run_daily_v2_live(
    *,
    options: DailyRunOptions | None = None,
    **kwargs: object,
):
    resolved = options or DailyRunOptions.from_kwargs(**kwargs)
    return legacy._run_daily_v2_live_impl(**resolved.workflow_kwargs())
