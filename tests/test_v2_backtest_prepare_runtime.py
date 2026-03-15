from __future__ import annotations

from types import SimpleNamespace

from src.application.v2_backtest_prepare_runtime import split_research_trajectory


def test_split_research_trajectory_preserves_prepared_payload() -> None:
    trajectory = SimpleNamespace(
        prepared=SimpleNamespace(tag="prepared"),
        steps=[SimpleNamespace(idx=i) for i in range(10)],
    )

    train, validation, holdout = split_research_trajectory(
        trajectory,
        split_mode="purged_wf",
        embargo_days=1,
    )

    assert [len(part.steps) for part in (train, validation, holdout)] == [6, 1, 1]
    assert [part.prepared for part in (train, validation, holdout)] == [trajectory.prepared] * 3


def test_split_research_trajectory_handles_tiny_inputs() -> None:
    trajectory = SimpleNamespace(
        prepared=SimpleNamespace(tag="prepared"),
        steps=[SimpleNamespace(idx=0), SimpleNamespace(idx=1)],
    )

    train, validation, holdout = split_research_trajectory(trajectory, split_mode="purged_wf", embargo_days=3)

    assert len(train.steps) == 0
    assert len(validation.steps) == 0
    assert len(holdout.steps) == 2
