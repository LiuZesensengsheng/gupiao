from tests.v2_research_cases import (
    test_backtest_summary_accepts_multi_horizon_metrics_payload,
    test_backtest_summary_carries_cross_section_metrics,
    test_calibrate_v2_policy_emits_progress_updates,
    test_calibrate_v2_policy_runs_expanded_validation_grid,
    test_calibrate_v2_policy_uses_two_stage_screening_when_trajectory_is_long,
    test_policy_objective_prefers_excess_and_ir_over_raw_return,
    test_research_workflow_emits_stage_progress,
    test_research_workflow_light_mode_skips_heavy_stages,
    test_research_workflow_passes_deep_backend,
    test_research_workflow_passes_training_window_days_to_trajectory_builder,
    test_research_workflow_reuses_single_trajectory_for_all_stages,
)
