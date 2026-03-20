# Formal Generator x Info Ablation

- created_at: 2026-03-19T22:42:53
- base_config: D:\gupiao\config\api_generated300_train_5y.json
- source: local
- training_window_days: 480
- split_mode: purged_wf
- embargo_days: 20

## Holdout Matrix

| combo | generator | info | days | excess annual | IR | MDD | period |
|---|---|---|---:|---:|---:|---:|---|
| v2_leaders__info_off | v2_leaders | off | 159 | -4.91% | -0.309 | -2.18% | 2025-06-23 -> 2026-02-10 |

## Full Cycle Matrix

| combo | generator | info | days | excess annual | IR | MDD | period |
|---|---|---|---:|---:|---:|---:|---|
| v2_leaders__info_off | v2_leaders | off | 892 | 13.32% | 1.025 | -7.37% | 2022-06-14 -> 2026-02-10 |

## Existing References

| name | days | excess annual | IR | period |
|---|---:|---:|---:|---|
| formal_best_holdout_20260316_132232 | 158 | 55.39% | 2.864 | 2025-06-20 -> 2026-02-06 |
| formal_current_holdout_20260319_100152 | 159 | -20.38% | -1.621 | 2025-06-23 -> 2026-02-10 |
| fast_rule_20260318 | 120 | 112.42% | 2.548 | 2025-09-12 -> 2026-03-18 |