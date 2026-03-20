# V2 研究回测报告

- 策略ID: swing_v2
- run_id: 20260319_234617
- release gate: pending
- best_score: 0.7855

## 核心指标

| 指标 | baseline | calibrated | learned |
|---|---:|---:|---:|
| 年化收益 | 40.5% | 49.9% | 19.8% |
| 超额年化 | 3.1% | 10.1% | -12.1% |
| 最大回撤 | -4.4% | -4.8% | -6.3% |
| 信息比率 | 0.287 | 0.698 | -0.854 |

## Horizon Metrics

| 方案 | 周期 | RankIC | 头部分层收益 | 头尾价差 | TopK命中率 |
|---|---|---:|---:|---:|---:|
| baseline | 1d | -0.002 | 0.4% | -0.4% | 51.1% |
| baseline | 2d | -0.022 | 0.9% | -0.9% | 48.1% |
| baseline | 3d | -0.025 | 1.3% | -1.4% | 49.2% |
| baseline | 5d | -0.043 | 2.0% | -2.5% | 53.0% |
| baseline | 20d | -0.063 | -2.0% | -3.5% | 41.4% |
| calibrated | 1d | -0.002 | 0.4% | -0.4% | 51.1% |
| calibrated | 2d | -0.022 | 0.9% | -0.9% | 48.1% |
| calibrated | 3d | -0.025 | 1.3% | -1.4% | 49.2% |
| calibrated | 5d | -0.043 | 2.0% | -2.5% | 53.0% |
| calibrated | 20d | -0.063 | -2.0% | -3.5% | 41.4% |
| learned | 1d | -0.002 | 0.4% | -0.4% | 51.1% |
| learned | 2d | -0.022 | 0.9% | -0.9% | 48.1% |
| learned | 3d | -0.025 | 1.3% | -1.4% | 49.2% |
| learned | 5d | -0.043 | 2.0% | -2.5% | 53.0% |
| learned | 20d | -0.063 | -2.0% | -3.5% | 41.4% |

## Validation Trials

| 排名 | risk_on_exposure | risk_on_positions | risk_on_turnover_cap | annual_return | benchmark_annual_return | excess_annual_return | information_ratio | max_drawdown | score |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.0% | 4 | 40.0% | 58.2% | 38.6% | 14.1% | 0.760 | -7.8% | 0.7855 |
| 2 | 75.0% | 4 | 34.0% | 58.2% | 38.6% | 14.1% | 0.760 | -7.8% | 0.7855 |
| 3 | 75.0% | 4 | 45.0% | 58.2% | 38.6% | 14.1% | 0.760 | -7.8% | 0.7855 |
| 4 | 85.0% | 4 | 40.0% | 58.2% | 38.6% | 14.1% | 0.760 | -7.8% | 0.7855 |
| 5 | 85.0% | 4 | 34.0% | 58.2% | 38.6% | 14.1% | 0.760 | -7.8% | 0.7855 |
| 6 | 85.0% | 4 | 45.0% | 58.2% | 38.6% | 14.1% | 0.760 | -7.8% | 0.7855 |
| 7 | 85.0% | 4 | 40.0% | 40.1% | 38.6% | 1.0% | 0.150 | -9.0% | 0.0417 |

## Learned Policy

- feature_names: mkt_up_1d, mkt_up_20d, mkt_drawdown_risk, mkt_liquidity_stress, cross_fund_flow, cross_margin_risk_on, cross_breadth, cross_leader_participation, cross_weak_ratio, top_sector_up_20d, top_sector_relative_strength, top_stock_up_20d, top_stock_tradeability, top_stock_excess_vs_sector, alpha_headroom, alpha_breadth, alpha_top_score, alpha_avg_top3, alpha_median_score, candidate_shortlist_ratio, candidate_shortlist_size_norm, candidate_alpha_breadth, candidate_durability
- train_rows: 158
- train_r2_exposure: 0.059
- train_r2_positions: 0.056
- train_r2_turnover: 0.034

## Artifacts

- run_dir: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617`
- run_id: `20260319_234617`
- baseline_reference_run_id: `20260308_211808`
- universe_tier: ``
- universe_id: `dynamic_300`
- universe_size: `19`
- source_universe_manifest_path: `D:\gupiao_repro_912874e\config\universe_all_a_3y_local_ready_nost_no_kc_cy_stable3y.json`
- info_manifest: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\info_manifest.json`
- info_shadow_report: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\info_shadow_report.json`
- info_hash: `6472e7bf05cb7051370a22836b202074efef8a019ce6c9af2996995b02237104`
- info_item_count: `0`
- info_shadow_enabled: `false`
- external_signal_manifest: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\external_signal_manifest.json`
- external_signal_version: `v1`
- external_signal_enabled: `true`
- generator_manifest: `D:\gupiao\artifacts\repro_912874e_cache\universe_catalog\dynamic_300_1e082ca7b73b.generator.json`
- generator_version: `dynamic_universe_v2_leaders`
- generator_hash: `1e082ca7b73b80911cd8b38eff907719e7c0411fdb48a0dda3340b84f887ed3f`
- coarse_pool_size: `19`
- refined_pool_size: `19`
- selected_pool_size: `19`
- use_us_index_context: `true`
- us_index_source: `akshare`
- dataset_manifest: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\dataset_manifest.json`
- policy_calibration: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\policy_calibration.json`
- learned_policy_model: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\learned_policy_model.json`
- forecast_models_manifest: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\forecast_models_manifest.json`
- frozen_forecast_bundle: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\frozen_forecast_bundle.json`
- frozen_daily_state: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\frozen_daily_state.json`
- backtest_summary: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\backtest_summary.json`
- consistency_checklist: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\consistency_checklist.json`
- rolling_oos_report: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\rolling_oos_report.json`
- research_manifest: `D:\gupiao\artifacts\repro_912874e\swing_v2\20260319_234617\research_manifest.json`
- published_policy_model: `D:\gupiao\artifacts\repro_912874e\swing_v2\latest_policy_model.json`
- strategy_memory: `D:\gupiao\artifacts\repro_912874e\memory\swing_v2_memory.json`
- capital_flow_snapshot: `{"northbound_net_flow": 0.0, "margin_balance_change": 0.0, "turnover_heat": 0.5, "large_order_bias": 0.0, "flow_regime": "neutral"}`
- macro_context_snapshot: `{"style_regime": "balanced", "commodity_pressure": 0.0, "fx_pressure": 0.0, "index_breadth_proxy": 0.5, "macro_risk_level": "low"}`
- release_gate_passed: `false`
- default_switch_gate_passed: `false`
- snapshot_hash: `a878a93c1ed9a67d39bf0151bd2d43716acd060d518081f9be93632ab7128967`
- config_hash: `1b92cb414a398c60d77b1f7f261989f17bee46e6ef49db008992942a3fae7559`