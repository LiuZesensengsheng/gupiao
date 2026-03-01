# Quant Research Architecture V2

## Goal

V2 splits the current monolithic daily pipeline into two explicit layers:

1. Forecast layer: predict what is likely to happen.
2. Policy layer: decide what to do under those predicted states.

This design targets:

- fast daily runs
- isolated backtests
- explicit model artifacts
- stable bounded contexts
- small, deterministic tests

## Core Design

The current `daily` flow mixes:

- data loading
- feature building
- model fitting
- parameter search
- backtest
- acceptance checks
- trading plan generation

V2 separates them into research-time and production-time workflows.

### Research-Time Workflow

```text
prepare-data
-> build-dataset
-> train-forecast-models
-> compose-state
-> calibrate-policy
-> run-backtest
-> publish-strategy-snapshot
```

### Production-Time Workflow

```text
prepare-data
-> load-published-strategy-snapshot
-> generate-forecast-state
-> compose-state
-> apply-policy
-> build-trade-plan
-> render-report
```

Production does not:

- run grid search
- retrain all models
- run full historical backtests

## Bounded Contexts

### 1. Market Data Context

Responsibility:

- ingest and normalize market facts
- expose data as stable time-series objects

Inputs:

- index bars
- stock bars
- sector membership
- margin data
- news events

Outputs:

- `BarSeries`
- `NewsEvent`
- `SectorMembershipSnapshot`
- `MarginSnapshot`

### 2. Research Context

Responsibility:

- build datasets
- define labels
- train and validate models
- publish versioned artifacts

Outputs:

- dataset artifact
- forecast model artifact
- calibration artifact
- experiment report

### 3. Forecast Context

Responsibility:

- predict structured states, not trades

V2 forecast layer is split into four model families:

1. Market forecast
2. Sector forecast
3. Stock forecast
4. Style / flow / breadth forecast

### 4. Strategy Context

Responsibility:

- compose the forecast outputs into a single investable state
- decide exposure, concentration, turnover mode, and target weights

This is the policy layer.

### 5. Portfolio Context

Responsibility:

- turn policy outputs into constrained target positions

Includes:

- weight allocation
- concentration limits
- liquidity guards
- rebalance thresholds

### 6. Execution Simulation Context

Responsibility:

- simulate trading under a fixed strategy snapshot
- estimate turnover, costs, and realized path

The simulator must not refit models or mutate strategy logic.

### 7. Reporting Context

Responsibility:

- render human-readable outputs
- never own business decisions

## Forecast Layer

Forecast models answer: what is happening and what is likely next?

### A. Market Forecast

Purpose:

- determine whether the book should be aggressive, neutral, or defensive

Recommended outputs:

- `mkt_up_1d_prob`
- `mkt_up_5d_prob`
- `mkt_up_20d_prob`
- `mkt_trend_state`
- `mkt_drawdown_risk`
- `mkt_volatility_regime`
- `mkt_liquidity_stress`

### B. Sector Forecast

Purpose:

- decide where capital should concentrate

Recommended outputs:

- `sector_up_5d_prob`
- `sector_up_20d_prob`
- `sector_relative_strength`
- `sector_rotation_speed`
- `sector_crowding_score`

### C. Stock Forecast

Purpose:

- rank candidates within their sector and market regime

Recommended outputs:

- `stock_up_1d_prob`
- `stock_up_5d_prob`
- `stock_up_20d_prob`
- `stock_excess_vs_sector_prob`
- `stock_event_impact_score`
- `stock_tradeability_score`

### D. Style / Flow / Breadth Forecast

Purpose:

- capture cross-sectional market state that should shape policy, not direct picks

Recommended outputs:

- `large_vs_small_bias`
- `growth_vs_value_bias`
- `fund_flow_strength`
- `margin_risk_on_score`
- `breadth_strength`
- `leader_participation`
- `weak_stock_ratio`

## State Composition

Raw forecast outputs are too granular for policy logic.

V2 introduces a `CompositeState` that summarizes:

- market regime
- style regime
- risk regime
- sector preference map
- stock conviction map

Example policy chain:

```text
market forecast -> total exposure
style/flow/breadth -> strategy mode
sector forecast -> sector budget
stock forecast -> final selection and weighting
```

## Policy Layer

Policy answers: given the current forecast state, what should the strategy do?

Recommended policy outputs:

- `target_exposure`
- `target_position_count`
- `sector_budgets`
- `symbol_target_weights`
- `rebalance_now`
- `rebalance_intensity`
- `intraday_t_allowed`
- `turnover_cap`
- `risk_notes`

### Important Rule

Policy is allowed to be:

- rule-driven
- parameterized
- learned

But it should consume forecast outputs as inputs.

Policy should not directly consume the full raw feature matrix in production mode.

## Strategy Snapshot

Daily production must run from a published strategy snapshot.

A snapshot contains:

- universe definition
- feature set version
- forecast model versions
- policy version
- risk limits
- execution assumptions

Backtests and daily runs must both reference the same snapshot format.

## Artifacts

V2 stores every expensive intermediate result.

### Artifact Types

- `data/normalized/*.parquet`
- `artifacts/datasets/<dataset_id>/`
- `artifacts/models/<model_id>/`
- `artifacts/policies/<policy_id>/`
- `artifacts/strategies/<strategy_id>/`
- `artifacts/backtests/<backtest_id>/`

### Cache Keys

Artifacts should be keyed by:

- universe id
- date range
- feature set version
- label version
- model spec hash
- strategy spec hash

This prevents repeated full retraining during daily runs.

## CLI Shape

V2 should live alongside the current CLI during migration.

Recommended commands:

```bash
python3 run_v2.py describe
python3 run_v2.py research-run --strategy swing_v2
python3 run_v2.py daily-run --strategy swing_v2
```

### `research-run`

Stages:

1. prepare-data
2. build-dataset
3. train-forecast-models
4. compose-state design validation
5. calibrate policy
6. run backtest
7. publish strategy snapshot

### `daily-run`

Stages:

1. prepare-data
2. load strategy snapshot
3. generate latest forecast state
4. compose state
5. apply policy
6. build trade plan
7. render report

## Testing Strategy

Tests must be able to run without network access.

### Unit Tests

Fast tests for:

- regime rules
- allocators
- turnover guards
- execution guards
- policy decisions

### Dataset Tests

Use tiny fixtures to verify:

- feature columns
- label alignment
- no future leakage

### Model Contract Tests

Verify:

- model artifact can train
- predict shape is stable
- calibration output is bounded

### Backtest Scenario Tests

Construct tiny deterministic scenarios for:

- suspended stocks
- limit-up / limit-down
- oversell prevention
- rebalance caps

### Integration Tests

A tiny end-to-end path should be able to run:

```text
fixture data -> forecast -> policy -> trade plan -> report
```

without retraining large models.

## Migration Plan

### Phase 1

- Keep `run_api.py`
- Add `run_v2.py`
- Introduce forecast/policy contracts
- Stop adding logic to the old monolithic `daily` path

### Phase 2

- Move state logic from `src/domain/policies.py` into a dedicated policy engine
- Move forecast outputs into typed contracts
- Freeze daily production to published strategy snapshots

### Phase 3

- Split current backtesting into a pure execution simulator
- Split training into dataset + model jobs

### Phase 4

- Retire the legacy `daily` command once V2 reaches parity

## Non-Goals

V2 does not require:

- immediate replacement of all models
- immediate replacement of all features
- deep model complexity on day one

The first win is boundary clarity.

Replace the workflow first, then improve models safely.
