# RFC: Quant Architecture Refactor

- Status: Proposed
- Date: 2026-03-14
- Owner: Codex + project maintainer
- Scope: `V2` research and daily production architecture

## Problem

Current `V2` has the right direction but an incomplete landing.

The repository already distinguishes:

- research workflow
- daily production workflow
- published artifacts
- reporting outputs

But the implementation boundaries are still blurry.

The clearest hotspot is [`src/application/v2_services.py`](/D:/gupiao/src/application/v2_services.py), which currently mixes:

- runtime config resolution
- research orchestration
- daily-run orchestration
- cache handling
- frozen snapshot loading
- policy learning
- artifact publishing
- summary generation

This creates four real problems:

1. Architecture intent exists in docs, but not yet in code boundaries.
2. Research and daily-run share logic implicitly instead of through stable contracts.
3. Performance optimization is hard because data preparation and scoring boundaries are not explicit.
4. Adding a second strategy or forecast backend will increase complexity faster than current structure can absorb.

This RFC proposes a practical refactor path that keeps the current `V2` direction, but makes it maintainable.

## Constraints

This proposal assumes the following project constraints:

- The project is a stock quant system, mainly batch-oriented, not a low-latency execution engine.
- The most important workflows are still `research-run` and `daily-run`.
- Existing tests should remain useful; full rewrite is not acceptable.
- Current code already has value and should be evolved incrementally.
- Artifact compatibility matters because research output must be consumed by production.
- The project needs both:
  - iteration speed for research
  - stable daily production execution

Non-goals:

- introducing microservices
- introducing distributed compute infrastructure
- redesigning strategy logic from scratch
- replacing all existing paths in one step

## Complexity

### Essential Complexity

These are normal and necessary in a quant system:

- market data ingestion and normalization
- feature engineering
- forecast model training and scoring
- state composition
- policy and portfolio construction
- backtesting and execution simulation
- artifact publication
- human-readable reporting

### Accidental Complexity

These are current architecture problems, not inherent domain complexity:

- one oversized orchestrator module
- duplicated parameter surfaces in CLI
- partial migration where support modules exist but core logic still lives in the monolith
- unclear ownership of cache boundaries
- business logic and artifact logic coupled in the same file
- reporting modules growing as passive copies of orchestration outputs instead of stable view-model consumers

### Current Hotspots

Repository hotspots that justify this RFC:

- [`src/application/v2_services.py`](/D:/gupiao/src/application/v2_services.py)
- [`src/interfaces/cli/run_v2_cli.py`](/D:/gupiao/src/interfaces/cli/run_v2_cli.py)
- [`src/interfaces/presenters/html_dashboard.py`](/D:/gupiao/src/interfaces/presenters/html_dashboard.py)
- [`src/interfaces/presenters/markdown_reports.py`](/D:/gupiao/src/interfaces/presenters/markdown_reports.py)

The problem is not that the model stack is too advanced. The problem is that orchestration, contracts, and artifact ownership are not yet cleanly separated.

## Options

### Option A: Keep the current layered layout and only continue local extraction

Description:

- keep `domain / application / infrastructure / interfaces`
- continue moving helper functions out of `v2_services.py`
- avoid any visible architecture shift

Pros:

- lowest short-term code churn
- minimal file movement
- easy to continue current work

Cons:

- likely keeps `application` as a catch-all layer
- boundaries remain implicit
- new modules will continue to accumulate around the orchestrator
- performance work will still lack clean ownership boundaries

When this works:

- only one strategy
- low change rate
- small team
- no serious need for artifact lifecycle governance

### Option B: Adopt explicit bounded contexts inside the current repo

Description:

- keep the repository as a single deployable codebase
- reorganize code by quant lifecycle ownership, not just generic layers
- preserve `interfaces` as the entry layer
- shrink `application` into workflow and facade modules
- move implementation into domain-specific packages

Target contexts:

- `data_platform`
- `feature_dataset`
- `forecast_research`
- `policy_portfolio`
- `backtest_simulator`
- `artifact_registry`
- `daily_runtime`
- `reporting`

Pros:

- best fit for current project scale
- preserves incremental migration
- makes research and production contracts explicit
- creates a clean place for caching and artifact responsibilities
- scales better to multiple strategies and forecast backends

Cons:

- moderate refactor cost
- some duplication may temporarily exist during migration
- requires discipline on module ownership

When this works:

- batch quant workflows
- daily or end-of-day production
- multiple research iterations
- desire for stable published artifacts

### Option C: Move to event-driven multi-service architecture

Description:

- split data, forecasting, portfolio, reporting, and scheduling into separate services
- communicate through queues or APIs

Pros:

- strong isolation
- scales for larger teams and real-time workflows

Cons:

- large operational complexity
- slows down research iteration
- not justified by the current system shape

When this works:

- high-frequency or intraday systems
- many accounts or strategies
- mature operations capability

## Risks

### Risks if We Do Nothing

- `v2_services.py` continues to absorb new logic
- research and daily-run drift apart in behavior
- artifact publishing remains difficult to reason about
- caching remains partial and inconsistent
- adding a second forecast backend becomes much harder than expected

### Risks of the Recommended Refactor

- temporary duplication during migration
- import breakage if move order is careless
- tests may pass while artifact compatibility subtly changes
- team may over-refactor reporting before fixing orchestration boundaries

### Mitigations

- keep public entrypoints stable during migration
- move code behind facades before changing CLI behavior
- add artifact contract checks before large refactors
- migrate by phase, not by grand rewrite

## Recommendation

Recommend **Option B: explicit bounded contexts inside the current monorepo**.

This is the most appropriate architecture for the project now.

It keeps the good parts of `V2`:

- separate research and production flows
- forecast layer and policy layer distinction
- artifact-based production consumption

But turns them into actual code boundaries.

### Architectural Principles

1. Research produces immutable artifacts.
2. Daily production consumes published artifacts and does not retrain by default.
3. Forecast logic predicts state, not direct trades.
4. Policy logic decides exposure, concentration, and target weights.
5. Reporting renders outputs and owns no business decisions.
6. Orchestrators should coordinate modules, not contain core logic.
7. Caching should attach to explicit data products, not random helper stages.

### Target Module Layout

Recommended medium-term layout:

```text
src/
  domain/
    entities.py
    policies.py
    news.py
    symbols.py

  contracts/
    artifacts.py
    runtime.py
    reporting.py

  workflows/
    research_workflow.py
    daily_workflow.py
    workflow_blueprints.py

  data_platform/
    market_data.py
    data_sync.py
    discovery.py
    security_metadata.py
    info_repository.py
    margin_sync.py

  feature_dataset/
    features.py
    market_context.py
    panel_dataset.py
    sector_data.py
    external_signal_features.py
    margin_features.py

  forecast_research/
    modeling.py
    forecast_engine.py
    sector_forecast.py
    cross_section_forecast.py
    v2_info_fusion.py

  policy_portfolio/
    candidate_selection.py
    sector_support.py
    mainline_support.py
    strategy_memory.py

  backtest_simulator/
    backtesting.py
    effect_analysis.py

  artifact_registry/
    publish_support.py
    snapshot_support.py
    backtest_cache_support.py

  daily_runtime/
    daily_runtime.py
    external_signal_support.py

  reporting/
    markdown_reports.py
    html_dashboard.py
    driver_explainer.py

  interfaces/
    cli/
      run_v2_cli.py
      run_api_cli.py
```

This does not need to happen in one commit. It is the target architecture.

### Runtime Boundaries

#### Research Workflow

```text
config
-> data_platform
-> feature_dataset
-> forecast_research
-> policy_portfolio
-> backtest_simulator
-> artifact_registry
-> reporting
```

Research outputs:

- `dataset_manifest`
- `forecast_bundle`
- `policy_calibration`
- `policy_model`
- `strategy_snapshot`
- `backtest_summary`
- `research_manifest`

#### Daily Workflow

```text
config
-> data_platform
-> artifact_registry(load published snapshot)
-> daily_runtime(score latest state)
-> policy_portfolio(apply policy)
-> reporting
```

Daily outputs:

- current state snapshot
- recommendation set
- trade plan
- daily report
- dashboard payload

### Contract Design

The most important architecture boundary is not a Python helper function. It is the artifact contract.

The following should become explicit stable contracts:

- `DatasetManifest`
- `ForecastBundle`
- `PolicyCalibrationArtifact`
- `LearnedPolicyArtifact`
- `StrategySnapshot`
- `ResearchManifest`
- `DailyReportViewModel`

Recommendation:

- keep these contracts versioned
- centralize serialization/deserialization rules
- validate required fields on load

### Migration Plan

#### Phase 1: Freeze public behavior, shrink the orchestrator

Goal:

- keep CLI and outputs stable
- move implementation details out of `v2_services.py`

Actions:

- keep [`src/application/v2_services.py`](/D:/gupiao/src/application/v2_services.py) only as facade or compatibility surface
- move research-run implementation into `workflows/research_workflow.py`
- move daily-run implementation into `workflows/daily_workflow.py`
- move artifact publication and loading into `artifact_registry/*`

Exit criteria:

- `v2_services.py` becomes mostly imports plus facade functions
- daily and research workflows can be read independently

#### Phase 2: Introduce typed runtime/config contracts

Goal:

- reduce parameter sprawl and accidental CLI duplication

Actions:

- define `ResearchRunOptions` and `DailyRunOptions`
- define shared CLI argument registration helpers
- stop passing dozens of raw keyword arguments through workflow boundaries

Exit criteria:

- workflow entry functions accept one options object plus dependencies
- CLI code becomes thin and obvious

#### Phase 3: Normalize artifact contracts

Goal:

- make research-to-production handoff explicit and testable

Actions:

- add versioned artifact dataclasses or schemas
- validate snapshot payloads before use
- centralize path resolution and manifest loading

Exit criteria:

- artifact format changes are visible and reviewed
- tests verify compatibility and missing-field behavior

#### Phase 4: Move heavy data preparation behind reusable products

Goal:

- target the real performance bottlenecks

Actions:

- separate:
  - raw data load
  - market frame build
  - panel dataset build
  - frozen forecast bundle build
- cache these steps by explicit keys and manifests

Exit criteria:

- repeated research runs reuse prepared datasets predictably
- daily-run reuses published forecast artifacts and avoids recomputation

#### Phase 5: Simplify reporting ownership

Goal:

- keep reporting thin and independent

Actions:

- introduce reporting view-models
- move formatting-only logic into `reporting`
- keep decision logic outside markdown/html renderers

Exit criteria:

- reporters consume structured payloads only
- report modules stop reaching into domain logic

## Counter-Review

The strongest objection to this RFC is:

"This may slow down research while the strategy is still evolving."

That objection is valid if the refactor becomes a rewrite.

This RFC should **not** be executed as a full folder move with broad renames first.
It should be executed as controlled extraction while preserving public entrypoints.

A second objection is:

"The current layered architecture is already good enough."

That is only partly true.

The current architecture documents are directionally correct, but the implementation does not yet enforce those boundaries. As long as orchestration, artifact management, and business logic remain concentrated in a few oversized modules, the architecture is descriptive rather than operational.

Therefore, the recommendation is not to replace the current architecture with something completely different. The recommendation is to make the documented `V2` architecture real.

## Decision Summary

Decision:

- Adopt bounded-context architecture within the current repo.
- Keep a single repository and single deployable runtime.
- Preserve `research-run` and `daily-run` as the only two primary workflows.
- Make artifacts the main boundary between research and production.
- Refactor incrementally, starting from `v2_services.py` and CLI option contracts.

What should happen next:

1. Approve this target architecture.
2. Execute Phase 1 and Phase 2 first.
3. Delay deeper reporting cleanup until workflow and artifact boundaries are stable.

What should not happen next:

- no big-bang rewrite
- no microservice split
- no simultaneous directory renaming across the whole repo

Success means:

- a new contributor can identify where research logic, daily logic, artifacts, and reporting each belong
- performance optimization can target explicit build products
- new strategy capabilities can be added without expanding a single god module
