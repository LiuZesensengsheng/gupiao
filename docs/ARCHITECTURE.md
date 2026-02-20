# Architecture Guide

## Why Refactor

The project started as a fast prototype and centralized orchestration, rules, and rendering in a single script.  
To support long-term iteration (news NLP, attention fusion, alpha/beta decomposition, more factor modules), the codebase now uses a layered architecture.

## Layered Structure

```text
src/
  domain/
    entities.py        # Core business objects
    symbols.py         # Symbol normalization/value object
    policies.py        # Regime/exposure/weighting rules
    news.py            # News semantics + sentiment aggregation rules

  application/
    config.py          # Use-case configuration objects
    watchlist.py       # Watchlist loading
    use_cases.py       # Forecast/Daily fusion orchestration

  infrastructure/
    market_data.py     # Eastmoney/local CSV data adapters
    features.py        # Factor engineering
    modeling.py        # Logistic model + metrics
    forecast_engine.py # Quant forecast engine
    news_repository.py # News CSV ingestion
    effect_analysis.py # Profit/loss/chip/capital/sector analytics

  interfaces/
    cli/
      run_forecast_cli.py
      run_daily_cli.py
    presenters/
      markdown_reports.py
      html_dashboard.py
```

## Dependency Direction

Allowed direction:

- `interfaces -> application -> domain`
- `application -> infrastructure` (through explicit use-cases)
- `infrastructure -> domain` (for shared entities/value objects)

Avoid:

- `domain` importing `application` / `interfaces` / specific external IO details
- business rules in CLI or presenters

## Design Principles

1. Domain first: keep market/news/exposure semantics in `domain`.
2. Use-case orchestration in `application`.
3. Data/model adapters in `infrastructure`.
4. Rendering-only in `interfaces`.

## Where to Add New Modules

For new analysis dimensions (fundamental, cost trend, shipment, commodity prices):

1. Add domain entities/policies for the concept.
2. Add infrastructure adapter for data ingestion.
3. Add application use-case step for orchestration.
4. Add presenter blocks for report/dashboard.

## Attention + Embedding Integration Plan

When introducing news embedding + attention fusion:

1. Keep news labeling/normalization in `domain.news`.
2. Add model implementation in `infrastructure` (e.g., `attention_fusion.py`).
3. Expose model through application use-case, not direct CLI calls.
4. Return explainability artifacts (attention weights/top contributing events) to presenters.

This keeps predictive power upgrades compatible with existing explainable reports.

## Compatibility

Legacy modules (`src/data.py`, `src/features.py`, `src/model.py`, `src/pipeline.py`, etc.) are now thin compatibility wrappers that forward to the new layered modules.

