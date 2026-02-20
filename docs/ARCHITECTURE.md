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
    use_cases.py       # Forecast/Daily fusion/Discovery orchestration + strategy objective search

  infrastructure/
    market_data.py     # Market data adapters (eastmoney/tushare/akshare/baostock/local + fallback chain)
    market_context.py  # Multi-index + breadth context feature builder
    margin_features.py # Margin financing/securities lending feature builder (market + stock)
    features.py        # Factor engineering
    modeling.py        # Logistic model + metrics
    forecast_engine.py # Quant forecast engine
    news_fusion.py     # Learned news impact + quant/news calibration
    backtesting.py     # Portfolio backtest + risk/return metrics
    discovery.py       # Candidate-universe builder + volume/chip risk scanner
    data_sync.py       # Local universe sync (300-1000 symbols) + universe file generation
    margin_sync.py     # Margin financing/securities lending (两融) data sync to CSV
    news_repository.py # News CSV ingestion
    effect_analysis.py # Profit/loss/chip/capital/sector analytics

  interfaces/
    cli/
      run_api_cli.py   # single API entrypoint + config merge
    presenters/
      markdown_reports.py
      html_dashboard.py

config/
  api.json             # unified runtime config (common + daily + forecast + discover)
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

For long-term iterative modeling, treat each analysis perspective as an independent module:

- quant timing module (price/volume/volatility/chip)
- news impact module (rule-based or learned fusion)
- market/beta module (regime and exposure)
- candidate discovery module (pool expansion + filtering)

Application use-cases compose these modules and keep their outputs explainable.

## Where to Add New Modules

For new analysis dimensions (fundamental, cost trend, shipment, commodity prices):

1. Add domain entities/policies for the concept.
2. Add infrastructure adapter for data ingestion.
3. Add application use-case step for orchestration.
4. Add presenter blocks for report/dashboard.

Recommended extensibility pattern:

1. Keep pure scoring logic in module-level analyzers (stateless, testable).
2. Use application-level service orchestration to merge analyzers.
3. Use domain entities as stable contracts (avoid passing raw DataFrame everywhere).
4. If construction complexity grows, add simple factories in application/infrastructure (not in domain entities).

## Attention + Embedding Integration Plan

When introducing news embedding + attention fusion:

1. Keep news labeling/normalization in `domain.news`.
2. Add model implementation in `infrastructure` (e.g., `attention_fusion.py`).
3. Expose model through application use-case, not direct CLI calls.
4. Return explainability artifacts (attention weights/top contributing events) to presenters.

This keeps predictive power upgrades compatible with existing explainable reports.

## Migration Notes

Legacy compatibility wrappers have been removed.  
Use the new module paths directly:

- Data access: `src.infrastructure.market_data`
- Feature engineering: `src.infrastructure.features`
- Modeling: `src.infrastructure.modeling`
- Forecast engine: `src.infrastructure.forecast_engine`
- Portfolio backtest: `src.infrastructure.backtesting`
- Candidate discovery: `src.infrastructure.discovery`
- News ingestion/fusion: `src.infrastructure.news_repository` + `src.domain.news`
- Effect analytics: `src.infrastructure.effect_analysis`
- Report rendering: `src.interfaces.presenters.*`

Root script `run_api.py` is the single stable entrypoint.
