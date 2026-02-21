# A-Share Multi-Horizon Forecast V1

This project builds a practical forecast pipeline for:

- Market-level direction probability (`1d` and `20d`)
- Stock-level direction probability for a selected watchlist (`1d` and `20d`)
- Position sizing suggestions based on market regime + stock scores

It is designed for A-shares and includes your initial symbols:

- `002772.SZ` 众兴菌业
- `603516.SH` 淳中科技
- `600160.SH` 巨化股份
- `000630.SZ` 铜陵有色
- `603619.SH` 中曼石油

## Features Included

- Short-term factors: momentum, reversal, volume shock, volatility expansion
- Mid-term factors: trend structure (20/60), drawdown, position-in-range
- Capital/chip proxies (without Level-2): OBV, Amihud proxy, volume concentration
- Volume risk flag: `高位巨量大阴线` detection and warning
- Market + stock layered modeling
- Multi-index market context (`000001.SH`, `399001.SZ`, `399006.SZ`) + breadth factors
- Margin financing/securities lending module (`两融`) for market + stock features
- Walk-forward out-of-sample evaluation
- Portfolio backtest (3y/5y windows) with equity curve and trading-cost simulation
- Learned news fusion: train `news -> impact` and calibrate `quant + news -> final probability`
- Candidate discovery mode: expand stock pool and rank top ideas for your manual review

## Architecture

The project now uses a layered architecture for long-term iteration:

- `src/domain`: business entities/policies (regime, exposure, news semantics)
- `src/application`: use-cases and orchestration
- `src/infrastructure`: data/model adapters and analytics engines
- `src/interfaces`: CLI entrypoints and markdown/html presenters

Architecture details:

- `docs/ARCHITECTURE.md`

## Current Module Map (for 7-14d swing + intraday T)

Based on current code, the core execution chain is:

1. CLI + config merge  
   - `run_api.py`  
   - `src/interfaces/cli/run_api_cli.py`
2. Use-case orchestration (forecast / daily fusion / discover / backtest optimizer)  
   - `src/application/use_cases.py`
3. Quant forecast core (market + stock probabilities)  
   - `src/infrastructure/forecast_engine.py`
4. Factor/feature engineering (price-volume + market context + margin)  
   - `src/infrastructure/features.py`  
   - `src/infrastructure/market_context.py`  
   - `src/infrastructure/margin_features.py`
5. News fusion (rule + learned calibration)  
   - `src/infrastructure/news_repository.py`  
   - `src/infrastructure/news_fusion.py`
6. Position sizing + policy rules  
   - `src/domain/policies.py`
7. Portfolio backtest + strategy parameter search  
   - `src/infrastructure/backtesting.py`  
   - `src/application/use_cases.py` (`_optimize_strategy_selection`)
8. Report/dashboard rendering  
   - `src/interfaces/presenters/markdown_reports.py`  
   - `src/interfaces/presenters/html_dashboard.py`

### What To Replace First

For your target workflow ("AI-assisted discretionary", 7-14d holding, allow intraday T), the first module to upgrade is:

- `signal routing / policy layer`, not raw factor set
- Current entry points:
  - `src/domain/policies.py` (`market_regime`, `target_exposure`, `allocate_weights`)
  - `src/application/use_cases.py` (where exposure/weights are applied)

Reason: your current regime is probability-threshold based (`Risk-On/Neutral/Risk-Off`), but not yet a dedicated state engine driving:

- strategy template switching
- T frequency limits
- risk budget by state

## Iteration Roadmap (Module-by-Module)

Use this order to avoid large refactors:

1. State Engine (highest priority)  
   Goal: classify each day into `trend / range / risk-off` and output state-specific params.  
   Minimal integration: replace direct `target_exposure(...)` calls with `state_engine -> exposure cap`.
2. Strategy Router  
   Goal: bind different execution templates to state (`trend push`, `range buy-low-sell-high`, `defensive`).  
   Integration point: `generate_daily_fusion(...)` before weight allocation.
3. Intraday-T Guardrail Module  
   Goal: only allow T in whitelist scenarios; cap attempts/day and weekly churn.  
   Integration point: extend turnover control fields in daily config + backtest assumptions.
4. Portfolio Risk Budget  
   Goal: add portfolio-level constraints (sector concentration, correlated crowding, per-position loss budget).  
   Integration point: before/after `allocate_weights(...)`.
5. Attribution Loop  
   Goal: decompose PnL into selection / timing / T / risk-control contribution.  
   Integration point: reporting + backtest post-processing.

### Keep Unchanged For Now

To keep momentum, do not rewrite these first:

- data adapters (`market_data.py`)
- base feature construction (`features.py`)
- learned news fusion pipeline

### Practical Working Rule

For each iteration cycle, change only one module and keep all others fixed:

1. define module inputs/outputs
2. run `daily` end-to-end
3. compare backtest + turnover + drawdown
4. update report fields and only then proceed to next module

### Intraday T Whitelist (Range State)

When market state is `range`, rebalance actions are filtered by T-style whitelist:

- allow reduce only when stock is relatively overextended:
  - `ret_1 >= +2%` and `price_pos_20 >= 0.80`
- allow add only when stock is in sharp pullback near lower range:
  - `ret_1 <= -2%` and `price_pos_20 <= 0.35`

This keeps T behavior aligned with your rule: only do one meaningful in-day adjustment instead of frequent micro-rebalancing.

## Quick Start

1. Prepare watchlist (already provided in `config/watchlist.json`).
2. Optional: edit unified runtime config in `config/api.json`.
3. Run forecast:

```bash
python3 run_api.py forecast
```

If your environment cannot access network, override from CLI:

```bash
python3 run_api.py forecast --source local --data-dir data
```

If your network is unstable, use automatic fallback:

```bash
python3 run_api.py forecast --source auto
```

3. Read report:

- `reports/latest_report.md`

4. Optional: discover candidate stocks for your pool:

```bash
python3 run_api.py discover
```

Read report:

- `reports/discovery_report.md`

5. Optional: sync a larger local universe (300-1000 symbols):

```bash
python3 run_api.py sync-data --source auto --universe-size 500
```

## Daily Workflow (Simple)

This is the practical flow you asked for:

1. Fill daily news events in `input/news.csv` (you can copy from `input/news_template.csv`).
   For large datasets, you can also split by target/year in a directory (recommended).
2. Optional: fill margin templates:
   - `input/margin_market_template.csv` -> `input/margin_market.csv`
   - `input/margin_stock_template.csv` -> `input/margin_stock.csv`
3. Run one command:

```bash
python3 run_api.py daily
```

4. Read fusion report:

- `reports/daily_report.md`
- `reports/daily_dashboard.html`

The daily report includes:

- Quant baseline probabilities (market + stocks)
- Fuzzy news matrix (bullish/bearish/neutral memberships)
- Learned news probabilities + calibrated final probabilities
- Suggested total exposure and stock weights
- Market-effect modules: profit effect, loss effect, chip structure, capital state, sector heat
- Learning diagnostics (samples, holdout metrics, learned coefficients)
- Backtest metrics: fused strategy vs quant baseline (annual return/excess, max drawdown, Sharpe, Sortino, Calmar, Information Ratio, turnover, cost drag)
- Volume/chip warning column (`量价风险`) for each stock row

### News CSV Fields

Required columns:

- `date`
- `target_type` (`market` or `stock`)
- `target` (use `MARKET` for market-level events; use symbol like `600160.SH` for stock-level)
- `direction` (`bullish` / `bearish` / `neutral`)

Optional columns:

- `horizon` (`short`, `mid`, or `both`; default `both`)
- `strength` (1-5; default 3)
- `confidence` (0-1; default 0.7)
- `source_weight` (0-1; default 0.7)
- `title`

### Scalable News Layout (Recommended)

When news volume gets large, use directory partitions by target and year:

```text
input/news_parts/
  MARKET/2024.csv
  MARKET/2025.csv
  MARKET/2026.csv
  600160.SH/2024.csv
  600160.SH/2025.csv
  600160.SH/2026.csv
  603619.SH/2024.csv
  ...
```

Each CSV keeps the same columns as `input/news_template.csv`.
The loader supports both a single CSV path and a directory path.

Example:

```bash
python3 run_api.py daily --news-file input/news_parts
```

Optional helper to initialize empty partition files:

```bash
python3 scripts/init_news_partitions.py \
  --watchlist config/watchlist.json \
  --out-dir input/news_parts \
  --start-year 2022 \
  --end-year 2026
```

Automatic collection helper (Eastmoney announcements -> news partitions):

```bash
python3 scripts/collect_notice_news.py \
  --watchlist config/watchlist.json \
  --start-year 2022 \
  --end-year 2026 \
  --out-dir input/news_parts
```

Notes:

- This collector currently focuses on stock announcements (not market macro news).
- Direction/horizon are inferred from title keywords and should be treated as weak labels.

You can tune blending sensitivity:

- `--market-news-strength` (default 0.9)
- `--stock-news-strength` (default 1.1)
- `--news-lookback-days` (default 45)
- `--news-half-life-days` (default 10)

Learned fusion controls (daily task):

- `--use-learned-news-fusion true|false` (default true)
- `--learned-news-lookback-days` (default 720)
- `--learned-news-min-samples` (default 80)
- `--learned-holdout-ratio` (default 0.2)
- `--learned-news-l2` (default 0.8)
- `--learned-fusion-l2` (default 0.6)

Example:

```bash
# Learned mode (default)
python3 run_api.py daily --source eastmoney

# Force rule-only mode (no learned fitting)
python3 run_api.py daily --source eastmoney --use-learned-news-fusion false
```

You can customize dashboard output path:

```bash
python3 run_api.py daily --source auto --news-file input/news.csv --dashboard reports/my_dashboard.html

# large partitioned dataset
python3 run_api.py daily --source auto --news-file input/news_parts --dashboard reports/my_dashboard.html
```

Dashboard highlights:

- Main curve supports one-click switch between `融合策略净值` and `量化基线净值`
- Extra spread curve shows `融合/基线 相对净值` (1.00 means tie)

Backtest parameters can be overridden from CLI:

```bash
python3 run_api.py daily \
  --backtest-years 3,5 \
  --backtest-retrain-days 20 \
  --backtest-time-budget-minutes 3 \
  --commission-bps 1.5 \
  --slippage-bps 2.0 \
  --backtest-weight-threshold 0.5 \
  --enable-acceptance-checks true \
  --acceptance-target-years 3
```

Turnover/frequency guardrail (recommended for discretionary style):

```bash
python3 run_api.py daily \
  --use-turnover-control true \
  --max-trades-per-stock-per-day 1 \
  --max-trades-per-stock-per-week 3 \
  --min-weight-change-to-trade 0.03
```

This means each stock can trade at most `1` time per day and `3` times in a rolling week (5 trading days),
and tiny rebalances under `3%` weight change are ignored.

Range-state T whitelist thresholds are configurable:

- `--range-t-sell-ret-1-min` (default `0.02`)
- `--range-t-sell-price-pos-20-min` (default `0.80`)
- `--range-t-buy-ret-1-max` (default `-0.02`)
- `--range-t-buy-price-pos-20-max` (default `0.35`)

Strategy optimizer (maximize excess return with turnover/drawdown controls):

```bash
python3 run_api.py daily \
  --use-strategy-optimizer true \
  --optimizer-retrain-days 20,40 \
  --optimizer-weight-thresholds 0.5,0.6 \
  --optimizer-max-positions 3,5 \
  --optimizer-time-budget-minutes 3 \
  --optimizer-turnover-penalty 0.0015 \
  --optimizer-drawdown-penalty 0.2 \
  --optimizer-target-years 3
```

Objective used by optimizer:

- `score = excess_annual_return - turnover_penalty * annual_turnover - drawdown_penalty * abs(max_drawdown)`
- annual turnover is the annualized sum of daily turnover (decimal form, e.g. `31.0` means `3100%`)

Position constraints:

- `--max-positions` controls max simultaneous stock holdings (default `5`)
- applied consistently in forecast, daily suggestions, discovery suggestions, and backtest

Margin module controls:

- `--use-margin-features true|false` (default `true`)
- `--margin-market-file input/margin_market.csv`
- `--margin-stock-file input/margin_stock.csv`

When margin files are missing or sparse, related features are auto-skipped (pipeline still runs).

## Discovery Workflow (You Pick, System Times)

Use this mode when you already have strong stock ideas and want timing + ranking support.

```bash
python3 run_api.py discover \
  --source local \
  --data-dir data \
  --candidate-limit 300 \
  --top-k 30 \
  --exclude-watchlist false
```

Options:

- `--universe-file`: optional custom universe (`csv/json`) with at least `symbol` column
- `--candidate-limit`: pre-filter pool size before model ranking
- `--top-k`: final output count
- `--exclude-watchlist true|false`: whether to exclude current watchlist symbols
- `--max-positions`: max simultaneous holdings in suggested weights (default `5`)

Discovery output includes:

- 1d/20d probability, combined score, suggested weight
- top feature drivers (short/mid)
- volume/chip risk flag (`高位巨量阴线风险`)

## Universe Sync (300-1000 Stocks)

Use this to build/refresh local A-share bars for broader cross-sectional modeling.

```bash
python3 run_api.py sync-data \
  --source auto \
  --data-dir data \
  --start 2018-01-01 \
  --end 2099-12-31 \
  --universe-size 500 \
  --include-indices true \
  --write-universe-file config/universe_auto.json
```

Useful options:

- `--universe-file`: use custom universe file (`csv/json`) instead of auto fetch
- `--force-refresh true|false`: force redownload even if local file is fresh
- `--sleep-ms`: throttle between requests
- `--max-failures`: stop early when failures accumulate

Output:

- local bars under `data/*.csv`
- generated universe file for discovery (default `config/universe_auto.json`)
- if network fetch fails, universe falls back to existing local `data/*.csv`

Market breadth factors are computed from local symbol files, so expanding local universe directly improves those features.

## Margin Sync (两融数据同步)

Use this to automatically pull/refresh:

- market-level margin file: `input/margin_market.csv`
- stock-level margin file: `input/margin_stock.csv`

```bash
# Use watchlist stocks by default
python3 run_api.py sync-margin \
  --source auto \
  --watchlist config/watchlist.json \
  --start 2023-01-01 \
  --end 2099-12-31

# Or specify symbols directly
python3 run_api.py sync-margin \
  --source akshare \
  --symbols 600160.SH,000630.SZ,603619.SH
```

Useful options:

- `--source`: `akshare` / `tushare` / `auto` (default `auto`)
- `--symbols`: comma-separated symbols; when empty, use watchlist stocks
- `--sleep-ms`: throttle request interval
- `--margin-market-file` / `--margin-stock-file`: output CSV paths

Notes:

- if a source only returns market-level rows, stock-level file is still written with empty schema
- if all sources fail, command returns error and prints failure reason

## Unified API Config

`run_api.py` loads `config/api.json` by default.

Merge priority:

- CLI args
- task section (`daily`/`forecast`/`discover`/`sync-data`/`sync-margin`) in config file
- `common` section in config file
- built-in defaults

Useful commands:

```bash
# Use custom config file
python3 run_api.py daily --config config/api.json

# Inspect final merged params
python3 run_api.py forecast --print-effective-config

# Inspect effective daily params (including backtest config)
python3 run_api.py daily --print-effective-config

# Inspect effective discovery params
python3 run_api.py discover --print-effective-config

# Inspect effective sync-data params
python3 run_api.py sync-data --print-effective-config

# Inspect effective sync-margin params
python3 run_api.py sync-margin --print-effective-config
```

## Data Sources

`source` now supports single source or fallback chain:

- `eastmoney`
- `tushare`
- `akshare`
- `baostock`
- `local`
- `auto` (equivalent to `eastmoney,tushare,akshare,baostock`)

Examples:

```bash
# Automatic fallback
python3 run_api.py daily --source auto

# Explicit fallback chain
python3 run_api.py daily --source eastmoney,tushare,akshare,baostock

# Force local CSV only
python3 run_api.py daily --source local --data-dir data
```

Optional packages for alternative sources:

```bash
pip install tushare akshare baostock
```

Tushare token configuration (any one is enough):

```bash
# Method 1: environment variable
export TUSHARE_TOKEN="your_token_here"
python3 run_api.py daily --source tushare

# Method 2: CLI override
python3 run_api.py daily --source tushare --tushare-token "your_token_here"
```

Or store in `config/api.json`:

```json
{
  "common": {
    "source": "auto",
    "tushare_token": "your_token_here"
  }
}
```

## Backtest Metrics

The dashboard and daily report now include these core quant metrics:

- Strategy total return / annualized return
- Benchmark total return / annualized return
- Excess return (total + annualized)
- Max Drawdown (MDD)
- Sharpe / Sortino / Calmar
- Information Ratio / Tracking Error
- Daily win rate
- Turnover (daily average + annualized)
- Cost drag (commission + slippage)

When daily news fusion is enabled, backtest will output both:

- `融合策略-*` (news-fused signal)
- `量化基线-*` (quant-only signal)

## Local CSV Format

Each symbol needs one CSV file in `data/`, for example:

- `data/002772.SZ.csv`
- `data/603516.SH.csv`
- `data/000300.SH.csv` (market index, default CSI300)

Required columns:

- `date`
- `open`
- `high`
- `low`
- `close`
- `volume`

Optional column:

- `amount`

If `amount` is missing, it will be approximated by `close * volume`.

## Notes

- This is a probability model, not certainty prediction.
- Use it as research support, not standalone trading advice.
- The script assumes daily bars and predicts next trading bar (`1d`) and `20d`.
- Root script `run_api.py` is the only API entrypoint; business logic lives in `src/application` and below.
- Legacy compatibility modules under `src/` were removed; use the layered paths in `docs/ARCHITECTURE.md`.

## High-Volume Bearish Candle Rule

`高位巨量大阴线` warning is triggered when all conditions are met on a day, then kept for 5 trading days:

- daily return `<= -4%`
- volume ratio vs 20d mean `>= 2.0`
- previous day 20d price position `>= 0.75`

This signal is designed as a risk flag, not a hard sell rule.
