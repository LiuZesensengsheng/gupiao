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
```

Dashboard highlights:

- Main curve supports one-click switch between `融合策略净值` and `量化基线净值`
- Extra spread curve shows `融合/基线 相对净值` (1.00 means tie)

Backtest parameters can be overridden from CLI:

```bash
python3 run_api.py daily \
  --backtest-years 3,5 \
  --backtest-retrain-days 20 \
  --commission-bps 1.5 \
  --slippage-bps 2.0 \
  --backtest-weight-threshold 0.5
```

Turnover/frequency guardrail (recommended for discretionary style):

```bash
python3 run_api.py daily \
  --use-turnover-control true \
  --max-trades-per-stock-per-week 3 \
  --min-weight-change-to-trade 0.03
```

This means each stock can trade at most `3` times in a rolling week (5 trading days),
and tiny rebalances under `3%` weight change are ignored.

Strategy optimizer (maximize excess return with turnover/drawdown controls):

```bash
python3 run_api.py daily \
  --use-strategy-optimizer true \
  --optimizer-retrain-days 20,40 \
  --optimizer-weight-thresholds 0.5,0.6 \
  --optimizer-max-positions 3,5 \
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
