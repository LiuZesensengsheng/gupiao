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
- Market + stock layered modeling
- Walk-forward out-of-sample evaluation
- Portfolio backtest (3y/5y windows) with equity curve and trading-cost simulation

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

3. Read report:

- `reports/latest_report.md`

## Daily Workflow (Simple)

This is the practical flow you asked for:

1. Fill daily news events in `input/news.csv` (you can copy from `input/news_template.csv`).
2. Run one command:

```bash
python3 run_api.py daily
```

3. Read fusion report:

- `reports/daily_report.md`
- `reports/daily_dashboard.html`

The daily report includes:

- Quant baseline probabilities (market + stocks)
- Fuzzy news matrix (bullish/bearish/neutral memberships)
- Blended probabilities after news adjustment
- Suggested total exposure and stock weights
- Market-effect modules: profit effect, loss effect, chip structure, capital state, sector heat
- Backtest metrics: annual return/excess, max drawdown, Sharpe, Sortino, Calmar, Information Ratio, turnover, cost drag

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

You can customize dashboard output path:

```bash
python3 run_api.py daily --source eastmoney --news-file input/news.csv --dashboard reports/my_dashboard.html
```

Backtest parameters can be overridden from CLI:

```bash
python3 run_api.py daily \
  --backtest-years 3,5 \
  --backtest-retrain-days 20 \
  --commission-bps 1.5 \
  --slippage-bps 2.0 \
  --backtest-weight-threshold 0.5
```

## Unified API Config

`run_api.py` loads `config/api.json` by default.

Merge priority:

- CLI args
- task section (`daily`/`forecast`) in config file
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
