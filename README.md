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

## Quick Start

1. Prepare watchlist (already provided in `config/watchlist.json`).
2. Run forecast:

```bash
python3 run_forecast.py --source eastmoney
```

If your environment cannot access network, use local CSV mode:

```bash
python3 run_forecast.py --source local --data-dir data
```

3. Read report:

- `reports/latest_report.md`

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
