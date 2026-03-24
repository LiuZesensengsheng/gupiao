# Dynamic 300 规则版操作计划

- 数据日期: 2026-03-18
- 方案类型: dynamic300 fast rule
- 逻辑口径: dynamic 300 + 分桶候选 + 同主题限额 + 次日纪律计划
- dynamic 300 入池数: 300
- fresh gate 通过数: 151

## Fresh Funnel

| 阶段 | 数量 |
|---|---:|
| post_min_history_liquidity | 2038 |
| close>=5 | 1805 |
| close>ma20>ma60 | 381 |
| ret20 band | 246 |
| ret60 band | 210 |
| volatility20<=cap | 189 |
| within_20d_high_gap | 172 |
| amount_ratio20 band | 151 |

## 当前主线

| 主线 | 入池数 | 强度 |
|---|---:|---:|
| 通信设备 | 11 | 0.359 |
| 电气设备 | 11 | 0.350 |
| 半导体 | 11 | 0.338 |
| 能源石油 | 10 | 0.373 |
| 化工原料 | 10 | 0.350 |
| 资源 | 10 | 0.347 |
| 建筑工程 | 10 | 0.339 |
| 元器件 | 10 | 0.333 |
| 煤炭 | 10 | 0.320 |
| 专用机械 | 10 | 0.314 |

## 次日买入候选

| 排名 | 股票 | 主题 | Bucket | 建议权重 | 买入区间 | 回避区间 |
|---:|---|---|---|---:|---|---|
| 1 | 陕西煤业 (601225.SH) | 煤炭 | trend | 25.5% | 25.24 - 25.88 | >26.52 不追高 / <24.73 不接 |
| 2 | 赤峰黄金 (600988.SH) | 资源 | pullback | 26.1% | 40.31 - 41.23 | >42.25 不追高 / <40.31 不接 |
| 3 | 二六三 (002467.SZ) | 科技软件 | breakout | 22.9% | 7.43 - 7.62 | >7.80 不追高 / <7.19 不接 |
| 4 | 南京熊猫 (600775.SH) | 通信设备 | pullback | 25.6% | 14.89 - 15.27 | >15.65 不追高 / <14.88 不接 |

### 1. 陕西煤业 (601225.SH)
- 最新价: 25.62
- 减仓条件: 收盘跌回 20 日线 24.73 下方先减仓；如果冲高回落明显、量能不跟，也先降仓位。
- 退出条件: 有效跌破 60 日线 22.87 直接退出；20 日线失守后继续弱化也退出。
- 理由: bucket=trend，fresh=0.812，refined=1.020
- 理由: 距 20 日高点 4.0%，量能比 1.15
- 理由: 主题 煤炭 当前入池 10 只，主题强度 0.320
- 理由: 近 20 日涨幅 11.1%，近 60 日涨幅 17.2%，120 日突破位置 0.84

### 2. 赤峰黄金 (600988.SH)
- 最新价: 40.82
- 减仓条件: 收盘跌回 20 日线 40.31 下方先减仓；如果冲高回落明显、量能不跟，也先降仓位。
- 退出条件: 有效跌破 60 日线 36.87 直接退出；20 日线失守后继续弱化也退出。
- 理由: bucket=pullback，fresh=0.718，refined=1.037
- 理由: 距 20 日高点 7.0%，量能比 1.09
- 理由: 主题 资源 当前入池 10 只，主题强度 0.347
- 理由: 近 20 日涨幅 11.0%，近 60 日涨幅 30.8%，120 日突破位置 0.65

### 3. 二六三 (002467.SZ)
- 最新价: 7.54
- 减仓条件: 收盘跌回 20 日线 7.19 下方先减仓；如果冲高回落明显、量能不跟，也先降仓位。
- 退出条件: 有效跌破 60 日线 6.69 直接退出；20 日线失守后继续弱化也退出。
- 理由: bucket=breakout，fresh=0.517，refined=0.910
- 理由: 距 20 日高点 1.0%，量能比 1.49
- 理由: 主题 科技软件 当前入池 10 只，主题强度 0.306
- 理由: 近 20 日涨幅 4.1%，近 60 日涨幅 24.6%，120 日突破位置 0.96

### 4. 南京熊猫 (600775.SH)
- 最新价: 15.12
- 减仓条件: 收盘跌回 20 日线 14.88 下方先减仓；如果冲高回落明显、量能不跟，也先降仓位。
- 退出条件: 有效跌破 60 日线 14.33 直接退出；20 日线失守后继续弱化也退出。
- 理由: bucket=pullback，fresh=0.560，refined=0.941
- 理由: 距 20 日高点 6.9%，量能比 1.18
- 理由: 主题 通信设备 当前入池 11 只，主题强度 0.359
- 理由: 近 20 日涨幅 4.8%，近 60 日涨幅 35.0%，120 日突破位置 0.50

## 观察名单

| 股票 | 主题 | Bucket | 备注 |
|---|---|---|---|
| 晋控煤业 (601001.SH) | 煤炭 | pullback | bucket=pullback; passed fresh gate, but stronger theme/candidate occupied the core slots. |
| 中国化学 (601117.SH) | 建筑工程 | pullback | bucket=pullback; passed fresh gate, but stronger theme/candidate occupied the core slots. |
| 中国神华 (601088.SH) | 煤炭 | trend | bucket=trend; passed fresh gate, but stronger theme/candidate occupied the core slots. |
| 润建股份 (002929.SZ) | 通信设备 | pullback | bucket=pullback; passed fresh gate, but stronger theme/candidate occupied the core slots. |
| 山金国际 (000975.SZ) | 资源 | pullback | bucket=pullback; passed fresh gate, but stronger theme/candidate occupied the core slots. |
| 璞泰来 (603659.SH) | 化工原料 | pullback | bucket=pullback; passed fresh gate, but stronger theme/candidate occupied the core slots. |
| 深南电路 (002916.SZ) | 元器件 | pullback | bucket=pullback; passed fresh gate, but stronger theme/candidate occupied the core slots. |
| 恩捷股份 (002812.SZ) | 电气设备 | pullback | bucket=pullback; passed fresh gate, but stronger theme/candidate occupied the core slots. |

## 交易纪律

- 默认最多开 4 只，且同主题默认不超过 1 只。
- 高开超过回避区间不追，跌破 20 日线不接。
- 这份计划先强调可执行性，不把高波动追涨包装成纪律化交易。