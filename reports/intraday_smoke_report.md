# V2 次日决策日报

- 策略ID: swing_v2
- artifact run_id: intraday_smoke_20260319
- 股票池: intraday_smoke
- 股票池规模: 1
- external signals enabled: false
- US index context: enabled (akshare)
- 数据日期: 2026-03-18
- 下一交易日: 2026-03-19
- 策略模式: trend_follow
- 风险状态: risk_on

## 市场总览

| 指标 | 数值 |
|---|---:|
| 情绪阶段 | 中性 |
| 情绪分 | 50.0 / 100 |
| 目标总仓位 | 15.0% |
| 目标持仓数 | 1 |
| 涨家数 / 跌家数 / 平家数 | 0 / 0 / 0 |
| 涨停 / 跌停 | 0 / 0 |
| 新高 / 新低 | 0 / 0 |
| 样本中位涨跌幅 | 0.0% |
| 样本覆盖数 | 0 |
| 样本成交额 | 0 |

## 大盘情绪

- 情绪结论: 

## 大盘多周期预测

| 周期 | 上涨概率 | 预期区间 | 中位预期 | 置信度 |
|---|---:|---:|---:|---:|

## 动态股票池

| 指标 | 数值 |
|---|---:|
| 粗排池 | 0 |
| 精排池 | 0 |
| 最终动态池 | 0 |

## 外部信号

| 指标 | 数值 |
|---|---:|
| 资金状态 | neutral |
| 北向强度 | 0.000 |
| 两融变化 | 0.000 |
| 成交热度 | 50.0% |
| 大单偏向 | 0.000 |
| 宏观风险 | neutral |
| 风格状态 | balanced |
| 商品压力 | 0.0% |
| 汇率压力 | 0.0% |
| 宽度代理 | 50.0% |

## 当日主线列表

| 主线 | phase | conviction | breadth | leadership | event_risk | 说明 |
|---|---|---:|---:|---:|---:|---|
| NA | NA | NA | NA | NA | NA | 暂无 insight 主线 |

## 龙头候选

| 股票 | 主线 | phase | role | candidate | conviction | negative | 说明 |
|---|---|---|---|---:|---:|---:|---|
| ???? (600000.SH) | ?? |  | core | 33.7% | 21.8% | 18.0% | theme role core | theme rank 1/1 | weak excess vs sector |

## 预测监控榜

| 排名 | 股票 | 行业 | 下一交易日区间 | 5日中位预期 | 20日中位预期 | 1日上涨概率 | 可执行分 | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | ???? (600000.SH) | ?? | 10.22 ~ 10.48 | 1.8% | 5.2% | 51.0% | NA | selected |

## 可执行候选榜

| 排名 | 股票 | 行业 | 可执行分 | 目标权重 | 下一交易日区间 | 1日上涨概率 | 备注 |
|---:|---|---|---:|---:|---|---:|---|
| 1 | NA | NA | NA | NA | NA | NA | 当前无新开仓候选 |

## 实际操作

| 股票 | 动作 | 当前权重 | 目标权重 | 权重变化 | 操作理由 |
|---|---|---:|---:|---:|---|
| 无 | HOLD | NA | NA | NA | 当前不触发调仓 |

## 持仓角色变化

| 股票 | 主线 | 当前角色 | 前序角色 | 角色降级 | 备注 |
|---|---|---|---|---|---|
| NA | NA | NA | NA | NA | 暂无角色变化 |

## 次日执行计划

| 股票 | bias | buy_zone | avoid_zone | intraday | levels | reduce_if | exit_if | reason |
|---|---|---|---|---|---|---|---|---|
| ???? (600000.SH) | reduce | 10.22 ~ 10.48 | gap-up chase above forecast range | hold_neutral | 15m | 2026-03-18 | stop 10.36 | tp 10.42 | near-term edge weakens | 20d probability breaks below 0.48 | held 6d; 15m close 10.37, VWAP gap +0.1%, drawdown 0.5%, close-pos 58% |

## Intraday Overlay

- ???? (600000.SH): hold_neutral | 15m | 2026-03-18 | stop 10.36 | tp 10.42 | 15m close 10.37, VWAP gap +0.1%, drawdown 0.5%, close-pos 58%

## 推荐解释卡

### ???? (600000.SH)
- 行业: ??
- 最新收盘价: 10.37
- 下一交易日(2026-03-19)预期区间: 10.22 ~ 10.48，中位预期 0.0%，上涨概率 51.0%，置信度 56.0%
- 多周期判断: 5日 1.8% / 10日 NA / 20日 5.2%
- 可执行状态: selected，可执行分 NA，备注: prediction-only monitor
- 入池原因: liquidity strong
- 排名原因: defensive relative strength
- 操作原因: smoke test position
- 仓位说明: baseline hold

## 预测复盘

| 窗口 | 命中参考 | 平均边际 | 近窗表现 | 样本数 | 说明 |
|---|---:|---:|---:|---:|---|
| 近窗 | NA | NA | NA | 0 | 暂无复盘数据 |

## 淇℃伅灞傛憳瑕?

| 鎸囨爣 | 鏁板€?|
|---|---:|
| info shadow enabled | false |
| info items | 0 |
| market catalyst | 0.0% |
| market event risk | 0.0% |
| market negative risk | 0.0% |
| market coverage | 0.0% |