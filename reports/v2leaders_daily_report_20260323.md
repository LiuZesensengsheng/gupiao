# V2 次日决策日报

- 策略ID: swing_v2
- artifact run_id: 20260322_211044
- 股票池: dynamic_300
- 股票池规模: 300
- generator manifest path: D:\gupiao\artifacts\v2\cache\universe_catalog\dynamic_300_43b3c5a76238.generator.json
- generator version: dynamic_universe_v3_fresh_pool
- external signal manifest path: artifacts\v2\swing_v2\20260322_211044\external_signal_manifest.json
- external signals enabled: false
- external signal version: v1
- US index context: enabled (akshare)
- 数据日期: 2026-03-20
- 下一交易日: 2026-03-23
- 策略模式: range_rotation
- 风险状态: risk_off

## 市场总览

| 指标 | 数值 |
|---|---:|
| 情绪阶段 | 冰点 |
| 情绪分 | 26.0 / 100 |
| 目标总仓位 | 0.0% |
| 目标持仓数 | 0 |
| 涨家数 / 跌家数 / 平家数 | 86 / 598 / 4 |
| 涨停 / 跌停 | 8 / 7 |
| 新高 / 新低 | 14 / 429 |
| 样本中位涨跌幅 | -2.2% |
| 样本覆盖数 | 688 |
| 样本成交额 | 343,804,256 |

## 大盘情绪

- 情绪结论: 冰点阶段，下一交易日情绪分 26/100。
- 主要驱动: 涨跌家数差 86/598；市场宽度强度 -69.5%；新高/新低 14/429；涨跌停差 8/7
- 风险提示: Leader weighting active: 2 promoted, 0 suppressed, 0 hard negatives.；Leader-promoted symbols: 000158.SZ, 002053.SZ；Absolute buy gate blocked 4 fresh candidates without positive edge.；Fresh-buy blocks: 001216.SZ(actionability<0.52, 5d_tail_risk_high), 603606.SH(actionability<0.52, short_prob_stack<0.500), 603110.SH(actionability<0.52, 5d_tail_risk_high)；Risk-off regime: hard floor reduced, but not forced into deep cash.；Near-term market stack below 0.50: mild exposure trim.；Cross-sectional alpha weak: exposure trimmed.；Candidate shortlist active: 7/281 names after macro-sector screening.；Fresh-buy eligible after absolute gate: 3/7 names.；Risk-off actionability weak: stay in cash instead of forcing a new long.；Fragile tape: single-name cap tightened.

## 大盘多周期预测

| 周期 | 上涨概率 | 预期区间 | 中位预期 | 置信度 |
|---|---:|---:|---:|---:|
| 未来1日 | 49.7% | -1.6% ~ 1.6% | -0.0% | 53.5% |
| 未来2日 | 49.6% | -3.0% ~ 3.1% | 0.0% | 54.3% |
| 未来3日 | 49.3% | -3.8% ~ 4.0% | 0.1% | 55.3% |
| 未来5日 | 47.4% | -7.0% ~ 7.5% | 0.2% | 56.9% |
| 未来10日 | 49.6% | -8.6% ~ 9.2% | 0.3% | 53.8% |
| 未来20日 | 51.4% | -9.9% ~ 10.6% | 0.4% | 53.5% |

## 动态股票池

| 指标 | 数值 |
|---|---:|
| 粗排池 | 1000 |
| 精排池 | 600 |
| 最终动态池 | 300 |

- shortlist notes: Macro shortlist active: 2 sectors prioritized before fine ranking. | Quant breadth fill added 5 names beyond macro-sector core to avoid an over-compressed shortlist. | Macro ranking keeps all actionable names ordered; top 7 names receive sector-prioritized placement. | Recommendation scope limited to main-board listings only. | Leader overlay reprioritized shortlist order and refreshed core membership.

## 策略记忆

- memory path: artifacts\v2\memory\swing_v2_memory.json
- 最近研究: run_id=20260322_211044, 截止=2026-02-12, 超额年化=-12.7%, IR=-1.03, release gate=未通过
- 近期日运行 5 次: 平均目标仓位=16.4%, 调仓触发占比=60.0%, 仓位趋势=-25.0%
- 高频标的: 601328.SH, 603757.SH, 002831.SZ
- 最近一次研究 run_id=20260322_211044，超额年化 -12.7%，IR -1.03，release gate 未通过。
- 近 5 次日运行平均目标仓位 16.4%，调仓触发占比 60.0%，近几次仓位有下调倾向。
- 高频关注标的: 601328.SH, 603757.SH, 002831.SZ。
- 重复出现的风险标签: regulatory_negative, Risk-off regime: hard floor reduced, but not forced into deep cash., Cross-sectional alpha weak: exposure trimmed.。
- 持续出现的正向线索: 福达合金, 中大力德, 泰嘉股份。
- 高频事件风险: regulatory_negative, delisting_risk。
- 高频催化标签: 福达合金, 中大力德, 泰嘉股份。
- 近期资金状态: neutral, neutral, neutral。
- 宏观风险持续性: low, neutral。

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
| 云南能投 (002053.SZ) | 新能源电力 |  | leader | 58.2% | 40.3% | 6.2% | theme role leader | theme rank 1/29 | tradeability supportive |
| 常山北明 (000158.SZ) | 科技软件 |  | leader | 58.7% | 40.6% | 8.9% | theme role leader | theme rank 1/12 | tradeability supportive |
| 道明光学 (002632.SZ) | 煤化工 |  | leader | 56.3% | 40.7% | 9.5% | theme role leader | theme rank 2/22 | tradeability supportive |
| 华瓷股份 (001216.SZ) | 资源 |  | core | 54.2% | 39.2% | 13.6% | theme role core | theme rank 1/29 | tradeability supportive |
| 大为股份 (002213.SZ) | 计算机、通信和其他电子设备制造业 |  | core | 53.1% | 36.8% | 9.6% | theme role core | theme rank 1/33 | tradeability supportive |
| 中国电建 (601669.SH) | 土木工程建筑业 |  | leader | 52.8% | 36.0% | 14.2% | theme role leader | theme rank 1/4 | tradeability supportive |
| 东方材料 (603110.SH) | 煤化工 |  | core | 51.8% | 37.4% | 13.8% | theme role core | theme rank 1/22 | tradeability supportive |
| 西藏矿业 (000762.SZ) | 资源 |  | core | 51.7% | 37.5% | 12.5% | theme role core | theme rank 2/29 | tradeability supportive |

## 预测监控榜

| 排名 | 股票 | 行业 | 下一交易日区间 | 5日中位预期 | 20日中位预期 | 1日上涨概率 | 可执行分 | 状态 |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | 常山北明 (000158.SZ) | 科技软件 | 19.97 ~ 20.62 | 0.4% | 0.3% | 49.0% | 52.8% | actionable |
| 2 | 华瓷股份 (001216.SZ) | 资源 | 17.93 ~ 18.83 | 0.4% | 0.3% | 49.0% | 52.0% | blocked |
| 3 | 云南能投 (002053.SZ) | 新能源电力 | 11.25 ~ 11.59 | -0.2% | -0.5% | 49.3% | 52.6% | actionable |
| 4 | 大为股份 (002213.SZ) | 计算机、通信和其他电子设备制造业 | 25.16 ~ 26.31 | -0.1% | -0.5% | 51.2% | 52.9% | actionable |
| 5 | 东方电缆 (603606.SH) | 电气机械和器材制造业 | 56.32 ~ 58.73 | 0.6% | 0.8% | 48.0% | 51.4% | blocked |
| 6 | 东方材料 (603110.SH) | 煤化工 | 16.75 ~ 17.47 | 0.2% | 0.1% | 48.7% | 51.6% | blocked |
| 7 | 西藏矿业 (000762.SZ) | 资源 | 25.98 ~ 27.41 | 0.0% | -0.2% | 49.3% | 51.3% | blocked |
| 8 | 道明光学 (002632.SZ) | 煤化工 | 12.12 ~ 12.71 | -0.0% | -0.3% | 48.9% | NA | monitor |
| 9 | 融捷股份 (002192.SZ) | 资源 | 51.67 ~ 55.02 | -0.0% | -0.3% | 49.6% | NA | monitor |
| 10 | 中国电建 (601669.SH) | 土木工程建筑业 | 5.43 ~ 5.61 | 0.5% | 0.5% | 48.9% | NA | monitor |
| 11 | 永兴材料 (002756.SZ) | 资源 | 47.84 ~ 51.17 | 0.1% | -0.0% | 49.0% | NA | monitor |
| 12 | 美诺华 (603538.SH) | 煤化工 | 20.98 ~ 21.95 | -0.0% | -0.2% | 48.2% | NA | monitor |
| 13 | 金健米业 (600127.SH) | 消费 | 6.63 ~ 6.80 | 0.1% | -0.0% | 48.8% | NA | monitor |
| 14 | 起帆电缆 (605222.SH) | 电气机械和器材制造业 | 20.33 ~ 21.53 | 0.4% | 0.5% | 49.3% | NA | monitor |
| 15 | 国电电力 (600795.SH) | 新能源电力 | 4.64 ~ 4.77 | -0.1% | -0.4% | 48.0% | NA | monitor |
| 16 | 华工科技 (000988.SZ) | 计算机、通信和其他电子设备制造业 | 73.58 ~ 77.07 | -0.2% | -0.5% | 49.4% | NA | monitor |
| 17 | 诺德股份 (600110.SH) | 计算机、通信和其他电子设备制造业 | 6.66 ~ 6.99 | -0.2% | -0.5% | 48.7% | NA | monitor |
| 18 | 兴业银行 (601166.SH) | 金融 | 18.74 ~ 19.14 | -0.3% | -0.6% | 47.5% | NA | monitor |
| 19 | 国投电力 (600886.SH) | 新能源电力 | 12.92 ~ 13.16 | -0.4% | -0.9% | 49.2% | NA | monitor |
| 20 | 百利电气 (600468.SH) | 电气机械和器材制造业 | 7.16 ~ 7.45 | 0.5% | 0.5% | 49.3% | NA | monitor |

## 可执行候选榜

| 排名 | 股票 | 行业 | 可执行分 | 目标权重 | 下一交易日区间 | 1日上涨概率 | 备注 |
|---:|---|---|---:|---:|---|---:|---|
| 1 | 大为股份 (002213.SZ) | 计算机、通信和其他电子设备制造业 | 52.9% | 0.0% | 25.16 ~ 26.31 | 51.2% | eligible |
| 2 | 常山北明 (000158.SZ) | 科技软件 | 52.8% | 0.0% | 19.97 ~ 20.62 | 49.0% | eligible |
| 3 | 云南能投 (002053.SZ) | 新能源电力 | 52.6% | 0.0% | 11.25 ~ 11.59 | 49.3% | eligible |

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
| NA | NA | NA | NA | NA | NA | NA | NA | 暂无执行计划 |

## Intraday Overlay

- 暂无分钟级执行覆盖

## 推荐解释卡

### 常山北明 (000158.SZ)
- 行业: 科技软件
- 最新收盘价: 20.17
- 下一交易日(2026-03-23)预期区间: 19.97 ~ 20.62，中位预期 0.4%，上涨概率 49.0%，置信度 44.1%
- 多周期判断: 5日 0.4% / 10日 0.3% / 20日 0.3%
- 可执行状态: actionable，可执行分 52.8%，备注: passed execution gate
- 入池原因: 行业内相对强度 57.2%，位于板块前列。；量价结构稳定，交易一致性 98.8%。
- 排名原因: 趋势延续与稳定性加分较高。；行业内相对强度为排名提供支撑。；多周期信号一致，分歧不大。
- 未入组合原因: 当前只开 0 个仓位，这只排位靠前但未进入前 0。
- 风险点: 当前未见明显硬性风险，但仍需服从仓位纪律。
- 失效条件: 若下一交易日收盘跌破 19.97，且 5 日上涨概率回落到 50% 下方，则本次信号失效。

### 华瓷股份 (001216.SZ)
- 行业: 资源
- 最新收盘价: 18.22
- 下一交易日(2026-03-23)预期区间: 17.93 ~ 18.83，中位预期 0.4%，上涨概率 49.0%，置信度 42.7%
- 多周期判断: 5日 0.4% / 10日 0.3% / 20日 0.3%
- 可执行状态: blocked，可执行分 52.0%，备注: actionability<0.52 / 5d_tail_risk_high
- 入池原因: 行业内相对强度 55.0%，位于板块前列。；量价结构稳定，交易一致性 97.9%。
- 排名原因: 趋势延续与稳定性加分较高。；行业内相对强度为排名提供支撑。；多周期信号一致，分歧不大。
- 未入组合原因: 当前只开 0 个仓位，这只排位靠前但未进入前 0。
- 风险点: 当前未见明显硬性风险，但仍需服从仓位纪律。
- 失效条件: 若下一交易日收盘跌破 17.93，且 5 日上涨概率回落到 50% 下方，则本次信号失效。

### 云南能投 (002053.SZ)
- 行业: 新能源电力
- 最新收盘价: 11.34
- 下一交易日(2026-03-23)预期区间: 11.25 ~ 11.59，中位预期 0.5%，上涨概率 49.3%，置信度 43.2%
- 多周期判断: 5日 -0.2% / 10日 -0.4% / 20日 -0.5%
- 可执行状态: actionable，可执行分 52.6%，备注: passed execution gate
- 入池原因: 行业内相对强度 56.2%，位于板块前列。；量价结构稳定，交易一致性 98.4%。
- 排名原因: 趋势延续与稳定性加分较高。；行业内相对强度为排名提供支撑。；多周期信号一致，分歧不大。
- 未入组合原因: 当前只开 0 个仓位，这只排位靠前但未进入前 0。
- 风险点: 当前未见明显硬性风险，但仍需服从仓位纪律。
- 失效条件: 若下一交易日收盘跌破 11.25，且 5 日上涨概率回落到 50% 下方，则本次信号失效。

### 大为股份 (002213.SZ)
- 行业: 计算机、通信和其他电子设备制造业
- 最新收盘价: 25.50
- 下一交易日(2026-03-23)预期区间: 25.16 ~ 26.31，中位预期 0.6%，上涨概率 51.2%，置信度 43.4%
- 多周期判断: 5日 -0.1% / 10日 -0.3% / 20日 -0.5%
- 可执行状态: actionable，可执行分 52.9%，备注: passed execution gate
- 入池原因: 行业内相对强度 56.0%，位于板块前列。；量价结构稳定，交易一致性 98.3%。
- 排名原因: 趋势延续与稳定性加分较高。；行业内相对强度为排名提供支撑。；多周期信号一致，分歧不大。
- 未入组合原因: 当前只开 0 个仓位，这只排位靠前但未进入前 0。
- 风险点: 当前未见明显硬性风险，但仍需服从仓位纪律。
- 失效条件: 若下一交易日收盘跌破 25.16，且 5 日上涨概率回落到 50% 下方，则本次信号失效。

### 东方电缆 (603606.SH)
- 行业: 电气机械和器材制造业
- 最新收盘价: 57.18
- 下一交易日(2026-03-23)预期区间: 56.32 ~ 58.73，中位预期 0.3%，上涨概率 48.0%，置信度 44.0%
- 多周期判断: 5日 0.6% / 10日 0.7% / 20日 0.8%
- 可执行状态: blocked，可执行分 51.4%，备注: actionability<0.52 / short_prob_stack<0.500
- 入池原因: 行业内相对强度 55.6%，位于板块前列。；量价结构稳定，交易一致性 98.3%。
- 排名原因: 趋势延续与稳定性加分较高。；行业内相对强度为排名提供支撑。；多周期信号一致，分歧不大。
- 未入组合原因: 当前只开 0 个仓位，这只排位靠前但未进入前 0。
- 风险点: 当前未见明显硬性风险，但仍需服从仓位纪律。
- 失效条件: 若下一交易日收盘跌破 56.32，且 5 日上涨概率回落到 50% 下方，则本次信号失效。

## 预测复盘

| 窗口 | 命中参考 | 平均边际 | 近窗表现 | 样本数 | 说明 |
|---|---:|---:|---:|---:|---|
| 5日预测命中参考 | 54.0% | -0.4% | 0.0% | 158 | 研究期内 5 日横截面命中率与头尾价差。 |
| 20日预测命中参考 | 41.8% | -0.5% | 0.0% | 158 | 研究期内 20 日横截面命中率与头尾价差。 |
| 60日策略表现 | 56.7% | -0.1% | 3.2% | 60 | 截至 2026-02-12 的滚动净值表现。 |
- 复盘参考来自研究 run_id=20260322_211044

## 淇℃伅灞傛憳瑕?

| 鎸囨爣 | 鏁板€?|
|---|---:|
| info shadow enabled | false |
| info items | 0 |
| market catalyst | 0.0% |
| market event risk | 0.0% |
| market negative risk | 0.0% |
| market coverage | 0.0% |