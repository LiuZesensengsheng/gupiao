# A 股量化研究与生产原型（V2 主线）

这个仓库当前的主线已经不是早期的 `V1` 单体脚本，而是以 `V2` 为中心的量化研究/生产原型。

当前目标是把这条链路稳定下来：

- 数据读取与标准化
- 多层预测（大盘 / 板块 / 个股 / 横截面）
- 策略层（仓位、板块预算、个股分配、交易计划）
- 回测与研究报告
- 生产日运行与 HTML 看板

`V1` 入口仍然保留用于兼容，但不再是推荐主入口。

## 当前状态

当前 V2 已经可以运行两条主流程：

1. 生产日运行
   - 入口：`python3 run_v2.py daily-run`
   - 输出：最新预测状态、策略决策、交易计划、Markdown 报告、HTML 看板

2. 研究回测
   - 入口：`python3 run_v2.py research-run`
   - 输出：基线回测、策略校准、学习型策略回测、研究报告、HTML 看板、研究产物

当前系统更准确地说是“可运行的量化研究平台原型”，还不是最终生产系统。

## 推荐入口

### 日常运行

```bash
python3 run_v2.py daily-run --strategy swing_v2 --source local --universe-file config/universe_smoke_5.json
```

常用输出：

- `reports/v2_daily_report.md`
- `reports/v2_daily_dashboard.html`

### 轻量研究（推荐先用）

```bash
python3 run_v2.py research-run --strategy swing_v2 --source local --universe-file config/universe_smoke_5.json --light
```

`--light` 会：

- 只跑基线回测
- 跳过策略校准
- 跳过学习型策略训练
- 不发布研究产物

这适合先做快速验证，尤其是大一些的股票池。

### 完整研究

```bash
python3 run_v2.py research-run --strategy swing_v2 --source local --universe-file config/universe_smoke_5.json
```

完整模式会执行：

- 基线回测
- 参数化策略校准
- 学习型策略训练与回测
- 产物发布到 `artifacts/v2`

## 训练栈（当前真实实现）

当前默认预测后端是 `linear`，不是深度学习。

### 预测层

- 大盘预测：
  - 逻辑回归（`1d / 5d / 20d`）
- 个股面板预测：
  - 逻辑回归（`1d / 5d / 20d`）
  - 分位数线性回归（`q10 / q30 / q50 / q70 / q90`）
- 板块层：
  - 独立板块模型
- 横截面层：
  - 规则型横截面状态聚合

核心实现位置：

- [src/infrastructure/modeling.py](/Users/liuzesen/gupiao/src/infrastructure/modeling.py)
- [src/infrastructure/forecast_engine.py](/Users/liuzesen/gupiao/src/infrastructure/forecast_engine.py)
- [src/application/v2_services.py](/Users/liuzesen/gupiao/src/application/v2_services.py)

### 策略学习层

- Ridge 回归
- 当前学习目标：
  - 总仓位
  - 持仓数
  - 换手上限

### 预测目标

当前个股层已经不是单纯“涨跌二分类”，而是组合使用：

- `1d / 5d / 20d` 概率头
- 分位数头（`q10/q30/q50/q70/q90`）
- 相对收益目标：
  - 相对大盘超额
  - 相对板块超额

## 当前架构

主目录分层：

- `src/domain`
  - 领域对象与策略规则
- `src/application`
  - V2 编排、研究流、日运行流
- `src/infrastructure`
  - 数据、特征、模型、预测引擎
- `src/interfaces`
  - CLI 与 Markdown/HTML 展示层

架构文档：

- [docs/ARCHITECTURE.md](/Users/liuzesen/gupiao/docs/ARCHITECTURE.md)
- [docs/ARCHITECTURE_V2.md](/Users/liuzesen/gupiao/docs/ARCHITECTURE_V2.md)

### 当前 V2 的关键边界

1. 预测层
   - 负责训练模型并生成可交易状态
2. 策略层
   - 负责将状态映射成仓位、板块预算、目标持仓
3. 执行层
   - 负责成交约束、换手、成本、滑点
4. 报告层
   - 负责 Markdown/HTML 输出

## V2 的主要能力

### 1. 多层预测

- 大盘预测
- 板块预测
- 个股统一面板预测
- 横截面状态（风格 / 资金 / 宽度）

### 2. 策略层

- 总仓位控制
- 板块预算
- 板块内选股
- 单票上限
- 最小调仓阈值
- 显式交易状态过滤：
  - `normal`
  - `halted`
  - `delisted`
  - `data_insufficient`

### 3. 回测

- 滚动重训
- 成本与滑点
- 流动性限制
- 部分成交
- 多周期横截面指标：
  - `1d / 5d / 20d`
  - `RankIC`
  - `TopK 命中率`
  - `头部分层收益`
  - `头尾价差`

### 4. 展示与产物

- Markdown 报告
- 中文 HTML 看板
- 研究产物：
  - `dataset_manifest.json`
  - `policy_calibration.json`
  - `learned_policy_model.json`
  - `backtest_summary.json`

## 性能现状

当前性能问题的核心不是“模型太复杂”，而是“研究流冷启动要先构建完整预测轨迹”。

### 已完成的性能优化

1. `research-run` 轻量模式
   - `--light`
   - 用于快速只跑基线验证

2. 单次研究流内部复用同一条预测轨迹
   - baseline / calibration / learning 不再各自重训预测层

3. 跨命令磁盘缓存
   - 新增参数：
     - `--cache-root`
     - `--refresh-cache`
   - 会缓存“准备数据 + 预测轨迹”

4. 预测后端抽象
   - 新增 `--forecast-backend`
   - 当前只支持：
     - `linear`

### 当前已验证的性能结论

在当前本地环境下：

- `research-run --light`（smoke 池，5 只）
  - 冷启动约 16.9 秒
  - 缓存命中约 1.6 秒

- `daily-run`（80 只，中等池）
  - 约 12 秒

- `daily-run`（300 只）
  - 约 36 秒

- `research-run --light`（80 只）首次冷启动
  - 仍然明显偏慢，当前已验证为主要瓶颈

### 推荐运行策略

1. 先用 `smoke_5` 做功能验证
2. 再用 `--light` 跑中池验证
3. 命中缓存后再做重复分析
4. 大池首次研究不要直接跑完整模式

## 可扩展性（深度模型）

当前还没有接入深度模型，但架构已经开始为它预留位置。

现在已经有：

- 预测轨迹构建与策略回放解耦
- `ForecastBackend` 抽象
- 默认 `LinearForecastBackend`

这意味着后续可以新增：

- `DeepForecastBackend`
- `TreeForecastBackend`

只要它们最终产出同样的预测轨迹对象，下游的：

- 策略层
- 执行层
- 回测
- 报告

都可以复用。

当前仍未完成的是：

- 深度模型后端的具体实现
- GPU / batch / checkpoint 管理
- 更适合深度模型的样本缓存与特征张量化

## 当前已知限制

1. `daily-run` 目前仍会在运行时拟合一轮最新模型，不是纯推理。
2. `research-run` 的首次冷启动在中池及以上仍然偏慢。
3. 默认预测后端仍是线性模型，不是深度模型。
4. 学习型策略目前仍落后于规则策略，暂时不应当视为主策略。
5. `V1` 代码仍保留，仓库还处于新旧并行阶段。

## 推荐的下一步

按优先级，当前最值得继续做的是：

1. 给 `daily-run` 也加上预测/特征缓存，减少重复训练。
2. 新增真正的 `DeepForecastBackend` 骨架，先打通 `linear -> deep` 的切换路径。
3. 给缓存加入“数据新鲜度”校验，避免底层数据变化后误命中旧缓存。
4. 继续优化研究冷启动，尤其是 80+ 股票池的首轮建轨迹速度。
5. 逐步收缩旧 `V1` 文档和入口，减少认知噪音。

## 旧入口说明

以下入口仍在仓库中，但不再是推荐主入口：

- `python3 run_api.py forecast`
- `python3 run_api.py daily`
- `python3 run_api.py discover`

它们主要用于兼容旧流程。新功能应优先放到 `V2`。

## 开发提醒

当前工作树里通常会同时存在三类内容：

1. 源码改动
2. 测试改动
3. 本地运行产物（`reports/`、`artifacts/`）

提交时建议把运行产物和源码分开，不要混在一个 commit 里。
