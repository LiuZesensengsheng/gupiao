# RFC：量化架构重构方案

- 状态：提议中
- 日期：2026-03-14
- 负责人：Codex + 项目维护者
- 范围：`V2` 研究流程与日常生产流程架构

## Problem

当前 `V2` 的方向是对的，但代码层面的落地还不完整。

仓库目前其实已经有这些关键概念：

- 研究工作流
- 日常生产工作流
- 已发布产物
- 报告输出

但这些概念在代码里的边界还不够清晰。

最明显的问题点是 [`src/application/v2_services.py`](/D:/gupiao/src/application/v2_services.py)，它现在同时承担了：

- 运行配置解析
- `research-run` 编排
- `daily-run` 编排
- 缓存处理
- 冻结快照加载
- 策略学习
- 产物发布
- 汇总结果生成

这会带来四个直接问题：

1. 架构意图已经写进文档，但还没有真正落进代码边界。
2. 研究流和日跑流共享了大量逻辑，但共享方式更多是隐式的，而不是通过稳定 contract。
3. 性能优化很难做，因为数据准备、打分、发布之间的边界不明确。
4. 未来如果增加第二套策略或第二种 forecast backend，复杂度会继续快速失控。

这份 RFC 的目标不是推翻 `V2`，而是把 `V2` 变成一个可以长期维护的架构。

## Constraints

这份方案基于下面这些实际约束：

- 这是一个股票量化系统，主要是批处理型，不是超低延迟撮合系统。
- 当前最核心的工作流仍然是 `research-run` 和 `daily-run`。
- 已有测试应当尽可能保留价值，不接受完全重写。
- 现有代码已经有业务价值，应该渐进演进，而不是推倒重来。
- 研究产物需要被生产流程消费，所以 artifact 兼容性很重要。
- 项目同时需要：
  - 足够快的研究迭代速度
  - 足够稳的日常运行能力

非目标：

- 不引入微服务
- 不引入分布式计算基础设施
- 不重写整套策略逻辑
- 不在一个阶段内完成全仓库的大迁移

## Complexity

### 本质复杂度

这些复杂度是量化系统本来就会有的：

- 行情与市场数据接入和标准化
- 特征工程
- 预测模型训练与打分
- 状态组合
- 策略与组合构建
- 回测与执行模拟
- 研究产物发布
- 面向人的报告输出

### 偶然复杂度

这些是当前架构带来的额外复杂度，不是业务本身必须要有的：

- 一个过大的 orchestrator 模块
- CLI 参数面过宽、重复太多
- 一部分 support 模块已经拆出去了，但核心逻辑仍然堆在单体里
- 缓存边界不清晰，不容易判断复用点
- 业务逻辑和 artifact 逻辑混在同一个文件里
- reporting 模块越来越大，但它们消费的数据边界并不稳定

### 当前热点

当前最值得关注的代码热点包括：

- [`src/application/v2_services.py`](/D:/gupiao/src/application/v2_services.py)
- [`src/interfaces/cli/run_v2_cli.py`](/D:/gupiao/src/interfaces/cli/run_v2_cli.py)
- [`src/interfaces/presenters/html_dashboard.py`](/D:/gupiao/src/interfaces/presenters/html_dashboard.py)
- [`src/interfaces/presenters/markdown_reports.py`](/D:/gupiao/src/interfaces/presenters/markdown_reports.py)

当前真正的问题，不是模型太先进，也不是策略太复杂，而是编排、contract 和 artifact ownership 还没有真正清晰。

## Options

### Option A：维持现有分层，只继续做局部抽取

说明：

- 保持 `domain / application / infrastructure / interfaces`
- 继续从 `v2_services.py` 往外抽帮助函数
- 尽量避免明显的架构变化

优点：

- 短期代码扰动最小
- 文件移动最少
- 能继续沿着现有方式推进

缺点：

- `application` 很可能继续变成兜底层
- 边界仍然偏隐式
- 新功能大概率还会继续往 orchestrator 周围堆
- 性能优化仍然缺乏明确的职责边界

适用条件：

- 只有一套策略
- 变化频率不高
- 团队规模小
- 对 artifact 生命周期治理要求不高

### Option B：在当前单仓库里引入明确的 bounded contexts

说明：

- 保持一个仓库、一个整体运行时
- 按量化生命周期职责拆模块，而不只是按通用技术层分层
- 保留 `interfaces` 作为入口层
- 让 `application` 收缩成 workflow 与 facade
- 让核心实现按业务上下文归位

目标上下文：

- `data_platform`
- `feature_dataset`
- `forecast_research`
- `policy_portfolio`
- `backtest_simulator`
- `artifact_registry`
- `daily_runtime`
- `reporting`

优点：

- 最适合当前项目规模
- 支持渐进迁移
- 能把研究和生产的边界真正做实
- 缓存和 artifact 的职责能有清晰归属
- 更适合未来增加多个策略或多个 forecast backend

缺点：

- 中等规模的重构成本
- 迁移过程中会短暂存在部分重复
- 需要对模块职责保持纪律

适用条件：

- 批处理量化工作流
- 日频或收盘后生产运行
- 研究迭代频繁
- 希望沉淀稳定的研究产物

### Option C：改成事件驱动、多服务架构

说明：

- 将数据、预测、组合、报告、调度拆成多个服务
- 通过队列或 API 通信

优点：

- 隔离性强
- 适合更大团队和更实时的运行模式

缺点：

- 运维复杂度高
- 会明显拖慢研究迭代
- 对当前项目规模来说不划算

适用条件：

- 高频或盘中系统
- 多账户、多策略并行
- 已有成熟运维能力

## Risks

### 如果什么都不做

- `v2_services.py` 会继续吸收新逻辑
- 研究流和日跑流会逐步出现行为漂移
- artifact 发布流程会越来越难理解
- 缓存会继续变多，但不透明
- 加第二种 forecast backend 的成本会比预期高很多

### 如果执行推荐的重构

- 迁移过程中会有阶段性重复
- 如果移动顺序不慎，容易出现 import 破坏
- 即使测试通过，也可能存在 artifact 兼容性细节变化
- 团队可能会过早去清理 reporting，而不是先收紧 workflow 边界

### 缓解措施

- 迁移期间保持公共入口不变
- 先把实现抽到 facade 背后，再考虑调整 CLI 行为
- 在大规模搬迁前补齐 artifact contract 校验
- 按阶段迁移，而不是整体重写

## Recommendation

推荐采用 **Option B：在当前单仓库中引入明确的 bounded contexts**。

这是现在最适合这个项目的选择。

它保留了 `V2` 已经做对的部分：

- 研究与生产分离
- forecast layer 与 policy layer 分离
- 通过 artifact 让生产消费研究结果

同时把这些设计真正落到代码边界里。

### 架构原则

1. 研究流程产出不可变 artifact。
2. 生产日跑默认只消费已发布 artifact，不主动重训。
3. Forecast 逻辑负责预测状态，不直接输出交易。
4. Policy 逻辑负责决定仓位、集中度和目标权重。
5. Reporting 只负责渲染，不持有业务决策。
6. Orchestrator 只做编排，不承载核心业务实现。
7. 缓存应绑定到明确的数据产品，而不是零散的临时步骤。

### 目标模块布局

建议的中期目标结构如下：

```text
src/
  domain/
    entities.py
    policies.py
    news.py
    symbols.py

  contracts/
    artifacts.py
    runtime.py
    reporting.py

  workflows/
    research_workflow.py
    daily_workflow.py
    workflow_blueprints.py

  data_platform/
    market_data.py
    data_sync.py
    discovery.py
    security_metadata.py
    info_repository.py
    margin_sync.py

  feature_dataset/
    features.py
    market_context.py
    panel_dataset.py
    sector_data.py
    external_signal_features.py
    margin_features.py

  forecast_research/
    modeling.py
    forecast_engine.py
    sector_forecast.py
    cross_section_forecast.py
    v2_info_fusion.py

  policy_portfolio/
    candidate_selection.py
    sector_support.py
    mainline_support.py
    strategy_memory.py

  backtest_simulator/
    backtesting.py
    effect_analysis.py

  artifact_registry/
    publish_support.py
    snapshot_support.py
    backtest_cache_support.py

  daily_runtime/
    daily_runtime.py
    external_signal_support.py

  reporting/
    markdown_reports.py
    html_dashboard.py
    driver_explainer.py

  interfaces/
    cli/
      run_v2_cli.py
      run_api_cli.py
```

这不是要求一次性全部改完，而是作为目标架构来推进。

### 运行流边界

#### Research Workflow

```text
config
-> data_platform
-> feature_dataset
-> forecast_research
-> policy_portfolio
-> backtest_simulator
-> artifact_registry
-> reporting
```

研究流程产出：

- `dataset_manifest`
- `forecast_bundle`
- `policy_calibration`
- `policy_model`
- `strategy_snapshot`
- `backtest_summary`
- `research_manifest`

#### Daily Workflow

```text
config
-> data_platform
-> artifact_registry(加载已发布快照)
-> daily_runtime(打最新状态分)
-> policy_portfolio(应用策略)
-> reporting
```

日跑流程产出：

- 当前状态快照
- 推荐结果
- 交易计划
- 日报
- dashboard 数据

### Contract 设计

这个架构里最重要的边界，不是某个 Python 帮助函数，而是 artifact contract。

建议把下面这些变成明确、稳定、可版本化的 contract：

- `DatasetManifest`
- `ForecastBundle`
- `PolicyCalibrationArtifact`
- `LearnedPolicyArtifact`
- `StrategySnapshot`
- `ResearchManifest`
- `DailyReportViewModel`

建议：

- 给这些 contract 做版本号
- 序列化/反序列化规则统一管理
- 加载时做必填字段校验

### 迁移计划

#### Phase 1：冻结外部行为，先瘦身 orchestrator

目标：

- 保持 CLI 与输出稳定
- 把实现细节从 `v2_services.py` 中迁出

动作：

- 让 [`src/application/v2_services.py`](/D:/gupiao/src/application/v2_services.py) 只保留 facade 或兼容入口
- 将 research-run 的实现迁到 `workflows/research_workflow.py`
- 将 daily-run 的实现迁到 `workflows/daily_workflow.py`
- 将 artifact 发布与加载迁到 `artifact_registry/*`

阶段完成标志：

- `v2_services.py` 主要只剩导入和 facade 函数
- 研究流和日跑流可以分别独立阅读

#### Phase 2：引入类型化运行配置 contract

目标：

- 缩小参数扩散和 CLI 重复

动作：

- 定义 `ResearchRunOptions` 与 `DailyRunOptions`
- 抽出共享的 CLI 参数注册函数
- workflow 边界不再传几十个零散 keyword arguments

阶段完成标志：

- workflow 入口函数只接收一个 options 对象和少量依赖
- CLI 代码明显变薄

#### Phase 3：规范 artifact contract

目标：

- 让研究到生产的交接边界清晰、可测试

动作：

- 增加版本化 artifact dataclass 或 schema
- 在使用前校验快照 payload
- 统一 path 解析和 manifest 加载逻辑

阶段完成标志：

- artifact 格式变化变得可见、可审查
- 测试覆盖兼容性与缺字段行为

#### Phase 4：把重计算步骤收敛成可复用数据产品

目标：

- 直接命中真实性能瓶颈

动作：

- 明确拆开：
  - 原始数据加载
  - `market_frame` 构建
  - `panel_dataset` 构建
  - `forecast_bundle` 冻结
- 对这些阶段建立明确 key 和 manifest 的缓存

阶段完成标志：

- 重复 research-run 能稳定复用 prepared datasets
- daily-run 能复用已发布 forecast artifact，避免重复重计算

#### Phase 5：收紧 reporting 职责

目标：

- 让 reporting 成为真正独立、薄的一层

动作：

- 引入 reporting view-model
- 把纯格式化逻辑留在 `reporting`
- 不让 markdown/html renderer 继续承载业务判断

阶段完成标志：

- reporter 只消费结构化 payload
- 报告模块不再向业务逻辑层反向渗透

## Counter-Review

这份 RFC 最强的反对意见会是：

“策略还在快速迭代，现在做架构会拖慢研究。”

这个反对意见在“如果把重构做成大重写”的前提下是成立的。

但这份 RFC 不建议这么做。

它明确要求：

- 不做大爆炸式重写
- 不先做全仓库重命名
- 不在一个阶段内同时搬所有目录

应该做的是：保持入口不动，逐步把实现抽离到合理边界里。

另一个反对意见会是：

“现在的 layered architecture 已经够用了。”

这只说对了一半。

当前的文档方向没问题，但代码实现还没有真正约束这些边界。只要 orchestrator、artifact 管理和核心业务逻辑还集中在少数几个大文件里，架构就还只是“描述性的”，不是“运行中的”。

所以这份 RFC 不是要否定当前架构，而是要把当前 `V2` 架构真正做实。

## Decision Summary

最终决策建议：

- 在当前单仓库中采用 bounded-context 架构。
- 保持一个仓库、一个整体运行时。
- 保留 `research-run` 和 `daily-run` 作为两条主工作流。
- 让 artifact 成为研究与生产之间的核心边界。
- 从 `v2_services.py` 与 CLI options contract 开始做渐进重构。

接下来应该做的事：

1. 确认这份目标架构。
2. 优先执行 Phase 1 和 Phase 2。
3. 等 workflow 与 artifact 边界稳定后，再去做更深的 reporting 清理。

接下来不应该做的事：

- 不做一次性大重写
- 不拆微服务
- 不在整个仓库里同时做大规模目录重命名

如果这次重构成功，最终效果应该是：

- 新人能很快分清研究逻辑、日跑逻辑、artifact、reporting 各自归属
- 性能优化可以明确打到具体数据产品和缓存边界
- 后续增加策略能力时，不需要继续扩大单个 god module
