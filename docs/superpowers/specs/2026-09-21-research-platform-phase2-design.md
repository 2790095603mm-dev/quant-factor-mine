# 量化研究平台第二阶段设计

日期：2026-09-21

## 目标与边界

本阶段在现有 Streamlit 单机量化研究平台上增加五项能力：Factor Compare、Correlation Analysis、Multi-Factor Lab、Dataset / Universe 管理、Strategy Compare，以及统一 Job / Cache。现有 Experiment、Factor Library、Factor Pipeline 和 Backtest Engine 是稳定基础设施，新功能必须通过复用公开接口接入，不改变已有计算口径，不迁移到 React，不增加 AI 或遗传算法。

完成后的主流程为：

```text
Dataset / Universe
        ↓
Job Manager → Cache
        ↓
Factor Compare ── Multi-Factor Lab
        ↓                ↓
Factor Pipeline / Factor Library
        ↓
Backtest Engine
        ↓
Experiment → Strategy Compare
```

## 总体架构

采用本地版本化注册表方案，与现有 JSON、CSV、Parquet 研究账本保持一致：

- `qfm/data/catalog.py` 管理 Dataset 与 Universe 元数据，不承载行情计算。
- `qfm/jobs/` 管理同步 Job 生命周期、确定性缓存键与 pickle/JSON 结果缓存。
- `qfm/analysis/` 提供因子对比、相关性和策略对比的纯计算接口。
- `qfm/multifactor/` 管理综合因子定义、持久化与恢复。
- 新页面只做参数收集和结果展示，核心计算继续调用现有 Factor Pipeline、`factor_report`、`synthesize`、`run_backtest`、`standard_metrics` 与 ResearchStore。

Job 第一阶段不引入后台队列。用户点击后在当前 Streamlit 进程同步执行，但状态严格经历 `PENDING → RUNNING → SUCCESS/FAILED` 并持久化，因此未来可以在不改变页面和缓存契约的前提下替换为异步执行器。

## 1. Factor Compare 与 Correlation Analysis

### 输入与口径

用户从 Factor Library 同时选择 2–12 个因子，选择前瞻周期和日期区间。每个因子使用当前确切 `factor_version`，统一经过现有 Factor Pipeline，并按方向调整为“数值越高越好”。

对比表包含：

- IC：方向调整后的 Pearson IC 均值。
- Rank IC：方向调整后的 Spearman IC 均值。
- ICIR：Rank IC 均值 / Rank IC 标准差。
- Long Short Return：现有五分层预测统计中的最高层减最低层平均前瞻收益。
- Turnover：现有 `turnover_ratio` 口径。
- Coverage：Pipeline 最终 signal 阶段有效单元格占比。
- Stability：有效交易日中 Rank IC 大于零的比例。因子已统一方向，因此数值越高表示预测方向越稳定。

### 相关性

Pearson 与 Spearman 都按“逐日截面相关系数，再对日期取均值”计算，避免把时间序列和横截面混为一谈：

- Pearson 使用方向调整、清洗后的因子值。
- Spearman 先在每日截面内排名，再计算 Pearson。
- 对角线固定为 1；有效股票不足 20 只的日期不参与均值。
- 任何非对角组合 `|corr| > 0.7` 都进入“高度重复因子”表，并在热力图中使用红色边界/注释明显标记。

结果页面由指标对比表、Pearson 热力图、Spearman 热力图和重复因子清单组成。计算通过 `FACTOR_ANALYSIS` Job 执行并缓存。

## 2. Multi-Factor Lab

### 支持的权重模式

首期支持：

- Equal Weight：所有因子等权。
- IC Weight：只使用当时已实现的历史收益，以滚动 IC 正值归一化。
- ICIR Weight：以滚动 ICIR 正值归一化。
- IC × IR Weight：以滚动 `IC × ICIR` 正值归一化。

后三种模式继续沿用现有 walk-forward 约束：估计日剔除最近 `horizon` 天尚未完全实现的未来收益，按设定窗口和调仓频率更新权重，避免未来函数。

### 综合因子持久化

用户必须输入唯一名称和说明。综合因子定义至少记录：

- 名称、版本、组成因子及各自版本/哈希。
- 权重模式、horizon、lookback、权重调仓频率。
- 是否正交化及控制变量。
- 创建时间和稳定定义哈希。

定义保存到 `data_cache/factors/composites.json`。应用启动时重建综合因子的 Python 计算闭包，并通过现有 `register_factor` 注册到 Factor Library 的“多因子”家族。因而它可以像内置因子一样进入 Factor Analysis、Factor Compare、再次合成和 Backtest。综合因子不会只存在于当前 Session；重启后仍可使用。

为避免递归和无法复现，综合因子只能引用已存在且版本可定位的基础/综合因子；禁止直接或间接引用自身。引用因子发生版本漂移时保留旧综合定义，并在页面提示版本不一致，不静默改写。

Multi-Factor 计算通过 `MULTI_FACTOR` Job 执行并缓存；保存定义本身不缓存。

## 3. Dataset 与 Universe 管理

### Dataset 数据契约

每个 Dataset 版本至少记录：

```text
dataset_id
dataset_version
source
start_date
end_date
last_update
symbols
fields
```

此外记录 `market_fingerprint`、`fundamental_fingerprint` 和 `created_at`。`dataset_version` 由数据源、日期轴、证券集合、字段集合及现有数据快照指纹生成，格式为 `dataset_<sha256前16位>`。相同内容得到相同版本；任何数据或字段变化都会得到新版本。

Dataset 注册表保存在 `data_cache/catalog/datasets.json`。现阶段 Dataset 指向现有本地行情/财务缓存，不复制一份 172MB 数据。

### Universe 数据契约

内置 Universe：

- `cn_hs300`：沪深300。
- `cn_zz500`：中证500。
- `cn_zz1000`：中证1000。
- `cn_all_a`：全A。

为兼容已有配置，保留 `index800` 和 `full` 作为旧别名，读取旧 Experiment 时仍可复现。新页面显示中文标准名称。

自定义股票池通过粘贴代码或上传 CSV/TXT 创建，标准化为六位代码、去重并排序。记录 `universe_id`、名称、来源、symbols、版本、创建/更新时间；存储在 `data_cache/catalog/universes.json`。空股票池、非法代码或重复名称必须给出明确错误。

### Experiment 绑定策略

所有新保存的 Experiment 必须同时写入：

- `dataset_id`
- `dataset_version`
- `universe_id`
- `universe_version`

ResearchStore 保存新格式时拒绝缺失绑定。格式版本升级但继续读取历史格式。

已有 Experiment 不伪造当前数据版本：读取时标记 `legacy_unbound=true`，页面明显显示“历史未绑定”，仍允许查看和导出，但重跑前要求用户明确选择一个 Dataset/Universe 版本。这样保留旧记录又不制造虚假的可复现性。

## 4. Strategy Compare

Strategy Compare 是独立页面，可从所有研究项目中选择 2–8 个已完成 Backtest Experiment。失败运行和缺少净值产物的运行不可选择。

统一指标全部基于现有 `standard_metrics()`：

- Annual Return：年化收益。
- Excess Return：年化超额。
- Sharpe：夏普比率。
- Max Drawdown：最大回撤。
- Calmar：卡玛比率。
- Turnover：年化换手。

图表包含：

- 多策略累计净值。
- Benchmark。若不同实验基准不一致，分别绘制并标明来源；相同曲线只绘制一次。
- Excess Return Curve：策略净值与同实验基准净值归一化后的相对净值减一。
- Drawdown：每个策略的回撤序列。

图表按共同时间轴对齐，但不强行截断指标计算；每个策略的指标仍基于自身完整保存区间。页面同时展示 Dataset/Universe 版本，避免把不同样本实验误当成纯参数对比。

## 5. Job 与 Cache

### Job 模型

Job 至少记录：

- `job_id`
- `job_type`
- `status`
- `cache_key`
- `created_at`、`started_at`、`finished_at`
- `request`
- `result_path`
- `cache_hit`
- `error`

状态只允许：`PENDING`、`RUNNING`、`SUCCESS`、`FAILED`。Job manifest 保存到 `data_cache/jobs/manifests/`，结果保存到 `data_cache/jobs/results/`。

统一纳管的 Job 类型：

- `FACTOR_COMPUTE`
- `FACTOR_ANALYSIS`
- `MULTI_FACTOR`
- `BACKTEST`

### 缓存键

缓存键使用排序后的规范 JSON 计算 SHA-256，必须包含：

- `factor_version`：多个因子时包含名称、版本、source_hash 的有序列表。
- `dataset_version`
- `universe_id` 与 `universe_version`
- `date_range`
- `pipeline_config`
- `backtest_config`
- `job_type`
- `cache_schema_version`

只有以上字段完全相同才命中。日期、元组、Numpy 标量等先转换为稳定 JSON 类型。缓存命中仍创建一条 `SUCCESS` Job 记录，并设置 `cache_hit=true`，便于审计“这次为什么没有重新计算”。

计算成功时先将结果写临时文件，再原子替换目标文件；失败不留下可命中的半成品。读取损坏缓存时将其视为 miss，重新计算并记录原因。

## UI 设计

沿用现有紫、粉、淡蓝玻璃金融研究台，不重做全局布局。新页面强调“研究工作台”而非营销仪表盘：

- 左侧导航新增“因子对比”“多因子实验室”“策略对比”“数据与股票池”“任务中心”。
- 比较页使用紧凑指标表和双热力图；高相关单元格使用红色描边和 `HIGH` 标记，成为本阶段的视觉识别元素。
- Job 状态使用一致的文字徽章：灰色 PENDING、蓝色 RUNNING、绿色 SUCCESS、红色 FAILED；不只依赖颜色。
- 空状态直接说明下一步动作；失败状态展示可操作原因，不吞掉底层错误。
- 不修改已有页面视觉层级和控件名称，避免破坏已有 AppTest 与用户习惯。

## 错误处理与兼容性

- Dataset/Universe 注册表采用原子写入；损坏单条记录被隔离并提示，不阻断整个应用。
- 因子缺失、版本漂移、循环依赖、样本不足、缓存损坏分别产生明确错误。
- 旧 ResearchRun v1/v2 继续读取；新格式为 v3。
- 旧 `index800/full` 配置继续解析。
- Job 层只包装现有计算，不改变 Factor Pipeline、Backtest Engine 的数值口径。
- 不自动修改或删除任何已有 Experiment、因子定义、行情缓存和报告。

## 测试与验收

测试按 TDD 增加：

1. Factor Compare 指标、Pearson/Spearman、阈值标记和日期轴对齐。
2. 四种多因子权重无未来函数、权重归一化、持久化、重启恢复与循环依赖拒绝。
3. Dataset/Universe 建档、版本稳定性、自定义池校验和 Experiment v3 强制绑定。
4. Strategy Compare 指标及四类曲线，兼容不同日期和基准。
5. Job 状态转换、成功/失败记录、缓存命中、配置变化导致 miss、损坏缓存回退。
6. Streamlit 页面冒烟测试，验证主要控件、空状态和结果呈现。
7. 运行现有完整测试集，确保既有 Experiment、Factor Library、Factor Pipeline 和 Backtest Engine 全部回归通过。
8. 使用真实 localhost 服务逐页验证，并检查后台日志无新 traceback。

## 明确不做

- AI 自动生成因子。
- 遗传算法/符号回归扩展。
- React 或前后端分离重构。
- 分布式任务队列、账户权限和云存储。
- 自动猜测旧 Experiment 的 Dataset/Universe 版本。
