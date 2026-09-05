# 研究运行约束执行审计归档设计

## 目标

让已保存的研究运行保留组合约束执行审计。研究员重新打开项目时，应能核对某次调仓的目标现金、实际现金、跟踪偏离和收盘实际暴露，而不必重新运行回测。

## 方案与决策

1. 仅将约束配置写入 JSON manifest。它能说明计划的上限，却不能复核实际执行差异。
2. 将审计明细塞入 manifest。表格字段会随回测演进，JSON 既臃肿又不利于用户导出。
3. 为存在审计记录的运行写入可选的 `constraint_history.csv`，并在 manifest 的 `artifacts` 中索引它。

采用方案 3。它和已保存的净值、权重、成交 CSV 保持一致，记录量只与调仓次数成正比。没有约束记录的普通运行不创建该文件；历史 manifest 缺少该索引时也照常读取。因此本期不升级 `FORMAT_VERSION`，不重写历史研究记录。

## 存储接口

`ResearchStore.save_run` 在最后增加可选参数 `constraint_history: pd.DataFrame | None = None`。只有传入非空表格时：

- `ResearchRun.artifacts` 增加 `constraint_history: constraint_history.csv`；
- CSV 不保存 DataFrame 索引；
- `signal_date` 和 `execution_date` 由 CSV 写出。

`LoadedResearchRun` 新增末尾默认字段 `constraint_history: pd.DataFrame | None = None`，使现有构造方式保持兼容。读取时，若 manifest 有该产物，则读取 CSV，并将已存在的 `signal_date`、`execution_date` 解析为 pandas 时间戳；若没有该索引，字段为 `None`。

`build_strategy_run_payload` 从 `BacktestResult.constraint_history` 复制非空表格到 `constraint_history` 键；空值保持为 `None`。策略页保存时将该键透传给 store，不触发重新回测。

## 页面呈现

研究项目详情在“成交流水”之后显示可折叠的“组合约束执行审计（N 次）”，内容为归档的原始审计表。导出区域在有该表时增加“约束审计 CSV”；无审计运行仍保持原来的五个导出按钮。

页面只呈现已归档的行，不在项目页重新计算目标、行业暴露或超限结论。策略页现有的即时执行审计继续保留。

## 验收

1. 含日期字段的非空约束审计表可以保存、读取，日期和全部数值/布尔字段保持一致，manifest 正确索引 CSV。
2. 未传审计表的新运行与历史 manifest 均可读取，`LoadedResearchRun.constraint_history is None`。
3. 策略载荷包含非空的回测审计表，不修改约束配置和既有回测产物。
4. 研究项目页面仅在归档表存在时显示审计详情和第六个下载项。
5. 零网络 pytest、研究账本冒烟、Python 编译、Streamlit 健康检查与隔离页面验收全部通过。

