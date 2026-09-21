# 量化因子挖掘流水线 (quant-factor-mine)

A 股多因子研究与挖掘工具：内置 49 个经典因子，统一因子管线 + 单因子 Tear Sheet + 带版本与复现能力的实验账本，以及 Factor/Strategy Compare、Multi-Factor Lab 和确定性 Job/Cache，附 Streamlit 操作台。

## 快速开始

```bash
# 1. 启动页面（首次会自动拉取数据，约 5-10 分钟，之后秒级）
streamlit run app.py

# 2. 命令行跑单因子检验
python -m qfm.cli --factor momentum_20 --horizon 20

# 3. 命令行跑自动挖掘
python -m qfm.cli --mine --max-candidates 200
```

## 架构

```
app.py              Streamlit 操作台（深色金融终端风）
qfm/
├── data/           数据层：akshare 拉取 + parquet 本地缓存
│   ├── loader.py   日线（前复权 + 不复权双拉，存精确复权因子）/ 财务指标
│   ├── panel.py    DataPanel（含 factor / close_raw / mv_float）+ raw_price/valuation_price
│   └── universe.py 股票池 + 显式成员生效区间（membership）
├── factors/        因子库：内置家族 + 持久化多因子家族，带版本历史
│   └── base.py     @register_factor（自动记录 version/tags/公式/source_hash）
├── pipeline/       统一因子管线与分析（自研，不依赖 alphalens）
│   ├── stages.py   管线各阶段：缺失/去极值/方向/中性化/标准化/Decay/Lag/Signal
│   ├── pipeline.py PipelineConfig + run_pipeline → SignalResult（全项目唯一信号来源）
│   ├── clean.py    去极值(MAD/分位数) + z-score（被 stages 复用）
│   ├── tests.py    IC / IC_IR / 分层 / 单调性 / 换手 / 前向对齐校验
│   ├── tearsheet.py 单因子 Tear Sheet（分组/多空双口径、分布、覆盖、行业与市值暴露）
│   └── report.py   HTML 检验报告（plotly 自包含页面）
├── orthogonalize.py 逐日截面 OLS（residualize_frame 被管线复用）
├── simulation/     单因子模拟：engine（管线入口）/ long_short / views / report
├── portfolio/      组合层：合成 / 回测（滑点、外部基准、每 N 日调仓、ST 位）/ 统一指标
├── research/       实验账本：store / models / snapshot（数据指纹）/ replay（重放复现）
├── analysis/       Factor Compare / Correlation Analysis / Strategy Compare
├── multifactor/    综合因子定义、版本与持久化注册表
├── jobs/           PENDING/RUNNING/SUCCESS/FAILED 任务记录与确定性缓存
├── mining/         自动挖掘引擎（未改动）
└── cli.py          命令行入口
```

## 第一阶段：可复现的研究基础设施

```bash
# 端到端验证（合成数据，零网络，秒级）
python scripts/verify_phase1.py

# 用真实缓存验证（需要先跑过一次数据加载）
python scripts/verify_phase1.py --real --max-stocks 300
```

六项交付：**统一 Factor Pipeline**、**Factor Tear Sheet**、**Factor Library 版本化**、
**Experiment 系统（状态/标签/版本快照/重新打开/复现）**、**Backtest 引擎补全**、
**数据复权口径修正**。关键不变量都有可执行断言（见 `tests/`）。

## 第二阶段：比较、组合、数据版本与任务缓存

新增五项能力，全部复用第一阶段的因子管线、回测引擎和实验账本：

- **因子对比与相关性**：同时比较 IC、Rank IC、ICIR、Long Short Return、Turnover、Coverage、Stability；提供 Pearson/Spearman 热力图，并突出 `|corr| > 0.7` 的重复因子。
- **多因子实验室**：支持 Equal Weight、IC Weight、ICIR Weight、IC × IR Weight。综合因子保存后进入 Factor Library，可继续检验、对比和回测；成分版本被固定，循环依赖会被拒绝。
- **Dataset / Universe 管理**：内置沪深300、中证500、中证1000、全A和自定义股票池。新实验明确绑定 `dataset_id`、`dataset_version`、`universe_id`、`universe_version`。
- **策略对比**：跨研究项目选择 2–8 个历史回测，比较 Annual Return、Excess Return、Sharpe、Max Drawdown、Calmar、Turnover，以及净值、基准、超额和回撤曲线。
- **Job / Cache**：因子计算、因子分析、多因子和回测统一记录 `PENDING → RUNNING → SUCCESS/FAILED`。因子版本、Dataset/Universe 版本、日期、Pipeline 和 Backtest 参数完全一致时直接读取缓存。

本地存储位置：

| 路径 | 内容 |
|---|---|
| `data_cache/catalog/` | Dataset 与 Universe 版本注册表 |
| `data_cache/factors/composites.json` | 综合因子的全部版本定义 |
| `data_cache/jobs/manifests/` | 每次任务的状态、请求和错误 |
| `data_cache/jobs/results/` | 按确定性缓存键保存的计算结果 |
| `data_cache/research/` | 研究项目、回测运行及导出产物 |

旧 v1/v2 实验不会被猜测性迁移：仍可查看和导出，但显示为 `legacy_unbound`，严格复现按钮停用。用当前数据重新运行并保存后，才生成可信的版本绑定。

本阶段明确不包含 AI 因子生成、遗传算法、符号回归扩展、React 重构、分布式队列或云存储。

### ⚠️ 复权口径修正（会改变既有结论）

修正前：OHLC 存前复权价，而 `outstanding_share` 是真实股本，于是
`mv_float = 前复权价 × 股本` 会**系统性低估历史市值**；`ep/bp = 真实每股指标 / 前复权价`
则分子分母量纲不一致。

修正后：行情双拉（`adjust="qfq"` 与 `adjust=""`）存精确 `factor = 前复权价 / 真实价`，
`close_raw = close / factor`，`mv_float = close_raw × 股本`，估值因子分母用 `close_raw`。

实测影响（799 只缓存、2015 年起、前瞻 20 日）：

| 因子 | 修复前 IC | 修复后 IC | 说明 |
|---|---|---|---|
| `ln_mv_float` | −0.0540 (t=−23.6) | **−0.0325** (t=−11.9) | 规模效应被高估约 66% |
| `ep_ttm` | +0.0346 | +0.0299 | −14% |
| `bp` | +0.0598 | +0.0538 | −10% |

受影响区间内旧口径对市值的中位低估为 **2.05 倍**，最大 **269 倍**（且逐股幅度不同 →
是横截面排名污染，不只是整体偏移）。**规模族与价值族的历史结论需要重跑**；
动量/波动/反转等只用收益率的因子不受影响。

### 已知的数据缺口（未实现，不是已实现功能）

- **ST 状态**：当前数据源（akshare 新浪/东财）不提供。引擎已支持 `st` 布尔面板
  （ST 按 5% 涨跌幅、且不进入选股目标），接入带标记的数据源即可生效；现在按普通股票处理。
- **历史指数成分股**：`get_universe` 取的是**当前**成分股快照，存在幸存者偏差。
  未上市期间会被 NaN 自然排除（实测 2020 年 363/400 只有效），但成分股调整前视无法用价格数据还原。
  提供 `data_cache/membership_<pool>.json`（`{code: [start, end]}`）后即按真实生效区间过滤，
  否则在运行快照里标注 `membership_source` 并给出告警。
- **财务修订链**：`merge_asof` 按公告日对齐已避开主要未来函数，但 `ak.stock_yjbb_em`
  返回的是最新版本的季报；若报表被更正过，回测用的是修正值。逐次留快照可逐步逼近真实 PIT。


## 自定义因子（页面 / 代码两入口）

```python
from qfm.factors import register_factor

@register_factor(name="my_idea", family="自定义", description="我的主观想法")
def my_idea(d: DataPanel) -> pd.DataFrame:
    # d.close / d.volume / d.turnover 是 日期×股票 的透视表
    return d.close.pct_change(20) / d.turnover.rolling(20).mean()
```

## 数据说明

- 默认股票池：沪深300 + 中证500（约 800 只，首次拉取 5-10 分钟）
- 全市场可选（约 5400 只，首次约 30-60 分钟，需耐心）
- 数据缓存在 `data_cache/`，二次加载秒级；日线为前复权
- 财务指标按"公告日期"对齐，避免未来函数

## 单因子检验与模拟

「因子检验 → 单因子检验与模拟」现在一次运行同时计算 IC 诊断和模拟组合：显示成本后日频 Sharpe、最大回撤、Fitness、年化收益、日均换手与净值曲线，支持分年表现、费用对照、持仓成交、离线 HTML/CSV 下载及研究项目归档。修改设置会保留上次结果并提示重跑。

「模拟设置」包含日期区间、持仓数量、日/周/月调仓、Delay、Decay、信号行业/市值中性化、单股上限、交易成本、无风险利率和资金规模。Delay=1 表示当日收盘信号在下一交易日开盘执行；Decay=N 表示最近 N 日的线性加权平滑。IC 使用处理后的信号和未来 H 日收盘收益，末尾不完整标签剔除。中性化是信号残差化，不保证选股后组合权重行业中性。

默认复用 A 股多头执行模型。可选的多空研究模型使用 50% 多头 / 50% 空头目标及借券费，但不模拟融券券源、涨跌停排队或成交参与率限制；小单股上限会降低总敞口。费用前后是独立模拟，成交数量可随资金差异改变。

统一绩效 Sharpe = (平均日收益 - 日无风险利率) / 日收益标准差 × sqrt(252)，取代旧版 CAGR/波动率近似。日均换手 = 每日买卖成交额 / 开盘交易前净资产，再对所有收益日（含零交易日）平均。本页 Fitness = Sharpe × sqrt(abs(平均日收益 × 252) / max(日均换手, 0.125))，是本地研究指标，不代表 BRAIN 官方评分。旧表达式排行榜仍采用月频/名单换手近似，页面已标明不可直接比较。已归档旧运行不改写。

## 保存与比较研究运行

在「研究项目」页创建并选择一个项目后，按以下顺序使用：

1. 在「策略回测」完成一次因子合成与成本后回测；
2. 在结果下方填写运行名称，点击「保存至当前项目」；
3. 回到「研究项目」横向比较保存运行的净值，并查看配置、数据快照、权重、分年绩效、成交记录和约束执行审计（如本次策略有约束记录）；
4. 按需下载保存的 CSV 产物。

每次保存会写入 `data_cache/research/`：包含参数、实际参与样本的证券代码、数据区间与覆盖率、确定性数据指纹、策略与基准净值、权重、分年绩效、成交，以及可选的约束执行审计。数据指纹可发现以后重新计算时的输入变化，但不替代原始数据备份，也不提供指数成分股的点时历史；它不会复制原始行情或重跑回测。这些记录是历史模拟研究，不能视为投资建议。

策略回测还可记录单股/行业目标权重上限和可选的单次调仓成交额预算。严格上限可能保留现金；涨跌停、停牌与参与率限制可能让实际仓位延后收敛到受约束目标。

## 方法论出处（QuantSkills 三技能融入）

| 功能 | 方法论来源 | 独立实现依据 |
|---|---|---|
| 逐日截面 OLS 正交化 | QuantSkills `skill-factor-orthogonalize`（GPL-3.0） | 公开标准截面回归；行业/市值/风格暴露剥离 |
| 回测过拟合检验 DSR/PBO/Haircut/MinTRL | QuantSkills `skill-backtest-overfit`（GPL-3.0） | Bailey & López de Prado (2012/2014)、Bailey et al. (2017)、Harvey & Liu (2015) 文献公式 |
| 无未来函数检查 | QuantSkills `skill-quant-factor-skill-factory`（GPL-3.0） | 静态泄漏模式扫描 + 挖掘候选结构性判定 |

本项目未拷贝上述仓库源码，算法按文献公式独立实现。

## 统一管线与口径约定

所有因子分析与回测共用**唯一**的信号路径（`qfm.pipeline.run_pipeline`）：

```
缺失处理 → 去极值 → 中性化 → 标准化 → 方向统一 → Decay 平滑 → Lag → Signal
```

- `SignalResult.signal` 是检验与持仓共同消费的唯一对象。此前检验用 `clean_factor`、
  回测用 `simulation/engine.py` 的内联 OLS，两者不复用，会出现「看到的 IC 与实际持仓
  来自不同信号」。现已合并，`prepare_signal` 逐值等于 `run_pipeline(...).signal`。
- 方向统一放在中性化/标准化**之后**：这两个阶段对符号不敏感，放在前后得到相同信号。
- ⚠️ **`standardize="rank"` 会破坏中性化的线性正交性**：z-score 是线性变换，保持残差与
  控制变量的正交（市值暴露≈0）；rank 是非线性单调变换，排名后 Pearson 暴露可达 0.78。
  同时启用中性化时应保持 zscore（详见 `qfm.pipeline.stages.apply_standardize`）。
- 对齐一律按**标签**而非位置：`compute_ic`/`layer_test` 内部按行向量化计算，
  传入轴不一致的未来收益会被自动 reindex，而不是静默错位。

## 实验记录（可复现）

每次保存运行会写入 `data_cache/research/runs/run_<hex>/`：

| 文件 | 内容 |
|---|---|
| `manifest.json` | 参数、数据快照、结果摘要、**状态**、**标签**、代码口径版本、**因子版本索引** |
| `nav.csv` / `benchmark_nav.csv` | 策略与基准净值 |
| `weights.csv` / `yearly_performance.csv` / `trades.csv` | 权重 / 分年绩效 / 成交流水 |
| `constraint_history.csv` | 约束执行审计（有约束时才写） |
| `factors.json` | **因子定义快照**（版本、公式源码、定义指纹） |

- 因子在 `config` 里以**定义对象**（name/version/source_hash）保存，而不是裸名字；
  因子被改动后重跑会明确告警「已从 vN 更新到 vM」，避免把复现不一致误判成数据问题。
- 「研究项目」页可**按存档参数重跑**并逐日比对存档净值，一致才算复现成功。
- 失败运行也会落盘（`status=failed` + `error.txt`），"这组参数跑不出来"本身是重要信息。

## 机器学习合成因子

因子检验页「ML 合成」Tab：LightGBM walk-forward 滚动训练把现有因子库非线性合成为一个新因子（仅样本外预测，结构性无未来函数）。合成后自动注册为 `ml_synth` 因子，可直接进入策略回测页参与合成与回测。

## 博主「每天一个因子」系列（9 个新增因子）

来源：博主公开渠道整理的 19 个因子（2026-08），公式按公开口径独立实现，实盘方向经真实缓存数据 IC 校准（5 个调整为 negative）。

新增：`bb_break_20`（布林上轨突破）、`amplitude_3`（振幅，需 OHLC）、`turnover_heat`（换手升温倍数）、`rav_4`（RSI4 变化率）、`gm_yoy`（毛利率同比近似）、`sentiment_20`（情绪）、`alpha144_191`（国泰君安191）、`rev_5`（5日反转）、`vol_ratio_20`（量能比）。

跳过（与现库重复/缺数据）：20日动量=mom_20、PIV=bp、PITTM=ep_ttm、20日波动=vol_20、低波动=vol_20(负向)、规模=ln_mv_float、长期动量=mom_250、Amihud=amihud_20、多空组合=分层检验"多空价差"、大单净流入=缺 Level2 数据源（二期）。

实证亮点（50 只 × 20 日前瞻）：`amplitude_3` IC -0.031(t=-12.4)、`alpha144_191` IC +0.033(t=11.1)、`rev_5` IC +0.024(t=9.4)。

## 「实战」家族（6 个行业格局/情绪因子，2026-08）

- `lead_cap` 龙头市值集中度（行业前3市值占比，**实证 IC +0.026 t=4.7**）
- `vol_div` 行业成交分化（成交额 Std/Mean）
- `volret_cov` 行业量价协方差（Cov(成交额,收益) 横截面）
- `lead_ret_pre` 龙头收益溢价（前3龙头均收益 - 行业等权收益）
- `range_bias` 振幅乖离（当日振幅 - 20日振幅中枢）
- `gap_sent` 隔夜跳空（(今开-昨收)/昨收）

行业级因子（前 4 个）按行业归属广播给成分股；龙头取流通市值前 3（mv_float 近似总市值），行业成分 <5 只时解读需谨慎。

「实战」家族增补（2026-08-18）：`res_mom`（60日动量去20日趋势，**实证 IC -0.035 t=-13.9，方向 negative**）、`sent_beta`（情绪Beta，市场情绪代理=20日平滑市场广度）、`rel_turn`（相对换手率，当日/20日均值）。
市场级指标（ADL 市场广度累计、Disp 全市场振幅分歧）为市场择时序列而非个股因子，未注册（横截面 IC 无法计算）。
