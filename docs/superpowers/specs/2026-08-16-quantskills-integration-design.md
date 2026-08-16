# QuantSkills 三技能融入设计文档

日期：2026-08-16
状态：已获用户批准（范围=三个全融入；PBO=保存试验矩阵全量支持）

## 1. 背景与目标

将 3 个 QuantSkills 开源 Skill 的方法论融入现有 Streamlit 量化因子挖掘应用（quant-factor-mine，localhost:8501）：

1. **skill-quant-factor-skill-factory** → 无未来函数检查（挖掘/自定义因子）
2. **skill-factor-orthogonalize** → 逐日截面 OLS 正交化（行业/市值/风格/旧因子）
3. **skill-backtest-overfit** → 回测过拟合统计检验（DSR/PBO/Haircut/MinTRL）

原则：零新页面；方法论模块化进 `qfm/` 包；复用现有 DataPanel/回测/挖掘管线；GPL-3.0 源码不拷贝，按公开学术文献公式独立实现。

## 2. 架构

```
qfm/
├── orthogonalize.py        # 新：逐日截面 OLS 正交化 + 诊断
├── overfit.py              # 新：DSR/PBO/Haircut/MinTRL + OverfitReport
├── pipeline/
│   ├── lookahead.py        # 新：无未来函数检查
│   └── tests.py            # 改：factor_report 增加 lookahead 标记
├── mining/
│   └── engine.py           # 改：run_mining 保存试验矩阵
└── portfolio/
    └── synthesis.py        # 改：synthesize 正交化升级（行业/风格可选）
data_cache/
└── trials/                 # 新：试验矩阵 parquet + manifest.json
```

### 2.1 qfm/orthogonalize.py

- 输入：信号 `date×stock` + 控制变量
- 控制变量（均来自 DataPanel，零外部 API）：
  - 行业：`panel.industry` L1 one-hot（drop_first）
  - 市值：`ln(panel.mv_float)`
  - 风格：beta_60d（60 日滚动 cov/var 对全池等权收益）、vol_20d（20 日滚动收益 std）
  - 旧因子：用户勾选的已有因子面板
- 流程（沿用 skill 7 步）：契约校验 → 控制变量对齐 → 5×MAD winsorize → z-score → 逐日 OLS（`np.linalg.lstsq`，含截距）→ 残差重新 z-score → 每日样本 <30 跳过
- 输出：残差因子 DataFrame + 诊断 dict：
  - 正交前/后：对控制变量的回归 R²（暴露）、IC（rank）、TOP10% 换手、覆盖率
  - IC 保留率 = 正交后 IC / 正交前 IC
- 函数签名：
  - `winsorize_zscore(x: pd.Series, n_mad=5.0) -> pd.Series`
  - `style_controls(panel) -> dict[str, pd.DataFrame]`（beta_60d / vol_20d / log_mv / industry_onehot 惰性计算）
  - `orthogonalize(signal, controls, min_samples=30) -> (residual, diagnostics)`

### 2.2 qfm/overfit.py

按公开文献公式独立实现（不拷贝 GPL 源码）：

- `deflated_sharpe_ratio(returns, n_trials, periods_per_year=252, sr_var=None)` — Bailey & López de Prado (2014)。sr_var 来自试验矩阵时精确；否则用 `E[max SR_N]` 理论期望（σ_SR = σ_returns / √T 退化估计），提示"偏宽松"
- `probabilistic_sharpe_ratio(observed_sr, benchmark_sr, n_obs, skew, kurt)` — LdP (2012)，含偏度/峰度修正
- `minimum_track_record_length(...)` — MinTRL（PSR=0.95 反解 n）
- `pbo_cscv(trials_matrix, n_blocks=16)` — Bailey et al. (2017)：CSCV 组合对称交叉验证，N≥10 才有效
- `haircut_sharpe(returns, n_trials, method='holm')` — Harvey & Liu (2015)：Bonferroni/Holm/BHY，t = SR·√T，p 值用 `norm.sf` 避免下溢
- `overfit_report(selected_returns, n_trials, trials_matrix=None) -> OverfitReport`（dataclass：verdict/passed/deflated_sharpe_ratio/pbo/haircut/minimum_track_record_length + 各项明细与降级说明）
- 判定：DSR≥0.95（PSR 概率阈值）且 PBO<0.5 且 haircut 后仍显著 → PASS；任一不过 → FAIL + 触发原因

### 2.3 qfm/pipeline/lookahead.py

- `check_structural(panel, factor_df) -> dict`：核对因子列名/生成方式。对挖掘候选（指标×窗口×变换：rolling 均值/std/累计，全部基于 T 日及之前数据）→ 结构安全"pass"
- `scan_source(src: str) -> list[str]`：静态泄漏模式扫描（自定义因子源码）：
  - `shift(-N)` / `.iloc[-N]` / `[t+1:]` / `fut` 等未来引用
  - 负窗口 rolling、`close.shift(-`、`pct_change` 后负 shift
  - 财务字段非公告日对齐用法（使用 `panel.fund` 即已对齐，标记安全）
- `factor_report` 增加 `lookahead` 字段：`"pass" | "review" | "fail"` + 命中模式列表

### 2.4 mining/engine.py 改造

- `run_mining(..., save_trials=True, trials_dir='data_cache/trials', top_n=30)`：
  - 每个候选：以候选因子为打分，月频调仓 TOP-N 等权组合收益序列（复用 backtest 的调仓日历/涨跌停过滤逻辑的轻量版，仅返回月频收益，不返回完整持仓）
  - 汇总 T×N 矩阵（列=候选名，行=调仓期收益）→ `data_cache/trials/trials_<horizon>_<YYYYMMDD_HHMMSS>.parquet`
  - `manifest.json`：n_trials、horizon、top_n、生成时间、候选名列表、无未来函数检查结果
- 排行榜 DataFrame 增加「无未来函数」列

### 2.5 synthesis.py 改造

- `synthesize(..., orthogonalize=False, ortho_controls=None)`：`orthogonalize=True` 时按 `ortho_controls`（["industry","size","style"] 子集）调用 `qfm.orthogonalize.orthogonalize`，默认保持现有行为（仅市值），向后兼容

## 3. 页面改动（零新页面）

### 3.1 因子检验页（page_test）
新增第三个 Tab「正交化」：
- 选因子（下拉，默认 ep_ttm）+ 勾选剥离项（行业/市值/beta/波动率/已有因子）
- 「运行正交化」→ st.status 流程：残差因子统计（IC 保留率、覆盖天数）+ 诊断对比表（暴露 R² / IC / 换手 / 覆盖率 前后并排）+ 残差因子 CSV 下载

### 3.2 策略回测页（page_strategy）
- ① 因子合成：「市值正交化」复选框 → 「正交化」expander：勾选剥离项（默认市值）
- ② 组合回测结果下新增「过拟合检验」区：
  - 自动取回测 nav 月频收益作为 selected_returns
  - n_trials 默认 = 最近一次试验矩阵 manifest 的 n_trials（可手动改）
  - 有试验矩阵（列名与本次因子集匹配或用户选择）→ PBO + 精确 DSR；否则降级提示
  - 输出：DSR / PBO / Haircut Sharpe（含原始 vs 调整后）/ MinTRL 四格指标 + PASS/FAIL 结论（中文说明）

### 3.3 自动挖掘页（page_mine）
- 「保存试验矩阵」checkbox（默认 True）
- 排行榜新增「无未来函数」列（全部"pass"，结构性安全）
- 完成后显示试验矩阵路径 + manifest 摘要（n_trials）

## 4. 数据流

```
挖掘（候选 × 检验）
  ├→ 排行榜（含无未来函数列）
  └→ [新] 每候选月频 TOP-N 组合收益 → T×N parquet + manifest
回测页：nav 月频收益 → DSR/MinTRL/Haircut（n_trials ← manifest）
         + trials 矩阵 → PBO(CSCV)
正交化：信号 + 行业/市值/风格控制 → 逐日 OLS 残差 → 残差因子 + 诊断
```

## 5. GPL-3.0 许可处理

三个 skill 均为 GPL-3.0。DSR/PSR/MinTRL（Bailey & López de Prado 2012/2014）、PBO（Bailey et al. 2017）、Haircut（Harvey & Liu 2015）、正交化（标准截面回归）公式均出自公开学术文献，本项目**按文献公式独立实现**，不拷贝仓库源码文件。`/tmp/quantskills/` 三个包仅作方法论参考。README 注明出处。项目不引入 GPL 代码拷贝。

## 6. 错误处理与边界

- 正交化：每日截面 <30 跳过该日；行业缺失日跳过行业 one-hot（降级为市值+风格，报告中标注）；覆盖率变化如实展示
- PBO：N<10 或矩阵缺失 → 明确降级（只给 DSR 退化版 + 提示）
- DSR：无矩阵 → 单次估计 + "偏宽松"提示
- 试验矩阵：与当日挖掘参数强相关，manifest 记录参数；回测页加载时校验行数/日期
- 所有新模块纯函数可单测，数据用合成数据，不依赖网络

## 7. 测试（pytest，新增 4 组）

- `test_orthogonalize.py`：合成数据（信号 = 行业 dummy × 系数 + 噪声）→ 残差与行业暴露正交（回归 R²≈0）、IC 保留率在 [0,1.2] 合理区间、样本<30 跳过
- `test_overfit.py`：纯噪声 N=200 试验取最优 → DSR<0.95 / PBO 高；构造真实 edge 信号 → PASS；haircut 单调性（n_trials 越大调整后 Sharpe 越低）
- `test_lookahead.py`：泄漏源码命中 fail；合法源码 pass；挖掘候选结构性 pass
- `test_mining_trials.py`：合成 DataPanel 小样本跑 run_mining → 矩阵形状 (T×N)、manifest 字段齐全、可加载
- 回归：现有挖掘/回测流程跑通（真实缓存数据可选）

## 8. 验收标准

1. 因子检验页正交化 Tab：正交前后 IC/暴露/换手/覆盖率对比可见，残差因子可下载
2. 策略回测页：DSR/PBO/Haircut/MinTRL 四项数值 + PASS/FAIL 中文结论
3. 自动挖掘页：试验矩阵落盘 data_cache/trials/，回测页可加载（n_trials 自动带出）
4. pytest 全绿
5. streamlit run app.py 正常启动，6 板块无回归
