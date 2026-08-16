# ML 合成因子（LightGBM walk-forward）设计文档

日期：2026-08-16
状态：已获用户批准

## 1. 目标

在 quant-factor-mine 中新增「机器学习合成因子」能力：用 LightGBM 对现有因子库做非线性合成，产出样本外预测因子，**完全复用现有检验/正交化/回测/过拟合闭环**。

## 2. 架构

```
qfm/
├── ml_synthesizer.py     # 新：数据准备 + walk-forward 训练 + 合成因子
app.py                    # 改：因子检验页新增第 4 个 Tab「ML 合成」
tests/
└── test_ml_synthesizer.py  # 新：合成数据测试（零网络）
```

### 2.1 qfm/ml_synthesizer.py

- `prepare_ml_data(panel, names, horizon) -> pd.DataFrame`
  长表 `(date, stock, f_<因子名>…, target)`；特征取 T 日截面（DataPanel 已公告日对齐），target = close[T+h]/close[T]-1；特征按日截面 z-score（`qfm.pipeline.clean` 或自实现）；dropna。
- `walk_forward_train(panel, names, horizon, train_cutoff, n_folds, lgb_params=None) -> tuple[pd.DataFrame, dict]`
  滚动训练：以 `train_cutoff` 前数据为首个训练集，之后每 fold 推进 `12/n_folds` 个月（或按月度粒度推进），训练 LightGBM（objective=regression，默认学习率 0.05、树数 200、叶子 31），只预测当前训练截止之后下一个区间；输出样本外预测面板 `date×stock`。
- `synthesize_ml_factor(panel, names=None, horizon=20, train_cutoff=None, n_folds=4, progress=None) -> tuple[pd.DataFrame, dict]`
  主入口。`names=None` 时用全部内置因子；meta 含：n_folds、训练区间、**特征重要性 TOP10**、预测日期范围、防未来函数声明（结构安全）。
- 防未来函数：特征均 T 日已知；训练集仅含 ≤ 截止日数据；预测仅输出截止日之后；结构性安全，无需静态扫描。

### 2.2 页面：因子检验页第 4 个 Tab「ML 合成」

- 控件：特征因子 multiselect（默认全选）、前瞻天数、训练截止日（默认 2023-12-31）、fold 数（2/4/6）
- 按钮「训练并合成」→ st.progress → 结果区：
  - 复用 `factor_report` 展示 IC/IC_IR/分层/换手（样本外区间）
  - 特征重要性 TOP10 条形图（plotly）
  - 合成因子注册为 `ml_synth`（`register_factor` 运行时注册，func 返回预测面板）→ 出现在策略回测页因子选择器
- 说明：样本外日期范围、训练耗时、防未来函数声明

### 2.3 测试 tests/test_ml_synthesizer.py

- 合成面板：特征 f1 = 未来收益 + 噪声（构造线性关系）→ `synthesize_ml_factor` 样本外预测 IC 显著 > 0
- 预测面板 index 全部 ≥ 训练截止日（无未来函数）
- `register_factor` 后 `get_factor("ml_synth")` 可用且值一致
- 小数据（60 只 × 260 日）快速跑通（<30s）

## 3. 依赖与边界

- 新依赖：`lightgbm`（已装 4.7.0）
- 训练时长：800 只 × 8600 日 × 31 特征，默认参数单 fold 几秒~十几秒；页面带进度条
- 全市场（5000 只）慎用（提示沿用现有模式）
- 失败处理：特征缺失/数据不足时给出明确 st.error；lightgbm 导入失败时降级提示

## 4. 验收标准

1. pytest 全绿（新增 test_ml_synthesizer + 既有 29 个）
2. 页面 ML Tab 真实数据跑通：样本外 IC 展示、特征重要性图、`ml_synth` 出现在策略回测因子选择器
3. streamlit 健康、日志无 Traceback
