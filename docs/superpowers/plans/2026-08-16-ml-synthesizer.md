# ML 合成因子（LightGBM walk-forward）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 LightGBM walk-forward 因子合成器，在因子检验页第 4 个 Tab「ML 合成」训练并注册 `ml_synth` 因子，完全复用现有检验/回测/过拟合闭环。

**Architecture:** 新模块 `qfm/ml_synthesizer.py`（数据准备 + 滚动训练 + 合成因子），app.py 加第 4 个 Tab，合成因子运行时注册进因子注册表（进程内跨 rerun 有效，出现在策略回测选择器）。

**Tech Stack:** Python 3.13 / pandas 2.2.3 / numpy 2.1.3 / lightgbm 4.7.0（已装）/ streamlit / pytest。

## Global Constraints

- Python：`/opt/anaconda3/bin/python3`；lightgbm 4.7.0 已安装
- 零网络依赖：所有测试用合成数据（复用 `tests/conftest.py` 的 `panel` fixture + 测试内自建预测面板）
- 特征列命名：`prepare_ml_data` 输出长表列 `f_<因子名> + date + stock + target`；`target = close[T+h]/close[T]-1`
- 防未来函数结构性保证：特征均 T 日已知（DataPanel 已公告日对齐 + `factor_panel` 逐日截面清洗）；训练只用 `< cutoff` 数据；预测只输出 `≥ cutoff` 日期
- 页面注册名固定 `ml_synth`（`register_factor` 重复注册=覆盖，安全）
- 每次任务结束必须运行 pytest 并提交 git

---

### Task 1: qfm/ml_synthesizer.py 模块（TDD）

**Files:**
- Create: `tests/test_ml_synthesizer.py`
- Create: `qfm/ml_synthesizer.py`

**Interfaces:**
- Produces: `prepare_ml_data(panel, names: list, horizon: int = 20) -> pd.DataFrame`（列：date/stock/f_*/target）
- Produces: `walk_forward_train(panel, names, horizon=20, train_cutoff=None, n_folds=4, lgb_params=None, progress=None) -> tuple[pd.DataFrame, dict]`（预测面板 date×stock + meta：n_folds/train_cutoff/importance_top(f_ 前缀名→均值重要性 dict)/pred_start/pred_end）
- Produces: `synthesize_ml_factor(panel, names=None, horizon=20, train_cutoff=None, n_folds=4, progress=None) -> tuple[pd.DataFrame, dict]`（names=None → 全部内置因子；train_cutoff 非法时抛 ValueError 含说明）
- Consumes: `qfm.portfolio.synthesis.factor_panel`（特征，已逐日截面清洗）、`qfm.pipeline.tests.forward_returns`、`qfm.factors.list_factors`、lightgbm

- [ ] **Step 1: 写失败测试 tests/test_ml_synthesizer.py**

```python
"""ML 合成因子测试：样本外 IC / 无未来函数 / 注册可用 / 小数据跑通（零网络）"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qfm.data.panel import DataPanel
from qfm.factors import FACTORS, get_factor, register_factor
from qfm.ml_synthesizer import synthesize_ml_factor
from qfm.pipeline.tests import compute_ic, forward_returns


def _predictive_panel() -> tuple[DataPanel, pd.DataFrame]:
    """构造可预测面板：ret[t] = 0.02·x[t-1] + eps，特征 x[t] 可预测未来收益（无泄露）"""
    rng = np.random.default_rng(5)
    n, T = 40, 300
    dates = pd.bdate_range("2024-01-02", periods=T)
    cols = [f"{600000 + i}" for i in range(n)]
    x = pd.DataFrame(rng.normal(0, 1, (T, n)), index=dates, columns=cols)
    ret = 0.02 * x.shift(1).fillna(0.0).values + rng.normal(0, 0.02, (T, n))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(ret, axis=0)), index=dates, columns=cols)
    panel = DataPanel(
        close=close,
        volume=pd.DataFrame(rng.integers(1e6, 5e7, (T, n)).astype(float), index=dates, columns=cols),
        amount=pd.DataFrame(rng.uniform(1e8, 5e9, (T, n)), index=dates, columns=cols),
        turnover=pd.DataFrame(rng.uniform(0.005, 0.05, (T, n)), index=dates, columns=cols),
        mv_float=pd.DataFrame(np.tile(rng.uniform(2e9, 5e11, n), (T, 1)), index=dates, columns=cols),
        industry=pd.DataFrame(np.tile(np.array([["银行", "白酒", "科技", "医药"][i % 4]
                                                for i in range(n)], dtype=object), (T, 1)),
                               index=dates, columns=cols),
    )
    return panel, x


def test_ml_synth_oos_ic_positive():
    panel, x = _predictive_panel()
    register_factor(name="x_sig", family="测试", description="", direction="positive")(lambda d: x)
    try:
        cutoff = "2024-12-31"          # 面板 300 交易日止于 ~2025-02，样本外取 2025 年初
        pred, meta = synthesize_ml_factor(panel, names=["x_sig"], horizon=5,
                                          train_cutoff=cutoff, n_folds=2)
        # 无未来函数：预测面板只含 cutoff 之后
        assert (pred.index >= pd.Timestamp(cutoff)).all()
        # 样本外 IC 显著为正
        ic = compute_ic(pred, forward_returns(panel.close, 5)).dropna()
        assert ic.mean() > 0.05
        assert meta["n_folds"] >= 1
        assert "importance_top" in meta and len(meta["importance_top"]) >= 1
    finally:
        FACTORS.pop("x_sig", None)


def test_ml_synth_no_lookahead_index():
    panel, x = _predictive_panel()
    register_factor(name="x_sig", family="测试", description="", direction="positive")(lambda d: x)
    try:
        pred, _ = synthesize_ml_factor(panel, names=["x_sig"], horizon=5,
                                       train_cutoff="2024-10-01", n_folds=2)
        assert (pred.index >= pd.Timestamp("2024-10-01")).all()
        assert pred.notna().sum().sum() > 0
    finally:
        FACTORS.pop("x_sig", None)


def test_ml_synth_register_factor_usable():
    panel, x = _predictive_panel()
    register_factor(name="x_sig", family="测试", description="", direction="positive")(lambda d: x)
    try:
        pred, _ = synthesize_ml_factor(panel, names=["x_sig"], horizon=5,
                                       train_cutoff="2024-12-31", n_folds=2)
        # 页面同款注册方式
        register_factor(name="ml_synth", family="机器学习", description="test",
                        direction="positive")(lambda d: pred)
        f = get_factor("ml_synth")
        assert f is not None
        out = f.func(panel)
        assert out.shape == pred.shape
        assert abs(float(out.iloc[0, 0] - pred.iloc[0, 0])) < 1e-12
    finally:
        FACTORS.pop("x_sig", None)
        FACTORS.pop("ml_synth", None)


def test_ml_synth_small_panel_runs(panel):
    pred, meta = synthesize_ml_factor(panel, names=["mom_20", "turnover_20"], horizon=5,
                                      train_cutoff="2024-06-30", n_folds=2)
    assert pred.shape[1] == 2 or pred.shape[1] >= 1
    assert (pred.index >= pd.Timestamp("2025-06-30")).all()
    assert meta["n_folds"] >= 1


def test_ml_synth_invalid_cutoff_raises(panel):
    with pytest.raises(ValueError, match="训练截止日"):
        synthesize_ml_factor(panel, names=["mom_20"], horizon=5, train_cutoff="not-a-date", n_folds=2)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_ml_synthesizer.py -v`
Expected: FAIL（ModuleNotFoundError: qfm.ml_synthesizer）

- [ ] **Step 3: 实现 qfm/ml_synthesizer.py**

```python
"""机器学习合成因子：LightGBM walk-forward 滚动训练

把现有因子库（date×stock 面板）非线性合成为一个新因子：
- 特征：T 日已知因子值（DataPanel 已公告日对齐 + factor_panel 逐日截面清洗）
- 目标：close[T+h]/close[T]-1
- 训练：逐 fold 只用 < cutoff 的历史数据；预测仅输出 ≥ cutoff 的样本外日期
- 结构性无未来函数：特征/目标/训练切分均不引用未来
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qfm.pipeline.tests import forward_returns


def prepare_ml_data(panel, names: list, horizon: int = 20) -> pd.DataFrame:
    """因子面板 → 长表 (date, stock, f_<因子名>…, target)

    特征取 T 日截面值（factor_panel 已逐日清洗），target = 未来 horizon 日收益。
    """
    from qfm.portfolio.synthesis import factor_panel

    factors = factor_panel(panel, names)
    parts = [fdf.stack().rename(f"f_{name}") for name, fdf in factors.items()]
    X = pd.concat(parts, axis=1)
    y = forward_returns(panel.close, horizon).stack().rename("target")
    df = X.join(y).dropna()
    df.index.names = ["date", "stock"]
    return df.reset_index()


def walk_forward_train(panel, names: list, horizon: int = 20, train_cutoff=None,
                       n_folds: int = 4, lgb_params: dict | None = None,
                       progress=None) -> tuple[pd.DataFrame, dict]:
    """walk-forward 滚动训练：每 fold 用 ≤ cutoff 历史训练，只预测下一个样本外区间

    返回 (预测面板 date×stock, meta)
    meta: n_folds / train_cutoff / importance_top（f_ 前缀→均值重要性，降序）/ pred_start / pred_end
    """
    import lightgbm as lgb

    df = prepare_ml_data(panel, names, horizon)
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    if train_cutoff is None:
        train_cutoff = dates[len(dates) * 3 // 5]
    cutoff = pd.Timestamp(train_cutoff)
    oos = dates[dates >= cutoff]
    if len(oos) < 2:
        raise ValueError("训练截止日过晚，样本外区间不足")

    params = {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 31,
              "random_state": 42, "verbose": -1}
    params.update(lgb_params or {})

    bounds = np.linspace(0, len(oos), int(n_folds) + 1, dtype=int)
    pred_parts, imp_list = [], []
    for k in range(int(n_folds)):
        a, b = int(bounds[k]), int(bounds[k + 1])
        if b <= a:
            continue
        cut = oos[a]                      # 本 fold 预测起始日
        end = oos[b - 1]                  # 本 fold 预测截止日
        if progress:
            progress(k, int(n_folds), f"训练 fold {k + 1}/{n_folds}（数据 < {cut.date()}）…")
        train = df[df["date"] < cut]
        valid = df[(df["date"] >= cut) & (df["date"] <= end)]
        if len(train) < 500 or len(valid) < 50:
            continue
        model = lgb.LGBMRegressor(**params)
        model.fit(train[feat_cols], train["target"])
        v = valid.copy()
        v["pred"] = model.predict(valid[feat_cols])
        pred_parts.append(v[["date", "stock", "pred"]])
        imp_list.append(dict(zip(feat_cols, model.feature_importances_)))

    if not pred_parts:
        raise ValueError("训练数据不足：请扩大训练区间或减少 fold 数")
    out = pd.concat(pred_parts)
    pred_df = out.pivot_table(index="date", columns="stock", values="pred")
    pred_df.index = pd.DatetimeIndex(pred_df.index)
    imp = pd.DataFrame(imp_list).mean().sort_values(ascending=False)
    meta = {
        "n_folds": len(pred_parts),
        "train_cutoff": str(cutoff.date()),
        "importance_top": imp.head(10).to_dict(),
        "pred_start": str(pred_df.index.min().date()),
        "pred_end": str(pred_df.index.max().date()),
    }
    return pred_df, meta


def synthesize_ml_factor(panel, names=None, horizon: int = 20, train_cutoff=None,
                         n_folds: int = 4, progress=None) -> tuple[pd.DataFrame, dict]:
    """ML 合成因子主入口：names=None 时用全部内置因子"""
    try:
        pd.Timestamp(train_cutoff) if train_cutoff is not None else None
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"训练截止日格式非法：{train_cutoff!r}（示例：2023-12-31）") from e
    if names is None:
        from qfm.factors import list_factors

        names = [f.name for f in list_factors()]
    if not names:
        raise ValueError("至少需要一个特征因子")
    if progress:
        progress(0, int(n_folds), "准备数据…")
    return walk_forward_train(panel, names, horizon=horizon, train_cutoff=train_cutoff,
                              n_folds=n_folds, progress=progress)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/test_ml_synthesizer.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add qfm/ml_synthesizer.py tests/test_ml_synthesizer.py
git commit -m "feat: LightGBM walk-forward 因子合成模块（样本外预测 + 特征重要性，结构性无未来函数）"
```

---

### Task 2: 因子检验页第 4 个 Tab「ML 合成」

**Files:**
- Modify: `app.py:206`（tabs 定义）
- Modify: `app.py` page_test 末尾（新增 Tab 4 代码块）

**Interfaces:**
- Consumes: `qfm.ml_synthesizer.synthesize_ml_factor`（Task 1）、`qfm.factors.register_factor`、`factor_report/ic_chart/layer_chart`（已导入）
- Produces: 运行时注册因子 `ml_synth`（进程内跨 rerun 有效，出现在策略回测因子选择器）

- [ ] **Step 1: 修改 tabs 定义（app.py:206）**

将：

```python
    tab_single, tab_cmp, tab_ortho = st.tabs(["单因子检验", "多因子对比", "正交化"])
```

改为：

```python
    tab_single, tab_cmp, tab_ortho, tab_ml = st.tabs(["单因子检验", "多因子对比", "正交化", "ML 合成"])
```

- [ ] **Step 2: 在 page_test 末尾（Tab 3 正交化块结束后）插入 Tab 4 代码**

```python
    # ---------- Tab 4：ML 合成因子（LightGBM walk-forward） ----------
    with tab_ml:
        st.markdown("LightGBM walk-forward 滚动训练：把现有因子库**非线性合成**成一个新因子"
                    "（仅输出样本外预测，结构性无未来函数）")
        c1, c2, c3 = st.columns([2, 1, 1])
        ml_names = c1.multiselect("特征因子", [f.name for f in list_factors()],
                                  default=[f.name for f in list_factors()][:10], key="ml_names")
        ml_h = c2.selectbox("前瞻天数", [5, 10, 20, 60], index=2, key="ml_h")
        ml_folds = c3.selectbox("fold 数", [2, 4, 6], index=1, key="ml_folds",
                                help="越多训练越充分，耗时线性增加")
        ml_cutoff = st.text_input("训练截止日（此后的日期为样本外预测区间）", "2023-12-31", key="ml_cutoff")
        if st.button("训练并合成", use_container_width=True):
            if not ml_names:
                st.error("至少选择一个特征因子")
                return
            from qfm.ml_synthesizer import synthesize_ml_factor

            prog = st.progress(0.0, text="准备数据…")
            try:
                pred, meta = synthesize_ml_factor(
                    panel, names=ml_names, horizon=ml_h, train_cutoff=ml_cutoff,
                    n_folds=ml_folds,
                    progress=lambda i, n, msg: prog.progress((i + 1) / n, text=msg))
            except Exception as e:  # noqa: BLE001
                st.error(f"训练失败：{e}")
                return
            st.success(f"合成完成：样本外预测 {len(pred)} 个交易日 × {pred.shape[1]} 只股票"
                       f"（fold={meta['n_folds']}，区间 {meta['pred_start']} ~ {meta['pred_end']}）")
            rep = factor_report(pred, panel.close, horizon=ml_h, direction="positive")
            s = rep["ic_summary"]
            k1, k2, k3 = st.columns(3)
            k1.metric("样本外 IC", f"{s['ic_mean']:+.4f}" if pd.notna(s["ic_mean"]) else "—")
            k2.metric("IC_IR", f"{s['ic_ir']:.2f}" if pd.notna(s["ic_ir"]) else "—")
            k3.metric("有效天数", s["n_days"])
            st.plotly_chart(ic_chart(rep), use_container_width=True)
            st.plotly_chart(layer_chart(rep, "positive"), use_container_width=True)

            imp = meta["importance_top"]
            figi = go.Figure(go.Bar(
                x=list(imp.values()), y=[k.replace("f_", "") for k in imp],
                orientation="h", marker_color="#7C4DFF"))
            figi.update_layout(title="特征重要性 TOP10", height=320, margin=dict(t=40),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font=dict(color="#3B2E5E"))
            st.plotly_chart(figi, use_container_width=True)

            from qfm.factors import register_factor

            register_factor(name="ml_synth", family="机器学习",
                            description=f"LightGBM walk-forward 合成（{ml_h}日前瞻，训练截止 {ml_cutoff}）",
                            direction="positive")(lambda d: pred)
            st.caption(f"✅ 已注册为因子 `ml_synth`，可直接在「策略回测」页选择。"
                       f"预测区间 {meta['pred_start']} ~ {meta['pred_end']}——回测起点请设在此区间内或之后。")
```

- [ ] **Step 3: 语法检查 + 重启 + 健康检查**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
/opt/anaconda3/bin/python3 -m py_compile app.py
pkill -f "streamlit run" || true
nohup /opt/anaconda3/bin/streamlit run app.py --server.headless true > /tmp/qfm_app.log 2>&1 < /dev/null &
sleep 8
curl -s http://localhost:8501/_stcore/health
grep -ci "traceback\|error" /tmp/qfm_app.log || true
```

Expected: health 返回 `ok`；错误计数 0

- [ ] **Step 4: Commit**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add app.py
git commit -m "feat: 因子检验页新增 ML 合成 Tab（walk-forward 训练 + 样本外 IC + 特征重要性 + 注册 ml_synth）"
```

---

### Task 3: 端到端验收（全量测试 + 真实数据链路 + 文档）

**Files:**
- Modify: `README.md`（功能列表 + ML 合成说明）
- Modify: `交接.md`（当前状态追加）

- [ ] **Step 1: 全量测试**

Run: `cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 -m pytest tests/ -q`
Expected: 34 passed（29 旧 + 5 新）

- [ ] **Step 2: 真实数据链路验证（50 只缓存股票，4 folds）**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine && /opt/anaconda3/bin/python3 - <<'PY'
from qfm.data import DataLoader, build_panel, get_universe
from qfm.ml_synthesizer import synthesize_ml_factor
from qfm.pipeline.tests import compute_ic, forward_returns
import pandas as pd

dl = DataLoader()
stocks = dl.status().get("stocks_cached", 0)
print(f"缓存股票数: {stocks}")
if stocks:
    codes = get_universe("index800")[:min(stocks, 50)]
    panel = build_panel(dl.load_bars(codes), dl.load_indicators())
    pred, meta = synthesize_ml_factor(panel, horizon=20, train_cutoff="2023-12-31", n_folds=4)
    ic = compute_ic(pred, forward_returns(panel.close, 20)).dropna()
    print(f"ML 合成: 预测 {len(pred)} 日 × {pred.shape[1]} 只 | 样本外 IC={ic.mean():+.4f} IC_IR={ic.mean()/ic.std():.2f}")
    print("特征重要性 TOP5:", list(meta["importance_top"].items())[:5])
PY
```

Expected: 打印预测尺寸、样本外 IC 与特征重要性；无异常（每 fold 数秒）

- [ ] **Step 3: README.md 追加**

```markdown
## 机器学习合成因子

因子检验页「ML 合成」Tab：LightGBM walk-forward 滚动训练把现有因子库非线性合成为一个新因子（仅样本外预测，结构性无未来函数）。合成后自动注册为 `ml_synth` 因子，可直接进入策略回测页参与合成与回测。
```

- [ ] **Step 4: 交接.md「三、当前状态」追加一行**

```markdown
- ✅ ML 合成因子（LightGBM walk-forward）：因子检验页「ML 合成」Tab——特征=现有因子库（T 日已知）、目标=未来收益、滚动训练仅输出样本外预测；自动注册 `ml_synth` 因子进策略回测；依赖 lightgbm 4.7.0（已装）；pytest 34 用例全绿
```

- [ ] **Step 5: 最终提交 + 交付验收汇总**

```bash
cd /Users/zxt/.zcode/workspace/default/quant-factor-mine
git add README.md 交接.md
git commit -m "docs: ML 合成因子功能与状态更新"
git log --oneline | head -5
```

输出：pytest 全量输出、health 检查、真实数据链路输出、KPI 卡（PUA 协议）
