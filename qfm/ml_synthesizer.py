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
    只要求 target 非 NaN：特征缺失交给 LightGBM 原生处理（inner join 全列 dropna
    会被单因子缺失毒化——历史教训：bb_break_20 覆盖率 0% 曾致全量训练崩溃）。
    """
    from qfm.portfolio.synthesis import factor_panel

    factors = factor_panel(panel, names)
    parts = [fdf.stack().rename(f"f_{name}") for name, fdf in factors.items()]
    X = pd.concat(parts, axis=1)
    y = forward_returns(panel.close, horizon).stack().rename("target")
    df = X.join(y).dropna(subset=["target"])
    df.index.names = ["date", "stock"]
    return df.reset_index()


def walk_forward_train(panel, names: list, horizon: int = 20, train_cutoff=None,
                       n_folds: int = 4, lgb_params: dict | None = None,
                       progress=None, min_coverage: float = 0.0) -> tuple[pd.DataFrame, dict]:
    """walk-forward 滚动训练：每 fold 用 ≤ cutoff 历史训练，只预测下一个样本外区间

    返回 (预测面板 date×stock, meta)
    meta: n_folds / train_cutoff / importance_top（f_ 前缀→均值重要性，降序）/ pred_start / pred_end
    """
    import lightgbm as lgb

    df = prepare_ml_data(panel, names, horizon)
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    # 剔除覆盖率 < min_coverage 的特征（无信息列；LightGBM 可处理部分缺失，全空列无意义）
    keep = [c for c in feat_cols if df[c].notna().mean() >= min_coverage]
    if len(keep) < len(feat_cols):
        df = df.drop(columns=[c for c in feat_cols if c not in keep])
    feat_cols = [c for c in df.columns if c.startswith("f_")]
    if not feat_cols:
        raise ValueError("全部特征因子缺失率过高，无可用特征")
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
