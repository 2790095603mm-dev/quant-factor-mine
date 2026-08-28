"""表达式挖掘（WorldQuant 方法论吸收，独立实现）：操作符 × 字段 → 流式候选

候选形式：`{op}[_{N}]__{field}`，如 `ts_mean_5__mom_20`、`rank__bp`。
全部操作符只用 t 及之前的数据（rolling/shift/rank 均滞后），结构性无未来函数。

字段池 = 已注册因子（49）+ 基础指标（ret/turnover/amount/volume）+ 技术指标（fields.py）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── 操作符 ───────────────────────────────────────────────────────
# 无参：rank（截面排名）/ abs
# 有参：ts_rank/ts_mean/ts_zscore/ts_std/decay_linear/delta（窗口 N）


def _decay_linear(s: pd.DataFrame, n: int) -> pd.DataFrame:
    """线性衰减加权均值：最新值权重 n，最旧权重 1（WQ decay_linear 语义）。

    向量化：Σ_{i=0}^{n-1}(n-i)x_{t-i} = n·C_t − Σ_{k=t-n}^{t-1} C_k（C=累积和），
    第二个求和 = C.rolling(n, min_periods=1).sum().shift(1)；完整窗口才出值。
    O(T·S)，避免逐窗口 apply。
    """
    c = s.cumsum()
    w = n * c - c.rolling(n, min_periods=1).sum().shift(1)
    out = w / (n * (n + 1) / 2)
    return out.where(s.rolling(n, min_periods=1).count() >= n)


OPS: dict[str, callable] = {
    "rank": lambda s: s.rank(axis=1),
    "ts_rank": lambda s, n: s.rolling(n).rank(pct=True),
    "ts_mean": lambda s, n: s.rolling(n).mean(),
    "ts_zscore": lambda s, n: (s - s.rolling(n).mean()) / s.rolling(n).std(),
    "ts_std": lambda s, n: s.rolling(n).std(),
    "decay_linear": _decay_linear,
    "delta": lambda s, n: s.diff(n),
    "abs": lambda s: s.abs(),
}

EXPR_OPS = list(OPS)                      # 全部操作符名（顺序即测试参数化顺序）
NO_PARAM_OPS = ("rank", "abs")            # 无窗口参数
PARAM_OPS = tuple(k for k in OPS if k not in NO_PARAM_OPS)


def apply_op(op: str, fdf: pd.DataFrame, n: int | None) -> pd.DataFrame:
    """对 date×stock 矩阵施加操作符；op 未知抛 KeyError"""
    fn = OPS[op]
    return fn(fdf) if n is None else fn(fdf, n)


def parse_candidate_name(name: str) -> tuple[str, int | None, str]:
    """`ts_mean_5__mom_20` → ('ts_mean', 5, 'mom_20')；`rank__mom_20` → ('rank', None, 'mom_20')"""
    op_part, field = name.split("__", 1)
    if "_" in op_part:
        op, n = op_part.rsplit("_", 1)
        return op, int(n), field
    return op_part, None, field


# ── 字段池 ───────────────────────────────────────────────────────
def build_field_map(panel, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """字段池（一次构建，供流式候选复用）：已注册因子 + 基础指标 + 技术指标"""
    from qfm.factors import list_factors
    from qfm.mining.engine import base_indicators
    from qfm.mining.fields import tech_fields

    base = base_indicators(panel)
    tech = tech_fields(panel)
    fmap = {f.name: f for f in list_factors()}

    if fields is None:
        fields = list(fmap) + list(base) + list(tech)

    out: dict[str, pd.DataFrame] = {}
    for name in fields:
        if name in fmap:
            out[name] = fmap[name].func(panel)
        elif name in base:
            out[name] = base[name]
        elif name in tech:
            out[name] = tech[name]
    return out


def generate_candidates_expr(panel, fields: list[str] | None = None,
                             ops: list[str] | None = None,
                             n_list: tuple[int, ...] = (5, 10, 20),
                             field_map: dict[str, pd.DataFrame] | None = None):
    """流式生成表达式候选：字段 × 操作符（有参操作符 × 窗口列表）。

    逐候选 yield (候选名, date×stock)，不囤积（峰值≈单候选矩阵）。
    """
    from qfm.mining.engine import base_indicators
    from qfm.mining.fields import tech_fields

    ops = tuple(ops) if ops else tuple(EXPR_OPS)
    fmap = field_map if field_map is not None else build_field_map(panel, fields)
    if fields is None:
        fields = list(fmap)
    for name in fields:
        if name not in fmap:
            continue
        fdf = fmap[name]
        for op in ops:
            if op in NO_PARAM_OPS:
                yield f"{op}__{name}", apply_op(op, fdf, None)
            elif op in PARAM_OPS:
                for n in n_list:
                    yield f"{op}_{n}__{name}", apply_op(op, fdf, n)


def expr_candidate_total(fields: list[str], ops: list[str] | None = None,
                         n_list: tuple[int, ...] = (5, 10, 20)) -> int:
    """候选总数（进度条用）：字段 × [无参操作符 + 有参操作符 × len(n_list)]"""
    ops = tuple(ops) if ops else tuple(EXPR_OPS)
    n_ops = sum(1 for op in ops if op in NO_PARAM_OPS) + \
            sum(len(n_list) for op in ops if op in PARAM_OPS)
    return len(fields) * n_ops
