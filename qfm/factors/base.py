"""因子注册表：@register_factor 一行注册一个因子，并保留完整版本历史。

版本语义
--------
每个因子携带 `version`、`tags`、`formula`（源码）、`source_hash`、`created_at`。
`register_factor` 会按 `source_hash` 判断是否真的是新版本：

- 同名 + 同 hash（模块重复导入、测试 reload）→ 复用已有记录，**不产生新版本**；
- 同名 + 不同 hash（因子被修改）→ 追加 version+1，**旧版本完整保留在 FACTOR_HISTORY**，
  绝不静默覆盖。这样任何一次实验归档都能引用到当时确切的因子定义。

`source_hash` 默认取自函数源码；运行时动态构造的因子（如 ml_synth 的 lambda）
源码不稳定，可通过 `source_key` 传入一个稳定的定义字符串（例如训练参数摘要）。
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from qfm.data.panel import DataPanel

FAMILIES = ["价值", "质量", "成长", "动量反转", "波动", "流动性", "规模", "实战", "多因子", "机器学习"]

# 家族 → 默认标签（避免为 49 个既有因子逐个补标签）
FAMILY_TAGS: dict[str, tuple[str, ...]] = {
    "价值": ("基本面", "估值"),
    "质量": ("基本面", "盈利质量"),
    "成长": ("基本面", "成长"),
    "动量反转": ("价量", "动量"),
    "波动": ("价量", "风险"),
    "流动性": ("价量", "流动性"),
    "规模": ("价量", "规模"),
    "实战": ("价量", "行业格局"),
    "多因子": ("合成", "多因子"),
    "机器学习": ("机器学习", "合成"),
}

REGISTRY_ENV = "QFM_FACTOR_REGISTRY"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_source(func: Callable) -> str:
    """函数源码（去掉装饰器与统一缩进），作为可展示的"公式"。"""
    try:
        raw = inspect.getsource(func)
    except (OSError, TypeError):  # C 扩展、内置、交互式定义
        return ""
    lines = textwrap.dedent(raw).splitlines()
    # 丢掉 @register_factor(...) 装饰器行，只保留函数体本身
    while lines and (lines[0].lstrip().startswith("@") or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _hash_definition(func: Callable, source_key: str | None) -> str:
    payload = source_key if source_key is not None else _clean_source(func)
    if not payload:
        # 源码不可得时退回函数限定名，至少保证同名同函数稳定
        payload = f"{getattr(func, '__module__', '?')}:{getattr(func, '__qualname__', '?')}"
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _closure_params(func: Callable) -> dict:
    """提取注册时固定的标量参数：关键字默认值与闭包自由变量。

    动量家族用 `_make(w)` 批量注册，形如 `def mom(d, _w=w)` —— 这里的 `w` 是**默认实参**
    而非闭包自由变量，因此两类都要取。拿不到参数时 `inspect.getsource` 只能给出通用模板，
    看不出具体窗口，公式就不自描述。
    """
    out: dict = {}
    code = getattr(func, "__code__", None)
    if code is None:
        return out

    def _keep(name: str, value: object) -> None:
        if isinstance(value, (int, float, str, bool)) or value is None:
            # 去掉 `_w` 这类内部前缀，展示为 w
            out[name.lstrip("_") or name] = value

    names = getattr(code, "co_varnames", ()) or ()
    defaults = getattr(func, "__defaults__", None) or ()
    if defaults:
        positional = names[: getattr(code, "co_argcount", 0)]
        for name, value in zip(positional[len(positional) - len(defaults):], defaults):
            _keep(name, value)

    closure = getattr(func, "__closure__", None)
    for name, cell in zip(getattr(code, "co_freevars", ()) or (), closure or ()):
        try:
            _keep(name, cell.cell_contents)
        except ValueError:  # 空 cell
            continue
    return out


def _formula_with_params(source: str, params: dict) -> str:
    if not params:
        return source
    rendered = ", ".join(f"{key}={value!r}" for key, value in sorted(params.items()))
    note = f"# 注册参数: {rendered}"
    return f"{source}\n{note}" if source else note


@dataclass
class Factor:
    name: str
    func: Callable[[DataPanel], pd.DataFrame]
    family: str
    description: str
    direction: str = "positive"  # positive: 值越大预期收益越高；negative: 值越小越高
    version: int = 1
    tags: tuple[str, ...] = ()
    formula: str = ""
    source_hash: str = ""
    created_at: str = ""
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """可序列化的因子定义（不含函数对象），用于实验归档与因子库页面。"""
        return {
            "name": self.name,
            "family": self.family,
            "description": self.description,
            "direction": self.direction,
            "version": self.version,
            "tags": list(self.tags),
            "formula": self.formula,
            "source_hash": self.source_hash,
            "created_at": self.created_at,
            "params": dict(self.params),
        }


FACTORS: dict[str, Factor] = {}
# 同名因子的全部历史版本（按注册顺序），旧版本永不删除
FACTOR_HISTORY: dict[str, list[Factor]] = {}

# 因子中文名映射（单一数据源）：页面显示 "英文（中文）"
ZH_NAMES: dict[str, str] = {
    # 价值
    "ep_ttm": "盈利收益率TTM", "bp": "账面市值比", "ep": "盈利收益率",
    # 质量
    "roe": "净资产收益率", "gross_margin": "销售毛利率", "ocf_eps": "盈余质量",
    # 成长
    "rev_yoy": "营收同比增速", "profit_yoy": "净利同比增速",
    "rev_qoq": "营收环比增速", "profit_qoq": "净利环比增速",
    # 动量反转
    "mom_5": "5日动量", "mom_10": "10日动量", "mom_20": "20日动量", "mom_60": "60日动量",
    "mom_120": "120日动量", "mom_250": "250日动量", "rev_20": "20日反转", "rev_5": "5日反转",
    # 波动
    "vol_10": "10日波动率", "vol_20": "20日波动率", "vol_60": "60日波动率",
    "vol_120": "120日波动率", "down_vol_20": "20日下行波动",
    "max_ret_20": "20日最大涨幅", "skew_60": "60日收益偏度",
    # 流动性
    "turnover_5": "5日均换手率", "turnover_20": "20日均换手率", "turnover_60": "60日均换手率",
    "amount_5": "5日均成交额", "amount_20": "20日均成交额", "amihud_20": "Amihud非流动性",
    # 规模
    "ln_mv_float": "流通市值对数",
    # 博主「每天一个因子」
    "bb_break_20": "布林上轨突破", "amplitude_3": "振幅", "turnover_heat": "换手升温倍数",
    "rav_4": "相对强弱极端", "gm_yoy": "毛利率同比", "sentiment_20": "情绪因子",
    "alpha144_191": "流动性冲击Alpha144", "vol_ratio_20": "量能比",
    # 「实战」家族
    "lead_cap": "龙头市值集中度", "vol_div": "行业成交分化", "volret_cov": "行业量价协方差",
    "lead_ret_pre": "龙头收益溢价", "range_bias": "振幅乖离", "gap_sent": "隔夜跳空",
    "res_mom": "滚动残差动量", "sent_beta": "情绪Beta", "rel_turn": "相对换手率",
    # 机器学习
    "ml_synth": "ML合成因子",
}


def factor_label(name: str) -> str:
    """页面显示名：英文（中文）；无中文名时退回原名"""
    zh = ZH_NAMES.get(name)
    return f"{name}（{zh}）" if zh else name


def register_factor(name: str, family: str, description: str = "", direction: str = "positive",
                    tags: tuple[str, ...] | list[str] | None = None, formula: str = "",
                    version: int | None = None, source_key: str | None = None,
                    params: dict | None = None):
    """注册因子：装饰器用法

    @register_factor("mom_20", "动量反转", "过去20日动量", "positive")
    def mom_20(d: DataPanel) -> pd.DataFrame:
        return d.close.pct_change(20)

    同名因子被修改时会追加新版本并保留旧版本，见模块 docstring。
    """

    def deco(func: Callable[[DataPanel], pd.DataFrame]):
        digest = _hash_definition(func, source_key)
        history = FACTOR_HISTORY.setdefault(name, [])
        previous = history[-1] if history else None

        if previous is not None and previous.source_hash == digest:
            # 同一份定义重复注册（模块重复导入）：复用记录，不产生新版本
            FACTORS[name] = previous
            return func

        resolved = version if version is not None else (previous.version + 1 if previous else 1)
        # 闭包参数（循环注册的窗口等）自动并入 params，并写进公式注释
        resolved_params = {**_closure_params(func), **(params or {})}
        source = formula or _clean_source(func)
        factor = Factor(
            name=name,
            func=func,
            family=family,
            description=description,
            direction=direction,
            version=resolved,
            tags=tuple(tags) if tags is not None else FAMILY_TAGS.get(family, ()),
            formula=_formula_with_params(source, resolved_params),
            source_hash=digest,
            created_at=_now(),
            params=resolved_params,
        )
        history.append(factor)
        FACTORS[name] = factor
        return func

    return deco


def get_factor(name: str, version: int | None = None) -> Factor | None:
    """取因子；指定 version 时从历史里取该版本（用于复现旧实验）。"""
    if version is None:
        return FACTORS.get(name)
    for candidate in FACTOR_HISTORY.get(name, []):
        if candidate.version == version:
            return candidate
    return None


def list_factor_versions(name: str) -> list[Factor]:
    """某因子的全部历史版本（按版本号）。"""
    return sorted(FACTOR_HISTORY.get(name, []), key=lambda f: f.version)


def list_factors(family: str | None = None) -> list[Factor]:
    fs = list(FACTORS.values())
    if family:
        fs = [f for f in fs if f.family == family]
    return sorted(fs, key=lambda f: (FAMILIES.index(f.family) if f.family in FAMILIES else 99, f.name))


def list_families() -> list[str]:
    """当前实际存在的家族（按 FAMILIES 顺序，未知家族排最后）。"""
    present = {f.family for f in FACTORS.values()}
    known = [f for f in FAMILIES if f in present]
    return known + sorted(present - set(FAMILIES))


def all_tags() -> list[str]:
    return sorted({tag for f in FACTORS.values() for tag in f.tags})


def factor_definitions(names: list[str] | None = None) -> list[dict]:
    """因子定义快照（供实验归档：记录当次运行所用因子的确切版本与源码指纹）。"""
    selected = [FACTORS[n] for n in names if n in FACTORS] if names else list_factors()
    return [f.to_dict() for f in selected]


def registry_path() -> str:
    root = os.environ.get("QFM_FACTOR_REGISTRY")
    if root:
        return root
    return os.path.join(os.path.dirname(__file__), "..", "..", "data_cache", "factors", "registry.json")


def save_registry(path: str | None = None) -> str:
    """把因子库当前状态落盘，便于跨会话核对"因子有没有被改过"。"""
    target = path or registry_path()
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    payload = {
        "saved_at": _now(),
        "count": len(FACTORS),
        "factors": factor_definitions(),
        "history": {
            name: [f.to_dict() for f in versions] for name, versions in sorted(FACTOR_HISTORY.items())
        },
    }
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return target


def load_registry(path: str | None = None) -> dict:
    """读回落盘的因子库快照；不存在时返回空结构。"""
    target = path or registry_path()
    if not os.path.exists(target):
        return {"factors": [], "history": {}}
    try:
        with open(target, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {"factors": [], "history": {}}


def compute_factor(name: str, panel: DataPanel) -> pd.DataFrame:
    f = get_factor(name)
    if f is None:
        raise KeyError(f"未知因子: {name}")
    return f.func(panel)
