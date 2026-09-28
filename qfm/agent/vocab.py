"""研究语义词表：把自然语言里的行业、股票池、因子主题映射到平台的实际标识。

这一层存在的理由：`qfm` 里的标识符是机器口径的（行业用东财标签 `银行Ⅱ`，
股票池用 `cn_hs300`，因子用 `bp` / `ep_ttm`），而研究员的提问是自然语言的
（「银行股」「沪深300」「低估值因子」）。词表是这两者之间唯一的翻译层，
放在独立模块里是为了让**规划器**和**工具层**共用同一套映射 —— 否则
「规划器选中的行业」和「工具解析出的行业」会漂移。

匹配策略是「精确 → 别名 → 子串」三级递降，并且**始终回报命中的是哪一级**：

- 精确命中（`银行Ⅱ` 直接给出）→ 可信；
- 别名命中（`银行` → `银行Ⅱ`）→ 需要人工核对词表；
- 子串命中（`半导体设备` → 含 `半导体` 的标签）→ 最宽泛，必须回报命中集合。

回报命中级别不是形式主义：把「银行」错配成「银行Ⅱ 以外的其他银行类标签」时，
只有回报了 `method` 才能让使用者发现口径差异（东财行业分类与申万分类不同，
这是研究员最常踩的坑之一）。
"""

from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

__all__ = [
    "SECTOR_ALIASES",
    "POOL_ALIASES",
    "FACTOR_THEMES",
    "match_industry",
    "resolve_pool",
    "strip_suffix",
    "match_factor_themes",
    "parse_period",
]

#: 常见市场术语 → 东财行业标签（部分：一个术语横跨多个细分行业）。
SECTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "银行": ("银行Ⅱ",),
    "白酒": ("白酒Ⅱ",),
    "券商": ("证券Ⅱ",),
    "证券": ("证券Ⅱ",),
    "保险": ("保险Ⅱ",),
    "医药": ("化学制药", "中药Ⅱ", "生物制品", "医疗器械", "医疗服务", "医药商业"),
    "医疗": ("医疗器械", "医疗服务"),
    "科技": (
        "半导体", "软件开发", "光学光电子", "消费电子",
        "计算机设备", "通信设备", "元件", "IT服务Ⅱ",
    ),
    "半导体": ("半导体",),
    "芯片": ("半导体",),
    "消费": ("白酒Ⅱ", "食品加工", "饮料乳品", "家居用品", "家电", "纺织服装"),
    "地产": ("房地产开发", "房地产服务"),
    "房地产": ("房地产开发", "房地产服务"),
    "新能源": ("电池", "光伏设备", "电网设备", "风电设备", "能源金属"),
    "电力": ("电力",),
    "汽车": ("汽车零部件", "汽车整车", "商用车", "乘用车"),
    "化工": ("化学制品", "化学原料", "化学制药"),
    "军工": ("航天装备", "航空装备", "地面兵装", "船舶制造"),
}

#: 自然语言股票池 → 平台股票池标识。键统一转小写比对。
POOL_ALIASES: dict[str, str] = {
    "沪深300": "cn_hs300",
    "hs300": "cn_hs300",
    "cn_hs300": "cn_hs300",
    "300": "cn_hs300",
    "中证500": "cn_zz500",
    "zz500": "cn_zz500",
    "cn_zz500": "cn_zz500",
    "500": "cn_zz500",
    "中证1000": "cn_zz1000",
    "zz1000": "cn_zz1000",
    "cn_zz1000": "cn_zz1000",
    "1000": "cn_zz1000",
    "全a": "cn_all_a",
    "全a股": "cn_all_a",
    "cn_all_a": "cn_all_a",
    "全市场": "full",
    "全部": "full",
    "full": "full",
    "沪深300+中证500": "index800",
    "index800": "index800",
}

#: 因子主题 → 候选因子名（按优先级）。用于把「低估值因子」这类说法映射到具体因子。
FACTOR_THEMES: dict[str, tuple[str, ...]] = {
    "低估值": ("bp", "ep_ttm"),
    "估值": ("bp", "ep_ttm"),
    "价值": ("bp", "ep_ttm"),
    "便宜": ("bp", "ep_ttm"),
    "盈利": ("roe", "ep_ttm"),
    "质量": ("roe", "gross_margin"),
    "成长": ("profit_yoy", "rev_yoy"),
    "景气": ("profit_yoy", "rev_yoy"),
    "动量": ("mom_20", "mom_60"),
    "趋势": ("mom_60", "mom_120"),
    "反转": ("rev_20", "mom_5"),
    "波动": ("vol_20", "down_vol_20"),
    "风险": ("vol_20", "max_ret_20"),
    "流动性": ("turnover_20", "amihud_20"),
    "换手": ("turnover_20", "turnover_heat"),
    "规模": ("ln_mv_float",),
    "小市值": ("ln_mv_float",),
}

#: 需要剥掉的量词/后缀，使「银行股」「沪深300指数」能命中词表。
_SUFFIXES = ("股票", "板块", "行业", "指数", "股", "类", "的")

_PERIOD_UNITS = {
    "年": "year",
    "个月": "month",
    "月": "month",
    "周": "week",
    "天": "day",
    "日": "day",
    "交易日": "bday",
}

_CN_NUMBERS = {
    "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "半": 0.5,
}


def strip_suffix(text: str) -> str:
    """剥掉「银行股」这类查询里的量词后缀，得到可匹配的词干。"""
    cleaned = text.strip()
    changed = True
    while changed and cleaned:
        changed = False
        for suffix in _SUFFIXES:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[: -len(suffix)]
                changed = True
    return cleaned


def match_industry(keyword: str, labels: Iterable[str]) -> tuple[list[str], str]:
    """把行业关键词匹配到实际行业标签。

    Args:
        keyword: 自然语言关键词，如 ``"银行股"``、``"科技板块"``。
        labels: 数据中实际存在的行业标签集合。

    Returns:
        ``(命中标签列表, 匹配方法)``。方法取值为
        ``"exact"`` / ``"alias"`` / ``"substring"`` / ``"none"``。
        命中为空时方法为 ``"none"``。
    """
    available = [str(label) for label in labels]
    raw = keyword.strip()
    stem = strip_suffix(raw)

    for candidate in (raw, stem):
        if candidate in available:
            return [candidate], "exact"

    for candidate in (raw, stem):
        aliased = SECTOR_ALIASES.get(candidate)
        if aliased:
            hits = [label for label in aliased if label in available]
            if hits:
                return hits, "alias"

    hits = [label for label in available if stem and stem in label]
    if hits:
        return hits, "substring"

    # 反向子串：关键词包含完整标签（如「沪深300银行权重」→「银行Ⅱ」不成立，
    # 但「半导体设备」→「半导体」成立，方向已在上一条覆盖；此处补「关键词更长」的情况）
    hits = [label for label in available if label in stem]
    if hits:
        return hits, "substring"
    return [], "none"


def resolve_pool(text: str) -> str | None:
    """把自然语言股票池说法解析成平台股票池标识；无法识别返回 ``None``。"""
    if not text:
        return None
    stem = strip_suffix(text.strip()).lower()
    if stem in POOL_ALIASES:
        return POOL_ALIASES[stem]
    for alias, pool in POOL_ALIASES.items():
        if alias and alias in stem:
            return pool
    return None


def match_factor_themes(text: str) -> tuple[list[str], list[str]]:
    """从文本里找出因子主题与直接提到的因子名。

    Returns:
        ``(主题列表, 因子名列表)``。主题按文本中出现的先后排序，
        因子名去重保序。
    """
    themes: list[tuple[int, str]] = []
    names: list[str] = []
    from qfm.factors import FACTORS, ZH_NAMES

    lowered = text.lower()
    for theme in FACTOR_THEMES:
        position = text.find(theme)
        if position >= 0:
            themes.append((position, theme))

    for name in sorted(FACTORS, key=len, reverse=True):
        if name.lower() in lowered:
            names.append(name)
    for name, zh in ZH_NAMES.items():
        if zh and zh in text and name not in names:
            names.append(name)
    for theme, candidates in FACTOR_THEMES.items():
        if theme in text:
            for candidate in candidates:
                if candidate not in names:
                    names.append(candidate)

    themes.sort()
    return [theme for _, theme in themes], names


def parse_period(text: str, as_of: str | None = None) -> tuple[str | None, str | None, str]:
    """从文本里解析日期区间，返回 ``(start, end, 说明)``。

    支持三类说法：

    - 相对区间：``最近一年`` / ``过去两年`` / ``近半年`` / ``最近12个月`` / ``近30个交易日``；
    - 绝对年份：``2023年`` / ``2023年到2025年``；
    - 无区间：返回 ``(None, None, "未指定")``，由调用方用默认区间。

    Args:
        text: 用户请求文本。
        as_of: 区间终点（通常是数据最晚交易日）。``None`` 时用当前日期。
    """
    end_ts = pd.Timestamp(as_of) if as_of else pd.Timestamp.today().normalize()

    # 绝对区间：2023年 / 2023年到2025年 / 2023-2025
    absolute = re.search(r"(20\d{2})\s*年?\s*(?:到|至|-|~)\s*(20\d{2})\s*年?", text)
    if absolute:
        first, last = int(absolute.group(1)), int(absolute.group(2))
        if first > last:
            first, last = last, first
        return f"{first}-01-01", f"{last}-12-31", f"{first} 年至 {last} 年"

    single_year = re.search(r"(20\d{2})\s*年", text)
    if single_year and "最近" not in text and "过去" not in text:
        year = int(single_year.group(1))
        return f"{year}-01-01", f"{year}-12-31", f"{year} 全年"

    # 相对区间：最近/过去/近 + 数字 + 单位
    relative = re.search(
        r"(最近|过去|近)\s*([0-9]+|[一两二三四五六七八九十半]+)?\s*(个?)(交易日|天|日|周|个月|月|年)",
        text,
    )
    if not relative:
        return None, None, "未指定"

    prefix, raw_number = relative.group(1), relative.group(2)
    unit = relative.group(4)
    if raw_number is None:
        # 「最近一年」这类量词省略时（正则里数字可选），默认 1 个单位
        amount = 1.0
    elif raw_number.isdigit():
        amount = float(raw_number)
    else:
        amount = float(_CN_NUMBERS.get(raw_number, 1))

    kind = _PERIOD_UNITS.get(unit, "year")
    start_ts = _shift_back(end_ts, kind, amount)
    trading_days = int(pd.bdate_range(start_ts, end_ts).size)
    label = f"{prefix}{raw_number or '一'}{'个' if relative.group(3) else ''}{unit}"
    return str(start_ts.date()), str(end_ts.date()), f"{label}（约 {trading_days} 个交易日）"


def _shift_back(end: pd.Timestamp, kind: str, amount: float) -> pd.Timestamp:
    """从区间终点往前推。

    用**日历偏移**而不是交易日偏移：研究员说「最近一年」指的是「一年前的今天」，
    用 `BDay(252)` 会因为节假日累积而偏出去一个多月，区间就不再是「一年」了。
    只有明确说「N 个交易日」时才按交易日回溯。
    """
    if kind == "bday":
        return end - pd.tseries.offsets.BDay(int(round(amount)))
    if kind == "day":
        return end - pd.Timedelta(days=int(round(amount)))
    if kind == "week":
        return end - pd.DateOffset(weeks=int(round(amount)))
    if kind == "month":
        whole = int(amount)
        result = end - pd.DateOffset(months=whole) if whole else end
        if amount - whole:
            result = result - pd.DateOffset(days=int(round((amount - whole) * 30)))
        return result
    whole = int(amount)
    result = end - pd.DateOffset(years=whole) if whole else end
    if amount - whole:
        # 「半年」这类半年为单位的情况按 6 个月折算
        result = result - pd.DateOffset(months=int(round((amount - whole) * 12)))
    return result
