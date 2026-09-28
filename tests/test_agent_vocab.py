"""语义词表测试：行业匹配三级递降、股票池解析、区间解析、因子主题识别。"""

from __future__ import annotations

import pytest

from qfm.agent.vocab import (
    match_factor_themes,
    match_industry,
    parse_period,
    resolve_pool,
    strip_suffix,
)

LABELS = [
    "银行Ⅱ", "白酒Ⅱ", "半导体", "软件开发", "化学制药", "中药Ⅱ",
    "医疗器械", "房地产开发", "汽车零部件", "电网设备", "电力", "食品加工",
]


# ---------------------------------------------------------------------------
# 行业匹配
# ---------------------------------------------------------------------------
def test_exact_label_match():
    assert match_industry("银行Ⅱ", LABELS) == (["银行Ⅱ"], "exact")


@pytest.mark.parametrize(
    "keyword, expected",
    [
        ("银行", ["银行Ⅱ"]),
        ("银行股", ["银行Ⅱ"]),
        ("白酒", ["白酒Ⅱ"]),
        ("白酒板块", ["白酒Ⅱ"]),
        # 剥掉「行业」后缀后与标签完全一致，因此是 exact ——这是最好的结果
        ("半导体行业", ["半导体"]),
    ],
)
def test_alias_and_suffix_stripping(keyword, expected):
    labels, method = match_industry(keyword, LABELS)
    assert labels == expected
    assert method in ("exact", "alias", "substring")


def test_multi_label_alias_returns_union():
    labels, method = match_industry("医药", LABELS)
    assert set(labels) == {"化学制药", "中药Ⅱ", "医疗器械"}
    assert method == "alias"


def test_unknown_keyword_returns_none_method():
    assert match_industry("不存在的行业", LABELS) == ([], "none")


def test_substring_fallback_for_unmapped_terms():
    labels, method = match_industry("汽车部件", LABELS)
    assert labels == []
    # 「汽车部件」不是「汽车零部件」的子串，反向也不成立 → 不该误匹配
    assert method == "none"


def test_substring_matches_longer_keyword_against_shorter_label():
    labels, method = match_industry("半导体设备", LABELS)
    assert labels == ["半导体"]
    assert method == "substring"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("银行股", "银行"),
        ("科技板块", "科技"),
        ("白酒行业", "白酒"),
        ("银行", "银行"),
    ],
)
def test_strip_suffix(text, expected):
    assert strip_suffix(text) == expected


# ---------------------------------------------------------------------------
# 股票池解析
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text, expected",
    [
        ("沪深300", "cn_hs300"),
        ("沪深300指数", "cn_hs300"),
        ("中证500", "cn_zz500"),
        ("中证1000", "cn_zz1000"),
        ("全市场", "full"),
        ("全A股", "cn_all_a"),
        ("index800", "index800"),
    ],
)
def test_resolve_pool(text, expected):
    assert resolve_pool(text) == expected


def test_resolve_pool_returns_none_for_unknown():
    assert resolve_pool("创业板指") is None
    assert resolve_pool("") is None


# ---------------------------------------------------------------------------
# 区间解析
# ---------------------------------------------------------------------------
def test_relative_year_uses_calendar_offset_not_trading_days():
    """「最近一年」必须落在日历一年前。

    用 BDay(252) 会因为节假日累积偏出去一个多月，区间就不再是「一年」。
    """
    start, end, label = parse_period("分析最近一年银行股", as_of="2026-09-18")
    assert start == "2025-09-18"
    assert end == "2026-09-18"
    assert "最近一年" in label


@pytest.mark.parametrize(
    "text, expected_start, expected_end",
    [
        ("最近6个月", "2026-03-18", "2026-09-18"),
        ("近半年", "2026-03-18", "2026-09-18"),
        ("最近三年", "2023-09-18", "2026-09-18"),
        ("最近30个交易日", "2026-08-07", "2026-09-18"),
    ],
)
def test_relative_periods(text, expected_start, expected_end):
    start, end, _ = parse_period(text, as_of="2026-09-18")
    assert (start, end) == (expected_start, expected_end)


def test_absolute_year_range():
    start, end, label = parse_period("2023年到2025年的价值因子", as_of="2026-09-18")
    assert (start, end) == ("2023-01-01", "2025-12-31")
    assert "2023" in label and "2025" in label


def test_absolute_year_range_reversed_is_normalised():
    start, end, _ = parse_period("2025到2023", as_of="2026-09-18")
    assert (start, end) == ("2023-01-01", "2025-12-31")


def test_single_year():
    start, end, label = parse_period("2023年的因子", as_of="2026-09-18")
    assert (start, end) == ("2023-01-01", "2023-12-31")
    assert "全年" in label


def test_no_period_returns_none():
    assert parse_period("分析一下茅台", as_of="2026-09-18") == (None, None, "未指定")


def test_period_defaults_to_today_when_as_of_missing():
    import pandas as pd

    start, end, _ = parse_period("最近一年")
    assert end == str(pd.Timestamp.today().normalize().date())


# ---------------------------------------------------------------------------
# 因子主题
# ---------------------------------------------------------------------------
def test_theme_maps_to_concrete_factors():
    themes, names = match_factor_themes("分析最近一年银行股低估值因子的表现")
    assert "低估值" in themes
    assert names[:2] == ["bp", "ep_ttm"]


def test_explicit_factor_name_is_recognised():
    _, names = match_factor_themes("看看 mom_20 的表现")
    assert "mom_20" in names


def test_chinese_factor_name_is_recognised():
    from qfm.factors import ZH_NAMES

    chinese = next(iter(ZH_NAMES.values()))
    _, names = match_factor_themes(f"分析{chinese}因子")
    assert names, f"中文因子名 {chinese!r} 应能被识别"


def test_multiple_themes_are_ordered_by_position():
    themes, names = match_factor_themes("先看动量再看反转")
    assert themes == ["动量", "反转"]
    assert "mom_20" in names and "rev_20" in names


def test_no_theme_returns_empty():
    themes, names = match_factor_themes("帮我看看这个股票")
    assert themes == []
    assert names == []
