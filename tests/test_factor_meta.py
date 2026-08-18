"""因子元数据测试：中文名全覆盖 / 标签格式"""

from __future__ import annotations

from qfm.factors import ZH_NAMES, factor_label, list_factors


def test_all_factors_have_zh_name():
    names = [f.name for f in list_factors()]
    missing = [n for n in names if n not in ZH_NAMES]
    assert not missing, f"缺中文名的因子: {missing}"


def test_factor_label_format():
    assert factor_label("ep_ttm") == "ep_ttm（盈利收益率TTM）"
    assert factor_label("no_such_factor") == "no_such_factor"   # 无中文名退回原名


def test_zh_names_no_duplicate_keys():
    assert len(ZH_NAMES) == len(set(ZH_NAMES))
