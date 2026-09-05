"""研究项目页面的纯展示辅助逻辑测试。"""

from __future__ import annotations

import math

import pytest

from qfm.research.views import _format_metric


@pytest.mark.parametrize(
    ("value", "template", "expected"),
    [
        (None, "{:+.1%}", "—"),
        (math.nan, "{:.2f}", "—"),
        (0.1234, "{:+.1%}", "+12.3%"),
        (1.234, "{:.2f}", "1.23"),
    ],
)
def test_format_metric_uses_placeholder_for_missing_or_nonfinite_values(value, template, expected):
    assert _format_metric(value, template) == expected
