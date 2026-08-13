"""动量反转族：过去 N 日累计收益（A 股短期反转效应强）"""

import pandas as pd

from qfm.factors.base import register_factor

WINDOWS = [5, 10, 20, 60, 120, 250]

for _w in WINDOWS:
    def _make(w):
        @register_factor(f"mom_{w}", "动量反转", f"过去 {w} 日动量（累计收益）", "positive")
        def mom(d, _w=w) -> pd.DataFrame:
            return d.close.pct_change(_w, fill_method=None)

        return mom

    _make(_w)


@register_factor("rev_20", "动量反转", "20 日反转 = -20 日动量（A股散户过度反应，跌多易反弹）", "positive")
def rev_20(d) -> pd.DataFrame:
    return -d.close.pct_change(20, fill_method=None)
