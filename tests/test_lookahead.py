"""无未来函数检查测试：泄漏模式命中 / 合法源码放行 / 结构性判定 / factor_report 集成"""

from __future__ import annotations

from qfm.factors import compute_factor
from qfm.pipeline.lookahead import check_structural, scan_source
from qfm.pipeline.tests import factor_report


def test_scan_hits_leaks():
    src = "def f(d):\n    return d.close.shift(-1) / d.close"
    r = scan_source(src)
    assert r["leaks"], "负 shift 必须命中"


def test_scan_hits_iloc_and_rolling():
    assert scan_source("x = s.iloc[-1]")["leaks"]
    assert scan_source("x = s.rolling(-5).mean()")["leaks"]


def test_scan_clean_code_no_leaks():
    src = "def f(d):\n    return d.close.pct_change(20) / d.turnover.rolling(20).mean()"
    r = scan_source(src)
    assert not r["leaks"]


def test_scan_shift1_is_warning_only():
    r = scan_source("x = d.close.shift(1)")
    assert not r["leaks"] and r["warnings"]


def test_check_structural():
    assert check_structural("mining") == "pass"
    assert check_structural("custom") == "review"


def test_factor_report_lookahead_field(panel):
    rep = factor_report(compute_factor("mom_20", panel), panel.close, horizon=20,
                        lookahead={"status": "pass", "hits": []})
    assert rep["lookahead"]["status"] == "pass"
    rep2 = factor_report(compute_factor("mom_20", panel), panel.close, horizon=20)
    assert "lookahead" not in rep2
