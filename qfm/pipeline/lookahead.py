"""无未来函数检查：结构性判定 + 静态泄漏模式扫描

方法论源自 QuantSkills skill-quant-factor-skill-factory 的 no-lookahead 检查思想，
本项目按因子计算语义独立实现（静态模式扫描，非 GPL 代码拷贝）。
"""

from __future__ import annotations

import re

# (正则, 描述) —— 命中即泄漏（引用未来数据）
LEAK_PATTERNS: list[tuple[str, str]] = [
    (r"\.shift\(\s*-\d", "负 shift：引用未来数据"),
    (r"\.iloc\[\s*-", "负 iloc：引用序列末尾（未来）数据"),
    (r"rolling\(\s*-", "负窗口 rolling"),
    (r"pct_change\s*\([^)]*-\d", "pct_change 负周期"),
    # \b 词边界：避免命中 from __future__ import 等合法标识
    (r"fwd|forward_return|\bfuture\b|\blabel\b|\btarget\b", "疑似未来收益/标签变量"),
]

# 提示级（不必然泄漏，但需人工确认）
WARN_PATTERNS: list[tuple[str, str]] = [
    (r"\.shift\(1\)", "shift(1)：引用 T-1 日数据（安全，请确认方向）"),
]


def scan_source(src: str) -> dict:
    """静态扫描因子源码，返回 {leaks: [描述], warnings: [描述]}"""
    leaks = [d for p, d in LEAK_PATTERNS if re.search(p, src)]
    warns = [d for p, d in WARN_PATTERNS if re.search(p, src)]
    return {"leaks": leaks, "warnings": warns}


def check_structural(kind: str) -> str:
    """结构性判定：挖掘候选（指标×窗口×变换）全部基于 T 日及以前数据 → pass"""
    return "pass" if kind == "mining" else "review"
