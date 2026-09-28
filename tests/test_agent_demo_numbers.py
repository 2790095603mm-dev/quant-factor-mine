"""锁定 `docs/agent-demo.md` 里引用的演示数字。

## 为什么要专门写这个测试

演示文档里的数字是手抄的，而手抄的数字迟早会和代码漂移。与其靠人工复查，不如
让数字变成断言：这里用合成面板（固定种子）跑演示文档里那句请求，把主实验与
对照实验的关键指标钉住。

数字一旦变化（改了因子实现、改了管线口径、改了合成面板参数、改了组合配置），
这个测试就会失败，从而强制同步文档——**文档里写错数字比不写数字更糟**。

本文件里的期望值对应 `docs/agent-demo.md` 的「二、研究报告节选」与
「这次运行里值得注意的几件事」两节。
"""

from __future__ import annotations

import pytest

from qfm.agent.planner import RulePlanner
from qfm.agent.providers import SyntheticProvider
from qfm.agent.retry import RetryPolicy
from qfm.agent.runtime import AgentConfig, AgentRuntime
from qfm.agent.synthetic import SyntheticSpec
from qfm.agent.tools import build_default_registry

#: 演示文档里使用的那句请求
DEMO_REQUEST = "分析最近一年银行股低估值因子的表现"

#: docs/agent-demo.md 中引用的主实验指标
EXPECTED_PRIMARY = {
    "name": "bp",
    "pool": "industry:银行Ⅱ",
    "n_symbols": 42,
    "ic_mean": 0.0346,
    "ic_t": 3.93,
    "ic_ir": 0.2529,
    "pos_ratio": 0.570,
    "annual_return": 0.0641,
    "excess_return": 0.0225,
    "sharpe": 1.6069,
    "max_drawdown": -0.0237,
    "turnover": 0.2769,
}

#: docs/agent-demo.md 中引用的宽基对照指标
EXPECTED_CONTROL = {
    "name": "bp",
    "pool": "index800",
    "n_symbols": 170,
    "ic_mean": 0.0252,
    "ic_t": 4.30,
    "ic_ir": 0.2765,
    "pos_ratio": 0.653,
    "annual_return": 0.1054,
    "excess_return": 0.0201,
    "sharpe": 2.0266,
    "max_drawdown": -0.0570,
    "turnover": 0.1998,
}

#: docs/agent-demo.md 中引用的主实验异常代码
EXPECTED_ANOMALIES = {"narrow_cross_section", "layer_test_unavailable"}


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    """跑一次演示请求（固定种子），模块内复用。"""
    provider = SyntheticProvider(SyntheticSpec(as_of="2026-09-18", n_days=620, seed=20260928))
    runs_root = tmp_path_factory.mktemp("demo_runs")
    runtime = AgentRuntime(
        build_default_registry(),
        RulePlanner(),
        AgentConfig(retry=RetryPolicy(max_attempts=1)),
        provider=provider,
        runs_root=runs_root,
        sleep=lambda seconds: None,
    )
    return runtime.run(DEMO_REQUEST)


def test_demo_run_succeeds(demo_run):
    assert demo_run.status.value == "SUCCESS"
    assert demo_run.findings["primary"] is not None
    assert demo_run.findings["control"] is not None


@pytest.mark.parametrize("field, expected", sorted(EXPECTED_PRIMARY.items()))
def test_primary_metrics_match_demo_doc(demo_run, field, expected):
    _assert_matches(EXPECTED_PRIMARY, demo_run.findings["primary"], field, "主实验")


@pytest.mark.parametrize("field, expected", sorted(EXPECTED_CONTROL.items()))
def test_control_metrics_match_demo_doc(demo_run, field, expected):
    _assert_matches(EXPECTED_CONTROL, demo_run.findings["control"], field, "对照实验")


def _assert_matches(expected_map, payload: dict, field: str, label: str) -> None:
    """按文档里的有效位数比较。

    文档写 `3.93` 而实际是 `3.934089…`，直接比数值永远过不了；正确的做法是
    把实际值四舍五入到文档的精度再比——这样既守住精度，又不会因为多写几位就报错。
    """
    expected = expected_map[field]
    actual = _extract(payload, field)
    if isinstance(expected, str):
        assert actual == expected, f"{label} {field} 变为 {actual!r}，文档记载 {expected!r}"
        return
    rounded = _round_like(actual, expected)
    assert rounded == pytest.approx(expected, abs=10.0 ** -(_decimals(expected) + 1)), (
        f"{label} {field} 变为 {actual}（四舍五入到文档精度为 {rounded}），"
        f"与 docs/agent-demo.md 记载的 {expected} 不一致；"
        "请确认是口径变化还是文档过期，并同步更新文档与本测试"
    )


def _decimals(value: float) -> int:
    text = repr(float(value))
    if "e" in text or "E" in text or "." not in text:
        return 0
    return len(text.split(".")[1])


def _round_like(actual: float, expected: float) -> float:
    return round(float(actual), _decimals(expected))


def test_demo_anomalies_match_demo_doc(demo_run):
    codes = {item["code"] for item in demo_run.findings["anomalies"]}
    assert codes == EXPECTED_ANOMALIES


def test_demo_structure_matches_demo_doc(demo_run):
    """演示文档里承诺的结构：12 步主计划 + 6 步补充计划。"""
    assert [len(plan.steps) for plan in demo_run.plans] == [12, 6]
    assert demo_run.plans[1].source == "remediation"
    assert len(demo_run.tasks) == 18
    assert all(task.status.value == "SUCCESS" for task in demo_run.tasks)


def test_demo_plan_notes_match_demo_doc(demo_run):
    """演示文档里引用的规划槽位。"""
    notes = " ".join(demo_run.plans[0].notes)
    assert "行业关键词：'银行'" in notes
    assert "最近一年" in notes
    assert "bp, ep_ttm" in notes


def test_demo_data_source_is_labeled_synthetic(demo_run):
    """演示用的是合成数据，这一点必须在运行记录与报告里都体现。"""
    assert demo_run.data_source == "synthetic"
    assert "合成" in demo_run.data_note
    report = open(demo_run.findings["research_report"], encoding="utf-8").read()
    assert "不是真实行情数据" in report
    assert "不构成任何实证结论" in report


def test_demo_context_stays_small(demo_run):
    """演示文档声称「整轮运行上下文只占数千字符」，这里守住这个量级。"""
    stats = demo_run.context_stats
    assert stats["items"] <= 30
    assert stats["chars"] < 5000, f"上下文膨胀到 {stats['chars']} 字符"


def _extract(payload: dict, field: str):
    return {
        "name": lambda: payload.get("name"),
        "pool": lambda: payload.get("pool"),
        "n_symbols": lambda: payload.get("n_symbols"),
        "turnover": lambda: payload.get("turnover"),
        "ic_mean": lambda: payload["ic"]["ic_mean"],
        "ic_t": lambda: payload["ic"]["ic_t"],
        "ic_ir": lambda: payload["ic"]["ic_ir"],
        "pos_ratio": lambda: payload["ic"]["pos_ratio"],
        "annual_return": lambda: payload["metrics"]["年化收益"],
        "excess_return": lambda: payload["metrics"]["年化超额"],
        "sharpe": lambda: payload["metrics"]["夏普比率"],
        "max_drawdown": lambda: payload["metrics"]["最大回撤"],
    }[field]()
