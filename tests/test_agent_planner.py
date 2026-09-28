"""规划器测试：规则规划的槽位解析、计划编排、LLM 规划的回落语义。"""

from __future__ import annotations

import json

import pytest

from qfm.agent.llm import LLMConfig, NullClient, OpenAICompatibleClient, build_llm_client
from qfm.agent.planner import (
    DEFAULT_POOL,
    LLMPlanner,
    PlanRequest,
    RulePlanner,
)


@pytest.fixture()
def catalog(registry):
    return registry.catalog()


def plan_for(text, catalog, **kwargs):
    return RulePlanner().plan(PlanRequest(text=text, as_of="2026-09-18", **kwargs), catalog)


# ---------------------------------------------------------------------------
# 槽位解析
# ---------------------------------------------------------------------------
def test_parses_industry_period_and_factor(catalog):
    plan = plan_for("分析最近一年银行股低估值因子的表现", catalog)
    notes = "\n".join(plan.notes)
    assert "银行" in notes
    assert "最近一年" in notes
    assert "低估值" in notes


def test_parses_pool_when_no_industry(catalog):
    plan = plan_for("分析沪深300最近两年的动量因子", catalog)
    assert any("cn_hs300" in note for note in plan.notes)


def test_defaults_to_index800_when_pool_unspecified(catalog):
    """未提到股票池时回落 800 只：全市场是数千只，Agent 演示里代价过高。"""
    plan = plan_for("看看动量因子", catalog)
    step = next(item for item in plan.steps if item.tool == "resolve_universe")
    assert step.arguments["pool"] == DEFAULT_POOL


def test_relative_period_becomes_concrete_dates(catalog):
    plan = plan_for("分析最近一年银行股低估值因子", catalog)
    step = next(item for item in plan.steps if item.tool == "run_experiment")
    assert step.arguments["start"] == "2025-09-18"
    assert step.arguments["end"] == "2026-09-18"


def test_unspecified_period_defaults_to_three_years(catalog):
    plan = plan_for("分析银行股的低估值因子", catalog)
    experiment = next(item for item in plan.steps if item.tool == "run_experiment")
    assert experiment.arguments["start"] == "2023-09-18"


def test_long_short_hint_switches_mode(catalog):
    plan = plan_for("最近一年银行股低估值因子做多空对冲", catalog)
    experiment = next(item for item in plan.steps if item.tool == "run_experiment")
    assert experiment.arguments["mode"] == "long_short"
    assert any("多空" in note for note in plan.notes)


def test_neutralization_hint_enables_industry_size_and_decay(catalog):
    plan = plan_for("最近一年白酒股质量因子做行业中性处理", catalog)
    experiment = next(item for item in plan.steps if item.tool == "run_experiment")
    assert experiment.arguments["neutralization"] == "industry_size"
    assert experiment.arguments["decay"] == 5


def test_compare_hint_adds_comparison_step(catalog):
    plan = plan_for("对比最近两年动量因子和反转因子的表现", catalog)
    assert any(step.tool == "compare_factors" for step in plan.steps)


def test_compare_step_requires_both_hint_and_multiple_factors(catalog):
    """对比步骤的触发条件是「有对比措辞」且「因子数 >= 2」，两个条件缺一不可。

    「动量因子」会展开成 mom_20 / mom_60 两个候选，所以下面第一种写法仍会对比
    ——这是有意的：同一主题的多个窗口值得并排看，且对比结果会提示它们高度相关。
    """
    expanded = plan_for("最近一年动量因子表现对比", catalog)
    assert any(step.tool == "compare_factors" for step in expanded.steps)

    single = plan_for("对比一下 bp 因子", catalog, factors=("bp",))
    assert not any(step.tool == "compare_factors" for step in single.steps)

    no_hint = plan_for("分析最近一年银行股低估值因子的表现", catalog)
    assert not any(step.tool == "compare_factors" for step in no_hint.steps)


def test_explicit_factors_override_parsed_ones(catalog):
    plan = plan_for("分析动量因子", catalog, factors=("roe", "bp"))
    experiments = [step for step in plan.steps if step.tool == "run_experiment"]
    assert [step.arguments["name"] for step in experiments] == ["roe", "bp"]


def test_explicit_pool_overrides_text(catalog):
    plan = plan_for("分析银行股的因子", catalog, pool="cn_zz500")
    step = next(item for item in plan.steps if item.tool == "resolve_universe")
    assert step.arguments["pool"] == "cn_zz500"


def test_explicit_symbols_do_not_leak_into_arguments(catalog):
    """显式标的列表通过上下文预设传递，不能写进计划参数——否则会灌满上下文。"""
    plan = plan_for("分析这批股票", catalog, universe_symbols=("600000", "600036"))
    step = next(item for item in plan.steps if item.tool == "resolve_universe")
    assert step.arguments == {}


# ---------------------------------------------------------------------------
# 计划编排
# ---------------------------------------------------------------------------
def test_plan_covers_the_whole_research_flow(catalog):
    plan = plan_for("分析最近一年银行股低估值因子的表现", catalog)
    tools = [step.tool for step in plan.steps]
    assert tools[0] == "resolve_universe"
    assert tools[1] == "load_panel"
    assert "compute_factor" in tools
    assert "run_experiment" in tools
    assert "check_anomalies" in tools
    assert tools[-1] == "generate_report"
    # 异常检查必须在实验之后、报告之前
    assert tools.index("run_experiment") < tools.index("check_anomalies") < tools.index("generate_report")


def test_load_panel_receives_the_analysis_window(catalog):
    plan = plan_for("分析最近一年银行股低估值因子", catalog)
    step = next(item for item in plan.steps if item.tool == "load_panel")
    assert step.arguments["start"] == "2025-09-18"
    assert step.arguments["end"] == "2026-09-18"


def test_failure_policy_differs_for_primary_and_secondary_factors(catalog):
    """主因子实验失败应中止（结论不可用），次因子失败只是少一个对照。"""
    plan = plan_for("分析最近一年银行股低估值因子", catalog)
    experiments = [step for step in plan.steps if step.tool == "run_experiment"]
    assert experiments[0].on_failure == "abort"
    assert experiments[1].on_failure == "continue"


def test_optional_audit_step_does_not_abort(catalog):
    plan = plan_for("分析最近一年银行股低估值因子", catalog)
    audit = next(item for item in plan.steps if item.tool == "audit_lookahead")
    assert audit.optional is True
    assert audit.on_failure == "skip"


def test_plan_is_json_serialisable(catalog):
    plan = plan_for("分析最近一年银行股低估值因子", catalog)
    payload = json.dumps(plan.to_dict(), ensure_ascii=False)
    assert "resolve_universe" in payload


def test_plan_records_source_as_rule(catalog):
    plan = plan_for("分析银行股低估值因子", catalog)
    assert plan.source == "rule"


# ---------------------------------------------------------------------------
# LLM 规划器
# ---------------------------------------------------------------------------
class FakeClient:
    name = "fake"

    def __init__(self, response):
        self.response = response
        self.calls = []

    @property
    def available(self):
        return True

    def complete(self, messages, **kwargs):
        self.calls.append(messages)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_llm_planner_uses_valid_response(catalog):
    response = json.dumps({
        "steps": [
            {"tool": "resolve_universe", "arguments": {"keyword": "银行"}, "goal": "选池"},
            {"tool": "load_panel", "arguments": {}, "goal": "取数", "on_failure": "abort"},
        ]
    }, ensure_ascii=False)
    plan = LLMPlanner(FakeClient(response)).plan(
        PlanRequest(text="分析银行股", as_of="2026-09-18"), catalog
    )
    assert plan.source == "llm"
    assert [step.tool for step in plan.steps] == ["resolve_universe", "load_panel"]


def test_llm_planner_tolerates_markdown_fence(catalog):
    response = "```json\n" + json.dumps({
        "steps": [{"tool": "load_panel", "arguments": {}, "goal": "取数"}]
    }, ensure_ascii=False) + "\n```"
    plan = LLMPlanner(FakeClient(response)).plan(
        PlanRequest(text="分析银行股", as_of="2026-09-18"), catalog
    )
    assert plan.source == "llm"


@pytest.mark.parametrize(
    "response, reason",
    [
        ("这不是 JSON", "JSON"),
        (json.dumps({"steps": "不是列表"}), "steps"),
        (json.dumps({"steps": [{"tool": "不存在的工具", "arguments": {}}]}), "不存在"),
        (json.dumps({"steps": [{"tool": "load_panel", "arguments": "不是对象"}]}), "arguments"),
        (RuntimeError("连接超时"), "RuntimeError"),
    ],
)
def test_llm_planner_falls_back_on_bad_output(catalog, response, reason):
    plan = LLMPlanner(FakeClient(response)).plan(
        PlanRequest(text="分析最近一年银行股低估值因子", as_of="2026-09-18"), catalog
    )
    assert plan.source == "llm→rule"
    assert any("回落" in note for note in plan.notes)
    # 回落后的计划必须是可执行的完整计划，而不是空壳
    assert any(step.tool == "run_experiment" for step in plan.steps)


def test_llm_planner_rejects_unknown_tool_instead_of_executing(catalog):
    """计划里的工具名必须来自工具目录，否则整份计划作废——不让不可执行的计划落地。"""
    response = json.dumps({
        "steps": [
            {"tool": "load_panel", "arguments": {}},
            {"tool": "drop_database", "arguments": {}},
        ]
    })
    planner = LLMPlanner(FakeClient(response))
    plan = planner.plan(PlanRequest(text="分析银行股", as_of="2026-09-18"), catalog)
    assert plan.source == "llm→rule"
    assert "drop_database" not in [step.tool for step in plan.steps]
    assert planner.last_fallback_reason


def test_llm_planner_prompt_contains_tool_catalog_and_slots(catalog):
    client = FakeClient(json.dumps({"steps": [{"tool": "load_panel", "arguments": {}}]}))
    LLMPlanner(client).plan(
        PlanRequest(text="分析最近一年银行股低估值因子", as_of="2026-09-18"), catalog
    )
    prompt = client.calls[0][1]["content"]
    assert "load_panel" in prompt
    assert "银行" in prompt
    assert "2025-09-18" in prompt


# ---------------------------------------------------------------------------
# LLM 客户端与配置
# ---------------------------------------------------------------------------
def test_build_llm_client_returns_null_without_env():
    client = build_llm_client(environ={})
    assert isinstance(client, NullClient)
    assert client.available is False


def test_build_llm_client_from_env():
    client = build_llm_client(environ={
        "QFM_AGENT_LLM_BASE_URL": "https://api.example.com/v1",
        "QFM_AGENT_LLM_API_KEY": "sk-test",
        "QFM_AGENT_LLM_MODEL": "some-model",
    })
    assert isinstance(client, OpenAICompatibleClient)
    assert client.available is True
    assert client.config.endpoint == "https://api.example.com/v1/chat/completions"


def test_null_client_raises_with_reason():
    with pytest.raises(RuntimeError, match="LLM 不可用"):
        NullClient().complete([])


@pytest.mark.parametrize(
    "base, expected",
    [
        ("https://x.com", "https://x.com/v1/chat/completions"),
        ("https://x.com/v1", "https://x.com/v1/chat/completions"),
        ("https://x.com/v1/chat/completions", "https://x.com/v1/chat/completions"),
        ("https://x.com/", "https://x.com/v1/chat/completions"),
    ],
)
def test_endpoint_normalisation(base, expected):
    config = LLMConfig(base_url=base, api_key="k", model="m")
    assert config.endpoint == expected


def test_llm_config_requires_fields():
    with pytest.raises(ValueError, match="base_url"):
        LLMConfig(base_url="", api_key="k", model="m")
    with pytest.raises(ValueError, match="model"):
        LLMConfig(base_url="https://x.com", api_key="k", model="")
    with pytest.raises(ValueError, match="timeout"):
        LLMConfig(base_url="https://x.com", api_key="k", model="m", timeout=0)


def test_llm_config_dict_excludes_api_key():
    """运行记录会落盘，密钥不能进报告。"""
    config = LLMConfig(base_url="https://x.com", api_key="sk-secret", model="m")
    assert "sk-secret" not in json.dumps(config.to_dict())
