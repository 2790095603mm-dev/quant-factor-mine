"""数据模型与报告渲染的边界测试：状态机、JSON 安全化、缺数据时的报告表现。"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from qfm.agent.models import (
    AgentRun,
    Plan,
    PlanStep,
    RunStatus,
    TaskRecord,
    TaskStatus,
    ToolCall,
    ToolKind,
    ToolSpec,
    _json_safe,
    new_id,
    utc_now,
)
from qfm.agent.reporting import format_metric, render_findings_markdown, render_run_markdown


# ---------------------------------------------------------------------------
# 状态机
# ---------------------------------------------------------------------------
def test_terminal_statuses():
    assert TaskStatus.SUCCESS.terminal
    assert TaskStatus.FAILED.terminal
    assert TaskStatus.DENIED.terminal
    assert TaskStatus.SKIPPED.terminal
    assert not TaskStatus.PENDING.terminal
    assert not TaskStatus.RUNNING.terminal


def test_status_maps_onto_existing_job_status_vocabulary():
    """Agent 层有自己的状态词表，但必须能映射到既有 JobStatus，避免两套语义打架。"""
    assert TaskStatus.SUCCESS.to_job_status() == "SUCCESS"
    assert TaskStatus.FAILED.to_job_status() == "FAILED"
    assert TaskStatus.RUNNING.to_job_status() == "RUNNING"
    assert TaskStatus.RETRYING.to_job_status() == "RUNNING"
    assert TaskStatus.DENIED.to_job_status() == "FAILED"


def test_run_counts_only_non_zero_statuses():
    run = AgentRun(run_id="r", request="q")
    run.tasks = [
        TaskRecord(task_id="1", index=1, goal="", tool="a", status=TaskStatus.SUCCESS),
        TaskRecord(task_id="2", index=2, goal="", tool="b", status=TaskStatus.SUCCESS),
        TaskRecord(task_id="3", index=3, goal="", tool="c", status=TaskStatus.DENIED),
    ]
    assert run.counts() == {"SUCCESS": 2, "DENIED": 1}


def test_run_duration_sums_task_durations():
    run = AgentRun(run_id="r", request="q")
    run.tasks = [
        TaskRecord(task_id="1", index=1, goal="", tool="a", duration_s=1.5),
        TaskRecord(task_id="2", index=2, goal="", tool="b", duration_s=2.0),
    ]
    assert run.duration_s == pytest.approx(3.5)


def test_task_to_dict_is_json_serialisable():
    task = TaskRecord(
        task_id="t1", index=1, goal="取数", tool="load_panel",
        arguments={"pool": "index800"}, status=TaskStatus.SUCCESS,
        attempts=2, cache_hit=True, duration_s=1.234,
        retry_log=({"attempt": 1, "error": "抖动", "will_retry": True},),
        artifacts=("a.json",),
    )
    payload = json.loads(json.dumps(task.to_dict(), ensure_ascii=False))
    assert payload["status"] == "SUCCESS"
    assert payload["attempts"] == 2
    assert payload["retry_log"][0]["error"] == "抖动"


# ---------------------------------------------------------------------------
# 计划与工具规格
# ---------------------------------------------------------------------------
def test_plan_step_rejects_unknown_failure_policy():
    with pytest.raises(ValueError, match="on_failure"):
        PlanStep(tool="load_panel", on_failure="忽略")


def test_plan_round_trips_to_dict():
    plan = Plan(
        request="分析银行股", source="rule", notes=("区间：最近一年",),
        steps=[PlanStep(tool="load_panel", arguments={"start": "2025-01-01"}, goal="取数")],
    )
    payload = plan.to_dict()
    assert payload["steps"][0]["tool"] == "load_panel"
    assert payload["notes"] == ["区间：最近一年"]


def test_tool_spec_catalog_line_shows_required_markers():
    spec = ToolSpec(
        name="run_experiment", kind=ToolKind.COMPUTE, description="跑实验",
        parameters={"name": "str，因子名", "horizon": "int，前瞻"}, required=("name",),
    )
    line = spec.catalog_line()
    assert "run_experiment" in line
    assert "必填: name" in line


def test_tool_call_gets_generated_id():
    call = ToolCall(tool="load_panel")
    assert call.call_id.startswith("call_")
    assert ToolCall(tool="load_panel", call_id="fixed").call_id == "fixed"


# ---------------------------------------------------------------------------
# JSON 安全化：上下文管理的关键依赖
# ---------------------------------------------------------------------------
def test_dataframe_is_summarised_by_shape_not_serialised():
    """这条约束是上下文管理能成立的根基：矩阵绝不能逐值进 JSON。"""
    frame = pd.DataFrame(np.random.rand(2000, 800))
    payload = _json_safe({"matrix": frame})
    assert payload["matrix"]["shape"] == [2000, 800]
    assert len(json.dumps(payload)) < 500


def test_series_is_summarised_not_stringified():
    payload = _json_safe(pd.Series([0.1, 0.2, 0.3], name="ic"))
    assert payload["name"] == "ic"
    assert payload["size"] == 3
    assert payload["head"] == [0.1, 0.2, 0.3]


def test_ndarray_and_numpy_scalars():
    assert _json_safe(np.arange(50))["shape"] == [50]
    assert _json_safe(np.int64(7)) == 7
    assert _json_safe(np.float64("nan")) is None
    assert _json_safe(np.float64("inf")) is None


def test_nested_containers_and_timestamps():
    payload = _json_safe({
        "date": pd.Timestamp("2026-09-18"),
        "items": [1, {"a": (2, 3)}],
        "tags": {"x"},
    })
    assert payload["date"].startswith("2026-09-18")
    assert payload["items"][1]["a"] == [2, 3]
    assert payload["tags"] == ["x"]


def test_unknown_objects_fall_back_to_str():
    class Weird:
        pass

    assert isinstance(_json_safe(Weird()), str)


def test_ids_are_unique_and_timestamped():
    assert new_id("task") != new_id("task")
    assert new_id("task").startswith("task_")
    assert "T" in utc_now()


# ---------------------------------------------------------------------------
# 指标格式化
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, kind, expected",
    [
        (0.0346, "ic", "+0.0346"),
        (3.93, "t", "+3.93"),
        (0.1234, "ratio", "12.34%"),
        (-0.0523, "pct", "-5.23%"),
        (2025.0, "int", "2025"),
        (None, "ic", "—"),
        (float("nan"), "number", "—"),
        (float("inf"), "number", "—"),
    ],
)
def test_format_metric(value, kind, expected):
    assert format_metric(value, kind) == expected


def test_format_metric_falls_back_for_non_numeric():
    assert format_metric("非数值") == "非数值"


# ---------------------------------------------------------------------------
# 报告渲染：缺数据与诚实性
# ---------------------------------------------------------------------------
def test_findings_report_without_experiment_says_so():
    body = render_findings_markdown("空报告", experiment=None)
    assert "没有产出实验结果" in body


def test_findings_report_marks_synthetic_data_prominently():
    experiment = _experiment()
    body = render_findings_markdown(
        "测试", experiment=experiment, panel_meta={"source": "synthetic"}
    )
    assert "不是真实行情数据" in body
    assert "不构成任何实证结论" in body


def test_findings_report_has_no_nan_or_none_tokens():
    experiment = _experiment()
    experiment["metrics"]["卡玛比率"] = float("nan")
    experiment["ic"]["ic_ir"] = None
    body = render_findings_markdown("测试", experiment=experiment, panel_meta={})
    assert "nan" not in body.lower()
    assert "None" not in body
    assert "—" in body


def test_findings_report_declares_missing_layer_test():
    experiment = _experiment()
    experiment["layer_available"] = False
    body = render_findings_markdown("测试", experiment=experiment, panel_meta={})
    assert "**不可用。**" in body
    assert "只能依据 IC 判断" in body


def test_findings_report_conclusion_reflects_negative_ic():
    """结论必须由证据推出：IC 显著为负就不能写成「观察到预测力」。"""
    experiment = _experiment()
    experiment["ic"]["ic_mean"] = -0.12
    experiment["ic"]["ic_t"] = -6.0
    body = render_findings_markdown("测试", experiment=experiment, panel_meta={})
    assert "IC 显著为负" in body
    assert "不应直接采用" in body


def test_findings_report_conclusion_flags_insignificant_ic():
    experiment = _experiment()
    experiment["ic"]["ic_mean"] = 0.01
    experiment["ic"]["ic_t"] = 0.7
    body = render_findings_markdown("测试", experiment=experiment, panel_meta={})
    assert "未达显著水平" in body


def test_findings_report_control_section_reports_opposite_direction():
    primary = _experiment()
    control = _experiment()
    control["pool"] = "index800"
    control["n_symbols"] = 170
    control["ic"]["ic_mean"] = -0.02
    control["ic"]["ic_t"] = -1.5
    body = render_findings_markdown(
        "测试", experiment=control, primary=primary, panel_meta={}
    )
    assert "## 五、宽基对照" in body
    assert "方向相反" in body
    assert "不能推广到更宽的市场范围" in body


def test_findings_report_control_section_reports_consistent_direction():
    primary = _experiment()
    control = _experiment()
    control["pool"] = "index800"
    control["ic"]["ic_mean"] = 0.025
    body = render_findings_markdown(
        "测试", experiment=control, primary=primary, panel_meta={}
    )
    assert "方向一致" in body
    assert "不是样本宽度造成的假象" in body


def test_run_report_lists_retry_and_permission_sections():
    run = AgentRun(run_id="agrun_x", request="分析银行股", status=RunStatus.PARTIAL)
    run.plans.append(Plan(request="分析银行股", steps=[PlanStep(tool="load_panel")]))
    run.tasks.append(TaskRecord(
        task_id="t", index=1, goal="取数", tool="load_panel",
        status=TaskStatus.SUCCESS, attempts=2, duration_s=0.5,
        retry_log=({"attempt": 1, "error_type": "ConnectionError",
                    "classification": "transient", "delay_s": 0.5, "will_retry": True},),
    ))
    run.permission_events = ({"tool": "generate_report", "decision": "denied",
                              "rule": "read_only", "reason": "只读模式"},)
    run.context_stats = {"items": 5, "chars": 1200, "original_chars": 9000,
                         "compression_ratio": 0.13, "offloaded_items": 2, "folded_items": 0}
    body = render_run_markdown(run)
    assert "## 重试轨迹" in body
    assert "ConnectionError" in body
    assert "## 权限事件" in body
    assert "read_only" in body
    assert "## 上下文管理" in body
    assert "压缩比" in body


def test_run_report_escapes_pipes_in_error_text():
    run = AgentRun(run_id="agrun_y", request="q")
    run.tasks.append(TaskRecord(
        task_id="t", index=1, goal="", tool="x", status=TaskStatus.FAILED,
        error="字段 a|b 非法",
    ))
    body = render_run_markdown(run)
    assert "a\\|b" in body, "表格里的竖线必须转义，否则报告表格会错位"


def _experiment() -> dict:
    return {
        "key": "bp@industry:银行Ⅱ",
        "name": "bp",
        "label": "账面市值比",
        "family": "价值",
        "direction": "positive",
        "pool": "industry:银行Ⅱ",
        "n_symbols": 42,
        "horizon": 20,
        "mode": "long_only",
        "rebalance": "ME",
        "neutralization": "none",
        "decay": 0,
        "top_n_requested": 30,
        "top_n_effective": 30,
        "layer_available": False,
        "turnover": 0.27,
        "coverage": 1.0,
        "cache_hit": False,
        "window": {"start": "2025-09-18", "end": "2026-09-18", "trading_days": 262},
        "ic": {"ic_mean": 0.034, "ic_ir": 0.25, "ic_t": 3.9, "pos_ratio": 0.57, "n_days": 242},
        "metrics": {"年化收益": 0.064, "年化超额": 0.022, "夏普比率": 1.6, "最大回撤": -0.024,
                    "卡玛比率": 2.7, "日均换手": 0.008, "累计交易成本": 0.003},
        "monotonicity": {"monotonic": False},
        "yearly": [{"年份": 2026.0, "收益": 0.03, "IC": 0.03, "最大回撤": -0.02, "夏普": 1.0}],
        "warnings": [],
    }
