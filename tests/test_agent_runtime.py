"""运行时端到端测试：状态机、权限路径、重试路径、异常驱动的重新规划、落盘产物。

全部在合成面板上离线运行，并注入 `sleep=lambda _: None` 使退避不实际等待。
"""

from __future__ import annotations

import json

import pytest

from qfm.agent.errors import PermanentError
from qfm.agent.models import RunStatus, TaskStatus
from qfm.agent.permissions import PermissionPolicy
from qfm.agent.planner import Plan, RulePlanner
from qfm.agent.retry import RetryPolicy
from qfm.agent.runtime import AgentConfig, AgentRuntime

BANK_REQUEST = "分析最近一年银行股低估值因子的表现"


def make_runtime(provider, registry, tmp_path, **overrides):
    overrides.setdefault("retry", RetryPolicy(max_attempts=2, base_delay=0.0, jitter=0.0))
    config = AgentConfig(**overrides)
    return AgentRuntime(
        registry,
        RulePlanner(),
        config,
        provider=provider,
        runs_root=tmp_path / "runs",
        sleep=lambda seconds: None,
    )


# ---------------------------------------------------------------------------
# 正常路径
# ---------------------------------------------------------------------------
def test_full_run_succeeds_on_synthetic_data(synthetic_provider, registry, tmp_path):
    runtime = make_runtime(synthetic_provider, registry, tmp_path)
    run = runtime.run(BANK_REQUEST)

    assert run.status is RunStatus.SUCCESS
    assert run.counts().get("SUCCESS") == len(run.tasks)
    assert not run.permission_events

    primary = run.findings["primary"]
    assert primary["name"] == "bp"
    assert primary["pool"] == "industry:银行Ⅱ"
    assert primary["n_symbols"] == 42
    assert primary["ic"]["ic_mean"] is not None
    assert primary["metrics"]["年化收益"] is not None


def test_run_records_data_source_and_labels_synthetic(synthetic_provider, registry, tmp_path):
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    assert run.data_source == "synthetic"
    assert "合成" in run.data_note
    # 合成数据必须留下可追踪的告警，避免结果被误读为实证结论
    assert any("合成" in warning for warning in run.warnings) or run.findings["primary"]["data_source"] == "synthetic"


def test_run_writes_report_and_run_json(synthetic_provider, registry, tmp_path):
    runtime = make_runtime(synthetic_provider, registry, tmp_path)
    run = runtime.run(BANK_REQUEST)

    run_dir = tmp_path / "runs" / run.run_id
    assert (run_dir / "run.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "context_digest.txt").exists()

    payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert payload["status"] == "SUCCESS"
    assert payload["task_counts"]["SUCCESS"] > 0
    assert payload["data_source"] == "synthetic"


def test_run_report_contains_trace_and_context_stats(synthetic_provider, registry, tmp_path):
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    body = run.report_markdown
    assert "## 逐步执行轨迹" in body
    assert "## 上下文管理" in body
    assert "resolve_universe" in body
    assert run.context_stats["items"] > 0
    assert run.context_stats["chars"] > 0


def test_findings_report_is_written_and_cited(synthetic_provider, registry, tmp_path):
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    report_path = run.findings["research_report"]
    assert report_path
    body = open(report_path, encoding="utf-8").read()
    assert "## 结论与局限" in body
    assert "合成面板" in body


# ---------------------------------------------------------------------------
# 异常驱动的重新规划
# ---------------------------------------------------------------------------
def test_narrow_universe_triggers_wide_control_run(synthetic_provider, registry, tmp_path):
    """银行池只有 42 只 → 异常检查报高优异常 → 自动补跑宽基对照。"""
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)

    assert len(run.plans) == 2, "应产生一份补充计划"
    assert run.plans[1].source == "remediation"
    assert any("截面宽度不足" in note for note in run.plans[1].notes)

    primary = run.findings["primary"]
    control = run.findings["control"]
    assert control is not None
    assert control["pool"] == "index800"
    assert control["n_symbols"] > primary["n_symbols"]
    assert control["layer_available"] is True, "宽基池应能进行分层检验"


def test_control_run_details_do_not_leak_into_primary(synthetic_provider, registry, tmp_path):
    """对照实验不得覆盖主实验，否则用户问的那个池子的结果就丢了。"""
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    primary = run.findings["primary"]
    assert primary["pool"] == "industry:银行Ⅱ"
    assert primary["n_symbols"] == 42
    assert run.findings["primary_universe"] == "industry:银行Ⅱ"


def test_anomalies_are_bound_to_their_own_experiment(synthetic_provider, registry, tmp_path):
    """主实验的异常不能混入对照实验的指标证据。"""
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    by_experiment = run.findings["anomalies_by_experiment"]
    assert "bp@industry:银行Ⅱ" in by_experiment
    assert "bp@index800" in by_experiment

    narrow = next(
        item for item in by_experiment["bp@industry:银行Ⅱ"]["anomalies"]
        if item["code"] == "narrow_cross_section"
    )
    assert narrow["evidence"]["n_symbols"] == 42
    # 宽基对照有 170 只，不应报截面不足
    control_codes = {item["code"] for item in by_experiment["bp@index800"]["anomalies"]}
    assert "narrow_cross_section" not in control_codes


def test_report_compares_primary_and_control(synthetic_provider, registry, tmp_path):
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    body = open(run.findings["research_report"], encoding="utf-8").read()
    assert "## 五、宽基对照" in body
    assert "两次实验**只有股票池不同**" in body
    assert "稳健性（宽基对照）" in body


def test_remediation_is_capped_and_does_not_loop(synthetic_provider, registry, tmp_path):
    runtime = make_runtime(synthetic_provider, registry, tmp_path, max_replan_rounds=1)
    run = runtime.run(BANK_REQUEST)
    assert len(run.plans) <= 2
    assert all(plan.round <= 1 for plan in run.plans)


def test_remediation_disabled_by_config(synthetic_provider, registry, tmp_path):
    runtime = make_runtime(synthetic_provider, registry, tmp_path, max_replan_rounds=0)
    run = runtime.run(BANK_REQUEST)
    assert len(run.plans) == 1
    assert run.findings["control"] is None
    # 即使没有补救，异常依然要如实报告
    assert any(item["severity"] == "high" for item in run.findings["anomalies"])


def test_high_severity_anomaly_is_promoted_to_warning(synthetic_provider, registry, tmp_path):
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    assert any("narrow_cross_section" in warning for warning in run.warnings)


# ---------------------------------------------------------------------------
# 权限路径
# ---------------------------------------------------------------------------
def test_read_only_mode_denies_write_tool_and_run_becomes_partial(
    synthetic_provider, registry, tmp_path
):
    runtime = make_runtime(
        synthetic_provider, registry, tmp_path, permissions=PermissionPolicy(read_only=True)
    )
    run = runtime.run(BANK_REQUEST)

    denied = [task for task in run.tasks if task.status is TaskStatus.DENIED]
    # 补救轮的补充计划里也有 generate_report，所以会有两条被拒记录
    assert denied and {task.tool for task in denied} == {"generate_report"}
    assert run.status is RunStatus.PARTIAL
    assert run.findings["primary"] is not None, "被拒的是写盘工具，研究结论仍然产出"
    assert any(event["rule"] == "read_only" for event in run.permission_events)


def test_denied_abort_step_stops_the_plan(synthetic_provider, registry, tmp_path):
    """关键前置步骤被拒时不能继续往下算，也不能伪造结论。"""
    runtime = make_runtime(
        synthetic_provider, registry, tmp_path,
        permissions=PermissionPolicy(denied=frozenset({"load_panel"})),
    )
    run = runtime.run(BANK_REQUEST)

    assert run.status is RunStatus.FAILED
    assert run.findings["primary"] is None
    assert any("前置依赖" in warning for warning in run.warnings)
    # 取数之后的步骤不应被执行
    tools = [task.tool for task in run.tasks]
    assert "run_experiment" not in tools
    assert "generate_report" not in tools


def test_allowlist_restricts_run(synthetic_provider, registry, tmp_path):
    runtime = make_runtime(
        synthetic_provider, registry, tmp_path,
        permissions=PermissionPolicy(allowed=frozenset({"resolve_universe", "load_panel"})),
    )
    run = runtime.run(BANK_REQUEST)
    assert run.status is RunStatus.FAILED
    # 白名单之外的步骤必须全部被拒，没有任何一个真正执行
    executed = {task.tool for task in run.tasks if task.status is TaskStatus.SUCCESS}
    assert executed <= {"resolve_universe", "load_panel"}
    assert any(task.status is TaskStatus.DENIED for task in run.tasks)


def test_call_budget_denies_after_limit(synthetic_provider, registry, tmp_path):
    runtime = make_runtime(
        synthetic_provider, registry, tmp_path,
        permissions=PermissionPolicy(max_total_calls=5),
    )
    run = runtime.run(BANK_REQUEST)
    denied = [task for task in run.tasks if task.status is TaskStatus.DENIED]
    assert denied, "预算耗尽后应出现被拒步骤"
    assert any(event["rule"] == "total_calls_budget" for event in run.permission_events)
    assert run.status in (RunStatus.PARTIAL, RunStatus.FAILED)


def test_approver_enables_confirmed_tool(synthetic_provider, registry, tmp_path):
    approvals = []

    def approver(spec, args):
        approvals.append(spec.name)
        return True

    config = AgentConfig(
        retry=RetryPolicy(max_attempts=1),
        permissions=PermissionPolicy(confirm_required=frozenset({"generate_report"})),
    )
    runtime = AgentRuntime(
        registry, RulePlanner(), config, provider=synthetic_provider,
        runs_root=tmp_path / "runs", sleep=lambda s: None, approver=approver,
    )
    run = runtime.run(BANK_REQUEST)
    assert "generate_report" in approvals
    assert run.status is RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# 重试路径
# ---------------------------------------------------------------------------
def test_transient_tool_failure_is_retried_then_succeeds(synthetic_provider, registry, tmp_path):
    """把第一次 compute_factor 调用打成瞬时故障，验证运行时确实重试并成功。"""
    original = registry.get("compute_factor").fn
    state = {"calls": 0}

    def flaky(context, **kwargs):
        state["calls"] += 1
        if state["calls"] == 1:
            raise ConnectionError("数据源临时不可用")
        return original(context=context, **kwargs)

    registry.replace("compute_factor", flaky)

    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    retried = [task for task in run.tasks if task.retry_log]
    assert retried, "应记录重试轨迹"
    assert retried[0].attempts == 2
    assert retried[0].status is TaskStatus.SUCCESS
    assert retried[0].retry_log[0]["classification"] == "transient"


def test_permanent_step_failure_is_not_retried(synthetic_provider, registry, tmp_path):
    state = {"calls": 0}

    def broken(context, **kwargs):
        state["calls"] += 1
        raise PermanentError("因子定义损坏")

    registry.replace("describe_factor", broken)
    run = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    assert state["calls"] == 1, "确定性错误不应重试"
    failed = [task for task in run.tasks if task.status is TaskStatus.FAILED]
    assert failed and failed[0].attempts == 1
    assert run.status is RunStatus.FAILED


def test_retry_policy_can_disable_retries(synthetic_provider, registry, tmp_path):
    state = {"calls": 0}

    def flaky(context, **kwargs):
        state["calls"] += 1
        raise ConnectionError("总是失败")

    registry.replace("compute_factor", flaky)
    runtime = make_runtime(synthetic_provider, registry, tmp_path, retry=RetryPolicy(max_attempts=1))
    run = runtime.run(BANK_REQUEST)
    assert state["calls"] == 1
    assert run.status is RunStatus.FAILED


# ---------------------------------------------------------------------------
# 事件流
# ---------------------------------------------------------------------------
def test_events_are_emitted_for_live_display(synthetic_provider, registry, tmp_path):
    events = []

    config = AgentConfig(retry=RetryPolicy(max_attempts=1))
    runtime = AgentRuntime(
        registry, RulePlanner(), config, provider=synthetic_provider,
        runs_root=tmp_path / "runs", sleep=lambda s: None,
        on_event=lambda event, payload: events.append((event, payload)),
    )
    runtime.run(BANK_REQUEST)

    names = [event for event, _ in events]
    assert "run_started" in names
    assert "planned" in names
    assert "step_started" in names
    assert "step_finished" in names
    assert "replanned" in names
    assert names[-1] == "run_finished"


def test_broken_event_hook_does_not_break_the_run(synthetic_provider, registry, tmp_path):
    def broken(event, payload):
        raise RuntimeError("展示层炸了")

    config = AgentConfig(retry=RetryPolicy(max_attempts=1))
    runtime = AgentRuntime(
        registry, RulePlanner(), config, provider=synthetic_provider,
        runs_root=tmp_path / "runs", sleep=lambda s: None, on_event=broken,
    )
    run = runtime.run(BANK_REQUEST)
    assert run.status is RunStatus.SUCCESS


# ---------------------------------------------------------------------------
# 失败与边界
# ---------------------------------------------------------------------------
def test_provider_failure_fails_early(registry, tmp_path):
    class BrokenProvider:
        name = "broken"

        def describe(self):
            return "坏数据源"

        def as_of(self):
            raise RuntimeError("数据源不可达")

    runtime = AgentRuntime(
        registry, RulePlanner(), AgentConfig(), provider=BrokenProvider(),
        runs_root=tmp_path / "runs", sleep=lambda s: None,
    )
    run = runtime.run(BANK_REQUEST)
    assert run.status is RunStatus.FAILED
    assert any("数据源不可达" in warning for warning in run.warnings)
    assert (tmp_path / "runs" / run.run_id / "report.md").exists()


def test_planner_failure_fails_early(synthetic_provider, registry, tmp_path):
    class BrokenPlanner:
        name = "broken"

        def plan(self, request, catalog):
            raise RuntimeError("规划器内部错误")

    runtime = AgentRuntime(
        registry, BrokenPlanner(), AgentConfig(), provider=synthetic_provider,
        runs_root=tmp_path / "runs", sleep=lambda s: None,
    )
    run = runtime.run(BANK_REQUEST)
    assert run.status is RunStatus.FAILED
    assert any("规划失败" in warning for warning in run.warnings)


def test_empty_plan_ends_as_failed_without_findings(synthetic_provider, registry, tmp_path):
    class EmptyPlanner:
        name = "empty"

        def plan(self, request, catalog):
            return Plan(request=request.text, steps=[], source="empty")

    runtime = AgentRuntime(
        registry, EmptyPlanner(), AgentConfig(), provider=synthetic_provider,
        runs_root=tmp_path / "runs", sleep=lambda s: None,
    )
    run = runtime.run(BANK_REQUEST)
    assert run.status is RunStatus.FAILED
    assert run.tasks == []


def test_explicit_universe_symbols_are_honoured(synthetic_provider, registry, tmp_path, bank_symbols):
    runtime = make_runtime(
        synthetic_provider, registry, tmp_path, max_replan_rounds=0
    )
    run = runtime.run(
        "分析这批股票最近一年的低估值因子",
        universe_symbols=bank_symbols[:20],
        pool="显式指定",
    )
    primary = run.findings["primary"]
    assert primary["n_symbols"] == 20


def test_unknown_factors_are_reported_not_invented(synthetic_provider, registry, tmp_path):
    runtime = make_runtime(synthetic_provider, registry, tmp_path)
    run = runtime.run("分析银行股", factors=("不存在的因子",))
    assert run.status is RunStatus.FAILED
    failed = [task for task in run.tasks if task.status is TaskStatus.FAILED]
    assert any("未知因子" in task.error for task in failed)


# ---------------------------------------------------------------------------
# 可重复性
# ---------------------------------------------------------------------------
def test_same_request_is_reproducible(synthetic_provider, registry, tmp_path):
    """同种子 + 同请求 → 同结论。这是「演示可复现」的前提。"""
    first = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    second = make_runtime(synthetic_provider, registry, tmp_path).run(BANK_REQUEST)
    assert first.findings["primary"]["ic"]["ic_mean"] == pytest.approx(
        second.findings["primary"]["ic"]["ic_mean"]
    )
    assert first.findings["primary"]["metrics"]["年化收益"] == pytest.approx(
        second.findings["primary"]["metrics"]["年化收益"]
    )
    assert len(first.tasks) == len(second.tasks)


def test_job_cache_makes_second_experiment_a_cache_hit(
    synthetic_provider, registry, tmp_path
):
    """接入 JobService 后，同一实验第二次运行应命中内容寻址缓存。"""
    from qfm.jobs import JobService

    jobs = JobService(tmp_path / "jobs")
    runtime = AgentRuntime(
        registry, RulePlanner(), AgentConfig(retry=RetryPolicy(max_attempts=1)),
        provider=synthetic_provider, runs_root=tmp_path / "runs",
        jobs=jobs, sleep=lambda s: None,
    )
    first = runtime.run(BANK_REQUEST)
    assert not any(task.cache_hit for task in first.tasks)

    second = runtime.run(BANK_REQUEST)
    experiment_tasks = [task for task in second.tasks if task.tool == "run_experiment"]
    assert experiment_tasks, "第二次运行仍应有实验步骤"
    assert any(task.cache_hit for task in experiment_tasks), "应命中缓存"
