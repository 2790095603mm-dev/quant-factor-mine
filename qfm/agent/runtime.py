"""Agent 运行时：计划 → 执行 → 观察 → 重试 → 重新规划 → 报告。

## 主循环

对计划里的每一步，依次穿过三道闸门，然后才真正调用工具：

```
第 1 道：权限门      PermissionGate.check()   —— 不允许就记 DENIED，不执行
第 2 道：重试层      call_with_retry()        —— 瞬时故障退避重试，确定性错误直接失败
第 3 道：结果校验    工具自身的后置校验        —— 覆盖率不达标也当作瞬时故障重试
```

三道闸门的位置有讲究：**权限必须在重试之外**。否则一个被拒绝的调用会被重试三次，
既浪费预算，又会把「这是权限问题而不是故障」这个事实淹没在重试日志里。

第 3 道没有做成独立的钩子，而是让工具在自己的实现里抛
`DegradedResultError`（`TransientError` 的子类）。理由是**只有工具自己知道
什么算结果不合格**：`load_panel` 关心覆盖率，`compute_factor` 关心因子非空比例，
`run_experiment` 关心有效交易日数。把这件事交给运行时的通用规则判断，就只能
猜到最粗的那一层。

## 异常驱动的重新规划

`check_anomalies` 是主循环的转折点。当它报出 high 级别异常时，运行时会**补一段
计划并继续执行**，而不是把异常写进报告就完事：

- `narrow_cross_section` → 换宽基池（沪深300+中证500）重跑同一因子作为对照；
- 对照结果与主结果一起进报告，于是「42 只银行股的结论有多可信」变成了一个
  **有对照样本的判断**，而不是一句免责声明。

这是整个 Agent 里唯一一处「自主决定下一步做什么」的地方，所以它被刻意做窄：
只对白名单里的异常代码生效（`REMEDIATION_CODES`）、最多补一轮
（`max_replan_rounds`）、补齐的计划走同一套权限与重试闸门、且必须在报告里留下
「为什么补这一步」的记录。

## 运行态不放在 self 上

上下文、工具工作集、权限门、任务清单都装在 `_Session` 里显式传参，而不是塞进
`self`。这样同一个 runtime 实例可以并发跑多次研究而不会互相污染，测试也能
单独构造 session 做断言。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from qfm.agent.context import ContextBudget, ContextManager
from qfm.agent.errors import AgentError
from qfm.agent.models import (
    AgentRun,
    Plan,
    PlanStep,
    RunStatus,
    TaskRecord,
    TaskStatus,
    new_id,
    utc_now,
)
from qfm.agent.permissions import PermissionGate, PermissionPolicy
from qfm.agent.planner import DEFAULT_POOL, PlanRequest
from qfm.agent.registry import ToolContext, ToolRegistry
from qfm.agent.reporting import render_run_markdown
from qfm.agent.retry import RetryPolicy, call_with_retry

__all__ = ["AgentConfig", "AgentRuntime", "Session", "REMEDIATION_CODES"]

#: 触发「补一段计划」的异常代码。刻意保持极窄：只有能通过换股票池解决的
#: 样本宽度问题才自动补救，其余异常一律只报告、不擅自改研究设计。
REMEDIATION_CODES = frozenset({"narrow_cross_section", "layer_test_unavailable"})


@dataclass
class AgentConfig:
    """运行配置。"""

    retry: RetryPolicy = field(default_factory=RetryPolicy)
    context_budget: ContextBudget = field(default_factory=ContextBudget)
    permissions: PermissionPolicy = field(default_factory=PermissionPolicy)
    max_replan_rounds: int = 1

    def __post_init__(self) -> None:
        if self.max_replan_rounds < 0:
            raise ValueError("max_replan_rounds 不能为负")


@dataclass
class Session:
    """一次运行的内部状态。所有执行方法都显式接收它，不依赖 `self`。"""

    run: AgentRun
    workdir: Path
    ctx: ToolContext
    context: ContextManager
    gate: PermissionGate
    aborted: bool = False
    #: 主实验（第一个成功跑完的 run_experiment）。
    #: 必须单独记：一个请求可能对多个因子各跑一次实验，
    #: 而"最后一个"和"用户主要问的那个"不是一回事，补救规划要针对后者。
    primary_experiment: dict | None = None

    def emit(self, hook: Callable[[str, dict[str, Any]], None] | None,
             event: str, payload: dict[str, Any]) -> None:
        if hook is None:
            return
        try:
            hook(event, payload)
        except Exception:  # noqa: BLE001 - 展示层异常不得影响研究执行
            pass


class AgentRuntime:
    """执行一次研究请求。

    Args:
        registry: 工具注册表。
        planner: 规划器。
        config: 运行配置。
        provider: 数据源（提供 `as_of()` / `load()` / `describe()`）。
        runs_root: 运行记录落盘目录，每次运行写一个子目录。
        store: 可选的研究台账（`ResearchStore`），供台账类工具使用。
        jobs: 可选的作业服务（`JobService`），提供实验级内容寻址缓存。
        catalog_root: 数据集目录根，用于数据集版本绑定。
        sleep / clock: 注入点，测试时替换掉真实休眠与时钟。
        on_event: 事件回调，用于页面实时展示。
        approver: 需要确认的工具的审批回调；缺省时此类工具一律拒绝。
    """

    def __init__(
        self,
        registry: ToolRegistry,
        planner,
        config: AgentConfig | None = None,
        *,
        provider,
        runs_root: Path | str = "reports/agent_runs",
        store: Any = None,
        jobs: Any = None,
        catalog_root: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        approver: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner
        self.config = config or AgentConfig()
        self.provider = provider
        self.runs_root = Path(runs_root)
        self.store = store
        self.jobs = jobs
        self.catalog_root = catalog_root
        self.sleep = sleep
        self.clock = clock
        self.on_event = on_event
        self.approver = approver

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def run(
        self,
        request: str | PlanRequest,
        *,
        universe_symbols: list[str] | None = None,
        pool: str | None = None,
        factors: tuple[str, ...] = (),
        max_stocks: int | None = None,
        horizon: int = 20,
        top_n: int = 30,
    ) -> AgentRun:
        """执行一次研究请求，返回完整运行记录（本方法不抛异常）。"""
        run_id = new_id("agrun")
        workdir = self.runs_root / run_id
        workdir.mkdir(parents=True, exist_ok=True)

        text = request.text if isinstance(request, PlanRequest) else str(request)
        run = AgentRun(run_id=run_id, request=text, created_at=utc_now())
        run.planner = getattr(self.planner, "name", "unknown")

        session = Session(
            run=run,
            workdir=workdir,
            ctx=ToolContext(
                workdir=workdir,
                data_dir=Path("data_cache"),
                provider=self.provider,
                store=self.store,
                jobs=self.jobs,
                catalog_root=self.catalog_root,
            ),
            context=ContextManager(self.config.context_budget, workdir / "artifacts"),
            gate=PermissionGate(self.config.permissions, approver=self.approver),
        )

        try:
            as_of = self.provider.as_of()
        except Exception as exc:  # noqa: BLE001 - 数据源不可用是运行级失败
            return self._fail_early(session, f"无法确定数据可用区间：{exc}")

        run.data_source = getattr(self.provider, "name", "unknown")
        run.data_note = _safe_describe(self.provider) or run.data_source

        plan_request = PlanRequest(
            text=text,
            as_of=as_of,
            pool=pool,
            factors=tuple(factors),
            universe_symbols=tuple(universe_symbols or ()),
            max_stocks=max_stocks,
            horizon=horizon,
            top_n=top_n,
        )
        if universe_symbols:
            session.ctx.put("preset_universe", {
                "pool": pool or "explicit",
                "symbols": [str(code) for code in universe_symbols],
                "as_of": as_of,
            })

        session.context.add("user", "研究请求", text, pinned=True)
        session.context.add(
            "system", "数据来源",
            f"{run.data_source}；可用区间截至 {as_of}",
            pinned=True,
        )
        self._emit("run_started", {"run_id": run_id, "request": text, "as_of": as_of})

        try:
            plan = self.planner.plan(plan_request, self.registry.catalog())
        except Exception as exc:  # noqa: BLE001 - 规划失败无法继续
            return self._fail_early(session, f"规划失败：{type(exc).__name__}: {exc}")

        run.plans.append(plan)
        session.context.add("plan", "执行计划", _plan_text(plan), pinned=True)
        self._emit("planned", {
            "steps": len(plan.steps), "source": plan.source, "notes": list(plan.notes),
        })

        self._loop(session, plan)
        self._finalize(session)
        return run

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _loop(self, session: Session, plan: Plan) -> None:
        index = 0
        round_index = 0
        handled: set[str] = set()

        while True:
            for step in plan.steps:
                index += 1
                task = self._execute_step(session, step, index)
                session.run.tasks.append(task)

                if task.status in (TaskStatus.FAILED, TaskStatus.DENIED, TaskStatus.SKIPPED):
                    if step.on_failure == "abort":
                        session.run.warnings += (
                            f"步骤 {index}（{step.tool}）失败或不可用，"
                            "且该步骤是后续步骤的前置依赖，计划已中止",
                        )
                        session.aborted = True
                        break

            if session.aborted or round_index >= self.config.max_replan_rounds:
                return
            remediation = self._remediation_plan(session, plan, round_index, handled)
            if remediation is None:
                return
            handled |= REMEDIATION_CODES
            round_index += 1
            plan = remediation
            session.run.plans.append(plan)
            session.context.add("plan", f"补充计划（第 {round_index + 1} 轮）", _plan_text(plan))
            self._emit("replanned", {
                "round": round_index + 1,
                "steps": len(plan.steps),
                "reason": plan.notes[0] if plan.notes else "",
            })

    # ------------------------------------------------------------------
    # 单步执行
    # ------------------------------------------------------------------
    def _execute_step(self, session: Session, step: PlanStep, index: int) -> TaskRecord:
        task = TaskRecord(
            task_id=new_id("task"),
            index=index,
            goal=step.goal,
            tool=step.tool,
            arguments=dict(step.arguments),
            status=TaskStatus.PENDING,
            started_at=utc_now(),
        )
        self._emit("step_started", {"index": index, "tool": step.tool, "goal": step.goal})

        # ---- 第 1 道闸门：权限 ----
        try:
            tool = self.registry.get(step.tool)
        except AgentError as exc:
            return self._finish_task(task, TaskStatus.FAILED, error=str(exc))

        decision = session.gate.check(tool.spec, step.arguments)
        if not decision.allowed:
            session.context.add("observation", f"{index}. {step.tool} 被拒绝", decision.reason)
            self._emit("step_denied", {
                "index": index, "tool": step.tool,
                "reason": decision.reason, "rule": decision.rule,
            })
            return self._finish_task(task, TaskStatus.DENIED, error=decision.reason)

        # ---- 第 2 道闸门：重试（工具内部做第 3 道：结果校验）----
        task.status = TaskStatus.RUNNING
        policy = self.config.retry if tool.spec.retryable else RetryPolicy(max_attempts=1)
        started_clock = self.clock()

        def attempt_fn():
            return self.registry.call(step.tool, step.arguments, session.ctx, clock=self.clock)

        def on_attempt(attempt) -> None:
            task.retry_log = task.retry_log + (attempt.to_dict(),)
            session.gate.record(tool.spec)
            self._emit("step_retrying", {
                "index": index, "tool": step.tool, "attempt": attempt.index,
                "error_type": attempt.error_type, "error": attempt.error,
                "classification": attempt.classification,
                "delay_s": round(attempt.delay_s, 3),
            })

        outcome = call_with_retry(
            attempt_fn, policy, sleep=self.sleep, on_attempt=on_attempt,
        )
        session.gate.record(tool.spec)  # 成功的那一次也要计入预算
        task.duration_s = self.clock() - started_clock
        task.attempts = outcome.attempt_count

        if outcome.ok:
            payload, summary, _ = outcome.value
            task.result_summary = summary
            task.cache_hit = bool(isinstance(payload, dict) and payload.get("cache_hit"))
            if step.tool == "run_experiment" and session.primary_experiment is None:
                session.primary_experiment = payload
                # 告诉异常检查「哪个实验是用户主要问的」——一次请求可能跑多个因子，
                # 主结论只应挂在第一个（也就是请求里排最前的那个）因子上。
                session.ctx.put("primary_factor_key", payload.get("key"))
            session.context.add(
                "tool",
                f"{index}. {step.tool}",
                summary,
                artifact_payload=payload if _is_heavy(payload) else None,
                offload_name=f"{index:02d}_{step.tool}.json",
            )
            session.context.compact()
            self._emit("step_finished", {
                "index": index, "tool": step.tool, "status": "SUCCESS",
                "summary": summary, "attempts": task.attempts,
                "duration_s": round(task.duration_s, 3), "cache_hit": task.cache_hit,
            })
            return self._finish_task(task, TaskStatus.SUCCESS)

        session.context.add(
            "observation",
            f"{index}. {step.tool} 失败",
            f"{outcome.error_type}: {outcome.error_message}（尝试 {task.attempts} 次）",
        )
        self._emit("step_finished", {
            "index": index, "tool": step.tool, "status": "FAILED",
            "error": outcome.error_message, "attempts": task.attempts,
            "duration_s": round(task.duration_s, 3),
        })
        return self._finish_task(task, TaskStatus.FAILED, error=outcome.error_message)

    def _finish_task(self, task: TaskRecord, status: TaskStatus, *, error: str = "") -> TaskRecord:
        task.status = status
        task.error = error
        task.finished_at = utc_now()
        return task

    # ------------------------------------------------------------------
    # 重新规划
    # ------------------------------------------------------------------
    def _remediation_plan(
        self,
        session: Session,
        current: Plan,
        round_index: int,
        handled: set[str],
    ) -> Plan | None:
        """根据异常检查结果决定是否补一段计划。"""
        anomalies = session.ctx.facts.get("anomalies") or []
        codes = {item.get("code") for item in anomalies}
        actionable = codes & REMEDIATION_CODES - handled
        if not actionable:
            return None

        experiment = session.primary_experiment or session.ctx.facts.get("experiment")
        if not experiment:
            return None
        n_symbols = int(experiment.get("n_symbols") or 0)
        if n_symbols >= 100:
            return None  # 截面已经够宽，异常另有原因，不擅自改研究设计

        factor = experiment.get("name")
        window = experiment.get("window") or {}

        # 把主实验固化成「主结果」，后续结果作为对照，避免互相覆盖。
        session.ctx.facts["primary_experiment"] = experiment
        session.ctx.facts["primary_anomalies"] = anomalies
        session.ctx.put("primary_experiment_key", experiment.get("key"))
        session.ctx.put("primary_universe", session.ctx.panel_meta.get("pool"))

        reason = (
            f"异常检查报出 {sorted(actionable)}：主实验仅 {n_symbols} 只标的，截面宽度不足。"
            f"补跑 {DEFAULT_POOL} 宽基对照，用于判断该因子在足够宽的截面上是否依然成立。"
        )
        steps = [
            PlanStep(
                tool="resolve_universe",
                arguments={"pool": DEFAULT_POOL},
                goal=f"切换到宽基股票池 {DEFAULT_POOL} 作为对照",
                on_failure="abort",
            ),
            PlanStep(
                tool="load_panel",
                arguments={
                    "start": window.get("start"),
                    "end": window.get("end"),
                },
                goal="加载宽基面板（窗口与主实验一致）",
                on_failure="abort",
            ),
            PlanStep(
                tool="compute_factor",
                arguments={"name": factor},
                goal=f"在宽基池上计算 {factor}",
                on_failure="abort",
            ),
            PlanStep(
                tool="run_experiment",
                arguments={
                    "name": factor,
                    "start": window.get("start"),
                    "end": window.get("end"),
                    "horizon": experiment.get("horizon", 20),
                    "top_n": experiment.get("top_n_requested", 30),
                    "mode": experiment.get("mode", "long_only"),
                    "rebalance": experiment.get("rebalance", "ME"),
                    "neutralization": experiment.get("neutralization", "none"),
                    "decay": experiment.get("decay", 0),
                },
                goal="在宽基池上重跑同一配置，得到有分层统计的对照结果",
                on_failure="continue",
            ),
            PlanStep(
                tool="check_anomalies",
                arguments={},
                goal="对对照实验再做一次异常检查",
                on_failure="continue",
            ),
            PlanStep(
                tool="generate_report",
                arguments={"title": f"{factor} 因子：窄池主实验与宽基对照"},
                goal="用主实验结果与对照结果一起重写报告",
                on_failure="continue",
            ),
        ]
        return Plan(
            request=current.request,
            steps=steps,
            source="remediation",
            notes=(reason, "补救范围仅限换股票池：不改变区间、因子定义与组合配置"),
            round=round_index + 1,
        )

    # ------------------------------------------------------------------
    # 收尾
    # ------------------------------------------------------------------
    def _finalize(self, session: Session) -> None:
        run = session.run
        run.finished_at = utc_now()
        run.artifacts = tuple(sorted(set(session.ctx.artifacts)))
        run.context_stats = session.context.compact().to_dict()
        run.warnings = tuple(dict.fromkeys(run.warnings))
        run.permission_events = tuple(dict(event) for event in session.gate.events)

        experiment = session.ctx.facts.get("experiment")
        primary = session.ctx.facts.get("primary_experiment") or session.primary_experiment or experiment
        # 只有对照轮真的跑过（facts 里存在 primary_experiment）才算有对照
        control = experiment if session.ctx.facts.get("primary_experiment") else None
        run.findings = {
            "primary": primary,
            "control": control,
            "primary_universe": session.ctx.get("primary_universe"),
            "anomalies": session.ctx.facts.get("primary_anomalies")
            or session.ctx.facts.get("anomalies") or [],
            "control_anomalies": _control_anomalies(session, control),
            "anomalies_by_experiment": session.ctx.facts.get("anomalies_by_experiment") or {},
            "panel": session.ctx.panel_meta,
            "compare": session.ctx.facts.get("compare"),
            "research_report": session.ctx.get("report_path", ""),
            "context": session.context.stats.to_dict(),
        }

        if primary is None:
            run.status = RunStatus.FAILED
            run.warnings += ("本次运行没有产出任何实验结果",)
        elif session.aborted:
            run.status = RunStatus.FAILED
        elif any(task.status in (TaskStatus.FAILED, TaskStatus.DENIED) for task in run.tasks):
            run.status = RunStatus.PARTIAL
        else:
            run.status = RunStatus.SUCCESS

        for item in run.findings["anomalies"]:
            if item.get("severity") == "high":
                run.warnings += (f"高优异常 {item.get('code')}：{item.get('message')}",)
        run.warnings = tuple(dict.fromkeys(run.warnings))

        run.report_markdown = render_run_markdown(run)
        self._persist(session)

    def _persist(self, session: Session) -> None:
        run = session.run
        try:
            (session.workdir / "report.md").write_text(run.report_markdown, encoding="utf-8")
            (session.workdir / "run.json").write_text(
                json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (session.workdir / "context_digest.txt").write_text(
                session.context.digest(limit=40), encoding="utf-8"
            )
            extra = tuple(
                item.artifact for item in session.context.items
                if item.artifact and item.artifact not in run.artifacts
            )
            run.artifacts = tuple(sorted(set(run.artifacts) | set(extra)))
        except OSError as exc:  # noqa: BLE001 - 落盘失败不应让运行结果丢失
            run.warnings += (f"运行记录落盘失败：{exc}",)

        self._emit("run_finished", {
            "run_id": run.run_id,
            "status": run.status.value,
            "counts": run.counts(),
            "duration_s": round(run.duration_s, 3),
        })

    def _fail_early(self, session: Session, reason: str) -> AgentRun:
        run = session.run
        run.status = RunStatus.FAILED
        run.finished_at = utc_now()
        run.warnings += (reason,)
        run.report_markdown = render_run_markdown(run)
        try:
            (session.workdir / "report.md").write_text(run.report_markdown, encoding="utf-8")
            (session.workdir / "run.json").write_text(
                json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
        self._emit("run_finished", {
            "run_id": run.run_id, "status": run.status.value, "error": reason,
        })
        return run

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event, payload)
        except Exception:  # noqa: BLE001 - 展示层异常不得影响研究执行
            pass


def _control_anomalies(session: Session, control) -> list:
    """取对照实验自己的异常，而不是主实验的。"""
    if not control:
        return []
    by_experiment = session.ctx.facts.get("anomalies_by_experiment") or {}
    entry = by_experiment.get(control.get("key"))
    return list(entry.get("anomalies") or []) if entry else []


def _plan_text(plan: Plan) -> str:
    lines = [f"来源：{plan.source}"]
    for note in plan.notes:
        lines.append(f"- {note}")
    lines.append("")
    for index, step in enumerate(plan.steps, start=1):
        args = ", ".join(f"{key}={value}" for key, value in step.arguments.items())
        flags = []
        if step.optional:
            flags.append("可选")
        if step.on_failure != "abort":
            flags.append(f"失败时{step.on_failure}")
        suffix = f" [{'/'.join(flags)}]" if flags else ""
        lines.append(f"{index}. {step.tool}({args}){suffix}")
    return "\n".join(lines)


def _is_heavy(payload: Any) -> bool:
    """判断工具返回值是否需要外置到 artifact。

    字典里若含 DataFrame 或超长列表，说明它不适合整体进上下文；
    工具给的摘要已覆盖判断下一步所需的信息，完整结构落到 artifact 供事后复盘。
    """
    if not isinstance(payload, dict):
        return False
    for value in payload.values():
        if type(value).__name__ == "DataFrame":
            return True
        if isinstance(value, (list, tuple)) and len(value) > 50:
            return True
    return False


def _safe_describe(provider) -> str:
    describe = getattr(provider, "describe", None)
    if not callable(describe):
        return ""
    try:
        return str(describe())
    except Exception:  # noqa: BLE001 - 描述信息失败不影响运行
        return ""
