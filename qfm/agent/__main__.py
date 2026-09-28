"""命令行入口：`python -m qfm.agent "研究请求"`。

CLI 是这个 Demo 的门面 —— 面试官 clone 后的第一条命令就是它。所以它必须：

1. **零配置可跑。** 没有 `data_cache/` 时自动回落到离线合成面板，并在输出里
   显著标注「合成数据」。不因为缺数据就报错退出。
2. **过程可见。** 默认打印计划、每一步的状态/耗时/重试/缓存命中，让「Agent
   到底做了什么」不需要读代码就能看懂。
3. **可演示边界。** `--read-only`、`--deny`、`--max-calls` 这些开关让权限门
   与失败处理可以被当场验证，而不是只在文档里声称有。

用法::

    python -m qfm.agent "分析最近一年银行股低估值因子的表现"
    python -m qfm.agent "分析最近一年白酒股动量因子" --read-only
    python -m qfm.agent "对比最近三年动量与反转因子" --data cache --json
    python -m qfm.agent "分析最近一年银行股低估值因子" --deny generate_report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qfm.agent.llm import build_llm_client
from qfm.agent.permissions import PermissionPolicy
from qfm.agent.planner import LLMPlanner, RulePlanner
from qfm.agent.providers import resolve_provider
from qfm.agent.retry import RetryPolicy
from qfm.agent.runtime import AgentConfig, AgentRuntime
from qfm.agent.synthetic import SyntheticSpec
from qfm.agent.tools import build_default_registry

EXAMPLES = (
    "分析最近一年银行股低估值因子的表现",
    "最近三年动量因子在全市场的表现",
    "对比最近两年白酒股的 roe 和 gross_margin 因子",
)

_ICON = {
    "SUCCESS": "✅", "FAILED": "❌", "DENIED": "🚫",
    "SKIPPED": "⏭", "RETRYING": "🔁", "PENDING": "·", "RUNNING": "▶",
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.request:
        parser.print_help()
        print("\n示例：")
        for example in EXAMPLES:
            print(f'  python -m qfm.agent "{example}"')
        return 0

    request = " ".join(args.request)
    provider = _build_provider(args)
    planner, planner_note = _build_planner(args)
    policy = _build_policy(args)
    config = AgentConfig(
        retry=RetryPolicy(
            max_attempts=args.max_attempts,
            base_delay=args.retry_delay,
        ),
        permissions=policy,
        max_replan_rounds=args.max_replan,
    )

    runtime = AgentRuntime(
        build_default_registry(),
        planner,
        config,
        provider=provider,
        runs_root=args.runs_root,
        sleep=lambda seconds: None if args.no_wait else _sleep(seconds),
        on_event=None if args.quiet else _make_printer(args),
    )

    print(f"▶ 研究请求：{request}")
    print(f"▶ 数据源：{getattr(provider, 'describe', lambda: provider.name)()}")
    print(f"▶ 规划器：{planner_note}")
    print(f"▶ 权限：{_describe_policy(policy)}")
    print()

    run = runtime.run(
        request,
        pool=args.pool,
        factors=tuple(args.factor or ()),
        max_stocks=args.max_stocks,
        horizon=args.horizon,
        top_n=args.top_n,
    )

    _print_summary(run, runs_root=args.runs_root, verbose=not args.quiet)
    if args.json:
        payload = run.to_dict()
        payload["findings"] = run.findings
        target = Path(args.json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n▶ 运行记录 JSON：{target}")
    return 0 if run.status.value == "SUCCESS" else 1


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qfm.agent",
        description="Quant Research Agent：把一句话研究请求跑成完整研究流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("request", nargs="*", help="自然语言研究请求")
    parser.add_argument("--data", default="auto", choices=("auto", "cache", "synthetic"),
                        help="数据源：auto 优先本地缓存、缺失则用合成面板（默认）")
    parser.add_argument("--planner", default="auto", choices=("auto", "rule", "llm"),
                        help="规划器：auto 有 LLM 配置则用 LLM、否则规则规划（默认）")
    parser.add_argument("--pool", default=None, help="显式指定股票池，跳过关键词解析")
    parser.add_argument("--factor", action="append", help="显式指定因子（可重复）")
    parser.add_argument("--max-stocks", type=int, default=None, help="限制股票数（加速调试）")
    parser.add_argument("--horizon", type=int, default=20, help="IC 前瞻交易日数（默认 20）")
    parser.add_argument("--top-n", type=int, default=30, help="组合持仓数（默认 30）")

    group = parser.add_argument_group("权限与预算")
    group.add_argument("--read-only", action="store_true",
                       help="只读模式：禁掉一切写盘工具（generate_report）")
    group.add_argument("--deny", action="append", default=[],
                       help="禁用指定工具（可重复），用于演示 Agent 如何适应")
    group.add_argument("--allow", action="append", default=None,
                       help="白名单：只允许这些工具（可重复）")
    group.add_argument("--max-calls", type=int, default=40, help="总调用次数上限（默认 40）")
    group.add_argument("--max-cost", type=float, default=200.0, help="累计成本上限（默认 200）")

    group = parser.add_argument_group("重试")
    group.add_argument("--max-attempts", type=int, default=3, help="单步最多尝试次数（默认 3）")
    group.add_argument("--retry-delay", type=float, default=0.5, help="首次退避秒数（默认 0.5）")
    group.add_argument("--no-wait", action="store_true", help="不真实等待退避（演示用）")
    group.add_argument("--max-replan", type=int, default=1, help="异常驱动的补救轮数上限（默认 1）")

    group = parser.add_argument_group("输出")
    group.add_argument("--runs-root", default="reports/agent_runs", help="运行记录目录")
    group.add_argument("--json", default=None, help="把运行记录写到指定 JSON 文件")
    group.add_argument("--quiet", action="store_true", help="只打印最终摘要")
    return parser


def _build_provider(args):
    spec = SyntheticSpec()
    return resolve_provider(args.data, cache_dir="data_cache", spec=spec)


def _build_planner(args) -> tuple[object, str]:
    rule = RulePlanner()
    if args.planner == "rule":
        return rule, "规则规划器（离线、确定性）"
    client = build_llm_client()
    if args.planner == "llm":
        if not client.available:
            print("⚠ 指定了 --planner llm 但没有配置 QFM_AGENT_LLM_* 环境变量，"
                  "已回落到规则规划器", file=sys.stderr)
            return rule, "规则规划器（LLM 未配置，已回落）"
        return LLMPlanner(client, fallback=rule), f"LLM 规划器（{client.config.model}）"
    if client.available:
        return LLMPlanner(client, fallback=rule), (
            f"LLM 规划器（{client.config.model}，失败自动回落规则规划）"
        )
    return rule, "规则规划器（未配置 QFM_AGENT_LLM_*，默认离线确定性规划）"


def _build_policy(args) -> PermissionPolicy:
    allowed = frozenset(args.allow) if args.allow else None
    return PermissionPolicy(
        allowed=allowed,
        denied=frozenset(args.deny),
        read_only=bool(args.read_only),
        max_total_calls=int(args.max_calls),
        max_cost=float(args.max_cost),
    )


def _describe_policy(policy: PermissionPolicy) -> str:
    bits = []
    if policy.read_only:
        bits.append("只读")
    if policy.denied:
        bits.append(f"禁用 {sorted(policy.denied)}")
    if policy.allowed is not None:
        bits.append(f"白名单 {len(policy.allowed)} 个工具")
    bits.append(f"调用上限 {policy.max_total_calls}")
    bits.append(f"成本上限 {policy.max_cost:g}")
    return "，".join(bits)


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def _make_printer(args):
    show_plan = not args.quiet

    def printer(event: str, payload: dict) -> None:
        if event == "planned":
            if show_plan:
                print(f"【规划】{payload.get('source')}，共 {payload.get('steps')} 步")
                for note in payload.get("notes") or []:
                    print(f"        · {note}")
        elif event == "step_started":
            print(f"  {payload.get('index'):>2}. ▶ {payload.get('tool'):<18} "
                  f"{payload.get('goal') or ''}")
        elif event == "step_finished":
            icon = _ICON.get(payload.get("status", ""), "•")
            extra = []
            if (payload.get("attempts") or 1) > 1:
                extra.append(f"尝试 {payload['attempts']} 次")
            if payload.get("cache_hit"):
                extra.append("命中缓存")
            if payload.get("error"):
                extra.append(str(payload["error"])[:120])
            suffix = ("　[" + "，".join(extra) + "]") if extra else ""
            print(f"      {icon} {payload.get('status')}　"
                  f"{payload.get('duration_s', 0):.2f}s　"
                  f"{(payload.get('summary') or '')[:110]}{suffix}")
        elif event == "step_retrying":
            print(f"      🔁 第 {payload.get('attempt')} 次失败（{payload.get('classification')}）："
                  f"{str(payload.get('error'))[:100]}　{payload.get('delay_s')}s 后重试")
        elif event == "step_denied":
            print(f"      🚫 被权限门拒绝（{payload.get('rule')}）：{payload.get('reason')}")
        elif event == "replanned":
            print(f"\n【重新规划】第 {payload.get('round')} 轮，{payload.get('steps')} 步")
            print(f"        {payload.get('reason')}")

    return printer


def _print_summary(run, runs_root: str = "reports/agent_runs", verbose: bool = True) -> None:
    print()
    print("=" * 78)
    print(f"运行状态：{run.status.value}　任务：{run.counts()}　"
          f"工具耗时合计 {run.duration_s:.2f}s")
    if run.warnings:
        print("\n告警：")
        for warning in run.warnings:
            print(f"  ⚠ {warning}")

    primary = run.findings.get("primary")
    control = run.findings.get("control")
    if primary:
        print()
        _print_experiment("主实验", primary)
    if control:
        _print_experiment("宽基对照", control)
        p_ic = (primary or {}).get("ic", {}).get("ic_mean") if primary else None
        c_ic = control.get("ic", {}).get("ic_mean")
        if isinstance(p_ic, float) and isinstance(c_ic, float):
            same = p_ic * c_ic > 0
            print(f"    对照结论：IC 方向{'一致，窄池结论非样本宽度假象' if same else '相反，窄池结论受行业构成驱动'}")

    anomalies = run.findings.get("anomalies") or []
    if anomalies:
        print(f"\n异常检查（{len(anomalies)} 项）：")
        for item in anomalies:
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}.get(item.get("severity"), "•")
            print(f"  {icon} [{item.get('code')}] {str(item.get('message'))[:130]}")

    report = run.findings.get("research_report")
    if report:
        print(f"\n研究报告：{report}")
    print(f"执行记录：{Path(runs_root) / run.run_id}")
    if verbose:
        stats = run.context_stats or {}
        print(f"上下文：{stats.get('items')} 条 / {stats.get('chars')} 字符"
              f"（外置 {stats.get('offloaded_items')} 条，折叠 {stats.get('folded_items')} 条）")
    print("=" * 78)


def _print_experiment(label: str, experiment: dict) -> None:
    ic = experiment.get("ic") or {}
    metrics = experiment.get("metrics") or {}

    def fmt(value, pattern="{:.4f}"):
        return pattern.format(value) if isinstance(value, (int, float)) else "—"

    print(f"【{label}】{experiment.get('label')}（{experiment.get('name')}）"
          f" @ {experiment.get('pool')}（{experiment.get('n_symbols')} 只）")
    window = experiment.get("window") or {}
    print(f"    区间 {window.get('start')} ~ {window.get('end')}"
          f"（{window.get('trading_days')} 个交易日）")
    print(f"    IC {fmt(ic.get('ic_mean'), '{:+.4f}')}　"
          f"t {fmt(ic.get('ic_t'), '{:+.2f}')}　"
          f"IC>0 占比 {fmt(ic.get('pos_ratio'), '{:.1%}')}　"
          f"ICIR {fmt(ic.get('ic_ir'))}")
    print(f"    年化 {fmt(metrics.get('年化收益'), '{:+.2%}')}　"
          f"超额 {fmt(metrics.get('年化超额'), '{:+.2%}')}　"
          f"夏普 {fmt(metrics.get('夏普比率'))}　"
          f"最大回撤 {fmt(metrics.get('最大回撤'), '{:.2%}')}　"
          f"换手 {fmt(experiment.get('turnover'), '{:.1%}')}")
    print(f"    分层检验：{'可用' if experiment.get('layer_available') else '不可用（截面不足 100 只）'}")


def _sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


if __name__ == "__main__":
    raise SystemExit(main())
