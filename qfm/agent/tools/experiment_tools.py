"""实验类工具：跑单因子实验、检查异常、生成报告。

## 三个设计要点

**① 复用既有的确定性缓存。** `run_experiment` 把请求整理成与既有页面
（`qfm/simulation/views.py`）完全同构的形状 ——
``{**dataset_binding, universe, date_range, factor_versions, pipeline_config, backtest_config}``
—— 然后交给 `JobService.run`。于是 Agent 的实验自动获得：内容寻址缓存、
`cache_hit` 标记、manifest 落盘、参数变化即失效。**没有新造一套缓存。**

**② 自适应约束。** `run_factor_simulation` 在 `top_n * 2 > 股票数` 时直接抛
`ValueError`。42 只银行股的池子里这必然发生。工具不是把参数错误抛给用户，
而是**按股票数收窄 `top_n` 并把这件事记进 payload** —— 研究员需要知道自己看的
不是他要求的配置。

**③ 异常检查复用上游告警而不是重复实现。** 引擎自己已经产出
「分层统计样本不足」「信号覆盖率低」「未产生成交」等告警
（`qfm/simulation/engine.py`）。异常检查把这些告警与本层新增的判据
（IC 方向、显著性、样本宽度、换手侵蚀、超额为负）合并成统一结构，
并给出**可执行的补救建议** —— 这样运行时才能据此重新规划。
"""

from __future__ import annotations

import math

import pandas as pd

from qfm.agent.errors import InvalidArgumentsError, PermanentError
from qfm.agent.models import ToolKind
from qfm.agent.registry import ToolContext

__all__ = ["register_experiment_tools", "ANOMALY_RULES", "remediation_for"]

#: 异常判据表：(代码, 严重级别)。级别 high 会触发运行时重新规划。
ANOMALY_RULES: dict[str, str] = {
    # 截面太窄会让 IC 的统计功效显著下降，且分层检验直接不可用。
    # 定为 high 是有意的：运行时会据此补跑一次宽基对照实验，
    # 而不是把「样本不足」当成结论免责声明糊过去了事。
    "narrow_cross_section": "high",
    "layer_test_unavailable": "low",
    "low_coverage": "medium",
    "insignificant_ic": "medium",
    "ic_direction_conflict": "high",
    "unstable_ic": "medium",
    "high_turnover": "medium",
    "weak_monotonicity": "low",
    "negative_excess": "medium",
    "large_drawdown": "medium",
    "no_trades": "high",
    "engine_warning": "low",
}

#: 异常 → 补救动作。运行时按此决定「重新规划时该加什么步骤」。
_REMEDIATION: dict[str, str] = {
    "narrow_cross_section": "改用更宽的股票池（如全市场或沪深300）重跑，以获得分层统计与更稳健的 IC",
    "layer_test_unavailable": "扩大股票池到每日至少 100 只有效股票",
    "low_coverage": "检查财务字段覆盖；必要时放宽 max_stocks 或改用覆盖更完整的因子",
    "insignificant_ic": "延长样本区间，或换用同主题的其他因子做对照",
    "ic_direction_conflict": "核对因子定义方向与中性化设置，确认是否为风格暴露导致的伪信号",
    "unstable_ic": "做行业中性与市值中性后再评估，或延长区间看稳定性",
    "high_turnover": "加入 Decay 平滑或降低调仓频率以压缩换手成本",
    "weak_monotonicity": "提高分层数或扩大样本，确认是否为排序能力不足",
    "negative_excess": "检查基准选择与持仓集中度，判断是因子失效还是组合构建问题",
    "large_drawdown": "检查区间内的极端行情与持仓集中度",
    "no_trades": "扩大股票池或放宽持仓数上限，确保有足够可成交标的",
    "engine_warning": "阅读上游引擎告警并逐条确认",
}


def remediation_for(code: str) -> str:
    """给出异常代码对应的补救建议。"""
    return _REMEDIATION.get(code, "人工复核该异常")


def register_experiment_tools(registry) -> None:
    """把实验类工具注册进 `registry`。"""

    @registry.register(
        "run_experiment",
        kind=ToolKind.COMPUTE,
        description="运行单因子实验：统一管线处理信号、IC/分层检验、成本后组合回测与绩效归因。结果进内容寻址缓存",
        parameters={
            "name": "str，因子名，如 bp",
            "start": "str，起始日期 YYYY-MM-DD",
            "end": "str，结束日期 YYYY-MM-DD",
            "horizon": "int，IC 前瞻交易日数（默认 20）",
            "top_n": "int，组合持仓数（会按股票池宽度自动收窄）",
            "mode": "str，long_only 或 long_short",
            "rebalance": "str，调仓频率 B/W-FRI/ME",
            "neutralization": "str，信号中性化 none/industry/size/industry_size",
            "decay": "int，信号平滑天数",
            "use_cache": "bool，是否使用作业缓存（默认 true）",
        },
        required=("name", "start", "end"),
        returns="payload: {name, window, ic, metrics, monotonicity, turnover, coverage, layer_available, top_n_effective, warnings, cache_hit, binding}",
        cost=4.0,
        tags=("experiment", "backtest"),
    )
    def run_experiment(
        context: ToolContext,
        name: str,
        start: str,
        end: str,
        horizon: int = 20,
        top_n: int = 30,
        mode: str = "long_only",
        rebalance: str = "ME",
        neutralization: str = "none",
        decay: int = 0,
        use_cache: bool = True,
    ):
        from qfm.simulation.engine import SimulationSettings, run_factor_simulation

        panel = context.panel
        if panel is None or getattr(panel.close, "empty", True):
            raise InvalidArgumentsError("尚未加载面板；请先执行 load_panel")
        factor = _require_factor(name)

        window = _window(panel, start, end, horizon)
        n_symbols = int(panel.close.shape[1])
        effective_top_n, capped_note = _fit_top_n(int(top_n), n_symbols, mode)

        settings = SimulationSettings(
            start=window["start"],
            end=window["end"],
            horizon=int(horizon),
            top_n=effective_top_n,
            mode=mode,
            rebalance=rebalance,
            neutralization=neutralization,
            decay=int(decay),
        )
        settings.validate()

        raw = context.get(f"factor:{name}")
        request = _job_request(context, name, factor, settings, window)
        warnings_extra: list[str] = []
        if capped_note:
            warnings_extra.append(capped_note)

        def compute():
            return run_factor_simulation(panel, name, settings, raw=raw)

        cache_hit = False
        if use_cache and context.jobs is not None:
            try:
                result = context.jobs.run("FACTOR_ANALYSIS", request, compute)
                simulation = result.value
                cache_hit = bool(result.job.cache_hit)
            except Exception as exc:  # noqa: BLE001 - 缓存层故障不应让实验失败
                warnings_extra.append(
                    f"作业缓存不可用（{type(exc).__name__}），本次直接计算"
                )
                simulation = compute()
        else:
            simulation = compute()

        context.put(f"simulation:{name}", simulation)
        report = simulation.report or {}
        ic = dict(report.get("ic_summary") or {})
        mono = dict(report.get("monotonicity") or {})
        layer = report.get("layer")
        layer_available = bool(layer is not None and not layer.empty)

        payload = {
            "name": name,
            "label": _label(name),
            "family": factor.family,
            "direction": factor.direction,
            "window": window,
            "top_n_requested": int(top_n),
            "top_n_effective": effective_top_n,
            "mode": mode,
            "rebalance": rebalance,
            "neutralization": neutralization,
            "decay": int(decay),
            "horizon": int(horizon),
            "n_symbols": n_symbols,
            "pool": context.panel_meta.get("pool"),
            "data_source": context.panel_meta.get("source"),
            "ic": _clean(ic),
            "metrics": _clean(simulation.stats),
            "monotonicity": _clean(mono),
            "layer_available": layer_available,
            "turnover": _finite(report.get("turnover")),
            "coverage": _finite(simulation.coverage),
            "yearly": _yearly(simulation.yearly),
            "warnings": list(simulation.warnings) + warnings_extra,
            "cache_hit": cache_hit,
            "binding": request.get("dataset_version", ""),
        }
        # 一次请求可能对多个因子、多个股票池各跑一次实验，所以既要记「最近的」，
        # 也要按 (因子@股票池) 累积。异常检查必须能精确对应到某一个实验，
        # 否则会出现「用 A 实验的指标配 B 实验的异常」这种报告内部矛盾。
        payload["key"] = _experiment_key(name, payload.get("pool"))
        experiments = dict(context.get("experiments") or {})
        experiments[payload["key"]] = payload
        context.put("experiments", experiments)
        context.put("experiment", payload)
        context.facts["experiment"] = payload
        summary = (
            f"{payload['label']}({name}) 实验完成：IC {_fmt(ic.get('ic_mean'))} "
            f"(t={_fmt(ic.get('ic_t'))})，年化 {_pct(payload['metrics'].get('年化收益'))}，"
            f"夏普 {_fmt(payload['metrics'].get('夏普比率'))}，"
            f"最大回撤 {_pct(payload['metrics'].get('最大回撤'))}"
            + ("，命中缓存" if cache_hit else "")
        )
        return payload, summary

    @registry.register(
        "check_anomalies",
        kind=ToolKind.READ,
        description="对实验结果做异常检查：样本宽度、IC 显著性与方向、换手侵蚀、单调性、超额收益、回撤，并给出可执行的补救建议",
        parameters={
            "name": "str，要检查的因子名；省略则检查最近一次实验",
            "min_abs_ic_t": "float，|t| 显著性下限（默认 2.0）",
            "max_turnover": "float，换手率上限（默认 0.6）",
        },
        required=(),
        returns="payload: {anomalies:[{code,severity,message,evidence,remediation}], severity_counts, needs_replan, layer_available}",
        cost=0.8,
        tags=("quality", "audit"),
    )
    def check_anomalies(
        context: ToolContext,
        name: str | None = None,
        min_abs_ic_t: float = 2.0,
        max_turnover: float = 0.6,
    ):
        experiments = dict(context.get("experiments") or {})
        if not experiments:
            raise InvalidArgumentsError("尚无实验结果；请先执行 run_experiment")

        if name:
            targets = {key: value for key, value in experiments.items() if value.get("name") == name}
            if not targets:
                raise InvalidArgumentsError(
                    f"没有因子 {name!r} 的实验记录；已跑过的因子: "
                    f"{sorted({value.get('name') for value in experiments.values()})}"
                )
        else:
            targets = experiments

        # 主实验 = 本次运行第一个成功跑完的那个（用户主要问的对象）。
        primary_key = context.get("primary_factor_key")
        if primary_key not in targets:
            primary_key = next(iter(targets))

        by_experiment: dict[str, dict] = {}
        for key, experiment in targets.items():
            items = _diagnose(experiment, float(min_abs_ic_t), float(max_turnover))
            counts: dict[str, int] = {}
            for item in items:
                counts[item["severity"]] = counts.get(item["severity"], 0) + 1
            by_experiment[key] = {
                "name": experiment.get("name"),
                "pool": experiment.get("pool"),
                "anomalies": items,
                "severity_counts": counts,
                "needs_replan": any(item["severity"] == "high" for item in items),
            }

        primary = by_experiment[primary_key]
        anomalies = primary["anomalies"]
        context.put("anomalies", anomalies)
        context.put("anomalies_by_experiment", by_experiment)
        context.facts["anomalies"] = anomalies
        context.facts["anomalies_by_experiment"] = by_experiment

        payload = {
            "n_experiments": len(by_experiment),
            "n_anomalies": len(anomalies),
            "primary_key": primary_key,
            "anomalies": anomalies,
            "severity_counts": primary["severity_counts"],
            "needs_replan": primary["needs_replan"],
            "layer_available": (targets[primary_key] or {}).get("layer_available"),
            "by_experiment": by_experiment,
        }
        if not anomalies:
            summary = (
                f"异常检查通过（主实验 {primary_key}）："
                "未发现样本宽度、显著性、换手或单调性问题"
            )
        else:
            top = ", ".join(f"{item['code']}({item['severity']})" for item in anomalies[:4])
            summary = (
                f"主实验 {primary_key} 发现 {len(anomalies)} 项异常：{top}"
                + ("；建议重新规划" if primary["needs_replan"] else "")
                + (f"（另检查了 {len(by_experiment) - 1} 个其他实验）" if len(by_experiment) > 1 else "")
            )
        return payload, summary

    @registry.register(
        "compare_factors",
        kind=ToolKind.COMPUTE,
        description="在相同区间与管线下对比多个因子，输出 IC 对照表与相关性，用于判断因子是否互相重复",
        parameters={
            "names": "list，因子名列表",
            "horizon": "int，IC 前瞻交易日数（默认 20）",
            "correlation_threshold": "float，高相关提示阈值（默认 0.7）",
        },
        required=("names",),
        returns="payload: {n, table:[{name, ic_mean, ic_ir, ic_t, pos_ratio}], high_correlations}",
        cost=5.0,
        tags=("experiment", "compare"),
    )
    def compare_factors(
        context: ToolContext,
        names: list,
        horizon: int = 20,
        correlation_threshold: float = 0.7,
    ):
        from qfm.analysis import compare_factors as compare_impl

        panel = context.panel
        if panel is None or getattr(panel.close, "empty", True):
            raise InvalidArgumentsError("尚未加载面板；请先执行 load_panel")
        targets = [str(item) for item in names]
        if len(targets) < 2:
            raise InvalidArgumentsError(f"对比至少需要 2 个因子，收到 {targets}")
        for target in targets:
            _require_factor(target)

        result = compare_impl(
            panel, targets, horizon=int(horizon),
            correlation_threshold=float(correlation_threshold),
        )

        # FactorCompareResult 的 metrics / high_correlations 都是 DataFrame
        # （因子按行存放，不是按索引），所以统一走 to_dict("records")，
        # 不能对 DataFrame 做真假判断——那会抛「truth value is ambiguous」。
        table = []
        for row in _records(result.metrics):
            name = str(row.get("factor") or row.get("name") or "")
            table.append({
                "name": name,
                "label": _label(name) if name else "",
                "ic_mean": _finite(row.get("ic")),
                "ic_ir": _finite(row.get("icir")),
                "ic_t": _finite(row.get("ic_t")),
                "rank_ic": _finite(row.get("rank_ic")),
                "long_short_return": _finite(row.get("long_short_return")),
                "turnover": _finite(row.get("turnover")),
                "coverage": _finite(row.get("coverage")),
                "stability": _finite(row.get("stability")),
            })

        pairs = []
        for row in _records(result.high_correlations):
            pairs.append({
                "left": row.get("factor_a"),
                "right": row.get("factor_b"),
                "correlation": _finite(row.get("correlation")),
                "absolute_correlation": _finite(row.get("absolute_correlation")),
            })

        payload = {
            "n": len(table),
            "table": table,
            "high_correlations": pairs,
            "horizon": int(horizon),
            "correlation_threshold": float(correlation_threshold),
        }
        context.facts["compare"] = payload
        summary = (
            f"{len(table)} 个因子对比完成；"
            + "；".join(f"{row['label']} IC={_fmt(row.get('ic_mean'))}" for row in table)
            + (f"；{len(pairs)} 对高相关（阈值 {correlation_threshold}）" if pairs else "")
        )
        return payload, summary

    @registry.register(
        "generate_report",
        kind=ToolKind.WRITE,
        description="把本次研究的计划、工具调用轨迹、实验指标、异常检查与结论写成 Markdown 报告并落盘",
        parameters={
            "title": "str，报告标题",
            "include_trace": "bool，是否包含完整工具调用轨迹（默认 true）",
        },
        required=(),
        returns="payload: {path, chars, sections}",
        cost=1.0,
        tags=("report",),
    )
    def generate_report(
        context: ToolContext,
        title: str | None = None,
        include_trace: bool = True,
    ):
        from qfm.agent.reporting import render_findings_markdown

        facts = dict(context.facts)
        control = facts.get("experiment")
        primary = facts.get("primary_experiment")
        if not control and not primary:
            raise InvalidArgumentsError(
                "还没有实验结果可写入报告；请先执行 run_experiment 与 check_anomalies"
            )
        # 报告以主实验为正文（用户问的是那个池子），对照作为稳健性检验单独成节。
        headline = primary or control or {}
        heading = title or f"因子研究报告：{headline.get('label')}"
        # 对照实验的异常必须按「对照自己的键」取：补救轮会跑同一个因子的另一个股票池，
        # 用 facts["anomalies"]（主实验的）会把主实验的异常错配到对照上。
        by_experiment = context.get("anomalies_by_experiment") or {}
        control_anomalies = []
        if control:
            control_anomalies = list(
                (by_experiment.get(control.get("key")) or {}).get("anomalies") or []
            )
        body = render_findings_markdown(
            heading,
            experiment=control,
            primary=primary,
            anomalies=control_anomalies,
            primary_anomalies=facts.get("primary_anomalies"),
            panel_meta=context.panel_meta,
            compare=facts.get("compare"),
            include_trace=bool(include_trace),
        )
        path = context.artifact_path("research_report.md")
        path.write_text(body, encoding="utf-8")
        context.artifacts.append(str(path))
        context.put("report_path", str(path))

        payload = {
            "path": str(path),
            "chars": len(body),
            "sections": body.count("\n## "),
        }
        return payload, f"报告已写入 {path}（{len(body)} 字符）"


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------


def _records(frame) -> list[dict]:
    """把 DataFrame / 列表统一成字典列表；None 与空表返回空列表。"""
    if frame is None:
        return []
    if hasattr(frame, "to_dict"):
        try:
            if getattr(frame, "empty", False):
                return []
            return [dict(row) for row in frame.to_dict("records")]
        except TypeError:
            pass
    if isinstance(frame, (list, tuple)):
        return [dict(item) if isinstance(item, dict) else {"value": item} for item in frame]
    return []


def _experiment_key(name: str, pool) -> str:
    """实验的唯一键：因子名 @ 股票池。

    只用因子名不行——补救轮会对**同一个因子**换股票池重跑，
    两次实验都叫 bp，必须靠股票池区分主实验与对照。
    """
    return f"{name}@{pool or 'unknown'}"


def _require_factor(name: str):
    from qfm.factors import get_factor

    factor = get_factor(name)
    if factor is None:
        raise PermanentError(f"未知因子 {name!r}")
    return factor


def _label(name: str) -> str:
    from qfm.factors import factor_label

    return factor_label(name)


def _window(panel, start: str, end: str, horizon: int) -> dict:
    """把请求区间夹到面板的实际可用范围内，并核对有效性。"""
    index = panel.close.index
    first, last = index.min(), index.max()
    try:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    except Exception as exc:  # noqa: BLE001
        raise InvalidArgumentsError(f"日期解析失败: {exc}") from exc
    if start_ts >= end_ts:
        raise InvalidArgumentsError(f"起始日期 {start} 必须早于结束日期 {end}")

    clipped_start = max(start_ts, first)
    clipped_end = min(end_ts, last)
    if clipped_start >= clipped_end:
        raise InvalidArgumentsError(
            f"请求区间 {start} ~ {end} 与数据区间 "
            f"{first.date()} ~ {last.date()} 没有交集"
        )
    days = int(((index >= clipped_start) & (index <= clipped_end)).sum())
    if days < max(21, horizon + 2):
        raise InvalidArgumentsError(
            f"区间内仅 {days} 个交易日，不足以做 {horizon} 日前瞻的检验"
            f"（至少需要 {max(21, horizon + 2)} 个）；请扩大区间"
        )
    clipped = clipped_start > start_ts or clipped_end < end_ts
    return {
        "start": str(clipped_start.date()),
        "end": str(clipped_end.date()),
        "requested_start": start,
        "requested_end": end,
        "trading_days": days,
        "clipped": bool(clipped),
        "note": (
            f"请求区间已按数据范围裁剪为 {clipped_start.date()} ~ {clipped_end.date()}"
            if clipped else ""
        ),
    }


def _fit_top_n(top_n: int, n_symbols: int, mode: str) -> tuple[int, str]:
    """按股票池宽度收窄持仓数：A 股多头要 top_n 只，多空要 top_n 只多头 + top_n 只空头。"""
    if top_n < 1:
        raise InvalidArgumentsError(f"top_n 必须 >= 1，得到 {top_n}")
    factor = 2 if mode == "long_short" else 1
    maximum = max(1, n_symbols // factor)
    if top_n <= maximum:
        return top_n, ""
    return maximum, (
        f"持仓数已从 {top_n} 收窄到 {maximum}：股票池仅 {n_symbols} 只，"
        f"{mode} 模式最多容纳 {maximum} 只；本结论对应的组合配置与请求不同"
    )


def _job_request(context: ToolContext, name: str, factor, settings, window) -> dict:
    """构造与既有页面同构的请求，使 Agent 实验共享同一套缓存键语义。"""
    from qfm.research.snapshot import build_data_snapshot

    request: dict = {
        "universe": context.panel_meta.get("pool", "unknown"),
        "date_range": [settings.start, settings.end],
        "factor_versions": [{
            "name": name,
            "version": factor.version,
            "source_hash": factor.source_hash,
        }],
        "pipeline_config": {
            "decay": settings.decay,
            "neutralization": settings.neutralization,
            "horizon": settings.horizon,
        },
        "backtest_config": {
            "top_n": settings.top_n,
            "mode": settings.mode,
            "rebalance": settings.rebalance,
        },
        "requested_window": [window["requested_start"], window["requested_end"]],
    }
    try:
        snapshot = build_data_snapshot(context.panel, request["universe"])
        from qfm.data.catalog import register_panel_dataset

        binding = register_panel_dataset(
            context.panel, request["universe"], snapshot,
            root=context.catalog_root,
            universe_symbols=context.panel_meta.get("symbols"),
        )
        request.update(binding)
    except Exception as exc:  # noqa: BLE001 - 绑定失败仍应能出结果，但需留痕
        request["binding_warning"] = f"{type(exc).__name__}: {exc}"
    return request


def _diagnose(experiment: dict, min_abs_ic_t: float, max_turnover: float) -> list[dict]:
    """把实验结果翻译成异常列表。每条异常都带证据与补救建议。"""
    anomalies: list[dict] = []

    def add(code: str, message: str, evidence: dict) -> None:
        anomalies.append({
            "code": code,
            "severity": ANOMALY_RULES.get(code, "low"),
            "message": message,
            "evidence": evidence,
            "remediation": remediation_for(code),
        })

    ic = experiment.get("ic") or {}
    metrics = experiment.get("metrics") or {}
    mono = experiment.get("monotonicity") or {}
    n_symbols = int(experiment.get("n_symbols") or 0)
    layer_available = bool(experiment.get("layer_available"))

    if n_symbols and n_symbols < 100:
        add(
            "narrow_cross_section",
            f"股票池仅 {n_symbols} 只，截面宽度不足；IC 的统计功效受限，"
            "结论对单只股票的异常表现更敏感",
            {"n_symbols": n_symbols, "threshold": 100},
        )
    if not layer_available:
        add(
            "layer_test_unavailable",
            "分层单调性检验不可用：每日有效股票数不足（需要至少 100 只），"
            "因此无法验证因子是否有分组区分度，只依赖 IC 判断",
            {"n_symbols": n_symbols, "required_per_day": 100},
        )

    coverage = experiment.get("coverage")
    if isinstance(coverage, (int, float)) and coverage == coverage and coverage < 0.8:
        add(
            "low_coverage",
            f"信号覆盖率 {coverage:.1%} 偏低，部分日期或标的没有有效因子值",
            {"coverage": round(float(coverage), 4), "threshold": 0.8},
        )

    ic_mean, ic_t = ic.get("ic_mean"), ic.get("ic_t")
    if isinstance(ic_t, (int, float)) and ic_t == ic_t:
        if abs(ic_t) < min_abs_ic_t:
            add(
                "insignificant_ic",
                f"IC 的 t 值 {ic_t:.2f} 未达到 |t| >= {min_abs_ic_t}，"
                "无法拒绝「该因子无预测力」的原假设",
                {"ic_mean": _finite(ic_mean), "ic_t": float(ic_t), "threshold": min_abs_ic_t},
            )
        elif isinstance(ic_mean, (int, float)) and ic_mean == ic_mean and ic_mean < 0:
            direction = experiment.get("direction", "positive")
            add(
                "ic_direction_conflict",
                f"IC 显著为负（{ic_mean:+.4f}，t={ic_t:.2f}）而因子定义为 {direction} 方向；"
                "可能是方向设置与实际含义不符，或是风格暴露造成的伪信号",
                {"ic_mean": float(ic_mean), "ic_t": float(ic_t), "declared_direction": direction},
            )

    pos_ratio = ic.get("pos_ratio")
    if isinstance(pos_ratio, (int, float)) and pos_ratio == pos_ratio and pos_ratio < 0.5:
        add(
            "unstable_ic",
            f"IC 为正的比例仅 {pos_ratio:.1%}（低于 50%），预测方向不稳定",
            {"pos_ratio": float(pos_ratio), "threshold": 0.5},
        )

    turnover = experiment.get("turnover")
    if isinstance(turnover, (int, float)) and turnover == turnover and turnover > max_turnover:
        add(
            "high_turnover",
            f"因子换手率 {turnover:.1%} 超过上限 {max_turnover:.0%}，"
            "交易成本可能吃掉大部分理论收益",
            {"turnover": float(turnover), "threshold": max_turnover},
        )

    if layer_available and not mono.get("monotonic", True):
        add(
            "weak_monotonicity",
            "分层收益不单调：因子对收益的排序能力不连续，"
            "多空价差可能只来自个别层级",
            {"monotonic": mono.get("monotonic"), "corr": _finite(mono.get("corr"))},
        )

    excess = metrics.get("年化超额")
    if isinstance(excess, (int, float)) and excess == excess and excess < 0:
        add(
            "negative_excess",
            f"年化超额收益为 {excess:.2%}：组合没有跑赢等权基准，"
            "说明在该区间内按此因子选股并未创造相对价值",
            {"年化超额": round(float(excess), 4)},
        )

    drawdown = metrics.get("最大回撤")
    if isinstance(drawdown, (int, float)) and drawdown == drawdown and drawdown < -0.3:
        add(
            "large_drawdown",
            f"最大回撤 {drawdown:.1%}，超过 30% 的容忍线",
            {"最大回撤": round(float(drawdown), 4), "threshold": -0.3},
        )

    for warning in experiment.get("warnings") or []:
        text = str(warning)
        if "未产生成交" in text:
            add("no_trades", text, {"engine_warning": text})
        elif "分层" in text and "不足" in text:
            continue  # 已由 layer_test_unavailable 覆盖
        elif "覆盖率" in text:
            continue  # 已由 low_coverage 覆盖
        else:
            add("engine_warning", text, {"engine_warning": text})
    return anomalies


def _clean(mapping) -> dict:
    """把指标字典里的 numpy 标量转成原生标量，并把 NaN/inf 统一成 None。"""
    out: dict = {}
    if not mapping:
        return out
    items = mapping.items() if hasattr(mapping, "items") else []
    for key, value in items:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[str(key)] = _finite(value)
        elif isinstance(value, str):
            out[str(key)] = value
    return out


def _yearly(frame) -> list[dict]:
    if frame is None or getattr(frame, "empty", True):
        return []
    rows = []
    for _, row in frame.iterrows():
        rows.append({str(key): _finite(value) if isinstance(value, float) else value
                     for key, value in row.items()})
    return rows


def _finite(value) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt(value) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:+.4f}"


def _pct(value) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:+.2%}"
