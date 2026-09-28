"""工具层测试：数据工具、因子工具、实验工具、异常检查、台账查询。

全部在合成面板上离线运行，并注入休眠替身使重试不实际等待。
"""

from __future__ import annotations

import pytest


def call(registry, context, tool, **kwargs):
    result = registry.invoke(tool, kwargs, context)
    assert result.ok, f"{tool} 失败: {result.error}"
    return result.value


# ---------------------------------------------------------------------------
# 数据工具：resolve_universe
# ---------------------------------------------------------------------------
def test_resolve_universe_by_industry_keyword(registry, tool_context):
    payload = call(registry, tool_context, "resolve_universe", keyword="银行股")
    assert payload["pool"] == "industry:银行Ⅱ"
    assert payload["n_symbols"] == 42
    assert payload["matched_labels"] == ["银行Ⅱ"]
    assert payload["method"] == "alias"
    # 通过别名匹配时必须提示口径差异
    assert "东财行业分类" in payload["note"]
    assert tool_context.get("universe")["symbols"] == payload["symbols"]


def test_resolve_universe_by_pool(registry, tool_context):
    payload = call(registry, tool_context, "resolve_universe", pool="沪深300")
    assert payload["pool"] == "cn_hs300"
    assert payload["n_symbols"] > 0
    assert payload["method"] == "pool"


def test_resolve_universe_prefers_industry_over_pool(registry, tool_context):
    """用户说「银行股」时不该落到全市场池。"""
    payload = call(registry, tool_context, "resolve_universe", keyword="银行", pool="cn_hs300")
    assert payload["pool"].startswith("industry:")


def test_resolve_universe_preset_symbols(registry, tool_context, bank_symbols):
    tool_context.put("preset_universe", {"pool": "explicit", "symbols": bank_symbols, "as_of": "2026-09-18"})
    payload = call(registry, tool_context, "resolve_universe")
    assert payload["n_symbols"] == len(bank_symbols)
    assert payload["method"] == "preset"


def test_resolve_universe_requires_something(registry, tool_context):
    result = registry.invoke("resolve_universe", {}, tool_context)
    assert not result.ok
    assert result.error_type == "InvalidArgumentsError"


def test_resolve_universe_unknown_industry_lists_available(registry, tool_context):
    result = registry.invoke("resolve_universe", {"keyword": "外星行业"}, tool_context)
    assert not result.ok
    assert "未匹配到任何行业标签" in result.error
    assert "银行Ⅱ" in result.error, "错误信息应列出可用行业，便于纠正"


def test_resolve_universe_max_stocks_truncates_and_notes(registry, tool_context):
    payload = call(registry, tool_context, "resolve_universe", keyword="银行股", max_stocks=10)
    assert payload["n_symbols"] == 10
    assert "截断" in payload["note"]


# ---------------------------------------------------------------------------
# 数据工具：load_panel / describe_panel
# ---------------------------------------------------------------------------
def test_load_panel_slices_to_analysis_window(registry, tool_context):
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    payload = call(
        registry, tool_context, "load_panel", start="2025-09-18", end="2026-09-18"
    )
    # 面板整体跨 620 个交易日；裁到「一年 + 320 日预热」后应显著变短
    assert payload["n_days"] < 620
    assert payload["end"] == "2026-09-18"
    assert payload["analysis_window"] == ["2025-09-18", "2026-09-18"]
    assert "含 320 日预热" in payload["window_note"]
    assert tool_context.panel is not None


def test_load_panel_warmup_covers_long_window_factors(registry, tool_context):
    """预热段必须足够长，否则 mom_250 这类因子在窗口首日会取不到值。"""
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    call(registry, tool_context, "load_panel", start="2025-09-18", end="2026-09-18")
    from qfm.factors import compute_factor

    mom = compute_factor("mom_250", tool_context.panel)
    tail = mom.loc[mom.index >= "2025-09-18"]
    assert tail.notna().to_numpy().mean() > 0.5, "预热不足会让长窗口因子在窗口内大面积缺失"


def test_load_panel_without_window_keeps_full_history(registry, tool_context):
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    payload = call(registry, tool_context, "load_panel")
    assert payload["n_days"] == 620
    assert payload["analysis_window"] is None


def test_load_panel_labels_synthetic_data(registry, tool_context):
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    payload = call(registry, tool_context, "load_panel")
    assert payload["source"] == "synthetic"
    assert payload.get("is_synthetic") is True
    assert any("合成" in warning for warning in payload["warnings"])


def test_load_panel_rejects_window_outside_data(registry, tool_context):
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    result = registry.invoke(
        "load_panel", {"start": "2030-01-01", "end": "2030-12-31"}, tool_context
    )
    assert not result.ok
    assert "没有交集" in result.error


def test_describe_panel_flags_narrow_cross_section(registry, tool_context):
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    call(registry, tool_context, "load_panel")
    payload = call(registry, tool_context, "describe_panel")
    assert payload["layer_test_available"] is False
    assert any("低于分层检验所需的 100 只" in warning for warning in payload["warnings"])


def test_describe_panel_reports_fund_coverage(registry, tool_context):
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    call(registry, tool_context, "load_panel")
    payload = call(registry, tool_context, "describe_panel")
    assert "bvps" in payload["fund_fields"]
    assert payload["fund_coverage"]["bvps"] > 0.9


def test_describe_panel_requires_panel(registry, tool_context):
    result = registry.invoke("describe_panel", {}, tool_context)
    assert not result.ok
    assert "尚未加载面板" in result.error


# ---------------------------------------------------------------------------
# 因子工具
# ---------------------------------------------------------------------------
def test_list_factors_by_family(registry, tool_context):
    payload = call(registry, tool_context, "list_factors", family="价值")
    names = {item["name"] for item in payload["factors"]}
    assert {"bp", "ep_ttm"} <= names
    assert payload["n_total"] == 3


def test_list_factors_keyword_filter(registry, tool_context):
    payload = call(registry, tool_context, "list_factors", keyword="动量")
    names = {item["name"] for item in payload["factors"]}
    assert "mom_20" in names
    # 过滤同时匹配名称、中文名、描述与家族（「动量反转」家族会带出反转因子）
    assert all(
        "动量" in item["name"] + item["label"] + item["description"] + item["family"]
        for item in payload["factors"]
    )


def test_describe_factor_returns_version_and_hash(registry, tool_context):
    payload = call(registry, tool_context, "describe_factor", name="bp")
    assert payload["family"] == "价值"
    assert payload["direction"] == "positive"
    assert payload["version"] >= 1
    assert payload["source_hash"].startswith("sha256:")
    assert payload["version_index"] == {
        "name": "bp", "version": payload["version"], "source_hash": payload["source_hash"],
    }


def test_describe_factor_unknown_suggests_near_names(registry, tool_context):
    result = registry.invoke("describe_factor", {"name": "mom"}, tool_context)
    assert not result.ok
    assert "未知因子" in result.error


def test_compute_factor_keeps_matrix_out_of_payload(registry, tool_context):
    """重对象必须留在工作集，payload 只带统计量——否则上下文会被撑爆。"""
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    call(registry, tool_context, "load_panel")
    payload = call(registry, tool_context, "compute_factor", name="bp")
    assert payload["coverage"] > 0.9
    assert payload["avg_cross_section"] == pytest.approx(42, abs=1)
    assert "value" not in payload and "frame" not in payload
    frame = tool_context.get("factor:bp")
    assert frame is not None and frame.shape[1] == 42


def test_compute_factor_raises_degraded_when_field_missing(registry, tool_context):
    """因子依赖的财务字段缺失时应报降级（可重试），而不是静默返回全空矩阵。"""
    call(registry, tool_context, "resolve_universe", keyword="银行股")
    call(registry, tool_context, "load_panel")
    tool_context.panel.fund.pop("bvps", None)  # 抹掉 bp 依赖的字段
    result = registry.invoke("compute_factor", {"name": "bp"}, tool_context)
    assert not result.ok
    assert result.error_type == "DegradedResultError"


def test_audit_lookahead_passes_for_builtin_factor(registry, tool_context):
    payload = call(registry, tool_context, "audit_lookahead", name="bp")
    assert payload["leaks"] == []
    assert payload["verdict"] in ("通过", "有提示")


# ---------------------------------------------------------------------------
# 实验工具
# ---------------------------------------------------------------------------
def _prepare(registry, context, keyword="银行股", window=("2025-09-18", "2026-09-18")):
    call(registry, context, "resolve_universe", keyword=keyword)
    call(registry, context, "load_panel", start=window[0], end=window[1])
    call(registry, context, "compute_factor", name="bp")


def test_run_experiment_returns_metrics(registry, tool_context):
    _prepare(registry, tool_context)
    payload = call(
        registry, tool_context, "run_experiment",
        name="bp", start="2025-09-18", end="2026-09-18",
    )
    assert payload["ic"]["ic_mean"] is not None
    assert payload["metrics"]["年化收益"] is not None
    assert payload["window"]["requested_start"] == "2025-09-18"
    assert payload["key"] == "bp@industry:银行Ⅱ"
    assert payload["pool"] == "industry:银行Ⅱ"


def test_run_experiment_narrows_top_n_for_narrow_universe(registry, tool_context):
    """42 只的池子里要 30 只多头没问题；多空模式需要 2×top_n，必须收窄并留痕。"""
    _prepare(registry, tool_context)
    payload = call(
        registry, tool_context, "run_experiment",
        name="bp", start="2025-09-18", end="2026-09-18", top_n=30, mode="long_short",
    )
    assert payload["top_n_requested"] == 30
    assert payload["top_n_effective"] == 21
    assert any("收窄" in warning for warning in payload["warnings"])


def test_run_experiment_rejects_bad_window(registry, tool_context):
    _prepare(registry, tool_context)
    result = registry.invoke(
        "run_experiment",
        {"name": "bp", "start": "2026-09-18", "end": "2025-09-18"},
        tool_context,
    )
    assert not result.ok
    assert "必须早于" in result.error


def test_run_experiment_rejects_short_window(registry, tool_context):
    _prepare(registry, tool_context)
    result = registry.invoke(
        "run_experiment", {"name": "bp", "start": "2026-09-10", "end": "2026-09-18"}, tool_context
    )
    assert not result.ok
    assert "交易日" in result.error


def test_run_experiment_clips_window_to_data_and_notes(registry, tool_context):
    _prepare(registry, tool_context, window=("2024-06-01", "2026-09-18"))
    payload = call(
        registry, tool_context, "run_experiment",
        name="bp", start="2020-01-01", end="2030-01-01",
    )
    assert payload["window"]["clipped"] is True
    assert payload["window"]["note"]


def test_run_experiment_accumulates_by_factor_and_pool(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    call(registry, tool_context, "compute_factor", name="roe")
    call(registry, tool_context, "run_experiment", name="roe", start="2025-09-18", end="2026-09-18")
    experiments = tool_context.get("experiments")
    assert set(experiments) == {"bp@industry:银行Ⅱ", "roe@industry:银行Ⅱ"}


# ---------------------------------------------------------------------------
# 异常检查
# ---------------------------------------------------------------------------
def test_check_anomalies_binds_to_primary_experiment(registry, tool_context):
    """异常必须对应到主实验，否则会出现「用 A 的指标配 B 的异常」。"""
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    tool_context.put("primary_factor_key", "bp@industry:银行Ⅱ")
    call(registry, tool_context, "compute_factor", name="roe")
    call(registry, tool_context, "run_experiment", name="roe", start="2025-09-18", end="2026-09-18")

    payload = call(registry, tool_context, "check_anomalies")
    assert payload["primary_key"] == "bp@industry:银行Ⅱ"
    assert set(payload["by_experiment"]) == {"bp@industry:银行Ⅱ", "roe@industry:银行Ⅱ"}
    # 主实验的异常里不应混入 roe 的指标证据
    codes = {item["code"] for item in payload["anomalies"]}
    assert "narrow_cross_section" in codes


def test_check_anomalies_requires_experiment(registry, tool_context):
    result = registry.invoke("check_anomalies", {}, tool_context)
    assert not result.ok
    assert "尚无实验结果" in result.error


def test_check_anomalies_can_filter_by_factor(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    call(registry, tool_context, "compute_factor", name="roe")
    call(registry, tool_context, "run_experiment", name="roe", start="2025-09-18", end="2026-09-18")
    payload = call(registry, tool_context, "check_anomalies", name="roe")
    assert list(payload["by_experiment"]) == ["roe@industry:银行Ⅱ"]
    assert payload["primary_key"] == "roe@industry:银行Ⅱ"


def test_check_anomalies_rejects_unknown_factor(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    result = registry.invoke("check_anomalies", {"name": "roe"}, tool_context)
    assert not result.ok
    assert "没有因子" in result.error and "已跑过的因子" in result.error


def test_check_anomalies_flags_narrow_cross_section_as_high(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    payload = call(registry, tool_context, "check_anomalies")
    narrow = next(item for item in payload["anomalies"] if item["code"] == "narrow_cross_section")
    assert narrow["severity"] == "high"
    assert narrow["remediation"]
    assert payload["needs_replan"] is True


def test_check_anomalies_detects_direction_conflict(registry, tool_context, monkeypatch):
    """IC 显著为负而因子定义为正向 → 必须报方向冲突，不能当作有效因子。"""
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    experiment = tool_context.get("experiment")
    experiment["ic"]["ic_mean"] = -0.12
    experiment["ic"]["ic_t"] = -6.5
    payload = call(registry, tool_context, "check_anomalies")
    codes = {item["code"] for item in payload["anomalies"]}
    assert "ic_direction_conflict" in codes


def test_check_anomalies_detects_insignificant_ic(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    tool_context.get("experiment")["ic"]["ic_t"] = 0.8
    payload = call(registry, tool_context, "check_anomalies")
    codes = {item["code"] for item in payload["anomalies"]}
    assert "insignificant_ic" in codes


def test_check_anomalies_detects_high_turnover(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    tool_context.get("experiment")["turnover"] = 0.85
    payload = call(registry, tool_context, "check_anomalies", max_turnover=0.6)
    codes = {item["code"] for item in payload["anomalies"]}
    assert "high_turnover" in codes


def test_check_anomalies_detects_large_drawdown(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    tool_context.get("experiment")["metrics"]["最大回撤"] = -0.55
    payload = call(registry, tool_context, "check_anomalies")
    codes = {item["code"] for item in payload["anomalies"]}
    assert "large_drawdown" in codes


def test_check_anomalies_does_not_duplicate_engine_warnings(registry, tool_context):
    """引擎已产出的分层/覆盖率告警不应再以 engine_warning 之名重复一遍。"""
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    payload = call(registry, tool_context, "check_anomalies")
    engine = [item for item in payload["anomalies"] if item["code"] == "engine_warning"]
    for item in engine:
        assert "分层" not in item["message"] or "不足" not in item["message"]


# ---------------------------------------------------------------------------
# 报告与对比
# ---------------------------------------------------------------------------
def test_generate_report_writes_markdown_with_sections(registry, tool_context):
    _prepare(registry, tool_context)
    call(registry, tool_context, "run_experiment", name="bp", start="2025-09-18", end="2026-09-18")
    call(registry, tool_context, "check_anomalies")
    payload = call(registry, tool_context, "generate_report", title="测试报告")
    body = open(payload["path"], encoding="utf-8").read()
    assert "# 测试报告" in body
    assert "## 一、研究配置" in body
    assert "## 二、核心指标" in body
    assert "## 结论与局限" in body
    assert "nan" not in body.lower().replace("财务", ""), "报告里不应出现 nan"
    assert "合成面板" in body, "合成数据必须显著标注"
    assert payload["path"] in tool_context.artifacts


def test_generate_report_requires_experiment(registry, tool_context):
    result = registry.invoke("generate_report", {}, tool_context)
    assert not result.ok
    assert "还没有实验结果" in result.error


def test_compare_factors(registry, tool_context):
    _prepare(registry, tool_context)
    payload = call(registry, tool_context, "compare_factors", names=["bp", "ep_ttm"])
    assert payload["n"] == 2
    assert {row["name"] for row in payload["table"]} == {"bp", "ep_ttm"}


def test_compare_factors_needs_two(registry, tool_context):
    _prepare(registry, tool_context)
    result = registry.invoke("compare_factors", {"names": ["bp"]}, tool_context)
    assert not result.ok
    assert "至少需要 2 个" in result.error


# ---------------------------------------------------------------------------
# 台账
# ---------------------------------------------------------------------------
def test_list_experiments_without_store_reports_unavailable(registry, tool_context):
    payload = call(registry, tool_context, "list_experiments")
    assert payload["available"] is False
    assert "未挂载" in payload["note"]


# ---------------------------------------------------------------------------
# 全量工具注册
# ---------------------------------------------------------------------------
def test_default_registry_has_all_twelve_tools(registry):
    assert registry.names() == [
        "audit_lookahead", "check_anomalies", "compare_factors", "compute_factor",
        "describe_factor", "describe_panel", "generate_report", "list_experiments",
        "list_factors", "load_panel", "resolve_universe", "run_experiment",
    ]


def test_every_tool_declares_parameters_and_purpose(registry):
    for spec in registry.specs():
        assert spec.description, f"{spec.name} 缺少描述"
        assert spec.returns, f"{spec.name} 未声明返回结构"


def test_catalog_text_lists_all_tools_with_required_markers(registry):
    catalog = registry.catalog()
    for name in registry.names():
        assert name in catalog
    assert "必填" in catalog
