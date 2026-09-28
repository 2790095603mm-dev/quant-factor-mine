"""研究助理页面的真实 Streamlit 控件冒烟测试。

用 `streamlit.testing.v1.AppTest` 真正渲染页面并模拟点击，而不是只 import 一下：
页面里最容易出问题的地方恰恰是「渲染时才求值」的分支（指标格式化、表格构造、
异常展开），这类问题 import 是发现不了的。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

# 页面脚本：用合成数据源，避免测试依赖本地 data_cache
AGENT_SCRIPT = '''
import streamlit as st
from qfm.agent.providers import SyntheticProvider
from qfm.agent.views import render_agent_page

# 强制使用合成面板：本测试不依赖本地缓存是否存在
import qfm.agent.providers as providers
providers.resolve_provider = lambda mode="auto", **kwargs: SyntheticProvider()

render_agent_page(
    runs_root=st.session_state["runs_root"],
    data_dir="data_cache",
)
'''

RUN_SCRIPT = AGENT_SCRIPT + '''
'''

EXAMPLE_RUN_SCRIPT = AGENT_SCRIPT + '''
# 再跑一次并点击「运行研究」，验证完整交互路径
'''

INVALID_REQUEST_SCRIPT = AGENT_SCRIPT


def _app(tmp_path, script=AGENT_SCRIPT) -> AppTest:
    app = AppTest.from_string(script)
    app.session_state["runs_root"] = str(tmp_path / "runs")
    return app


def test_agent_page_renders_without_exception(tmp_path):
    app = _app(tmp_path)
    app.run(timeout=60)
    assert not app.exception, [str(item) for item in app.exception]
    assert app.markdown, "页面应渲染出标题区"
    titles = " ".join(block.value for block in app.markdown)
    assert "研究助理" in titles


def test_agent_page_exposes_request_input_and_controls(tmp_path):
    app = _app(tmp_path)
    app.run(timeout=60)
    assert not app.exception
    # 研究请求输入框
    assert any("研究请求" in area.label for area in app.text_area)
    # 数据源与规划器选择
    labels = {box.label for box in app.selectbox}
    assert {"数据源", "规划器"} <= labels
    # 权限控件
    assert any("只读模式" in box.label for box in app.checkbox)
    assert any("禁用工具" in box.label for box in app.multiselect)


def test_agent_page_example_button_fills_and_runs(tmp_path):
    """示例按钮在表单之外，点击后应填入请求并一键跑完。"""
    app = _app(tmp_path)
    app.run(timeout=60)
    assert not app.exception
    example_buttons = [button for button in app.button if button.label.startswith("示例")]
    assert len(example_buttons) == 4, "应有 4 个示例按钮"

    example_buttons[1].click().run(timeout=240)
    assert not app.exception, [str(item) for item in app.exception]
    assert "动量" in app.session_state["agent_request_text"]
    assert app.session_state["agent_last_run"] is not None


def test_agent_page_runs_end_to_end_and_shows_results(tmp_path):
    """真正点击「运行研究」并断言页面出现结论、轨迹与异常区块。"""
    app = _app(tmp_path)
    app.run(timeout=60)
    assert not app.exception

    app.text_area(key="agent_request_text").set_value("分析最近一年银行股低估值因子的表现")
    run_buttons = [button for button in app.button if "运行研究" in str(button.label)]
    assert run_buttons, "应有运行按钮"
    run_buttons[0].click().run(timeout=180)

    assert not app.exception, [str(item) for item in app.exception]

    run = app.session_state["agent_last_run"]
    assert run is not None
    assert run.findings["primary"]["name"] == "bp"
    assert run.findings["primary"]["n_symbols"] == 42

    # 结果区应包含关键指标表与轨迹表
    frames = [frame.value for frame in app.dataframe]
    assert frames, "应渲染出结果表格"
    all_text = " ".join(str(frame) for frame in frames)
    assert "IC 均值" in all_text
    assert "run_experiment" in all_text

    # 告警区应提示截面过窄
    warnings = " ".join(str(item.value) for item in app.warning)
    assert "narrow_cross_section" in warnings or "截面宽度不足" in warnings


def test_agent_page_reports_read_only_denial(tmp_path):
    """只读模式下应看到被拒绝的调用，而不是静默少了报告。"""
    app = _app(tmp_path)
    app.run(timeout=60)
    app.text_area(key="agent_request_text").set_value("分析最近一年银行股低估值因子的表现")
    app.checkbox(key="agent_read_only").set_value(True)
    for button in app.button:
        if "运行研究" in str(button.label):
            button.click().run(timeout=180)
            break

    assert not app.exception
    run = app.session_state["agent_last_run"]
    assert run.status.value == "PARTIAL"
    denied = [task.tool for task in run.tasks if task.status.value == "DENIED"]
    assert denied and set(denied) == {"generate_report"}


def test_agent_page_handles_empty_request(tmp_path):
    app = _app(tmp_path)
    app.run(timeout=60)
    app.text_area(key="agent_request_text").set_value("   ")
    for button in app.button:
        if "运行研究" in str(button.label):
            button.click().run(timeout=60)
            break
    assert not app.exception
    assert any("请先填写研究请求" in str(item.value) for item in app.warning)


def test_agent_page_disabled_tool_is_surfaced(tmp_path):
    app = _app(tmp_path)
    app.run(timeout=60)
    app.text_area(key="agent_request_text").set_value("分析最近一年银行股低估值因子")
    app.multiselect(key="agent_denied").set_value(["load_panel"])
    for button in app.button:
        if "运行研究" in str(button.label):
            button.click().run(timeout=180)
            break
    assert not app.exception
    run = app.session_state["agent_last_run"]
    assert run.status.value == "FAILED"
    assert run.findings["primary"] is None


def test_agent_page_caps_replan_rounds(tmp_path):
    app = _app(tmp_path)
    app.run(timeout=60)
    app.text_area(key="agent_request_text").set_value("分析最近一年银行股低估值因子的表现")
    app.number_input(key="agent_max_replan").set_value(0)
    for button in app.button:
        if "运行研究" in str(button.label):
            button.click().run(timeout=180)
            break
    assert not app.exception
    run = app.session_state["agent_last_run"]
    assert len(run.plans) == 1
    assert run.findings["control"] is None


def test_agent_page_download_buttons_present(tmp_path):
    app = _app(tmp_path)
    app.run(timeout=60)
    app.text_area(key="agent_request_text").set_value("分析最近一年银行股低估值因子的表现")
    for button in app.button:
        if "运行研究" in str(button.label):
            button.click().run(timeout=180)
            break
    assert not app.exception
    labels = {button.label for button in app.get("download_button")}
    assert any("研究报告" in label for label in labels)
    assert any("运行记录" in label for label in labels)
