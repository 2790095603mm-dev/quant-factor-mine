"""真实 Streamlit 控件流程：运行、改参、重跑和归档。"""
from streamlit.testing.v1 import AppTest

from qfm.research import ResearchStore

SCRIPT = '''
import streamlit as st
from qfm.research import ResearchStore
from qfm.simulation.views import render_single_factor
render_single_factor(st.session_state['fixture_panel'], 'test_pool', ResearchStore(st.session_state['store_root']))
'''


def test_simulate_change_settings_preserve_results_and_save(panel, tmp_path):
    store = ResearchStore(tmp_path)
    project = store.create_project("single-factor-test")
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture_panel"] = panel
    app.session_state["store_root"] = str(tmp_path)
    app.session_state["active_project_id"] = project.id
    app.session_state["test_factor"] = "mom_5"
    app.run()
    assert not app.exception
    app.button(key="sim_run").click().run()
    assert not app.exception
    assert not app.error
    labels = {metric.label for metric in app.metric}
    assert {"Sharpe", "最大回撤", "Fitness", "年化收益", "日均换手", "平均 IC"}.issubset(labels)
    initial = app.session_state["factor_simulation"]
    assert "plotly.js" in initial["html"]
    app.number_input(key="sim_delay").set_value(3).run()
    assert not app.exception
    assert any("参数已改变" in info.value for info in app.info)
    assert app.session_state["factor_simulation"]["fingerprint"] == initial["fingerprint"]
    app.button(key="sim_run").click().run()
    assert not app.exception
    assert app.session_state["factor_simulation"]["result"].settings.delay == 3
    assert not any("参数已改变" in info.value for info in app.info)
    app.text_input(key="factor_sim_research_run_name").set_value("delay-three").run()
    app.button(key="factor_sim_save_research_run").click().run()
    assert not app.exception
    assert any("已保存研究运行" in success.value for success in app.success)
    saved, errors = store.list_runs(project.id)
    assert not errors
    assert len(saved) == 1
    assert saved[0].config["single_factor_simulation"]["delay"] == 3
    assert "Fitness" in saved[0].summary


def test_invalid_range_keeps_previous_result(panel, tmp_path):
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture_panel"] = panel
    app.session_state["store_root"] = str(tmp_path)
    app.session_state["test_factor"] = "mom_5"
    app.run().button(key="sim_run").click().run()
    assert not app.exception
    old = app.session_state["factor_simulation"]["fingerprint"]
    app.date_input(key="sim_start").set_value(panel.close.index[-1].date()).run()
    app.button(key="sim_run").click().run()
    assert not app.exception
    assert any("开始日期" in error.value for error in app.error)
    assert app.session_state["factor_simulation"]["fingerprint"] == old
