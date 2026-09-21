"""Phase 2 新页面的真实 Streamlit 控件冒烟测试。"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest


FACTOR_COMPARE_SCRIPT = '''
import streamlit as st
from qfm.analysis.factor_views import render_factor_compare
from qfm.jobs import JobService
render_factor_compare(
    st.session_state["fixture_panel"], "test_pool",
    JobService(st.session_state["job_root"]), st.session_state["catalog_root"],
)
'''

MULTIFACTOR_SCRIPT = '''
import streamlit as st
from qfm.jobs import JobService
from qfm.multifactor import CompositeRegistry
from qfm.multifactor.views import render_multifactor_lab
render_multifactor_lab(
    st.session_state["fixture_panel"], "test_pool",
    JobService(st.session_state["job_root"]),
    CompositeRegistry(st.session_state["composite_root"]),
    st.session_state["catalog_root"],
)
'''

CATALOG_SCRIPT = '''
import streamlit as st
from qfm.data import DataCatalog
from qfm.data.catalog_views import render_data_catalog
render_data_catalog(DataCatalog(st.session_state["catalog_root"]))
'''

JOB_SCRIPT = '''
import streamlit as st
from qfm.jobs import JobService
from qfm.jobs.views import render_job_center
render_job_center(JobService(st.session_state["job_root"]).store)
'''


def test_phase2_navigation_is_present():
    app = AppTest.from_file("app.py", default_timeout=90).run()

    assert not app.exception
    navigation = app.radio(key="section")
    assert {"因子对比", "多因子实验室", "策略对比", "数据与股票池", "任务中心"} <= set(navigation.options)


def test_factor_compare_page_runs_and_records_both_job_types(panel, tmp_path):
    app = AppTest.from_string(FACTOR_COMPARE_SCRIPT, default_timeout=90)
    app.session_state["fixture_panel"] = panel
    app.session_state["job_root"] = str(tmp_path / "jobs")
    app.session_state["catalog_root"] = str(tmp_path / "catalog")
    app.run().button(key="factor_compare_run").click().run()

    assert not app.exception
    assert not app.error
    assert "IC" in [column for frame in app.dataframe for column in frame.value.columns]
    from qfm.jobs import JobStore
    assert {job.job_type for job in JobStore(tmp_path / "jobs").list_jobs()} == {
        "FACTOR_COMPUTE", "FACTOR_ANALYSIS"
    }


def test_multifactor_page_runs_and_saves_composite(panel, tmp_path):
    app = AppTest.from_string(MULTIFACTOR_SCRIPT, default_timeout=90)
    app.session_state["fixture_panel"] = panel
    app.session_state["job_root"] = str(tmp_path / "jobs")
    app.session_state["catalog_root"] = str(tmp_path / "catalog")
    app.session_state["composite_root"] = str(tmp_path / "factors")
    app.run().button(key="multifactor_run").click().run()

    assert not app.exception
    assert not app.error
    app.text_input(key="multifactor_save_name").set_value("ui_composite").run()
    app.text_input(key="multifactor_save_desc").set_value("UI 冒烟综合因子").run()
    app.button(key="multifactor_save").click().run()

    assert not app.exception
    assert any("已保存 ui_composite" in item.value for item in app.success)


def test_catalog_and_job_center_render_empty_states(tmp_path):
    catalog = AppTest.from_string(CATALOG_SCRIPT, default_timeout=30)
    catalog.session_state["catalog_root"] = str(tmp_path / "catalog")
    catalog.run()
    assert not catalog.exception
    assert {metric.label for metric in catalog.metric} >= {"Dataset 版本", "已登记股票池版本"}

    jobs = AppTest.from_string(JOB_SCRIPT, default_timeout=30)
    jobs.session_state["job_root"] = str(tmp_path / "jobs")
    jobs.run()
    assert not jobs.exception
    assert {metric.label for metric in jobs.metric} == {"PENDING", "RUNNING", "SUCCESS", "FAILED"}
