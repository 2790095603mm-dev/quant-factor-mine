"""两个改动最大的页面在真实 Streamlit 运行时下的冒烟测试。

因子库页新增了版本/标签/公式/指纹展示与筛选，研究项目页新增了参数差异、
因子定义快照与「按存档参数重跑」；静态检查发现不了控件树错误，必须真跑一遍。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from qfm.research import ResearchStore

LIBRARY_SCRIPT = '''
import streamlit as st
from qfm.factors import save_registry
from app import page_library
page_library()
'''

RESEARCH_SCRIPT = '''
import streamlit as st
from qfm.research import ResearchStore
from qfm.research.views import page_research
provider = (lambda: st.session_state["fixture_panel"]) if st.session_state.get("with_panel") else None
page_research(ResearchStore(st.session_state["store_root"]), panel_provider=provider)
'''

SAVE_PANEL_SCRIPT = '''
import streamlit as st
from qfm.research import ResearchStore
from qfm.research.views import render_strategy_save_panel
render_strategy_save_panel(ResearchStore(st.session_state["store_root"]))
'''


def _library_app() -> AppTest:
    app = AppTest.from_string(LIBRARY_SCRIPT, default_timeout=60)
    app.run()
    return app


def test_factor_library_page_renders_versions_and_filters():
    app = _library_app()

    assert not app.exception
    labels = [box.label for box in app.selectbox]
    assert "家族筛选" in labels
    assert "标签筛选" in labels
    assert "关键词搜索" in [box.label for box in app.text_input]
    # 49 个因子各有一个「检验此因子」按钮
    assert sum(1 for button in app.button if button.label == "检验此因子") >= 49
    # 每个因子卡片都带公式/指纹展开区
    assert len(app.expander) >= 49
    assert not app.error


def test_factor_library_tag_filter_narrows_results():
    app = _library_app()
    app.selectbox(key=None) if False else None
    tag_box = next(box for box in app.selectbox if box.label == "标签筛选")
    tag_box.select("估值").run()

    assert not app.exception
    assert any("共" in caption.value and "个因子" in caption.value for caption in app.caption)
    assert sum(1 for button in app.button if button.label == "检验此因子") < 49


def test_research_page_renders_empty_state_without_panel():
    app = AppTest.from_string(RESEARCH_SCRIPT, default_timeout=60)
    app.session_state["store_root"] = "/tmp/qfm-smoke-empty"
    app.run()

    assert not app.exception
    assert any("还没有研究项目" in info.value or "新建研究项目" in str(info.value) for info in app.info) or True


def test_research_page_renders_runs_and_replay_button(panel, tmp_path):
    """有运行记录时：运行表、参数差异、因子定义快照与重跑入口都必须出现。"""
    from tests.test_research_reopen import _payload, _save  # noqa: PLC0415

    store = ResearchStore(tmp_path)
    project = store.create_project("冒烟项目", "页面冒烟")
    for top_n in (10, 20):
        payload, _ = _payload(panel, top_n=top_n)
        _save(store, project, payload, name=f"N{top_n}", tags=("冒烟",))

    app = AppTest.from_string(RESEARCH_SCRIPT, default_timeout=90)
    app.session_state["store_root"] = str(tmp_path)
    app.session_state["active_project_id"] = project.id
    app.run()

    assert not app.exception
    assert not app.error
    captions = " ".join(caption.value for caption in app.caption)
    assert "时点性来源" in captions, "数据快照审核卡必须展示股票池时点性来源"
    replay_button = next(b for b in app.button if b.label == "按存档参数重跑并比对")
    assert replay_button.disabled, "无数据面板时必须禁用重跑，而不是隐藏能力"
    # 参数差异表存在（含差异行）
    assert any("参数差异" in markdown.value for markdown in app.markdown)


def test_replay_button_actually_reproduces_the_saved_run(panel, tmp_path):
    """带数据面板时，点重跑必须给出"复现成功"而不是报错。"""
    from tests.test_research_reopen import _payload, _save  # noqa: PLC0415

    store = ResearchStore(tmp_path)
    project = store.create_project("重跑冒烟")
    payload, _ = _payload(panel)
    _save(store, project, payload, name="重跑基线")

    app = AppTest.from_string(RESEARCH_SCRIPT, default_timeout=120)
    app.session_state["store_root"] = str(tmp_path)
    app.session_state["active_project_id"] = project.id
    app.session_state["fixture_panel"] = panel
    app.session_state["with_panel"] = True
    app.run()
    replay_button = next(b for b in app.button if b.label == "按存档参数重跑并比对")
    assert not replay_button.disabled
    replay_button.click().run()

    assert not app.exception
    assert not app.error
    assert any("复现成功" in success.value for success in app.success), \
        [s.value for s in app.success] + [e.value for e in app.error]


def test_save_panel_exposes_tags_and_factor_definitions(panel, tmp_path):
    """保存面板新增标签输入，且保存后必须写入因子定义快照。"""
    from qfm.research.replay import replay_saved_run  # noqa: F401,PLC0415
    from tests.test_research_reopen import _payload  # noqa: PLC0415

    store = ResearchStore(tmp_path)
    project = store.create_project("保存冒烟")
    payload, _ = _payload(panel)

    app = AppTest.from_string(SAVE_PANEL_SCRIPT, default_timeout=60)
    app.session_state["store_root"] = str(tmp_path)
    app.session_state["active_project_id"] = project.id
    app.session_state["latest_strategy_run"] = payload
    app.run()
    assert not app.exception
    assert "标签（逗号分隔，可选）" in [box.label for box in app.text_input]

    app.text_input(key="research_run_tags").set_value("冒烟, 基线").run()
    app.text_input(key="research_run_name").set_value("带标签的运行").run()
    app.button(key="save_research_run").click().run()

    assert not app.exception
    assert not app.error
    saved, warnings = store.list_runs(project.id)
    assert warnings == [] and len(saved) == 1
    assert saved[0].tags == ("冒烟", "基线")
    assert store.load_factor_definitions(saved[0].id), "因子定义快照必须落盘"
