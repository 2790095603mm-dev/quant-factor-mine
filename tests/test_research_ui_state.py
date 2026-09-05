"""研究项目页面的无 Streamlit 状态逻辑。"""

from __future__ import annotations

import pandas as pd

from qfm.research.ui_state import can_save_strategy_run, set_active_project


def test_active_project_selection_is_preserved():
    state: dict[str, object] = {"active_project_id": None}

    set_active_project(state, "project_123")

    assert state["active_project_id"] == "project_123"


def test_save_disabled_without_project_or_result():
    assert can_save_strategy_run(active_project_id=None, payload={}) is False
    payload = {"nav": pd.Series([1.0])}
    assert can_save_strategy_run(active_project_id="project_123", payload=payload) is True
