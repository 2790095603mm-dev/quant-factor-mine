"""研究项目页面可独立测试的轻量状态逻辑。"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pandas as pd


def set_active_project(state: MutableMapping[str, object], project_id: str | None) -> None:
    """在任意兼容映射中更新当前选择的研究项目。"""
    state["active_project_id"] = project_id


def can_save_strategy_run(active_project_id: str | None, payload: dict[str, Any] | None) -> bool:
    """仅在用户已选项目且存在完整净值结果时允许持久化。"""
    return bool(active_project_id and payload and isinstance(payload.get("nav"), pd.Series))
