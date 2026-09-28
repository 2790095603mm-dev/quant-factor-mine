"""台账类工具：查询历史实验。

复用 `qfm.research.store.ResearchStore`，但**不改动它的落盘格式**。
仓库现有的台账查询只有 `list_runs` / `list_all_runs`，两者都是线性扫目录 +
全量 JSON 解析，没有按指标过滤的能力。Agent 需要「找出历史上 IC 最高的实验」
这类查询，所以在工具层做一层轻量筛选，而不是去改 store 的存储结构 ——
既有台账的向后兼容性比查询便利性更重要。
"""

from __future__ import annotations

from qfm.agent.models import ToolKind
from qfm.agent.registry import ToolContext

__all__ = ["register_ledger_tools"]


def register_ledger_tools(registry) -> None:
    """把台账类工具注册进 `registry`。"""

    @registry.register(
        "list_experiments",
        kind=ToolKind.READ,
        description="列出已保存的历史实验（研究台账），可按指标排序或过滤，用于复用此前的研究结论",
        parameters={
            "limit": "int，最多返回多少条（默认 10）",
            "sort_by": "str，按哪个指标排序，如 夏普比率 / ic_mean / 年化收益",
            "min_value": "float，该指标的下限过滤",
        },
        required=(),
        returns="payload: {n_total, n_returned, available, experiments:[{run_id, project, name, created_at, status, metrics}]}",
        cost=0.8,
        tags=("ledger",),
    )
    def list_experiments(
        context: ToolContext,
        limit: int = 10,
        sort_by: str | None = None,
        min_value: float | None = None,
    ):
        store = context.store
        if store is None:
            return {
                "available": False,
                "n_total": 0,
                "n_returned": 0,
                "experiments": [],
                "note": "本次运行未挂载研究台账（ResearchStore），无法查询历史实验",
            }, "未挂载研究台账，跳过历史查询"

        pairs = store.list_all_runs()
        rows = []
        for project, run in pairs:
            summary = dict(run.summary or {})
            rows.append({
                "run_id": run.id,
                "project": project.name,
                "name": run.name,
                "created_at": run.created_at,
                "status": run.status,
                "tags": list(run.tags),
                "metrics": {
                    key: value
                    for key, value in summary.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                },
            })

        if sort_by:
            if not any(sort_by in row["metrics"] for row in rows):
                available = sorted({key for row in rows for key in row["metrics"]})
                return {
                    "available": True,
                    "n_total": len(rows),
                    "n_returned": 0,
                    "experiments": [],
                    "note": f"没有实验记录指标 {sort_by!r}；可用指标: {available}",
                }, f"没有实验包含指标 {sort_by}"
            rows.sort(
                key=lambda row: row["metrics"].get(sort_by, float("-inf")), reverse=True
            )
        if min_value is not None and sort_by:
            rows = [row for row in rows if row["metrics"].get(sort_by, float("-inf")) >= min_value]

        total = len(rows)
        shown = rows[: max(1, int(limit))]
        payload = {
            "available": True,
            "n_total": total,
            "n_returned": len(shown),
            "sort_by": sort_by,
            "experiments": shown,
        }
        summary = (
            f"台账共 {total} 条实验记录"
            + (f"，按 {sort_by} 排序" if sort_by else "")
            + (f"；首条 {shown[0]['name']}" if shown else "；无匹配记录")
        )
        return payload, summary
