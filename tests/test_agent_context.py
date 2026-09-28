"""上下文管理测试：摘要代替原文、超限外置、超预算折叠最旧、置顶豁免。"""

from __future__ import annotations

import json

import pytest

from qfm.agent.context import ContextBudget, ContextManager


def _manager(tmp_path, **budget):
    settings = {"max_chars": 600, "max_item_chars": 200, "keep_recent": 2}
    settings.update(budget)
    return ContextManager(ContextBudget(**settings), tmp_path / "artifacts")


# ---------------------------------------------------------------------------
# 基本读写
# ---------------------------------------------------------------------------
def test_add_and_render_keeps_order(tmp_path):
    ctx = _manager(tmp_path)
    ctx.add("user", "请求", "分析银行股")
    ctx.add("tool", "第一步", "取数完成")
    rendered = ctx.render()
    assert rendered.index("请求") < rendered.index("第一步")
    assert len(ctx) == 2


def test_render_messages_maps_roles(tmp_path):
    ctx = _manager(tmp_path)
    ctx.add("user", "请求", "问题")
    ctx.add("plan", "计划", "步骤")
    ctx.add("tool", "工具", "结果")
    messages = ctx.render_messages()
    assert [item["role"] for item in messages] == ["user", "assistant", "user"]


def test_digest_marks_pinned_and_offloaded(tmp_path):
    ctx = _manager(tmp_path)
    ctx.add("user", "请求", "问题", pinned=True)
    ctx.add("tool", "大结果", "x" * 500)
    digest = ctx.digest()
    assert "📌" in digest and "🗂" in digest


# ---------------------------------------------------------------------------
# 超长内容外置
# ---------------------------------------------------------------------------
def test_long_content_is_offloaded_to_artifact(tmp_path):
    ctx = _manager(tmp_path)
    long_text = "A" * 5000
    item = ctx.add("tool", "巨长结果", long_text)
    assert item.offloaded is True
    # 上下文里只留截断后的摘要
    assert item.chars < len(long_text)
    assert "已截断" in item.content
    # 完整内容落盘，事后可复盘
    assert item.artifact
    assert long_text in open(item.artifact, encoding="utf-8").read()


def test_artifact_payload_is_written_as_json_separately_from_summary(tmp_path):
    """摘要进上下文、完整结构落盘——这是重对象不撑爆上下文的关键。"""
    ctx = _manager(tmp_path)
    payload = {"ic": 0.034, "rows": list(range(500))}
    item = ctx.add(
        "tool",
        "run_experiment",
        "IC +0.034，命中缓存",
        artifact_payload=payload,
        offload_name="experiment.json",
    )
    assert "IC +0.034" in item.content
    assert item.chars < 100
    saved = json.loads(open(item.artifact, encoding="utf-8").read())
    assert saved["rows"] == list(range(500))


def test_short_content_stays_inline(tmp_path):
    ctx = _manager(tmp_path)
    item = ctx.add("tool", "小结果", "42 只股票")
    assert item.offloaded is False
    assert item.artifact == ""
    assert item.content == "42 只股票"


def test_dataframe_payload_is_summarised_not_serialised(tmp_path):
    """DataFrame 不该被逐值塞进 artifact，只留形状与列名。"""
    import pandas as pd

    ctx = _manager(tmp_path)
    frame = pd.DataFrame({"a": range(10), "b": range(10)})
    item = ctx.add("tool", "矩阵", "矩阵已计算", artifact_payload={"frame": frame},
                   offload_name="frame.json")
    saved = json.loads(open(item.artifact, encoding="utf-8").read())
    assert saved["frame"]["shape"] == [10, 2]


# ---------------------------------------------------------------------------
# 折叠
# ---------------------------------------------------------------------------
def test_compact_folds_oldest_first(tmp_path):
    ctx = _manager(tmp_path, max_chars=700, keep_recent=1)
    ctx.add("observation", "第一步", "x" * 200)
    ctx.add("observation", "第二步", "y" * 200)
    ctx.add("observation", "第三步", "z" * 200)
    ctx.add("observation", "第四步", "w" * 200)
    stats = ctx.compact()
    assert stats.folded_items >= 1
    assert stats.chars <= 700
    titles = [item.title for item in ctx.items]
    folded = [item.title for item in ctx.items if item.folded]
    assert "第一步" in folded, "应优先折叠最旧的条目"
    assert "第四步" not in folded, "最新条目应保留"
    assert titles == ["第一步", "第二步", "第三步", "第四步"], "折叠不应改变条目顺序"


def test_folded_item_keeps_a_placeholder_not_silence(tmp_path):
    """折叠成占位说明而不是删除：否则模型会以为这一步从未发生。

    注意尺寸：条目必须小于 max_item_chars（否则会先被外置截断，总长降下来就不再
    触发折叠）。这里单条 200 字符 < 400，三条合计 600 > max_chars=500。
    """
    ctx = _manager(tmp_path, max_chars=500, max_item_chars=400, keep_recent=1)
    ctx.add("tool", "旧步骤", "x" * 200)
    ctx.add("tool", "中步骤", "y" * 200)
    ctx.add("tool", "新步骤", "z" * 200)
    ctx.compact()
    folded = [item for item in ctx.items if item.folded][0]
    assert "被折叠" in folded.content
    assert f"{folded.original_chars} 字符" in folded.content


def test_pinned_items_are_never_folded(tmp_path):
    ctx = _manager(tmp_path, max_chars=500, max_item_chars=400, keep_recent=1)
    ctx.add("user", "研究请求", "x" * 200, pinned=True)
    ctx.add("tool", "步骤一", "y" * 200)
    ctx.add("tool", "步骤二", "z" * 200)
    ctx.compact()
    pinned = [item for item in ctx.items if item.pinned][0]
    assert pinned.folded is False
    assert pinned.content == "x" * 200
    assert any(item.folded for item in ctx.items), "非置顶的旧条目仍应被折叠"


def test_keep_pinned_can_be_disabled(tmp_path):
    ctx = _manager(tmp_path, max_chars=500, max_item_chars=400, keep_recent=1, keep_pinned=False)
    ctx.add("user", "请求", "x" * 200, pinned=True)
    ctx.add("tool", "步骤", "y" * 200)
    ctx.add("tool", "步骤2", "z" * 200)
    ctx.compact()
    # keep_pinned=False 时置顶条目也参与折叠
    assert any(item.pinned and item.folded for item in ctx.items)


def test_compact_is_idempotent(tmp_path):
    ctx = _manager(tmp_path, max_chars=600, keep_recent=1)
    for index in range(5):
        ctx.add("tool", f"步骤{index}", "x" * 150)
    first = ctx.compact().to_dict()
    second = ctx.compact().to_dict()
    assert first["chars"] == second["chars"]
    assert first["folded_items"] == second["folded_items"]


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def test_stats_report_compression_ratio(tmp_path):
    ctx = _manager(tmp_path)
    ctx.add("tool", "大块", "x" * 3000)
    ctx.add("tool", "小块", "ok")
    stats = ctx.compact()
    assert stats.original_chars > stats.chars
    assert 0 < stats.compression_ratio < 1
    assert stats.offloaded_items == 1
    assert stats.budget["max_chars"] == 600


def test_stats_serialise_into_run_record(tmp_path):
    ctx = _manager(tmp_path)
    ctx.add("tool", "步骤", "结果")
    payload = ctx.compact().to_dict()
    assert set(payload) >= {
        "items", "chars", "original_chars", "offloaded_items",
        "folded_items", "compression_ratio", "budget",
    }


def test_non_string_content_is_compactified(tmp_path):
    ctx = _manager(tmp_path)
    item = ctx.add("tool", "字典结果", {"a": 1, "b": [1, 2, 3]})
    assert "a" in item.content and '"a"' in item.content


# ---------------------------------------------------------------------------
# 预算校验
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_chars": 100}, "max_chars"),
        ({"max_item_chars": 10}, "max_item_chars"),
        ({"keep_recent": -1}, "keep_recent"),
    ],
)
def test_invalid_budget_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ContextBudget(**kwargs)


def test_render_respects_explicit_limit(tmp_path):
    ctx = _manager(tmp_path, max_chars=2000)
    for index in range(20):
        ctx.add("tool", f"步骤{index:02d}", "内容" * 20)
    rendered = ctx.render(max_chars=300)
    assert len(rendered) <= 300 + 200
