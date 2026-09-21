"""因子库版本化测试：定义快照、版本递增、旧版本保留、注册表落盘。

注册新因子的测试必须用 `clean_registry` 夹具复原全局状态，
否则会污染 tests/test_factor_meta.py（它要求每个已注册因子都有中文名）。
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from qfm.factors import (
    FACTORS,
    FACTOR_HISTORY,
    FAMILIES,
    all_tags,
    compute_factor,
    factor_definitions,
    get_factor,
    list_factor_versions,
    list_factors,
    list_families,
    load_registry,
    register_factor,
    save_registry,
)


@pytest.fixture
def clean_registry():
    """快照并复原全局注册表，避免测试用因子泄漏到其它测试。"""
    factors, history = dict(FACTORS), {k: list(v) for k, v in FACTOR_HISTORY.items()}
    yield
    FACTORS.clear()
    FACTORS.update(factors)
    FACTOR_HISTORY.clear()
    FACTOR_HISTORY.update(history)


# ---------- 既有因子自动获得完整元数据 ----------

def test_every_registered_factor_carries_version_metadata():
    """49 个既有因子无需逐个改动，就应自动带上版本/标签/公式/指纹/时间。"""
    factors = list_factors()

    assert len(factors) >= 49
    for factor in factors:
        assert factor.version >= 1
        assert factor.source_hash.startswith("sha256:")
        assert factor.created_at, f"{factor.name} 缺少创建时间"
        assert factor.tags, f"{factor.name} 缺少标签"
        assert factor.formula, f"{factor.name} 缺少公式源码"
        assert factor.family in FAMILIES or factor.family


def test_tags_default_from_family():
    by_name = {f.name: f for f in list_factors()}

    assert "估值" in by_name["bp"].tags
    assert "动量" in by_name["mom_20"].tags
    assert "规模" in by_name["ln_mv_float"].tags
    assert all_tags() == sorted({t for f in list_factors() for t in f.tags})


def test_machine_learning_family_is_declared():
    """ml_synth 运行时注册到「机器学习」家族，该家族必须在 FAMILIES 里（否则排序落到末尾）。"""
    assert "机器学习" in FAMILIES
    assert FAMILIES.index("机器学习") == len(FAMILIES) - 1


def test_list_families_reports_present_families_in_canonical_order():
    families = list_families()

    assert families == [f for f in FAMILIES if f in set(families)]
    assert "价值" in families and "动量反转" in families


def test_formula_excludes_the_decorator_line():
    """公式应展示函数体，而不是 @register_factor(...) 那一行。"""
    formula = {f.name: f for f in list_factors()}["bp"].formula

    assert "register_factor" not in formula
    assert "valuation_price" in formula


def test_batch_registered_factors_record_their_window_parameter():
    """批量注册的家族只共享一份源码，必须靠注册参数让公式自描述。"""
    by_name = {f.name: f for f in list_factors()}

    for name, window in [("mom_5", 5), ("mom_20", 20), ("mom_250", 250)]:
        factor = by_name[name]
        assert factor.params.get("w") == window, f"{name} 未记录窗口参数"
        assert f"w={window}" in factor.formula, f"{name} 的公式缺少窗口参数注释"


def test_standalone_factors_have_no_spurious_params():
    """独立定义的因子不应凭空多出参数（前向 _w 提取不能误伤）。"""
    by_name = {f.name: f for f in list_factors()}

    assert by_name["vol_20"].params == {}
    assert by_name["bp"].params == {}


def test_factor_definitions_snapshot_is_serialisable():
    snapshot = factor_definitions(["bp", "mom_20"])

    assert [item["name"] for item in snapshot] == ["bp", "mom_20"]
    assert json.dumps(snapshot, ensure_ascii=False)  # 必须可 JSON 序列化
    assert set(snapshot[0]) == {
        "name", "family", "description", "direction", "version",
        "tags", "formula", "source_hash", "created_at", "params",
    }


# ---------- 版本递增与旧版本保留 ----------

def test_reimport_same_definition_does_not_create_a_new_version(clean_registry):
    """模块被重复导入/reload 时不应刷版本号。"""

    def body(panel):
        return panel.close

    source = "def body(panel):\n    return panel.close"

    @register_factor("dup_probe", "价值", "探针", source_key=source)
    def first(panel):
        return body(panel)

    assert first is not None
    version_after_first = get_factor("dup_probe").version

    @register_factor("dup_probe", "价值", "探针", source_key=source)
    def second(panel):
        return body(panel)

    assert get_factor("dup_probe").version == version_after_first
    assert len(list_factor_versions("dup_probe")) == 1


def test_modifying_a_factor_appends_a_new_version_and_keeps_the_old(clean_registry):
    """修改因子定义 → 版本 +1，旧版本必须完整保留（绝不静默覆盖）。"""

    @register_factor("ver_probe", "动量反转", "v1", source_key="return close.pct_change(5)")
    def probe_v1(panel):
        return panel.close.pct_change(5)

    original = get_factor("ver_probe")

    @register_factor("ver_probe", "动量反转", "v2", source_key="return close.pct_change(10)")
    def probe_v2(panel):
        return panel.close.pct_change(10)

    updated = get_factor("ver_probe")

    assert original.version == 1
    assert updated.version == 2
    assert updated.description == "v2"
    assert get_factor("ver_probe").version == 2
    # 旧版本仍可取回，且保留各自的描述与指纹
    assert get_factor("ver_probe", version=1).description == "v1"
    assert get_factor("ver_probe", version=1).source_hash == original.source_hash
    assert [f.version for f in list_factor_versions("ver_probe")] == [1, 2]
    assert get_factor("ver_probe", version=99) is None


def test_explicit_version_overrides_auto_increment(clean_registry):
    @register_factor("exp_probe", "规模", "显式版本", version=7, source_key="x")
    def probe(panel):
        return panel.close

    assert get_factor("exp_probe").version == 7


def test_runtime_generated_factor_gets_stable_hash(clean_registry):
    """ml_synth 这类运行时 lambda 源码不稳定，必须能用 source_key 固定指纹。"""
    key = "lightgbm:horizon=20:cutoff=2024-06-01:features=49"
    version_holder = {}

    @register_factor("ml_probe", "机器学习", "ML 合成", source_key=key)
    def ml_a(panel):
        return panel.close

    version_holder["v"] = get_factor("ml_probe").version
    digest = get_factor("ml_probe").source_hash

    @register_factor("ml_probe", "机器学习", "ML 合成", source_key=key)
    def ml_b(panel):
        return panel.close

    assert get_factor("ml_probe").version == version_holder["v"]
    assert get_factor("ml_probe").source_hash == digest


def test_registering_named_version_keeps_history_consistent(clean_registry):
    """同一因子多版本共存时，注册表快照必须包含全部历史。"""
    for index, key in enumerate(["a", "b", "c"], start=1):
        register_factor("hist_probe", "价值", f"gen{index}", source_key=key)(
            lambda panel: panel.close
        )

    versions = list_factor_versions("hist_probe")

    assert [f.version for f in versions] == [1, 2, 3]
    assert [f.description for f in versions] == ["gen1", "gen2", "gen3"]


# ---------- 注册表落盘 ----------

def test_registry_round_trip(tmp_path, monkeypatch):
    target = tmp_path / "registry.json"

    written = save_registry(str(target))

    assert written == str(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["count"] == len(list_factors())
    assert {item["name"] for item in payload["factors"]} == {f.name for f in list_factors()}
    assert set(payload["history"]) >= {"mom_20", "bp"}
    assert all("func" not in item for item in payload["factors"]), "函数对象不可序列化，必须排除"
    assert load_registry(str(target))["count"] == payload["count"]


def test_load_registry_returns_empty_structure_when_missing(tmp_path):
    assert load_registry(str(tmp_path / "absent.json")) == {"factors": [], "history": {}}


def test_load_registry_survives_corrupt_file(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")

    assert load_registry(str(broken)) == {"factors": [], "history": {}}


def test_registry_path_honours_environment(monkeypatch, tmp_path):
    from qfm.factors.base import registry_path

    monkeypatch.setenv("QFM_FACTOR_REGISTRY", str(tmp_path / "custom.json"))

    assert registry_path() == str(tmp_path / "custom.json")


# ---------- 与计算路径的兼容 ----------

def test_compute_factor_works_for_a_versioned_factor(clean_registry, panel):
    @register_factor("compute_probe", "波动", "探针", source_key="probe")
    def probe(d):
        return d.close.pct_change(3)

    result = compute_factor("compute_probe", panel)

    assert result.shape == panel.close.shape
    assert result.notna().any().any()


def test_compute_factor_reports_unknown_names(panel):
    with pytest.raises(KeyError, match="未知因子"):
        compute_factor("not_a_factor", panel)


def test_historical_version_is_recomputable(clean_registry, panel):
    """归档旧实验时，必须能按记录里的版本号取回当时的因子定义并重算。"""
    register_factor("repro_probe", "价值", "v1", source_key="close")(
        lambda d: d.close
    )
    register_factor("repro_probe", "价值", "v2", source_key="close*2")(
        lambda d: d.close * 2
    )

    v1 = get_factor("repro_probe", version=1)
    v2 = get_factor("repro_probe", version=2)

    pd.testing.assert_frame_equal(v1.func(panel), panel.close)
    pd.testing.assert_frame_equal(v2.func(panel), panel.close * 2)
