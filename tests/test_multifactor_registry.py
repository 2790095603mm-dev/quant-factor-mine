"""综合因子持久化、重载与循环依赖保护。"""

from __future__ import annotations

import pytest

from qfm.factors import FACTORS, FACTOR_HISTORY, compute_factor, get_factor
from qfm.multifactor import CompositeDefinition, CompositeRegistry


@pytest.fixture
def clean_factor_registry():
    factors = dict(FACTORS)
    history = {name: list(items) for name, items in FACTOR_HISTORY.items()}
    yield
    FACTORS.clear()
    FACTORS.update(factors)
    FACTOR_HISTORY.clear()
    FACTOR_HISTORY.update(history)


def test_saved_composite_reloads_into_factor_library(panel, tmp_path, clean_factor_registry):
    registry = CompositeRegistry(tmp_path)
    definition = CompositeDefinition.create(
        "value_quality_combo", ["ep_ttm", "roe"], description="价值质量综合因子"
    )

    saved = registry.save(definition)
    FACTORS.pop(saved.name, None)
    FACTOR_HISTORY.pop(saved.name, None)
    warnings = registry.register_all()

    assert warnings == []
    assert get_factor(saved.name).family == "多因子"
    result = compute_factor(saved.name, panel)
    assert result.shape == panel.close.shape
    assert result.notna().any().any()


def test_registry_preserves_component_versions(tmp_path):
    definition = CompositeDefinition.create("versioned_combo", ["bp", "roe"], mode="icir")

    saved = CompositeRegistry(tmp_path).save(definition)

    assert saved.version == 1
    assert saved.mode == "icir"
    assert all(item["version"] >= 1 for item in saved.components)
    assert all(item["source_hash"].startswith("sha256:") for item in saved.components)


def test_registry_rejects_direct_cycle(tmp_path):
    registry = CompositeRegistry(tmp_path)

    with pytest.raises(ValueError, match="循环"):
        registry.save(CompositeDefinition.create("cycle_a", ["cycle_a"], validate_components=False))


def test_registry_rejects_indirect_cycle(tmp_path, clean_factor_registry):
    registry = CompositeRegistry(tmp_path)
    registry.save(CompositeDefinition.create("cycle_a", ["mom_20"]))
    registry.register_all()
    registry.save(CompositeDefinition.create("cycle_b", ["cycle_a"]))
    registry.register_all()

    with pytest.raises(ValueError, match="循环"):
        registry.save(
            CompositeDefinition.create("cycle_a", ["cycle_b"], validate_components=False)
        )


def test_same_definition_is_idempotent_and_changed_definition_versions(tmp_path):
    registry = CompositeRegistry(tmp_path)
    first = registry.save(CompositeDefinition.create("combo_version", ["bp", "roe"]))
    same = registry.save(CompositeDefinition.create("combo_version", ["bp", "roe"]))
    changed = registry.save(CompositeDefinition.create("combo_version", ["bp", "mom_20"]))

    assert same.version == first.version == 1
    assert changed.version == 2
    assert [item.version for item in registry.list_definitions("combo_version")] == [1, 2]
