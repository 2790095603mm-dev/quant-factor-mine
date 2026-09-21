"""把第一阶段验证脚本本身纳入回归：离线模式下必须全项通过。

脚本是给人和 CI 都能跑的端到端检查，如果它自己坏了，"验证"就失去了意义。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "verify_phase1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_phase1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verifier():
    return _load_module()


def test_script_exists_and_is_importable(verifier):
    assert hasattr(verifier, "main")
    assert hasattr(verifier, "Checker")


def test_offline_verification_passes_end_to_end(verifier, capsys):
    """合成数据下 6 个环节必须全部通过（零网络、秒级）。"""
    exit_code = verifier.main([])
    output = capsys.readouterr().out

    assert exit_code == 0, f"验证脚本未通过：\n{output}"
    assert "0 项失败" in output
    for stage in ("【1/6】", "【2/6】", "【3/6】", "【4/6】", "【5/6】", "【6/6】"):
        assert stage in output


def test_checker_collects_failures_without_short_circuiting(verifier):
    """检查器必须跑完全部断言再汇报，而不是遇到第一个失败就中断。"""
    checker = verifier.Checker()

    assert checker.check("通过项", True) is True
    assert checker.check("失败项", False) is False
    assert checker.check("后续项仍会执行", True) is True
    assert checker.passed == ["通过项", "后续项仍会执行"]
    assert checker.failed == ["失败项"]
