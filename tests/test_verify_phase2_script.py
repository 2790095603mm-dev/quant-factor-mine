"""第二阶段离线验证脚本。"""

from scripts import verify_phase2


def test_phase2_verifier_passes(capsys):
    assert verify_phase2.main([]) == 0
    output = capsys.readouterr().out
    assert "0 项失败" in output
    for label in ("Job/Cache", "Dataset/Universe", "Factor Compare", "Multi-Factor", "Strategy Compare"):
        assert label in output
