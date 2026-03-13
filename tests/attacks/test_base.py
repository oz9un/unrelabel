import numpy as np
import pytest
from unrelabel.attacks.base import AttackResult, BaseAttack


def test_attack_result_accuracy_drop():
    result = AttackResult(
        attack_type="label_flip",
        clean_accuracy=0.95,
        poisoned_accuracy=0.70,
        accuracy_drop=0.25,
        vulnerability_score=62.5,
        confusion_matrices={"clean": [], "poisoned": []},
        plots=[],
        config={"poison_rate": 0.2},
        timestamp="2026-03-09T00:00:00",
    )
    assert result.accuracy_drop == pytest.approx(0.25)
    assert result.vulnerability_score == pytest.approx(62.5)


def test_attack_result_to_dict():
    result = AttackResult(
        attack_type="label_flip",
        clean_accuracy=0.95,
        poisoned_accuracy=0.80,
        accuracy_drop=0.15,
        vulnerability_score=37.5,
        confusion_matrices={},
        plots=[],
        config={},
        timestamp="2026-03-09T00:00:00",
    )
    d = result.to_dict()
    assert d["attack_type"] == "label_flip"
    assert "clean_accuracy" in d
    assert "vulnerability_score" in d


def test_base_attack_is_abstract():
    with pytest.raises(TypeError):
        BaseAttack()
