import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from unrelabel.attacks.targeted_label import TargetedLabelAttack, targeted_flip_labels
from unrelabel.attacks.base import AttackResult
from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.model_loader import ModelWrapper


def test_targeted_flip_correct_count():
    """Poison rate applied to source_class only, not all samples."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])  # 4 of class 0, 4 of class 1
    y_p, idx = targeted_flip_labels(y, poison_rate=0.5, source_class=0, target_class=1, seed=0)
    # 50% of 4 class-0 samples = 2 flipped
    assert len(idx) == 2
    assert all(y_p[idx] == 1)  # all flipped go to target_class


def test_targeted_flip_only_source_class_affected():
    """Class 1 samples must remain unchanged regardless of target_class value."""
    y = np.array([0, 0, 0, 1, 1, 1])
    y_p, idx = targeted_flip_labels(y, poison_rate=1.0, source_class=0, target_class=1, seed=0)
    # All class-0 flipped to class-1
    assert all(y_p[:3] == 1)
    # Class-1 samples are untouched (compare against original, not just value)
    assert np.array_equal(y_p[3:], y[3:])


def test_targeted_flip_zero_rate_returns_copy():
    y = np.array([0, 1, 0, 1])
    y_p, idx = targeted_flip_labels(y, poison_rate=0.0, source_class=0, target_class=1, seed=0)
    assert len(idx) == 0
    assert np.array_equal(y, y_p)
    # Mutating the returned array must not corrupt the original
    y_p[0] = 999
    assert y[0] != 999


def test_targeted_flip_invalid_rate():
    with pytest.raises(ValueError, match="poison_rate"):
        targeted_flip_labels(np.array([0, 1]), poison_rate=1.5, source_class=0, target_class=1)


def test_targeted_flip_same_source_target_raises():
    with pytest.raises(ValueError, match="must differ"):
        targeted_flip_labels(np.array([0, 1]), poison_rate=0.5, source_class=0, target_class=0)


def test_targeted_flip_source_class_not_in_y_raises():
    with pytest.raises(ValueError, match="source_class"):
        targeted_flip_labels(np.array([0, 1]), poison_rate=0.5, source_class=2, target_class=1)


def test_targeted_flip_target_class_not_in_y_raises():
    with pytest.raises(ValueError, match="target_class"):
        targeted_flip_labels(np.array([0, 1]), poison_rate=0.5, source_class=0, target_class=5)


def test_targeted_flip_reproducible():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    _, idx1 = targeted_flip_labels(y, poison_rate=0.5, source_class=0, target_class=1, seed=99)
    _, idx2 = targeted_flip_labels(y, poison_rate=0.5, source_class=0, target_class=1, seed=99)
    assert np.array_equal(sorted(idx1), sorted(idx2))


@pytest.fixture
def binary_dataset():
    X, y = make_classification(n_samples=300, n_features=4, random_state=42)
    return PoisonDataset(
        X_train=X[:210], X_test=X[210:],
        y_train=y[:210], y_test=y[210:],
        class_names=["0", "1"],
        feature_names=["f1", "f2", "f3", "f4"],
        source="test",
        metadata={},
    )


@pytest.fixture
def sklearn_wrapper():
    return ModelWrapper(LogisticRegression(random_state=42), backend="sklearn")


def test_attack_run_returns_result(binary_dataset, sklearn_wrapper):
    attack = TargetedLabelAttack(source_class=0, target_class=1, poison_rate=0.3, seed=42)
    result = attack.run(binary_dataset, sklearn_wrapper)
    assert isinstance(result, AttackResult)
    assert result.attack_type == "targeted_label"
    assert 0.0 <= result.clean_accuracy <= 1.0
    assert 0.0 <= result.poisoned_accuracy <= 1.0
    assert 0.0 <= result.vulnerability_score <= 100.0


def test_attack_result_has_tmr(binary_dataset, sklearn_wrapper):
    """targeted_misclassification_rate must be in config and sweep_results."""
    attack = TargetedLabelAttack(source_class=0, target_class=1, poison_rate=0.5, seed=42)
    result = attack.run(binary_dataset, sklearn_wrapper)
    assert "targeted_misclassification_rate" in result.config
    tmr = result.config["targeted_misclassification_rate"]
    assert 0.0 <= tmr <= 1.0
    assert "targeted_misclassification_rate" in result.sweep_results[0]


def test_attack_sweep_mode(binary_dataset, sklearn_wrapper):
    attack = TargetedLabelAttack(
        source_class=0, target_class=1, poison_rates=[0.1, 0.3, 0.5], seed=42
    )
    result = attack.run(binary_dataset, sklearn_wrapper)
    assert len(result.sweep_results) == 3
    rates = [r["poison_rate"] for r in result.sweep_results]
    assert rates == [0.1, 0.3, 0.5]


def test_attack_summary(binary_dataset, sklearn_wrapper):
    attack = TargetedLabelAttack(source_class=0, target_class=1, poison_rate=0.3, seed=42)
    attack.run(binary_dataset, sklearn_wrapper)
    s = attack.summary()
    assert "attack_type" in s
    assert "targeted_misclassification_rate" in s
    assert "vulnerability_score" in s


def test_attack_init_requires_rate():
    with pytest.raises(ValueError, match="poison_rate"):
        TargetedLabelAttack(source_class=0, target_class=1)


def test_attack_init_same_classes_raises():
    with pytest.raises(ValueError, match="must differ"):
        TargetedLabelAttack(source_class=1, target_class=1, poison_rate=0.3)


def test_attack_summary_before_run_raises():
    attack = TargetedLabelAttack(source_class=0, target_class=1, poison_rate=0.3)
    with pytest.raises(RuntimeError):
        attack.summary()


def test_vulnerability_score_high_at_high_tmr(binary_dataset, sklearn_wrapper):
    """At high poison rate, TMR should be significant and score > 0."""
    attack = TargetedLabelAttack(source_class=0, target_class=1, poison_rate=0.9, seed=42)
    result = attack.run(binary_dataset, sklearn_wrapper)
    assert result.vulnerability_score >= 10.0  # 90% poison on separable data must yield meaningful score
