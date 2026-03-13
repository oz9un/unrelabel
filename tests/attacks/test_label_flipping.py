import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from unrelabel.attacks.label_flipping import LabelFlippingAttack, flip_labels
from unrelabel.attacks.base import AttackResult
from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.model_loader import ModelWrapper


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


# --- flip_labels unit tests ---

def test_flip_labels_correct_count():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_p, idx = flip_labels(y, poison_rate=0.3, seed=0)
    assert len(idx) == 3
    assert len(y_p) == len(y)


def test_flip_labels_binary_actually_flips():
    # Use target_class when only one class exists, since random flipping is undefined
    y = np.zeros(10, dtype=int)
    y_p, idx = flip_labels(y, poison_rate=0.5, seed=0, target_class=1)
    assert all(y_p[idx] == 1)
    assert all(y_p[~np.isin(np.arange(len(y)), idx)] == 0)


def test_flip_labels_multiclass_random():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_p, idx = flip_labels(y, poison_rate=0.4, seed=42)
    for i in idx:
        assert y_p[i] != y[i]


def test_flip_labels_targeted_source_target():
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0])
    y_p, idx = flip_labels(y, poison_rate=1.0, seed=0, source_class=0, target_class=1)
    source_indices = np.where(y == 0)[0]
    assert set(idx) == set(source_indices)
    assert all(y_p[source_indices] == 1)


def test_flip_labels_zero_rate_returns_copy():
    y = np.array([0, 1, 0, 1])
    y_p, idx = flip_labels(y, poison_rate=0.0, seed=0)
    assert len(idx) == 0
    assert np.array_equal(y, y_p)


def test_flip_labels_invalid_rate():
    with pytest.raises(ValueError):
        flip_labels(np.array([0, 1]), poison_rate=1.5)


# --- LabelFlippingAttack integration tests ---

def test_attack_run_returns_result(binary_dataset, sklearn_wrapper):
    attack = LabelFlippingAttack(poison_rate=0.2, seed=42)
    result = attack.run(binary_dataset, sklearn_wrapper)
    assert isinstance(result, AttackResult)
    assert result.attack_type == "label_flip"
    assert 0.0 <= result.clean_accuracy <= 1.0
    assert 0.0 <= result.poisoned_accuracy <= 1.0
    assert result.accuracy_drop == pytest.approx(
        result.clean_accuracy - result.poisoned_accuracy, abs=1e-6
    )


def test_attack_vulnerability_score_in_range(binary_dataset, sklearn_wrapper):
    attack = LabelFlippingAttack(poison_rate=0.3, seed=42)
    result = attack.run(binary_dataset, sklearn_wrapper)
    assert 0.0 <= result.vulnerability_score <= 100.0


def test_attack_sweep_mode(binary_dataset, sklearn_wrapper):
    attack = LabelFlippingAttack(poison_rates=[0.1, 0.2, 0.3], seed=42)
    result = attack.run(binary_dataset, sklearn_wrapper)
    assert len(result.sweep_results) == 3
    rates = [r["poison_rate"] for r in result.sweep_results]
    assert rates == [0.1, 0.2, 0.3]


def test_attack_summary(binary_dataset, sklearn_wrapper):
    attack = LabelFlippingAttack(poison_rate=0.2, seed=42)
    attack.run(binary_dataset, sklearn_wrapper)
    s = attack.summary()
    assert "attack_type" in s
    assert "clean_accuracy" in s
    assert "vulnerability_score" in s


def test_attack_init_requires_rate():
    with pytest.raises(ValueError, match="poison_rate"):
        LabelFlippingAttack()


def test_attack_summary_before_run_raises():
    attack = LabelFlippingAttack(poison_rate=0.1)
    with pytest.raises(RuntimeError):
        attack.summary()


def test_flip_labels_single_class_without_target_raises():
    y = np.array([2, 2, 2, 2, 2])
    with pytest.raises(ValueError, match="one unique class"):
        flip_labels(y, poison_rate=0.5, seed=0)


def test_flip_labels_same_source_target_raises():
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="source_class and target_class must differ"):
        flip_labels(y, poison_rate=0.5, source_class=0, target_class=0)
