import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, make_blobs

from unrelabel.attacks.clean_label import select_target_point, perturb_neighbors


@pytest.fixture
def fitted_binary_lr():
    """LogisticRegression fitted on a clean binary dataset."""
    X, y = make_classification(
        n_samples=400, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    lr = LogisticRegression(random_state=42)
    lr.fit(X, y)
    return lr, X, y


@pytest.fixture
def fitted_multiclass_lr():
    """LogisticRegression (OvR) fitted on a clean 3-class dataset."""
    X, y = make_blobs(n_samples=600, centers=3, n_features=2, random_state=42)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)
    return lr, X, y


# --- select_target_point ---

def test_select_target_point_returns_valid_index(fitted_binary_lr):
    lr, X, y = fitted_binary_lr
    X_target, idx = select_target_point(X, y, lr, source_class=0, target_class=1)
    assert 0 <= idx < len(X)
    assert np.array_equal(X_target, X[idx])


def test_select_target_point_picks_target_class(fitted_binary_lr):
    lr, X, y = fitted_binary_lr
    _, idx = select_target_point(X, y, lr, source_class=0, target_class=1)
    assert y[idx] == 1  # Must be a target_class sample


def test_select_target_point_correctly_classified(fitted_binary_lr):
    """Selected point must be correctly classified (baseline model predicts target_class)."""
    lr, X, y = fitted_binary_lr
    X_target, idx = select_target_point(X, y, lr, source_class=0, target_class=1)
    pred = lr.predict(X_target.reshape(1, -1))[0]
    assert pred == 1  # Currently correct, attack goal is to flip this


def test_select_target_point_multiclass(fitted_multiclass_lr):
    lr, X, y = fitted_multiclass_lr
    X_target, idx = select_target_point(X, y, lr, source_class=0, target_class=1)
    assert y[idx] == 1
    pred = lr.predict(X_target.reshape(1, -1))[0]
    assert pred == 1


def test_select_target_point_source_not_in_y_raises(fitted_binary_lr):
    lr, X, y = fitted_binary_lr
    with pytest.raises(ValueError, match="source_class"):
        select_target_point(X, y, lr, source_class=99, target_class=1)


def test_select_target_point_same_classes_raises(fitted_binary_lr):
    lr, X, y = fitted_binary_lr
    with pytest.raises(ValueError, match="must differ"):
        select_target_point(X, y, lr, source_class=0, target_class=0)


def test_select_target_point_requires_coef(fitted_binary_lr):
    """Non-LR models without coef_ raise a clear error."""
    from sklearn.tree import DecisionTreeClassifier
    lr, X, y = fitted_binary_lr
    dt = DecisionTreeClassifier().fit(X, y)
    with pytest.raises(ValueError, match="coef_"):
        select_target_point(X, y, dt, source_class=0, target_class=1)


# --- perturb_neighbors ---

def test_perturb_neighbors_returns_correct_shape(fitted_binary_lr):
    lr, X, y = fitted_binary_lr
    X_target, idx = select_target_point(X, y, lr, source_class=0, target_class=1)
    X_poisoned, perturbed = perturb_neighbors(X, y, X_target, lr, source_class=0, target_class=1, n_neighbors=5, epsilon=0.25)
    assert X_poisoned.shape == X.shape
    assert len(perturbed) == 5


def test_perturb_neighbors_only_source_class_changed(fitted_binary_lr):
    lr, X, y = fitted_binary_lr
    X_target, _ = select_target_point(X, y, lr, source_class=0, target_class=1)
    X_poisoned, perturbed = perturb_neighbors(X, y, X_target, lr, source_class=0, target_class=1, n_neighbors=5, epsilon=0.25)
    # Non-perturbed rows must be identical
    unchanged_mask = np.ones(len(X), dtype=bool)
    unchanged_mask[perturbed] = False
    assert np.allclose(X[unchanged_mask], X_poisoned[unchanged_mask])


def test_perturb_neighbors_only_source_class_eligible(fitted_binary_lr):
    """Perturbed indices must all be source_class samples."""
    lr, X, y = fitted_binary_lr
    X_target, _ = select_target_point(X, y, lr, source_class=0, target_class=1)
    _, perturbed = perturb_neighbors(X, y, X_target, lr, source_class=0, target_class=1, n_neighbors=5, epsilon=0.25)
    assert all(y[i] == 0 for i in perturbed)


def test_perturb_neighbors_does_not_change_labels(fitted_binary_lr):
    """Labels must stay unchanged: this is the 'clean label' invariant."""
    lr, X, y = fitted_binary_lr
    y_orig = y.copy()
    X_target, _ = select_target_point(X, y, lr, source_class=0, target_class=1)
    X_poisoned, _ = perturb_neighbors(X, y, X_target, lr, source_class=0, target_class=1, n_neighbors=5, epsilon=0.25)
    assert np.array_equal(y, y_orig)  # y must not be mutated


def test_perturb_neighbors_fewer_neighbors_than_requested(fitted_binary_lr):
    """If fewer source_class samples exist than n_neighbors, use all available."""
    lr, X, y = fitted_binary_lr
    X_target, _ = select_target_point(X, y, lr, source_class=0, target_class=1)
    _, perturbed = perturb_neighbors(X, y, X_target, lr, source_class=0, target_class=1, n_neighbors=9999, epsilon=0.25)
    n_source = (y == 0).sum()
    assert len(perturbed) == n_source


def test_perturb_neighbors_multiclass(fitted_multiclass_lr):
    lr, X, y = fitted_multiclass_lr
    X_target, _ = select_target_point(X, y, lr, source_class=0, target_class=1)
    X_poisoned, perturbed = perturb_neighbors(X, y, X_target, lr, source_class=0, target_class=1, n_neighbors=5, epsilon=0.25)
    assert X_poisoned.shape == X.shape
    assert all(y[i] == 0 for i in perturbed)


def test_perturb_neighbors_moves_toward_target_side(fitted_binary_lr):
    """Perturbed points must move toward the target_class side of the boundary."""
    from unrelabel.attacks.clean_label import _get_boundary_vector
    lr, X, y = fitted_binary_lr
    X_target, _ = select_target_point(X, y, lr, source_class=0, target_class=1)
    X_poisoned, perturbed = perturb_neighbors(
        X, y, X_target, lr, source_class=0, target_class=1, n_neighbors=5, epsilon=0.5
    )
    w_diff, b_diff = _get_boundary_vector(lr, source_class=0, target_class=1)
    before = X[perturbed] @ w_diff + b_diff
    after  = X_poisoned[perturbed] @ w_diff + b_diff
    # All perturbed points should have moved toward target side (smaller decision value)
    assert np.all(after < before)


# =============================================================================
# Integration tests: CleanLabelAttack class + _compute_clean_label_vulnerability_score
# =============================================================================

from unrelabel.attacks.clean_label import CleanLabelAttack, _compute_clean_label_vulnerability_score
from unrelabel.attacks.base import AttackResult
from unrelabel.loaders.dataset import PoisonDataset


@pytest.fixture
def binary_dataset_2f():
    """2-feature binary dataset, well-separated."""
    X, y = make_classification(
        n_samples=400, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42
    )
    return PoisonDataset(
        X_train=X[:280], X_test=X[280:],
        y_train=y[:280], y_test=y[280:],
        class_names=["0", "1"],
        feature_names=["f1", "f2"],
        source="test",
        metadata={},
    )


@pytest.fixture
def lr_wrapper():
    from unrelabel.loaders.model_loader import ModelWrapper
    return ModelWrapper(LogisticRegression(random_state=42), backend="sklearn")


# --- Vulnerability score ---

def test_vuln_score_success_high():
    """Successful attack with near-zero accuracy drop → score near 100."""
    score = _compute_clean_label_vulnerability_score(attack_success=True, accuracy_drop=0.001)
    assert score > 90.0


def test_vuln_score_success_with_drop():
    """Successful attack with 5% accuracy drop → score at boundary of stealth."""
    score = _compute_clean_label_vulnerability_score(attack_success=True, accuracy_drop=0.05)
    assert 60.0 <= score <= 80.0


def test_vuln_score_failure_small():
    score = _compute_clean_label_vulnerability_score(attack_success=False, accuracy_drop=0.01)
    assert 0.0 <= score <= 20.0


def test_vuln_score_failure_zero_drop():
    score = _compute_clean_label_vulnerability_score(attack_success=False, accuracy_drop=0.0)
    assert score == 0.0


def test_vuln_score_bounds():
    """Score must always be in [0, 100]."""
    for success in [True, False]:
        for drop in [-0.5, 0.0, 0.05, 0.5, 1.0]:
            score = _compute_clean_label_vulnerability_score(success, drop)
            assert 0.0 <= score <= 100.0


# --- CleanLabelAttack class ---

def test_clean_label_attack_returns_result(binary_dataset_2f, lr_wrapper):
    attack = CleanLabelAttack(source_class=0, target_class=1, n_neighbors=5, epsilon=0.25, seed=42)
    result = attack.run(binary_dataset_2f, lr_wrapper)
    assert isinstance(result, AttackResult)
    assert result.attack_type == "clean_label"


def test_clean_label_attack_config_fields(binary_dataset_2f, lr_wrapper):
    attack = CleanLabelAttack(source_class=0, target_class=1, n_neighbors=5, epsilon=0.25, seed=42)
    result = attack.run(binary_dataset_2f, lr_wrapper)
    cfg = result.config
    assert cfg["source_class"] == 0
    assert cfg["target_class"] == 1
    assert cfg["n_neighbors"] == 5
    assert cfg["epsilon"] == 0.25
    assert "target_index" in cfg
    assert "attack_success" in cfg
    assert isinstance(cfg["attack_success"], bool)


def test_clean_label_attack_accuracy_fields(binary_dataset_2f, lr_wrapper):
    attack = CleanLabelAttack(source_class=0, target_class=1)
    result = attack.run(binary_dataset_2f, lr_wrapper)
    assert 0.0 <= result.clean_accuracy <= 1.0
    assert 0.0 <= result.poisoned_accuracy <= 1.0
    assert 0.0 <= result.vulnerability_score <= 100.0


def test_clean_label_attack_sweep_results_empty(binary_dataset_2f, lr_wrapper):
    """Clean label attack has no sweep: sweep_results must be empty list."""
    attack = CleanLabelAttack(source_class=0, target_class=1)
    result = attack.run(binary_dataset_2f, lr_wrapper)
    assert result.sweep_results == []


def test_clean_label_attack_perturbed_indices_stored(binary_dataset_2f, lr_wrapper):
    """flipped_indices stores the perturbed neighbor indices."""
    attack = CleanLabelAttack(source_class=0, target_class=1, n_neighbors=3)
    result = attack.run(binary_dataset_2f, lr_wrapper)
    assert len(result.flipped_indices) == 3


def test_clean_label_attack_summary(binary_dataset_2f, lr_wrapper):
    attack = CleanLabelAttack(source_class=0, target_class=1)
    attack.run(binary_dataset_2f, lr_wrapper)
    s = attack.summary()
    assert "attack_type" in s
    assert "attack_success" in s
    assert "vulnerability_score" in s
    assert "accuracy_drop" in s


def test_clean_label_attack_summary_before_run_raises():
    attack = CleanLabelAttack(source_class=0, target_class=1)
    with pytest.raises(RuntimeError):
        attack.summary()


def test_clean_label_attack_same_classes_raises():
    with pytest.raises(ValueError, match="must differ"):
        CleanLabelAttack(source_class=0, target_class=0)


def test_clean_label_attack_non_lr_raises(binary_dataset_2f):
    """Models without coef_ should raise a clear error."""
    from sklearn.tree import DecisionTreeClassifier
    from unrelabel.loaders.model_loader import ModelWrapper
    dt_wrapper = ModelWrapper(DecisionTreeClassifier(), backend="sklearn")
    attack = CleanLabelAttack(source_class=0, target_class=1)
    with pytest.raises(ValueError, match="coef_"):
        attack.run(binary_dataset_2f, dt_wrapper)


def test_clean_label_attack_does_not_mutate_dataset(binary_dataset_2f, lr_wrapper):
    """CleanLabelAttack.run() must not mutate the caller's X_train or y_train."""
    X_train_orig = binary_dataset_2f.X_train.copy()
    y_train_orig = binary_dataset_2f.y_train.copy()
    attack = CleanLabelAttack(source_class=0, target_class=1, n_neighbors=5, epsilon=0.25, seed=42)
    attack.run(binary_dataset_2f, lr_wrapper)
    assert np.array_equal(binary_dataset_2f.X_train, X_train_orig), "X_train was mutated"
    assert np.array_equal(binary_dataset_2f.y_train, y_train_orig), "y_train was mutated"
