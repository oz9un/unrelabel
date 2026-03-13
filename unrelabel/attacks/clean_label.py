from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _get_boundary_vector(
    fitted_model,
    source_class: int,
    target_class: int,
) -> tuple[np.ndarray, float]:
    """
    Extract (w_diff, b_diff) for the source-vs-target decision boundary.

    For binary LR: coef_ has shape (1, n_features); intercept_ has shape (1,).
    For multiclass OvR LR: coef_ has shape (n_classes, n_features).

    The boundary is: w_diff @ x + b_diff = 0
    where w_diff = w_source - w_target, b_diff = b_source - b_target.
    A point is on the source_class side if w_diff @ x + b_diff > 0.
    """
    if not hasattr(fitted_model, "coef_"):
        raise ValueError(
            "Model must have coef_ attribute (e.g. LogisticRegression). "
            f"Got: {type(fitted_model).__name__}"
        )

    coef = fitted_model.coef_
    intercept = fitted_model.intercept_

    model_classes = list(fitted_model.classes_)
    for cls, name in [(source_class, "source_class"), (target_class, "target_class")]:
        if cls not in model_classes:
            raise ValueError(
                f"{name} {cls!r} not found in model.classes_: {model_classes}"
            )

    if coef.shape[0] == 1:
        # Binary classifier: row 0 is the positive class (class 1 by default)
        # Decision function = coef[0] @ x + intercept[0]
        # Positive → class 1; Negative → class 0
        classes = fitted_model.classes_
        # w_diff oriented so that w_diff @ x + b_diff > 0 → source_class region
        if classes[1] == target_class:
            w_diff = -coef[0]
            b_diff = -intercept[0]
        else:
            w_diff = coef[0]
            b_diff = intercept[0]
    else:
        # Multiclass OvR: row i corresponds to classes_[i]
        classes = list(fitted_model.classes_)
        src_idx = classes.index(source_class)
        tgt_idx = classes.index(target_class)
        w_diff = coef[src_idx] - coef[tgt_idx]
        b_diff = intercept[src_idx] - intercept[tgt_idx]

    return w_diff, float(b_diff)


def select_target_point(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fitted_model,
    source_class: int,
    target_class: int,
) -> tuple[np.ndarray, int]:
    """
    Find the target_class training point closest to the source/target decision boundary
    that is still correctly classified.

    Returns:
        (X_target, absolute_index_in_X_train)
    """
    if source_class == target_class:
        raise ValueError(
            f"source_class and target_class must differ, both are {source_class!r}."
        )
    classes = np.unique(y_train)
    if source_class not in classes:
        raise ValueError(
            f"source_class {source_class!r} not found in y_train. Available: {classes.tolist()}"
        )
    if target_class not in classes:
        raise ValueError(
            f"target_class {target_class!r} not found in y_train. Available: {classes.tolist()}"
        )

    w_diff, b_diff = _get_boundary_vector(fitted_model, source_class, target_class)

    # Restrict to target_class samples
    target_mask = y_train == target_class
    target_indices = np.where(target_mask)[0]
    X_target_class = X_train[target_indices]

    # Decision values: f01(x) = w_diff @ x + b_diff
    # > 0 → source_class side; < 0 → target_class side (correctly classified)
    decision_values = X_target_class @ w_diff + b_diff

    # Keep only correctly classified: f01 < 0
    correct_mask = decision_values < 0
    if correct_mask.any():
        # Largest negative = closest to boundary
        rel_idx = np.where(correct_mask)[0][np.argmax(decision_values[correct_mask])]
    else:
        # Fallback: closest to boundary by absolute value
        rel_idx = int(np.argmin(np.abs(decision_values)))

    abs_idx = int(target_indices[rel_idx])
    return X_train[abs_idx].copy(), abs_idx


def perturb_neighbors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_target: np.ndarray,
    fitted_model,
    source_class: int,
    target_class: int,
    n_neighbors: int = 5,
    epsilon: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the n_neighbors source_class training points nearest to X_target,
    push them across the source/target decision boundary by epsilon,
    and return the poisoned training set (labels unchanged).

    Returns:
        (X_train_poisoned, perturbed_indices_in_X_train)
    """
    w_diff, _ = _get_boundary_vector(fitted_model, source_class, target_class)

    # Push direction: from source_class side INTO target_class side
    # source side: w_diff @ x + b_diff > 0; target side: < 0
    # Direction to cross INTO target_class region: negative of w_diff normal
    norm = np.linalg.norm(w_diff)
    if norm < 1e-9:
        raise ValueError("Boundary normal vector has near-zero norm — cannot determine push direction.")
    unit_push = -w_diff / norm
    perturbation = epsilon * unit_push

    # Find source_class neighbors
    source_mask = y_train == source_class
    source_indices = np.where(source_mask)[0]
    X_source = X_train[source_indices]

    actual_k = min(n_neighbors, len(source_indices))
    nn = NearestNeighbors(n_neighbors=actual_k, algorithm="auto")
    nn.fit(X_source)
    _, rel_indices = nn.kneighbors(X_target.reshape(1, -1))
    neighbor_abs_indices = source_indices[rel_indices.flatten()]

    # Apply perturbation (labels stay unchanged)
    X_poisoned = X_train.copy()
    X_poisoned[neighbor_abs_indices] = X_train[neighbor_abs_indices] + perturbation

    return X_poisoned, neighbor_abs_indices


# =============================================================================
# CleanLabelAttack class + vulnerability score helper
# =============================================================================

from datetime import datetime, timezone

from unrelabel.attacks.base import AttackResult, BaseAttack
from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.model_loader import ModelWrapper


def _compute_clean_label_vulnerability_score(
    attack_success: bool,
    accuracy_drop: float,
) -> float:
    """
    Score 0-100 for clean label attacks.

    Success path (70-100): high stealth = minimal accuracy drop = higher score.
        stealth = max(0, 1 - min(accuracy_drop / 0.05, 1.0))
        score = (0.7 + 0.3 * stealth) * 100

    Failure path (0-20): score is proportional to observed accuracy drop.
        score = min(20.0, abs(accuracy_drop) * 100)
    """
    if attack_success:
        stealth = max(0.0, 1.0 - min(abs(accuracy_drop) / 0.05, 1.0))
        raw = (0.7 + 0.3 * stealth) * 100.0
    else:
        raw = min(20.0, abs(accuracy_drop) * 100.0)
    return round(max(0.0, min(100.0, raw)), 2)


class CleanLabelAttack(BaseAttack):
    """
    Clean Label Attack: perturb source_class training features (labels unchanged)
    to shift the decision boundary and cause a specific target_class training point
    to be misclassified after retraining.

    Requires a model with coef_ and intercept_ (e.g. LogisticRegression).

    Note: ``seed`` is stored in config for reproducibility metadata but has no
    effect on algorithm output — target selection and neighbor perturbation are
    both fully deterministic given fixed training data and a fitted model.
    """

    def __init__(
        self,
        source_class: int,
        target_class: int,
        n_neighbors: int = 5,
        epsilon: float = 0.25,
        seed: int = 42,
    ):
        if source_class == target_class:
            raise ValueError(
                f"source_class and target_class must differ, both are {source_class!r}."
            )
        self.source_class = source_class
        self.target_class = target_class
        self.n_neighbors = n_neighbors
        self.epsilon = epsilon
        self.seed = seed
        self._last_result: AttackResult | None = None

    def run(self, dataset: PoisonDataset, model: ModelWrapper) -> AttackResult:
        from sklearn.metrics import accuracy_score, confusion_matrix

        # Fit baseline model on clean data
        try:
            baseline_model = model.clone()
        except NotImplementedError:
            raise ValueError(
                f"CleanLabelAttack requires a model that supports clone(). "
                f"Got backend='{model.backend}'."
            )

        baseline_model.fit(dataset.X_train, dataset.y_train)

        # Validate that the underlying estimator has coef_ (needed for boundary math)
        sk_model = baseline_model._model
        if not hasattr(sk_model, "coef_"):
            raise ValueError(
                f"CleanLabelAttack requires a model with coef_ (e.g. LogisticRegression). "
                f"Got: {type(sk_model).__name__}"
            )

        clean_preds = baseline_model.predict(dataset.X_test)
        clean_accuracy = float(accuracy_score(dataset.y_test, clean_preds))
        clean_cm = confusion_matrix(dataset.y_test, clean_preds).tolist()

        # Find target point: closest target_class sample to the decision boundary
        X_target, target_idx = select_target_point(
            dataset.X_train, dataset.y_train, sk_model,
            source_class=self.source_class, target_class=self.target_class,
        )
        target_true_label = int(dataset.y_train[target_idx])

        # Perturb source_class neighbors to push boundary toward target point
        X_poisoned, perturbed_indices = perturb_neighbors(
            dataset.X_train, dataset.y_train, X_target, sk_model,
            source_class=self.source_class, target_class=self.target_class,
            n_neighbors=self.n_neighbors, epsilon=self.epsilon,
        )

        # Retrain on poisoned data (labels unchanged — clean label invariant)
        poisoned_model = model.clone()
        poisoned_model.fit(X_poisoned, dataset.y_train)

        poisoned_preds = poisoned_model.predict(dataset.X_test)
        poisoned_accuracy = float(accuracy_score(dataset.y_test, poisoned_preds))
        poisoned_cm = confusion_matrix(dataset.y_test, poisoned_preds).tolist()
        accuracy_drop = clean_accuracy - poisoned_accuracy

        # Check if attack succeeded: target point now misclassified as source_class
        target_pred = int(poisoned_model.predict(X_target.reshape(1, -1))[0])
        # Both conditions: (1) target misclassified AND (2) specifically as source_class.
        # In multiclass, a prediction of a third class does not count as attack success.
        attack_success = (target_pred != target_true_label) and (target_pred == self.source_class)

        vuln_score = _compute_clean_label_vulnerability_score(attack_success, accuracy_drop)

        result = AttackResult(
            attack_type="clean_label",
            clean_accuracy=clean_accuracy,
            poisoned_accuracy=poisoned_accuracy,
            accuracy_drop=accuracy_drop,
            vulnerability_score=vuln_score,
            confusion_matrices={"clean": clean_cm, "poisoned": poisoned_cm},
            plots=[],
            config={
                "attack_type": "clean_label",
                "source_class": self.source_class,
                "target_class": self.target_class,
                "n_neighbors": self.n_neighbors,
                "epsilon": self.epsilon,
                "seed": self.seed,
                "target_index": target_idx,
                "target_pred": target_pred,
                "attack_success": attack_success,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            flipped_indices=perturbed_indices,
            sweep_results=[],
        )
        self._last_result = result
        return result

    def summary(self) -> dict:
        if self._last_result is None:
            raise RuntimeError("Run the attack first.")
        r = self._last_result
        return {
            "attack_type": r.attack_type,
            "clean_accuracy": r.clean_accuracy,
            "poisoned_accuracy": r.poisoned_accuracy,
            "accuracy_drop": r.accuracy_drop,
            "attack_success": r.config["attack_success"],
            "target_index": r.config["target_index"],
            "target_pred": r.config["target_pred"],
            "vulnerability_score": r.vulnerability_score,
            "n_perturbed": len(r.flipped_indices),
        }
