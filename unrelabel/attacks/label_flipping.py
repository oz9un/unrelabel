from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from unrelabel.attacks.base import AttackResult, BaseAttack
from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.model_loader import ModelWrapper


def flip_labels(
    y: np.ndarray,
    poison_rate: float,
    seed: int = 42,
    source_class: int | None = None,
    target_class: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip a fraction of labels in y.

    Args:
        y: Original label array.
        poison_rate: Fraction of labels to flip (0.0 to 1.0).
        seed: Random seed for reproducibility.
        source_class: If set, only flip labels belonging to this class.
        target_class: If set, flip selected labels to this class.
                      If None, flip to a random different class.

    Returns:
        (y_poisoned, flipped_indices)
    """
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate must be between 0 and 1, got {poison_rate}")

    if source_class is not None and target_class is not None and source_class == target_class:
        raise ValueError(
            f"source_class and target_class must differ, both are {source_class!r}."
        )

    y = np.asarray(y)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)

    if source_class is not None:
        candidate_indices = np.where(y == source_class)[0]
    else:
        candidate_indices = np.arange(len(y))

    n_to_flip = int(len(candidate_indices) * poison_rate)

    if n_to_flip == 0:
        return y.copy(), np.array([], dtype=int)

    flipped_indices = rng.choice(candidate_indices, size=n_to_flip, replace=False)
    y_poisoned = y.copy()

    for idx in flipped_indices:
        if target_class is not None:
            y_poisoned[idx] = target_class
        else:
            other_classes = classes[classes != y[idx]]
            if len(other_classes) == 0:
                raise ValueError(
                    f"Cannot randomly flip label {y[idx]!r}: only one unique class exists. "
                    "Provide target_class explicitly."
                )
            y_poisoned[idx] = rng.choice(other_classes)

    return y_poisoned, flipped_indices


def _compute_vulnerability_score(
    accuracy_drop: float,
    poison_rate: float,
    n_flipped: int,
    n_train: int,
) -> float:
    """
    Compute a normalized vulnerability score (0-100).

    Weights:
        - 60% accuracy drop (normalized to max expected drop at poison_rate)
        - 40% effective poison ratio (n_flipped / n_train)
    """
    max_expected_drop = min(poison_rate * 2, 1.0)
    drop_score = min(accuracy_drop / max_expected_drop, 1.0) if max_expected_drop > 0 else 0.0
    ratio_score = min(n_flipped / n_train, 1.0)
    raw = (0.6 * drop_score + 0.4 * ratio_score) * 100
    return round(max(0.0, min(100.0, raw)), 2)


class LabelFlippingAttack(BaseAttack):
    """
    Label Flipping attack: corrupt training labels to degrade model performance.

    Supports binary and multiclass datasets. Can target a specific source class
    and/or direct flips to a specific target class.
    """

    def __init__(
        self,
        poison_rate: float | None = None,
        poison_rates: list[float] | None = None,
        source_class: int | None = None,
        target_class: int | None = None,
        seed: int = 42,
    ):
        if poison_rate is None and not poison_rates:
            raise ValueError("Provide either poison_rate or poison_rates.")
        self.poison_rate = poison_rate
        self.poison_rates = poison_rates or ([poison_rate] if poison_rate is not None else [])
        self.source_class = source_class
        self.target_class = target_class
        self.seed = seed
        self._last_result: AttackResult | None = None

    def run(self, dataset: PoisonDataset, model: ModelWrapper) -> AttackResult:
        from sklearn.metrics import accuracy_score, confusion_matrix

        # Train and evaluate clean baseline
        try:
            clean_model = model.clone()
        except NotImplementedError:
            raise ValueError(
                f"LabelFlippingAttack requires a model that supports clone() "
                f"(e.g. sklearn). Got backend='{model.backend}'."
            )
        clean_model.fit(dataset.X_train, dataset.y_train)
        clean_preds = clean_model.predict(dataset.X_test)
        clean_accuracy = float(accuracy_score(dataset.y_test, clean_preds))
        clean_cm = confusion_matrix(dataset.y_test, clean_preds).tolist()

        sweep_results = []
        final_result_data = {}

        for rate in self.poison_rates:
            y_poisoned, flipped_idx = flip_labels(
                dataset.y_train,
                poison_rate=rate,
                seed=self.seed,
                source_class=self.source_class,
                target_class=self.target_class,
            )
            try:
                poisoned_model = model.clone()
            except NotImplementedError:
                raise ValueError(
                    f"LabelFlippingAttack requires a model that supports clone() "
                    f"(e.g. sklearn). Got backend='{model.backend}'."
                )
            poisoned_model.fit(dataset.X_train, y_poisoned)
            poisoned_preds = poisoned_model.predict(dataset.X_test)
            poisoned_accuracy = float(accuracy_score(dataset.y_test, poisoned_preds))
            poisoned_cm = confusion_matrix(dataset.y_test, poisoned_preds).tolist()
            accuracy_drop = clean_accuracy - poisoned_accuracy
            vscore = _compute_vulnerability_score(
                accuracy_drop, rate, len(flipped_idx), dataset.n_train
            )
            sweep_results.append({
                "poison_rate": rate,
                "poisoned_accuracy": poisoned_accuracy,
                "accuracy_drop": accuracy_drop,
                "vulnerability_score": vscore,
                "n_flipped": len(flipped_idx),
                "confusion_matrix": poisoned_cm,
            })
            final_result_data = {
                "poisoned_accuracy": poisoned_accuracy,
                "accuracy_drop": accuracy_drop,
                "vulnerability_score": vscore,
                "flipped_indices": flipped_idx,
                "poisoned_cm": poisoned_cm,
            }

        result = AttackResult(
            attack_type="label_flip",
            clean_accuracy=clean_accuracy,
            poisoned_accuracy=final_result_data["poisoned_accuracy"],
            accuracy_drop=final_result_data["accuracy_drop"],
            vulnerability_score=final_result_data["vulnerability_score"],
            confusion_matrices={
                "clean": clean_cm,
                "poisoned": final_result_data["poisoned_cm"],
            },
            plots=[],
            config={
                "poison_rate": self.poison_rates[-1],
                "poison_rates": self.poison_rates,
                "source_class": self.source_class,
                "target_class": self.target_class,
                "seed": self.seed,
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
            flipped_indices=final_result_data["flipped_indices"],
            sweep_results=sweep_results,
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
            "vulnerability_score": r.vulnerability_score,
            "n_flipped": len(r.flipped_indices),
        }
