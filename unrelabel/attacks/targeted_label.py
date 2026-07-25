from __future__ import annotations
from datetime import datetime, timezone

import numpy as np

from unrelabel.attacks.base import AttackResult, BaseAttack
from unrelabel.loaders.dataset import PoisonDataset
from unrelabel.loaders.model_loader import ModelWrapper


def targeted_flip_labels(
    y: np.ndarray,
    poison_rate: float,
    source_class: int,
    target_class: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Flip a fraction of source_class labels to target_class.

    Args:
        y: Original label array.
        poison_rate: Fraction of source_class samples to flip (0.0 to 1.0).
        source_class: Only samples of this class are eligible to be flipped.
        target_class: Flipped samples are assigned this class.
        seed: Random seed for reproducibility.

    Returns:
        (y_poisoned, flipped_indices)
    """
    if not 0.0 <= poison_rate <= 1.0:
        raise ValueError(f"poison_rate must be between 0 and 1, got {poison_rate}")
    if source_class == target_class:
        raise ValueError(
            f"source_class and target_class must differ, both are {source_class!r}."
        )

    y = np.asarray(y)
    classes = np.unique(y)

    if source_class not in classes:
        raise ValueError(f"source_class {source_class!r} not found in y. Available: {classes.tolist()}")
    if target_class not in classes:
        raise ValueError(f"target_class {target_class!r} not found in y. Available: {classes.tolist()}")

    source_indices = np.where(y == source_class)[0]
    n_to_flip = int(len(source_indices) * poison_rate)

    if n_to_flip == 0:
        return y.copy(), np.array([], dtype=int)

    rng = np.random.default_rng(seed)
    flipped_indices = rng.choice(source_indices, size=n_to_flip, replace=False)
    y_poisoned = y.copy()
    y_poisoned[flipped_indices] = target_class

    return y_poisoned, flipped_indices


def _compute_targeted_vulnerability_score(
    targeted_misclassification_rate: float,
    n_flipped: int,
    n_source_train: int,
) -> float:
    """
    Compute a normalized vulnerability score (0-100) for targeted attacks.

    Weights:
        - 70% targeted misclassification rate (primary signal)
        - 30% effective poison ratio within source_class training samples
    """
    ratio_score = min(n_flipped / n_source_train, 1.0) if n_source_train > 0 else 0.0
    raw = (0.7 * targeted_misclassification_rate + 0.3 * ratio_score) * 100
    return round(max(0.0, min(100.0, raw)), 2)


class TargetedLabelAttack(BaseAttack):
    """
    Targeted Label Attack: flip source_class labels to target_class to cause
    specific misclassifications at inference time.

    Primary metric: targeted_misclassification_rate: fraction of source_class
    test samples the poisoned model predicts as target_class.
    """

    def __init__(
        self,
        source_class: int,
        target_class: int,
        poison_rate: float | None = None,
        poison_rates: list[float] | None = None,
        seed: int = 42,
    ):
        if source_class == target_class:
            raise ValueError(
                f"source_class and target_class must differ, both are {source_class!r}."
            )
        if poison_rate is None and not poison_rates:
            raise ValueError("Provide either poison_rate or poison_rates.")
        self.source_class = source_class
        self.target_class = target_class
        self.poison_rate = poison_rate
        self.poison_rates = poison_rates or ([poison_rate] if poison_rate is not None else [])
        self.seed = seed
        self._last_result: AttackResult | None = None

    def run(self, dataset: PoisonDataset, model: ModelWrapper) -> AttackResult:
        from sklearn.metrics import accuracy_score, confusion_matrix

        try:
            clean_model = model.clone()
        except NotImplementedError:
            raise ValueError(
                f"TargetedLabelAttack requires a model that supports clone() "
                f"(e.g. sklearn). Got backend='{model.backend}'."
            )

        clean_model.fit(dataset.X_train, dataset.y_train)
        clean_preds = clean_model.predict(dataset.X_test)
        clean_accuracy = float(accuracy_score(dataset.y_test, clean_preds))
        clean_cm = confusion_matrix(dataset.y_test, clean_preds).tolist()

        # Source class counts (computed once)
        source_test_mask = dataset.y_test == self.source_class
        n_source_test = int(source_test_mask.sum())
        n_source_train = int((dataset.y_train == self.source_class).sum())

        sweep_results = []
        final_result_data: dict = {}

        for rate in self.poison_rates:
            y_poisoned, flipped_idx = targeted_flip_labels(
                dataset.y_train,
                poison_rate=rate,
                source_class=self.source_class,
                target_class=self.target_class,
                seed=self.seed,
            )
            try:
                poisoned_model = model.clone()
            except NotImplementedError:
                raise ValueError(
                    f"TargetedLabelAttack requires a model that supports clone() "
                    f"(e.g. sklearn). Got backend='{model.backend}'."
                )

            poisoned_model.fit(dataset.X_train, y_poisoned)
            poisoned_preds = poisoned_model.predict(dataset.X_test)
            poisoned_accuracy = float(accuracy_score(dataset.y_test, poisoned_preds))
            poisoned_cm = confusion_matrix(dataset.y_test, poisoned_preds).tolist()
            accuracy_drop = clean_accuracy - poisoned_accuracy

            # Targeted misclassification rate
            if n_source_test > 0:
                tmr = float(
                    ((dataset.y_test == self.source_class) &
                     (poisoned_preds == self.target_class)).sum()
                ) / n_source_test
            else:
                tmr = 0.0

            vscore = _compute_targeted_vulnerability_score(
                tmr, len(flipped_idx), n_source_train
            )

            sweep_results.append({
                "poison_rate": rate,
                "poisoned_accuracy": poisoned_accuracy,
                "accuracy_drop": accuracy_drop,
                "targeted_misclassification_rate": tmr,
                "vulnerability_score": vscore,
                "n_flipped": len(flipped_idx),
                "confusion_matrix": poisoned_cm,
            })
            final_result_data = {
                "poisoned_accuracy": poisoned_accuracy,
                "accuracy_drop": accuracy_drop,
                "targeted_misclassification_rate": tmr,
                "vulnerability_score": vscore,
                "flipped_indices": flipped_idx,
                "poisoned_cm": poisoned_cm,
            }

        if not final_result_data:
            raise RuntimeError("No sweep iterations completed; poison_rates list is empty.")

        result = AttackResult(
            attack_type="targeted_label",
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
                "attack_type": "targeted_label",
                "source_class": self.source_class,
                "target_class": self.target_class,
                "poison_rate": self.poison_rates[-1],
                "poison_rates": self.poison_rates,
                "seed": self.seed,
                "targeted_misclassification_rate": final_result_data["targeted_misclassification_rate"],
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
            "targeted_misclassification_rate": r.config["targeted_misclassification_rate"],
            "vulnerability_score": r.vulnerability_score,
            "n_flipped": len(r.flipped_indices),
        }
