from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def compute_per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[int, float]:
    classes = np.unique(y_true)
    return {
        int(c): float(np.mean(y_pred[y_true == c] == c))
        for c in classes
    }


def severity_label(score: float) -> str:
    """Map a vulnerability score (0-100) to a severity string."""
    if score <= 5:
        return "Clean"
    elif score <= 40:
        return "Low"
    elif score <= 65:
        return "Medium"
    elif score <= 85:
        return "High"
    else:
        return "Critical"
