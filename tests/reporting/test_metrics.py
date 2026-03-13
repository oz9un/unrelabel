import numpy as np
import pytest
from unrelabel.reporting.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_per_class_accuracy,
    severity_label,
)


def test_compute_accuracy():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    assert compute_accuracy(y_true, y_pred) == pytest.approx(0.8)


def test_compute_confusion_matrix_shape():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 1, 2])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2


def test_per_class_accuracy():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    pca = compute_per_class_accuracy(y_true, y_pred)
    assert pca[0] == pytest.approx(0.5)
    assert pca[1] == pytest.approx(2/3)


def test_severity_label():
    assert severity_label(0) == "Clean"
    assert severity_label(25) == "Low"
    assert severity_label(50) == "Medium"
    assert severity_label(75) == "High"
    assert severity_label(90) == "Critical"
