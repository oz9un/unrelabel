# tests/reporting/test_visualizer.py
import numpy as np
import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from unrelabel.reporting.visualizer import Visualizer
from unrelabel.attacks.base import AttackResult


@pytest.fixture
def sample_result(tmp_path):
    result = AttackResult(
        attack_type="label_flip",
        clean_accuracy=0.95,
        poisoned_accuracy=0.75,
        accuracy_drop=0.20,
        vulnerability_score=55.0,
        confusion_matrices={"clean": [[40, 2], [1, 17]], "poisoned": [[35, 7], [5, 13]]},
        plots=[],
        config={"poison_rate": 0.3},
        timestamp="2026-03-09T00:00:00Z",
        flipped_indices=np.array([0, 5, 10, 15]),
        sweep_results=[
            {"poison_rate": 0.1, "poisoned_accuracy": 0.92, "accuracy_drop": 0.03},
            {"poison_rate": 0.2, "poisoned_accuracy": 0.85, "accuracy_drop": 0.10},
            {"poison_rate": 0.3, "poisoned_accuracy": 0.75, "accuracy_drop": 0.20},
        ],
    )
    return result, tmp_path


def test_plot_confusion_matrices_saves_file(sample_result):
    result, out = sample_result
    viz = Visualizer()
    path = viz.plot_confusion_matrices(result, out)
    assert path.exists()
    assert path.suffix == ".png"


def test_plot_accuracy_curve_saves_file(sample_result):
    result, out = sample_result
    viz = Visualizer()
    path = viz.plot_accuracy_curve(result, out)
    assert path.exists()


def test_plot_poisoned_scatter_saves_file(sample_result):
    result, out = sample_result
    viz = Visualizer()
    X = np.random.rand(100, 2)
    y_original = np.random.randint(0, 2, 100)
    y_poisoned = y_original.copy()
    y_poisoned[result.flipped_indices] = 1 - y_poisoned[result.flipped_indices]
    path = viz.plot_poisoned_scatter(
        X, y_original, y_poisoned, result.flipped_indices, out
    )
    assert path.exists()
