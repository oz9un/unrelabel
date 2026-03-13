# tests/test_e2e.py
"""
Full pipeline integration test: load → attack → visualize → report
"""
import json
import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from unrelabel.loaders.dataset_loader import DatasetLoader
from unrelabel.loaders.model_loader import ModelWrapper
from unrelabel.attacks.label_flipping import LabelFlippingAttack
from unrelabel.attacks.targeted_label import TargetedLabelAttack
from unrelabel.reporting.report import ReportBuilder
from unrelabel.reporting.visualizer import Visualizer
from unrelabel.reporting.metrics import severity_label


def test_full_label_flip_pipeline(tmp_path):
    # 1. Load dataset
    ds = DatasetLoader().load_sklearn("breast_cancer", test_size=0.2, seed=42)
    assert ds.n_train > 0

    # 2. Load model
    model = ModelWrapper(LogisticRegression(max_iter=1000, random_state=42), backend="sklearn")

    # 3. Run attack sweep
    attack = LabelFlippingAttack(poison_rates=[0.1, 0.2, 0.3], seed=42)
    result = attack.run(ds, model)

    assert result.clean_accuracy > 0.8
    assert len(result.sweep_results) == 3
    assert 0 <= result.vulnerability_score <= 100

    # 4. Generate visualizations
    viz = Visualizer()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    cm_path = viz.plot_confusion_matrices(result, plots_dir)
    acc_path = viz.plot_accuracy_curve(result, plots_dir)
    assert cm_path.exists()
    assert acc_path.exists()
    result.plots = [cm_path, acc_path]

    # 5. Generate report
    json_path, html_path = ReportBuilder().build(result, tmp_path)
    assert json_path.exists()
    assert html_path.exists()

    # 6. Verify JSON is valid and complete
    data = json.loads(json_path.read_text())
    assert data["attack_type"] == "label_flip"
    assert "sweep_results" in data
    assert len(data["sweep_results"]) == 3

    # 7. Verify HTML contains severity
    html = html_path.read_text()
    assert severity_label(result.vulnerability_score) in html


def test_full_targeted_label_pipeline(tmp_path):
    """Full pipeline: load → targeted attack → visualize → report."""
    # 1. Load iris (3 classes, well-separated, class 0 and 1 are distinct)
    ds = DatasetLoader().load_sklearn("iris", test_size=0.2, seed=42)
    assert ds.n_train > 0
    assert ds.n_classes == 3

    # 2. Model
    model = ModelWrapper(LogisticRegression(max_iter=1000, random_state=42), backend="sklearn")

    # 3. Run targeted attack sweep: class 0 → class 1
    attack = TargetedLabelAttack(
        source_class=0,
        target_class=1,
        poison_rates=[0.2, 0.4, 0.6],
        seed=42,
    )
    result = attack.run(ds, model)

    assert result.attack_type == "targeted_label"
    assert result.clean_accuracy > 0.8
    assert len(result.sweep_results) == 3
    assert 0 <= result.vulnerability_score <= 100
    assert "targeted_misclassification_rate" in result.config
    assert 0.0 <= result.config["targeted_misclassification_rate"] <= 1.0
    assert "targeted_misclassification_rate" in result.sweep_results[0]
    # At 60% poison of class 0, TMR must be non-trivial on this linearly separable dataset
    high_rate_row = result.sweep_results[-1]  # poison_rate=0.6
    assert high_rate_row["targeted_misclassification_rate"] > 0.1, (
        "Expected non-trivial TMR at 60% poison rate"
    )

    # 4. Generate visualizations
    viz = Visualizer()
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    cm_path = viz.plot_confusion_matrices(result, plots_dir)
    acc_path = viz.plot_accuracy_curve(result, plots_dir)
    assert cm_path.exists()
    assert acc_path.exists()
    result.plots = [cm_path, acc_path]

    # 5. Generate report
    json_path, html_path = ReportBuilder().build(result, tmp_path)
    assert json_path.exists()
    assert html_path.exists()

    # 6. Verify JSON
    data = json.loads(json_path.read_text())
    assert data["attack_type"] == "targeted_label"
    assert "sweep_results" in data
    assert len(data["sweep_results"]) == 3
    assert all("targeted_misclassification_rate" in r for r in data["sweep_results"])

    # 7. Verify HTML contains severity and TMR
    html = html_path.read_text()
    assert severity_label(result.vulnerability_score) in html
    assert "Targeted Misclassification Rate" in html


def test_full_clean_label_pipeline(tmp_path):
    """Full pipeline: load → clean label attack → report."""
    from unrelabel.attacks.clean_label import CleanLabelAttack

    # 1. Load iris (multiclass, well-separated, LR works well)
    ds = DatasetLoader().load_sklearn("iris", test_size=0.2, seed=42)
    assert ds.n_train > 0

    # 2. Model (must have coef_)
    model = ModelWrapper(LogisticRegression(max_iter=1000, random_state=42), backend="sklearn")

    # 3. Run clean label attack: class 0 features perturbed, target = class 1
    attack = CleanLabelAttack(source_class=0, target_class=1, n_neighbors=5, epsilon=0.25, seed=42)
    result = attack.run(ds, model)

    assert result.attack_type == "clean_label"
    assert 0.0 <= result.clean_accuracy <= 1.0
    assert 0.0 <= result.vulnerability_score <= 100.0
    assert result.sweep_results == []
    assert "attack_success" in result.config
    assert isinstance(result.config["attack_success"], bool)
    assert len(result.flipped_indices) == 5  # exactly n_neighbors perturbed (iris has 40 class-0 train samples, deterministic with seed=42)

    # 4. Generate report
    json_path, html_path = ReportBuilder().build(result, tmp_path)
    assert json_path.exists()
    assert html_path.exists()

    # 5. Verify JSON
    data = json.loads(json_path.read_text())
    assert data["attack_type"] == "clean_label"
    assert data["sweep_results"] == []
    assert "attack_success" in data["config"]

    # 6. Verify HTML
    html = html_path.read_text()
    assert severity_label(result.vulnerability_score) in html
    assert "Attack Succeeded" in html
